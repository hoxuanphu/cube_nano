"""Developer entry point for the local satellite simulator."""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable, Iterator, TextIO

from link_sim.transport import TransportFrame, UdpMissionEndpoint
from protocol.ccsds import TcTypeBdFrame
from protocol.schemas import Command, CommandOpcode, ROI, RequestKey, SceneRef, decode_command

from .cloud_payload import CloudPayload
from .deployment import SatelliteDeployment
from .mission_udp_adapter import MissionUdpAdapter


logger = logging.getLogger("flight.satellite_simulator")


def configure_realtime_logging(level: str = "INFO", *, stream: TextIO | None = None) -> None:
    """Configure concise satellite logs without contaminating JSON stdout."""
    normalized = str(level).upper()
    numeric_level = getattr(logging, normalized, None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"unsupported log level: {level}")

    target_stream = stream or sys.stderr
    flight_logger = logging.getLogger("flight")
    flight_logger.setLevel(numeric_level)
    flight_logger.propagate = False
    handler = next(
        (item for item in flight_logger.handlers if getattr(item, "_cube_nano_satellite", False)),
        None,
    )
    if handler is None:
        handler = logging.StreamHandler(target_stream)
        handler._cube_nano_satellite = True
        flight_logger.addHandler(handler)
    elif hasattr(handler, "setStream"):
        handler.setStream(target_stream)
    handler.setLevel(numeric_level)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s.%(msecs)03d %(levelname)-8s [SAT] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _log_health(health: dict, *, label: str = "status") -> None:
    scheduler = health.get("scheduler_metrics", {})
    queues = health.get("scheduler_queue_depths", {})
    logger.info(
        "%s state=%s worker=%s heartbeat_age_ms=%s queue=%s/%s active_request_id=%s "
        "restarts=%s scheduler=ACK:%s CONTROL:%s FILE:%s",
        label,
        health.get("state"),
        health.get("worker_state"),
        health.get("worker_heartbeat_age_ms"),
        health.get("queue_depth"),
        health.get("queue_capacity"),
        health.get("active_request_id"),
        health.get("worker_restart_count"),
        queues.get("ACK", 0),
        queues.get("CONTROL", 0),
        queues.get("FILE", 0),
    )


class SatelliteSimulator:
    def __init__(
        self,
        root: str | Path = ".",
        *,
        state_directory: str | Path | None = None,
        product_directory: str | Path | None = None,
        device: str = "cpu",
        start_worker: bool = True,
        event_clock: Callable[[], int] | None = None,
        event_time_base: str = "monotonic",
    ):
        self.deployment = SatelliteDeployment(
            root,
            state_directory=state_directory,
            product_directory=product_directory,
        )
        self.payload = CloudPayload(
            self.deployment,
            device=device,
            start_worker=start_worker,
            event_clock=event_clock,
            event_time_base=event_time_base,
        )
        # APID 2 control/event telemetry has a small, explicit bound. APID 3
        # file data never enters this queue; it remains completion-gated in
        # CloudPayload's FILE scheduler.
        self._pending_tm_capacity = self.payload.profile.ack_mailbox_capacity
        self._pending_tm_frames: deque[tuple[bytes, int]] = deque()
        self._pending_lock = threading.RLock()
        self._last_journal_event_id = 0
        self._queue_status_tm()

    def _queue_tm(self, message: dict) -> None:
        """Encode one bounded APID 2 event without allowing telemetry overflow."""

        try:
            frame, ordering_key = self.payload.encode_ack_tm_frame_with_order(message)
        except ValueError as exc:
            # Events must never make the flight endpoint unavailable.  Keep a
            # small, correlated diagnostic instead of silently dropping it.
            fallback = {
                "event_name": "TM_EVENT_TRUNCATED",
                "stage": message.get("stage", "FAILED"),
                "message": {
                    "error_code": "TM_EVENT_OVERSIZE",
                    "original_event_name": str(message.get("event_name", "MISSION_EVENT")),
                    "detail": type(exc).__name__,
                },
            }
            if message.get("request_key") is not None:
                fallback["request_key"] = message["request_key"]
            frame, ordering_key = self.payload.encode_ack_tm_frame_with_order(fallback)
        with self._pending_lock:
            if len(self._pending_tm_frames) >= self._pending_tm_capacity:
                raise RuntimeError("TM_PENDING_QUEUE_FULL")
            self._pending_tm_frames.append((frame, ordering_key))

    def _queue_status_tm(self) -> None:
        health = self.payload.health()
        slo = self.payload.deployment.slo_profile
        self._queue_tm(
            {
                "event_name": "SATELLITE_STATUS",
                "message": {
                    "state": health["state"],
                    "config": health["config"],
                    "model_release_id": health["model_release_id"],
                    "assurance_level": health["assurance_level"],
                    "queue_capacity": health["queue_capacity"],
                    "scheduler_queue_depths": health["scheduler_queue_depths"],
                    # Keep the recurring readiness frame within the fixed TM
                    # packet budget.  The immutable full SLO follows once as
                    # a separate APID 2 event below.
                    "slo": None if slo is None else {
                        "oldest_ack_age_ms": slo.oldest_ack_age_ms,
                        "health_max_latency_ms": slo.health_max_latency_ms,
                    },
                },
            }
        )
        if slo is not None:
            self._queue_tm({"event_name": "SATELLITE_SLO", "message": slo.as_dict()})

    def queue_status_tm(self) -> None:
        """Queue a compact APID 2 readiness update for the mission endpoint."""

        self._queue_status_tm()

    def queue_catalog_snapshot(self) -> None:
        """Publish the canonical catalog only through APID 2 TM."""
        self._queue_tm(
            {
                "event_name": "CATALOG_SNAPSHOT",
                "catalog_bundle": self.payload.catalog.bundle_bytes(),
            }
        )

    @staticmethod
    def _request_key_from_transport_frame(transport_frame: TransportFrame) -> dict | None:
        try:
            tc = TcTypeBdFrame.decode(transport_frame.frame_bytes)
            return decode_command(tc.packet.payload).request_key.as_dict()
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _compact_command_result(result: dict, request_key: dict | None) -> dict:
        """Retain ACK correlation fields while excluding variable-size payloads."""

        compact: dict = {
            "stage": str(result.get("stage", "MISSION_EVENT")),
        }
        effective_key = result.get("request_key", request_key)
        if effective_key is not None:
            compact["request_key"] = effective_key
        for name in (
            "error_code",
            "transfer_id",
            "product_ref",
            "config_snapshot",
            "job_key",
            "state",
            "cancel_outcome",
            "preview_policy",
            "scene_ref",
        ):
            if name in result:
                compact[name] = result[name]
        return compact

    @staticmethod
    def _compact_journal_body(event_name: str, body: dict) -> dict:
        """Send lifecycle correlation data, not worker/product implementation data."""

        compact: dict = {}
        for name in (
            "error_code",
            "transfer_id",
            "state",
            "science_decision",
            "progress_bp",
            "job_key",
            "product_ref",
        ):
            if name in body:
                compact[name] = body[name]
        product = body.get("product")
        if "product_ref" not in compact and isinstance(product, dict) and product.get("product_ref") is not None:
            compact["product_ref"] = product["product_ref"]
        result = body.get("result")
        if "product_ref" not in compact and isinstance(result, dict):
            nested_product = result.get("product")
            if result.get("product_ref") is not None:
                compact["product_ref"] = result["product_ref"]
            elif isinstance(nested_product, dict) and nested_product.get("product_ref") is not None:
                compact["product_ref"] = nested_product["product_ref"]
        if not compact:
            compact["event"] = event_name
        return compact

    def _queue_command_tm(self, result: dict, request_key: dict | None = None) -> None:
        stage = str(result.get("stage", "MISSION_EVENT"))
        event_name = "COMMAND_REJECTED" if stage.endswith("REJECTED") else "COMMAND_ACK"
        compact = self._compact_command_result(result, request_key)
        self._queue_tm({"event_name": event_name, "message": compact, **compact})

    def publish_journal_events(self) -> int:
        """Encode durable async job lifecycle records as APID 2 TM events."""
        count = 0
        for row in self.payload.journal.events_after(self._last_journal_event_id):
            self._last_journal_event_id = int(row["event_id"])
            try:
                body = json.loads(str(row["body_json"]))
                request_key = None if row["request_key_json"] is None else json.loads(str(row["request_key_json"]))
            except (TypeError, ValueError, json.JSONDecodeError):
                logger.warning("journal event %s cannot be encoded as TM", row["event_id"])
                continue
            if not isinstance(body, dict):
                continue
            message: dict = {
                "event_name": str(row["event_name"]),
                "message": self._compact_journal_body(str(row["event_name"]), body),
                "satellite_journal_event_id": int(row["event_id"]),
            }
            if request_key is not None:
                message["request_key"] = request_key
            self._queue_tm(message)
            count += 1
        return count

    def receive_tc_frame(self, frame: bytes) -> dict:
        return self.payload.dispatch_tc_frame(frame)

    def receive_space_packet(self, packet: bytes) -> dict:
        return self.payload.dispatch_space_packet(packet)

    def receive_transport_frame(self, transport_frame) -> dict:
        """Endpoint used by MissionLink after egress envelope validation."""
        request_key = self._request_key_from_transport_frame(transport_frame)
        result = self.payload.receive_transport_frame(transport_frame)
        self._queue_command_tm(result, request_key)
        if result.get("stage") == "EXECUTED" and isinstance(result.get("catalog"), dict):
            self.queue_catalog_snapshot()
        # Queue journal events before allocating the first file frame. The
        # scheduler also carries the durable MCFC order key, but this keeps a
        # new transfer's initial control telemetry naturally adjacent.
        self.publish_journal_events()
        if result.get("stage") == "DISPATCHED" and result.get("transfer_id") is not None:
            self.payload.enqueue_downlink_frame(int(result["transfer_id"]))
        return result

    def next_pending_tm_item(self) -> tuple[bytes, int] | None:
        """Pop one APID 2 frame and its durable global TM ordering key."""

        with self._pending_lock:
            return self._pending_tm_frames.popleft() if self._pending_tm_frames else None

    def next_pending_tm_frame(self) -> bytes | None:
        """Pop one bounded APID 2 control/event frame for a transport adapter."""

        item = self.next_pending_tm_item()
        return None if item is None else item[0]

    def drain_pending_tm_frames(self) -> Iterator[bytes]:
        """Compatibility iterator for bounded APID 2 control/event telemetry."""

        self.publish_journal_events()
        while True:
            frame = self.next_pending_tm_frame()
            if frame is None:
                return
            yield frame

    def enqueue_pending_tm_frames(self) -> int:
        """Move APID 2 telemetry into the shared completion-gated scheduler.

        Both the local loopback and UDP service call this method so priority
        selection cannot reorder frames after MCFC/VCFC allocation.
        """

        self.publish_journal_events()
        queued = 0
        with self._pending_lock:
            while self._pending_tm_frames:
                frame, ordering_key = self._pending_tm_frames[0]
                self.payload.scheduler.enqueue_ack(frame, ordering_key=ordering_key)
                self._pending_tm_frames.popleft()
                queued += 1
        return queued

    def iter_scheduled_tm_frames(self) -> Iterator[bytes]:
        """Stream scheduler-owned frames and release each only after the caller resumes."""

        self.publish_journal_events()
        while True:
            item = self.payload.scheduler.poll()
            if item is None:
                return
            try:
                yield item.frame
            except BaseException:
                self.payload.scheduler.mark_status(item.item_id, "FRAME_FAILED")
                self.payload.scheduler.mark_upstream_return(item.item_id)
                raise
            else:
                self.payload.scheduler.mark_status(item.item_id, "LINK_CONSUMED")
                self.payload.scheduler.mark_upstream_return(item.item_id)

    def drain_downlink_frames(self, transfer_id: int) -> Iterator[bytes]:
        """Compatibility streaming adapter for a single scheduled file transfer."""

        self.payload.enqueue_downlink_frame(transfer_id)
        return self.iter_scheduled_tm_frames()

    def health(self) -> dict:
        return self.payload.health()

    def close(self) -> None:
        self.payload.close()
        self.deployment.close()


class SatelliteUdpService:
    """Run the flight endpoint against LinkSimulator over the common UDP ABI."""

    def __init__(
        self,
        simulator: SatelliteSimulator,
        *,
        bind_host: str,
        bind_port: int,
        link_host: str,
        link_port: int,
        link_session_id: int,
        expected_sender_boot_id: int | None = None,
        status_interval_s: float = 1.0,
    ) -> None:
        self.simulator = simulator
        actual_boot_id = simulator.payload.journal.boot_id
        if expected_sender_boot_id is not None and actual_boot_id != expected_sender_boot_id:
            raise RuntimeError(
                "SATELLITE_BOOT_ID_MISMATCH: configured link sender boot ID "
                f"{expected_sender_boot_id} does not match durable satellite boot ID {actual_boot_id}"
            )
        if status_interval_s <= 0:
            raise ValueError("status_interval_s must be positive")
        self._session_binding: dict[str, int] | None = None
        self.session_resets = 0
        self.adapter: MissionUdpAdapter | None = None
        self.endpoint = UdpMissionEndpoint(
            bind_addr=(bind_host, bind_port),
            link_addr=(link_host, link_port),
            spacecraft_instance_id=simulator.payload.profile.spacecraft_instance_id,
            link_session_id=link_session_id,
            sender_boot_id=actual_boot_id,
            handshake_role="satellite",
            on_session=self._on_session_binding,
        )
        self.adapter = MissionUdpAdapter(simulator.payload.scheduler, self._send_frame)
        self.endpoint.establish_session()
        self.frames_sent = 0
        self.frames_received = 0
        self.tm_seen = False
        self.tc_seen = False
        self._status_interval_s = float(status_interval_s)
        self._last_status_monotonic = time.monotonic()

    def _on_session_binding(self, binding: dict[str, int]) -> None:
        previous = self._session_binding
        self._session_binding = dict(binding)
        if previous is None or self.adapter is None:
            return
        if binding != previous:
            # No stale file/control frame may survive a LinkSimulator reset.
            self.adapter.reset()
            self.simulator.payload.scheduler.set_ready()
            self.session_resets += 1

    @property
    def ready(self) -> bool:
        return self.endpoint.ready and self.tm_seen

    def _send_frame(self, frame: bytes, status_callback, return_callback) -> None:
        try:
            self.endpoint.send_ingress(frame)
            self.frames_sent += 1
            self.tm_seen = True
            status_callback("UDP_SENT")
        except Exception as exc:
            logger.warning("TM UDP send failed: %s", exc)
            status_callback("FRAME_FAILED")
        finally:
            return_callback()

    def _enqueue_pending_tm(self) -> int:
        try:
            return self.simulator.enqueue_pending_tm_frames()
        except Exception as exc:
            logger.warning("TM scheduler enqueue failed: %s", exc)
            return 0

    def pump(self, timeout_ms: int = 0) -> int:
        """Receive one TC, collect async TM, and drain the completion-gated queue."""
        frame: TransportFrame | None = self.endpoint.receive_egress(timeout_ms)
        if frame is not None:
            self.frames_received += 1
            self.tc_seen = True
            self.simulator.receive_transport_frame(frame)
        now = time.monotonic()
        if now - self._last_status_monotonic >= self._status_interval_s:
            self.simulator.queue_status_tm()
            self._last_status_monotonic = now
        self._enqueue_pending_tm()
        sent = 0
        assert self.adapter is not None
        while self.adapter.gate is None:
            item = self.adapter.send_next()
            if item is None:
                break
            sent += 1
        return sent

    def health(self) -> dict:
        return {
            **self.simulator.health(),
            "udp": {
                "ready": self.ready,
                "bound_address": self.endpoint.bound_address,
                "frames_sent": self.frames_sent,
                "frames_received": self.frames_received,
                "tc_seen": self.tc_seen,
                "tm_seen": self.tm_seen,
                "session_resets": self.session_resets,
                "link_generation": getattr(self.endpoint, "link_generation", None),
            },
        }

    def close(self) -> None:
        self.endpoint.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local satellite simulator reference profile")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--state-directory", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    parser.add_argument(
        "--status-interval",
        type=_non_negative_float,
        default=5.0,
        help="seconds between realtime status lines; use 0 to disable",
    )
    parser.add_argument("--health-once", action="store_true")
    parser.add_argument("--roi-smoke", action="store_true")
    parser.add_argument("--udp", action="store_true", help="bind the satellite mission UDP endpoint")
    parser.add_argument("--udp-bind-host", default="0.0.0.0")
    parser.add_argument("--udp-bind-port", type=int, default=9002)
    parser.add_argument("--link-host", default=None)
    parser.add_argument("--link-port", type=int, default=None)
    parser.add_argument("--link-session-id", type=int, default=None)
    args = parser.parse_args()
    configure_realtime_logging(args.log_level)
    logger.info(
        "startup root=%s device=%s status_interval_s=%s",
        args.root.resolve(),
        args.device,
        args.status_interval,
    )
    try:
        simulator = SatelliteSimulator(args.root, state_directory=args.state_directory, device=args.device)
    except Exception:
        logger.exception("startup_failed")
        raise
    link_host = args.link_host or os.environ.get("CUBE_NANO_LINK_HOST", "127.0.0.1")
    link_port = args.link_port if args.link_port is not None else int(os.environ.get("CUBE_NANO_LINK_PORT", "9000"))
    link_session_id = args.link_session_id if args.link_session_id is not None else int(os.environ.get("CUBE_NANO_LINK_SESSION_ID", "1"))
    expected_boot_text = os.environ.get("CUBE_NANO_SENDER_BOOT_ID")
    sender_boot_id = None if expected_boot_text is None else int(expected_boot_text)
    udp_enabled = args.udp or os.environ.get("CUBE_NANO_LINK_MODE", "").lower() == "udp"
    udp_service: SatelliteUdpService | None = None
    if udp_enabled:
        try:
            udp_service = SatelliteUdpService(
                simulator,
                bind_host=args.udp_bind_host,
                bind_port=args.udp_bind_port,
                link_host=link_host,
                link_port=link_port,
                link_session_id=link_session_id,
                expected_sender_boot_id=sender_boot_id,
            )
        except Exception:
            simulator.close()
            logger.exception("udp_startup_failed")
            raise
    _log_health(simulator.health(), label="ready")
    print(json.dumps(udp_service.health() if udp_service is not None else simulator.health(), sort_keys=True))
    if args.roi_smoke:
        logger.info("roi_smoke_started")
        config = simulator.payload.journal.current_config()
        request_key = RequestKey(0x736D6F6B65000001, time.time_ns() & 0xFFFFFFFF)
        command = Command(
            CommandOpcode.ROI_REQUEST,
            simulator.payload.profile.spacecraft_instance_id,
            request_key,
            {
                "scene_ref": SceneRef(1, 1, 1).as_dict(),
                "roi": ROI(0, 0, 256, 256).as_dict(),
                "expected_config_epoch": config.epoch,
                "expected_config_revision": config.revision,
                "model_threshold_bp": config.model_threshold_bp,
                "coverage_limit_bp": config.coverage_limit_bp,
            },
        )
        admission = simulator.payload.handle_command(command)
        simulator.payload.wait_for_jobs(60)
        row = simulator.payload.journal.get_job(request_key)
        downlink = None
        transfer_state = None
        frame_count = 0
        if row is not None and row["state"] == "SUCCEEDED":
            downlink_key = RequestKey(
                request_key.ground_instance_id,
                (request_key.request_id + 1) & 0xFFFFFFFF,
            )
            downlink_command = Command(
                CommandOpcode.PRODUCT_REQUEST_DOWNLINK,
                simulator.payload.profile.spacecraft_instance_id,
                downlink_key,
                {
                    "origin_request_key": request_key.as_dict(),
                    "product_ref": admission["product_ref"],
                },
            )
            downlink = simulator.payload.handle_command(downlink_command)
            if downlink.get("stage") == "DISPATCHED":
                frame_count = sum(
                    1
                    for _frame in simulator.payload.drain_downlink(
                        int(downlink["transfer_id"])
                    )
                )
                transfer = simulator.payload.journal.get_transfer(int(downlink["transfer_id"]))
                transfer_state = transfer["state"] if transfer else None
        print(
            json.dumps(
                {
                    "admission": admission,
                    "job_state": row["state"] if row else None,
                    "error_code": row["error_code"] if row else "JOB_MISSING",
                    "downlink": downlink,
                    "downlink_frame_count": frame_count,
                    "transfer_state": transfer_state,
                },
                sort_keys=True,
            )
        )
        simulator.close()
        logger.info("stopped mode=roi_smoke")
        return
    if args.health_once:
        if udp_service is not None:
            udp_service.pump(timeout_ms=0)
            print(json.dumps(udp_service.health(), sort_keys=True))
            udp_service.close()
        simulator.close()
        logger.info("stopped mode=health_once")
        return
    stop = False

    def request_stop(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, request_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, request_stop)
    try:
        last_status = 0.0
        while not stop:
            now = time.monotonic()
            if udp_service is not None:
                udp_service.pump(timeout_ms=100)
            if args.status_interval > 0 and now - last_status >= args.status_interval:
                _log_health(udp_service.health() if udp_service is not None else simulator.health())
                last_status = now
            if udp_service is None:
                time.sleep(0.5)
    finally:
        logger.info("stopping")
        if udp_service is not None:
            udp_service.close()
        simulator.close()
        logger.info("stopped mode=daemon")


if __name__ == "__main__":
    main()
