"""CloudPayload command dispatcher and AI worker for the local satellite SIL."""

from __future__ import annotations

import json
import logging
import math
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Iterator

from protocol.ccsds import TcTypeBdFrame, encode_tm_frame
from protocol.messages import PacketDescriptor, encode_tm_application
from protocol.schemas import (
    Command,
    CommandOpcode,
    ConfigSnapshot,
    ProductRef,
    RequestKey,
    ROI,
    mission_digest,
)
from sat_ai.roi import iter_patch_windows
from sat_ai.products import cleanup_staging_products
from sat_ai.worker_contract import DeadlineContract, WorkerRequest, WorkerResult, WorkerResultState

from .catalog import SceneCatalog
from .deployment import SatelliteDeployment
from .file_downlink import FileDownlinkCoordinator, FrameLease, TransferState
from .journal import SatelliteJournal
from .mission_com_scheduler import MissionComScheduler, QueueKind
from .stock_router import StockApidRouter
from .worker_client import WorkerProcessClient, WorkerProcessPolicy, WorkerQueueFull


logger = logging.getLogger(__name__)


class CloudPayload:
    """The business boundary behind stock APID 0 command dispatch."""

    def __init__(
        self,
        deployment: SatelliteDeployment,
        *,
        device: str = "cpu",
        start_worker: bool = True,
        event_clock: Callable[[], int] | None = None,
        event_time_base: str = "monotonic",
    ):
        if not deployment.ready:
            raise RuntimeError(f"satellite deployment is not READY: {deployment.readiness.reason}")
        assert deployment.profile and deployment.manifest and deployment.lut and deployment.catalog and deployment.journal
        self.deployment = deployment
        self.profile = deployment.profile
        self.manifest = deployment.manifest
        self.lut = deployment.lut
        self.catalog: SceneCatalog = deployment.catalog
        self.journal: SatelliteJournal = deployment.journal
        self._event_clock = event_clock or time.monotonic_ns
        self._event_time_base = str(event_time_base)
        self.router = StockApidRouter(
            self.profile.tc_apid,
            expected_packet_type=1,
            expected_secondary_header_present=False,
            expected_sequence_flags=3,
        )
        self.scheduler = MissionComScheduler(
            capacities={
                QueueKind.ACK: self.profile.ack_mailbox_capacity,
                QueueKind.CONTROL: self.profile.control_queue_capacity,
                QueueKind.FILE: self.profile.file_queue_capacity,
            },
            ack_burst=self.profile.ack_burst,
            control_burst=self.profile.control_burst,
            file_burst=self.profile.file_burst,
        )
        self.file_downlink = FileDownlinkCoordinator(
            tm_file_apid=self.profile.tm_file_apid,
            spacecraft_id=self.profile.spacecraft_id,
            state_callback=self._persist_transfer_state,
            tm_counter_allocator=lambda apid: self.journal.allocate_tm_frame_counters(
                apid,
                virtual_channel_id=self.profile.tm_virtual_channel,
            ),
        )
        self.worker_client: WorkerProcessClient | None = None
        logger.info(
            "payload_start spacecraft_instance_id=%016x boot_id=%s device=%s",
            self.profile.spacecraft_instance_id,
            self.journal.boot_id,
            device,
        )
        if start_worker:
            self.worker_client = WorkerProcessClient(
                deployment.root,
                device=device,
                policy=WorkerProcessPolicy(
                    max_pending_jobs=self.profile.max_pending_jobs,
                    heartbeat_interval_ms=self.profile.worker_heartbeat_interval_ms,
                    heartbeat_timeout_ms=self.profile.worker_heartbeat_timeout_ms,
                    max_restarts=self.profile.max_worker_restarts,
                    restart_window_ms=self.profile.worker_restart_window_ms,
                    initial_backoff_ms=self.profile.worker_initial_backoff_ms,
                    cancel_grace_ms=self.profile.worker_cancel_grace_ms,
                ),
                on_result=self._handle_worker_result,
                on_started=self._handle_worker_started,
                on_state_change=self.deployment.set_worker_state,
            )
            try:
                self.worker_client.start()
            except Exception:
                self.deployment.set_worker_state("FAULT")
                logger.exception("worker_start_failed")
                raise
            logger.info("payload_ready worker_state=%s", self.worker_client.health()["state"])

    def _next_tm_sequence(self, apid: int) -> int:
        return self.journal.allocate_tm_packet_sequence(apid)[0]

    def health(self) -> dict[str, Any]:
        worker_health = self.worker_client.health() if self.worker_client is not None else {
            "state": "STOPPED",
            "heartbeat_age_ms": None,
            "queue_depth": 0,
            "queue_capacity": self.profile.max_pending_jobs,
            "active_request_id": None,
            "restart_count": 0,
        }
        return {
            "state": self.deployment.state.value,
            "spacecraft_instance_id": f"{self.profile.spacecraft_instance_id:016x}",
            "sender_boot_id": self.journal.boot_id,
            "worker_state": worker_health["state"],
            "worker_heartbeat_age_ms": worker_health["heartbeat_age_ms"],
            "worker_restart_count": worker_health["restart_count"],
            "active_request_id": worker_health["active_request_id"],
            "queue_depth": worker_health["queue_depth"],
            "queue_capacity": worker_health["queue_capacity"],
            "config": self.journal.current_config().as_dict(),
            "model_release_id": self.manifest.model_release_id,
            "model_sha256": self.manifest.checkpoint_sha256,
            "assurance_level": self.manifest.assurance_level,
            "catalog_epoch": self.catalog.epoch,
            "catalog_revision": self.catalog.revision,
            "scheduler_queue_depths": self.scheduler.queue_depths(),
            "scheduler_metrics": dict(self.scheduler.metrics),
        }

    def dispatch_tc_frame(self, frame_bytes: bytes) -> dict[str, Any]:
        try:
            frame = TcTypeBdFrame.decode(frame_bytes)
        except (TypeError, ValueError) as exc:
            message = str(exc).upper()
            if "CRC" in message or "FECF" in message:
                error_code = "TC_CRC_INVALID"
            elif "TYPE-BD" in message or "RESERVED" in message or "VERSION" in message:
                error_code = "TC_FRAME_HEADER_INVALID"
            elif "LENGTH" in message or "SPACE PACKET" in message:
                error_code = "TC_PACKET_INVALID"
            else:
                error_code = "TC_FRAME_INVALID"
            self._record_frame_rejection(error_code, frame_bytes)
            return self._error_response(None, error_code)
        if frame.spacecraft_id != self.profile.spacecraft_id:
            error_code = "TC_SCID_MISMATCH"
            self._record_frame_rejection(error_code, frame_bytes)
            return self._error_response(None, error_code)
        if frame.virtual_channel_id != self.profile.tc_virtual_channel:
            error_code = "TC_VCID_MISMATCH"
            self._record_frame_rejection(error_code, frame_bytes)
            return self._error_response(None, error_code)
        return self.dispatch_space_packet(frame.packet.encode())

    def receive_transport_frame(self, transport_frame) -> dict[str, Any]:
        """Receive only a validated egress frame from the mission link."""
        envelope = transport_frame.envelope
        envelope.validate_egress(
            expected_spacecraft_instance_id=self.profile.spacecraft_instance_id,
            expected_sender_boot_id=self.journal.boot_id,
        )
        return self.dispatch_tc_frame(transport_frame.frame_bytes)

    def dispatch_space_packet(self, packet_bytes: bytes) -> dict[str, Any]:
        routed = self.router.route_tc(packet_bytes)
        if not routed.accepted or routed.command is None:
            error_code = routed.error_code or "INVALID_COMMAND"
            self._record_frame_rejection(error_code, packet_bytes)
            return self._error_response(None, error_code)
        return self.handle_command(routed.command)

    def _record_frame_rejection(self, error_code: str, frame_bytes: bytes) -> None:
        """Persist malformed/no-command rejects so they remain auditable."""
        import hashlib

        self.journal.append_event(
            "TC_FRAME_REJECTED",
            {
                "error_code": str(error_code),
                "frame_sha256": hashlib.sha256(bytes(frame_bytes)).hexdigest(),
                "frame_length": len(frame_bytes),
            },
        )

    def _error_response(self, command: Command | None, error_code: str) -> dict[str, Any]:
        result = {"stage": "COMMAND_REJECTED", "error_code": error_code}
        if command is not None:
            result["request_key"] = command.request_key.as_dict()
        return result

    def _record_rejection(self, command: Command, digest: str, error_code: str) -> dict[str, Any]:
        result = self._error_response(command, error_code)
        logger.warning(
            "command_rejected opcode=%s request_key=%s error_code=%s",
            getattr(command.opcode, "name", int(command.opcode)),
            command.request_key.as_dict(),
            error_code,
        )
        self.journal.record_command(command.request_key, int(command.opcode), digest, command.semantic_dict(), "COMMAND_REJECTED", result)
        self.journal.append_event("COMMAND_REJECTED", result, command.request_key)
        return result

    def handle_command(self, command: Command) -> dict[str, Any]:
        digest = mission_digest(command)
        logger.info(
            "command_received opcode=%s request_key=%s target=%016x",
            getattr(command.opcode, "name", int(command.opcode)),
            command.request_key.as_dict(),
            command.target_spacecraft_instance_id,
        )
        if command.target_spacecraft_instance_id != self.profile.spacecraft_instance_id:
            return self._record_rejection(command, digest, "TARGET_INSTANCE_MISMATCH")
        lookup, cached = self.journal.lookup_request(command.request_key, digest)
        if lookup == "DUPLICATE":
            logger.info("command_duplicate request_key=%s", command.request_key.as_dict())
            return cached or {"stage": "DUPLICATE_REQUEST_RETIRED", "request_key": command.request_key.as_dict()}
        if lookup == "CONFLICT":
            logger.warning("command_conflict request_key=%s", command.request_key.as_dict())
            return self._error_response(command, "DUPLICATE_REQUEST_CONFLICT")
        if lookup == "RETIRED":
            logger.warning("command_retired request_key=%s", command.request_key.as_dict())
            return self._error_response(command, "DUPLICATE_REQUEST_RETIRED")
        try:
            if command.opcode == CommandOpcode.CLOUD_SET_CONFIG:
                return self._handle_set_config(command, digest)
            if command.opcode == CommandOpcode.SCENE_REQUEST_CATALOG:
                result = {"stage": "EXECUTED", "catalog": self.catalog.snapshot()}
                self.journal.record_command(command.request_key, int(command.opcode), digest, command.semantic_dict(), "EXECUTED", result)
                return result
            if command.opcode == CommandOpcode.SCENE_REQUEST_PREVIEW:
                self.catalog.get(self._scene_ref(command))
                result = {"stage": "EXECUTED", "preview_policy": "PRODUCT_REQUEST_REQUIRED", "scene_ref": command.payload["scene_ref"]}
                self.journal.record_command(command.request_key, int(command.opcode), digest, command.semantic_dict(), "EXECUTED", result)
                return result
            if command.opcode in {CommandOpcode.SCENE_ANALYZE, CommandOpcode.ROI_REQUEST}:
                return self._admit_analysis(command, digest)
            if command.opcode == CommandOpcode.JOB_GET_STATUS:
                return self._job_status(command, digest)
            if command.opcode == CommandOpcode.JOB_CANCEL:
                return self._job_cancel(command, digest)
            if command.opcode == CommandOpcode.PRODUCT_REQUEST_DOWNLINK:
                return self._admit_downlink(command, digest)
            if command.opcode == CommandOpcode.PRODUCT_CANCEL_DOWNLINK:
                result = self._cancel_downlink(command)
                self.journal.record_command(command.request_key, int(command.opcode), digest, command.semantic_dict(), "EXECUTED", result)
                return result
        except ValueError as exc:
            return self._record_rejection(command, digest, str(exc))
        except RuntimeError as exc:
            return self._record_rejection(command, digest, str(exc))
        return self._record_rejection(command, digest, "UNKNOWN_OPCODE")

    @staticmethod
    def _scene_ref(command: Command):
        from protocol.schemas import SceneRef

        return SceneRef.from_dict(command.payload.get("scene_ref"))

    def _handle_set_config(self, command: Command, digest: str) -> dict[str, Any]:
        payload = command.payload
        snapshot, result = self.journal.apply_config_command(
            command.request_key,
            int(command.opcode),
            digest,
            command.semantic_dict(),
            int(payload["expected_config_epoch"]),
            int(payload["expected_config_revision"]),
            int(payload["model_threshold_bp"]),
            int(payload["coverage_limit_bp"]),
        )
        self.journal.append_event("COMMAND_ACCEPTED", result, command.request_key)
        return result

    def _validate_config_payload(self, payload: dict[str, Any]) -> ConfigSnapshot:
        current = self.journal.current_config()
        expected = ConfigSnapshot(
            int(payload["expected_config_epoch"]),
            int(payload["expected_config_revision"]),
            int(payload["model_threshold_bp"]),
            int(payload["coverage_limit_bp"]),
        )
        if (current.epoch, current.revision) != (expected.epoch, expected.revision):
            raise ValueError("CONFIG_REVISION_MISMATCH")
        if current.model_threshold_bp != expected.model_threshold_bp or current.coverage_limit_bp != expected.coverage_limit_bp:
            raise ValueError("CONFIG_SNAPSHOT_MISMATCH")
        return current

    def _job_deadline_ms(self, scene_shape: tuple[int, int, int], roi: ROI) -> tuple[int, int]:
        patch_count = len(
            list(iter_patch_windows(scene_shape, roi, self.manifest.input_spec.patch_size))
        )
        if patch_count <= 0:
            raise ValueError("INVALID_ROI")
        assert self.deployment.benchmark_artifact is not None
        p99_ms = float(self.deployment.benchmark_artifact["measurements"]["p99_latency_ms"])
        deadline_ms = max(5000, math.ceil(p99_ms * patch_count * 4.0))
        return patch_count, min(deadline_ms, 0xFFFFFFFF)

    def _admit_analysis(self, command: Command, digest: str) -> dict[str, Any]:
        scene_ref = self._scene_ref(command)
        scene = self.catalog.get(scene_ref)
        domain_status = self.manifest.domain_status(scene.domain)
        if domain_status == "DOMAIN_MISMATCH":
            raise ValueError("DOMAIN_MISMATCH")
        snapshot = self._validate_config_payload(command.payload)
        if command.opcode == CommandOpcode.ROI_REQUEST:
            roi = ROI.from_dict(command.payload.get("roi"))
        else:
            roi = ROI(0, 0, scene.shape[1], scene.shape[0])
        if roi.x_end > scene.shape[1] or roi.y_end > scene.shape[0]:
            raise ValueError("INVALID_ROI")
        product_id = self.journal.allocate_product_id()
        product_ref = ProductRef(self.profile.spacecraft_instance_id, self.journal.boot_id, product_id)
        patch_count, deadline_ms = self._job_deadline_ms(scene.shape, roi)
        deadline = DeadlineContract.after_ms(deadline_ms)
        immutable_snapshot = {
            "model_release_id": self.manifest.model_release_id,
            "model_sha256": self.manifest.checkpoint_sha256,
            "input_spec_id": self.manifest.input_spec.input_spec_id,
            "input_contract": self.manifest.input_contract(),
            "domain_status": domain_status,
            "threshold_mapping_id": self.lut.lut_id,
            "threshold_lut_sha256": self.lut.sha256,
            "config_snapshot": snapshot.as_dict(),
            "scene_ref": scene_ref.as_dict(),
            "roi": roi.as_dict(),
            "scene": scene.as_dict(),
            "product_ref": product_ref.as_dict(),
            "origin_request_key": command.request_key.as_dict(),
            "product_root": str(self.deployment.product_directory),
            "patch_count": patch_count,
            "job_deadline_ms": deadline_ms,
        }
        result = self.journal.admit_analysis(
            command.request_key,
            int(command.opcode),
            digest,
            command.semantic_dict(),
            scene_ref.as_dict(),
            roi.as_dict(),
            snapshot,
            immutable_snapshot,
            product_ref,
        )
        try:
            if self.worker_client is None:
                raise RuntimeError("SERVICE_FAULT")
            self.worker_client.submit(WorkerRequest(command.request_key, immutable_snapshot, deadline))
        except (WorkerQueueFull, RuntimeError) as exc:
            error_code = "QUEUE_FULL" if isinstance(exc, WorkerQueueFull) else str(exc)
            self.journal.transition_job(command.request_key, {"QUEUED"}, "FAILED", error_code=error_code)
            self.journal.fail_product_for_job(command.request_key, error_code)
            result = {"stage": "COMMAND_REJECTED", "error_code": error_code}
            self.journal.update_command_result(command.request_key, "COMMAND_REJECTED", result)
            self.journal.append_event(error_code, result, command.request_key)
            return result
        self.journal.append_event("COMMAND_ACCEPTED", result, command.request_key)
        self.journal.append_event("JOB_QUEUED", result, command.request_key)
        return result

    def _job_status(self, command: Command, digest: str) -> dict[str, Any]:
        target = RequestKey.from_dict(command.payload["target_request_key"])
        row = self.journal.get_job(target)
        if row is None:
            result = self._error_response(command, "TARGET_RETIRED")
        else:
            result = {"stage": "EXECUTED", "job_key": target.as_dict(), "state": row["state"], "result": json.loads(row["result_json"]) if row["result_json"] else None, "error_code": row["error_code"]}
        self.journal.record_command(command.request_key, int(command.opcode), digest, command.semantic_dict(), "EXECUTED", result)
        return result

    def _job_cancel(self, command: Command, digest: str) -> dict[str, Any]:
        target = RequestKey.from_dict(command.payload["target_request_key"])
        row = self.journal.get_job(target)
        if row is None:
            result = self._error_response(command, "TARGET_RETIRED")
        elif row["state"] in {"SUCCEEDED", "FAILED", "TIMEOUT", "CANCELED"}:
            result = {"stage": "EXECUTED", "cancel_outcome": "ALREADY_TERMINAL", "state": row["state"]}
        else:
            self.journal.transition_job(target, {"QUEUED", "RUNNING"}, "CANCEL_REQUESTED", error_code="CANCEL_REQUESTED")
            try:
                outcome = self.worker_client.cancel(target) if self.worker_client is not None else "CANCELED"
            except WorkerQueueFull:
                outcome = "CANCEL_REQUESTED"
            except RuntimeError as exc:
                outcome = "CANCEL_FAILED"
                self.journal.transition_job(
                    target,
                    {"CANCEL_REQUESTED"},
                    "FAILED",
                    error_code="CANCEL_FAILED",
                    result={"error": str(exc)},
                )
                self.journal.fail_product_for_job(target, "CANCEL_FAILED")
            if self.worker_client is None:
                self.journal.transition_job(target, {"CANCEL_REQUESTED"}, "CANCELED", error_code="WORKER_CANCELED")
                self.journal.fail_product_for_job(target, "WORKER_CANCELED")
            current = self.journal.get_job(target)
            result = {
                "stage": "EXECUTED",
                "cancel_outcome": outcome,
                "state": current["state"] if current is not None else "CANCELED",
            }
        self.journal.record_command(command.request_key, int(command.opcode), digest, command.semantic_dict(), "EXECUTED", result)
        return result

    def _admit_downlink(self, command: Command, digest: str) -> dict[str, Any]:
        if self.journal.get_active_transfer() is not None:
            raise RuntimeError("TRANSFER_BUSY")
        product = ProductRef.from_dict(command.payload["product_ref"])
        if product.spacecraft_instance_id != self.profile.spacecraft_instance_id:
            raise ValueError("PRODUCT_TARGET_INSTANCE_MISMATCH")
        row = self.journal.get_product(product)
        if row is None or row["state"] != "READY":
            raise ValueError("PRODUCT_NOT_READY")
        origin = RequestKey.from_dict(command.payload["origin_request_key"])
        if row["origin_ground_instance_id"] != origin.ground_instance_id.to_bytes(8, "big") or row["origin_request_id"] != origin.request_id:
            raise ValueError("ORIGIN_REQUEST_MISMATCH")
        transfer_id = self.journal.allocate_transfer_id()
        result = self.journal.admit_downlink(command.request_key, int(command.opcode), digest, command.semantic_dict(), transfer_id, product)
        self.journal.append_event("PRODUCT_DOWNLINK_STARTED", result, command.request_key)
        return result

    def _persist_transfer_state(self, transfer_id: int, state: TransferState, reason: str | None) -> None:
        if self.journal.get_transfer(transfer_id) is not None:
            self.journal.update_transfer(transfer_id, state.value, reason)

    def _start_downlink(self, transfer_id: int) -> None:
        row = self.journal.get_transfer(transfer_id)
        if row is None:
            raise ValueError("TRANSFER_NOT_FOUND")
        if row["state"] != "QUEUED":
            raise RuntimeError("TRANSFER_NOT_QUEUED")
        product = ProductRef.from_dict(json.loads(row["product_ref_json"]))
        product_row = self.journal.get_product(product)
        if product_row is None or product_row["state"] != "READY" or not product_row["path"]:
            raise ValueError("PRODUCT_NOT_READY")
        bundle_path = Path(product_row["path"]) / "bundle.tar"
        self.file_downlink.start(transfer_id, product, bundle_path)

    def has_queued_downlink(self, transfer_id: int) -> bool:
        """Whether the bounded FILE slot already owns a frame for this transfer."""

        active = self.file_downlink.active
        return bool(
            active is not None
            and active.transfer_id == transfer_id
            and active.current_lease is not None
        )

    def _abort_downlink(self, transfer_id: int, reason: str) -> None:
        active = self.file_downlink.active
        if active is None or active.transfer_id != transfer_id or active.state in {
            TransferState.SEND_COMPLETED,
            TransferState.SEND_FAILED,
            TransferState.CANCELED,
        }:
            return
        self.file_downlink.begin_abort(transfer_id, reason)
        if active.current_lease is None:
            self.close_downlink_abort_fence(transfer_id)

    def _finish_downlink_if_terminal(self, transfer_id: int) -> None:
        active = self.file_downlink.active
        if active is None or active.transfer_id != transfer_id:
            return
        if active.state == TransferState.ABORTING and active.current_lease is None:
            self.close_downlink_abort_fence(transfer_id)
            return
        if active.state == TransferState.SEND_COMPLETED:
            self.journal.append_event(
                "PRODUCT_SEND_COMPLETED",
                {"transfer_id": transfer_id, "frame_count": active.next_file_sequence},
            )

    def _on_downlink_frame_complete(self, lease: FrameLease, _status: str, completed: bool) -> None:
        """Advance exactly one completion-gated file frame at a time."""

        if not completed:
            return
        active = self.file_downlink.active
        if active is None or active.transfer_id != lease.transfer_id:
            return
        if active.state in {TransferState.SENDING, TransferState.CANCEL_REQUESTED}:
            try:
                self.enqueue_downlink_frame(lease.transfer_id)
            except Exception as exc:
                self._abort_downlink(lease.transfer_id, type(exc).__name__)
            return
        self._finish_downlink_if_terminal(lease.transfer_id)

    def enqueue_downlink_frame(self, transfer_id: int) -> int | None:
        """Queue one FILE frame; its completion schedules at most one successor."""

        active = self.file_downlink.active
        if active is None or active.transfer_id != transfer_id:
            self._start_downlink(transfer_id)
            active = self.file_downlink.active
        assert active is not None
        if active.current_lease is not None:
            return None
        try:
            return self.file_downlink.enqueue_next(
                transfer_id,
                self.scheduler,
                on_complete=self._on_downlink_frame_complete,
            )
        except Exception as exc:
            self._abort_downlink(transfer_id, type(exc).__name__)
            raise

    def next_downlink_frame(self, transfer_id: int) -> FrameLease | None:
        """Return one leased file frame for a non-scheduler transport adapter."""

        active = self.file_downlink.active
        if active is None or active.transfer_id != transfer_id:
            self._start_downlink(transfer_id)
            active = self.file_downlink.active
        assert active is not None
        if active.current_lease is not None:
            return None
        return self.file_downlink.next_frame(transfer_id)

    def complete_downlink_frame(self, lease: FrameLease, status: str) -> bool:
        """Complete one lease; callers request the next frame explicitly."""

        completed = self.file_downlink.complete_frame(lease, status)
        if completed:
            active = self.file_downlink.active
            if active is not None and active.transfer_id == lease.transfer_id:
                while active.state == TransferState.COOLDOWN and active.epoch_closed:
                    self.file_downlink.cooldown_tick(lease.transfer_id)
                self._finish_downlink_if_terminal(lease.transfer_id)
        return completed

    def close_downlink_abort_fence(self, transfer_id: int) -> str:
        """Apply stock-Cancel/file-epoch acknowledgements, then drain cooldown."""
        self.file_downlink.close_abort_fence(transfer_id)
        active = self.file_downlink.active
        assert active is not None
        while active.state == TransferState.COOLDOWN:
            self.file_downlink.cooldown_tick(transfer_id)
        self.journal.append_event(
            "PRODUCT_SEND_FAILED",
            {"transfer_id": transfer_id, "error_code": active.abort_reason},
        )
        return active.state.value

    def iter_downlink_frames(self, transfer_id: int) -> Iterator[bytes]:
        """Compatibility iterator that streams a fault-free transfer one frame at a time."""

        try:
            while True:
                lease = self.next_downlink_frame(transfer_id)
                if lease is None:
                    break
                yield lease.frame
                if not self.complete_downlink_frame(lease, "SUCCESS"):
                    raise RuntimeError("FILE_FRAME_COMPLETION_REJECTED")
        except Exception as exc:
            self._abort_downlink(transfer_id, type(exc).__name__)
            raise
        active = self.file_downlink.active
        if active is None or active.state != TransferState.SEND_COMPLETED:
            raise RuntimeError("TRANSFER_DID_NOT_COMPLETE")

    def drain_downlink(self, transfer_id: int) -> Iterator[bytes]:
        """Deprecated bounded alias retained for local-SIL callers."""

        return self.iter_downlink_frames(transfer_id)

    def _cancel_downlink(self, command: Command) -> dict[str, Any]:
        payload = command.payload
        product = ProductRef.from_dict(payload["product_ref"])
        if product.spacecraft_instance_id != self.profile.spacecraft_instance_id:
            raise ValueError("PRODUCT_TARGET_INSTANCE_MISMATCH")
        transfer_id = int(payload["transfer_id"])
        active = self.file_downlink.active
        if active is not None and active.transfer_id == transfer_id:
            if active.product_ref != product:
                raise ValueError("TRANSFER_PRODUCT_MISMATCH")
            outcome = self.file_downlink.cancel(transfer_id)
            return {"stage": "EXECUTED", "cancel_outcome": outcome, "transfer_id": transfer_id}
        row = self.journal.get_transfer(transfer_id)
        if row is None:
            raise ValueError("TRANSFER_NOT_FOUND")
        if ProductRef.from_dict(json.loads(row["product_ref_json"])) != product:
            raise ValueError("TRANSFER_PRODUCT_MISMATCH")
        if row["state"] == "QUEUED":
            self.journal.update_transfer(transfer_id, "CANCELED", "MISSION_CANCEL")
            outcome = "CANCELED"
        else:
            outcome = "ALREADY_TERMINAL"
        return {"stage": "EXECUTED", "cancel_outcome": outcome, "transfer_id": transfer_id}

    def _handle_worker_started(self, request: WorkerRequest) -> None:
        if self.journal.transition_job(request.request_key, {"QUEUED"}, "RUNNING"):
            self.journal.append_event(
                "JOB_STARTED",
                {
                    "job_key": request.request_key.as_dict(),
                    "deadline_monotonic_ns": request.deadline.deadline_monotonic_ns,
                },
                request.request_key,
            )

    def _discard_worker_product(self, result: dict[str, Any] | None) -> None:
        if not result or not isinstance(result.get("product"), dict):
            return
        path_value = result["product"].get("product_directory")
        if not path_value:
            return
        product_root = self.deployment.product_directory.resolve()
        path = Path(str(path_value)).resolve()
        if path != product_root and product_root in path.parents and path.is_dir():
            shutil.rmtree(path)

    def _discard_product_ref(self, product_ref: ProductRef | None) -> None:
        if product_ref is None:
            return
        path = (
            self.deployment.product_directory
            / f"{product_ref.origin_boot_id:08x}"
            / f"{product_ref.product_id:08x}"
        )
        if path.is_dir():
            shutil.rmtree(path)

    def _handle_worker_result(self, worker_result: WorkerResult) -> None:
        try:
            self._handle_worker_result_inner(worker_result)
        except Exception as exc:
            # Worker callbacks are part of the terminal-state transaction. A
            # malformed result, product identity, or callback exception must
            # become a durable FAILED job rather than escape the monitor.
            self._terminalize_worker_protocol_error(worker_result.request_key, type(exc).__name__)

    def _terminalize_worker_protocol_error(self, request_key: RequestKey, detail: str) -> None:
        row = self.journal.get_job(request_key)
        if row is None:
            return
        current = str(row["state"])
        if current in {"SUCCEEDED", "FAILED", "TIMEOUT", "CANCELED"}:
            return
        if self.journal.transition_job(
            request_key,
            {"QUEUED", "RUNNING", "CANCEL_REQUESTED"},
            "FAILED",
            result={"failure_stage": "WORKER_PROTOCOL", "detail": detail},
            error_code="WORKER_PROTOCOL_ERROR",
        ):
            product_ref = self.journal.fail_product_for_job(request_key, "WORKER_PROTOCOL_ERROR")
            self._discard_product_ref(product_ref)
            cleanup_staging_products(self.deployment.product_directory)
            self.journal.append_event(
                "JOB_FAILED",
                {"error_code": "WORKER_PROTOCOL_ERROR", "detail": detail},
                request_key,
            )

    def _handle_worker_result_inner(self, worker_result: WorkerResult) -> None:
        row = self.journal.get_job(worker_result.request_key)
        if row is None:
            self._discard_worker_product(worker_result.result)
            return
        current = str(row["state"])
        terminal = {"SUCCEEDED", "FAILED", "TIMEOUT", "CANCELED"}
        if current in terminal:
            self._discard_worker_product(worker_result.result)
            return
        if worker_result.state == WorkerResultState.SUCCEEDED:
            result = worker_result.result or {}
            product_summary = result.get("product")
            if not isinstance(product_summary, dict):
                raise RuntimeError("successful worker result is missing product summary")
            product_ref = ProductRef.from_dict(product_summary.get("product_ref"))
            expected_ref = ProductRef.from_dict(json.loads(row["product_ref_json"]))
            if product_ref != expected_ref:
                self._discard_worker_product(result)
                raise RuntimeError("worker product identity mismatch")
            completed = self.journal.complete_job_with_product(
                worker_result.request_key,
                {"RUNNING", "CANCEL_REQUESTED"},
                result,
                product_ref,
                product_summary,
            )
            if not completed:
                self._discard_worker_product(result)
                return
            self.journal.append_event("JOB_COMPLETED", result, worker_result.request_key)
            return
        if worker_result.state == WorkerResultState.REJECTED:
            result = worker_result.result or {"science_decision": "REJECTED"}
            self.journal.transition_job(
                worker_result.request_key,
                {"RUNNING", "CANCEL_REQUESTED"},
                "SUCCEEDED",
                result=result,
                error_code=worker_result.error_code,
            )
            product_ref = self.journal.fail_product_for_job(worker_result.request_key, worker_result.error_code or "INSUFFICIENT_VALID_DATA")
            self._discard_product_ref(product_ref)
            self.journal.append_event("SCIENCE_DECISION_REJECTED", result, worker_result.request_key)
            return
        target_state = {
            WorkerResultState.CANCELED: "CANCELED",
            WorkerResultState.TIMEOUT: "TIMEOUT",
            WorkerResultState.FAILED: "FAILED",
        }[worker_result.state]
        expected = {"CANCEL_REQUESTED"} if target_state == "CANCELED" else {"QUEUED", "RUNNING", "CANCEL_REQUESTED"}
        if self.journal.transition_job(
            worker_result.request_key,
            expected,
            target_state,
            result=worker_result.result,
            error_code=worker_result.error_code,
        ):
            product_ref = self.journal.fail_product_for_job(worker_result.request_key, worker_result.error_code or target_state)
            self._discard_product_ref(product_ref)
            cleanup_staging_products(self.deployment.product_directory)
            self.journal.append_event(
                "JOB_" + target_state,
                {"error_code": worker_result.error_code},
                worker_result.request_key,
            )

    def wait_for_jobs(self, timeout: float | None = None) -> None:
        if self.worker_client is not None:
            self.worker_client.wait(timeout)

    def _encode_ack_tm(self, result: dict[str, Any], sequence_count: int) -> bytes:
        event_time_ns = self._event_clock()
        return encode_tm_application(
            self.profile.tm_event_apid,
            PacketDescriptor.EVENT_ACK,
            {
                "source_spacecraft_instance_id": f"{self.profile.spacecraft_instance_id:016x}",
                "sender_boot_id": self.journal.boot_id,
                "satellite_event_time": event_time_ns,
                "satellite_event_time_ns": event_time_ns,
                "satellite_time_base": self._event_time_base,
                **result,
            },
            sequence_count,
        )

    def encode_ack_tm(self, result: dict[str, Any]) -> bytes:
        """Encode an APID 2 packet for fixture callers without a TM frame."""

        return self._encode_ack_tm(
            result,
            self._next_tm_sequence(self.profile.tm_event_apid),
        )

    def encode_ack_tm_frame_with_order(self, result: dict[str, Any]) -> tuple[bytes, int]:
        """Allocate an APID 2 frame and its durable global TM order key."""

        counters = self.journal.allocate_tm_frame_counters(
            self.profile.tm_event_apid,
            virtual_channel_id=self.profile.tm_virtual_channel,
        )
        frame = encode_tm_frame(
            self._encode_ack_tm(result, counters.packet_sequence),
            spacecraft_id=self.profile.spacecraft_id,
            virtual_channel_id=counters.virtual_channel_id,
            master_channel_count=counters.master_channel_count,
            virtual_channel_count=counters.virtual_channel_count,
        )
        return frame, counters.master_channel_epoch * 256 + counters.master_channel_count

    def encode_ack_tm_frame(self, result: dict[str, Any]) -> bytes:
        """Allocate and persist APID 2/MCFC/VCFC before emitting a TM frame."""

        frame, _ = self.encode_ack_tm_frame_with_order(result)
        return frame

    def close(self) -> None:
        if self.worker_client is not None:
            self.worker_client.close()
        logger.info("payload_stopped")
