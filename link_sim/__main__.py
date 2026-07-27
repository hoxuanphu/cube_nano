"""Runnable UDP Link Simulator service for Compose deployments."""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import socket
import tempfile
import time
from dataclasses import fields
from pathlib import Path
from typing import Any, Callable

import yaml

from .contact_schedule import ContactSchedule, ContactState
from .fault_model import FaultProfile
from .link_simulator import LinkSimulator
from .session_manager import SessionManager
from .transport import (
    Direction,
    SidebandEnvelope,
    Transport,
    TransportFrame,
    UdpTransport,
    decode_session_control,
    encode_session_binding,
)
from .virtual_clock import SimulationTime


logger = logging.getLogger("link_sim.service")
_HEALTH_PROBE = b"CSH1?"
_HEALTH_REPLY = b"CSH1!"
_LINK_CHECKPOINT_SCHEMA_VERSION = 1


def _parse_int(value: str | int, label: str) -> int:
    if isinstance(value, int):
        return value
    text = str(value).strip()
    try:
        if text.lower().startswith("0x"):
            return int(text, 16)
        if any(character in "abcdefABCDEF" for character in text):
            return int(text, 16)
        return int(text, 10)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{label} must be an integer") from exc


def _address(host: str, port: int) -> tuple[str, int]:
    if not 1 <= port <= 65535:
        raise ValueError("port must be in [1, 65535]")
    return host, port


class _LinkCheckpointStore:
    """Atomically retain monotonic IDs across a LinkSimulator process restart."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory).resolve()
        self.path = self.directory / "link_checkpoint.json"
        self.loaded_checkpoint = self.path.is_file()

    @staticmethod
    def _positive_u64(value: Any, label: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 0xFFFFFFFFFFFFFFFF:
            raise ValueError(f"{label} must be a positive U64")
        return int(value)

    @staticmethod
    def _u64(value: Any, label: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 0xFFFFFFFFFFFFFFFF:
            raise ValueError(f"{label} must be a U64")
        return int(value)

    def load(self) -> tuple[int, int, dict[int, int]]:
        if not self.loaded_checkpoint:
            return 1, 1, {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"cannot read link checkpoint {self.path}") from exc
        if not isinstance(raw, dict) or raw.get("schema_version") != _LINK_CHECKPOINT_SCHEMA_VERSION:
            raise RuntimeError("link checkpoint schema is invalid")
        next_session_id = self._positive_u64(raw.get("next_session_id"), "next_session_id")
        next_link_frame_id = self._positive_u64(raw.get("next_link_frame_id"), "next_link_frame_id")
        raw_floors = raw.get("generation_floors")
        if not isinstance(raw_floors, dict):
            raise RuntimeError("link checkpoint generation_floors is invalid")
        floors: dict[int, int] = {}
        for raw_instance, raw_generation in raw_floors.items():
            try:
                instance = self._positive_u64(int(raw_instance), "generation_floors instance")
            except (TypeError, ValueError) as exc:
                raise RuntimeError("link checkpoint generation_floors key is invalid") from exc
            floors[instance] = self._u64(raw_generation, "generation_floors generation")
        return next_session_id, next_link_frame_id, floors

    def save(self, simulator: LinkSimulator) -> None:
        checkpoint = simulator.checkpoint()
        session_state = checkpoint.get("session_manager")
        if not isinstance(session_state, dict):
            raise RuntimeError("LinkSimulator checkpoint is invalid")
        next_session_id = self._positive_u64(session_state.get("next_session_id"), "next_session_id")
        next_link_frame_id = self._positive_u64(checkpoint.get("next_link_frame_id"), "next_link_frame_id")
        floors = session_state.get("generation_floors")
        if not isinstance(floors, dict):
            raise RuntimeError("LinkSimulator generation checkpoint is invalid")
        payload = {
            "schema_version": _LINK_CHECKPOINT_SCHEMA_VERSION,
            "next_session_id": next_session_id,
            "next_link_frame_id": next_link_frame_id,
            "generation_floors": {
                str(self._positive_u64(instance, "generation_floors instance")): self._u64(
                    generation,
                    "generation_floors generation",
                )
                for instance, generation in floors.items()
            },
        }
        self.directory.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".link_checkpoint-",
            suffix=".tmp",
            dir=self.directory,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, self.path)
        finally:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def _resolved_addresses(address: tuple[str, int]) -> set[tuple[str, int]]:
    host, port = address
    try:
        results = socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_DGRAM)
    except socket.gaierror as exc:
        raise ValueError(f"unable to resolve configured peer {host}:{port}") from exc
    return {(str(value[4][0]), int(value[4][1])) for value in results}


class _PeerUdpTransport(Transport):
    """Central UDP bridge that accepts only the two configured mission peers."""

    def __init__(
        self,
        bind: tuple[str, int],
        *,
        gds_peer: tuple[str, int],
        satellite_peer: tuple[str, int],
        health_payload: Callable[[], dict[str, Any]],
    ) -> None:
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(bind)
        self.gds_peer = gds_peer
        self.satellite_peer = satellite_peer
        self._gds_addresses = _resolved_addresses(gds_peer)
        self._satellite_addresses = _resolved_addresses(satellite_peer)
        self.peer: tuple[str, int] | None = None
        self._health_payload = health_payload

    @property
    def bound_address(self) -> tuple[str, int]:
        host, port = self.socket.getsockname()
        return str(host), int(port)

    def _role(self, peer: tuple[str, int]) -> str:
        normalized = str(peer[0]), int(peer[1])
        if normalized in self._gds_addresses:
            return "gds"
        if normalized in self._satellite_addresses:
            return "satellite"
        raise ValueError(f"unrecognized link peer: {peer}")

    def select_peer(self, peer: tuple[str, int]) -> None:
        self._role(peer)
        self.peer = peer

    def send(self, envelope: SidebandEnvelope, frame_bytes: bytes) -> None:
        if self.peer is None:
            raise RuntimeError("link egress peer is not selected")
        envelope.validate_egress()
        if len(frame_bytes) != envelope.frame_length:
            raise ValueError("link egress frame length mismatch")
        datagram = envelope.to_bytes() + bytes(frame_bytes)
        if len(datagram) > UdpTransport.MAX_DATAGRAM_BYTES:
            raise ValueError("link datagram exceeds the UDP payload limit")
        self.socket.sendto(datagram, self.peer)

    def send_transport_frame(self, frame: TransportFrame) -> None:
        """Send LinkSimulator output without dropping duplicate identity."""

        envelope = frame.envelope
        if frame.copy_index != envelope.copy_index:
            from dataclasses import replace

            envelope = replace(
                envelope,
                version=(
                    SidebandEnvelope.VERSION
                    if frame.copy_index == 0 and envelope.version == SidebandEnvelope.VERSION
                    else SidebandEnvelope.VERSION_WITH_COPY_INDEX
                ),
                copy_index=frame.copy_index,
            )
        self.send(envelope, frame.frame_bytes)

    def send_control(self, peer: tuple[str, int], datagram: bytes) -> None:
        """Reply only to one configured control-plane peer."""

        self._role(peer)
        if decode_session_control(datagram) is None:
            raise ValueError("link control datagram has an unknown prefix")
        self.socket.sendto(bytes(datagram), peer)

    def receive(self, timeout_ms: int | None = None) -> tuple[TransportFrame | dict[str, Any], str] | None:
        self.socket.settimeout(None if timeout_ms is None else timeout_ms / 1000)
        try:
            datagram, peer = self.socket.recvfrom(UdpTransport.MAX_DATAGRAM_BYTES)
        except socket.timeout:
            return None
        if datagram == _HEALTH_PROBE:
            payload = json.dumps(self._health_payload(), sort_keys=True, separators=(",", ":")).encode("utf-8")
            self.socket.sendto(_HEALTH_REPLY + payload, peer)
            return None
        role = self._role(peer)
        control = decode_session_control(datagram)
        if control is not None:
            return control, role
        if len(datagram) < SidebandEnvelope.HEADER_SIZE:
            raise ValueError("link datagram is shorter than its envelope")
        envelope = SidebandEnvelope.from_bytes(datagram)
        frame = datagram[SidebandEnvelope.header_size_for_version(envelope.version):]
        if len(frame) != envelope.frame_length:
            raise ValueError("link datagram frame length mismatch")
        envelope.validate_ingress()
        return TransportFrame(envelope, frame, envelope.copy_index), role

    def close(self) -> None:
        self.socket.close()


def _profile_from_mapping(value: Any, label: str) -> FaultProfile:
    if value is None:
        return FaultProfile()
    if not isinstance(value, dict):
        raise ValueError(f"{label} fault profile must be an object")
    allowed = {field.name for field in fields(FaultProfile)}
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{label} fault profile has unsupported fields: {unknown}")
    profile = FaultProfile(**{str(key): int(item) for key, item in value.items()})
    profile.validate()
    return profile


def _load_fault_configuration(path: str | Path | None) -> tuple[FaultProfile, FaultProfile, int | None, ContactSchedule, dict[str, Any]]:
    if path is None:
        return FaultProfile(), FaultProfile(), None, ContactSchedule(), {"profile_id": "default", "revision": 0}
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or int(raw.get("schema_version", 0)) != 1:
        raise ValueError("fault profile must be a schema_version 1 object")
    uplink = _profile_from_mapping(raw.get("uplink"), "uplink")
    downlink = _profile_from_mapping(raw.get("downlink"), "downlink")
    revision = int(raw.get("revision", 0))
    uplink.profile_revision = revision
    downlink.profile_revision = revision
    configured_seed = raw.get("seed")
    seed = None if configured_seed is None else _parse_int(configured_seed, "fault profile seed")
    schedule = ContactSchedule()
    contact = str(raw.get("contact", "OPEN")).upper()
    if contact == "BLACKOUT":
        schedule.add_window(SimulationTime(0), SimulationTime((1 << 63) - 1), ContactState.BLACKOUT)
    elif contact not in {"OPEN", "CONTACT_OPEN", "NO_CONTACT"}:
        raise ValueError("fault profile contact must be OPEN, NO_CONTACT, or BLACKOUT")
    return uplink, downlink, seed, schedule, {
        "profile_id": str(raw.get("profile_id", Path(path).stem)),
        "revision": revision,
        "contact": contact,
    }


def _probe_health(host: str, port: int, timeout_ms: int) -> dict[str, Any]:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.settimeout(timeout_ms / 1000)
        probe.sendto(_HEALTH_PROBE, _address(host, port))
        try:
            data, _ = probe.recvfrom(UdpTransport.MAX_DATAGRAM_BYTES)
        except socket.timeout as exc:
            raise RuntimeError("link health probe timed out") from exc
    if not data.startswith(_HEALTH_REPLY):
        raise RuntimeError("link health probe returned an invalid response")
    try:
        payload = json.loads(data[len(_HEALTH_REPLY) :].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("link health probe returned invalid JSON") from exc
    if not isinstance(payload, dict) or payload.get("state") != "READY":
        raise RuntimeError("link health probe reported not ready")
    return payload


def serve(args: argparse.Namespace) -> None:
    gds_peer = _address(args.gds_host, args.gds_port)
    satellite_peer = _address(args.satellite_host, args.satellite_port)
    uplink, downlink, profile_seed, schedule, profile_metadata = _load_fault_configuration(args.fault_profile)
    seed = args.seed if args.seed is not None else (profile_seed if profile_seed is not None else 1)
    if not 0 <= seed <= 0xFFFFFFFFFFFFFFFF or not 0 <= args.run_id <= 0xFFFFFFFFFFFFFFFF:
        raise ValueError("seed and run-id must fit U64")

    checkpoint_store = _LinkCheckpointStore(args.state_directory)
    next_session_id, next_link_frame_id, generation_floors = checkpoint_store.load()
    simulator: LinkSimulator | None = None
    exchange = {
        "gds_ingress_frames": 0,
        "satellite_ingress_frames": 0,
        "uplink_admitted": 0,
        "downlink_admitted": 0,
        "rejected_frames": 0,
        "last_exchange_at_ms": None,
    }

    def health_payload() -> dict[str, Any]:
        active = (
            None
            if simulator is None
            else simulator.session_manager.get_active_session(args.spacecraft_instance_id)
        )
        exchange_ready = (
            active is not None
            and exchange["gds_ingress_frames"] > 0
            and exchange["satellite_ingress_frames"] > 0
            and exchange["uplink_admitted"] > 0
            and exchange["downlink_admitted"] > 0
        )
        return {
            "state": "READY" if exchange_ready else "NOT_READY",
            "service": "link-simulator",
            "bound_address": transport.bound_address,
            "session_id": None if active is None else active.session_id,
            "link_generation": None if active is None else active.generation,
            "sender_boot_id": None if active is None else active.sender_boot_id,
            "fault_profile": profile_metadata,
            "peer_exchange": {**exchange, "ready": exchange_ready},
        }

    transport = _PeerUdpTransport(
        _address(args.bind_host, args.bind_port),
        gds_peer=gds_peer,
        satellite_peer=satellite_peer,
        health_payload=health_payload,
    )
    session_manager = SessionManager(
        next_session_id=next_session_id,
        generation_floors=generation_floors,
    )
    simulator = LinkSimulator(
        simulation_run_id=args.run_id,
        seed=seed,
        uplink_profile=uplink,
        downlink_profile=downlink,
        contact_schedule=schedule,
        transport=transport,
        session_manager=session_manager,
        next_link_frame_id=next_link_frame_id,
        state_changed=lambda: checkpoint_store.save(simulator),
    )
    hellos: dict[str, dict[str, Any]] = {}

    def publish_session_binding() -> None:
        """Create/reset only after both peer-authenticated HELLOs agree."""

        gds_hello = hellos.get("gds")
        satellite_hello = hellos.get("satellite")
        if gds_hello is None or satellite_hello is None:
            return
        if (
            gds_hello["spacecraft_instance_id"] != args.spacecraft_instance_id
            or satellite_hello["spacecraft_instance_id"] != args.spacecraft_instance_id
        ):
            raise ValueError("session HELLO spacecraft instance does not match LinkSimulator configuration")
        boot_id = int(satellite_hello["sender_boot_id"])
        if args.sender_boot_id is not None and boot_id != args.sender_boot_id:
            raise ValueError(
                f"configured sender_boot_id {args.sender_boot_id} does not match satellite HELLO {boot_id}"
            )
        active = simulator.session_manager.get_active_session(args.spacecraft_instance_id)
        if active is None:
            session_id = simulator.create_session(args.spacecraft_instance_id, boot_id)
            active = simulator.session_manager.get_session(session_id)
            if active is None:
                raise RuntimeError("LinkSimulator did not retain the opened session")
            if (
                not checkpoint_store.loaded_checkpoint
                and args.link_session_id is not None
                and session_id != args.link_session_id
            ):
                raise ValueError(
                    f"configured initial link_session_id {args.link_session_id} does not match HELLO session {session_id}"
                )
            if (
                not checkpoint_store.loaded_checkpoint
                and args.link_generation is not None
                and active.generation != args.link_generation
            ):
                raise ValueError(
                    f"configured initial link_generation {args.link_generation} does not match HELLO generation "
                    f"{active.generation}"
                )
        elif active.sender_boot_id != boot_id:
            session_id = simulator.reset_session(args.spacecraft_instance_id, boot_id)
            active = simulator.session_manager.get_session(session_id)
            if active is None:
                raise RuntimeError("LinkSimulator did not retain the reset session")
            logger.info(
                "session reset from satellite HELLO boot=%s session=%s generation=%s",
                boot_id,
                active.session_id,
                active.generation,
            )
        binding = encode_session_binding(
            spacecraft_instance_id=args.spacecraft_instance_id,
            sender_boot_id=active.sender_boot_id,
            link_session_id=active.session_id,
            link_generation=active.generation,
        )
        transport.send_control(gds_peer, binding)
        transport.send_control(satellite_peer, binding)

    stopped = False

    def stop(_signum, _frame) -> None:
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, stop)
    logger.info(
        "listening bind=%s awaiting_peer_hello fault_profile=%s seed=%s",
        transport.bound_address,
        profile_metadata["profile_id"],
        seed,
    )
    try:
        while not stopped:
            try:
                received = transport.receive(timeout_ms=500)
            except ValueError as exc:
                logger.warning("rejected UDP datagram: %s", exc)
                continue
            if received is None:
                continue
            frame, role = received
            if isinstance(frame, dict):
                if frame.get("kind") != "HELLO" or frame.get("role") != role:
                    logger.warning("rejected session control role=%s payload=%s", role, frame.get("kind"))
                    continue
                hellos[role] = frame
                try:
                    publish_session_binding()
                except ValueError as exc:
                    logger.warning("rejected session HELLO role=%s: %s", role, exc)
                continue
            active = simulator.session_manager.get_active_session(args.spacecraft_instance_id)
            if active is None:
                exchange["rejected_frames"] += 1
                logger.warning("rejected mission frame before peer HELLO role=%s", role)
                continue
            if role == "gds":
                exchange["gds_ingress_frames"] += 1
                direction = Direction.INGRESS
                transport.select_peer(satellite_peer)
            else:
                exchange["satellite_ingress_frames"] += 1
                direction = Direction.EGRESS
                transport.select_peer(gds_peer)
            admitted = simulator.admit_frame(frame, direction=direction)
            if admitted is None:
                exchange["rejected_frames"] += 1
                logger.warning("rejected frame peer=%s session=%s", role, frame.envelope.link_session_id)
                continue
            if role == "gds":
                exchange["uplink_admitted"] += 1
            else:
                exchange["downlink_admitted"] += 1
            exchange["last_exchange_at_ms"] = int(time.time() * 1000)
            simulator.run_until_idle()
    finally:
        transport.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Cube Nano deterministic UDP Link Simulator")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--health-once", action="store_true")
    parser.add_argument("--health-host", default="127.0.0.1")
    parser.add_argument("--health-port", type=int, default=9000)
    parser.add_argument("--health-timeout-ms", type=int, default=1000)
    parser.add_argument("--bind-host", default="0.0.0.0")
    parser.add_argument("--bind-port", type=int, default=9000)
    parser.add_argument("--gds-host", default="gds")
    parser.add_argument("--gds-port", type=int, default=9001)
    parser.add_argument("--satellite-host", default="satellite")
    parser.add_argument("--satellite-port", type=int, default=9002)
    parser.add_argument("--spacecraft-instance-id", type=lambda value: _parse_int(value, "spacecraft-instance-id"), default=1)
    parser.add_argument(
        "--sender-boot-id",
        type=lambda value: _parse_int(value, "sender-boot-id"),
        help="optional expected satellite durable boot ID; HELLO is authoritative",
    )
    parser.add_argument("--link-session-id", type=lambda value: _parse_int(value, "link-session-id"))
    parser.add_argument("--link-generation", type=lambda value: _parse_int(value, "link-generation"))
    parser.add_argument("--fault-profile", type=Path)
    parser.add_argument(
        "--state-directory",
        type=Path,
        default=Path(os.environ.get("CUBE_NANO_LINK_STATE_DIRECTORY", "data/link")),
        help="durable LinkSimulator checkpoint directory",
    )
    parser.add_argument("--seed", type=lambda value: _parse_int(value, "seed"))
    parser.add_argument("--run-id", type=lambda value: _parse_int(value, "run-id"), default=1)
    args = parser.parse_args()
    if args.health_once:
        print(json.dumps(_probe_health(args.health_host, args.health_port, args.health_timeout_ms), sort_keys=True))
        return
    if not args.serve:
        parser.error("--serve or --health-once is required")
    serve(args)


if __name__ == "__main__":
    main()
