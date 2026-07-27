"""Deterministic CCSDS link boundary with explicit session and file epochs."""

from __future__ import annotations

import json
import logging
import threading
from typing import Callable, Dict, Optional

from protocol.ccsds import decode_tm_frame
from protocol.file_packet import FilePacket, FilePacketType, decode_file_packet

from .contact_schedule import ContactSchedule
from .control import LinkControlMessage, LinkControlType
from .fault_model import FaultModel, FaultProfile
from .file_epoch import FileEpochManager
from .session_manager import Session, SessionManager, SessionState
from .transport import Direction, SidebandEnvelope, Transport, TransportFrame
from .virtual_clock import SimulationTime, VirtualClock

logger = logging.getLogger(__name__)


DeliveryCallback = Callable[[TransportFrame], str | None]
AcceptedCallback = Callable[[LinkControlMessage], None]
ConsumedCallback = Callable[[LinkControlMessage], None]


class LinkSimulator:
    """Serialize ingress, inject independent copy faults, and deliver egress.

    ``direction`` describes the simulated path: ``INGRESS`` is uplink and
    ``EGRESS`` is downlink. The sideband on the caller-facing frame remains an
    ingress envelope; the simulator creates the validated egress envelope.
    """

    def __init__(
        self,
        simulation_run_id: int,
        seed: int,
        uplink_profile: FaultProfile,
        downlink_profile: FaultProfile,
        clock: Optional[VirtualClock] = None,
        transport: Optional[Transport] = None,
        contact_schedule: Optional[ContactSchedule] = None,
        *,
        session_manager: SessionManager | None = None,
        file_epoch_manager: FileEpochManager | None = None,
        next_link_frame_id: int = 1,
        state_changed: Callable[[], None] | None = None,
    ):
        if isinstance(next_link_frame_id, bool) or not 1 <= next_link_frame_id <= 0xFFFFFFFFFFFFFFFF:
            raise ValueError("next_link_frame_id must fit U64 and be positive")
        self.simulation_run_id = simulation_run_id
        self.seed = seed
        self.uplink_profile = uplink_profile
        self.downlink_profile = downlink_profile
        self.clock = clock or VirtualClock()
        self.transport = transport
        self.contact_schedule = contact_schedule or ContactSchedule()
        self.fault_model = FaultModel(seed=seed, simulation_run_id=simulation_run_id)
        self.session_manager = session_manager or SessionManager()
        self.file_epoch_manager = file_epoch_manager or FileEpochManager()

        self._lock = threading.Lock()
        self._next_link_frame_id = int(next_link_frame_id)
        self._state_changed = state_changed
        # Compatibility mirrors used by the phase-3 replay tests.
        self._sessions: Dict[int, Session] = self.session_manager._session_history
        self._file_epochs: dict[int, object] = {}
        self._uplink_available_ns = 0
        self._downlink_available_ns = 0
        self._admission_log: list[dict] = []
        self.control_events: list[LinkControlMessage] = []

        logger.info(
            "LinkSimulator initialized: run_id=%#018x, seed=%#018x",
            simulation_run_id,
            seed,
        )

    def checkpoint(self) -> dict[str, object]:
        """Expose restart-critical allocators for the UDP service checkpoint."""

        with self._lock:
            return {
                "next_link_frame_id": self._next_link_frame_id,
                "session_manager": self.session_manager.checkpoint(),
            }

    def _checkpoint_changed(self) -> None:
        if self._state_changed is not None:
            self._state_changed()

    def _control(
        self,
        message_type: LinkControlType,
        *,
        session: Session | None,
        spacecraft_instance_id: int,
        link_frame_id: int = 0,
        file_epoch_id: int = 0,
        copy_index: int = 0,
        status: str | None = None,
        reason: str | None = None,
    ) -> LinkControlMessage:
        message = LinkControlMessage(
            message_type,
            spacecraft_instance_id,
            0 if session is None else session.sender_boot_id,
            0 if session is None else session.session_id,
            0 if session is None else session.generation,
            link_frame_id,
            file_epoch_id,
            copy_index,
            status,
            reason,
        )
        self.control_events.append(message)
        return message

    def create_session(self, spacecraft_instance_id: int, sender_boot_id: int) -> int:
        """Open a session and its file-epoch fence."""
        session_id = self.session_manager.create_session(
            spacecraft_instance_id,
            sender_boot_id,
            self.clock.now.ns,
        )
        self._checkpoint_changed()
        self.file_epoch_manager.open_session(sender_boot_id)
        session = self.session_manager.get_session(session_id)
        self._control(
            LinkControlType.OPEN_SESSION,
            session=session,
            spacecraft_instance_id=spacecraft_instance_id,
        )
        self._control(
            LinkControlType.SESSION_READY,
            session=session,
            spacecraft_instance_id=spacecraft_instance_id,
            status="READY",
        )
        return session_id

    def close_session(self, session_id: int) -> None:
        session = self.session_manager.close_session(session_id, self.clock.now.ns)
        if session is None:
            raise ValueError(f"Unknown session: {session_id:#018x}")
        self.file_epoch_manager.close_session(session.sender_boot_id)
        self._control(
            LinkControlType.SESSION_RESET,
            session=session,
            spacecraft_instance_id=session.spacecraft_instance_id,
            reason="SESSION_CLOSED",
        )

    def reset_session(self, spacecraft_instance_id: int, sender_boot_id: int) -> int:
        """Close the active session and establish a new boot boundary."""
        active = self.session_manager.get_active_session(spacecraft_instance_id)
        if active is not None:
            self.session_manager.close_session(active.session_id, self.clock.now.ns)
        self.file_epoch_manager.open_session(sender_boot_id)
        session_id = self.session_manager.create_session(
            spacecraft_instance_id,
            sender_boot_id,
            self.clock.now.ns,
        )
        self._checkpoint_changed()
        session = self.session_manager.get_session(session_id)
        self._control(
            LinkControlType.SESSION_RESET,
            session=session,
            spacecraft_instance_id=spacecraft_instance_id,
        )
        self._control(
            LinkControlType.SESSION_READY,
            session=session,
            spacecraft_instance_id=spacecraft_instance_id,
            status="READY",
        )
        return session_id

    def abort_file_epoch(self, file_epoch_id: int, reason: str = "ABORT_REQUESTED") -> bool:
        current = self.file_epoch_manager.get_current_attempt()
        if current is None or current.epoch_id != file_epoch_id:
            return False
        result = self.file_epoch_manager.abort_attempt(current.attempt_id, 0)
        session = self.session_manager.get_session(current.session_id)
        self._control(
            LinkControlType.ABORT_FILE_EPOCH,
            session=session,
            spacecraft_instance_id=current.spacecraft_instance_id,
            file_epoch_id=file_epoch_id,
            reason=reason,
        )
        return result

    def _session_for_ingress(self, envelope: SidebandEnvelope) -> Session | None:
        # A few phase-3 unit fixtures intentionally exercise the fault engine
        # without a handshake. Production callers always create a session and
        # therefore take the strict path below.
        if not self._sessions:
            return None
        session = self.session_manager.get_session(envelope.link_session_id)
        if session is None:
            return None
        if session.state is not SessionState.ACTIVE:
            return None
        if session.spacecraft_instance_id != envelope.spacecraft_instance_id:
            return None
        active = self.session_manager.get_active_session(envelope.spacecraft_instance_id)
        if active is None or active.session_id != session.session_id:
            return None
        return session

    @staticmethod
    def _file_packet_info(frame_bytes: bytes) -> FilePacket | None:
        try:
            tm = decode_tm_frame(frame_bytes)
            if tm.packet.apid != 3:
                return None
            packet = decode_file_packet(tm.packet.payload)
            return packet
        except (TypeError, ValueError):
            return None

    def _admit_file_epoch(
        self,
        *,
        direction: Direction,
        session: Session | None,
        link_frame_id: int,
        frame_bytes: bytes,
    ) -> tuple[int, str | None, FilePacketType | None]:
        if direction is not Direction.EGRESS:
            return 0, None, None
        info = self._file_packet_info(frame_bytes)
        if info is None:
            return 0, None, None
        packet_type = info.packet_type
        payload = info.payload
        if session is None:
            return 0, "NO_ACTIVE_SESSION", packet_type
        boot = session.sender_boot_id
        if packet_type is FilePacketType.START:
            try:
                metadata = json.loads(payload.decode("utf-8"))
                file_path = str(metadata.get("destination", "")) if isinstance(metadata, dict) else ""
            except (UnicodeDecodeError, json.JSONDecodeError):
                file_path = ""
            attempt = self.file_epoch_manager.admit_start(
                session.session_id,
                session.spacecraft_instance_id,
                boot,
                link_frame_id,
                file_path,
            )
            return (0, "TRANSFER_BUSY", packet_type) if attempt is None else (attempt, None, packet_type)
        current = self.file_epoch_manager.get_current_attempt()
        if current is None or current.session_id != session.session_id:
            return 0, "NO_ACTIVE_TRANSFER", packet_type
        if packet_type is FilePacketType.DATA:
            _, reason = self.file_epoch_manager.admit_data(boot, link_frame_id, info.offset, len(payload))
            return current.epoch_id, reason, packet_type
        if packet_type is FilePacketType.END:
            _, reason = self.file_epoch_manager.admit_end(boot, link_frame_id)
            return current.epoch_id, reason, packet_type
        self.file_epoch_manager.abort_attempt(current.attempt_id, link_frame_id)
        return current.epoch_id, None, packet_type

    def admit_frame(
        self,
        transport_frame: TransportFrame,
        *,
        direction: Direction = Direction.INGRESS,
        on_accepted: AcceptedCallback | None = None,
        on_consumed: ConsumedCallback | None = None,
        on_delivered: DeliveryCallback | None = None,
    ) -> Optional[int]:
        """Admit a frame, schedule each copy independently, and expose gates."""
        envelope = transport_frame.envelope
        frame_bytes = transport_frame.frame_bytes
        try:
            envelope.validate_ingress()
            direction = Direction(direction)
        except (TypeError, ValueError) as exc:
            logger.error("Ingress validation failed: %s", exc)
            return None

        current_time = self.clock.now
        if self.contact_schedule.should_drop_frame(current_time):
            logger.info("Frame dropped: BLACKOUT at %s", current_time.ns)
            return None

        session = self._session_for_ingress(envelope)
        if self._sessions and session is None:
            logger.warning("Frame rejected at session boundary: session=%s", envelope.link_session_id)
            return None

        with self._lock:
            admission_order = self.clock.get_admission_order()
            link_frame_id = self._next_link_frame_id
            if link_frame_id >= 0xFFFFFFFFFFFFFFFF:
                raise RuntimeError("link frame ID allocator exhausted")
            self._next_link_frame_id += 1
            ingress_time = self.clock.now
            self._admission_log.append(
                {
                    "admission_order": admission_order,
                    "link_frame_id": link_frame_id,
                    "ingress_time_ns": ingress_time.ns,
                    "spacecraft_instance_id": envelope.spacecraft_instance_id,
                    "sender_frame_id": envelope.sender_frame_id,
                    "session_id": envelope.link_session_id,
                    "direction": direction.name,
                    "frame_length": envelope.frame_length,
                }
            )
        # Persist before fault scheduling/delivery so a crash cannot replay a
        # delivered link_frame_id after the process comes back.
        self._checkpoint_changed()

        epoch_id, epoch_error, packet_type = self._admit_file_epoch(
            direction=direction,
            session=session,
            link_frame_id=link_frame_id,
            frame_bytes=frame_bytes,
        )
        if epoch_error is not None:
            logger.warning("File epoch rejected: frame=%s reason=%s", link_frame_id, epoch_error)
            if on_consumed is not None:
                on_consumed(
                    self._control(
                        LinkControlType.FRAME_CONSUMED,
                        session=session,
                        spacecraft_instance_id=envelope.spacecraft_instance_id,
                        link_frame_id=link_frame_id,
                        file_epoch_id=epoch_id,
                        status="FILE_EPOCH_REJECTED",
                        reason=epoch_error,
                    )
                )
            return None

        accepted = self._control(
            LinkControlType.FRAME_ACCEPTED,
            session=session,
            spacecraft_instance_id=envelope.spacecraft_instance_id,
            link_frame_id=link_frame_id,
            file_epoch_id=epoch_id,
            status="ACCEPTED",
        )
        if on_accepted is not None:
            on_accepted(accepted)

        profile = self.uplink_profile if direction is Direction.INGRESS else self.downlink_profile
        available = self._uplink_available_ns if direction is Direction.INGRESS else self._downlink_available_ns
        decision = self.fault_model.apply_faults(
            profile,
            direction.value,
            link_frame_id,
            len(frame_bytes) * 8,
            ingress_time.ns,
            available,
            copy_index=0,
        )
        if direction is Direction.INGRESS:
            self._uplink_available_ns = decision.release_ns
        else:
            self._downlink_available_ns = decision.release_ns

        if decision.is_lost:
            self._emit_consumed(
                on_consumed,
                self._control(
                    LinkControlType.FRAME_CONSUMED,
                    session=session,
                    spacecraft_instance_id=envelope.spacecraft_instance_id,
                    link_frame_id=link_frame_id,
                    file_epoch_id=epoch_id,
                    status="FRAME_LOST",
                    reason="FAULT_LOSS",
                ),
            )
            return link_frame_id

        self._schedule_delivery(
            link_frame_id,
            0,
            frame_bytes,
            decision,
            envelope,
            direction,
            session,
            epoch_id,
            packet_type,
            on_consumed,
            on_delivered,
        )

        if decision.has_duplicate:
            duplicate_available = self._uplink_available_ns if direction is Direction.INGRESS else self._downlink_available_ns
            duplicate = self.fault_model.apply_faults(
                profile,
                direction.value,
                link_frame_id,
                len(frame_bytes) * 8,
                ingress_time.ns,
                duplicate_available,
                copy_index=1,
                include_duplicate=False,
            )
            if direction is Direction.INGRESS:
                self._uplink_available_ns = duplicate.release_ns
            else:
                self._downlink_available_ns = duplicate.release_ns
            if not duplicate.is_lost:
                self._schedule_delivery(
                    link_frame_id,
                    1,
                    frame_bytes,
                    duplicate,
                    envelope,
                    direction,
                    session,
                    epoch_id,
                    packet_type,
                    on_consumed,
                    on_delivered,
                )
            else:
                self._emit_consumed(
                    on_consumed,
                    self._control(
                        LinkControlType.FRAME_CONSUMED,
                        session=session,
                        spacecraft_instance_id=envelope.spacecraft_instance_id,
                        link_frame_id=link_frame_id,
                        file_epoch_id=epoch_id,
                        copy_index=1,
                        status="FRAME_LOST",
                        reason="FAULT_DUPLICATE_COPY_LOSS",
                    ),
                )
        return link_frame_id

    @staticmethod
    def _emit_consumed(callback: ConsumedCallback | None, message: LinkControlMessage) -> None:
        if callback is not None:
            callback(message)

    def _schedule_delivery(
        self,
        link_frame_id: int,
        copy_index: int,
        frame_bytes: bytes,
        decision,
        envelope: SidebandEnvelope,
        direction: Direction,
        session: Session | None,
        epoch_id: int,
        packet_type: FilePacketType | None,
        on_consumed: ConsumedCallback | None,
        on_delivered: DeliveryCallback | None,
    ) -> None:
        if decision.is_corrupted and decision.corrupted_bits:
            frame_bytes = self.fault_model.corrupt_frame(frame_bytes, decision.corrupted_bits)

        egress_envelope = SidebandEnvelope(
            magic=SidebandEnvelope.MAGIC,
            version=(
                SidebandEnvelope.VERSION
                if copy_index == 0
                else SidebandEnvelope.VERSION_WITH_COPY_INDEX
            ),
            direction=Direction.EGRESS,
            reserved=0,
            spacecraft_instance_id=envelope.spacecraft_instance_id,
            sender_boot_id=(session.sender_boot_id if session is not None else envelope.sender_boot_id),
            link_session_id=envelope.link_session_id,
            sender_frame_id=envelope.sender_frame_id,
            link_frame_id=link_frame_id,
            file_epoch_id=epoch_id,
            frame_length=len(frame_bytes),
            copy_index=copy_index,
        )
        egress_envelope.validate_egress()
        delivery_time = SimulationTime(decision.release_ns)

        def deliver() -> None:
            transport_frame = TransportFrame(egress_envelope, frame_bytes, copy_index)
            try:
                if self.transport is not None:
                    sender = getattr(self.transport, "send_transport_frame", None)
                    if sender is not None:
                        sender(transport_frame)
                    else:
                        self.transport.send(egress_envelope, frame_bytes)
                status = "FRAME_CONSUMED"
                if on_delivered is not None:
                    callback_status = on_delivered(transport_frame)
                    if callback_status:
                        status = str(callback_status)
                if packet_type in {FilePacketType.END, FilePacketType.CANCEL} and epoch_id:
                    current = self.file_epoch_manager.get_current_attempt()
                    if current is not None and current.epoch_id == epoch_id:
                        self.file_epoch_manager.complete_attempt(current.attempt_id)
                self._emit_consumed(
                    on_consumed,
                    self._control(
                        LinkControlType.FRAME_CONSUMED,
                        session=session,
                        spacecraft_instance_id=envelope.spacecraft_instance_id,
                        link_frame_id=link_frame_id,
                        file_epoch_id=epoch_id,
                        copy_index=copy_index,
                        status=status,
                    ),
                )
            except Exception as exc:
                logger.exception("Egress delivery failed: frame=%s copy=%s", link_frame_id, copy_index)
                self._emit_consumed(
                    on_consumed,
                    self._control(
                        LinkControlType.FRAME_CONSUMED,
                        session=session,
                        spacecraft_instance_id=envelope.spacecraft_instance_id,
                        link_frame_id=link_frame_id,
                        file_epoch_id=epoch_id,
                        copy_index=copy_index,
                        status="FRAME_FAILED",
                        reason=type(exc).__name__,
                    ),
                )

        self.clock.schedule_at(
            due_time=delivery_time,
            callback=deliver,
            direction=direction.value,
            link_frame_id=link_frame_id,
            copy_index=copy_index,
        )

    def run_until_idle(self) -> int:
        return self.clock.run_until_idle()

    def get_stats(self) -> Dict:
        with self._lock:
            session_stats = self.session_manager.get_stats()
            epoch_stats = self.file_epoch_manager.get_stats()
            return {
                "simulation_run_id": f"{self.simulation_run_id:#018x}",
                "seed": f"{self.seed:#018x}",
                "next_link_frame_id": self._next_link_frame_id,
                "frames_admitted": len(self._admission_log),
                "active_sessions": session_stats["active_sessions"],
                "session": session_stats,
                "file_epoch": epoch_stats,
                "current_time_ns": self.clock.now.ns,
            }
