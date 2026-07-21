"""Completion-driven single global FilePacket downlink coordinator."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Iterator

from protocol.canonical import canonical_json, checked_u32
from protocol.ccsds import encode_space_packet, encode_tm_frame
from protocol.file_packet import MAX_FILE_DATA_PER_FRAME, FilePacket, FilePacketType, encode_file_packet
from protocol.schemas import ProductRef

from .journal import TmFrameCounters
from .mission_com_scheduler import MissionComScheduler


class TransferState(str, Enum):
    IDLE = "IDLE"
    SENDING = "SENDING"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    CANCEL_DRAINING = "CANCEL_DRAINING"
    ABORTING = "ABORTING"
    COOLDOWN = "COOLDOWN"
    SEND_COMPLETED = "SEND_COMPLETED"
    SEND_FAILED = "SEND_FAILED"
    CANCELED = "CANCELED"


TERMINAL_TRANSFER_STATES = {
    TransferState.SEND_COMPLETED,
    TransferState.SEND_FAILED,
    TransferState.CANCELED,
}

IO_BUFFER_BYTES = 1024 * 1024


@dataclass(frozen=True)
class _BundleStamp:
    device: int
    inode: int
    size: int
    mtime_ns: int


def _bundle_stamp(path: Path) -> _BundleStamp:
    value = path.stat()
    return _BundleStamp(
        int(getattr(value, "st_dev", 0)),
        int(getattr(value, "st_ino", 0)),
        int(value.st_size),
        int(value.st_mtime_ns),
    )


def _stream_bundle_integrity(path: Path) -> tuple[int, str, int]:
    """Return size, SHA-256 and CFDP checksum without loading the bundle."""

    digest = hashlib.sha256()
    checksum = 0
    size = 0
    trailing = b""
    with path.open("rb") as stream:
        while chunk := stream.read(IO_BUFFER_BYTES):
            size += len(chunk)
            digest.update(chunk)
            words = trailing + chunk
            word_bytes = len(words) - (len(words) % 4)
            if word_bytes:
                for (word,) in struct.iter_unpack(">I", words[:word_bytes]):
                    checksum = (checksum + word) & 0xFFFFFFFF
            trailing = words[word_bytes:]
    if trailing:
        checksum = (checksum + int.from_bytes(trailing.ljust(4, b"\0"), "big")) & 0xFFFFFFFF
    return size, digest.hexdigest(), checksum


@dataclass(frozen=True)
class FrameLease:
    attempt_epoch: int
    lease_id: int
    transfer_id: int
    packet: FilePacket
    frame: bytes
    ordering_key: int | None = None


@dataclass
class ActiveTransfer:
    transfer_id: int
    product_ref: ProductRef
    bundle_path: Path
    bundle_size: int
    bundle_sha256: str
    bundle_checksum: int
    bundle_stamp: _BundleStamp
    attempt_epoch: int
    state: TransferState = TransferState.SENDING
    next_offset: int = 0
    next_file_sequence: int = 0
    packet_sequence: int = 0
    current_lease: FrameLease | None = None
    output_gate_open: bool = True
    abort_reason: str | None = None
    epoch_closed: bool = False
    cooldown_ticks: int = 0
    terminal_after_cooldown: TransferState | None = None


class FileDownlinkCoordinator:
    """Owns one wire attempt until terminal completion and cooldown finish."""

    # The local MissionLink and UDP transport use distinct acknowledgement
    # vocabulary, but both mean the completion-gated frame was accepted.
    SUCCESS_STATUSES = frozenset({"OK", "SUCCESS", "FRAME_ACCEPTED", "UDP_SENT", "LINK_CONSUMED"})

    def __init__(
        self,
        *,
        tm_file_apid: int = 3,
        spacecraft_id: int = 68,
        cooldown_ticks: int = 2,
        state_callback: Callable[[int, TransferState, str | None], None] | None = None,
        tm_counter_allocator: Callable[[int], TmFrameCounters] | None = None,
    ):
        if cooldown_ticks <= 0:
            raise ValueError("cooldown_ticks must be positive")
        self.tm_file_apid = tm_file_apid
        self.spacecraft_id = spacecraft_id
        self.required_cooldown_ticks = cooldown_ticks
        self.state_callback = state_callback
        self.tm_counter_allocator = tm_counter_allocator
        self.active: ActiveTransfer | None = None
        self.closed_attempts: set[int] = set()
        self._next_attempt_epoch = 1
        self._next_lease_id = 1
        self.metrics = {
            "frames_completed": 0,
            "terminal_failures": 0,
            "late_callbacks": 0,
            "duplicate_callbacks": 0,
            "abort_fences_closed": 0,
        }

    def _transition(self, active: ActiveTransfer, state: TransferState, reason: str | None = None) -> None:
        active.state = state
        if reason is not None:
            active.abort_reason = reason
        if self.state_callback is not None:
            self.state_callback(active.transfer_id, state, reason)

    def start(self, transfer_id: int, product_ref: ProductRef, bundle_path: str | Path) -> ActiveTransfer:
        checked_u32(transfer_id, "transfer_id")
        if self.active is not None and self.active.state not in TERMINAL_TRANSFER_STATES:
            raise RuntimeError("TRANSFER_BUSY")
        if transfer_id in self.closed_attempts:
            raise RuntimeError("TRANSFER_RETIRED")
        path = Path(bundle_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"bundle does not exist: {path}")
        before = _bundle_stamp(path)
        if before.size > 0xFFFFFFFF:
            raise ValueError("bundle exceeds FilePacket U32 offset range")
        bundle_size, bundle_sha256, bundle_checksum = _stream_bundle_integrity(path)
        if _bundle_stamp(path) != before or bundle_size != before.size:
            raise RuntimeError("BUNDLE_MUTATED")
        active = ActiveTransfer(
            transfer_id,
            product_ref,
            path,
            bundle_size,
            bundle_sha256,
            bundle_checksum,
            before,
            self._next_attempt_epoch,
        )
        self._next_attempt_epoch += 1
        self.active = active
        self._transition(active, TransferState.SENDING)
        return active

    def _active_for(self, transfer_id: int) -> ActiveTransfer:
        active = self.active
        if active is None or active.transfer_id != transfer_id:
            raise RuntimeError("TRANSFER_NOT_ACTIVE")
        return active

    def _start_packet(self, active: ActiveTransfer) -> FilePacket:
        source = f"b/{active.product_ref.origin_boot_id:08x}/{active.product_ref.product_id:08x}.tar"
        destination = (
            f"p/{active.product_ref.origin_boot_id:08x}/{active.product_ref.product_id:08x}/"
            f"{active.transfer_id:08x}/{active.bundle_sha256}.tar"
        )
        metadata = canonical_json(
            {
                "source": source,
                "destination": destination,
                "file_size": active.bundle_size,
                "checksum": active.bundle_checksum,
                "product_ref": active.product_ref.as_dict(),
                "transfer_id": active.transfer_id,
            }
        )
        return FilePacket(FilePacketType.START, 0, 0, metadata)

    @staticmethod
    def _read_data(active: ActiveTransfer) -> bytes:
        if _bundle_stamp(active.bundle_path) != active.bundle_stamp:
            raise RuntimeError("BUNDLE_MUTATED")
        with active.bundle_path.open("rb") as stream:
            stream.seek(active.next_offset)
            payload = stream.read(min(MAX_FILE_DATA_PER_FRAME, active.bundle_size - active.next_offset))
        if _bundle_stamp(active.bundle_path) != active.bundle_stamp:
            raise RuntimeError("BUNDLE_MUTATED")
        if not payload:
            raise RuntimeError("BUNDLE_TRUNCATED")
        return payload

    def _next_packet(self, active: ActiveTransfer) -> FilePacket | None:
        if not active.output_gate_open or active.state in {
            TransferState.ABORTING,
            TransferState.COOLDOWN,
            *TERMINAL_TRANSFER_STATES,
        }:
            return None
        if _bundle_stamp(active.bundle_path) != active.bundle_stamp:
            raise RuntimeError("BUNDLE_MUTATED")
        if active.next_file_sequence == 0:
            return self._start_packet(active)
        if active.state == TransferState.CANCEL_REQUESTED and active.next_offset < active.bundle_size:
            self._transition(active, TransferState.CANCEL_DRAINING)
            return FilePacket(
                FilePacketType.CANCEL,
                active.next_file_sequence,
                active.next_offset,
                b"MISSION_CANCEL",
            )
        if active.next_offset < active.bundle_size:
            return FilePacket(
                FilePacketType.DATA,
                active.next_file_sequence,
                active.next_offset,
                self._read_data(active),
            )
        return FilePacket(
            FilePacketType.END,
            active.next_file_sequence,
            active.bundle_size,
            active.bundle_checksum.to_bytes(4, "big"),
        )

    def next_frame(self, transfer_id: int) -> FrameLease | None:
        active = self._active_for(transfer_id)
        if active.current_lease is not None:
            raise RuntimeError("FRAME_IN_FLIGHT")
        packet = self._next_packet(active)
        if packet is None:
            return None
        application = encode_file_packet(packet)
        counters = (
            None
            if self.tm_counter_allocator is None
            else self.tm_counter_allocator(self.tm_file_apid)
        )
        packet_sequence = active.packet_sequence if counters is None else counters.packet_sequence
        space_packet = encode_space_packet(self.tm_file_apid, application, packet_sequence)
        frame = encode_tm_frame(
            space_packet,
            spacecraft_id=self.spacecraft_id,
            virtual_channel_id=0 if counters is None else counters.virtual_channel_id,
            master_channel_count=0 if counters is None else counters.master_channel_count,
            virtual_channel_count=0 if counters is None else counters.virtual_channel_count,
        )
        ordering_key = (
            None
            if counters is None
            else counters.master_channel_epoch * 256 + counters.master_channel_count
        )
        lease = FrameLease(
            active.attempt_epoch,
            self._next_lease_id,
            active.transfer_id,
            packet,
            frame,
            ordering_key,
        )
        self._next_lease_id += 1
        active.current_lease = lease
        return lease

    def complete_frame(self, lease: FrameLease, status: str) -> bool:
        active = self.active
        if active is None or active.attempt_epoch != lease.attempt_epoch or active.transfer_id != lease.transfer_id:
            self.metrics["late_callbacks"] += 1
            return False
        if active.current_lease is None:
            self.metrics["duplicate_callbacks"] += 1
            return False
        if active.current_lease.lease_id != lease.lease_id:
            self.metrics["late_callbacks"] += 1
            return False
        active.current_lease = None
        if active.state == TransferState.ABORTING:
            return True
        if str(status).upper() not in self.SUCCESS_STATUSES:
            self.metrics["terminal_failures"] += 1
            active.output_gate_open = False
            active.terminal_after_cooldown = TransferState.SEND_FAILED
            self._transition(active, TransferState.ABORTING, f"LINK_{status}")
            return True
        packet = lease.packet
        active.packet_sequence = (active.packet_sequence + 1) % 16384
        active.next_file_sequence = packet.sequence_index + 1
        self.metrics["frames_completed"] += 1
        if packet.packet_type == FilePacketType.DATA:
            active.next_offset = packet.offset + len(packet.payload)
        elif packet.packet_type == FilePacketType.END:
            active.output_gate_open = False
            active.epoch_closed = True
            active.terminal_after_cooldown = TransferState.SEND_COMPLETED
            self._transition(active, TransferState.COOLDOWN)
        elif packet.packet_type == FilePacketType.CANCEL:
            active.output_gate_open = False
            active.epoch_closed = True
            active.terminal_after_cooldown = TransferState.CANCELED
            self._transition(active, TransferState.COOLDOWN)
        return True

    def begin_abort(self, transfer_id: int, reason: str) -> None:
        active = self._active_for(transfer_id)
        if active.state in TERMINAL_TRANSFER_STATES:
            return
        active.output_gate_open = False
        active.terminal_after_cooldown = TransferState.SEND_FAILED
        self._transition(active, TransferState.ABORTING, reason)

    def close_abort_fence(self, transfer_id: int) -> None:
        active = self._active_for(transfer_id)
        if active.state != TransferState.ABORTING:
            raise RuntimeError("TRANSFER_NOT_ABORTING")
        if active.current_lease is not None:
            raise RuntimeError("BUFFER_STILL_IN_FLIGHT")
        active.epoch_closed = True
        self.metrics["abort_fences_closed"] += 1
        self._transition(active, TransferState.COOLDOWN, active.abort_reason)

    def cooldown_tick(self, transfer_id: int) -> TransferState:
        active = self._active_for(transfer_id)
        if active.state != TransferState.COOLDOWN:
            return active.state
        if not active.epoch_closed:
            raise RuntimeError("FILE_EPOCH_NOT_CLOSED")
        active.cooldown_ticks += 1
        if active.cooldown_ticks >= self.required_cooldown_ticks:
            terminal = active.terminal_after_cooldown or TransferState.SEND_FAILED
            self._transition(active, terminal, active.abort_reason)
            self.closed_attempts.add(active.transfer_id)
        return active.state

    def cancel(self, transfer_id: int) -> str:
        active = self._active_for(transfer_id)
        if active.state in TERMINAL_TRANSFER_STATES:
            return "ALREADY_TERMINAL"
        if active.state in {TransferState.ABORTING, TransferState.COOLDOWN, TransferState.CANCEL_DRAINING}:
            return "CANCEL_REQUESTED"
        if active.state == TransferState.SENDING:
            self._transition(active, TransferState.CANCEL_REQUESTED)
        return "CANCEL_REQUESTED"

    def enqueue_next(
        self,
        transfer_id: int,
        scheduler: MissionComScheduler,
        *,
        on_complete: Callable[[FrameLease, str, bool], None] | None = None,
    ) -> int | None:
        """Queue exactly one frame and expose its completion to the owner.

        The caller must schedule a subsequent frame from ``on_complete``. This
        prevents a large bundle from turning into a list of resident TM frames.
        """

        lease = self.next_frame(transfer_id)
        if lease is None:
            return None

        def complete(_item, status: str) -> None:
            completed = self.complete_frame(lease, status)
            active = self.active
            if active is not None and active.state == TransferState.COOLDOWN and active.epoch_closed:
                while active.state == TransferState.COOLDOWN:
                    self.cooldown_tick(transfer_id)
            if on_complete is not None:
                on_complete(lease, str(status), completed)

        try:
            return scheduler.enqueue_file(
                lease.frame,
                complete,
                ordering_key=lease.ordering_key,
            )
        except Exception:
            # ``next_frame`` reserves the lease before scheduler admission.
            # If the bounded queue rejects it, no transport owns that frame.
            active = self.active
            if active is not None and active.current_lease is lease:
                active.current_lease = None
            raise

    def packets(self, transfer_id: int) -> Iterator[FilePacket]:
        """Compatibility iterator that still advances only after completion."""
        while True:
            lease = self.next_frame(transfer_id)
            if lease is None:
                break
            yield lease.packet
            self.complete_frame(lease, "SUCCESS")
        active = self._active_for(transfer_id)
        while active.state == TransferState.COOLDOWN:
            self.cooldown_tick(transfer_id)

    def frames(self, transfer_id: int) -> Iterator[bytes]:
        """Fault-free local helper; each resume simulates one link completion."""
        while True:
            lease = self.next_frame(transfer_id)
            if lease is None:
                break
            yield lease.frame
            self.complete_frame(lease, "SUCCESS")
        active = self._active_for(transfer_id)
        while active.state == TransferState.COOLDOWN:
            self.cooldown_tick(transfer_id)
