"""Durable FilePacket START/DATA/END reassembly with one global wire attempt."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar

from protocol.canonical import canonical_json, checked_u32, checked_u64
from protocol.file_packet import MAX_FILE_DATA_PER_FRAME, FilePacket, FilePacketType
from protocol.schemas import ProductRef, RequestKey

from .product_store import ProductStore, ProductVerificationError, stream_file_integrity, verify_bundle
from .storage import StorageFullError, StorageGuard
from .audit import append_audit_in_transaction
from .u64 import encode_sqlite_u64
from .writer import MutationPriority, SQLiteWriter


class FileReassemblyError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class FileEpochKey:
    spacecraft_instance_id: int
    link_session_id: int
    file_epoch_id: int

    def __post_init__(self) -> None:
        checked_u64(self.spacecraft_instance_id, "spacecraft_instance_id")
        checked_u64(self.link_session_id, "link_session_id")
        checked_u64(self.file_epoch_id, "file_epoch_id")
        if self.file_epoch_id == 0:
            raise ValueError("file_epoch_id must be non-zero for a FilePacket epoch")


@dataclass(frozen=True)
class ReassemblyResult:
    key: FileEpochKey
    state: str
    transfer_id: int | None
    product_ref: ProductRef | None
    bytes_received: int
    expected_size: int | None
    reason: str | None = None
    product: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "spacecraft_instance_id": f"{self.key.spacecraft_instance_id:016x}",
            "link_session_id": f"{self.key.link_session_id:016x}",
            "file_epoch_id": f"{self.key.file_epoch_id:016x}",
            "state": self.state,
            "transfer_id": self.transfer_id,
            "product_ref": None if self.product_ref is None else self.product_ref.as_dict(),
            "bytes_received": self.bytes_received,
            "expected_size": self.expected_size,
            "reason": self.reason,
            "product": self.product,
        }


@dataclass
class _State:
    key: FileEpochKey
    transfer_id: int
    product_ref: ProductRef
    expected_size: int
    expected_file_checksum: int
    expected_bundle_sha256: str
    source_name: str
    destination_name: str
    part_path: Path
    state: str = "RECEIVING"
    start_payload: bytes = b""
    terminal_reason: str | None = None
    origin_request_key: RequestKey | None = None
    created_at_us: int = 0
    updated_at_us: int = 0
    reservation_id: int | None = None
    bytes_received: int = 0
    terminal_packet_type: FilePacketType | None = None
    terminal_sequence_index: int | None = None
    terminal_payload_sha256: str | None = None


_DESTINATION_RE = re.compile(r"^p/(?P<boot>[0-9a-f]{8})/(?P<product>[0-9a-f]{8})/(?P<transfer>[0-9a-f]{8})/(?P<sha>[0-9a-f]{64})\.tar$")


def _read_start(payload: bytes) -> tuple[dict[str, Any], ProductRef]:
    try:
        value = json.loads(bytes(payload).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FileReassemblyError("INVALID_START", "START payload is not UTF-8 JSON") from exc
    if not isinstance(value, dict) or canonical_json(value) != bytes(payload):
        raise FileReassemblyError("INVALID_START", "START payload must be canonical JSON")
    required = {"source", "destination", "file_size", "checksum", "product_ref", "transfer_id"}
    if set(value) != required:
        raise FileReassemblyError("INVALID_START", "START payload has an unexpected field set")
    try:
        product_ref = ProductRef.from_dict(value["product_ref"])
        size = int(value["file_size"])
        checksum = int(value["checksum"])
        transfer_id = checked_u32(value["transfer_id"], "transfer_id")
    except (TypeError, ValueError) as exc:
        raise FileReassemblyError("INVALID_START", "START identity fields are invalid") from exc
    if size < 0 or size > 0xFFFFFFFF:
        raise FileReassemblyError("INVALID_START", "START file_size must fit U32")
    checked_u32(checksum, "checksum")
    source = str(value["source"])
    destination = str(value["destination"])
    expected_source = f"b/{product_ref.origin_boot_id:08x}/{product_ref.product_id:08x}.tar"
    if source != expected_source:
        raise FileReassemblyError("INVALID_START_PATH", "START source path is not canonical")
    match = _DESTINATION_RE.fullmatch(destination)
    if match is None or int(match.group("boot"), 16) != product_ref.origin_boot_id or int(match.group("product"), 16) != product_ref.product_id or int(match.group("transfer"), 16) != transfer_id:
        raise FileReassemblyError("INVALID_START_PATH", "START destination path is not canonical")
    bundle_sha = match.group("sha")
    if not all(char in "0123456789abcdef" for char in bundle_sha):
        raise FileReassemblyError("INVALID_START_PATH", "START destination hash is invalid")
    if size == 0:
        raise FileReassemblyError("INVALID_START", "empty product bundles are not supported")
    return value, product_ref


_T = TypeVar("_T")
_MAX_OVERLAP_ROWS = MAX_FILE_DATA_PER_FRAME + 2


class FilePacketReassembler:
    """Consume validated FilePackets and publish one verified product bundle."""

    def __init__(
        self,
        root: str | Path,
        *,
        writer: SQLiteWriter | None = None,
        product_store: ProductStore | None = None,
        storage_guard: StorageGuard | None = None,
        max_bundle_bytes: int = 1 << 30,
        max_extract_bytes: int = 2 << 30,
        max_artifacts: int = 256,
        reassembly_timeout_us: int = 24 * 3_600_000_000,
        clock_us: Any | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.writer = writer
        self.product_store = product_store
        self.storage_guard = storage_guard
        self.max_bundle_bytes = max_bundle_bytes
        self.max_extract_bytes = max_extract_bytes
        self.max_artifacts = max_artifacts
        self.reassembly_timeout_us = reassembly_timeout_us
        self._clock_injected = clock_us is not None
        self._clock_us = clock_us or (lambda: time.time_ns() // 1_000)
        if isinstance(max_bundle_bytes, bool) or not 0 < max_bundle_bytes <= 0xFFFFFFFF:
            raise ValueError("max_bundle_bytes must fit positive U32")
        if isinstance(max_extract_bytes, bool) or not 0 < max_extract_bytes:
            raise ValueError("max_extract_bytes must be positive")
        if isinstance(max_artifacts, bool) or not 1 <= max_artifacts <= 100_000:
            raise ValueError("max_artifacts must be in [1, 100000]")
        self._lock = threading.RLock()
        self._states: dict[FileEpochKey, _State] = {}
        self._missing_start: dict[FileEpochKey, str] = {}
        self._local_index_path = self.root / ".file-reassembly-index.sqlite3"
        self._index_connection: sqlite3.Connection | None = None
        self._load_durable_states()
        self._release_orphaned_reservations()
        self.reconcile()

    @staticmethod
    def _identity_params(key: FileEpochKey) -> tuple[bytes, bytes, bytes]:
        return (
            encode_sqlite_u64(key.spacecraft_instance_id),
            encode_sqlite_u64(key.link_session_id),
            encode_sqlite_u64(key.file_epoch_id),
        )

    def _local_index(self) -> sqlite3.Connection:
        if self._index_connection is None:
            connection = sqlite3.connect(self._local_index_path, check_same_thread=False)
            connection.execute("PRAGMA foreign_keys=ON")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS file_reassembly_packets (
                    source_spacecraft_instance_id BLOB NOT NULL,
                    link_session_id BLOB NOT NULL,
                    file_epoch_id BLOB NOT NULL,
                    sequence_index INTEGER NOT NULL,
                    offset INTEGER NOT NULL,
                    payload_length INTEGER NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    PRIMARY KEY (source_spacecraft_instance_id, link_session_id, file_epoch_id, sequence_index)
                );
                CREATE TABLE IF NOT EXISTS file_reassembly_ranges (
                    source_spacecraft_instance_id BLOB NOT NULL,
                    link_session_id BLOB NOT NULL,
                    file_epoch_id BLOB NOT NULL,
                    start_offset INTEGER NOT NULL,
                    end_offset INTEGER NOT NULL,
                    PRIMARY KEY (source_spacecraft_instance_id, link_session_id, file_epoch_id, start_offset)
                );
                CREATE INDEX IF NOT EXISTS ix_file_reassembly_ranges_overlap
                    ON file_reassembly_ranges(
                        source_spacecraft_instance_id, link_session_id, file_epoch_id,
                        start_offset, end_offset
                    );
                """
            )
            self._index_connection = connection
        return self._index_connection

    def _index_read(self, callback: Callable[[sqlite3.Connection], _T]) -> _T:
        if self.writer is not None:
            with self.writer.reader() as connection:
                return callback(connection)
        return callback(self._local_index())

    def _index_mutate(self, name: str, callback: Callable[[sqlite3.Connection], _T]) -> _T:
        if self.writer is not None:
            return self.writer.mutate(name, callback, priority=MutationPriority.HIGH)
        connection = self._local_index()
        with connection:
            return callback(connection)

    def _load_durable_states(self) -> None:
        if self.writer is None:
            return
        with self.writer.reader() as connection:
            rows = connection.execute("SELECT * FROM file_reassemblies WHERE state='RECEIVING'").fetchall()
        for row in rows:
            key = FileEpochKey(
                int.from_bytes(bytes(row["source_spacecraft_instance_id"]), "big"),
                int.from_bytes(bytes(row["link_session_id"]), "big"),
                int.from_bytes(bytes(row["file_epoch_id"]), "big"),
            )
            terminal_type = row["terminal_packet_type"]
            state = _State(
                key=key,
                transfer_id=int(row["transfer_id"]),
                product_ref=ProductRef(
                    int.from_bytes(bytes(row["product_spacecraft_instance_id"]), "big"),
                    int(row["origin_boot_id"]),
                    int(row["product_id"]),
                ),
                expected_size=int(row["expected_size"]),
                expected_file_checksum=int(row["expected_file_checksum"]),
                expected_bundle_sha256=bytes(row["expected_bundle_sha256"]).hex(),
                source_name=str(row["source_name"]),
                destination_name=str(row["destination_name"]),
                part_path=Path(str(row["part_path"])),
                state=str(row["state"]),
                start_payload=bytes(row["start_payload"]),
                terminal_reason=row["terminal_reason"],
                created_at_us=int(row["created_at_us"]),
                updated_at_us=int(row["updated_at_us"]),
                reservation_id=None if row["reservation_id"] is None else int(row["reservation_id"]),
                bytes_received=int(row["received_bytes"]),
                terminal_packet_type=None if terminal_type is None else FilePacketType(int(terminal_type)),
                terminal_sequence_index=None if row["terminal_sequence_index"] is None else int(row["terminal_sequence_index"]),
                terminal_payload_sha256=None if row["terminal_payload_sha256"] is None else str(row["terminal_payload_sha256"]),
            )
            self._states[key] = state
            if not state.part_path.is_file():
                state.state = "CANCELED"
                state.terminal_reason = "STAGING_MISSING"
                self._persist_state(state, clear_metadata=True)
                self._release_state(state)

    def _release_orphaned_reservations(self) -> None:
        if self.writer is None or self.storage_guard is None:
            return
        active_owners = {
            f"file-reassembly:{state.key.spacecraft_instance_id:016x}:{state.key.link_session_id:016x}:{state.key.file_epoch_id:016x}"
            for state in self._states.values()
            if state.state == "RECEIVING" and state.reservation_id is not None
        }
        with self.writer.reader() as connection:
            rows = connection.execute(
                "SELECT reservation_id,owner FROM storage_reservations WHERE volume=? AND state='ACTIVE' AND owner LIKE 'file-reassembly:%'",
                (self.storage_guard.volume,),
            ).fetchall()
        for row in rows:
            if str(row[1]) not in active_owners:
                self.storage_guard.release(int(row[0]))

    def _packet_metadata(self, state: _State, sequence_index: int) -> tuple[int, int, str] | None:
        row = self._index_read(
            lambda connection: connection.execute(
                "SELECT offset,payload_length,payload_sha256 FROM file_reassembly_packets "
                "WHERE source_spacecraft_instance_id=? AND link_session_id=? AND file_epoch_id=? AND sequence_index=?",
                (*self._identity_params(state.key), sequence_index),
            ).fetchone()
        )
        if row is None:
            return None
        return int(row[0]), int(row[1]), str(row[2])

    def _overlap_ranges(self, state: _State, start: int, end: int) -> list[tuple[int, int]]:
        rows = self._index_read(
            lambda connection: connection.execute(
                "SELECT start_offset,end_offset FROM file_reassembly_ranges "
                "WHERE source_spacecraft_instance_id=? AND link_session_id=? AND file_epoch_id=? "
                "AND start_offset<=? AND end_offset>=? ORDER BY start_offset LIMIT ?",
                (*self._identity_params(state.key), end, start, _MAX_OVERLAP_ROWS + 1),
            ).fetchall()
        )
        if len(rows) > _MAX_OVERLAP_ROWS:
            raise FileReassemblyError("REASSEMBLY_METADATA_LIMIT", "DATA packet touches too many persisted ranges")
        return [(int(row[0]), int(row[1])) for row in rows]

    def _next_sequence(self, state: _State) -> int:
        row = self._index_read(
            lambda connection: connection.execute(
                "SELECT COALESCE(MAX(sequence_index),0) FROM file_reassembly_packets "
                "WHERE source_spacecraft_instance_id=? AND link_session_id=? AND file_epoch_id=?",
                self._identity_params(state.key),
            ).fetchone()
        )
        return int(row[0]) + 1

    def _coverage_complete(self, state: _State) -> bool:
        if state.bytes_received != state.expected_size:
            return False
        rows = self._index_read(
            lambda connection: connection.execute(
                "SELECT start_offset,end_offset FROM file_reassembly_ranges "
                "WHERE source_spacecraft_instance_id=? AND link_session_id=? AND file_epoch_id=? "
                "ORDER BY start_offset LIMIT 2",
                self._identity_params(state.key),
            ).fetchall()
        )
        return len(rows) == 1 and int(rows[0][0]) == 0 and int(rows[0][1]) == state.expected_size

    def _persist_data_packet(
        self,
        state: _State,
        *,
        sequence_index: int,
        offset: int,
        payload_length: int,
        payload_sha256: str,
        old_ranges: list[tuple[int, int]],
        merged_start: int,
        merged_end: int,
        received_bytes: int,
        updated_at_us: int,
    ) -> None:
        identity = self._identity_params(state.key)

        def persist(connection: sqlite3.Connection) -> None:
            for old_start, _old_end in old_ranges:
                connection.execute(
                    "DELETE FROM file_reassembly_ranges WHERE source_spacecraft_instance_id=? AND link_session_id=? "
                    "AND file_epoch_id=? AND start_offset=?",
                    (*identity, old_start),
                )
            connection.execute(
                "INSERT INTO file_reassembly_ranges(source_spacecraft_instance_id,link_session_id,file_epoch_id,start_offset,end_offset) "
                "VALUES(?,?,?,?,?)",
                (*identity, merged_start, merged_end),
            )
            connection.execute(
                "INSERT INTO file_reassembly_packets(source_spacecraft_instance_id,link_session_id,file_epoch_id,sequence_index,offset,payload_length,payload_sha256) "
                "VALUES(?,?,?,?,?,?,?)",
                (*identity, sequence_index, offset, payload_length, payload_sha256),
            )
            if self.writer is not None:
                connection.execute(
                    "UPDATE file_reassemblies SET received_bytes=?,updated_at_us=?,ranges_json='[]',sequence_map_json='{}' "
                    "WHERE source_spacecraft_instance_id=? AND link_session_id=? AND file_epoch_id=?",
                    (received_bytes, updated_at_us, *identity),
                )

        self._index_mutate("persist_file_reassembly_data_packet", persist)

    def _clear_index_metadata(self, state: _State, connection: sqlite3.Connection | None = None) -> None:
        identity = self._identity_params(state.key)

        def clear(current: sqlite3.Connection) -> None:
            current.execute(
                "DELETE FROM file_reassembly_packets WHERE source_spacecraft_instance_id=? AND link_session_id=? AND file_epoch_id=?",
                identity,
            )
            current.execute(
                "DELETE FROM file_reassembly_ranges WHERE source_spacecraft_instance_id=? AND link_session_id=? AND file_epoch_id=?",
                identity,
            )

        if connection is None:
            self._index_mutate("clear_file_reassembly_metadata", clear)
        else:
            clear(connection)

    def receive(
        self,
        packet: FilePacket,
        *,
        spacecraft_instance_id: int,
        link_session_id: int,
        file_epoch_id: int,
        received_at_us: int = 0,
        origin_request_key: RequestKey | None = None,
    ) -> ReassemblyResult:
        if not isinstance(packet, FilePacket):
            raise TypeError("packet must be a FilePacket")
        key = FileEpochKey(spacecraft_instance_id, link_session_id, file_epoch_id)
        checked_u64(spacecraft_instance_id, "spacecraft_instance_id")
        if received_at_us < 0:
            raise ValueError("received_at_us must be non-negative")
        with self._lock:
            if packet.packet_type is FilePacketType.START:
                return self._start(packet, key, received_at_us, origin_request_key)
            state = self._states.get(key)
            if state is None:
                self._missing_start[key] = "MISSING_START"
                return ReassemblyResult(key, "INCOMPLETE", None, None, 0, None, "MISSING_START")
            if state.state != "RECEIVING":
                if self._is_duplicate_terminal(state, packet):
                    return self._result(state)
                raise FileReassemblyError("FILE_PACKET_CONFLICT", "packet arrived after terminal epoch")
            if packet.packet_type is FilePacketType.DATA:
                return self._data(state, packet, received_at_us)
            if packet.packet_type is FilePacketType.END:
                return self._end(state, packet, received_at_us)
            if packet.packet_type is FilePacketType.CANCEL:
                return self._cancel(state, packet, received_at_us)
            raise FileReassemblyError("UNKNOWN_FILE_PACKET", "unsupported FilePacket type")

    def reconcile(self, *, now_us: int | None = None) -> tuple[FileEpochKey, ...]:
        """Expire stale receiving epochs and release their durable reservations."""
        now = self._clock_us() if now_us is None else now_us
        if isinstance(now, bool) or not isinstance(now, int) or now < 0:
            raise ValueError("now_us must be a non-negative integer")
        if self.storage_guard is not None:
            self.storage_guard.expire()
        expired: list[FileEpochKey] = []
        with self._lock:
            for state in tuple(self._states.values()):
                if (
                    state.state != "RECEIVING"
                    or (state.updated_at_us == 0 and not self._clock_injected)
                    or state.updated_at_us + self.reassembly_timeout_us > now
                ):
                    continue
                state.state = "CANCELED"
                state.terminal_reason = "REASSEMBLY_TIMEOUT"
                state.updated_at_us = now
                self._persist_state(state, clear_metadata=True)
                self._audit_failure(state, "REASSEMBLY_TIMEOUT", "reassembly inactivity timeout")
                self._release_state(state)
                self._remove_part(state)
                expired.append(state.key)
        return tuple(expired)

    def _start(self, packet: FilePacket, key: FileEpochKey, received_at_us: int, origin_request_key: RequestKey | None) -> ReassemblyResult:
        if packet.sequence_index != 0 or packet.offset != 0:
            raise FileReassemblyError("INVALID_START", "START must use sequenceIndex=0 and offset=0")
        value, product_ref = _read_start(packet.payload)
        if product_ref.spacecraft_instance_id != key.spacecraft_instance_id:
            raise FileReassemblyError("PRODUCT_TARGET_INSTANCE_MISMATCH", "START ProductRef instance differs from source instance")
        existing = self._states.get(key)
        if existing is not None:
            if existing.start_payload == packet.payload:
                return self._result(existing)
            raise FileReassemblyError("START_CONFLICT", "duplicate START differs from the active epoch")
        for other in self._states.values():
            if other.key.spacecraft_instance_id == key.spacecraft_instance_id and other.state == "RECEIVING":
                raise FileReassemblyError("TRANSFER_BUSY", "another global FilePacket attempt is still receiving")
        destination = str(value["destination"])
        bundle_sha = destination.rsplit("/", 1)[-1][:-4]
        expected_size = int(value["file_size"])
        if expected_size > self.max_bundle_bytes:
            raise FileReassemblyError("BUNDLE_TOO_LARGE", "START file_size exceeds the configured bundle limit")
        if expected_size > 0xFFFFFFFF:
            raise FileReassemblyError("BUNDLE_TOO_LARGE", "START file_size exceeds FilePacket U32 range")
        part_path = self.root / f"{key.spacecraft_instance_id:016x}" / f"{key.link_session_id:016x}" / f"{key.file_epoch_id:016x}.part"
        reservation_id: int | None = None
        if self.storage_guard is not None:
            try:
                reservation = self.storage_guard.reserve(
                    f"file-reassembly:{key.spacecraft_instance_id:016x}:{key.link_session_id:016x}:{key.file_epoch_id:016x}",
                    expected_size,
                )
            except StorageFullError as exc:
                raise FileReassemblyError("STORAGE_FULL", str(exc)) from exc
            reservation_id = reservation.reservation_id
        try:
            part_path.parent.mkdir(parents=True, exist_ok=True)
            with part_path.open("xb") as stream:
                stream.flush()
                os.fsync(stream.fileno())
        except Exception:
            if reservation_id is not None and self.storage_guard is not None:
                self.storage_guard.release(reservation_id)
            raise FileReassemblyError("STAGING_OPEN_FAILED", "unable to create the bounded reassembly file")
        state = _State(
            key,
            int(value["transfer_id"]),
            product_ref,
            expected_size,
            int(value["checksum"]),
            bundle_sha,
            str(value["source"]),
            destination,
            part_path,
            start_payload=packet.payload,
            origin_request_key=origin_request_key,
            created_at_us=received_at_us,
            updated_at_us=received_at_us,
            reservation_id=reservation_id,
        )
        if self.writer is None:
            # A completed in-process epoch may have left a local sidecar entry
            # behind if the process exited between terminal persistence and cleanup.
            self._clear_index_metadata(state)
        self._states[key] = state
        if self.writer is not None:
            try:
                self._persist_start(state)
            except Exception as exc:
                self._states.pop(key, None)
                self._release_state(state)
                self._remove_part(state)
                raise FileReassemblyError("DURABLE_STATE_FAILED", "unable to persist reassembly state") from exc
        return self._result(state)

    def _data(self, state: _State, packet: FilePacket, received_at_us: int) -> ReassemblyResult:
        if packet.sequence_index == 0:
            raise FileReassemblyError("INVALID_DATA", "DATA sequenceIndex must be greater than zero")
        if packet.offset > state.expected_size or not packet.payload or len(packet.payload) > state.expected_size - packet.offset:
            raise FileReassemblyError("DATA_RANGE_INVALID", "DATA range is outside START file size")
        end = packet.offset + len(packet.payload)
        digest = hashlib.sha256(packet.payload).hexdigest()
        old = self._packet_metadata(state, packet.sequence_index)
        if old is not None:
            if old != (packet.offset, len(packet.payload), digest):
                raise FileReassemblyError("FILE_PACKET_CONFLICT", "duplicate DATA sequence differs")
            with state.part_path.open("rb") as stream:
                stream.seek(packet.offset)
                if stream.read(len(packet.payload)) != packet.payload:
                    raise FileReassemblyError("FILE_PACKET_CONFLICT", "duplicate DATA bytes differ")
            return self._result(state)
        ranges = self._overlap_ranges(state, packet.offset, end)
        with state.part_path.open("r+b") as stream:
            for start, finish in ranges:
                overlap_start = max(start, packet.offset)
                overlap_end = min(finish, end)
                if overlap_start < overlap_end:
                    stream.seek(overlap_start)
                    existing = stream.read(overlap_end - overlap_start)
                    incoming = packet.payload[overlap_start - packet.offset : overlap_end - packet.offset]
                    if existing != incoming:
                        raise FileReassemblyError("FILE_PACKET_CONFLICT", "overlapping DATA bytes differ")
            stream.seek(packet.offset)
            stream.write(packet.payload)
            stream.flush()
            os.fsync(stream.fileno())
        merged_start = min([packet.offset, *(start for start, _ in ranges)])
        merged_end = max([end, *(finish for _, finish in ranges)])
        previous_coverage = sum(finish - start for start, finish in ranges)
        bytes_received = state.bytes_received + (merged_end - merged_start - previous_coverage)
        self._persist_data_packet(
            state,
            sequence_index=packet.sequence_index,
            offset=packet.offset,
            payload_length=len(packet.payload),
            payload_sha256=digest,
            old_ranges=ranges,
            merged_start=merged_start,
            merged_end=merged_end,
            received_bytes=bytes_received,
            updated_at_us=received_at_us,
        )
        state.bytes_received = bytes_received
        state.updated_at_us = received_at_us
        return self._result(state)

    def _end(self, state: _State, packet: FilePacket, received_at_us: int) -> ReassemblyResult:
        if packet.sequence_index != self._next_sequence(state) or packet.offset != state.expected_size or len(packet.payload) != 4:
            raise FileReassemblyError("INVALID_END", "END must use the next sequence and exact file size")
        state.terminal_packet_type = FilePacketType.END
        state.terminal_sequence_index = packet.sequence_index
        state.terminal_payload_sha256 = hashlib.sha256(packet.payload).hexdigest()
        state.updated_at_us = received_at_us
        if not self._coverage_complete(state):
            state.state = "INCOMPLETE"
            state.terminal_reason = "GAP"
            self._persist_state(state, clear_metadata=True)
            self._release_state(state)
            self._remove_part(state)
            return self._result(state)
        try:
            integrity = stream_file_integrity(state.part_path, max_bytes=state.expected_size)
            if integrity.size != state.expected_size:
                raise ProductVerificationError("FILE_SIZE_MISMATCH", "reassembled file size differs from START metadata")
            if int.from_bytes(packet.payload, "big") != integrity.cfdp_checksum or integrity.cfdp_checksum != state.expected_file_checksum:
                raise ProductVerificationError("FILE_CHECKSUM_MISMATCH", "END checksum does not match file")
            if integrity.sha256 != state.expected_bundle_sha256:
                raise ProductVerificationError("BUNDLE_SHA_MISMATCH", "START bundle SHA does not match file")
            verification_root: Path | None = None
            if self.product_store is not None:
                verification_root = self.root / f".verified-{state.transfer_id:08x}"
                verified = verify_bundle(
                    state.part_path,
                    expected_bundle_sha256=state.expected_bundle_sha256,
                    expected_file_checksum=state.expected_file_checksum,
                    expected_product_ref=state.product_ref,
                    temporary_root=verification_root,
                    max_bundle_bytes=self.max_bundle_bytes,
                    max_extract_bytes=self.max_extract_bytes,
                    max_files=self.max_artifacts,
                )
                summary = self.product_store.publish(verified)
            else:
                summary = None
        except (ProductVerificationError, OSError, ValueError) as exc:
            state.state = "CHECKSUM_FAILED"
            state.terminal_reason = getattr(exc, "code", "PRODUCT_VERIFY_FAILED")
            self._persist_state(state, clear_metadata=True)
            self._audit_failure(state, state.terminal_reason, str(exc))
            self._release_state(state)
            self._remove_part(state)
            return self._result(state)
        finally:
            if self.product_store is not None:
                shutil.rmtree(self.root / f".verified-{state.transfer_id:08x}", ignore_errors=True)
        state.state = "VERIFIED"
        state.terminal_reason = None
        self._persist_state(state, clear_metadata=True)
        self._release_state(state)
        self._remove_part(state)
        return self._result(state, product=summary)

    def _cancel(self, state: _State, packet: FilePacket, received_at_us: int) -> ReassemblyResult:
        if packet.sequence_index != self._next_sequence(state):
            raise FileReassemblyError("INVALID_CANCEL", "CANCEL must use the next sequence")
        state.terminal_packet_type = FilePacketType.CANCEL
        state.terminal_sequence_index = packet.sequence_index
        state.terminal_payload_sha256 = hashlib.sha256(packet.payload).hexdigest()
        state.state = "CANCELED"
        state.terminal_reason = "REMOTE_CANCEL"
        state.updated_at_us = received_at_us
        self._persist_state(state, clear_metadata=True)
        self._release_state(state)
        self._remove_part(state)
        return self._result(state)

    @staticmethod
    def _is_duplicate_terminal(state: _State, packet: FilePacket) -> bool:
        return (
            state.terminal_packet_type == packet.packet_type
            and state.terminal_sequence_index == packet.sequence_index
            and state.terminal_payload_sha256 == hashlib.sha256(packet.payload).hexdigest()
        )

    @staticmethod
    def _result(state: _State, *, product: dict[str, Any] | None = None) -> ReassemblyResult:
        return ReassemblyResult(state.key, state.state, state.transfer_id, state.product_ref, state.bytes_received, state.expected_size, state.terminal_reason, product)

    @staticmethod
    def _remove_part(state: _State) -> None:
        try:
            state.part_path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass

    def _release_state(self, state: _State) -> None:
        reservation_id = state.reservation_id
        if reservation_id is None or self.storage_guard is None:
            return
        state.reservation_id = None
        self.storage_guard.release(reservation_id)
        if self.writer is not None:
            self.writer.mutate(
                "clear_file_reassembly_reservation",
                lambda connection: connection.execute(
                    "UPDATE file_reassemblies SET reservation_id=NULL WHERE source_spacecraft_instance_id=? AND link_session_id=? AND file_epoch_id=?",
                    (
                        encode_sqlite_u64(state.key.spacecraft_instance_id),
                        encode_sqlite_u64(state.key.link_session_id),
                        encode_sqlite_u64(state.key.file_epoch_id),
                    ),
                ),
                priority=MutationPriority.HIGH,
            )

    def _audit_failure(self, state: _State, error_code: str, message: str) -> None:
        if self.writer is None:
            return
        self.writer.mutate(
            "audit_file_reassembly_failure",
            lambda connection: append_audit_in_transaction(
                connection,
                principal="gds-file-reassembler",
                action="FILE_REASSEMBLY_FAILED",
                target_type="file_epoch",
                target_identity={
                    "spacecraft_instance_id": f"{state.key.spacecraft_instance_id:016x}",
                    "link_session_id": f"{state.key.link_session_id:016x}",
                    "file_epoch_id": f"{state.key.file_epoch_id:016x}",
                },
                old_value=None,
                new_value={"error_code": error_code, "message": message},
                created_at_us=max(0, state.updated_at_us),
            ),
            priority=MutationPriority.HIGH,
        )

    def _ensure_product_in_transaction(self, connection, state: _State) -> None:
        origin = state.origin_request_key or RequestKey(0, 0)
        connection.execute(
            "INSERT INTO products(spacecraft_instance_id,origin_boot_id,product_id,origin_ground_instance_id,origin_request_id,product_type,state,bundle_size,bundle_sha256,origin_request_key_json,created_at_us,retention_until_us) VALUES(?,?,?,?,?,'UNKNOWN','RECEIVING',?,?,?, ?,?) ON CONFLICT(spacecraft_instance_id,origin_boot_id,product_id) DO UPDATE SET bundle_size=COALESCE(products.bundle_size,excluded.bundle_size),bundle_sha256=COALESCE(products.bundle_sha256,excluded.bundle_sha256),origin_request_key_json=COALESCE(products.origin_request_key_json,excluded.origin_request_key_json)",
            (encode_sqlite_u64(state.product_ref.spacecraft_instance_id), state.product_ref.origin_boot_id, state.product_ref.product_id, encode_sqlite_u64(origin.ground_instance_id), origin.request_id, state.expected_size, bytes.fromhex(state.expected_bundle_sha256), json.dumps(origin.as_dict(), sort_keys=True, separators=(",", ":")), state.created_at_us, state.created_at_us),
        )

    def _persist_start(self, state: _State) -> None:
        assert self.writer is not None

        def persist(connection: sqlite3.Connection) -> None:
            self._ensure_product_in_transaction(connection, state)
            connection.execute(
                "INSERT INTO file_reassemblies("
                "source_spacecraft_instance_id,link_session_id,file_epoch_id,transfer_id,"
                "product_spacecraft_instance_id,origin_boot_id,product_id,expected_size,"
                "expected_file_checksum,expected_bundle_sha256,source_name,destination_name,"
                "part_path,state,ranges_json,sequence_map_json,received_bytes,start_payload,"
                "terminal_packet_type,terminal_sequence_index,terminal_payload_sha256,"
                "terminal_reason,created_at_us,updated_at_us,verified_at_us,reservation_id"
                ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    encode_sqlite_u64(state.key.spacecraft_instance_id),
                    encode_sqlite_u64(state.key.link_session_id),
                    encode_sqlite_u64(state.key.file_epoch_id),
                    state.transfer_id,
                    encode_sqlite_u64(state.product_ref.spacecraft_instance_id),
                    state.product_ref.origin_boot_id,
                    state.product_ref.product_id,
                    state.expected_size,
                    state.expected_file_checksum,
                    bytes.fromhex(state.expected_bundle_sha256),
                    state.source_name,
                    state.destination_name,
                    str(state.part_path),
                    state.state,
                    "[]",
                    "{}",
                    state.bytes_received,
                    state.start_payload,
                    None if state.terminal_packet_type is None else int(state.terminal_packet_type),
                    state.terminal_sequence_index,
                    state.terminal_payload_sha256,
                    state.terminal_reason,
                    state.created_at_us,
                    state.updated_at_us,
                    None,
                    state.reservation_id,
                ),
            )

        self.writer.mutate(
            "persist_file_reassembly_start",
            persist,
            priority=MutationPriority.HIGH,
        )

    def _persist_state(self, state: _State, *, clear_metadata: bool = False) -> None:
        if self.writer is None:
            if clear_metadata:
                self._clear_index_metadata(state)
            return

        identity = self._identity_params(state.key)

        def persist(connection: sqlite3.Connection) -> None:
            if clear_metadata:
                self._clear_index_metadata(state, connection)
            connection.execute(
                "UPDATE file_reassemblies SET "
                "state=?,ranges_json='[]',sequence_map_json='{}',received_bytes=?,"
                "terminal_packet_type=?,terminal_sequence_index=?,terminal_payload_sha256=?,"
                "terminal_reason=?,updated_at_us=?,verified_at_us=?,reservation_id=? "
                "WHERE source_spacecraft_instance_id=? AND link_session_id=? AND file_epoch_id=?",
                (
                    state.state,
                    state.bytes_received,
                    None if state.terminal_packet_type is None else int(state.terminal_packet_type),
                    state.terminal_sequence_index,
                    state.terminal_payload_sha256,
                    state.terminal_reason,
                    state.updated_at_us,
                    state.updated_at_us if state.state == "VERIFIED" else None,
                    state.reservation_id,
                    *identity,
                ),
            )

        self.writer.mutate(
            "persist_file_reassembly_state",
            persist,
            priority=MutationPriority.HIGH,
        )
