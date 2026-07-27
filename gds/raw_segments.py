"""Append-only raw frame segments with crash-torn record recovery."""

from __future__ import annotations

import hashlib
import os
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


MAGIC = b"GDSR"
VERSION = 1
HEADER = struct.Struct(">4sB3xII")
DEFAULT_MAX_RECORD_BYTES = 256 * 1024 * 1024


class RawSegmentError(RuntimeError):
    pass


@dataclass(frozen=True)
class RawRecord:
    path: Path
    offset: int
    length: int
    payload_length: int
    payload_sha256: str


class RawSegmentStore:
    """Keep frame bytes outside SQLite while references remain durable."""

    def __init__(
        self,
        path: str | Path,
        *,
        max_record_bytes: int = DEFAULT_MAX_RECORD_BYTES,
    ) -> None:
        self.path = Path(path)
        if max_record_bytes <= 0:
            raise ValueError("max_record_bytes must be positive")
        self.max_record_bytes = max_record_bytes

    def append(self, payload: bytes) -> RawRecord:
        payload = bytes(payload)
        if not payload:
            raise ValueError("raw payload must not be empty")
        if len(payload) > self.max_record_bytes:
            raise ValueError("raw payload is larger than the segment record limit")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.scan(repair=True)
        mode = "r+b" if self.path.exists() else "w+b"
        with self.path.open(mode) as stream:
            stream.seek(0, os.SEEK_END)
            offset = stream.tell()
            stream.write(
                HEADER.pack(MAGIC, VERSION, len(payload), zlib.crc32(payload) & 0xFFFFFFFF)
            )
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        return RawRecord(
            self.path,
            offset,
            HEADER.size + len(payload),
            len(payload),
            hashlib.sha256(payload).hexdigest(),
        )

    def append_before_db(
        self,
        payload: bytes,
        persist_reference: Callable[[RawRecord], object],
    ) -> object:
        """Fsync bytes before invoking the DB reference transaction callback."""

        record = self.append(payload)
        return persist_reference(record)

    def scan(
        self,
        *,
        repair: bool = False,
        committed_offset: int | None = None,
    ) -> tuple[RawRecord, ...]:
        if committed_offset is not None and (
            isinstance(committed_offset, bool)
            or not isinstance(committed_offset, int)
            or committed_offset < 0
        ):
            raise ValueError("committed_offset must be a non-negative integer")
        if not self.path.exists():
            return ()
        records: list[RawRecord] = []
        good_offset = 0
        with self.path.open("rb") as stream:
            while True:
                offset = stream.tell()
                header = stream.read(HEADER.size)
                if not header:
                    good_offset = offset
                    break
                if len(header) != HEADER.size:
                    break
                magic, version, payload_length, expected_crc = HEADER.unpack(header)
                if (
                    magic != MAGIC
                    or version != VERSION
                    or payload_length <= 0
                    or payload_length > self.max_record_bytes
                ):
                    break
                payload = stream.read(payload_length)
                if len(payload) != payload_length:
                    break
                if (zlib.crc32(payload) & 0xFFFFFFFF) != expected_crc:
                    break
                records.append(
                    RawRecord(
                        self.path,
                        offset,
                        HEADER.size + payload_length,
                        payload_length,
                        hashlib.sha256(payload).hexdigest(),
                    )
                )
                good_offset = stream.tell()
        actual_size = self.path.stat().st_size
        if committed_offset is not None and committed_offset > good_offset:
            raise RawSegmentError(
                "committed_offset is beyond the last valid raw record boundary"
            )
        repair_offset = good_offset
        if committed_offset is not None and committed_offset < repair_offset:
            if committed_offset not in {
                record.offset + record.length for record in records
            } and committed_offset != 0:
                raise RawSegmentError("committed_offset is not a raw record boundary")
            repair_offset = committed_offset
            records = [
                record
                for record in records
                if record.offset + record.length <= committed_offset
            ]
        if repair and repair_offset != actual_size:
            with self.path.open("r+b") as stream:
                stream.truncate(repair_offset)
                stream.flush()
                os.fsync(stream.fileno())
        return tuple(records)

    recover = scan

    def read(self, record: RawRecord) -> bytes:
        if record.path.resolve() != self.path.resolve():
            raise RawSegmentError("record belongs to another segment")
        with self.path.open("rb") as stream:
            stream.seek(record.offset)
            header = stream.read(HEADER.size)
            if len(header) != HEADER.size:
                raise RawSegmentError("raw record header is truncated")
            magic, version, payload_length, expected_crc = HEADER.unpack(header)
            if magic != MAGIC or version != VERSION or payload_length != record.payload_length:
                raise RawSegmentError("raw record header does not match reference")
            payload = stream.read(payload_length)
            if len(payload) != payload_length or (zlib.crc32(payload) & 0xFFFFFFFF) != expected_crc:
                raise RawSegmentError("raw record CRC check failed")
            if hashlib.sha256(payload).hexdigest() != record.payload_sha256:
                raise RawSegmentError("raw record SHA-256 check failed")
            return payload

    def prune_after_db(
        self,
        delete_db_references: Callable[[], object],
        *,
        remove_file: bool = True,
    ) -> object:
        """Delete DB references first; a later crash leaves only an orphan file."""

        result = delete_db_references()
        if remove_file:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
        return result

    @staticmethod
    def sweep_orphans(
        directory: str | Path,
        *,
        referenced_paths: Iterable[str | Path],
        suffix: str = ".seg",
    ) -> tuple[Path, ...]:
        root = Path(directory)
        referenced = {Path(item).resolve() for item in referenced_paths}
        removed: list[Path] = []
        if not root.exists():
            return ()
        for path in root.glob(f"*{suffix}"):
            if path.resolve() not in referenced:
                path.unlink()
                removed.append(path)
        return tuple(removed)
