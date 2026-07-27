"""Durable APID-scoped CCSDS Space Packet sequence allocation."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from typing import Callable

from protocol.canonical import MAX_U32, checked_u32, checked_u64

from .u64 import encode_sqlite_u64
from .writer import MutationPriority, SQLiteWriter

SPACE_PACKET_SEQUENCE_MODULUS = 16_384


class SequenceAllocatorError(RuntimeError):
    pass


@dataclass(frozen=True)
class SequenceAllocation:
    spacecraft_instance_id: int
    apid: int
    sequence: int
    sequence_epoch: int
    rollover: bool
    reset_marker: str | None


@dataclass(frozen=True)
class SequenceState:
    spacecraft_instance_id: int
    apid: int
    next_sequence: int
    sequence_epoch: int
    last_reset_reason: str | None


class TcSequenceAllocator:
    """Persist Space Packet sequence counts independently from RequestKey."""

    def __init__(
        self,
        writer: SQLiteWriter,
        *,
        clock_us: Callable[[], int] | None = None,
    ) -> None:
        self.writer = writer
        self._clock_us = clock_us or (lambda: time.time_ns() // 1_000)

    @staticmethod
    def _validate_apid(apid: int) -> int:
        if isinstance(apid, bool) or not isinstance(apid, int) or not 0 <= apid <= 0x7FF:
            raise ValueError("APID must be an integer in [0, 2047]")
        return apid

    def _now_us(self) -> int:
        value = int(self._clock_us())
        if value < 0:
            raise SequenceAllocatorError("clock_us must be non-negative")
        return value

    def allocate(
        self,
        spacecraft_instance_id: int,
        apid: int,
    ) -> SequenceAllocation:
        return self.writer.mutate(
            "allocate_tc_sequence",
            lambda connection: self.allocate_in_transaction(
                connection, spacecraft_instance_id, apid
            ),
            priority=MutationPriority.HIGH,
        )

    def allocate_in_transaction(
        self,
        connection: sqlite3.Connection,
        spacecraft_instance_id: int,
        apid: int,
    ) -> SequenceAllocation:
        instance = checked_u64(spacecraft_instance_id, "spacecraft_instance_id")
        apid = self._validate_apid(apid)
        now_us = self._now_us()
        instance_blob = encode_sqlite_u64(instance)
        connection.execute(
            "INSERT OR IGNORE INTO tc_sequence_allocators("
            "spacecraft_instance_id,apid,next_sequence,sequence_epoch,"
            "last_reset_reason,updated_at_us) VALUES(?,?,0,0,NULL,?)",
            (instance_blob, apid, now_us),
        )
        row = connection.execute(
            "SELECT next_sequence,sequence_epoch,last_reset_reason "
            "FROM tc_sequence_allocators WHERE spacecraft_instance_id=? AND apid=?",
            (instance_blob, apid),
        ).fetchone()
        if row is None:
            raise SequenceAllocatorError("sequence allocator row disappeared")
        next_sequence = int(row[0])
        epoch = checked_u32(int(row[1]), "sequence_epoch")
        rollover = next_sequence == SPACE_PACKET_SEQUENCE_MODULUS
        sequence = 0 if rollover else next_sequence
        if rollover:
            if epoch == MAX_U32:
                raise SequenceAllocatorError(
                    "Space Packet sequence epoch exhausted; migrate spacecraft instance"
                )
            epoch += 1
            next_sequence = 1
        else:
            next_sequence = sequence + 1
        connection.execute(
            "UPDATE tc_sequence_allocators SET next_sequence=?,sequence_epoch=?,"
            "last_reset_reason=CASE WHEN ? THEN 'SPACE_PACKET_SEQUENCE_ROLLOVER' "
            "ELSE last_reset_reason END,updated_at_us=? "
            "WHERE spacecraft_instance_id=? AND apid=?",
            (next_sequence, epoch, int(rollover), now_us, instance_blob, apid),
        )
        return SequenceAllocation(
            instance,
            apid,
            sequence,
            epoch,
            rollover,
            "SPACE_PACKET_SEQUENCE_ROLLOVER" if rollover else None,
        )

    def reset(
        self,
        spacecraft_instance_id: int,
        apid: int,
        *,
        reason: str,
    ) -> SequenceState:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("sequence reset reason must not be empty")

        def mutation(connection: sqlite3.Connection) -> SequenceState:
            instance = checked_u64(spacecraft_instance_id, "spacecraft_instance_id")
            normalized_apid = self._validate_apid(apid)
            now_us = self._now_us()
            instance_blob = encode_sqlite_u64(instance)
            connection.execute(
                "INSERT OR IGNORE INTO tc_sequence_allocators("
                "spacecraft_instance_id,apid,next_sequence,sequence_epoch,"
                "last_reset_reason,updated_at_us) VALUES(?,?,0,0,NULL,?)",
                (instance_blob, normalized_apid, now_us),
            )
            row = connection.execute(
                "SELECT sequence_epoch FROM tc_sequence_allocators "
                "WHERE spacecraft_instance_id=? AND apid=?",
                (instance_blob, normalized_apid),
            ).fetchone()
            assert row is not None
            epoch = int(row[0])
            if epoch == MAX_U32:
                raise SequenceAllocatorError(
                    "sequence reset epoch exhausted; migrate spacecraft instance"
                )
            epoch += 1
            connection.execute(
                "UPDATE tc_sequence_allocators SET next_sequence=0,"
                "sequence_epoch=?,last_reset_reason=?,updated_at_us=? "
                "WHERE spacecraft_instance_id=? AND apid=?",
                (epoch, reason, now_us, instance_blob, normalized_apid),
            )
            return SequenceState(instance, normalized_apid, 0, epoch, reason)

        return self.writer.mutate(
            "reset_tc_sequence",
            mutation,
            priority=MutationPriority.HIGH,
        )

    def state(self, spacecraft_instance_id: int, apid: int) -> SequenceState | None:
        instance = checked_u64(spacecraft_instance_id, "spacecraft_instance_id")
        normalized_apid = self._validate_apid(apid)
        with self.writer.reader() as connection:
            row = connection.execute(
                "SELECT next_sequence,sequence_epoch,last_reset_reason "
                "FROM tc_sequence_allocators WHERE spacecraft_instance_id=? AND apid=?",
                (encode_sqlite_u64(instance), normalized_apid),
            ).fetchone()
            if row is None:
                return None
            return SequenceState(
                instance,
                normalized_apid,
                int(row[0]),
                int(row[1]),
                row[2],
            )
