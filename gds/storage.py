"""Per-volume storage watermarks and durable reservation guard."""

from __future__ import annotations

import shutil
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Callable

from .idempotency import datetime_to_unix_us, unix_us_to_datetime
from .writer import MutationPriority, SQLiteWriter


class StorageFullError(RuntimeError):
    status_code = 507
    error_code = "STORAGE_FULL"


@dataclass(frozen=True)
class StorageSnapshot:
    volume: str
    used_bytes: int
    reserved_bytes: int
    cap_bytes: int
    high_watermark_bytes: int
    hard_watermark_bytes: int

    @property
    def hard_full(self) -> bool:
        return self.used_bytes + self.reserved_bytes >= self.hard_watermark_bytes


@dataclass(frozen=True)
class StorageReservation:
    reservation_id: int
    volume: str
    owner: str
    reserved_bytes: int
    expires_at: datetime


class StorageGuard:
    """Use a logical cap so tests and deployments can share one admission rule."""

    def __init__(
        self,
        writer: SQLiteWriter,
        root: str | Path,
        *,
        cap_bytes: int | None = None,
        high_watermark: float = 0.80,
        hard_watermark: float = 0.90,
        usage_provider: Callable[[], int] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.writer = writer
        self.root = Path(root)
        if cap_bytes is None:
            cap_bytes = int(shutil.disk_usage(self.root).total)
        if isinstance(cap_bytes, bool) or not isinstance(cap_bytes, int) or cap_bytes <= 0:
            raise ValueError("cap_bytes must be positive")
        if not 0 < high_watermark < hard_watermark <= 1:
            raise ValueError("watermarks must satisfy 0 < high < hard <= 1")
        self.volume = str(self.root.resolve())
        self.cap_bytes = cap_bytes
        self.high_watermark_bytes = int(cap_bytes * high_watermark)
        self.hard_watermark_bytes = int(cap_bytes * hard_watermark)
        self._usage_provider = usage_provider or (
            lambda: int(shutil.disk_usage(self.root).used)
        )
        self._clock = clock or (lambda: datetime.now(UTC))

    def _now_us(self) -> int:
        now = self._clock()
        if now.tzinfo is None:
            raise ValueError("storage clock must be timezone-aware")
        return datetime_to_unix_us(now.astimezone(UTC))

    def used_bytes(self) -> int:
        value = int(self._usage_provider())
        if value < 0:
            raise ValueError("usage provider returned a negative value")
        return value

    def snapshot(self) -> StorageSnapshot:
        now_us = self._now_us()
        with self.writer.reader() as connection:
            reserved = int(
                connection.execute(
                    "SELECT COALESCE(sum(reserved_bytes),0) FROM storage_reservations "
                    "WHERE volume=? AND state='ACTIVE' AND expires_at_us>?",
                    (self.volume, now_us),
                ).fetchone()[0]
            )
        return StorageSnapshot(
            self.volume,
            self.used_bytes(),
            reserved,
            self.cap_bytes,
            self.high_watermark_bytes,
            self.hard_watermark_bytes,
        )

    def ensure_admission(self, additional_bytes: int = 0) -> None:
        if isinstance(additional_bytes, bool) or not isinstance(additional_bytes, int):
            raise ValueError("additional_bytes must be an integer")
        if additional_bytes < 0:
            raise ValueError("additional_bytes must be non-negative")
        snapshot = self.snapshot()
        if snapshot.used_bytes + snapshot.reserved_bytes + additional_bytes >= snapshot.hard_watermark_bytes:
            raise StorageFullError(
                f"storage volume {self.volume} is above the hard watermark"
            )

    def reserve(
        self,
        owner: str,
        reserved_bytes: int,
        *,
        ttl: timedelta = timedelta(hours=1),
    ) -> StorageReservation:
        if not isinstance(owner, str) or not owner:
            raise ValueError("reservation owner must not be empty")
        if isinstance(reserved_bytes, bool) or not isinstance(reserved_bytes, int) or reserved_bytes <= 0:
            raise ValueError("reserved_bytes must be positive")
        if ttl <= timedelta(0):
            raise ValueError("reservation ttl must be positive")

        def mutation(connection: sqlite3.Connection) -> StorageReservation:
            now_us = self._now_us()
            connection.execute(
                "UPDATE storage_reservations SET state='EXPIRED' "
                "WHERE volume=? AND state='ACTIVE' AND expires_at_us<=?",
                (self.volume, now_us),
            )
            active = int(
                connection.execute(
                    "SELECT COALESCE(sum(reserved_bytes),0) FROM storage_reservations "
                    "WHERE volume=? AND state='ACTIVE'",
                    (self.volume,),
                ).fetchone()[0]
            )
            if self.used_bytes() + active + reserved_bytes >= self.hard_watermark_bytes:
                raise StorageFullError("storage reservation would cross hard watermark")
            expires_at_us = now_us + int(ttl.total_seconds() * 1_000_000)
            cursor = connection.execute(
                "INSERT INTO storage_reservations(volume,owner,reserved_bytes,state,"
                "expires_at_us,created_at_us) VALUES(?,?,?,'ACTIVE',?,?)",
                (self.volume, owner, reserved_bytes, expires_at_us, now_us),
            )
            return StorageReservation(
                int(cursor.lastrowid),
                self.volume,
                owner,
                reserved_bytes,
                unix_us_to_datetime(expires_at_us),
            )

        return self.writer.mutate(
            "reserve_storage_headroom",
            mutation,
            priority=MutationPriority.HIGH,
        )

    def release(self, reservation_id: int) -> None:
        if isinstance(reservation_id, bool) or not isinstance(reservation_id, int) or reservation_id <= 0:
            raise ValueError("reservation_id must be positive")
        self.writer.mutate(
            "release_storage_reservation",
            lambda connection: connection.execute(
                "UPDATE storage_reservations SET state='RELEASED',released_at_us=? "
                "WHERE reservation_id=? AND state='ACTIVE'",
                (self._now_us(), reservation_id),
            ),
            priority=MutationPriority.HIGH,
        )

    def expire(self) -> int:
        return self.writer.mutate(
            "expire_storage_reservations",
            lambda connection: connection.execute(
                "UPDATE storage_reservations SET state='EXPIRED' "
                "WHERE volume=? AND state='ACTIVE' AND expires_at_us<=?",
                (self.volume, self._now_us()),
            ).rowcount,
            priority=MutationPriority.HIGH,
        )
