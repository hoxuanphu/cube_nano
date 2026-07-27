"""Durable monotonic event cursor and keyset event reader."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Callable, Mapping, Sequence

from protocol.canonical import MAX_U64, checked_u32, checked_u64, u64_to_json
from protocol.schemas import RequestKey

from .audit import _json_value
from .idempotency import datetime_to_unix_us, unix_us_to_datetime
from .u64 import decode_sqlite_u64, decode_u64_cursor, encode_sqlite_u64, encode_u64_cursor
from .writer import MutationPriority, SQLiteWriter


class EventStoreError(RuntimeError):
    pass


@dataclass(frozen=True)
class EventRecord:
    event_id: int
    event_name: str
    severity: str
    message: Any
    server_time: datetime
    source_spacecraft_instance_id: int | None = None
    target_spacecraft_instance_id: int | None = None
    source_boot_id: int | None = None
    request_key: RequestKey | None = None
    dictionary_version: str | None = None

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "event_id": u64_to_json(self.event_id),
            "event_name": self.event_name,
            "severity": self.severity,
            "message": self.message,
            "server_time": self.server_time.isoformat().replace("+00:00", "Z"),
        }
        if self.source_spacecraft_instance_id is not None:
            result["source_spacecraft_instance_id"] = u64_to_json(
                self.source_spacecraft_instance_id
            )
        if self.target_spacecraft_instance_id is not None:
            result["target_spacecraft_instance_id"] = u64_to_json(
                self.target_spacecraft_instance_id
            )
        if self.source_boot_id is not None:
            result["source_boot_id"] = self.source_boot_id
        if self.request_key is not None:
            result["request_key"] = self.request_key.as_dict()
        if self.dictionary_version is not None:
            result["dictionary_version"] = self.dictionary_version
        return result


def _optional_u64(value: int | None, label: str) -> int | None:
    return None if value is None else checked_u64(value, label)


def _optional_u32(value: int | None, label: str) -> int | None:
    return None if value is None else checked_u32(value, label)


class EventStore:
    def __init__(
        self,
        writer: SQLiteWriter,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.writer = writer
        self._clock = clock or (lambda: datetime.now(UTC))

    def _now_us(self) -> int:
        now = self._clock()
        if now.tzinfo is None:
            raise ValueError("event clock must be timezone-aware")
        return datetime_to_unix_us(now.astimezone(UTC))

    @staticmethod
    def append_in_transaction(
        connection: sqlite3.Connection,
        *,
        event_name: str,
        severity: str,
        message: Any,
        server_time_us: int,
        source_spacecraft_instance_id: int | None = None,
        target_spacecraft_instance_id: int | None = None,
        source_boot_id: int | None = None,
        request_key: RequestKey | None = None,
        dictionary_version: str | None = None,
    ) -> EventRecord:
        if not isinstance(event_name, str) or not event_name:
            raise ValueError("event_name must not be empty")
        if not isinstance(severity, str) or not severity:
            raise ValueError("severity must not be empty")
        if server_time_us < 0:
            raise ValueError("server_time_us must be non-negative")
        source = _optional_u64(source_spacecraft_instance_id, "source_spacecraft_instance_id")
        target = _optional_u64(target_spacecraft_instance_id, "target_spacecraft_instance_id")
        boot = _optional_u32(source_boot_id, "source_boot_id")
        if request_key is not None and not isinstance(request_key, RequestKey):
            raise TypeError("request_key must be a RequestKey")
        message_json = _json_value(message)
        connection.execute(
            "INSERT OR IGNORE INTO event_sequences(singleton,next_event_id,updated_at_us) "
            "VALUES(1,?,?)",
            (encode_sqlite_u64(1), server_time_us),
        )
        row = connection.execute(
            "SELECT next_event_id FROM event_sequences WHERE singleton=1"
        ).fetchone()
        if row is None:
            raise EventStoreError("event sequence row is missing")
        event_id = decode_sqlite_u64(row[0], "next_event_id")
        if event_id >= MAX_U64:
            raise EventStoreError("event cursor exhausted; create a new GDS epoch")
        connection.execute(
            "UPDATE event_sequences SET next_event_id=?,updated_at_us=? WHERE singleton=1",
            (encode_sqlite_u64(event_id + 1), server_time_us),
        )
        event_blob = encode_sqlite_u64(event_id)
        ground_blob = None
        request_id = None
        if request_key is not None:
            ground_blob = encode_sqlite_u64(request_key.ground_instance_id)
            request_id = request_key.request_id
        connection.execute(
            "INSERT INTO events(event_id,source_spacecraft_instance_id,"
            "target_spacecraft_instance_id,source_boot_id,ground_instance_id,request_id,"
            "severity,event_name,dictionary_version,message_json,server_time_us) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (
                event_blob,
                None if source is None else encode_sqlite_u64(source),
                None if target is None else encode_sqlite_u64(target),
                boot,
                ground_blob,
                request_id,
                severity,
                event_name,
                dictionary_version,
                message_json,
                server_time_us,
            ),
        )
        return EventRecord(
            event_id,
            event_name,
            severity,
            json.loads(message_json),
            unix_us_to_datetime(server_time_us),
            source,
            target,
            boot,
            request_key,
            dictionary_version,
        )

    def append(
        self,
        event_name: str,
        *,
        severity: str = "INFO",
        message: Any = None,
        source_spacecraft_instance_id: int | None = None,
        target_spacecraft_instance_id: int | None = None,
        source_boot_id: int | None = None,
        request_key: RequestKey | None = None,
        dictionary_version: str | None = None,
    ) -> EventRecord:
        return self.writer.mutate(
            "append_event",
            lambda connection: self.append_in_transaction(
                connection,
                event_name=event_name,
                severity=severity,
                message=message,
                server_time_us=self._now_us(),
                source_spacecraft_instance_id=source_spacecraft_instance_id,
                target_spacecraft_instance_id=target_spacecraft_instance_id,
                source_boot_id=source_boot_id,
                request_key=request_key,
                dictionary_version=dictionary_version,
            ),
            priority=MutationPriority.HIGH,
        )

    def append_batch(self, events: Sequence[Mapping[str, Any]]) -> tuple[EventRecord, ...]:
        if not 1 <= len(events) <= 100:
            raise ValueError("event batch must contain 1..100 events")
        now_us = self._now_us()
        normalized = []
        for event in events:
            if not isinstance(event, Mapping):
                raise TypeError("event batch entries must be objects")
            item = dict(event)
            item.pop("server_time_us", None)
            normalized.append(item)
        return self.writer.mutate(
            "append_event_batch",
            lambda connection: tuple(
                self.append_in_transaction(
                    connection,
                    server_time_us=now_us,
                    **dict(event),
                )
                for event in normalized
            ),
            priority=MutationPriority.HIGH,
        )

    def list_events(
        self,
        *,
        after_event_id: int | str | None = None,
        limit: int = 100,
    ) -> tuple[tuple[EventRecord, ...], str | None]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1_000:
            raise ValueError("limit must be in [1, 1000]")
        if isinstance(after_event_id, str):
            after = decode_u64_cursor(after_event_id)
        elif after_event_id is None:
            after = None
        else:
            after = checked_u64(after_event_id, "after_event_id")
        with self.writer.reader() as connection:
            if after is None:
                rows = connection.execute(
                    "SELECT * FROM events ORDER BY event_id LIMIT ?", (limit,)
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM events WHERE event_id>? ORDER BY event_id LIMIT ?",
                    (encode_sqlite_u64(after), limit),
                ).fetchall()
        records = tuple(self._from_row(row) for row in rows)
        cursor = None if not records else encode_u64_cursor(records[-1].event_id)
        return records, cursor

    def latest_event_id(self) -> int:
        """Return the highest durable cursor, or zero before the first event."""

        with self.writer.reader() as connection:
            row = connection.execute(
                "SELECT event_id FROM events ORDER BY event_id DESC LIMIT 1"
            ).fetchone()
        return 0 if row is None else decode_sqlite_u64(row[0], "event_id")

    def oldest_event_id(self) -> int:
        """Return the first retained cursor, or the next cursor when empty."""

        with self.writer.reader() as connection:
            row = connection.execute(
                "SELECT event_id FROM events ORDER BY event_id LIMIT 1"
            ).fetchone()
        return 1 if row is None else decode_sqlite_u64(row[0], "event_id")

    @staticmethod
    def _from_row(row: sqlite3.Row) -> EventRecord:
        source = (
            None
            if row["source_spacecraft_instance_id"] is None
            else decode_sqlite_u64(row["source_spacecraft_instance_id"], "source")
        )
        target = (
            None
            if row["target_spacecraft_instance_id"] is None
            else decode_sqlite_u64(row["target_spacecraft_instance_id"], "target")
        )
        request_key = None
        if row["ground_instance_id"] is not None:
            request_key = RequestKey(
                decode_sqlite_u64(row["ground_instance_id"], "ground_instance_id"),
                int(row["request_id"]),
            )
        return EventRecord(
            decode_sqlite_u64(row["event_id"], "event_id"),
            str(row["event_name"]),
            str(row["severity"]),
            json.loads(str(row["message_json"])),
            unix_us_to_datetime(int(row["server_time_us"])),
            source,
            target,
            None if row["source_boot_id"] is None else int(row["source_boot_id"]),
            request_key,
            row["dictionary_version"],
        )
