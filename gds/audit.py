"""Durable operator and system audit records."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Callable, Mapping

from .idempotency import datetime_to_unix_us, unix_us_to_datetime
from .writer import MutationPriority, SQLiteWriter


def _json_value(value: Any) -> str:
    def fallback(item: Any) -> str:
        if isinstance(item, bytes):
            return item.hex()
        if isinstance(item, datetime):
            return item.astimezone(UTC).isoformat().replace("+00:00", "Z")
        raise TypeError(f"unsupported audit value {type(item).__name__}")

    return json.dumps(
        value,
        default=fallback,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def append_audit_in_transaction(
    connection: sqlite3.Connection,
    *,
    principal: str,
    action: str,
    target_type: str,
    target_identity: Mapping[str, Any],
    old_value: Any,
    new_value: Any,
    created_at_us: int,
) -> int:
    if not isinstance(principal, str) or not principal:
        raise ValueError("audit principal must not be empty")
    if not isinstance(action, str) or not action:
        raise ValueError("audit action must not be empty")
    if not isinstance(target_type, str) or not target_type:
        raise ValueError("audit target_type must not be empty")
    if created_at_us < 0:
        raise ValueError("audit timestamp must be non-negative")
    cursor = connection.execute(
        "INSERT INTO audit_log(principal,action,target_type,target_identity_json,"
        "old_value_json,new_value_json,created_at_us) VALUES(?,?,?,?,?,?,?)",
        (
            principal,
            action,
            target_type,
            _json_value(dict(target_identity)),
            None if old_value is None else _json_value(old_value),
            None if new_value is None else _json_value(new_value),
            created_at_us,
        ),
    )
    return int(cursor.lastrowid)


@dataclass(frozen=True)
class AuditEntry:
    audit_id: int
    principal: str
    action: str
    target_type: str
    target_identity: dict[str, Any]
    old_value: Any
    new_value: Any
    created_at: datetime


class AuditStore:
    def __init__(
        self,
        writer: SQLiteWriter,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.writer = writer
        self._clock = clock or (lambda: datetime.now(UTC))

    def append(
        self,
        *,
        principal: str,
        action: str,
        target_type: str,
        target_identity: Mapping[str, Any],
        old_value: Any = None,
        new_value: Any = None,
    ) -> int:
        now = self._clock()
        if now.tzinfo is None:
            raise ValueError("audit clock must be timezone-aware")
        return self.writer.mutate(
            "append_audit",
            lambda connection: append_audit_in_transaction(
                connection,
                principal=principal,
                action=action,
                target_type=target_type,
                target_identity=target_identity,
                old_value=old_value,
                new_value=new_value,
                created_at_us=datetime_to_unix_us(now),
            ),
            priority=MutationPriority.HIGH,
        )

    def list(
        self,
        *,
        after_audit_id: int | None = None,
        limit: int = 100,
    ) -> tuple[AuditEntry, ...]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1_000:
            raise ValueError("limit must be in [1, 1000]")
        if after_audit_id is not None and (
            isinstance(after_audit_id, bool)
            or not isinstance(after_audit_id, int)
            or after_audit_id < 0
        ):
            raise ValueError("after_audit_id must be a non-negative integer")
        with self.writer.reader() as connection:
            if after_audit_id is None:
                rows = connection.execute(
                    "SELECT * FROM audit_log ORDER BY audit_id LIMIT ?", (limit,)
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM audit_log WHERE audit_id>? "
                    "ORDER BY audit_id LIMIT ?",
                    (after_audit_id, limit),
                ).fetchall()
        return tuple(self._from_row(row) for row in rows)

    @staticmethod
    def _from_row(row: sqlite3.Row) -> AuditEntry:
        return AuditEntry(
            int(row["audit_id"]),
            str(row["principal"]),
            str(row["action"]),
            str(row["target_type"]),
            json.loads(str(row["target_identity_json"])),
            None if row["old_value_json"] is None else json.loads(str(row["old_value_json"])),
            None if row["new_value_json"] is None else json.loads(str(row["new_value_json"])),
            unix_us_to_datetime(int(row["created_at_us"])),
        )
