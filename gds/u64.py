"""Lossless SQLite and API cursor codec for mission U64 values."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from typing import Any

from protocol.canonical import (
    checked_u64,
    u64_from_bytes,
    u64_from_json,
    u64_to_bytes,
    u64_to_json,
)

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class U64CodecError(ValueError):
    """A U64 crossed a boundary in a lossy or non-canonical form."""


def encode_sqlite_u64(value: int) -> bytes:
    """Encode U64 as fixed-width big-endian BLOB for unsigned lexical order."""

    try:
        return u64_to_bytes(value)
    except (TypeError, ValueError) as exc:
        raise U64CodecError(str(exc)) from exc


def decode_sqlite_u64(value: object, label: str = "SQLite U64") -> int:
    try:
        return u64_from_bytes(value, label)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise U64CodecError(str(exc)) from exc


def encode_u64_cursor(value: int) -> str:
    try:
        return u64_to_json(value)
    except (TypeError, ValueError) as exc:
        raise U64CodecError(str(exc)) from exc


def decode_u64_cursor(value: object) -> int:
    try:
        return u64_from_json(value, "cursor")
    except (TypeError, ValueError) as exc:
        raise U64CodecError(str(exc)) from exc


@dataclass(frozen=True)
class U64KeysetPage:
    rows: tuple[sqlite3.Row, ...]
    next_cursor: str | None


def _checked_identifier(value: str, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{label} is not a safe SQLite identifier")
    return value


def select_u64_keyset_page(
    connection: sqlite3.Connection,
    *,
    table: str,
    u64_column: str,
    after: str | None = None,
    limit: int = 100,
) -> U64KeysetPage:
    """Read one ascending page using ``BLOB(8) > cursor`` and no OFFSET."""

    table = _checked_identifier(table, "table")
    u64_column = _checked_identifier(u64_column, "u64_column")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1_000:
        raise ValueError("limit must be an integer in [1, 1000]")
    if connection.row_factory is not sqlite3.Row:
        raise ValueError("connection.row_factory must be sqlite3.Row")
    if after is None:
        sql = f"SELECT * FROM {table} ORDER BY {u64_column} ASC LIMIT ?"
        parameters: tuple[Any, ...] = (limit,)
    else:
        cursor = encode_sqlite_u64(decode_u64_cursor(after))
        sql = (
            f"SELECT * FROM {table} WHERE {u64_column} > ? "
            f"ORDER BY {u64_column} ASC LIMIT ?"
        )
        parameters = (cursor, limit)
    rows = tuple(connection.execute(sql, parameters).fetchall())
    next_cursor = None
    if rows:
        last_value = decode_sqlite_u64(rows[-1][u64_column], u64_column)
        next_cursor = encode_u64_cursor(checked_u64(last_value, u64_column))
    return U64KeysetPage(rows, next_cursor)
