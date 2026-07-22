"""SQLite connection profile and WAL health primitives for the GDS."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

DEFAULT_BUSY_TIMEOUT_MS = 5_000
DEFAULT_WAL_AUTOCHECKPOINT_PAGES = 1_000
DEFAULT_WAL_WARNING_BYTES = 128 * 1024 * 1024
DEFAULT_WAL_THROTTLE_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class SQLiteProfile:
    busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS
    wal_autocheckpoint_pages: int = DEFAULT_WAL_AUTOCHECKPOINT_PAGES
    wal_warning_bytes: int = DEFAULT_WAL_WARNING_BYTES
    wal_throttle_bytes: int = DEFAULT_WAL_THROTTLE_BYTES
    max_reader_seconds: float = 2.0

    def __post_init__(self) -> None:
        if self.busy_timeout_ms < 0:
            raise ValueError("busy_timeout_ms must be non-negative")
        if self.wal_autocheckpoint_pages <= 0:
            raise ValueError("wal_autocheckpoint_pages must be positive")
        if not 0 < self.wal_warning_bytes < self.wal_throttle_bytes:
            raise ValueError("WAL warning threshold must be below throttle threshold")
        if self.max_reader_seconds <= 0:
            raise ValueError("max_reader_seconds must be positive")


class WalLevel(str, Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    THROTTLED = "THROTTLED"


@dataclass(frozen=True)
class WalStatus:
    size_bytes: int
    level: WalLevel
    active_readers: int
    overdue_readers: int

    @property
    def admit_low_priority(self) -> bool:
        return self.level is not WalLevel.THROTTLED


def _pragma_value(connection: sqlite3.Connection, pragma: str) -> object:
    row = connection.execute(f"PRAGMA {pragma}").fetchone()
    return None if row is None else row[0]


def open_writer_connection(
    path: str | Path, profile: SQLiteProfile
) -> sqlite3.Connection:
    database_path = Path(path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(
        database_path,
        timeout=profile.busy_timeout_ms / 1_000,
        isolation_level=None,
    )
    connection.row_factory = sqlite3.Row
    try:
        mode = str(_pragma_value(connection, "journal_mode=WAL")).lower()
        if mode != "wal":
            raise RuntimeError(f"SQLite refused WAL mode (reported {mode!r})")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute(f"PRAGMA busy_timeout={profile.busy_timeout_ms}")
        connection.execute(
            f"PRAGMA wal_autocheckpoint={profile.wal_autocheckpoint_pages}"
        )
        verify_connection_profile(connection, profile, writer=True)
        return connection
    except Exception:
        connection.close()
        raise


def open_reader_connection(
    path: str | Path, profile: SQLiteProfile
) -> sqlite3.Connection:
    absolute = Path(path).resolve().as_posix()
    connection = sqlite3.connect(
        f"file:{absolute}?mode=ro",
        uri=True,
        timeout=profile.busy_timeout_ms / 1_000,
        isolation_level=None,
    )
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute(f"PRAGMA busy_timeout={profile.busy_timeout_ms}")
        verify_connection_profile(connection, profile, writer=False)
        return connection
    except Exception:
        connection.close()
        raise


def verify_connection_profile(
    connection: sqlite3.Connection, profile: SQLiteProfile, *, writer: bool
) -> None:
    mode = str(_pragma_value(connection, "journal_mode")).lower()
    synchronous = int(_pragma_value(connection, "synchronous"))
    foreign_keys = int(_pragma_value(connection, "foreign_keys"))
    busy_timeout = int(_pragma_value(connection, "busy_timeout"))
    if mode != "wal":
        raise RuntimeError(f"journal_mode must be WAL, got {mode!r}")
    if writer and synchronous != 2:
        raise RuntimeError(f"synchronous must be FULL (2), got {synchronous}")
    if foreign_keys != 1:
        raise RuntimeError("foreign_keys must be enabled")
    if busy_timeout != profile.busy_timeout_ms:
        raise RuntimeError(
            f"busy_timeout must be {profile.busy_timeout_ms}, got {busy_timeout}"
        )
    if writer:
        autocheckpoint = int(_pragma_value(connection, "wal_autocheckpoint"))
        if autocheckpoint != profile.wal_autocheckpoint_pages:
            raise RuntimeError(
                "wal_autocheckpoint must be "
                f"{profile.wal_autocheckpoint_pages}, got {autocheckpoint}"
            )


def wal_size_bytes(path: str | Path) -> int:
    wal_path = Path(f"{Path(path)}-wal")
    try:
        return wal_path.stat().st_size
    except FileNotFoundError:
        return 0


def classify_wal(size_bytes: int, profile: SQLiteProfile) -> WalLevel:
    if size_bytes >= profile.wal_throttle_bytes:
        return WalLevel.THROTTLED
    if size_bytes >= profile.wal_warning_bytes:
        return WalLevel.WARNING
    return WalLevel.NORMAL
