"""Forward-only SQLite schema migrations for the GDS ledger."""

from __future__ import annotations

import hashlib
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

SCHEMA_VERSION = 9


class SchemaError(RuntimeError):
    """Base class for schema startup/readiness failures."""


class SchemaCompatibilityError(SchemaError):
    """The database schema cannot be used by this binary."""


class SchemaIntegrityError(SchemaError):
    """The version metadata or physical schema is inconsistent."""


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    sql: str

    @property
    def checksum(self) -> str:
        return hashlib.sha256(self.sql.encode("utf-8")).hexdigest()


def _load_migrations() -> tuple[Migration, ...]:
    root = Path(__file__).with_name("migrations")
    migrations = []
    for path in sorted(root.glob("[0-9][0-9][0-9]_*.sql")):
        version_text, _, name = path.stem.partition("_")
        migrations.append(
            Migration(int(version_text), name, path.read_text(encoding="utf-8"))
        )
    versions = [item.version for item in migrations]
    if versions != list(range(1, SCHEMA_VERSION + 1)):
        raise SchemaIntegrityError(
            f"migration set {versions!r} does not cover 1..{SCHEMA_VERSION}"
        )
    return tuple(migrations)


MIGRATIONS = _load_migrations()

REQUIRED_TABLES = frozenset(
    {
        "schema_migrations",
        "ground_namespaces",
        "gds_metadata",
        "spacecraft_instances",
        "commands",
        "command_outbox",
        "command_attempts",
        "system_state",
        "catalog_snapshots",
        "scenes",
        "telemetry_samples",
        "telemetry_rollups",
        "events",
        "link_frames",
        "jobs",
        "products",
        "product_transfers",
        "simulation_runs",
        "replay_segments",
        "http_idempotency_retired",
        "product_tombstones",
        "audit_log",
        "tc_sequence_allocators",
        "event_sequences",
        "storage_reservations",
        "scene_packages",
        "file_reassemblies",
        "file_reassembly_packets",
        "file_reassembly_ranges",
        "product_artifacts",
        "product_downlink_ledger",
        "product_downlink_pending_files",
        "tm_source_generations",
        "tm_channel_counter_states",
        "tm_packet_counter_states",
        "tm_counter_states",
        "tm_counter_observations",
    }
)


def current_schema_version(connection: sqlite3.Connection) -> int:
    return int(connection.execute("PRAGMA user_version").fetchone()[0])


def _application_tables(connection: sqlite3.Connection) -> set[str]:
    rows = connection.execute(
        "SELECT name FROM sqlite_schema "
        "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    return {str(row[0]) for row in rows}


def _verify_applied_migrations(connection: sqlite3.Connection, version: int) -> None:
    if version == 0:
        if _application_tables(connection):
            raise SchemaCompatibilityError(
                "unversioned SQLite database is not a GDS database"
            )
        return
    if "schema_migrations" not in _application_tables(connection):
        raise SchemaIntegrityError("schema_migrations is missing")
    rows = connection.execute(
        "SELECT version,name,checksum_sha256 FROM schema_migrations ORDER BY version"
    ).fetchall()
    if len(rows) != version:
        raise SchemaIntegrityError(
            f"user_version={version} but {len(rows)} migration rows exist"
        )
    for row, expected in zip(rows, MIGRATIONS[:version], strict=True):
        if (int(row[0]), str(row[1]), str(row[2])) != (
            expected.version,
            expected.name,
            expected.checksum,
        ):
            raise SchemaIntegrityError(
                f"migration {expected.version} metadata/checksum mismatch"
            )


def migrate(connection: sqlite3.Connection, *, now_us: int | None = None) -> int:
    """Apply every missing migration; downgrade and unknown DBs fail closed."""

    version = current_schema_version(connection)
    if version > SCHEMA_VERSION:
        raise SchemaCompatibilityError(
            f"database schema {version} is newer than binary schema {SCHEMA_VERSION}"
        )
    _verify_applied_migrations(connection, version)
    applied_at_us = int(time.time_ns() // 1_000 if now_us is None else now_us)
    for migration in MIGRATIONS[version:]:
        checksum = migration.checksum.replace("'", "''")
        name = migration.name.replace("'", "''")
        script = (
            "BEGIN IMMEDIATE;\n"
            + migration.sql
            + "\nINSERT INTO schema_migrations"
            + "(version,name,checksum_sha256,applied_at_us) VALUES "
            + f"({migration.version},'{name}','{checksum}',{applied_at_us});\n"
            + f"PRAGMA user_version={migration.version};\nCOMMIT;"
        )
        try:
            connection.executescript(script)
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
    validate_schema(connection)
    return SCHEMA_VERSION


def validate_schema(connection: sqlite3.Connection) -> None:
    version = current_schema_version(connection)
    if version != SCHEMA_VERSION:
        raise SchemaCompatibilityError(
            f"database schema {version} does not match binary schema {SCHEMA_VERSION}"
        )
    _verify_applied_migrations(connection, version)
    missing = REQUIRED_TABLES - _application_tables(connection)
    if missing:
        raise SchemaIntegrityError(f"required tables are missing: {sorted(missing)!r}")
    foreign_key_errors = connection.execute("PRAGMA foreign_key_check").fetchall()
    if foreign_key_errors:
        raise SchemaIntegrityError(
            f"foreign key check failed for {len(foreign_key_errors)} row(s)"
        )
    quick_check = str(connection.execute("PRAGMA quick_check").fetchone()[0])
    if quick_check != "ok":
        raise SchemaIntegrityError(f"SQLite quick_check failed: {quick_check}")
