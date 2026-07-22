"""Foundation exit-gate tests for P4A-01 through P4A-07."""

from __future__ import annotations

import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from gds.database import SQLiteProfile, WalLevel, classify_wal
from gds.idempotency import (
    DEFAULT_EXPIRY_SENTINEL,
    IdempotencyValidationError,
    build_semantic_idempotency,
)
from gds.ledger import (
    AtomicCommandLedger,
    IdempotencyConflictError,
    IdempotencyKeyRetiredError,
    NoContactError,
    OutboxCapacityError,
)
from gds.request_keys import RequestKeyAllocator, RequestNamespaceDrainingError
from gds.schema import SCHEMA_VERSION, SchemaCompatibilityError, SchemaIntegrityError
from gds.u64 import (
    U64CodecError,
    decode_sqlite_u64,
    decode_u64_cursor,
    encode_sqlite_u64,
    encode_u64_cursor,
    select_u64_keyset_page,
)
from gds.writer import (
    ActiveReaderError,
    LowPriorityDroppedError,
    MutationIntent,
    MutationPriority,
    SQLiteWriter,
    WriterAlreadyRunningError,
    WriterBackpressureError,
)
from protocol.canonical import MAX_U32
from protocol.schemas import CommandOpcode, http_idempotency_digest


class MutableClock:
    def __init__(self, value: datetime):
        self.value = value

    def __call__(self) -> datetime:
        return self.value


def admit_catalog(
    ledger: AtomicCommandLedger,
    key: str,
    *,
    target: int = 1,
    contact_available: bool = True,
    delivery_mode: str = "immediate",
):
    return ledger.admit(
        idempotency_key=key,
        target_spacecraft_instance_id=target,
        opcode=CommandOpcode.SCENE_REQUEST_CATALOG,
        payload={},
        contact_available=contact_available,
        delivery_mode=delivery_mode,
    )


def test_versioned_schema_profile_and_forward_only_compatibility(tmp_path: Path):
    path = tmp_path / "gds.sqlite3"
    with SQLiteWriter(path) as writer:
        profile = writer.mutate(
            "read_profile",
            lambda connection: (
                connection.execute("PRAGMA user_version").fetchone()[0],
                connection.execute("PRAGMA journal_mode").fetchone()[0],
                connection.execute("PRAGMA synchronous").fetchone()[0],
                connection.execute("PRAGMA foreign_keys").fetchone()[0],
                connection.execute("PRAGMA busy_timeout").fetchone()[0],
                connection.execute("PRAGMA wal_autocheckpoint").fetchone()[0],
            ),
            transactional=False,
        )
        assert profile == (SCHEMA_VERSION, "wal", 2, 1, 5000, 1000)
        with writer.reader() as reader:
            tables = {
                row[0]
                for row in reader.execute(
                    "SELECT name FROM sqlite_schema WHERE type='table'"
                )
            }
            assert {
                "commands",
                "command_outbox",
                "command_attempts",
                "system_state",
                "telemetry_samples",
                "events",
                "products",
                "simulation_runs",
                "replay_segments",
                "audit_log",
            } <= tables

    # Reopening is idempotent and does not create another migration row.
    with SQLiteWriter(path) as writer:
        count = writer.mutate(
            "migration_count",
            lambda connection: connection.execute(
                "SELECT count(*) FROM schema_migrations"
            ).fetchone()[0],
            transactional=False,
        )
        assert count == SCHEMA_VERSION

    connection = sqlite3.connect(path)
    connection.execute(f"PRAGMA user_version={SCHEMA_VERSION + 1}")
    connection.close()
    with pytest.raises(SchemaCompatibilityError, match="newer than binary"):
        SQLiteWriter(path)


def test_migration_checksum_tamper_fails_readiness(tmp_path: Path):
    path = tmp_path / "gds.sqlite3"
    with SQLiteWriter(path):
        pass
    connection = sqlite3.connect(path)
    connection.execute(
        "UPDATE schema_migrations SET checksum_sha256=? WHERE version=1",
        ("0" * 64,),
    )
    connection.commit()
    connection.close()
    with pytest.raises(SchemaIntegrityError, match="checksum mismatch"):
        SQLiteWriter(path)


def test_u64_blob_order_strict_cursor_and_keyset_pagination():
    boundaries = (0, 2**53 - 1, 2**53, 2**63 - 1, 2**63, 2**64 - 1)
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute(
        "CREATE TABLE u64_values(value BLOB NOT NULL CHECK(length(value)=8))"
    )
    for value in reversed(boundaries):
        connection.execute("INSERT INTO u64_values(value) VALUES(?)", (encode_sqlite_u64(value),))

    first = select_u64_keyset_page(
        connection, table="u64_values", u64_column="value", limit=3
    )
    second = select_u64_keyset_page(
        connection,
        table="u64_values",
        u64_column="value",
        after=first.next_cursor,
        limit=3,
    )
    values = [decode_sqlite_u64(row["value"]) for row in first.rows + second.rows]
    assert values == list(boundaries)
    assert first.next_cursor == "0020000000000000"
    assert second.next_cursor == "ffffffffffffffff"
    assert all(
        row[0] == "blob"
        for row in connection.execute("SELECT typeof(value) FROM u64_values")
    )
    for value in boundaries:
        assert decode_u64_cursor(encode_u64_cursor(value)) == value
    for malformed in ("000000000000000A", "0x00000000000001", "1", 1):
        with pytest.raises(U64CodecError):
            decode_u64_cursor(malformed)
    for malformed_blob in (b"", b"\x00" * 7, b"\x00" * 9, 1):
        with pytest.raises(U64CodecError):
            decode_sqlite_u64(malformed_blob)


def test_single_writer_priority_reserve_and_backpressure(tmp_path: Path):
    path = tmp_path / "gds.sqlite3"
    gate = threading.Event()
    started = threading.Event()

    def blocking(connection: sqlite3.Connection) -> str:
        started.set()
        assert gate.wait(5)
        return "released"

    writer = SQLiteWriter(path, queue_capacity=4, high_priority_reserve=1)
    try:
        with pytest.raises(WriterAlreadyRunningError):
            SQLiteWriter(path)
        active = writer.submit(MutationIntent("blocking", blocking))
        assert started.wait(2)
        low_futures = [
            writer.submit(
                MutationIntent(
                    f"low-{index}",
                    lambda connection, value=index: value,
                    MutationPriority.LOW,
                )
            )
            for index in range(3)
        ]
        with pytest.raises(LowPriorityDroppedError):
            writer.submit(
                MutationIntent("low-overflow", lambda connection: None, MutationPriority.LOW)
            )
        reserved = writer.submit(
            MutationIntent("reserved-high", lambda connection: "high")
        )
        with pytest.raises(WriterBackpressureError) as exc_info:
            writer.submit(MutationIntent("high-overflow", lambda connection: None))
        assert exc_info.value.status_code == 503
        gate.set()
        assert active.result(2) == "released"
        assert reserved.result(2) == "high"
        assert [future.result(2) for future in low_futures] == [0, 1, 2]
        metrics = writer.metrics()
        assert metrics.dropped_low == 1
        assert metrics.rejected_high == 1
        assert metrics.max_queue_depth == 4
        assert writer.writer_thread_id is not None
    finally:
        gate.set()
        writer.close()


def test_reader_is_query_only_and_truncate_waits_for_readers(tmp_path: Path):
    profile = SQLiteProfile(max_reader_seconds=0.01)
    with SQLiteWriter(tmp_path / "gds.sqlite3", profile=profile) as writer:
        with writer.reader() as reader:
            assert reader.execute("PRAGMA query_only").fetchone()[0] == 1
            with pytest.raises(sqlite3.OperationalError, match="readonly"):
                reader.execute(
                    "INSERT INTO system_state(state_key,value_json,updated_at_us) "
                    "VALUES('invalid','{}',0)"
                )
            with pytest.raises(ActiveReaderError):
                writer.checkpoint("TRUNCATE")
            time.sleep(0.02)
            status = writer.wal_status()
            assert status.active_readers == 1
            assert status.overdue_readers == 1
        assert writer.metrics().reader_overruns == 1
        assert writer.checkpoint("TRUNCATE")[0] == 0

    thresholds = SQLiteProfile(wal_warning_bytes=100, wal_throttle_bytes=200)
    assert classify_wal(99, thresholds) is WalLevel.NORMAL
    assert classify_wal(100, thresholds) is WalLevel.WARNING
    assert classify_wal(200, thresholds) is WalLevel.THROTTLED


def test_request_key_allocator_survives_restart(tmp_path: Path):
    path = tmp_path / "gds.sqlite3"
    random_values = iter((11, 22))
    with SQLiteWriter(path) as writer:
        allocator = RequestKeyAllocator(
            writer,
            random_u64=lambda: next(random_values),
            clock_us=lambda: 1,
        )
        initial = allocator.initialize()
        assert (initial.gds_installation_epoch, initial.ground_instance_id) == (11, 22)
        assert allocator.allocate().request_id == 1
        assert allocator.allocate().request_id == 2

    def unexpected_random() -> int:
        raise AssertionError("restart must not regenerate durable identity")

    with SQLiteWriter(path) as writer:
        allocator = RequestKeyAllocator(
            writer, random_u64=unexpected_random, clock_us=lambda: 2
        )
        restarted = allocator.initialize()
        assert restarted.gds_installation_epoch == 11
        assert restarted.ground_instance_id == 22
        assert allocator.allocate().request_id == 3


def test_request_id_wrap_rotates_only_after_old_namespace_drains(tmp_path: Path):
    clock = MutableClock(datetime(2026, 7, 19, 12, tzinfo=UTC))
    random_values = iter((100, 200, 300))
    with SQLiteWriter(tmp_path / "gds.sqlite3") as writer:
        allocator = RequestKeyAllocator(
            writer,
            random_u64=lambda: next(random_values),
            clock_us=lambda: 1,
        )
        ledger = AtomicCommandLedger(writer, allocator=allocator, clock=clock)
        writer.mutate(
            "near_wrap",
            lambda connection: connection.execute(
                "UPDATE ground_namespaces SET next_request_id=?,state='ACTIVE' "
                "WHERE ground_instance_id=?",
                (MAX_U32, encode_sqlite_u64(200)),
            ),
        )
        last = admit_catalog(ledger, "wrap-last")
        assert last.request_key.request_id == MAX_U32
        assert last.request_key.ground_instance_id == 200
        with pytest.raises(RequestNamespaceDrainingError):
            admit_catalog(ledger, "wrap-new")

        writer.mutate(
            "terminalize_old_namespace",
            lambda connection: (
                connection.execute(
                    "UPDATE commands SET command_state='ACKED',"
                    "terminal_at_us=updated_at_us"
                ),
                connection.execute(
                    "UPDATE command_outbox SET state='ACKED'"
                ),
            ),
        )
        rotated = admit_catalog(ledger, "wrap-new")
        assert rotated.request_key.request_id == 1
        assert rotated.request_key.ground_instance_id == 300
        assert allocator.state().gds_installation_epoch == 100


def test_jcs_semantics_default_sentinel_and_utc_normalization():
    first = build_semantic_idempotency(
        {
            "payload": {"\ue000": 1, "\U00010000": 2},
            "expires_at": "2026-07-19T19:00:00+07:00",
        }
    )
    second = build_semantic_idempotency(
        {
            "expires_at": "2026-07-19T12:00:00Z",
            "delivery_mode": "immediate",
            "payload": {"\U00010000": 2, "\ue000": 1},
        }
    )
    assert first.digest == second.digest
    assert first.digest_hex == http_idempotency_digest(
        {
            "payload": {"\ue000": 1, "\U00010000": 2},
            "expires_at": "2026-07-19T19:00:00+07:00",
        }
    )
    assert first.normalized_body["expires_at"] == "2026-07-19T12:00:00Z"
    encoded_supplementary = "\U00010000".encode("utf-8")
    encoded_private_use = "\ue000".encode("utf-8")
    assert first.canonical_jcs.index(encoded_supplementary) < first.canonical_jcs.index(
        encoded_private_use
    )

    omitted = build_semantic_idempotency({"payload": {}})
    explicit_mode = build_semantic_idempotency(
        {"payload": {}, "delivery_mode": "immediate"}
    )
    assert omitted.digest == explicit_mode.digest
    assert omitted.normalized_body["expires_at"] == DEFAULT_EXPIRY_SENTINEL
    with pytest.raises(IdempotencyValidationError, match="floating-point"):
        build_semantic_idempotency({"payload": {"threshold": 0.5}})
    with pytest.raises(IdempotencyValidationError, match="IEEE-754"):
        build_semantic_idempotency({"payload": {"unsafe": 2**53}})


def test_default_expiry_replay_conflict_and_contact_ordering(tmp_path: Path):
    clock = MutableClock(datetime(2026, 7, 19, 12, tzinfo=UTC))
    with SQLiteWriter(tmp_path / "gds.sqlite3") as writer:
        ledger = AtomicCommandLedger(writer, clock=clock)
        first = admit_catalog(ledger, "same-key", target=2**63)
        assert first.effective_expires_at == clock.value + timedelta(minutes=5)
        clock.value += timedelta(hours=1)
        replay = admit_catalog(
            ledger, "same-key", target=2**63, contact_available=False
        )
        assert replay.replayed
        assert replay.request_key == first.request_key
        assert replay.effective_expires_at == first.effective_expires_at

        with pytest.raises(IdempotencyConflictError) as exc_info:
            admit_catalog(ledger, "same-key", target=2**63 + 1)
        assert exc_info.value.request_key == first.request_key
        with pytest.raises(NoContactError):
            admit_catalog(ledger, "no-contact", contact_available=False)
        held = admit_catalog(
            ledger,
            "held",
            contact_available=False,
            delivery_mode="next_contact",
        )
        assert held.outbox_state == "HELD_NO_CONTACT"
        assert ledger.orphan_counts() == (0, 0)


def test_concurrent_same_key_creates_one_command_and_outbox(tmp_path: Path):
    clock = MutableClock(datetime(2026, 7, 19, 12, tzinfo=UTC))
    with SQLiteWriter(tmp_path / "gds.sqlite3") as writer:
        ledger = AtomicCommandLedger(writer, clock=clock)
        with ThreadPoolExecutor(max_workers=16) as pool:
            results = list(
                pool.map(
                    lambda _: admit_catalog(ledger, "concurrent-key", target=2**64 - 1),
                    range(32),
                )
            )
        assert len({result.request_key for result in results}) == 1
        assert sum(not result.replayed for result in results) == 1
        with writer.reader() as reader:
            assert reader.execute("SELECT count(*) FROM commands").fetchone()[0] == 1
            assert reader.execute("SELECT count(*) FROM command_outbox").fetchone()[0] == 1
        assert ledger.orphan_counts() == (0, 0)


def test_atomic_rollback_capacity_and_retry_before_capacity(tmp_path: Path):
    clock = MutableClock(datetime(2026, 7, 19, 12, tzinfo=UTC))
    fail = {"enabled": True}

    def inject(stage: str) -> None:
        if fail["enabled"] and stage == "after_command_insert":
            raise RuntimeError("simulated process failure")

    with SQLiteWriter(tmp_path / "gds.sqlite3") as writer:
        ledger = AtomicCommandLedger(
            writer,
            clock=clock,
            outbox_capacity=2,
            fault_injector=inject,
        )
        with pytest.raises(RuntimeError, match="simulated process failure"):
            admit_catalog(ledger, "crash-key")
        assert ledger.orphan_counts() == (0, 0)
        with writer.reader() as reader:
            assert reader.execute("SELECT count(*) FROM commands").fetchone()[0] == 0
            assert reader.execute("SELECT count(*) FROM command_outbox").fetchone()[0] == 0

        fail["enabled"] = False
        first = admit_catalog(ledger, "crash-key")
        assert first.request_key.request_id == 1
        second = admit_catalog(ledger, "capacity-2")
        with pytest.raises(OutboxCapacityError) as exc_info:
            admit_catalog(ledger, "capacity-3")
        assert exc_info.value.status_code == 429
        replay = admit_catalog(ledger, "crash-key", contact_available=False)
        assert replay.request_key == first.request_key
        assert second.request_key.request_id == 2
        assert ledger.orphan_counts() == (0, 0)


def test_retired_idempotency_marker_conflict_and_90_day_expiry(tmp_path: Path):
    clock = MutableClock(datetime(2026, 7, 19, 12, tzinfo=UTC))
    with SQLiteWriter(tmp_path / "gds.sqlite3") as writer:
        ledger = AtomicCommandLedger(writer, clock=clock)
        original = admit_catalog(ledger, "retained-key")
        writer.mutate(
            "terminalize",
            lambda connection: (
                connection.execute(
                    "UPDATE commands SET command_state='ACKED',"
                    "terminal_at_us=updated_at_us"
                ),
                connection.execute(
                    "UPDATE command_outbox SET state='ACKED'"
                ),
            ),
        )
        retained_until = ledger.retire_terminal_command(original.request_key)
        assert retained_until == clock.value + timedelta(days=90)
        assert ledger.get(original.request_key) is None
        with pytest.raises(IdempotencyKeyRetiredError) as retired:
            admit_catalog(ledger, "retained-key")
        assert retired.value.request_key == original.request_key
        with pytest.raises(IdempotencyConflictError):
            admit_catalog(ledger, "retained-key", target=2)

        clock.value = retained_until
        replacement = admit_catalog(ledger, "retained-key")
        assert replacement.request_key != original.request_key
        assert replacement.request_key.request_id == original.request_key.request_id + 1
