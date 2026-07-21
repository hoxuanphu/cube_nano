"""Runtime and exit-gate tests for P4A-08 through P4A-14."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Iterator

import pytest

from gds.api import GDSApi
from gds.binding import BindingChangedError, SpacecraftBindingManager
from gds.events import EventStore
from gds.ledger import AtomicCommandLedger, NoContactError
from gds.outbox import ContactState, OutboxService
from gds.raw_segments import RawSegmentStore
from gds.sequence import TcSequenceAllocator
from gds.storage import StorageFullError, StorageGuard
from gds.telemetry import TelemetryConflictError, TelemetrySample, TelemetryStore
from gds.writer import SQLiteWriter, WriterBackpressureError
from protocol.schemas import CommandOpcode


class Clock:
    def __init__(self, value: datetime | None = None):
        self.value = value or datetime(2026, 7, 19, 12, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.value

    def advance(self, **kwargs: int) -> None:
        self.value += timedelta(**kwargs)


@contextmanager
def stack(
    tmp_path: Path,
    *,
    contact_state: ContactState = ContactState.CONTACT_OPEN,
    outbox_capacity: int = 1_024,
) -> Iterator[tuple[SQLiteWriter, AtomicCommandLedger, OutboxService, Clock]]:
    clock = Clock()
    with SQLiteWriter(tmp_path / "gds.sqlite3") as writer:
        ledger = AtomicCommandLedger(
            writer,
            clock=clock,
            outbox_capacity=outbox_capacity,
        )
        outbox = OutboxService(writer, clock=clock)
        outbox.register_instance(
            1,
            link_generation=1,
            link_session_id=1,
            contact_state=contact_state,
        )
        yield writer, ledger, outbox, clock


def admit(
    ledger: AtomicCommandLedger,
    key: str,
    *,
    target: int = 1,
    delivery_mode: str = "immediate",
    contact_available: bool = True,
):
    return ledger.admit(
        idempotency_key=key,
        target_spacecraft_instance_id=target,
        opcode=CommandOpcode.SCENE_REQUEST_CATALOG,
        payload={},
        delivery_mode=delivery_mode,
        contact_available=contact_available,
    )


def test_outbox_lease_crash_recovery_and_ack_retry(tmp_path: Path):
    with stack(tmp_path) as (writer, ledger, outbox, clock):
        accepted = admit(ledger, "retry")
        lease = outbox.claim_next()
        assert lease is not None
        attempt = outbox.persist_attempt(lease, b"tc-1", apid=7)
        clock.advance(seconds=10, microseconds=1)
        report = outbox.reconcile()
        assert report.recovered_leases == 1
        with writer.reader() as reader:
            assert tuple(reader.execute(
                "SELECT state,send_result FROM command_outbox o "
                "JOIN command_attempts a ON a.ground_instance_id=o.ground_instance_id "
                "AND a.request_id=o.request_id"
            ).fetchone()) == ("OUTBOX_PENDING", "PERSISTED_NOT_SENT")

        lease = outbox.claim_next()
        assert lease is not None
        retry = outbox.persist_attempt(lease, b"tc-2", apid=7)
        assert retry.packet_sequence != attempt.packet_sequence
        outbox.mark_sent(lease, retry)
        clock.advance(seconds=5, microseconds=1)
        report = outbox.reconcile()
        assert report.timed_out_sends == 1
        with writer.reader() as reader:
            state, available_at = reader.execute(
                "SELECT state,available_at_us FROM command_outbox"
            ).fetchone()
            assert state == "OUTBOX_PENDING"
            assert available_at > int(clock.value.timestamp() * 1_000_000)
        assert ledger.get(accepted.request_key).command_state == "ADMITTED"


def test_contact_modes_pause_next_contact_and_fail_immediate(tmp_path: Path):
    with stack(
        tmp_path,
        contact_state=ContactState.NO_CONTACT,
    ) as (writer, ledger, outbox, clock):
        with pytest.raises(NoContactError):
            admit(ledger, "immediate-no-contact", contact_available=False)
        held = admit(
            ledger,
            "next-contact",
            delivery_mode="next_contact",
            contact_available=False,
        )
        assert held.outbox_state == "HELD_NO_CONTACT"
        outbox.set_contact_state(1, ContactState.CONTACT_OPEN)
        lease = outbox.claim_next()
        assert lease is not None
        attempt = outbox.persist_attempt(lease, b"next-contact-tc")
        outbox.mark_sent(lease, attempt)
        outbox.set_contact_state(1, ContactState.NO_CONTACT)
        clock.advance(seconds=30)
        outbox.reconcile()
        with writer.reader() as reader:
            state, deadline = reader.execute(
                "SELECT state,ack_deadline_at_us FROM command_outbox "
                "WHERE ground_instance_id=? AND request_id=?",
                (bytes.fromhex(ledger.allocator.state().ground_instance_id.to_bytes(8, "big").hex()), held.request_key.request_id),
            ).fetchone()
            assert state == "SENT"
            assert deadline is None
        outbox.set_contact_state(1, ContactState.CONTACT_OPEN)
        with writer.reader() as reader:
            deadline = reader.execute(
                "SELECT ack_deadline_at_us FROM command_outbox "
                "WHERE ground_instance_id=? AND request_id=?",
                (held.request_key.ground_instance_id.to_bytes(8, "big"), held.request_key.request_id),
            ).fetchone()[0]
            assert deadline is not None

        immediate = admit(ledger, "immediate-open")
        lease = outbox.claim_next()
        assert lease is not None
        attempt = outbox.persist_attempt(lease, b"immediate-tc")
        outbox.set_contact_state(1, ContactState.NO_CONTACT)
        assert ledger.get(immediate.request_key).command_state == "FAILED"
        late = outbox.ingest_ack(immediate.request_key)
        assert late.late is True
        assert late.state == "DELIVERY_FAILED"
        with writer.reader() as reader:
            assert reader.execute(
                "SELECT count(*) FROM audit_log WHERE action='LATE_RECEIPT'"
            ).fetchone()[0] == 1


def test_tc_sequence_rollover_and_reset_are_durable(tmp_path: Path):
    with SQLiteWriter(tmp_path / "gds.sqlite3") as writer:
        allocator = TcSequenceAllocator(writer, clock_us=lambda: 1)
        first = allocator.allocate(1, 10)
        writer.mutate(
            "force_sequence_boundary",
            lambda connection: connection.execute(
                "UPDATE tc_sequence_allocators SET next_sequence=16384 "
                "WHERE spacecraft_instance_id=? AND apid=10",
                ((1).to_bytes(8, "big"),),
            ),
        )
        rollover = allocator.allocate(1, 10)
        assert first.sequence == 0
        assert rollover.sequence == 0
        assert rollover.sequence_epoch == 1
        assert rollover.rollover is True
        assert rollover.reset_marker == "SPACE_PACKET_SEQUENCE_ROLLOVER"
        assert allocator.state(1, 10).last_reset_reason == "SPACE_PACKET_SEQUENCE_ROLLOVER"
        reset = allocator.reset(1, 10, reason="spacecraft-reboot")
        assert reset.sequence_epoch == 2
        assert allocator.allocate(1, 10).sequence == 0


def test_api_status_mapping_and_same_key_replay(tmp_path: Path):
    with stack(tmp_path, outbox_capacity=1) as (writer, ledger, outbox, clock):
        api = GDSApi(ledger, outbox=outbox)
        body = {
            "target_spacecraft_instance_id": "0000000000000001",
            "opcode": int(CommandOpcode.SCENE_REQUEST_CATALOG),
            "payload": {},
        }
        accepted = api.post_command(body, headers={"Idempotency-Key": "api-key"})
        assert accepted.status_code == 202
        request_key = accepted.body["request_key"]
        assert api.get_command(
            request_key["ground_instance_id"], request_key["request_id"]
        ).status_code == 200
        assert api.get_command(request_key["ground_instance_id"], 99).status_code == 404
        conflict_body = dict(body, target_spacecraft_instance_id="0000000000000002")
        assert api.post_command(
            conflict_body, headers={"Idempotency-Key": "api-key"}
        ).status_code == 409
        assert api.post_command(body).status_code == 422

        full = api.post_command(
            dict(body, target_spacecraft_instance_id="0000000000000001"),
            headers={"Idempotency-Key": "second"},
        )
        assert full.status_code == 429
        outbox.set_contact_state(1, ContactState.NO_CONTACT)
        replay = api.post_command(body, headers={"idempotency-key": "api-key"})
        assert replay.status_code == 202
        outbox.set_contact_state(1, ContactState.CONTACT_OPEN)

        class FullGuard:
            def ensure_admission(self):
                raise StorageFullError("full")

        storage_api = GDSApi(ledger, outbox=outbox, storage_guard=FullGuard())
        assert storage_api.post_command(
            dict(body, target_spacecraft_instance_id="0000000000000001"),
            headers={"Idempotency-Key": "third"},
        ).status_code == 507

        class BusyGuard:
            def ensure_admission(self):
                raise WriterBackpressureError("busy")

        busy_api = GDSApi(ledger, outbox=outbox, storage_guard=BusyGuard())
        assert busy_api.post_command(
            dict(body, target_spacecraft_instance_id="0000000000000001"),
            headers={"Idempotency-Key": "fourth"},
        ).status_code == 503


def test_binding_migration_terminals_old_target_and_rejects_stale_fence(tmp_path: Path):
    with SQLiteWriter(tmp_path / "gds.sqlite3") as writer:
        clock = Clock()
        ledger = AtomicCommandLedger(writer, clock=clock)
        outbox = OutboxService(writer, clock=clock)
        manager = SpacecraftBindingManager(writer, outbox, clock=clock)
        manager.bind(1, link_generation=1, link_session_id=10)
        accepted = admit(ledger, "migrate")
        lease = outbox.claim_next()
        assert lease is not None
        before = writer.mutate(
            "read_semantic_before_migration",
            lambda connection: connection.execute(
                "SELECT semantic_body_jcs FROM commands WHERE ground_instance_id=? "
                "AND request_id=?",
                (accepted.request_key.ground_instance_id.to_bytes(8, "big"), accepted.request_key.request_id),
            ).fetchone()[0],
            transactional=False,
        )
        migration = manager.bind(2, link_generation=1, link_session_id=20)
        assert migration.retired_instance_id == 1
        assert migration.terminalized_commands == 1
        assert ledger.get(accepted.request_key).outbox_state == "DELIVERY_FAILED"
        with writer.reader() as reader:
            row = reader.execute(
                "SELECT last_error_code,lease_owner FROM command_outbox"
            ).fetchone()
            assert tuple(row) == ("TARGET_INSTANCE_RETIRED", None)
        after = writer.mutate(
            "read_semantic_after_migration",
            lambda connection: connection.execute(
                "SELECT semantic_body_jcs FROM commands WHERE ground_instance_id=? "
                "AND request_id=?",
                (accepted.request_key.ground_instance_id.to_bytes(8, "big"), accepted.request_key.request_id),
            ).fetchone()[0],
            transactional=False,
        )
        assert before == after
        with pytest.raises(BindingChangedError):
            with manager.read_fence(lease.binding):
                pass
        api = GDSApi(ledger, outbox=outbox)
        old_body = {
            "target_spacecraft_instance_id": "0000000000000001",
            "opcode": int(CommandOpcode.SCENE_REQUEST_CATALOG),
            "payload": {},
        }
        assert api.post_command(
            old_body, headers={"Idempotency-Key": "migrate"}
        ).status_code == 202
        new_response = api.post_command(
            dict(old_body, target_spacecraft_instance_id="0000000000000002"),
            headers={"Idempotency-Key": "migrate-b"},
        )
        assert new_response.status_code == 202
        assert new_response.body["request_key"] != accepted.request_key.as_dict()


def test_event_cursor_telemetry_dedupe_rollup_and_audit(tmp_path: Path):
    with SQLiteWriter(tmp_path / "gds.sqlite3") as writer:
        events = EventStore(writer, clock=lambda: datetime(2026, 7, 19, 12, tzinfo=UTC))
        first = events.append("ONE", message={"n": 1})
        second = events.append("TWO", message={"n": 2})
        records, cursor = events.list_events(limit=2)
        assert [item.event_id for item in records] == [first.event_id, second.event_id]
        later, _ = events.list_events(after_event_id=cursor)
        assert later == ()

        telemetry = TelemetryStore(writer)
        sample = TelemetrySample(
            1, 1, 99, "DOWNLINK", 2, 3, 0, 0, 10, 20,
            1_000_000, b"raw-1", 2.0,
        )
        assert telemetry.ingest(sample).inserted
        assert telemetry.ingest(sample).duplicate
        telemetry.ingest(
            TelemetrySample(
                1, 1, 99, "DOWNLINK", 2, 4, 0, 0, 10, 20,
                1_500_000, b"raw-2", 4.0,
            )
        )
        with pytest.raises(TelemetryConflictError):
            telemetry.ingest(
                TelemetrySample(
                    1, 1, 99, "DOWNLINK", 2, 3, 0, 0, 10, 20,
                    1_000_000, b"different", 3.0,
                )
            )
        with writer.reader() as reader:
            count, mean = reader.execute(
                "SELECT sample_count,mean_value FROM telemetry_rollups"
            ).fetchone()
            assert (count, mean) == (2, 3.0)
            assert reader.execute(
                "SELECT count(*) FROM audit_log WHERE action='TELEMETRY_DEDUPE_CONFLICT'"
            ).fetchone()[0] == 1


def test_raw_segment_fsync_recovery_prune_and_storage_reservation(tmp_path: Path):
    raw = RawSegmentStore(tmp_path / "frames.seg")
    first = raw.append(b"frame-1")
    raw.append(b"frame-2")
    assert len(raw.scan(committed_offset=first.offset + first.length)) == 1
    assert len(raw.scan(repair=True, committed_offset=first.offset + first.length)) == 1
    raw.append(b"frame-2")
    with raw.path.open("ab") as stream:
        stream.write(b"torn")
    assert len(raw.scan(repair=True)) == 2
    assert raw.read(first) == b"frame-1"
    order: list[str] = []
    raw.prune_after_db(lambda: order.append("db"), remove_file=True)
    assert order == ["db"]
    assert not raw.path.exists()

    with SQLiteWriter(tmp_path / "storage.sqlite3") as writer:
        guard = StorageGuard(
            writer,
            tmp_path,
            cap_bytes=1_000,
            usage_provider=lambda: 500,
        )
        reservation = guard.reserve("test", 200, ttl=timedelta(minutes=1))
        assert guard.snapshot().reserved_bytes == 200
        guard.release(reservation.reservation_id)
        assert guard.snapshot().reserved_bytes == 0
        full = StorageGuard(
            writer,
            tmp_path,
            cap_bytes=1_000,
            usage_provider=lambda: 950,
        )
        with pytest.raises(StorageFullError):
            full.ensure_admission()
