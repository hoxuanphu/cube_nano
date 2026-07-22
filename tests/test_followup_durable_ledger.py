"""Regression coverage for F-03 through F-05 durable transport remediation."""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from flight.file_downlink import FileDownlinkCoordinator
from flight.journal import SatelliteJournal
from flight.stock_router import StockApidRouter
from gds.binding import SpacecraftBindingManager
from gds.ingest import TmIngestService
from gds.ledger import AtomicCommandLedger
from gds.outbox import ContactState, OutboxService, TcWireProfile
from gds.schema import SCHEMA_VERSION
from gds.tm import TMDecoder, TmCounterStatus, ValidatedTransportEnvelope
from gds.writer import SQLiteWriter
from protocol.canonical import canonical_json
from protocol.ccsds import TcTypeBdFrame, decode_tm_frame, encode_space_packet, encode_tm_frame
from protocol.file_packet import FilePacket, FilePacketType, encode_file_packet
from protocol.messages import PacketDescriptor, encode_tm_application
from protocol.profile import MissionProfile
from protocol.schemas import CommandOpcode, ProductRef, RequestKey


class Clock:
    def __init__(self) -> None:
        self.value = datetime(2026, 7, 21, 12, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.value

    def advance(self, **kwargs: int) -> None:
        self.value += timedelta(**kwargs)


def _admit(ledger: AtomicCommandLedger, key: str):
    return ledger.admit(
        idempotency_key=key,
        target_spacecraft_instance_id=1,
        opcode=CommandOpcode.SCENE_REQUEST_CATALOG,
        payload={},
    )


def _tm_envelope(
    frame: bytes,
    *,
    frame_id: int,
    generation: int = 1,
    session: int = 1,
    file_epoch: int = 0,
    copy_index: int = 0,
) -> ValidatedTransportEnvelope:
    return ValidatedTransportEnvelope(
        1,
        1,
        session,
        generation,
        77,
        frame_id,
        frame_id,
        file_epoch,
        copy_index,
        frame_id,
        "DOWNLINK",
        frame,
    )


def _event_frame(message: dict, *, sequence: int, counter: int) -> bytes:
    packet = encode_tm_application(
        2,
        PacketDescriptor.EVENT_ACK,
        message,
        sequence,
    )
    return encode_tm_frame(
        packet,
        master_channel_count=counter,
        virtual_channel_count=counter,
    )


def _telemetry_frame(*, sequence: int, counter: int) -> bytes:
    packet = encode_tm_application(
        1,
        PacketDescriptor.TELEMETRY,
        {"channel_id": 1, "value": counter},
        sequence,
    )
    return encode_tm_frame(
        packet,
        master_channel_count=counter,
        virtual_channel_count=counter,
    )


def _file_frame(packet: FilePacket, *, sequence: int, counter: int) -> bytes:
    return encode_tm_frame(
        encode_space_packet(3, encode_file_packet(packet), sequence),
        master_channel_count=counter,
        virtual_channel_count=counter,
    )


def test_schema_008_registers_durable_transport_tables(tmp_path: Path):
    with SQLiteWriter(tmp_path / "gds.sqlite3") as writer:
        version, name = writer.mutate(
            "read_durable_migration",
            lambda connection: connection.execute(
                "SELECT version,name FROM schema_migrations WHERE version=?",
                (8,),
            ).fetchone(),
            transactional=False,
        )
        assert (version, name) == (8, "product_downlink_transfer_identity")
        assert SCHEMA_VERSION >= 8
        tables = writer.mutate(
            "read_durable_tables",
            lambda connection: {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_schema WHERE type='table'"
                )
            },
            transactional=False,
        )
        assert {
            "product_downlink_ledger",
            "product_downlink_pending_files",
            "tm_source_generations",
            "tm_channel_counter_states",
            "tm_packet_counter_states",
            "tm_counter_states",
            "tm_counter_observations",
        } <= tables


def test_keyed_claim_and_prepared_tc_match_persisted_profile_bytes(tmp_path: Path):
    clock = Clock()
    with SQLiteWriter(tmp_path / "gds.sqlite3") as writer:
        ledger = AtomicCommandLedger(writer, clock=clock)
        outbox = OutboxService(writer, clock=clock)
        outbox.register_instance(
            1,
            link_generation=1,
            link_session_id=1,
            contact_state=ContactState.CONTACT_OPEN,
        )
        first = _admit(ledger, "first")
        second = _admit(ledger, "second")

        with ThreadPoolExecutor(max_workers=8) as pool:
            leases = list(
                pool.map(
                    lambda index: outbox.claim(
                        second.request_key,
                        lease_owner=f"dispatcher-{index}",
                    ),
                    range(8),
                )
            )
        claimed = [lease for lease in leases if lease is not None]
        assert len(claimed) == 1
        assert claimed[0].request_key == second.request_key
        assert ledger.get(first.request_key).outbox_state == "OUTBOX_PENDING"

        clock.advance(seconds=11)
        assert outbox.reconcile().recovered_leases == 1
        lease = outbox.claim(first.request_key)
        assert lease is not None
        profile = TcWireProfile.from_mission_profile(
            MissionProfile.from_file("protocol/mission_profile.yaml")
        )
        attempt = outbox.prepare_attempt(lease, profile=profile)
        decoded = TcTypeBdFrame.decode(attempt.encoded_tc)
        assert decoded.packet.packet_type == 1
        assert decoded.packet.sequence_count == attempt.packet_sequence
        assert decoded.sequence_number == attempt.frame_sequence
        assert StockApidRouter().route_tc(decoded.packet.encode()).accepted
        assert hashlib.sha256(attempt.encoded_tc).hexdigest() == attempt.encoded_tc_sha256

        with writer.reader() as reader:
            stored = reader.execute(
                "SELECT packet_sequence,frame_sequence,encoded_tc,encoded_tc_sha256,"
                "tc_profile_id,tc_profile_sha256,space_packet_type,"
                "space_packet_sequence_flags FROM command_attempts WHERE attempt_id=?",
                (attempt.attempt_id,),
            ).fetchone()
        assert int(stored["packet_sequence"]) == attempt.packet_sequence
        assert int(stored["frame_sequence"]) == attempt.frame_sequence
        assert bytes(stored["encoded_tc"]) == attempt.encoded_tc
        assert bytes(stored["encoded_tc_sha256"]).hex() == attempt.encoded_tc_sha256
        assert (stored["tc_profile_id"], stored["tc_profile_sha256"]) == (
            profile.profile_id,
            profile.profile_sha256,
        )
        assert (stored["space_packet_type"], stored["space_packet_sequence_flags"]) == (1, 3)


def test_only_correlated_apid2_tm_advances_the_outbox(tmp_path: Path):
    clock = Clock()
    with SQLiteWriter(tmp_path / "gds.sqlite3") as writer:
        ledger = AtomicCommandLedger(writer, clock=clock)
        outbox = OutboxService(writer, clock=clock)
        outbox.register_instance(
            1,
            link_generation=1,
            link_session_id=1,
            contact_state=ContactState.CONTACT_OPEN,
        )
        profile = TcWireProfile.from_mission_profile(
            MissionProfile.from_file("protocol/mission_profile.yaml")
        )
        accepted = _admit(ledger, "correlated")
        lease = outbox.claim(accepted.request_key)
        assert lease is not None
        attempt = outbox.prepare_attempt(lease, profile=profile)
        outbox.mark_sent(lease, attempt)
        service = TmIngestService(writer, TMDecoder(), outbox=outbox, ledger=ledger)
        result = service.ingest(
            _tm_envelope(
                _event_frame(
                    {"request_key": accepted.request_key.as_dict(), "stage": "EXECUTED"},
                    sequence=0,
                    counter=0,
                ),
                frame_id=1,
            )
        )
        assert result.ack is not None and result.ack.state == "ACKED"
        assert ledger.get(accepted.request_key).outbox_state == "ACKED"

        unsent = _admit(ledger, "not-sent")
        unsent_lease = outbox.claim(unsent.request_key)
        assert unsent_lease is not None
        outbox.prepare_attempt(unsent_lease, profile=profile)
        uncorrelated = service.ingest(
            _tm_envelope(
                _event_frame(
                    {"request_key": unsent.request_key.as_dict(), "stage": "EXECUTED"},
                    sequence=1,
                    counter=1,
                ),
                frame_id=2,
            )
        )
        assert uncorrelated.ack is not None
        assert uncorrelated.ack.reason == "UNCORRELATED_TM_ACK"
        assert ledger.get(unsent.request_key).outbox_state == "DISPATCHING"


def test_send_fence_arms_tm_correlation_before_udp_send_and_recovers_send_error(tmp_path: Path):
    clock = Clock()
    profile = TcWireProfile.from_mission_profile(
        MissionProfile.from_file("protocol/mission_profile.yaml")
    )
    with SQLiteWriter(tmp_path / "gds.sqlite3") as writer:
        ledger = AtomicCommandLedger(writer, clock=clock)
        outbox = OutboxService(writer, clock=clock)
        bindings = SpacecraftBindingManager(writer, outbox, clock=clock)
        bindings.bind(1, link_generation=1, link_session_id=1)

        accepted = _admit(ledger, "inline-udp-ack")
        lease = outbox.claim(accepted.request_key, binding=bindings.active_binding())
        assert lease is not None
        attempt = outbox.prepare_attempt(lease, profile=profile)

        def ack_during_send(_: bytes) -> str:
            ack = outbox.ingest_correlated_tm(
                accepted.request_key,
                source_spacecraft_instance_id=1,
                link_generation=1,
                link_session_id=1,
                success=True,
            )
            assert ack.state == "ACKED"
            return "UDP_SENT"

        assert outbox.send_with_fence(
            lease,
            attempt,
            fence=bindings.fence,
            send=ack_during_send,
        ) == "UDP_SENT"
        assert ledger.get(accepted.request_key).outbox_state == "ACKED"

        failed = _admit(ledger, "udp-send-error")
        failed_lease = outbox.claim(failed.request_key, binding=bindings.active_binding())
        assert failed_lease is not None
        failed_attempt = outbox.prepare_attempt(failed_lease, profile=profile)
        with pytest.raises(OSError):
            outbox.send_with_fence(
                failed_lease,
                failed_attempt,
                fence=bindings.fence,
                send=lambda _: (_ for _ in ()).throw(OSError("udp unavailable")),
            )
        assert ledger.get(failed.request_key).outbox_state == "OUTBOX_PENDING"
        with writer.reader() as reader:
            row = reader.execute(
                "SELECT send_result FROM command_attempts WHERE attempt_id=?",
                (failed_attempt.attempt_id,),
            ).fetchone()
        assert row[0] == "NOT_SENT:SEND_ERROR"


def test_same_instance_rebind_requeues_sent_command_with_new_session_correlation(tmp_path: Path):
    clock = Clock()
    profile = TcWireProfile.from_mission_profile(
        MissionProfile.from_file("protocol/mission_profile.yaml")
    )
    with SQLiteWriter(tmp_path / "gds.sqlite3") as writer:
        ledger = AtomicCommandLedger(writer, clock=clock)
        outbox = OutboxService(writer, clock=clock)
        bindings = SpacecraftBindingManager(writer, outbox, clock=clock)
        bindings.bind(1, link_generation=1, link_session_id=1)
        accepted = _admit(ledger, "same-instance-rebind")
        first_lease = outbox.claim(accepted.request_key, binding=bindings.active_binding())
        assert first_lease is not None
        first_attempt = outbox.prepare_attempt(first_lease, profile=profile)
        outbox.mark_sent(first_lease, first_attempt)

        migration = bindings.bind(1, link_generation=2, link_session_id=2)
        assert migration.retired_instance_id is None
        assert migration.terminalized_commands == 0
        assert ledger.get(accepted.request_key).outbox_state == "OUTBOX_PENDING"
        with writer.reader() as reader:
            old_attempt = reader.execute(
                "SELECT send_result FROM command_attempts WHERE attempt_id=?",
                (first_attempt.attempt_id,),
            ).fetchone()
        assert old_attempt[0] == "NOT_SENT:REBIND"

        rebound = bindings.active_binding()
        assert rebound is not None
        retry_lease = outbox.claim(accepted.request_key, binding=rebound)
        assert retry_lease is not None
        retry_attempt = outbox.prepare_attempt(retry_lease, profile=profile)
        outbox.mark_sent(retry_lease, retry_attempt)
        ack = outbox.ingest_correlated_tm(
            accepted.request_key,
            source_spacecraft_instance_id=1,
            link_generation=2,
            link_session_id=2,
            success=True,
        )
        assert ack.state == "ACKED"
        assert ledger.get(accepted.request_key).outbox_state == "ACKED"


def test_outbox_startup_recovery_bounds_crashes_before_and_after_send(tmp_path: Path):
    path = tmp_path / "gds.sqlite3"
    clock = Clock()
    profile = TcWireProfile.from_mission_profile(
        MissionProfile.from_file("protocol/mission_profile.yaml")
    )
    with SQLiteWriter(path) as writer:
        ledger = AtomicCommandLedger(writer, clock=clock)
        outbox = OutboxService(writer, clock=clock)
        outbox.register_instance(
            1,
            link_generation=1,
            link_session_id=1,
            contact_state=ContactState.CONTACT_OPEN,
        )
        admitted = _admit(ledger, "restart")
        lease = outbox.claim(admitted.request_key)
        assert lease is not None
        outbox.prepare_attempt(lease, profile=profile)
        # Simulate process death after attempt persistence but before send.

    clock.advance(seconds=11)
    with SQLiteWriter(path) as writer:
        ledger = AtomicCommandLedger(writer, clock=clock)
        outbox = OutboxService(writer, clock=clock)
        assert ledger.get(admitted.request_key).outbox_state == "OUTBOX_PENDING"
        lease = outbox.claim(admitted.request_key)
        assert lease is not None
        attempt = outbox.prepare_attempt(lease, profile=profile)
        outbox.mark_sent(lease, attempt)
        # Simulate a crash after transport send but before correlated APID 2 TM.

    clock.advance(seconds=6)
    with SQLiteWriter(path) as writer:
        ledger = AtomicCommandLedger(writer, clock=clock)
        OutboxService(writer, clock=clock)
        assert ledger.get(admitted.request_key).outbox_state == "OUTBOX_PENDING"


def test_tm_counter_ledger_survives_restart_rollover_gap_duplicate_and_stale_generation(tmp_path: Path):
    path = tmp_path / "gds.sqlite3"
    frames = [
        _telemetry_frame(sequence=16_382, counter=254),
        _telemetry_frame(sequence=16_383, counter=255),
        _telemetry_frame(sequence=0, counter=0),
        _telemetry_frame(sequence=1, counter=1),
    ]
    with SQLiteWriter(path) as writer:
        service = TmIngestService(writer, TMDecoder())
        statuses = [
            service.ingest(_tm_envelope(frame, frame_id=index + 1)).counter.status
            for index, frame in enumerate(frames)
        ]
        assert statuses == [
            TmCounterStatus.BASELINE,
            TmCounterStatus.IN_ORDER,
            TmCounterStatus.ROLLOVER,
            TmCounterStatus.IN_ORDER,
        ]

    with SQLiteWriter(path) as writer:
        service = TmIngestService(writer, TMDecoder())
        continued = service.ingest(
            _tm_envelope(_telemetry_frame(sequence=2, counter=2), frame_id=5)
        )
        assert continued.counter is not None
        assert continued.counter.status is TmCounterStatus.IN_ORDER
        duplicate = service.ingest(
            _tm_envelope(_telemetry_frame(sequence=2, counter=2), frame_id=6)
        )
        assert duplicate.counter is not None
        assert duplicate.counter.status is TmCounterStatus.DUPLICATE
        gap = service.ingest(
            _tm_envelope(_telemetry_frame(sequence=4, counter=4), frame_id=7)
        )
        assert gap.counter is not None
        assert gap.counter.status is TmCounterStatus.GAP
        assert gap.counter.packet_gap == 1
        stale = service.ingest(
            _tm_envelope(
                _telemetry_frame(sequence=5, counter=5),
                frame_id=8,
                generation=0,
            )
        )
        assert stale.counter is not None
        assert stale.counter.status is TmCounterStatus.STALE_GENERATION


def test_tm_channel_counters_span_apid2_and_apid3_file_epochs(tmp_path: Path):
    with SQLiteWriter(tmp_path / "gds.sqlite3") as writer:
        service = TmIngestService(writer, TMDecoder())
        event = _event_frame({"event_name": "COMMAND_ACCEPTED", "stage": "DISPATCHED"}, sequence=10, counter=40)
        first = service.ingest(_tm_envelope(event, frame_id=1))
        assert first.counter is not None
        assert first.counter.status is TmCounterStatus.BASELINE

        product = ProductRef(1, 1, 9)
        start_one = FilePacket(
            FilePacketType.START,
            0,
            0,
            canonical_json(
                {
                    "product_ref": product.as_dict(),
                    "transfer_id": 10,
                    "file_size": 1,
                    "checksum": 1,
                }
            ),
        )
        second = service.ingest(
            _tm_envelope(
                _file_frame(start_one, sequence=100, counter=41),
                frame_id=2,
                file_epoch=10,
            )
        )
        assert second.counter is not None
        assert second.counter.status is TmCounterStatus.IN_ORDER

        start_two = FilePacket(
            FilePacketType.START,
            0,
            0,
            canonical_json(
                {
                    "product_ref": product.as_dict(),
                    "transfer_id": 11,
                    "file_size": 1,
                    "checksum": 1,
                }
            ),
        )
        third = service.ingest(
            _tm_envelope(
                _file_frame(start_two, sequence=101, counter=42),
                frame_id=3,
                file_epoch=11,
            )
        )
        assert third.counter is not None
        assert third.counter.status is TmCounterStatus.IN_ORDER

        with writer.reader() as reader:
            channel = reader.execute(
                "SELECT last_master_channel_count,last_virtual_channel_count "
                "FROM tm_channel_counter_states"
            ).fetchone()
            packet = reader.execute(
                "SELECT last_packet_sequence FROM tm_packet_counter_states WHERE apid=3"
            ).fetchone()
        assert tuple(channel) == (42, 42)
        assert int(packet[0]) == 101


def test_downlink_gets_fresh_request_key_and_links_apid3_transfer_state(tmp_path: Path):
    clock = Clock()
    with SQLiteWriter(tmp_path / "gds.sqlite3") as writer:
        ledger = AtomicCommandLedger(writer, clock=clock)
        origin = RequestKey(7, 0x80000000)
        product = ProductRef(1, 1, 9)
        first = ledger.admit_product_downlink(
            origin_request_key=origin,
            product_ref=product,
            target_spacecraft_instance_id=1,
            contact_available=True,
        )
        duplicate = ledger.admit_product_downlink(
            origin_request_key=origin,
            product_ref=product,
            target_spacecraft_instance_id=1,
            contact_available=True,
        )
        assert first.request_key != origin
        assert duplicate.request_key == first.request_key
        assert duplicate.replayed

        writer.mutate(
            "terminalize_first_downlink",
            lambda connection: (
                connection.execute(
                    "UPDATE command_outbox SET state='DELIVERY_FAILED' WHERE "
                    "ground_instance_id=? AND request_id=?",
                    (
                        first.request_key.ground_instance_id.to_bytes(8, "big"),
                        first.request_key.request_id,
                    ),
                ),
                connection.execute(
                    "UPDATE commands SET command_state='FAILED' WHERE "
                    "ground_instance_id=? AND request_id=?",
                    (
                        first.request_key.ground_instance_id.to_bytes(8, "big"),
                        first.request_key.request_id,
                    ),
                ),
            ),
        )
        retry = ledger.admit_product_downlink(
            origin_request_key=origin,
            product_ref=product,
            target_spacecraft_instance_id=1,
            contact_available=True,
            retry=True,
        )
        assert retry.request_key != first.request_key
        assert ledger.update_product_downlink_transfer(
            retry.request_key,
            transfer_id=44,
            transfer_state="DISPATCHED",
        )
        assert ledger.update_product_downlink_file_state(
            product,
            transfer_id=44,
            transfer_state="VERIFIED",
        ) == retry.request_key
        assert ledger.update_product_downlink_transfer(
            retry.request_key,
            transfer_id=44,
            transfer_state="DISPATCHED",
        )
        with writer.reader() as reader:
            rows = reader.execute(
                "SELECT admission_ordinal,transfer_id,transfer_state FROM "
                "product_downlink_ledger ORDER BY admission_ordinal"
            ).fetchall()
        assert [tuple(row) for row in rows] == [
            (1, None, "ADMITTED"),
            (2, 44, "VERIFIED"),
        ]


def test_apid3_transfer_identity_never_claims_an_unassigned_retry(tmp_path: Path):
    clock = Clock()
    with SQLiteWriter(tmp_path / "gds.sqlite3") as writer:
        ledger = AtomicCommandLedger(writer, clock=clock)
        origin = RequestKey(7, 0x80000000)
        product = ProductRef(1, 1, 10)
        first = ledger.admit_product_downlink(
            origin_request_key=origin,
            product_ref=product,
            target_spacecraft_instance_id=1,
            contact_available=True,
        )
        assert ledger.update_product_downlink_transfer(
            first.request_key,
            transfer_id=40,
            transfer_state="DISPATCHED",
        )
        writer.mutate(
            "terminalize_first_transfer_for_retry",
            lambda connection: connection.execute(
                "UPDATE command_outbox SET state='DELIVERY_FAILED' "
                "WHERE ground_instance_id=? AND request_id=?",
                (
                    first.request_key.ground_instance_id.to_bytes(8, "big"),
                    first.request_key.request_id,
                ),
            ),
        )
        retry = ledger.admit_product_downlink(
            origin_request_key=origin,
            product_ref=product,
            target_spacecraft_instance_id=1,
            contact_available=True,
            retry=True,
        )

        # A delayed old transfer is attached only to its original admission.
        assert ledger.update_product_downlink_file_state(
            product,
            transfer_id=40,
            transfer_state="VERIFIED",
        ) == first.request_key

        # New APID 3 data before APID 2 is durably deferred, not assigned to
        # the retry's NULL transfer_id.  The APID 2 receipt later resolves it.
        assert ledger.update_product_downlink_file_state(
            product,
            transfer_id=41,
            transfer_state="VERIFIED",
        ) is None
        with writer.reader() as reader:
            retry_before = reader.execute(
                "SELECT transfer_id,transfer_state FROM product_downlink_ledger "
                "WHERE downlink_ground_instance_id=? AND downlink_request_id=?",
                (
                    retry.request_key.ground_instance_id.to_bytes(8, "big"),
                    retry.request_key.request_id,
                ),
            ).fetchone()
        assert tuple(retry_before) == (None, "ADMITTED")

        assert ledger.update_product_downlink_transfer(
            retry.request_key,
            transfer_id=41,
            transfer_state="DISPATCHED",
        )
        with writer.reader() as reader:
            retry_after = reader.execute(
                "SELECT transfer_id,transfer_state FROM product_downlink_ledger "
                "WHERE downlink_ground_instance_id=? AND downlink_request_id=?",
                (
                    retry.request_key.ground_instance_id.to_bytes(8, "big"),
                    retry.request_key.request_id,
                ),
            ).fetchone()
        assert tuple(retry_after) == (41, "VERIFIED")

        assert ledger.update_product_downlink_file_state(
            product,
            transfer_id=99,
            transfer_state="RECEIVING",
        ) is None
        clock.advance(days=2)
        assert ledger.reconcile_pending_product_downlink_files() == 1


def test_satellite_tm_and_apid3_counters_are_durable(tmp_path: Path):
    path = tmp_path / "satellite.sqlite3"
    journal = SatelliteJournal(path, 1)
    first = journal.allocate_tm_frame_counters(3)
    assert (first.packet_sequence, first.master_channel_count, first.virtual_channel_count) == (0, 0, 0)
    journal.close()

    journal = SatelliteJournal(path, 1)
    second = journal.allocate_tm_frame_counters(3)
    assert (second.packet_sequence, second.master_channel_count, second.virtual_channel_count) == (1, 1, 1)
    with journal.transaction() as connection:
        connection.execute("UPDATE tm_packet_counters SET next_sequence=16384 WHERE apid=3")
        connection.execute("UPDATE tm_master_counter SET next_count=256 WHERE singleton=1")
        connection.execute("UPDATE tm_virtual_counters SET next_count=256 WHERE virtual_channel_id=0")
    rollover = journal.allocate_tm_frame_counters(3)
    assert (rollover.packet_sequence, rollover.master_channel_count, rollover.virtual_channel_count) == (0, 0, 0)
    assert (
        rollover.packet_sequence_epoch,
        rollover.master_channel_epoch,
        rollover.virtual_channel_epoch,
    ) == (1, 1, 1)

    bundle = tmp_path / "bundle.tar"
    bundle.write_bytes(b"payload")
    coordinator = FileDownlinkCoordinator(
        tm_counter_allocator=journal.allocate_tm_frame_counters,
    )
    coordinator.start(5, ProductRef(1, 1, 1), bundle)
    lease = coordinator.next_frame(5)
    assert lease is not None
    decoded = decode_tm_frame(lease.frame)
    assert decoded.packet.apid == 3
    assert decoded.packet.sequence_count == 1
    assert decoded.master_channel_count == 1
    assert decoded.virtual_channel_count == 1
    journal.close()
