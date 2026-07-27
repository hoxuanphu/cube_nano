"""Tests for Link Simulator Phase 3 components.

Test coverage for transport, virtual clock, fault model, and link simulator.
"""

import pytest
from link_sim.transport import (
    Direction,
    InMemoryTransport,
    SidebandEnvelope,
    TransportFrame,
    UdpTransport,
)
from link_sim.virtual_clock import SimulationTime, VirtualClock
from link_sim.fault_model import FaultModel, FaultProfile, FaultStage
from link_sim.link_simulator import LinkSimulator


class TestSidebandEnvelope:
    """Test sideband envelope serialization and validation."""

    def test_ingress_validation_success(self):
        """Test valid ingress envelope."""
        envelope = SidebandEnvelope(
            magic=SidebandEnvelope.MAGIC,
            version=1,
            direction=Direction.INGRESS,
            reserved=0,
            spacecraft_instance_id=0x1234567890ABCDEF,
            sender_boot_id=0,
            link_session_id=0xFEDCBA0987654321,
            sender_frame_id=42,
            link_frame_id=0,
            file_epoch_id=0,
            frame_length=1024,
        )
        envelope.validate_ingress()  # Should not raise

    def test_ingress_validation_nonzero_boot_fails(self):
        """Test ingress with nonzero sender_boot_id fails."""
        envelope = SidebandEnvelope(
            magic=SidebandEnvelope.MAGIC,
            version=1,
            direction=Direction.INGRESS,
            reserved=0,
            spacecraft_instance_id=0x1234567890ABCDEF,
            sender_boot_id=1,  # Must be 0
            link_session_id=0xFEDCBA0987654321,
            sender_frame_id=42,
            link_frame_id=0,
            file_epoch_id=0,
            frame_length=1024,
        )
        with pytest.raises(ValueError, match="sender_boot_id must be 0"):
            envelope.validate_ingress()

    def test_egress_validation_success(self):
        """Test valid egress envelope."""
        envelope = SidebandEnvelope(
            magic=SidebandEnvelope.MAGIC,
            version=1,
            direction=Direction.EGRESS,
            reserved=0,
            spacecraft_instance_id=0x1234567890ABCDEF,
            sender_boot_id=5,
            link_session_id=0xFEDCBA0987654321,
            sender_frame_id=42,
            link_frame_id=100,
            file_epoch_id=1,
            frame_length=1024,
        )
        envelope.validate_egress()  # Should not raise

    def test_egress_validation_zero_link_frame_fails(self):
        """Test egress with link_frame_id=0 fails."""
        envelope = SidebandEnvelope(
            magic=SidebandEnvelope.MAGIC,
            version=1,
            direction=Direction.EGRESS,
            reserved=0,
            spacecraft_instance_id=0x1234567890ABCDEF,
            sender_boot_id=5,
            link_session_id=0xFEDCBA0987654321,
            sender_frame_id=42,
            link_frame_id=0,  # Must be > 0
            file_epoch_id=1,
            frame_length=1024,
        )
        with pytest.raises(ValueError, match="link_frame_id must be > 0"):
            envelope.validate_egress()

    def test_serialization_roundtrip(self):
        """Test envelope serialization round-trip."""
        original = SidebandEnvelope(
            magic=SidebandEnvelope.MAGIC,
            version=1,
            direction=Direction.INGRESS,
            reserved=0,
            spacecraft_instance_id=0x1234567890ABCDEF,
            sender_boot_id=0,
            link_session_id=0xFEDCBA0987654321,
            sender_frame_id=42,
            link_frame_id=0,
            file_epoch_id=0,
            frame_length=1024,
        )
        serialized = original.to_bytes()
        deserialized = SidebandEnvelope.from_bytes(serialized)
        assert deserialized == original

    def test_udp_roundtrip_preserves_duplicate_copy_index(self):
        """UDP version 2 preserves fault-copy identity rather than defaulting to zero."""

        receiver = UdpTransport(("127.0.0.1", 0), expected_direction=Direction.EGRESS)
        sender = UdpTransport(("127.0.0.1", 0), receiver.bound_address)
        try:
            legacy_envelope = SidebandEnvelope(
                magic=SidebandEnvelope.MAGIC,
                version=SidebandEnvelope.VERSION,
                direction=Direction.EGRESS,
                reserved=0,
                spacecraft_instance_id=1,
                sender_boot_id=7,
                link_session_id=9,
                sender_frame_id=11,
                link_frame_id=13,
                file_epoch_id=0,
                frame_length=4,
            )
            sender.send_transport_frame(TransportFrame(legacy_envelope, b"copy", copy_index=1))
            received = receiver.receive(timeout_ms=1_000)
            assert received is not None
            assert received.copy_index == 1
            assert received.envelope.copy_index == 1
            assert received.envelope.version == SidebandEnvelope.VERSION_WITH_COPY_INDEX
            assert received.frame_bytes == b"copy"
        finally:
            sender.close()
            receiver.close()


class TestVirtualClock:
    """Test virtual clock and event scheduling."""

    def test_monotonic_time(self):
        """Test clock advances monotonically."""
        clock = VirtualClock()
        assert clock.now == SimulationTime(0)

        clock.schedule(100, lambda: None, direction=0, link_frame_id=1)
        clock.advance_to_next_event()
        assert clock.now == SimulationTime(100)

    def test_event_ordering_tie_breaker(self):
        """Test events with same due time are ordered by tie-breaker."""
        clock = VirtualClock()
        results = []

        # Schedule 3 events at same time with different tie-breakers
        clock.schedule(100, lambda: results.append("frame_3"), direction=0, link_frame_id=3)
        clock.schedule(100, lambda: results.append("frame_1"), direction=0, link_frame_id=1)
        clock.schedule(100, lambda: results.append("frame_2"), direction=0, link_frame_id=2)

        clock.run_until_idle()

        # Should be ordered by link_frame_id (tie-breaker)
        assert results == ["frame_1", "frame_2", "frame_3"]

    def test_run_until_time(self):
        """Test run_until advances to specified time."""
        clock = VirtualClock()
        executed = []

        clock.schedule(50, lambda: executed.append(1), direction=0, link_frame_id=1)
        clock.schedule(150, lambda: executed.append(2), direction=0, link_frame_id=2)
        clock.schedule(250, lambda: executed.append(3), direction=0, link_frame_id=3)

        count = clock.run_until(SimulationTime(200))
        assert count == 2
        assert executed == [1, 2]
        assert clock.now == SimulationTime(200)

    def test_admission_order_counter(self):
        """Test admission order counter is monotonic."""
        clock = VirtualClock()
        orders = [clock.get_admission_order() for _ in range(5)]
        assert orders == [0, 1, 2, 3, 4]


class TestFaultModel:
    """Test deterministic fault injection."""

    def test_loss_deterministic(self):
        """Test loss decision is deterministic for same inputs."""
        model = FaultModel(seed=0x1111111111111111, simulation_run_id=0x2222222222222222)
        profile = FaultProfile(frame_loss_rate_ppm=500_000)  # 50%

        decision1 = model.apply_faults(
            profile=profile,
            direction=0,
            link_frame_id=100,
            frame_bits=8192,
            ingress_time_ns=1000,
            link_available_ns=0,
        )

        decision2 = model.apply_faults(
            profile=profile,
            direction=0,
            link_frame_id=100,
            frame_bits=8192,
            ingress_time_ns=1000,
            link_available_ns=0,
        )

        assert decision1.is_lost == decision2.is_lost

    def test_loss_rate_extremes(self):
        """Test loss rate 0 and 1_000_000."""
        model = FaultModel(seed=0xAAAAAAAAAAAAAAAA, simulation_run_id=0xBBBBBBBBBBBBBBBB)

        # Rate 0: never lost
        profile_0 = FaultProfile(frame_loss_rate_ppm=0)
        decision_0 = model.apply_faults(profile_0, 0, 1, 8192, 1000, 0)
        assert decision_0.is_lost is False

        # Rate 1_000_000: always lost
        profile_max = FaultProfile(frame_loss_rate_ppm=1_000_000)
        decision_max = model.apply_faults(profile_max, 0, 1, 8192, 1000, 0)
        assert decision_max.is_lost is True

    def test_corruption_bit_selection(self):
        """Test corruption selects unique bit offsets."""
        model = FaultModel(seed=0x3333333333333333, simulation_run_id=0x4444444444444444)
        profile = FaultProfile(corrupt_frame_rate_ppm=1_000_000, bits_per_corrupt_frame=5)

        decision = model.apply_faults(profile, 0, 1, 8192, 1000, 0)

        assert decision.is_corrupted is True
        assert len(decision.corrupted_bits) == 5
        assert len(set(decision.corrupted_bits)) == 5  # All unique

    def test_jitter_symmetric(self):
        """Test jitter is symmetric around base latency."""
        model = FaultModel(seed=0x5555555555555555, simulation_run_id=0x6666666666666666)
        profile = FaultProfile(base_latency_ns=1000, jitter_abs_ns=200)

        jitters = []
        for frame_id in range(100):
            decision = model.apply_faults(profile, 0, frame_id, 8192, 0, 0)
            jitters.append(decision.jitter_ns)

        # Jitter should be in [-200, +200]
        assert all(-200 <= j <= 200 for j in jitters)
        # Should have both positive and negative (statistically)
        assert any(j < 0 for j in jitters)
        assert any(j > 0 for j in jitters)

    def test_bandwidth_serialization(self):
        """Test bandwidth serializer enforces link availability."""
        model = FaultModel(seed=0x7777777777777777, simulation_run_id=0x8888888888888888)
        profile = FaultProfile(base_latency_ns=0, bitrate_bps=1_000_000_000)  # 1 Gbps

        # First frame: 8192 bits at 1 Gbps
        # tx_duration = ceil(8192 * 1e9 / 1e9) = ceil(8192) = 8192 ns
        decision1 = model.apply_faults(profile, 0, 1, 8192, 1000, 0)
        assert decision1.tx_start_ns == decision1.due_ns  # Link idle
        assert decision1.tx_duration_ns == 8192  # 8192 bits / 1 Gbps = 8192 ns
        assert decision1.release_ns == 1000 + 8192  # ingress + latency(0) + duration

        # Second frame must wait for first to complete
        decision2 = model.apply_faults(
            profile, 0, 2, 8192, 1000, decision1.release_ns
        )
        assert decision2.tx_start_ns == decision1.release_ns

    def test_corrupt_frame_applies_xor(self):
        """Test frame corruption applies XOR mask correctly."""
        model = FaultModel(seed=0x9999999999999999, simulation_run_id=0xAAAAAAAAAAAAAAAA)

        frame = b"\xff\xff\xff\xff"  # All bits set
        bit_offsets = [0, 7, 8, 15]  # First and last bit of first two bytes

        corrupted = model.corrupt_frame(frame, bit_offsets)

        # Bit 0: MSB of byte 0 (0x80)
        # Bit 7: LSB of byte 0 (0x01)
        # Bit 8: MSB of byte 1 (0x80)
        # Bit 15: LSB of byte 1 (0x01)
        # 0xFF XOR 0x81 = 0x7E
        assert corrupted == b"\x7e\x7e\xff\xff"


class TestInMemoryTransport:
    """Test in-memory transport."""

    def test_send_receive_roundtrip(self):
        """Test send and receive."""
        transport = InMemoryTransport()

        envelope = SidebandEnvelope(
            magic=SidebandEnvelope.MAGIC,
            version=1,
            direction=Direction.EGRESS,
            reserved=0,
            spacecraft_instance_id=1,
            sender_boot_id=1,
            link_session_id=1,
            sender_frame_id=1,
            link_frame_id=100,
            file_epoch_id=0,
            frame_length=4,
        )
        frame_bytes = b"\xde\xad\xbe\xef"

        transport.send(envelope, frame_bytes)
        received = transport.receive()

        assert received is not None
        assert received.envelope == envelope
        assert received.frame_bytes == frame_bytes

    def test_receive_empty_queue(self):
        """Test receive on empty queue returns None."""
        transport = InMemoryTransport()
        assert transport.receive() is None


class TestLinkSimulator:
    """Test Link Simulator integration."""

    def test_admit_frame_assigns_link_frame_id(self):
        """Test frame admission assigns monotonic link_frame_id."""
        clock = VirtualClock()
        transport = InMemoryTransport()

        sim = LinkSimulator(
            simulation_run_id=0x1111111111111111,
            seed=0x2222222222222222,
            uplink_profile=FaultProfile(),
            downlink_profile=FaultProfile(),
            clock=clock,
            transport=transport,
        )

        session_id = sim.create_session(
            spacecraft_instance_id=0xAAAAAAAAAAAAAAAA,
            sender_boot_id=1,
        )

        # Create ingress frame
        envelope = SidebandEnvelope(
            magic=SidebandEnvelope.MAGIC,
            version=1,
            direction=Direction.INGRESS,
            reserved=0,
            spacecraft_instance_id=0xAAAAAAAAAAAAAAAA,
            sender_boot_id=0,
            link_session_id=session_id,
            sender_frame_id=1,
            link_frame_id=0,
            file_epoch_id=0,
            frame_length=1024,
        )
        frame = TransportFrame(envelope=envelope, frame_bytes=b"\x00" * 1024)

        link_frame_id = sim.admit_frame(frame)
        assert link_frame_id == 1

        # Second frame gets next ID
        envelope2 = SidebandEnvelope(
            magic=SidebandEnvelope.MAGIC,
            version=1,
            direction=Direction.INGRESS,
            reserved=0,
            spacecraft_instance_id=0xAAAAAAAAAAAAAAAA,
            sender_boot_id=0,
            link_session_id=session_id,
            sender_frame_id=2,
            link_frame_id=0,
            file_epoch_id=0,
            frame_length=1024,
        )
        frame2 = TransportFrame(envelope=envelope2, frame_bytes=b"\x00" * 1024)

        link_frame_id2 = sim.admit_frame(frame2)
        assert link_frame_id2 == 2

    def test_frame_with_loss_not_delivered(self):
        """Test frame with 100% loss is not delivered."""
        clock = VirtualClock()
        transport = InMemoryTransport()

        # 100% loss rate
        profile = FaultProfile(frame_loss_rate_ppm=1_000_000)

        sim = LinkSimulator(
            simulation_run_id=0x3333333333333333,
            seed=0x4444444444444444,
            uplink_profile=profile,
            downlink_profile=profile,
            clock=clock,
            transport=transport,
        )

        session_id = sim.create_session(
            spacecraft_instance_id=0xBBBBBBBBBBBBBBBB,
            sender_boot_id=1,
        )

        envelope = SidebandEnvelope(
            magic=SidebandEnvelope.MAGIC,
            version=1,
            direction=Direction.INGRESS,
            reserved=0,
            spacecraft_instance_id=0xBBBBBBBBBBBBBBBB,
            sender_boot_id=0,
            link_session_id=session_id,
            sender_frame_id=1,
            link_frame_id=0,
            file_epoch_id=0,
            frame_length=1024,
        )
        frame = TransportFrame(envelope=envelope, frame_bytes=b"\x00" * 1024)

        sim.admit_frame(frame)
        sim.run_until_idle()

        # Frame should not be delivered
        received = transport.receive()
        assert received is None

    def test_deterministic_replay_same_seed(self):
        """Test same seed produces same fault decisions."""
        seed = 0x5555555555555555
        run_id = 0x6666666666666666

        results1 = []
        results2 = []

        for trial in [results1, results2]:
            clock = VirtualClock()
            transport = InMemoryTransport()
            profile = FaultProfile(frame_loss_rate_ppm=500_000)

            sim = LinkSimulator(
                simulation_run_id=run_id,
                seed=seed,
                uplink_profile=profile,
                downlink_profile=profile,
                clock=clock,
                transport=transport,
            )

            session_id = sim.create_session(
                spacecraft_instance_id=0xCCCCCCCCCCCCCCCC,
                sender_boot_id=1,
            )

            for sender_frame_id in range(10):
                envelope = SidebandEnvelope(
                    magic=SidebandEnvelope.MAGIC,
                    version=1,
                    direction=Direction.INGRESS,
                    reserved=0,
                    spacecraft_instance_id=0xCCCCCCCCCCCCCCCC,
                    sender_boot_id=0,
                    link_session_id=session_id,
                    sender_frame_id=sender_frame_id,
                    link_frame_id=0,
                    file_epoch_id=0,
                    frame_length=1024,
                )
                frame = TransportFrame(envelope=envelope, frame_bytes=b"\x00" * 1024)
                link_frame_id = sim.admit_frame(frame)
                trial.append(link_frame_id)

        # Same seed should produce same results
        assert results1 == results2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
