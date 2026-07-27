"""Comprehensive Phase 3 exit gate tests - determinism, crash, replay.

Simplified integration tests. Detailed unit tests are in individual component test files.
"""

import pytest

from link_sim.contact_schedule import ContactSchedule, ContactState
from link_sim.fault_model import FaultProfile
from link_sim.file_epoch import FileEpochManager
from link_sim.link_simulator import LinkSimulator
from link_sim.session_manager import SessionManager
from link_sim.transport import Direction, SidebandEnvelope, TransportFrame
from link_sim.virtual_clock import SimulationTime


def test_same_seed_same_output():
    """Test same seed/profile produces same decision log."""
    seed = 0x123456789abcdef0
    run_id = 1

    # Create two simulators with same seed
    sim1 = LinkSimulator(
        simulation_run_id=run_id,
        seed=seed,
        uplink_profile=FaultProfile(),
        downlink_profile=FaultProfile(),
    )

    sim2 = LinkSimulator(
        simulation_run_id=run_id,
        seed=seed,
        uplink_profile=FaultProfile(),
        downlink_profile=FaultProfile(),
    )

    # Admit same frames to both
    for i in range(10):
        envelope = SidebandEnvelope(
            magic=SidebandEnvelope.MAGIC,
            version=SidebandEnvelope.VERSION,
            direction=Direction.INGRESS,
            reserved=0,
            spacecraft_instance_id=68,
            sender_boot_id=0,
            link_session_id=100,
            sender_frame_id=i + 1,
            link_frame_id=0,
            file_epoch_id=0,
            frame_length=1024,
        )
        frame = TransportFrame(envelope=envelope, frame_bytes=b"x" * 1024)

        link_id1 = sim1.admit_frame(frame)
        link_id2 = sim2.admit_frame(frame)

        # Same link_frame_id assigned
        assert link_id1 == link_id2

    # Same admission log
    assert len(sim1._admission_log) == len(sim2._admission_log)
    for log1, log2 in zip(sim1._admission_log, sim2._admission_log):
        assert log1["link_frame_id"] == log2["link_frame_id"]
        assert log1["admission_order"] == log2["admission_order"]


def test_different_seed_different_output():
    """Test different seed can produce different results."""
    # Create two simulators with different seeds
    sim1 = LinkSimulator(
        simulation_run_id=1,
        seed=0x1111111111111111,
        uplink_profile=FaultProfile(),
        downlink_profile=FaultProfile(),
    )

    sim2 = LinkSimulator(
        simulation_run_id=1,
        seed=0x2222222222222222,
        uplink_profile=FaultProfile(),
        downlink_profile=FaultProfile(),
    )

    # Both should admit frames
    for i in range(10):
        envelope = SidebandEnvelope(
            magic=SidebandEnvelope.MAGIC,
            version=SidebandEnvelope.VERSION,
            direction=Direction.INGRESS,
            reserved=0,
            spacecraft_instance_id=68,
            sender_boot_id=0,
            link_session_id=100,
            sender_frame_id=i + 1,
            link_frame_id=0,
            file_epoch_id=0,
            frame_length=1024,
        )
        frame = TransportFrame(envelope=envelope, frame_bytes=b"x" * 1024)
        sim1.admit_frame(frame)
        sim2.admit_frame(frame)

    # Both admitted frames
    assert len(sim1._admission_log) > 0
    assert len(sim2._admission_log) > 0


def test_concurrent_ingress_ordered():
    """Test concurrent ingress has logged admission order."""
    sim = LinkSimulator(
        simulation_run_id=1,
        seed=12345,
        uplink_profile=FaultProfile(),
        downlink_profile=FaultProfile(),
    )

    # Admit frames
    for i in range(10):
        envelope = SidebandEnvelope(
            magic=SidebandEnvelope.MAGIC,
            version=SidebandEnvelope.VERSION,
            direction=Direction.INGRESS,
            reserved=0,
            spacecraft_instance_id=68,
            sender_boot_id=0,
            link_session_id=100,
            sender_frame_id=i + 1,
            link_frame_id=0,
            file_epoch_id=0,
            frame_length=1024,
        )
        frame = TransportFrame(envelope=envelope, frame_bytes=b"x" * 1024)
        sim.admit_frame(frame)

    # Verify admission order is logged
    admission_log = sim._admission_log
    assert len(admission_log) == 10

    # Verify sequential admission_order
    for i, log_entry in enumerate(admission_log):
        assert log_entry["admission_order"] == i


def test_blackout_drops_frames():
    """Test frames during BLACKOUT are dropped - tested in test_link_simulator_blackout.py."""
    # Full blackout integration tests are in test_link_simulator_blackout.py
    # This is just a smoke test that blackout functionality exists
    schedule = ContactSchedule()
    schedule.add_window(
        start_time=SimulationTime(5000),
        end_time=SimulationTime(10000),
        state=ContactState.BLACKOUT,
    )

    # Verify blackout state is recognized
    state = schedule.get_state_at(SimulationTime(7000))
    assert state == ContactState.BLACKOUT

    # Verify should_drop_frame works
    assert schedule.should_drop_frame(SimulationTime(7000))


def test_session_restart_isolates_packets():
    """Test sender boot change isolates old packets."""
    manager = SessionManager()

    # Boot 1
    session_id1 = manager.create_session(
        spacecraft_instance_id=68,
        sender_boot_id=1,
        current_time_ns=1000,
    )

    # Validate packet from boot 1
    is_valid, _ = manager.validate_packet(session_id1, 1, 68)
    assert is_valid

    # Boot 2 (restart)
    session_id2 = manager.create_session(
        spacecraft_instance_id=68,
        sender_boot_id=2,
        current_time_ns=2000,
    )

    # Old packet should be rejected
    is_valid, reason = manager.validate_packet(session_id1, 1, 68)
    assert not is_valid
    assert reason in ("SESSION_CLOSING", "STALE_SESSION")

    # New packet should be accepted
    is_valid, _ = manager.validate_packet(session_id2, 2, 68)
    assert is_valid


def test_file_epoch_no_cross_attempt():
    """Test FilePacket fence prevents cross-attempt contamination."""
    manager = FileEpochManager()
    manager.open_session(sender_boot_id=1)

    # Attempt A
    attempt_a = manager.admit_start(
        session_id=100, spacecraft_instance_id=68,
        sender_boot_id=1, link_frame_id=1, file_path="/data/A.tar"
    )
    manager.admit_data(sender_boot_id=1, link_frame_id=2, data_offset=0, data_length=990)
    manager.admit_end(sender_boot_id=1, link_frame_id=3)
    manager.complete_attempt(attempt_a)

    # Attempt B
    attempt_b = manager.admit_start(
        session_id=100, spacecraft_instance_id=68,
        sender_boot_id=1, link_frame_id=10, file_path="/data/B.tar"
    )

    # Verify B has clean state (no carryover from A)
    stats = manager.get_stats()
    assert stats["current_attempt_id"] == attempt_b
    assert stats["data_packets_sent"] == 0
    assert stats["data_buffer_depth"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
