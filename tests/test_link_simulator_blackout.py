"""Tests for blackout and bandwidth shaping (P3-04).

Section 9.2: Frames during blackout are dropped; bandwidth shaper serializes transmission.
"""

import pytest
from link_sim.contact_schedule import BandwidthProfile, ContactSchedule, ContactState
from link_sim.fault_model import FaultProfile
from link_sim.link_simulator import LinkSimulator
from link_sim.transport import Direction, SidebandEnvelope, TransportFrame
from link_sim.virtual_clock import SimulationTime, VirtualClock


def test_blackout_drops_frame():
    """Frames arriving during BLACKOUT are dropped."""
    clock = VirtualClock()
    schedule = ContactSchedule()

    # Add blackout window: 1s to 2s
    schedule.add_window(
        start_time=SimulationTime(1_000_000_000),
        end_time=SimulationTime(2_000_000_000),
        state=ContactState.BLACKOUT,
    )

    profile = FaultProfile(base_latency_ns=100_000)
    sim = LinkSimulator(
        simulation_run_id=1,
        seed=42,
        uplink_profile=profile,
        downlink_profile=profile,
        clock=clock,
        contact_schedule=schedule,
    )

    session_id = sim.create_session(spacecraft_instance_id=100, sender_boot_id=1)

    # Frame before blackout: should be admitted
    envelope = SidebandEnvelope(
        magic=SidebandEnvelope.MAGIC,
        version=SidebandEnvelope.VERSION,
        direction=Direction.INGRESS,
        reserved=0,
        spacecraft_instance_id=100,
        sender_boot_id=0,  # Ingress must have sender_boot_id=0
        link_session_id=session_id,
        sender_frame_id=1,
        link_frame_id=0,
        file_epoch_id=0,
        frame_length=1024,
    )
    frame = TransportFrame(envelope=envelope, frame_bytes=b"X" * 1024)

    link_frame_id = sim.admit_frame(frame)
    assert link_frame_id == 1  # Admitted

    # Advance into blackout
    clock.run_until(SimulationTime(1_500_000_000))

    # Frame during blackout: should be dropped
    envelope2 = SidebandEnvelope(
        magic=SidebandEnvelope.MAGIC,
        version=SidebandEnvelope.VERSION,
        direction=Direction.INGRESS,
        reserved=0,
        spacecraft_instance_id=100,
        sender_boot_id=0,
        link_session_id=session_id,
        sender_frame_id=2,
        link_frame_id=0,
        file_epoch_id=0,
        frame_length=1024,
    )
    frame2 = TransportFrame(envelope=envelope2, frame_bytes=b"Y" * 1024)

    link_frame_id2 = sim.admit_frame(frame2)
    assert link_frame_id2 is None  # Dropped during blackout

    # Advance past blackout
    clock.run_until(SimulationTime(2_500_000_000))

    # Frame after blackout: should be admitted
    envelope3 = SidebandEnvelope(
        magic=SidebandEnvelope.MAGIC,
        version=SidebandEnvelope.VERSION,
        direction=Direction.INGRESS,
        reserved=0,
        spacecraft_instance_id=100,
        sender_boot_id=0,
        link_session_id=session_id,
        sender_frame_id=3,
        link_frame_id=0,
        file_epoch_id=0,
        frame_length=1024,
    )
    frame3 = TransportFrame(envelope=envelope3, frame_bytes=b"Z" * 1024)

    link_frame_id3 = sim.admit_frame(frame3)
    assert link_frame_id3 == 2  # Admitted (link_frame_id continues from 1)


def test_no_contact_allows_admission():
    """NO_CONTACT does not drop frames at link layer."""
    clock = VirtualClock()
    schedule = ContactSchedule()

    # Add NO_CONTACT window: 1s to 2s
    schedule.add_window(
        start_time=SimulationTime(1_000_000_000),
        end_time=SimulationTime(2_000_000_000),
        state=ContactState.NO_CONTACT,
    )

    profile = FaultProfile(base_latency_ns=100_000)
    sim = LinkSimulator(
        simulation_run_id=1,
        seed=42,
        uplink_profile=profile,
        downlink_profile=profile,
        clock=clock,
        contact_schedule=schedule,
    )

    session_id = sim.create_session(spacecraft_instance_id=100, sender_boot_id=1)

    # Advance into NO_CONTACT
    clock.run_until(SimulationTime(1_500_000_000))

    # Frame during NO_CONTACT: should still be admitted (link layer does not drop)
    envelope = SidebandEnvelope(
        magic=SidebandEnvelope.MAGIC,
        version=SidebandEnvelope.VERSION,
        direction=Direction.INGRESS,
        reserved=0,
        spacecraft_instance_id=100,
        sender_boot_id=0,
        link_session_id=session_id,
        sender_frame_id=1,
        link_frame_id=0,
        file_epoch_id=0,
        frame_length=1024,
    )
    frame = TransportFrame(envelope=envelope, frame_bytes=b"X" * 1024)

    link_frame_id = sim.admit_frame(frame)
    assert link_frame_id == 1  # Admitted (GDS may reject commands, but link admits)


def test_bandwidth_shaper_serialization():
    """Bandwidth shaper serializes frame transmission."""
    profile = FaultProfile(
        base_latency_ns=0,
        bitrate_bps=1_000_000,  # 1 Mbps
    )

    clock = VirtualClock()
    sim = LinkSimulator(
        simulation_run_id=1,
        seed=42,
        uplink_profile=profile,
        downlink_profile=profile,
        clock=clock,
    )

    session_id = sim.create_session(spacecraft_instance_id=100, sender_boot_id=1)

    # Frame 1: 1024 bytes = 8192 bits
    # tx_duration = ceil(8192 * 1e9 / 1e6) = 8192 us = 8192000 ns
    envelope1 = SidebandEnvelope(
        magic=SidebandEnvelope.MAGIC,
        version=SidebandEnvelope.VERSION,
        direction=Direction.INGRESS,
        reserved=0,
        spacecraft_instance_id=100,
        sender_boot_id=0,
        link_session_id=session_id,
        sender_frame_id=1,
        link_frame_id=0,
        file_epoch_id=0,
        frame_length=1024,
    )
    frame1 = TransportFrame(envelope=envelope1, frame_bytes=b"X" * 1024)

    link_frame_id1 = sim.admit_frame(frame1)
    assert link_frame_id1 == 1

    # Frame 1 should be released at tx_duration_ns = 8192000 ns
    # (base_latency=0, jitter=0, no reorder, tx_start=0)

    # Frame 2: should start after frame 1 completes
    envelope2 = SidebandEnvelope(
        magic=SidebandEnvelope.MAGIC,
        version=SidebandEnvelope.VERSION,
        direction=Direction.INGRESS,
        reserved=0,
        spacecraft_instance_id=100,
        sender_boot_id=0,
        link_session_id=session_id,
        sender_frame_id=2,
        link_frame_id=0,
        file_epoch_id=0,
        frame_length=1024,
    )
    frame2 = TransportFrame(envelope=envelope2, frame_bytes=b"Y" * 1024)

    link_frame_id2 = sim.admit_frame(frame2)
    assert link_frame_id2 == 2

    # Frame 2 should be released at 8192000 + 8192000 = 16384000 ns
    # Verify serialization by checking uplink_available_ns
    assert sim._uplink_available_ns == 16384000


def test_bandwidth_profile_validation():
    """BandwidthProfile validates bitrate."""
    profile = BandwidthProfile(bitrate_bps=1_000_000)
    profile.validate()  # Should pass

    with pytest.raises(ValueError, match="bitrate_bps must be > 0"):
        bad_profile = BandwidthProfile(bitrate_bps=0)
        bad_profile.validate()

    with pytest.raises(ValueError, match="bitrate_bps must be > 0"):
        bad_profile = BandwidthProfile(bitrate_bps=-1000)
        bad_profile.validate()


def test_contact_schedule_overlapping_windows():
    """ContactSchedule rejects overlapping windows."""
    schedule = ContactSchedule()

    # Add first window: 1s to 3s
    schedule.add_window(
        start_time=SimulationTime(1_000_000_000),
        end_time=SimulationTime(3_000_000_000),
        state=ContactState.BLACKOUT,
    )

    # Try to add overlapping window: 2s to 4s (should fail)
    with pytest.raises(ValueError, match="Window overlaps existing"):
        schedule.add_window(
            start_time=SimulationTime(2_000_000_000),
            end_time=SimulationTime(4_000_000_000),
            state=ContactState.NO_CONTACT,
        )

    # Add non-overlapping window: 4s to 5s (should succeed)
    schedule.add_window(
        start_time=SimulationTime(4_000_000_000),
        end_time=SimulationTime(5_000_000_000),
        state=ContactState.NO_CONTACT,
    )
