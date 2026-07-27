"""Tests for FilePacket epoch with START/DATA/END drain fence."""

import pytest

from link_sim.file_epoch import (
    EpochState,
    FileEpochManager,
    PacketType,
)


def test_admit_start_success():
    """Test successful START admission."""
    manager = FileEpochManager()
    manager.open_session(sender_boot_id=1)

    attempt_id = manager.admit_start(
        session_id=100,
        spacecraft_instance_id=68,
        sender_boot_id=1,
        link_frame_id=1,
        file_path="/data/product1.tar",
    )

    assert attempt_id == 1
    attempt = manager.get_current_attempt()
    assert attempt is not None
    assert attempt.state == EpochState.STARTED
    assert attempt.file_path == "/data/product1.tar"


def test_start_busy_when_transfer_active():
    """Test START rejected when transfer is busy."""
    manager = FileEpochManager()
    manager.open_session(sender_boot_id=1)

    # Start first transfer
    attempt_id = manager.admit_start(
        session_id=100,
        spacecraft_instance_id=68,
        sender_boot_id=1,
        link_frame_id=1,
        file_path="/data/product1.tar",
    )
    assert attempt_id == 1

    # Try to start second transfer - should fail
    can_start, reason = manager.can_start_transfer(
        session_id=100, spacecraft_instance_id=68, sender_boot_id=1
    )
    assert not can_start
    assert reason == "TRANSFER_BUSY"

    # Attempt to admit START anyway
    attempt_id2 = manager.admit_start(
        session_id=100,
        spacecraft_instance_id=68,
        sender_boot_id=1,
        link_frame_id=10,
        file_path="/data/product2.tar",
    )
    assert attempt_id2 is None


def test_admit_data():
    """Test DATA packet admission."""
    manager = FileEpochManager()
    manager.open_session(sender_boot_id=1)

    # Start transfer
    attempt_id = manager.admit_start(
        session_id=100,
        spacecraft_instance_id=68,
        sender_boot_id=1,
        link_frame_id=1,
        file_path="/data/product.tar",
    )

    # Admit DATA packets
    result, error = manager.admit_data(
        sender_boot_id=1, link_frame_id=2, data_offset=0, data_length=990
    )
    assert result == attempt_id
    assert error is None

    result, error = manager.admit_data(
        sender_boot_id=1, link_frame_id=3, data_offset=990, data_length=990
    )
    assert result == attempt_id
    assert error is None

    stats = manager.get_stats()
    assert stats["data_packets_sent"] == 2
    assert stats["data_buffer_depth"] == 2


def test_admit_end_transitions_to_draining():
    """Test END admission transitions to DRAINING."""
    manager = FileEpochManager()
    manager.open_session(sender_boot_id=1)

    # Start transfer
    attempt_id = manager.admit_start(
        session_id=100,
        spacecraft_instance_id=68,
        sender_boot_id=1,
        link_frame_id=1,
        file_path="/data/product.tar",
    )

    # Send some DATA
    manager.admit_data(
        sender_boot_id=1, link_frame_id=2, data_offset=0, data_length=990
    )

    # Admit END
    result, error = manager.admit_end(sender_boot_id=1, link_frame_id=3)
    assert result == attempt_id
    assert error is None

    attempt = manager.get_current_attempt()
    assert attempt.state == EpochState.DRAINING
    assert attempt.end_link_frame_id == 3


def test_complete_attempt_releases_fence():
    """Test completing attempt releases fence for next transfer."""
    manager = FileEpochManager()
    manager.open_session(sender_boot_id=1)

    # Start and complete first transfer
    attempt_id1 = manager.admit_start(
        session_id=100,
        spacecraft_instance_id=68,
        sender_boot_id=1,
        link_frame_id=1,
        file_path="/data/product1.tar",
    )
    manager.admit_end(sender_boot_id=1, link_frame_id=2)
    assert manager.complete_attempt(attempt_id1)

    # Now can start second transfer
    can_start, reason = manager.can_start_transfer(
        session_id=100, spacecraft_instance_id=68, sender_boot_id=1
    )
    assert can_start
    assert reason is None

    attempt_id2 = manager.admit_start(
        session_id=100,
        spacecraft_instance_id=68,
        sender_boot_id=1,
        link_frame_id=10,
        file_path="/data/product2.tar",
    )
    assert attempt_id2 == 2


def test_stale_sender_boot_rejected():
    """Test packets from stale sender boot are rejected."""
    manager = FileEpochManager()
    manager.open_session(sender_boot_id=1)

    # Start transfer with boot 1
    attempt_id = manager.admit_start(
        session_id=100,
        spacecraft_instance_id=68,
        sender_boot_id=1,
        link_frame_id=1,
        file_path="/data/product.tar",
    )

    # Send DATA with boot 1
    result, error = manager.admit_data(
        sender_boot_id=1, link_frame_id=2, data_offset=0, data_length=990
    )
    assert result == attempt_id

    # Sender reboots (boot 2)
    manager.open_session(sender_boot_id=2)

    # Late DATA from boot 1 should be rejected
    result, error = manager.admit_data(
        sender_boot_id=1, link_frame_id=3, data_offset=990, data_length=990
    )
    assert result is None
    assert error == "STALE_SENDER_BOOT"

    # Late END from boot 1 should be rejected
    result, error = manager.admit_end(sender_boot_id=1, link_frame_id=4)
    assert result is None
    assert error == "STALE_SENDER_BOOT"

    # START with boot 1 should be rejected
    attempt_id2 = manager.admit_start(
        session_id=100,
        spacecraft_instance_id=68,
        sender_boot_id=1,
        link_frame_id=10,
        file_path="/data/product2.tar",
    )
    assert attempt_id2 is None


def test_data_without_start_rejected():
    """Test DATA without START is rejected."""
    manager = FileEpochManager()
    manager.open_session(sender_boot_id=1)

    # Try to send DATA without START
    result, error = manager.admit_data(
        sender_boot_id=1, link_frame_id=1, data_offset=0, data_length=990
    )
    assert result is None
    assert error == "NO_ACTIVE_TRANSFER"


def test_end_without_start_rejected():
    """Test END without START is rejected."""
    manager = FileEpochManager()
    manager.open_session(sender_boot_id=1)

    # Try to send END without START
    result, error = manager.admit_end(sender_boot_id=1, link_frame_id=1)
    assert result is None
    assert error == "NO_ACTIVE_TRANSFER"


def test_abort_attempt():
    """Test aborting attempt enters ABORTING state."""
    manager = FileEpochManager()
    manager.open_session(sender_boot_id=1)

    # Start transfer
    attempt_id = manager.admit_start(
        session_id=100,
        spacecraft_instance_id=68,
        sender_boot_id=1,
        link_frame_id=1,
        file_path="/data/product.tar",
    )

    # Send some DATA
    manager.admit_data(
        sender_boot_id=1, link_frame_id=2, data_offset=0, data_length=990
    )

    # Abort
    assert manager.abort_attempt(attempt_id, link_frame_id=3)

    attempt = manager.get_current_attempt()
    assert attempt.state == EpochState.ABORTING
    assert attempt.abort_link_frame_id == 3

    # Further DATA should be rejected
    result, error = manager.admit_data(
        sender_boot_id=1, link_frame_id=4, data_offset=990, data_length=990
    )
    assert result is None
    assert error == "INVALID_STATE_ABORTING"


def test_data_after_end_rejected():
    """Test DATA after END is rejected."""
    manager = FileEpochManager()
    manager.open_session(sender_boot_id=1)

    # Start transfer
    attempt_id = manager.admit_start(
        session_id=100,
        spacecraft_instance_id=68,
        sender_boot_id=1,
        link_frame_id=1,
        file_path="/data/product.tar",
    )

    # Send END
    manager.admit_end(sender_boot_id=1, link_frame_id=2)

    # Try to send DATA after END
    result, error = manager.admit_data(
        sender_boot_id=1, link_frame_id=3, data_offset=0, data_length=990
    )
    assert result is None
    assert error == "INVALID_STATE_DRAINING"


def test_boot_change_closes_epoch():
    """Test sender boot change closes all epochs."""
    manager = FileEpochManager()
    manager.open_session(sender_boot_id=1)

    # Start transfer with boot 1
    attempt_id1 = manager.admit_start(
        session_id=100,
        spacecraft_instance_id=68,
        sender_boot_id=1,
        link_frame_id=1,
        file_path="/data/product1.tar",
    )
    assert attempt_id1 == 1

    attempt = manager.get_current_attempt()
    assert attempt.state == EpochState.STARTED

    # Reboot
    manager.open_session(sender_boot_id=2)

    # Old attempt should be closed
    attempt = manager._attempts.get(attempt_id1)
    assert attempt.state == EpochState.CLOSED

    # Can start new transfer with boot 2
    attempt_id2 = manager.admit_start(
        session_id=200,
        spacecraft_instance_id=68,
        sender_boot_id=2,
        link_frame_id=10,
        file_path="/data/product2.tar",
    )
    assert attempt_id2 == 2


def test_no_cross_attempt_contamination():
    """Test packets from attempt A don't cross into attempt B."""
    manager = FileEpochManager()
    manager.open_session(sender_boot_id=1)

    # Attempt A
    attempt_a = manager.admit_start(
        session_id=100,
        spacecraft_instance_id=68,
        sender_boot_id=1,
        link_frame_id=1,
        file_path="/data/productA.tar",
    )
    manager.admit_data(
        sender_boot_id=1, link_frame_id=2, data_offset=0, data_length=990
    )
    manager.admit_end(sender_boot_id=1, link_frame_id=3)
    manager.complete_attempt(attempt_a)

    # Attempt B
    attempt_b = manager.admit_start(
        session_id=100,
        spacecraft_instance_id=68,
        sender_boot_id=1,
        link_frame_id=10,
        file_path="/data/productB.tar",
    )

    # Verify attempt B has clean state
    stats = manager.get_stats()
    assert stats["current_attempt_id"] == attempt_b
    assert stats["data_packets_sent"] == 0
    assert stats["data_buffer_depth"] == 0  # Buffer cleared


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
