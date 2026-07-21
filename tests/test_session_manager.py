"""Tests for Session Manager with boot/session handshake."""

import pytest

from link_sim.session_manager import (
    Session,
    SessionManager,
    SessionState,
)


def test_create_session():
    """Test creating a new session."""
    manager = SessionManager()

    session_id = manager.create_session(
        spacecraft_instance_id=68,
        sender_boot_id=1,
        current_time_ns=1000,
    )

    assert session_id == 1
    session = manager.get_session(session_id)
    assert session is not None
    assert session.state == SessionState.ACTIVE
    assert session.spacecraft_instance_id == 68
    assert session.sender_boot_id == 1
    assert session.generation == 1


def test_create_session_closes_old():
    """Test creating new session closes old session."""
    manager = SessionManager()

    # Create first session
    session_id1 = manager.create_session(
        spacecraft_instance_id=68,
        sender_boot_id=1,
        current_time_ns=1000,
    )

    # Create second session (sender reboot)
    session_id2 = manager.create_session(
        spacecraft_instance_id=68,
        sender_boot_id=2,
        current_time_ns=2000,
    )

    assert session_id2 == 2

    # Old session should be CLOSING
    old_session = manager.get_session(session_id1)
    assert old_session.state == SessionState.CLOSING
    assert old_session.closed_at_ns == 2000

    # New session should be ACTIVE
    new_session = manager.get_session(session_id2)
    assert new_session.state == SessionState.ACTIVE
    assert new_session.generation == 2


def test_validate_packet_success():
    """Test successful packet validation."""
    manager = SessionManager()

    session_id = manager.create_session(
        spacecraft_instance_id=68,
        sender_boot_id=1,
        current_time_ns=1000,
    )

    # Valid packet
    is_valid, reason = manager.validate_packet(
        session_id=session_id,
        sender_boot_id=1,
        spacecraft_instance_id=68,
    )

    assert is_valid
    assert reason is None


def test_validate_packet_unknown_session():
    """Test validation fails for unknown session."""
    manager = SessionManager()

    is_valid, reason = manager.validate_packet(
        session_id=999,
        sender_boot_id=1,
        spacecraft_instance_id=68,
    )

    assert not is_valid
    assert reason == "UNKNOWN_SESSION"


def test_validate_packet_sender_boot_mismatch():
    """Test validation fails for boot mismatch."""
    manager = SessionManager()

    session_id = manager.create_session(
        spacecraft_instance_id=68,
        sender_boot_id=1,
        current_time_ns=1000,
    )

    # Wrong sender_boot_id
    is_valid, reason = manager.validate_packet(
        session_id=session_id,
        sender_boot_id=2,
        spacecraft_instance_id=68,
    )

    assert not is_valid
    assert reason == "SENDER_BOOT_MISMATCH"


def test_validate_packet_stale_session():
    """Test validation fails for stale session after new boot."""
    manager = SessionManager()

    # First boot
    session_id1 = manager.create_session(
        spacecraft_instance_id=68,
        sender_boot_id=1,
        current_time_ns=1000,
    )

    # Second boot (closes first session)
    session_id2 = manager.create_session(
        spacecraft_instance_id=68,
        sender_boot_id=2,
        current_time_ns=2000,
    )

    # Packet from old session should be stale
    is_valid, reason = manager.validate_packet(
        session_id=session_id1,
        sender_boot_id=1,
        spacecraft_instance_id=68,
    )

    assert not is_valid
    assert reason == "SESSION_CLOSING"  # Old session is CLOSING


def test_validate_packet_closed_session():
    """Test validation fails for closed session."""
    manager = SessionManager()

    session_id = manager.create_session(
        spacecraft_instance_id=68,
        sender_boot_id=1,
        current_time_ns=1000,
    )

    # Close session
    manager.close_session(session_id, current_time_ns=2000)

    # Packet should be rejected
    is_valid, reason = manager.validate_packet(
        session_id=session_id,
        sender_boot_id=1,
        spacecraft_instance_id=68,
    )

    assert not is_valid
    assert reason == "SESSION_CLOSED"


def test_should_drop_packet_closing_session():
    """Test packets are dropped during session drain."""
    manager = SessionManager()

    # First boot
    session_id1 = manager.create_session(
        spacecraft_instance_id=68,
        sender_boot_id=1,
        current_time_ns=1000,
    )

    # Second boot (puts first in CLOSING)
    session_id2 = manager.create_session(
        spacecraft_instance_id=68,
        sender_boot_id=2,
        current_time_ns=2000,
    )

    # Packet from CLOSING session should be dropped
    should_drop, reason = manager.should_drop_packet(
        session_id=session_id1,
        sender_boot_id=1,
    )

    assert should_drop
    assert reason == "SESSION_CLOSING"


def test_should_drop_packet_active_session():
    """Test packets are NOT dropped for active session."""
    manager = SessionManager()

    session_id = manager.create_session(
        spacecraft_instance_id=68,
        sender_boot_id=1,
        current_time_ns=1000,
    )

    # Packet from ACTIVE session should NOT be dropped
    should_drop, reason = manager.should_drop_packet(
        session_id=session_id,
        sender_boot_id=1,
    )

    assert not should_drop
    assert reason == "ACTIVE"


def test_get_active_session():
    """Test getting active session."""
    manager = SessionManager()

    session_id = manager.create_session(
        spacecraft_instance_id=68,
        sender_boot_id=1,
        current_time_ns=1000,
    )

    active = manager.get_active_session(spacecraft_instance_id=68)
    assert active is not None
    assert active.session_id == session_id
    assert active.state == SessionState.ACTIVE


def test_close_session():
    """Test closing a session."""
    manager = SessionManager()

    session_id = manager.create_session(
        spacecraft_instance_id=68,
        sender_boot_id=1,
        current_time_ns=1000,
    )

    # Close
    closed = manager.close_session(session_id, current_time_ns=3000)
    assert closed is not None
    assert closed.state == SessionState.CLOSED
    assert closed.closed_at_ns == 3000

    # No longer active
    active = manager.get_active_session(spacecraft_instance_id=68)
    assert active is None


def test_generation_counter():
    """Test generation counter increments."""
    manager = SessionManager()

    # First session
    session_id1 = manager.create_session(
        spacecraft_instance_id=68,
        sender_boot_id=1,
        current_time_ns=1000,
    )
    session1 = manager.get_session(session_id1)
    assert session1.generation == 1

    # Second session
    session_id2 = manager.create_session(
        spacecraft_instance_id=68,
        sender_boot_id=2,
        current_time_ns=2000,
    )
    session2 = manager.get_session(session_id2)
    assert session2.generation == 2

    # Third session
    session_id3 = manager.create_session(
        spacecraft_instance_id=68,
        sender_boot_id=3,
        current_time_ns=3000,
    )
    session3 = manager.get_session(session_id3)
    assert session3.generation == 3


def test_checkpoint_preserves_monotonic_session_and_generation_allocators():
    """A fresh link process must not reuse a prior session/generation pair."""

    manager = SessionManager()
    first = manager.create_session(68, 1, 1_000)
    assert first == 1
    checkpoint = manager.checkpoint()
    restored = SessionManager(
        next_session_id=int(checkpoint["next_session_id"]),
        generation_floors=checkpoint["generation_floors"],
    )
    second = restored.create_session(68, 1, 2_000)
    session = restored.get_session(second)
    assert session is not None
    assert second == 2
    assert session.generation == 2


def test_multiple_spacecraft_instances():
    """Test managing multiple spacecraft instances."""
    manager = SessionManager()

    # Instance 68, boot 1
    session_id1 = manager.create_session(
        spacecraft_instance_id=68,
        sender_boot_id=1,
        current_time_ns=1000,
    )

    # Instance 69, boot 1
    session_id2 = manager.create_session(
        spacecraft_instance_id=69,
        sender_boot_id=1,
        current_time_ns=1000,
    )

    # Both should be active
    active1 = manager.get_active_session(spacecraft_instance_id=68)
    active2 = manager.get_active_session(spacecraft_instance_id=69)

    assert active1.session_id == session_id1
    assert active2.session_id == session_id2

    # Validate both independently
    is_valid1, _ = manager.validate_packet(session_id1, 1, 68)
    is_valid2, _ = manager.validate_packet(session_id2, 1, 69)

    assert is_valid1
    assert is_valid2


def test_get_stats():
    """Test session statistics."""
    manager = SessionManager()

    # Initial stats
    stats = manager.get_stats()
    assert stats["active_sessions"] == 0
    assert stats["total_sessions"] == 0

    # Create first session
    session_id1 = manager.create_session(
        spacecraft_instance_id=68,
        sender_boot_id=1,
        current_time_ns=1000,
    )

    stats = manager.get_stats()
    assert stats["active_sessions"] == 1
    assert stats["total_sessions"] == 1

    # Create second session (closes first)
    session_id2 = manager.create_session(
        spacecraft_instance_id=68,
        sender_boot_id=2,
        current_time_ns=2000,
    )

    stats = manager.get_stats()
    assert stats["active_sessions"] == 1
    assert stats["closing_sessions"] == 1
    assert stats["total_sessions"] == 2

    # Close first session explicitly
    manager.close_session(session_id1, current_time_ns=2500)

    # Close second session
    manager.close_session(session_id2, current_time_ns=3000)

    stats = manager.get_stats()
    assert stats["active_sessions"] == 0
    assert stats["closed_sessions"] == 2
    assert stats["total_sessions"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
