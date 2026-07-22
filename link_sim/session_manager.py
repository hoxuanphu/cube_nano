"""Session handshake and restart resolution for Link Simulator.

Section 9.10: Spacecraft boot/session handshake to prevent cross-boot/session
packet contamination during sender restart.
"""

import logging
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Mapping, Optional

logger = logging.getLogger(__name__)


class SessionState(Enum):
    """Session lifecycle state."""
    ACTIVE = "ACTIVE"
    CLOSING = "CLOSING"
    CLOSED = "CLOSED"


@dataclass
class Session:
    """Link session metadata."""
    session_id: int  # U64
    spacecraft_instance_id: int  # U64
    sender_boot_id: int  # U32
    state: SessionState
    generation: int  # Session generation counter
    created_at_ns: int  # Simulation time
    closed_at_ns: Optional[int] = None


class SessionManager:
    """Manages session lifecycle and boot handshake.

    Section 9.10: Enforces boot/session boundary to prevent stale packets
    from old boot/session cross into new one. Key invariants:

    1. Each sender_boot_id gets unique session_id
    2. Session close epoch isolates old packets
    3. New boot waits for old session drain/close
    4. Startup delivery policy: drop old boot queued packets
    5. Generation counter detects session reuse bugs
    """

    def __init__(
        self,
        *,
        next_session_id: int = 1,
        generation_floors: Mapping[int, int] | None = None,
    ):
        """Initialize session manager from an optional durable checkpoint."""
        if isinstance(next_session_id, bool) or not 1 <= int(next_session_id) <= 0xFFFFFFFFFFFFFFFF:
            raise ValueError("next_session_id must fit U64 and be positive")
        normalized_floors: Dict[int, int] = {}
        for raw_instance, raw_generation in (generation_floors or {}).items():
            if (
                isinstance(raw_instance, bool)
                or isinstance(raw_generation, bool)
                or not 1 <= int(raw_instance) <= 0xFFFFFFFFFFFFFFFF
                or not 0 <= int(raw_generation) <= 0xFFFFFFFFFFFFFFFF
            ):
                raise ValueError("generation_floors must contain U64 instance and generation values")
            normalized_floors[int(raw_instance)] = int(raw_generation)
        self._lock = threading.Lock()

        # Active sessions by spacecraft_instance_id
        self._active_sessions: Dict[int, Session] = {}

        # Session history (for stale packet detection)
        self._session_history: Dict[int, Session] = {}  # session_id -> Session

        # Next session ID allocator
        self._next_session_id = int(next_session_id)
        # History is process-local, but these floors survive a LinkSimulator
        # restart so a new HELLO cannot reuse a prior generation.
        self._generation_floors = normalized_floors

        logger.info("SessionManager initialized")

    def create_session(
        self,
        spacecraft_instance_id: int,
        sender_boot_id: int,
        current_time_ns: int,
    ) -> int:
        """Create new session for sender boot.

        Args:
            spacecraft_instance_id: Spacecraft instance
            sender_boot_id: Sender boot ID
            current_time_ns: Current simulation time

        Returns:
            session_id (U64)
        """
        with self._lock:
            # Close existing session for this spacecraft
            old_session = self._active_sessions.get(spacecraft_instance_id)
            if old_session:
                logger.info(
                    f"Closing old session: session_id={old_session.session_id:#018x}, "
                    f"old_boot={old_session.sender_boot_id:#010x}, "
                    f"new_boot={sender_boot_id:#010x}"
                )
                old_session.state = SessionState.CLOSING
                old_session.closed_at_ns = current_time_ns

            # Allocate new session
            session_id = self._next_session_id
            if session_id >= 0xFFFFFFFFFFFFFFFF:
                raise RuntimeError("link session ID allocator exhausted")
            self._next_session_id += 1

            # A reset closes/removes the active entry before allocating the
            # replacement, so generation must be derived from retained
            # history as well as a currently active session.  Otherwise a
            # boot change can reuse generation 1 and defeat endpoint fences.
            prior_generations = [
                item.generation
                for item in self._session_history.values()
                if item.spacecraft_instance_id == spacecraft_instance_id
            ]
            floor = self._generation_floors.get(spacecraft_instance_id, 0)
            generation = max([floor, *prior_generations]) + 1
            if generation > 0xFFFFFFFFFFFFFFFF:
                raise RuntimeError("link generation allocator exhausted")

            session = Session(
                session_id=session_id,
                spacecraft_instance_id=spacecraft_instance_id,
                sender_boot_id=sender_boot_id,
                state=SessionState.ACTIVE,
                generation=generation,
                created_at_ns=current_time_ns,
            )

            self._active_sessions[spacecraft_instance_id] = session
            self._session_history[session_id] = session
            self._generation_floors[spacecraft_instance_id] = generation

            logger.info(
                f"Session created: session_id={session_id:#018x}, "
                f"instance={spacecraft_instance_id:#018x}, "
                f"boot={sender_boot_id:#010x}, gen={generation}"
            )

            return session_id

    def checkpoint(self) -> dict[str, object]:
        """Return only the monotonic allocators needed after process restart."""

        with self._lock:
            return {
                "next_session_id": self._next_session_id,
                "generation_floors": dict(self._generation_floors),
            }

    def close_session(
        self, session_id: int, current_time_ns: int
    ) -> Optional[Session]:
        """Close session and transition to CLOSED.

        Args:
            session_id: Session ID
            current_time_ns: Current simulation time

        Returns:
            Closed session or None if not found
        """
        with self._lock:
            session = self._session_history.get(session_id)
            if not session:
                logger.warning(f"Session not found: session_id={session_id:#018x}")
                return None

            if session.state == SessionState.CLOSED:
                logger.debug(f"Session already closed: session_id={session_id:#018x}")
                return session

            session.state = SessionState.CLOSED
            session.closed_at_ns = current_time_ns

            # Remove from active if still there
            if self._active_sessions.get(session.spacecraft_instance_id) == session:
                del self._active_sessions[session.spacecraft_instance_id]

            logger.info(
                f"Session closed: session_id={session_id:#018x}, "
                f"duration_ns={current_time_ns - session.created_at_ns}"
            )

            return session

    def validate_packet(
        self,
        session_id: int,
        sender_boot_id: int,
        spacecraft_instance_id: int,
    ) -> tuple[bool, Optional[str]]:
        """Validate packet against active session.

        Args:
            session_id: Session ID from packet
            sender_boot_id: Sender boot ID from packet
            spacecraft_instance_id: Spacecraft instance

        Returns:
            (is_valid, reason) - True if valid, else (False, reason)
        """
        with self._lock:
            # Check if session exists
            session = self._session_history.get(session_id)
            if not session:
                return False, "UNKNOWN_SESSION"

            # Check if session is active
            if session.state != SessionState.ACTIVE:
                return False, f"SESSION_{session.state.value}"

            # Check sender boot match
            if session.sender_boot_id != sender_boot_id:
                return False, "SENDER_BOOT_MISMATCH"

            # Check spacecraft instance match
            if session.spacecraft_instance_id != spacecraft_instance_id:
                return False, "INSTANCE_MISMATCH"

            # Check if this is the active session for this spacecraft
            active = self._active_sessions.get(spacecraft_instance_id)
            if not active or active.session_id != session_id:
                return False, "STALE_SESSION"

            return True, None

    def get_active_session(
        self, spacecraft_instance_id: int
    ) -> Optional[Session]:
        """Get active session for spacecraft instance.

        Args:
            spacecraft_instance_id: Spacecraft instance

        Returns:
            Active session or None
        """
        with self._lock:
            return self._active_sessions.get(spacecraft_instance_id)

    def get_session(self, session_id: int) -> Optional[Session]:
        """Get session by ID.

        Args:
            session_id: Session ID

        Returns:
            Session or None
        """
        with self._lock:
            return self._session_history.get(session_id)

    def should_drop_packet(
        self, session_id: int, sender_boot_id: int
    ) -> tuple[bool, str]:
        """Determine if packet should be dropped during startup/drain.

        Args:
            session_id: Session ID
            sender_boot_id: Sender boot ID

        Returns:
            (should_drop, reason)
        """
        with self._lock:
            session = self._session_history.get(session_id)

            # Unknown session -> drop
            if not session:
                return True, "UNKNOWN_SESSION"

            # Closed session -> drop
            if session.state == SessionState.CLOSED:
                return True, "SESSION_CLOSED"

            # Boot mismatch -> drop
            if session.sender_boot_id != sender_boot_id:
                return True, "BOOT_MISMATCH"

            # CLOSING session -> drop (drain policy)
            if session.state == SessionState.CLOSING:
                return True, "SESSION_CLOSING"

            # ACTIVE -> deliver
            return False, "ACTIVE"

    def get_stats(self) -> dict:
        """Get session statistics.

        Returns:
            Statistics dict
        """
        with self._lock:
            active_count = len(self._active_sessions)
            total_sessions = len(self._session_history)
            closing_count = sum(
                1 for s in self._session_history.values()
                if s.state == SessionState.CLOSING
            )
            closed_count = sum(
                1 for s in self._session_history.values()
                if s.state == SessionState.CLOSED
            )

            return {
                "active_sessions": active_count,
                "closing_sessions": closing_count,
                "closed_sessions": closed_count,
                "total_sessions": total_sessions,
                "next_session_id": self._next_session_id,
            }
