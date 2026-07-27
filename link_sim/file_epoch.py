"""FilePacket epoch management with START/DATA/END drain fence.

Section 9.9: File transfer epoch with attempt barrier, DATA ordering,
abort fence, and transfer busy lock to prevent cross-attempt/boot reordering.
"""

import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class EpochState(Enum):
    """File epoch state."""
    IDLE = "IDLE"  # No active transfer
    STARTED = "STARTED"  # START sent, waiting for DATA/END
    DRAINING = "DRAINING"  # END sent, waiting for drain
    ABORTING = "ABORTING"  # Abort requested, waiting for fence
    CLOSED = "CLOSED"  # Epoch closed (sender boot closed or explicit close)


class PacketType(Enum):
    """FilePacket type."""
    START = "START"
    DATA = "DATA"
    END = "END"
    CANCEL = "CANCEL"


@dataclass
class FileTransferAttempt:
    """Single file transfer attempt."""
    attempt_id: int  # Global monotonic attempt ID
    epoch_id: int
    session_id: int
    spacecraft_instance_id: int
    sender_boot_id: int
    file_path: str
    state: EpochState
    start_link_frame_id: int
    data_packets_sent: int = 0
    end_link_frame_id: Optional[int] = None
    abort_link_frame_id: Optional[int] = None


@dataclass
class FileEpochManager:
    """Manages file transfer epochs with drain fence.

    Section 9.9: Enforces START/DATA/END drain fence to prevent packet
    cross-contamination between attempts. Key invariants:

    1. Only ONE global attempt is IN_FLIGHT at a time
    2. DATA/END from attempt A cannot cross into attempt B
    3. START B only admitted after attempt A fully drained
    4. Sender boot change closes all epochs, drops stale packets
    5. Abort fence ensures late callbacks don't violate attempt boundary
    """

    def __init__(self):
        """Initialize file epoch manager."""
        self._lock = threading.Lock()
        self._next_attempt_id = 1

        # Current active attempt (at most ONE globally)
        self._current_attempt: Optional[FileTransferAttempt] = None

        # Attempt history (for late packet detection)
        self._attempts: Dict[int, FileTransferAttempt] = {}

        # Sender boot tracking (for restart resolution)
        self._active_sender_boot_id: Optional[int] = None

        # Reorder buffer for DATA packets (simple list for MVP)
        self._data_buffer: list[dict] = []

        logger.info("FileEpochManager initialized")

    def open_session(self, sender_boot_id: int) -> None:
        """Open new sender boot session.

        Closes all previous epochs and resets state.

        Args:
            sender_boot_id: New sender boot ID
        """
        with self._lock:
            if self._active_sender_boot_id == sender_boot_id:
                logger.warning(
                    f"Session already open: sender_boot_id={sender_boot_id:#010x}"
                )
                return

            # Close previous session
            if self._current_attempt:
                old_boot_str = f"{self._active_sender_boot_id:#010x}" if self._active_sender_boot_id else "0x00000000"
                logger.info(
                    f"Closing previous session: attempt_id={self._current_attempt.attempt_id}, "
                    f"old_boot={old_boot_str}, new_boot={sender_boot_id:#010x}"
                )
                self._current_attempt.state = EpochState.CLOSED

            self._active_sender_boot_id = sender_boot_id
            self._current_attempt = None
            self._data_buffer.clear()

            logger.info(f"Session opened: sender_boot_id={sender_boot_id:#010x}")

    def close_session(self, sender_boot_id: int) -> None:
        """Close the current sender boot and reject late file packets."""
        with self._lock:
            if self._active_sender_boot_id != sender_boot_id:
                return
            if self._current_attempt is not None:
                self._current_attempt.state = EpochState.CLOSED
            self._current_attempt = None
            self._data_buffer.clear()
            self._active_sender_boot_id = None

    def can_start_transfer(
        self, session_id: int, spacecraft_instance_id: int, sender_boot_id: int
    ) -> tuple[bool, Optional[str]]:
        """Check if new transfer can start (drain fence check).

        Args:
            session_id: Link session ID
            spacecraft_instance_id: Spacecraft instance
            sender_boot_id: Sender boot ID

        Returns:
            (can_start, reason) - True if can start, else (False, reason)
        """
        with self._lock:
            # Check sender boot match
            if self._active_sender_boot_id is None:
                return False, "NO_ACTIVE_SESSION"

            if sender_boot_id != self._active_sender_boot_id:
                return False, f"STALE_SENDER_BOOT"

            # Check if transfer busy
            if self._current_attempt:
                if self._current_attempt.state in (
                    EpochState.STARTED,
                    EpochState.DRAINING,
                    EpochState.ABORTING,
                ):
                    return False, "TRANSFER_BUSY"

            # Can start
            return True, None

    def admit_start(
        self,
        session_id: int,
        spacecraft_instance_id: int,
        sender_boot_id: int,
        link_frame_id: int,
        file_path: str,
    ) -> Optional[int]:
        """Admit START packet and create new attempt.

        Args:
            session_id: Link session ID
            spacecraft_instance_id: Spacecraft instance
            sender_boot_id: Sender boot ID
            link_frame_id: Link frame ID
            file_path: File path

        Returns:
            attempt_id if admitted, None if rejected
        """
        with self._lock:
            # Check sender boot match (inline to avoid nested lock)
            if self._active_sender_boot_id is None:
                logger.warning(f"START rejected: NO_ACTIVE_SESSION")
                return None

            if sender_boot_id != self._active_sender_boot_id:
                logger.warning(f"START rejected: STALE_SENDER_BOOT")
                return None

            # Check if transfer busy
            if self._current_attempt:
                if self._current_attempt.state in (
                    EpochState.STARTED,
                    EpochState.DRAINING,
                    EpochState.ABORTING,
                ):
                    logger.warning(f"START rejected: TRANSFER_BUSY")
                    return None

            # Now can start - create new attempt
            # Create new attempt
            epoch_id = self._next_attempt_id  # For MVP, epoch_id = attempt_id
            attempt_id = self._next_attempt_id
            self._next_attempt_id += 1

            attempt = FileTransferAttempt(
                attempt_id=attempt_id,
                epoch_id=epoch_id,
                session_id=session_id,
                spacecraft_instance_id=spacecraft_instance_id,
                sender_boot_id=sender_boot_id,
                file_path=file_path,
                state=EpochState.STARTED,
                start_link_frame_id=link_frame_id,
            )

            self._current_attempt = attempt
            self._attempts[attempt_id] = attempt

            logger.info(
                f"START admitted: attempt_id={attempt_id}, "
                f"link_frame_id={link_frame_id}, file={file_path}"
            )

            return attempt_id

    def admit_data(
        self,
        sender_boot_id: int,
        link_frame_id: int,
        data_offset: int,
        data_length: int,
    ) -> tuple[Optional[int], Optional[str]]:
        """Admit DATA packet.

        Args:
            sender_boot_id: Sender boot ID
            link_frame_id: Link frame ID
            data_offset: Byte offset in file
            data_length: Data length

        Returns:
            (attempt_id, error_reason) - attempt_id if admitted, else (None, reason)
        """
        with self._lock:
            # Check sender boot
            if sender_boot_id != self._active_sender_boot_id:
                return None, "STALE_SENDER_BOOT"

            # Check active attempt
            if not self._current_attempt:
                return None, "NO_ACTIVE_TRANSFER"

            if self._current_attempt.state != EpochState.STARTED:
                return None, f"INVALID_STATE_{self._current_attempt.state.value}"

            # Admit DATA
            self._current_attempt.data_packets_sent += 1

            # Add to reorder buffer (MVP: simple append, full impl: ordered insert)
            self._data_buffer.append({
                "link_frame_id": link_frame_id,
                "offset": data_offset,
                "length": data_length,
            })

            logger.debug(
                f"DATA admitted: attempt_id={self._current_attempt.attempt_id}, "
                f"link_frame_id={link_frame_id}, offset={data_offset}, len={data_length}"
            )

            return self._current_attempt.attempt_id, None

    def admit_end(
        self, sender_boot_id: int, link_frame_id: int
    ) -> tuple[Optional[int], Optional[str]]:
        """Admit END packet and transition to DRAINING.

        Args:
            sender_boot_id: Sender boot ID
            link_frame_id: Link frame ID

        Returns:
            (attempt_id, error_reason)
        """
        with self._lock:
            # Check sender boot
            if sender_boot_id != self._active_sender_boot_id:
                return None, "STALE_SENDER_BOOT"

            # Check active attempt
            if not self._current_attempt:
                return None, "NO_ACTIVE_TRANSFER"

            if self._current_attempt.state != EpochState.STARTED:
                return None, f"INVALID_STATE_{self._current_attempt.state.value}"

            # Admit END
            self._current_attempt.end_link_frame_id = link_frame_id
            self._current_attempt.state = EpochState.DRAINING

            logger.info(
                f"END admitted: attempt_id={self._current_attempt.attempt_id}, "
                f"link_frame_id={link_frame_id}, draining..."
            )

            return self._current_attempt.attempt_id, None

    def complete_attempt(self, attempt_id: int) -> bool:
        """Mark attempt as complete and release fence.

        Args:
            attempt_id: Attempt ID

        Returns:
            True if completed, False if not found
        """
        with self._lock:
            if not self._current_attempt or self._current_attempt.attempt_id != attempt_id:
                logger.warning(f"Attempt not current: attempt_id={attempt_id}")
                return False

            # Close attempt
            self._current_attempt.state = EpochState.CLOSED
            self._current_attempt = None
            self._data_buffer.clear()

            logger.info(f"Attempt complete: attempt_id={attempt_id}, fence released")

            return True

    def abort_attempt(self, attempt_id: int, link_frame_id: int) -> bool:
        """Abort current attempt and enter abort fence.

        Args:
            attempt_id: Attempt ID
            link_frame_id: Link frame ID where abort occurred

        Returns:
            True if aborted, False if not found
        """
        with self._lock:
            if not self._current_attempt or self._current_attempt.attempt_id != attempt_id:
                logger.warning(f"Attempt not current: attempt_id={attempt_id}")
                return False

            # Transition to ABORTING
            self._current_attempt.state = EpochState.ABORTING
            self._current_attempt.abort_link_frame_id = link_frame_id

            logger.info(
                f"Attempt aborting: attempt_id={attempt_id}, "
                f"abort_frame={link_frame_id}, entering fence..."
            )

            return True

    def get_current_attempt(self) -> Optional[FileTransferAttempt]:
        """Get current active attempt.

        Returns:
            Current attempt or None
        """
        with self._lock:
            return self._current_attempt

    def get_stats(self) -> dict:
        """Get file epoch statistics.

        Returns:
            Statistics dict
        """
        with self._lock:
            current = self._current_attempt
            return {
                "active_sender_boot_id": (
                    f"{self._active_sender_boot_id:#010x}"
                    if self._active_sender_boot_id
                    else None
                ),
                "current_attempt_id": current.attempt_id if current else None,
                "current_state": current.state.value if current else "IDLE",
                "data_packets_sent": current.data_packets_sent if current else 0,
                "data_buffer_depth": len(self._data_buffer),
                "total_attempts": len(self._attempts),
            }
