"""Queue manager for Link Simulator with overflow/backpressure.

Section 9.8: Bounded queue with overflow counter, backpressure policy,
and fallback logging when queue is full.
"""

import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class QueueOverflowPolicy(Enum):
    """Policy when queue is full."""
    DROP = "drop"  # Drop new frame, increment counter
    BLOCK = "block"  # Block until space available (in-memory only)
    LOG_ONLY = "log_only"  # Log to fallback, continue (UDP)


@dataclass
class QueueStats:
    """Queue statistics."""
    capacity: int
    current_depth: int
    total_enqueued: int
    total_dequeued: int
    total_dropped: int
    total_logged_fallback: int
    high_water_mark: int


@dataclass
class QueueConfig:
    """Queue configuration."""
    capacity: int = 1000  # Max frames in queue
    overflow_policy: QueueOverflowPolicy = QueueOverflowPolicy.DROP
    enable_fallback_log: bool = True
    fallback_log_path: Optional[str] = None


@dataclass
class QueuedFrame:
    """Frame queued for processing."""
    link_frame_id: int
    admission_order: int
    ingress_time_ns: int
    frame_bytes: bytes
    metadata: dict = field(default_factory=dict)


class QueueManager:
    """Bounded queue with overflow handling.

    Section 9.8: Implements bounded queue with configurable overflow policy,
    metrics, and optional fallback logging when queue is full.

    For InMemoryTransport: DROP or BLOCK policy.
    For UdpTransport: LOG_ONLY (frame delivered via UDP but overflow logged).
    """

    def __init__(self, config: QueueConfig):
        """Initialize queue manager.

        Args:
            config: Queue configuration
        """
        self.config = config
        self._lock = threading.Lock()
        self._queue: list[QueuedFrame] = []
        self._condition = threading.Condition(self._lock)

        # Metrics
        self._total_enqueued = 0
        self._total_dequeued = 0
        self._total_dropped = 0
        self._total_logged_fallback = 0
        self._high_water_mark = 0

        # Fallback log (simple in-memory for MVP; file-based for production)
        self._fallback_log: list[dict] = []

        logger.info(
            f"QueueManager initialized: capacity={config.capacity}, "
            f"policy={config.overflow_policy.value}"
        )

    def enqueue(self, frame: QueuedFrame) -> bool:
        """Enqueue frame with overflow handling.

        Args:
            frame: Frame to enqueue

        Returns:
            True if enqueued, False if dropped (DROP policy)
        """
        with self._condition:
            # Check capacity
            if len(self._queue) >= self.config.capacity:
                if self.config.overflow_policy == QueueOverflowPolicy.DROP:
                    self._total_dropped += 1
                    logger.warning(
                        f"Queue full: frame dropped (link_frame_id={frame.link_frame_id})"
                    )
                    return False

                elif self.config.overflow_policy == QueueOverflowPolicy.LOG_ONLY:
                    # Frame delivered via UDP, just log overflow
                    self._total_logged_fallback += 1
                    if self.config.enable_fallback_log:
                        self._fallback_log.append({
                            "link_frame_id": frame.link_frame_id,
                            "admission_order": frame.admission_order,
                            "ingress_time_ns": frame.ingress_time_ns,
                            "reason": "queue_full_udp_delivered",
                        })
                    logger.info(
                        f"Queue full: frame logged to fallback "
                        f"(link_frame_id={frame.link_frame_id})"
                    )
                    return True  # Considered "enqueued" for UDP

                elif self.config.overflow_policy == QueueOverflowPolicy.BLOCK:
                    # Block until space available
                    logger.debug(
                        f"Queue full: blocking (link_frame_id={frame.link_frame_id})"
                    )
                    while len(self._queue) >= self.config.capacity:
                        self._condition.wait()

            # Enqueue frame
            self._queue.append(frame)
            self._total_enqueued += 1

            # Update high water mark
            current_depth = len(self._queue)
            if current_depth > self._high_water_mark:
                self._high_water_mark = current_depth

            # Notify waiters
            self._condition.notify()

            logger.debug(
                f"Frame enqueued: link_frame_id={frame.link_frame_id}, "
                f"depth={current_depth}/{self.config.capacity}"
            )

            return True

    def dequeue(self, timeout_ms: Optional[int] = None) -> Optional[QueuedFrame]:
        """Dequeue frame with optional timeout.

        Args:
            timeout_ms: Timeout in milliseconds, None for blocking

        Returns:
            Frame or None if timeout/empty
        """
        timeout_s = timeout_ms / 1000.0 if timeout_ms is not None else None

        with self._condition:
            # Wait for frame with timeout
            if not self._queue:
                if timeout_s is None:
                    self._condition.wait()
                else:
                    if not self._condition.wait(timeout=timeout_s):
                        return None  # Timeout

            # Still empty after wait?
            if not self._queue:
                return None

            # Dequeue
            frame = self._queue.pop(0)
            self._total_dequeued += 1

            # Notify blocked enqueue threads
            self._condition.notify()

            logger.debug(
                f"Frame dequeued: link_frame_id={frame.link_frame_id}, "
                f"remaining={len(self._queue)}"
            )

            return frame

    def peek(self) -> Optional[QueuedFrame]:
        """Peek at next frame without removing.

        Returns:
            Next frame or None if empty
        """
        with self._lock:
            return self._queue[0] if self._queue else None

    def get_stats(self) -> QueueStats:
        """Get queue statistics.

        Returns:
            Queue statistics
        """
        with self._lock:
            return QueueStats(
                capacity=self.config.capacity,
                current_depth=len(self._queue),
                total_enqueued=self._total_enqueued,
                total_dequeued=self._total_dequeued,
                total_dropped=self._total_dropped,
                total_logged_fallback=self._total_logged_fallback,
                high_water_mark=self._high_water_mark,
            )

    def get_fallback_log(self) -> list[dict]:
        """Get fallback log entries.

        Returns:
            List of fallback log entries
        """
        with self._lock:
            return self._fallback_log.copy()

    def clear(self) -> int:
        """Clear queue and return number of dropped frames.

        Returns:
            Number of frames dropped
        """
        with self._condition:
            dropped = len(self._queue)
            self._queue.clear()
            self._total_dropped += dropped
            self._condition.notify_all()
            logger.info(f"Queue cleared: {dropped} frames dropped")
            return dropped
