"""Tests for Queue Manager with overflow/backpressure."""

import pytest
import threading
import time

from link_sim.queue_manager import (
    QueueConfig,
    QueueManager,
    QueueOverflowPolicy,
    QueuedFrame,
)


def test_enqueue_dequeue_basic():
    """Test basic enqueue/dequeue."""
    config = QueueConfig(capacity=10)
    queue = QueueManager(config)

    frame = QueuedFrame(
        link_frame_id=1,
        admission_order=1,
        ingress_time_ns=1000,
        frame_bytes=b"test",
    )

    assert queue.enqueue(frame)
    stats = queue.get_stats()
    assert stats.current_depth == 1
    assert stats.total_enqueued == 1

    dequeued = queue.dequeue()
    assert dequeued is not None
    assert dequeued.link_frame_id == 1

    stats = queue.get_stats()  # Get fresh stats after dequeue
    assert stats.total_dequeued == 1


def test_drop_policy_overflow():
    """Test DROP policy when queue is full."""
    config = QueueConfig(capacity=3, overflow_policy=QueueOverflowPolicy.DROP)
    queue = QueueManager(config)

    # Fill queue
    for i in range(3):
        frame = QueuedFrame(
            link_frame_id=i + 1,
            admission_order=i + 1,
            ingress_time_ns=(i + 1) * 1000,
            frame_bytes=b"test",
        )
        assert queue.enqueue(frame)

    stats = queue.get_stats()
    assert stats.current_depth == 3

    # Overflow - should drop
    overflow_frame = QueuedFrame(
        link_frame_id=999,
        admission_order=999,
        ingress_time_ns=999000,
        frame_bytes=b"overflow",
    )
    assert not queue.enqueue(overflow_frame)

    stats = queue.get_stats()
    assert stats.current_depth == 3
    assert stats.total_dropped == 1


def test_log_only_policy():
    """Test LOG_ONLY policy for UDP (overflow logged, not dropped)."""
    config = QueueConfig(
        capacity=2,
        overflow_policy=QueueOverflowPolicy.LOG_ONLY,
        enable_fallback_log=True,
    )
    queue = QueueManager(config)

    # Fill queue
    for i in range(2):
        frame = QueuedFrame(
            link_frame_id=i + 1,
            admission_order=i + 1,
            ingress_time_ns=(i + 1) * 1000,
            frame_bytes=b"test",
        )
        assert queue.enqueue(frame)

    # Overflow - should log but return True (UDP delivered)
    overflow_frame = QueuedFrame(
        link_frame_id=999,
        admission_order=999,
        ingress_time_ns=999000,
        frame_bytes=b"overflow",
    )
    assert queue.enqueue(overflow_frame)  # Returns True for UDP

    stats = queue.get_stats()
    assert stats.current_depth == 2  # Still full
    assert stats.total_logged_fallback == 1
    assert stats.total_dropped == 0  # Not counted as dropped

    fallback = queue.get_fallback_log()
    assert len(fallback) == 1
    assert fallback[0]["link_frame_id"] == 999
    assert fallback[0]["reason"] == "queue_full_udp_delivered"


def test_block_policy():
    """Test BLOCK policy with background dequeue."""
    config = QueueConfig(capacity=2, overflow_policy=QueueOverflowPolicy.BLOCK)
    queue = QueueManager(config)

    # Fill queue
    for i in range(2):
        frame = QueuedFrame(
            link_frame_id=i + 1,
            admission_order=i + 1,
            ingress_time_ns=(i + 1) * 1000,
            frame_bytes=b"test",
        )
        queue.enqueue(frame)

    # Background thread to dequeue after delay
    def dequeue_after_delay():
        time.sleep(0.1)
        queue.dequeue()

    thread = threading.Thread(target=dequeue_after_delay)
    thread.start()

    # This should block until dequeue makes space
    start = time.time()
    overflow_frame = QueuedFrame(
        link_frame_id=999,
        admission_order=999,
        ingress_time_ns=999000,
        frame_bytes=b"overflow",
    )
    assert queue.enqueue(overflow_frame)  # Blocks then succeeds
    elapsed = time.time() - start

    assert elapsed >= 0.1  # Should have blocked
    thread.join()

    stats = queue.get_stats()
    assert stats.total_enqueued == 3
    assert stats.total_dequeued == 1


def test_high_water_mark():
    """Test high water mark tracking."""
    config = QueueConfig(capacity=10)
    queue = QueueManager(config)

    # Enqueue 5 frames
    for i in range(5):
        frame = QueuedFrame(
            link_frame_id=i + 1,
            admission_order=i + 1,
            ingress_time_ns=(i + 1) * 1000,
            frame_bytes=b"test",
        )
        queue.enqueue(frame)

    stats = queue.get_stats()
    assert stats.high_water_mark == 5

    # Dequeue 3
    for _ in range(3):
        queue.dequeue()

    stats = queue.get_stats()
    assert stats.current_depth == 2
    assert stats.high_water_mark == 5  # Should remain at peak

    # Enqueue 7 more (total depth = 9)
    for i in range(7):
        frame = QueuedFrame(
            link_frame_id=i + 10,
            admission_order=i + 10,
            ingress_time_ns=(i + 10) * 1000,
            frame_bytes=b"test",
        )
        queue.enqueue(frame)

    stats = queue.get_stats()
    assert stats.current_depth == 9
    assert stats.high_water_mark == 9  # Updated


def test_dequeue_timeout():
    """Test dequeue with timeout."""
    config = QueueConfig(capacity=10)
    queue = QueueManager(config)

    # Empty queue, dequeue with timeout
    start = time.time()
    result = queue.dequeue(timeout_ms=100)
    elapsed = time.time() - start

    assert result is None
    assert elapsed >= 0.1
    assert elapsed < 0.2


def test_peek():
    """Test peek without removing."""
    config = QueueConfig(capacity=10)
    queue = QueueManager(config)

    frame = QueuedFrame(
        link_frame_id=42,
        admission_order=1,
        ingress_time_ns=1000,
        frame_bytes=b"test",
    )
    queue.enqueue(frame)

    # Peek
    peeked = queue.peek()
    assert peeked is not None
    assert peeked.link_frame_id == 42

    # Queue still has frame
    stats = queue.get_stats()
    assert stats.current_depth == 1

    # Dequeue
    dequeued = queue.dequeue()
    assert dequeued.link_frame_id == 42

    # Now empty
    assert queue.peek() is None


def test_clear():
    """Test clearing queue."""
    config = QueueConfig(capacity=10)
    queue = QueueManager(config)

    # Enqueue 5 frames
    for i in range(5):
        frame = QueuedFrame(
            link_frame_id=i + 1,
            admission_order=i + 1,
            ingress_time_ns=(i + 1) * 1000,
            frame_bytes=b"test",
        )
        queue.enqueue(frame)

    stats = queue.get_stats()
    assert stats.current_depth == 5

    # Clear
    dropped = queue.clear()
    assert dropped == 5

    stats = queue.get_stats()
    assert stats.current_depth == 0
    assert stats.total_dropped == 5


def test_concurrent_enqueue_dequeue():
    """Test concurrent operations."""
    config = QueueConfig(capacity=100)
    queue = QueueManager(config)

    enqueue_count = 50
    dequeue_count = 50

    def enqueue_worker():
        for i in range(enqueue_count):
            frame = QueuedFrame(
                link_frame_id=i + 1,
                admission_order=i + 1,
                ingress_time_ns=(i + 1) * 1000,
                frame_bytes=b"test",
            )
            queue.enqueue(frame)

    def dequeue_worker():
        for _ in range(dequeue_count):
            while queue.dequeue(timeout_ms=10) is None:
                pass

    # Start threads
    enqueue_thread = threading.Thread(target=enqueue_worker)
    dequeue_thread = threading.Thread(target=dequeue_worker)

    enqueue_thread.start()
    dequeue_thread.start()

    enqueue_thread.join()
    dequeue_thread.join()

    stats = queue.get_stats()
    assert stats.total_enqueued == enqueue_count
    assert stats.total_dequeued == dequeue_count


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
