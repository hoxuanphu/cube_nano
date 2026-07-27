"""Virtual clock and ordered event queue for deterministic simulation.

Section 9.2: Simulation uses monotonic virtual time (sim_time_ns) for deterministic
scheduling. Event queue uses tie-breaker ordering for concurrent events.
"""

import heapq
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional


@dataclass(frozen=True)
class SimulationTime:
    """Simulation time in nanoseconds. Checked integer, no overflow."""

    ns: int  # U64 in spec

    def __post_init__(self):
        if self.ns < 0:
            raise ValueError(f"Simulation time cannot be negative: {self.ns}")
        if self.ns > (2**64 - 1):
            raise ValueError(f"Simulation time overflow: {self.ns}")

    def __add__(self, delta_ns: int) -> "SimulationTime":
        """Add duration with overflow check."""
        result = self.ns + delta_ns
        if result < 0 or result > (2**64 - 1):
            raise OverflowError(f"Time arithmetic overflow: {self.ns} + {delta_ns}")
        return SimulationTime(result)

    def __sub__(self, other: "SimulationTime") -> int:
        """Compute duration between two times."""
        return self.ns - other.ns

    def __lt__(self, other: "SimulationTime") -> bool:
        return self.ns < other.ns

    def __le__(self, other: "SimulationTime") -> bool:
        return self.ns <= other.ns

    def __gt__(self, other: "SimulationTime") -> bool:
        return self.ns > other.ns

    def __ge__(self, other: "SimulationTime") -> bool:
        return self.ns >= other.ns

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SimulationTime):
            return NotImplemented
        return self.ns == other.ns

    def __hash__(self) -> int:
        return hash(self.ns)


@dataclass(order=True)
class ScheduledEvent:
    """Event scheduled for future delivery.

    Ordering: (due_time, direction, link_frame_id, copy_index) for tie-breaking.
    Section 9.2: deterministic ordering even when events have same due time.
    """

    due_time: SimulationTime
    direction: int  # 0=uplink, 1=downlink
    link_frame_id: int
    copy_index: int

    # Payload is not part of ordering
    callback: Callable[[], None] = field(compare=False)
    payload: Any = field(default=None, compare=False)


class VirtualClock:
    """Virtual clock for deterministic simulation.

    Section 9.2: Monotonic simulation time with ordered event queue.
    Replay does not depend on wall-clock thread timing.
    """

    def __init__(self, start_time: SimulationTime = None):
        """Initialize virtual clock."""
        self._current_time = start_time or SimulationTime(0)
        self._event_queue: List[ScheduledEvent] = []
        self._admission_order = 0  # Monotonic counter for ingress ordering

    @property
    def now(self) -> SimulationTime:
        """Get current simulation time."""
        return self._current_time

    def schedule(
        self,
        delay_ns: int,
        callback: Callable[[], None],
        direction: int,
        link_frame_id: int,
        copy_index: int = 0,
        payload: Any = None,
    ) -> None:
        """Schedule event for future delivery.

        Args:
            delay_ns: Delay from current time
            callback: Function to call at due time
            direction: 0=uplink, 1=downlink (for tie-breaking)
            link_frame_id: Frame ID (for tie-breaking)
            copy_index: Copy index for duplicates (for tie-breaking)
            payload: Optional payload passed to callback
        """
        if delay_ns < 0:
            raise ValueError(f"Delay cannot be negative: {delay_ns}")

        due_time = self._current_time + delay_ns
        event = ScheduledEvent(
            due_time=due_time,
            direction=direction,
            link_frame_id=link_frame_id,
            copy_index=copy_index,
            callback=callback,
            payload=payload,
        )
        heapq.heappush(self._event_queue, event)

    def schedule_at(
        self,
        due_time: SimulationTime,
        callback: Callable[[], None],
        direction: int,
        link_frame_id: int,
        copy_index: int = 0,
        payload: Any = None,
    ) -> None:
        """Schedule event at absolute time."""
        if due_time < self._current_time:
            raise ValueError(
                f"Cannot schedule in past: {due_time.ns} < {self._current_time.ns}"
            )

        event = ScheduledEvent(
            due_time=due_time,
            direction=direction,
            link_frame_id=link_frame_id,
            copy_index=copy_index,
            callback=callback,
            payload=payload,
        )
        heapq.heappush(self._event_queue, event)

    def advance_to_next_event(self) -> bool:
        """Advance clock to next event and execute it.

        Returns:
            True if event was processed, False if queue empty
        """
        if not self._event_queue:
            return False

        event = heapq.heappop(self._event_queue)
        self._current_time = event.due_time
        event.callback()
        return True

    def run_until(self, end_time: SimulationTime) -> int:
        """Run all events until specified time.

        Returns:
            Number of events processed
        """
        count = 0
        while self._event_queue and self._event_queue[0].due_time <= end_time:
            self.advance_to_next_event()
            count += 1

        # Advance clock even if no events
        if end_time > self._current_time:
            self._current_time = end_time

        return count

    def run_until_idle(self, max_events: Optional[int] = None) -> int:
        """Run all pending events.

        Args:
            max_events: Optional limit on events processed

        Returns:
            Number of events processed
        """
        count = 0
        while self._event_queue:
            if max_events is not None and count >= max_events:
                break
            self.advance_to_next_event()
            count += 1
        return count

    def peek_next_event_time(self) -> Optional[SimulationTime]:
        """Get time of next event without advancing clock."""
        if not self._event_queue:
            return None
        return self._event_queue[0].due_time

    def has_pending_events(self) -> bool:
        """Check if any events are scheduled."""
        return len(self._event_queue) > 0

    def clear(self) -> None:
        """Clear all pending events."""
        self._event_queue.clear()

    def get_admission_order(self) -> int:
        """Get next admission order counter for ingress serialization.

        Section 9.2: Concurrent ingress has logged admission order.
        """
        order = self._admission_order
        self._admission_order += 1
        return order

    def reset(self, start_time: SimulationTime = None) -> None:
        """Reset clock and clear events."""
        self._current_time = start_time or SimulationTime(0)
        self._event_queue.clear()
        self._admission_order = 0
