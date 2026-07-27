"""Blackout and contact window scheduling.

Section 9.2: Contact windows, blackout periods, and frame admission policy.
Frames arriving during blackout are dropped according to policy.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from .virtual_clock import SimulationTime


class ContactState(Enum):
    """Contact state for link."""
    CONTACT_OPEN = "CONTACT_OPEN"
    NO_CONTACT = "NO_CONTACT"
    BLACKOUT = "BLACKOUT"


@dataclass
class ContactWindow:
    """Contact window definition."""
    start_time: SimulationTime
    end_time: SimulationTime
    state: ContactState

    def contains(self, time: SimulationTime) -> bool:
        """Check if time is within this window."""
        return self.start_time <= time < self.end_time


class ContactSchedule:
    """Contact window and blackout scheduler.

    Section 9.2: Blackout admission drops frames according to policy.
    NO_CONTACT pauses immediate commands, BLACKOUT drops frames.
    """

    def __init__(self, bandwidth_profile: Optional['BandwidthProfile'] = None):
        """Initialize schedule.

        Args:
            bandwidth_profile: Optional bandwidth shaping profile
        """
        self._windows: List[ContactWindow] = []
        self._default_state = ContactState.CONTACT_OPEN
        self._bandwidth_profile = bandwidth_profile

    def add_window(self, start_time: SimulationTime, end_time: SimulationTime, state: ContactState) -> None:
        """Add contact window to schedule.

        Windows must not overlap. Sort by start time.
        """
        if end_time <= start_time:
            raise ValueError(f"Invalid window: {end_time} <= {start_time}")

        # Check for overlaps
        for window in self._windows:
            if not (end_time <= window.start_time or start_time >= window.end_time):
                raise ValueError(f"Window overlaps existing: {start_time}-{end_time}")

        window = ContactWindow(start_time=start_time, end_time=end_time, state=state)
        self._windows.append(window)
        self._windows.sort(key=lambda w: w.start_time.ns)

    def get_state_at(self, time: SimulationTime) -> ContactState:
        """Get contact state at given time."""
        for window in self._windows:
            if window.contains(time):
                return window.state
        return self._default_state

    def get_next_state_change(self, current_time: SimulationTime) -> Optional[SimulationTime]:
        """Get time of next state change after current time."""
        for window in self._windows:
            if window.start_time > current_time:
                return window.start_time
            if window.end_time > current_time:
                return window.end_time
        return None

    def is_admission_allowed(self, time: SimulationTime) -> bool:
        """Check if frame admission is allowed at given time.

        Section 9.2: Frames during BLACKOUT are dropped.
        NO_CONTACT does not drop frames at link layer, but GDS may reject commands.
        """
        state = self.get_state_at(time)
        return state != ContactState.BLACKOUT

    def clear(self) -> None:
        """Clear all scheduled windows."""
        self._windows.clear()

    def should_drop_frame(self, time: SimulationTime) -> bool:
        """Check if frame should be dropped at given time.

        Section 9.2: Frames during BLACKOUT are dropped.
        """
        return self.get_state_at(time) == ContactState.BLACKOUT

    def get_bandwidth_budget(self, frame_bits: int) -> int:
        """Get transmission duration for frame in nanoseconds.

        Returns:
            Transmission duration in ns, or 0 if no bandwidth limit
        """
        if self._bandwidth_profile:
            return self._bandwidth_profile.compute_tx_duration_ns(frame_bits)
        return 0


@dataclass
class BandwidthProfile:
    """Bandwidth shaping profile.

    Section 9.2: Strict serializer, not token bucket.
    """
    bitrate_bps: int  # Bits per second

    def validate(self) -> None:
        """Validate bandwidth profile."""
        if self.bitrate_bps <= 0:
            raise ValueError(f"bitrate_bps must be > 0: {self.bitrate_bps}")

    def compute_tx_duration_ns(self, frame_bits: int) -> int:
        """Compute transmission duration in nanoseconds.

        Section 9.2: ceil(frame_bits * 1e9 / bitrate_bps) using U128.
        """
        # Python: (frame_bits * 10**9 + bitrate_bps - 1) // bitrate_bps
        return (frame_bits * 10**9 + self.bitrate_bps - 1) // self.bitrate_bps
