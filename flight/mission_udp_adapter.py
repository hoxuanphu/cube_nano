"""Completion gate for the SpacePacketFramer -> TmFramer chain."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .mission_com_scheduler import MissionComScheduler, ScheduledItem


@dataclass
class CompletionGate:
    item_id: int
    status: str | None = None
    frame_returned: bool = False
    completed: bool = False

    def status_callback(self, status: str) -> None:
        if self.status is not None:
            return
        self.status = str(status)

    def return_callback(self) -> None:
        if self.frame_returned:
            return
        self.frame_returned = True


class MissionUdpAdapter:
    """Own one frame until both propagated status and upstream return arrive."""

    def __init__(self, scheduler: MissionComScheduler, send_frame: Callable[[bytes, Callable[[str], None], Callable[[], None]], None] | None = None):
        self.scheduler = scheduler
        self.send_frame = send_frame
        self.gate: CompletionGate | None = None
        self.completion_tuples: list[tuple[int, str, bool]] = []

    def send_next(self) -> ScheduledItem | None:
        item = self.scheduler.poll()
        if item is None:
            return None
        self.gate = CompletionGate(item.item_id)
        if self.send_frame is not None:
            self.send_frame(item.frame, self.receive_status, self.receive_return)
        return item

    def receive_status(self, status: str) -> None:
        if self.gate is None:
            return
        self.gate.status_callback(status)
        self._flush_gate()

    def receive_return(self) -> None:
        if self.gate is None:
            return
        self.gate.return_callback()
        self._flush_gate()

    def _flush_gate(self) -> None:
        gate = self.gate
        if gate is None or gate.completed or gate.status is None or not gate.frame_returned:
            return
        gate.completed = True
        status = gate.status
        # Clear the old ownership record before invoking scheduler callbacks;
        # a callback may synchronously enqueue/poll the next frame.
        self.gate = None
        self.completion_tuples.append((gate.item_id, status, gate.frame_returned))
        self.scheduler.mark_upstream_return(gate.item_id)
        self.scheduler.mark_status(gate.item_id, status)

    def reset(self) -> None:
        self.scheduler.set_not_ready()
        if self.gate is not None:
            self.gate.return_callback()
            self.gate.status_callback("SESSION_RESET")
            self._flush_gate()
