"""Bounded single-in-flight scheduler with explicit ownership completion."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable


class SchedulerState(str, Enum):
    READY = "READY"
    IN_FLIGHT = "IN_FLIGHT"
    NOT_READY = "NOT_READY"


class QueueKind(str, Enum):
    ACK = "ACK"
    CONTROL = "CONTROL"
    FILE = "FILE"


@dataclass
class ScheduledItem:
    item_id: int
    kind: QueueKind
    frame: bytes
    accepted_at_ms: int
    completion: Callable[["ScheduledItem", str], None] | None = None
    ordering_key: int | None = None
    status: str | None = None
    upstream_returned: bool = False


class QueueOverflow(RuntimeError):
    code = "QUEUE_FULL"


class MissionComScheduler:
    """Select packets fairly while never releasing a buffer twice."""

    def __init__(
        self,
        *,
        capacities: dict[QueueKind, int] | None = None,
        ack_burst: int = 8,
        control_burst: int = 4,
        file_burst: int = 8,
    ):
        self.capacities = capacities or {
            QueueKind.ACK: 32,
            QueueKind.CONTROL: 64,
            QueueKind.FILE: 16,
        }
        self.bursts = {
            QueueKind.ACK: ack_burst,
            QueueKind.CONTROL: control_burst,
            QueueKind.FILE: file_burst,
        }
        if any(value <= 0 for value in self.capacities.values()) or any(value <= 0 for value in self.bursts.values()):
            raise ValueError("scheduler capacities and bursts must be positive")
        self.queues = {kind: deque() for kind in QueueKind}
        self.state = SchedulerState.READY
        self.current: ScheduledItem | None = None
        self._next_item_id = 1
        self._ack_run = 0
        self._control_run = 0
        self._ack_relief_stage = 0
        self.metrics = {
            "accepted": {kind.value: 0 for kind in QueueKind},
            "rejected": {kind.value: 0 for kind in QueueKind},
            "completed": {kind.value: 0 for kind in QueueKind},
            "duplicate_callbacks": 0,
            "oldest_ack_age_ms": 0,
            "oldest_control_age_ms": 0,
        }
        self.not_ready_after_completion = False

    def enqueue(
        self,
        kind: QueueKind,
        frame: bytes,
        completion: Callable[[ScheduledItem, str], None] | None = None,
        *,
        ordering_key: int | None = None,
    ) -> int:
        frame = bytes(frame)
        if ordering_key is not None and (
            isinstance(ordering_key, bool) or not isinstance(ordering_key, int) or ordering_key < 0
        ):
            raise ValueError("ordering_key must be a non-negative integer when provided")
        if len(self.queues[kind]) >= self.capacities[kind]:
            self.metrics["rejected"][kind.value] += 1
            raise QueueOverflow(f"{kind.value} scheduler queue is full")
        item_id = self._next_item_id
        self._next_item_id += 1
        self.queues[kind].append(
            ScheduledItem(
                item_id,
                kind,
                frame,
                int(time.monotonic() * 1000),
                completion,
                ordering_key,
            )
        )
        self.metrics["accepted"][kind.value] += 1
        return item_id

    def enqueue_ack(self, frame: bytes, completion=None, *, ordering_key: int | None = None) -> int:
        return self.enqueue(QueueKind.ACK, frame, completion, ordering_key=ordering_key)

    def enqueue_control(self, frame: bytes, completion=None, *, ordering_key: int | None = None) -> int:
        return self.enqueue(QueueKind.CONTROL, frame, completion, ordering_key=ordering_key)

    def enqueue_file(self, frame: bytes, completion=None, *, ordering_key: int | None = None) -> int:
        return self.enqueue(QueueKind.FILE, frame, completion, ordering_key=ordering_key)

    def _ordered_kind(self) -> QueueKind | None:
        """Keep globally counted TM frames in their allocated wire order.

        ACK priority is useful for ordinary traffic, but a later APID 2 frame
        cannot overtake an already allocated APID 3 frame on the same TM
        channel.  Callers opt into this rule with an ``ordering_key`` derived
        from the durable MCFC allocation.
        """

        heads = [(kind, queue[0]) for kind, queue in self.queues.items() if queue]
        if not heads or any(item.ordering_key is None for _, item in heads):
            return None
        return min(heads, key=lambda pair: (pair[1].ordering_key, pair[1].item_id))[0]

    def _select_kind(self) -> QueueKind | None:
        has_ack = bool(self.queues[QueueKind.ACK])
        has_control = bool(self.queues[QueueKind.CONTROL])
        has_file = bool(self.queues[QueueKind.FILE])
        if not (has_ack or has_control or has_file):
            return None
        if has_ack:
            if self._ack_run >= self.bursts[QueueKind.ACK] and (has_control or has_file):
                if self._ack_relief_stage == 0:
                    self._ack_relief_stage = 1
                if self._ack_relief_stage == 1:
                    self._ack_relief_stage = 2
                    if has_control:
                        return QueueKind.CONTROL
                if self._ack_relief_stage == 2:
                    self._ack_relief_stage = 0
                    self._ack_run = 0
                    if has_file:
                        return QueueKind.FILE
                self._ack_relief_stage = 0
                self._ack_run = 0
            self._ack_run += 1
            self._control_run = 0
            return QueueKind.ACK
        self._ack_run = 0
        self._ack_relief_stage = 0
        if has_control:
            if self._control_run < self.bursts[QueueKind.CONTROL] or not has_file:
                self._control_run += 1
                return QueueKind.CONTROL
            self._control_run = 0
            return QueueKind.FILE
        self._control_run = 0
        return QueueKind.FILE

    def poll(self) -> ScheduledItem | None:
        if self.state != SchedulerState.READY or self.current is not None:
            return None
        kind = self._ordered_kind()
        if kind is None:
            kind = self._select_kind()
        if kind is None:
            return None
        item = self.queues[kind].popleft()
        self.current = item
        self.state = SchedulerState.IN_FLIGHT
        now_ms = int(time.monotonic() * 1000)
        if item.kind == QueueKind.ACK:
            self.metrics["oldest_ack_age_ms"] = max(
                self.metrics["oldest_ack_age_ms"], now_ms - item.accepted_at_ms
            )
        elif item.kind == QueueKind.CONTROL:
            self.metrics["oldest_control_age_ms"] = max(
                self.metrics["oldest_control_age_ms"], now_ms - item.accepted_at_ms
            )
        if self.queues[QueueKind.ACK]:
            self.metrics["oldest_ack_age_ms"] = max(
                self.metrics["oldest_ack_age_ms"],
                now_ms - self.queues[QueueKind.ACK][0].accepted_at_ms,
            )
        if self.queues[QueueKind.CONTROL]:
            self.metrics["oldest_control_age_ms"] = max(
                self.metrics["oldest_control_age_ms"],
                now_ms - self.queues[QueueKind.CONTROL][0].accepted_at_ms,
            )
        return item

    def mark_status(self, item_id: int, status: str) -> bool:
        if self.current is None or self.current.item_id != item_id:
            self.metrics["duplicate_callbacks"] += 1
            return False
        if self.current.status is not None:
            self.metrics["duplicate_callbacks"] += 1
            return False
        self.current.status = str(status)
        return self._try_complete()

    def mark_upstream_return(self, item_id: int) -> bool:
        if self.current is None or self.current.item_id != item_id:
            self.metrics["duplicate_callbacks"] += 1
            return False
        if self.current.upstream_returned:
            self.metrics["duplicate_callbacks"] += 1
            return False
        self.current.upstream_returned = True
        return self._try_complete()

    def _try_complete(self) -> bool:
        item = self.current
        if item is None or item.status is None or not item.upstream_returned:
            return False
        status = item.status
        self.current = None
        self.state = SchedulerState.NOT_READY if self.not_ready_after_completion else SchedulerState.READY
        self.not_ready_after_completion = False
        self.metrics["completed"][item.kind.value] += 1
        if item.completion is not None:
            item.completion(item, status)
        return True

    def set_not_ready(self) -> None:
        if self.current is not None:
            self.not_ready_after_completion = True
        else:
            self.state = SchedulerState.NOT_READY

    def set_ready(self) -> None:
        if self.current is None:
            self.state = SchedulerState.READY

    def queue_depths(self) -> dict[str, int]:
        return {kind.value: len(self.queues[kind]) for kind in QueueKind}
