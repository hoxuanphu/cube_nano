"""Bounded event replay and snapshot/resync contract for realtime clients."""

from __future__ import annotations

import json
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

from protocol.canonical import checked_u64, u64_to_json

from .events import EventRecord, EventStore
from .u64 import decode_u64_cursor


# RFC 6455 reserves 4000-4999 for application-defined close codes. Clients
# receiving this code must fetch a fresh snapshot before reconnecting.
RESYNC_CLOSE_CODE = 4009


class RealtimeError(RuntimeError):
    code = "REALTIME_ERROR"


class ResyncRequired(RealtimeError):
    code = "RESYNC_REQUIRED"


class SlowClient(RealtimeError):
    code = "SLOW_CLIENT"


@dataclass(frozen=True)
class SnapshotEnvelope:
    state: dict[str, Any]
    as_of_event_id: int
    last_event_id: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "as_of_event_id": u64_to_json(self.as_of_event_id),
            "last_event_id": u64_to_json(self.last_event_id),
        }


class RealtimeClient:
    def __init__(
        self,
        client_id: int,
        *,
        max_events: int,
        max_bytes: int,
        live_after_event_id: int = 0,
    ):
        self.client_id = client_id
        self.max_events = max_events
        self.max_bytes = max_bytes
        self.live_after_event_id = checked_u64(live_after_event_id, "live_after_event_id")
        self._queue: deque[dict[str, Any]] = deque()
        self._bytes = 0
        self.closed = False
        self.close_reason: str | None = None
        self.close_code: int | None = None
        self._terminal_envelope: dict[str, Any] | None = None

    @property
    def queue_depth(self) -> int:
        return len(self._queue)

    @property
    def queue_bytes(self) -> int:
        return self._bytes

    @property
    def resync_envelope(self) -> dict[str, Any] | None:
        """The control envelope that must be sent before a resync close."""

        return None if self._terminal_envelope is None else dict(self._terminal_envelope)

    def take_terminal_envelope(self) -> dict[str, Any] | None:
        """Return the pending control envelope once for websocket delivery."""

        envelope = self.resync_envelope
        self._terminal_envelope = None
        return envelope

    @staticmethod
    def _encoded_size(value: dict[str, Any]) -> int:
        return len(json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode("utf-8"))

    def require_resync(self, message: str) -> None:
        """Transition to an explicit, bounded resync-required terminal state."""

        self._queue.clear()
        self._bytes = 0
        self.closed = True
        self.close_reason = ResyncRequired.code
        self.close_code = RESYNC_CLOSE_CODE
        self._terminal_envelope = {
            "type": "error",
            "error": ResyncRequired.code,
            "message": message,
        }

    def enqueue(self, event: EventRecord | dict[str, Any]) -> None:
        if self.closed:
            return
        if isinstance(event, EventRecord) and event.event_id <= self.live_after_event_id:
            # The event belongs to the durable replay snapshot. A publisher
            # that was waiting for the hub lock must not duplicate it live.
            return
        value = event.as_dict() if isinstance(event, EventRecord) else dict(event)
        encoded_size = self._encoded_size(value)
        if len(self._queue) >= self.max_events or self._bytes + encoded_size > self.max_bytes:
            self.require_resync("realtime client exceeded its bounded replay buffer")
            raise SlowClient("realtime client exceeded its bounded replay buffer")
        self._queue.append(value)
        self._bytes += encoded_size

    def drain(self, limit: int = 100) -> tuple[dict[str, Any], ...]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        result = []
        while self._queue and len(result) < limit:
            value = self._queue.popleft()
            self._bytes -= self._encoded_size(value)
            result.append(value)
        return tuple(result)

    def close(self, reason: str = "CLIENT_CLOSED", *, code: int | None = None) -> None:
        self.closed = True
        self.close_reason = reason
        self.close_code = code
        self._queue.clear()
        self._bytes = 0
        if reason != ResyncRequired.code:
            self._terminal_envelope = None


class RealtimeHub:
    """Coordinate the snapshot/read cursor race under one publish lock."""

    def __init__(
        self,
        event_store: EventStore,
        state_provider: Callable[[], dict[str, Any]],
        *,
        max_client_events: int = 1_000,
        max_client_bytes: int = 4 * 1024 * 1024,
    ) -> None:
        if max_client_events <= 0 or max_client_bytes <= 0:
            raise ValueError("realtime client limits must be positive")
        self.event_store = event_store
        self.state_provider = state_provider
        self.max_client_events = max_client_events
        self.max_client_bytes = max_client_bytes
        self._lock = threading.RLock()
        self._next_client_id = 1
        self._clients: dict[int, RealtimeClient] = {}
        self._retention_floor = 1
        self.metrics = {
            "connected": 0,
            "disconnected": 0,
            "replayed": 0,
            "slow_clients": 0,
            "resync_required": 0,
            "published": 0,
        }

    def set_retention_floor(self, event_id: int) -> None:
        self._retention_floor = checked_u64(event_id, "retention_floor")

    def snapshot(self) -> SnapshotEnvelope:
        with self._lock:
            state = dict(self.state_provider())
            latest = self.event_store.latest_event_id()
            return SnapshotEnvelope(state, latest, latest)

    def connect(self, last_event_id: int | str | None = None) -> tuple[SnapshotEnvelope, RealtimeClient, tuple[dict[str, Any], ...]]:
        with self._lock:
            cursor = None
            if isinstance(last_event_id, str):
                cursor = decode_u64_cursor(last_event_id)
            elif last_event_id is not None:
                cursor = checked_u64(last_event_id, "last_event_id")
            if cursor is not None and cursor + 1 < self._retention_floor:
                raise ResyncRequired("requested realtime cursor is older than retention")
            state = dict(self.state_provider())
            as_of = self.event_store.latest_event_id()
            if cursor is not None and cursor > as_of:
                raise ResyncRequired("requested realtime cursor is ahead of the retained event stream")
            records, _ = self.event_store.list_events(after_event_id=cursor, limit=self.max_client_events)
            replay_values = tuple(record.as_dict() for record in records if record.event_id <= as_of)
            replay_bytes = sum(RealtimeClient._encoded_size(value) for value in replay_values)
            if (
                len(replay_values) >= self.max_client_events
                and records
                and records[-1].event_id < as_of
            ) or replay_bytes > self.max_client_bytes:
                raise ResyncRequired("replay exceeds the bounded client buffer")
            client = RealtimeClient(
                self._next_client_id,
                max_events=self.max_client_events,
                max_bytes=self.max_client_bytes,
                live_after_event_id=as_of,
            )
            self._next_client_id += 1
            # Replay is returned to the caller exactly once. Live events are
            # queued only after the client is registered, so replay is never
            # duplicated by a second drain of the live queue.
            self._clients[client.client_id] = client
            self.metrics["connected"] += 1
            self.metrics["replayed"] += len(replay_values)
            return SnapshotEnvelope(state, as_of, as_of), client, replay_values

    def publish(self, event: EventRecord) -> None:
        if not isinstance(event, EventRecord):
            raise TypeError("publish requires an EventRecord")
        with self._lock:
            self.metrics["published"] += 1
            for client_id, client in tuple(self._clients.items()):
                try:
                    client.enqueue(event)
                except SlowClient:
                    self.metrics["slow_clients"] += 1
                    self.metrics["resync_required"] += 1
                    self._clients.pop(client_id, None)
                    self.metrics["disconnected"] += 1

    def disconnect(self, client_id: int, reason: str = "CLIENT_CLOSED") -> None:
        with self._lock:
            client = self._clients.pop(client_id, None)
            if client is not None:
                client.close(reason)
                self.metrics["disconnected"] += 1

    def clients(self) -> tuple[RealtimeClient, ...]:
        with self._lock:
            return tuple(self._clients.values())
