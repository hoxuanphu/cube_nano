"""Versioned, fail-closed contract between CloudPayload and the AI worker."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from protocol.canonical import canonical_json, checked_u32, checked_u64
from protocol.schemas import RequestKey

WORKER_API_VERSION = 1
WORKER_VERSION = "sat-ai-worker-v1"


class WorkerProtocolError(ValueError):
    code = "WORKER_PROTOCOL_ERROR"


class WorkerMessageType(str, Enum):
    JOB = "JOB"
    CONTROL = "CONTROL"
    RESULT = "RESULT"
    HEARTBEAT = "HEARTBEAT"


class WorkerControlAction(str, Enum):
    CANCEL = "CANCEL"
    SHUTDOWN = "SHUTDOWN"


class WorkerResultState(str, Enum):
    SUCCEEDED = "SUCCEEDED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"
    CANCELED = "CANCELED"
    TIMEOUT = "TIMEOUT"


WORKER_ERROR_CODES = frozenset(
    {
        "DEADLINE_EXCEEDED",
        "INSUFFICIENT_VALID_DATA",
        "INVALID_WORKER_REQUEST",
        "QUEUE_FULL",
        "SERVICE_FAULT",
        "WORKER_CANCELED",
        "WORKER_LOST",
        "WORKER_PROTOCOL_ERROR",
        "WORKER_RESOURCE_EXHAUSTED",
        "WORKER_TIMEOUT",
        "WORKER_UNEXPECTED_ERROR",
    }
)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise WorkerProtocolError(f"{label} must be an object")
    return value


def _decode(encoded: bytes) -> Mapping[str, Any]:
    if not isinstance(encoded, (bytes, bytearray, memoryview)):
        raise WorkerProtocolError("worker message must be bytes")
    try:
        value = json.loads(bytes(encoded).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorkerProtocolError("worker message is not valid UTF-8 JSON") from exc
    data = _mapping(value, "worker message")
    if data.get("api_version") != WORKER_API_VERSION:
        raise WorkerProtocolError("unsupported worker api_version")
    return data


def _require_type(data: Mapping[str, Any], expected: WorkerMessageType) -> None:
    if data.get("message_type") != expected.value:
        raise WorkerProtocolError(f"expected worker message_type {expected.value}")


@dataclass(frozen=True)
class DeadlineContract:
    admitted_monotonic_ns: int
    deadline_monotonic_ns: int

    def __post_init__(self) -> None:
        checked_u64(self.admitted_monotonic_ns, "admitted_monotonic_ns")
        checked_u64(self.deadline_monotonic_ns, "deadline_monotonic_ns")
        if self.deadline_monotonic_ns <= self.admitted_monotonic_ns:
            raise WorkerProtocolError("worker deadline must be after admission")

    @classmethod
    def after_ms(cls, timeout_ms: int, *, now_ns: int | None = None) -> "DeadlineContract":
        timeout_ms = checked_u32(timeout_ms, "timeout_ms")
        if timeout_ms == 0:
            raise WorkerProtocolError("worker timeout must be positive")
        admitted = time.monotonic_ns() if now_ns is None else checked_u64(now_ns, "now_ns")
        deadline = admitted + timeout_ms * 1_000_000
        checked_u64(deadline, "deadline_monotonic_ns")
        return cls(admitted, deadline)

    def expired(self, *, now_ns: int | None = None) -> bool:
        now = time.monotonic_ns() if now_ns is None else checked_u64(now_ns, "now_ns")
        return now >= self.deadline_monotonic_ns

    def remaining_ms(self, *, now_ns: int | None = None) -> int:
        now = time.monotonic_ns() if now_ns is None else checked_u64(now_ns, "now_ns")
        return max(0, (self.deadline_monotonic_ns - now) // 1_000_000)

    def as_dict(self) -> dict[str, int]:
        return {
            "admitted_monotonic_ns": self.admitted_monotonic_ns,
            "deadline_monotonic_ns": self.deadline_monotonic_ns,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "DeadlineContract":
        data = _mapping(value, "deadline")
        return cls(
            checked_u64(data.get("admitted_monotonic_ns"), "admitted_monotonic_ns"),
            checked_u64(data.get("deadline_monotonic_ns"), "deadline_monotonic_ns"),
        )


@dataclass(frozen=True)
class WorkerRequest:
    request_key: RequestKey
    job_snapshot: dict[str, Any]
    deadline: DeadlineContract

    def __post_init__(self) -> None:
        if not isinstance(self.job_snapshot, dict):
            raise WorkerProtocolError("job_snapshot must be an object")

    def encode(self) -> bytes:
        return canonical_json(
            {
                "api_version": WORKER_API_VERSION,
                "message_type": WorkerMessageType.JOB.value,
                "request_key": self.request_key.as_dict(),
                "job_snapshot": self.job_snapshot,
                "deadline": self.deadline.as_dict(),
            }
        )

    @classmethod
    def decode(cls, encoded: bytes) -> "WorkerRequest":
        data = _decode(encoded)
        _require_type(data, WorkerMessageType.JOB)
        snapshot = _mapping(data.get("job_snapshot"), "job_snapshot")
        return cls(
            RequestKey.from_dict(data.get("request_key")),
            dict(snapshot),
            DeadlineContract.from_dict(data.get("deadline")),
        )


@dataclass(frozen=True)
class WorkerControl:
    action: WorkerControlAction
    request_key: RequestKey | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.action == WorkerControlAction.CANCEL and self.request_key is None:
            raise WorkerProtocolError("CANCEL requires request_key")
        if self.action == WorkerControlAction.SHUTDOWN and self.request_key is not None:
            raise WorkerProtocolError("SHUTDOWN must not carry request_key")

    def encode(self) -> bytes:
        return canonical_json(
            {
                "api_version": WORKER_API_VERSION,
                "message_type": WorkerMessageType.CONTROL.value,
                "action": self.action.value,
                "request_key": self.request_key.as_dict() if self.request_key else None,
                "reason": self.reason,
            }
        )

    @classmethod
    def decode(cls, encoded: bytes) -> "WorkerControl":
        data = _decode(encoded)
        _require_type(data, WorkerMessageType.CONTROL)
        try:
            action = WorkerControlAction(str(data.get("action")))
        except ValueError as exc:
            raise WorkerProtocolError("invalid worker control action") from exc
        request_value = data.get("request_key")
        request_key = None if request_value is None else RequestKey.from_dict(request_value)
        reason = data.get("reason")
        if reason is not None and not isinstance(reason, str):
            raise WorkerProtocolError("worker control reason must be a string or null")
        return cls(action, request_key, reason)


@dataclass(frozen=True)
class WorkerResult:
    request_key: RequestKey
    state: WorkerResultState
    result: dict[str, Any] | None = None
    error_code: str | None = None

    def __post_init__(self) -> None:
        if self.result is not None and not isinstance(self.result, dict):
            raise WorkerProtocolError("worker result must be an object or null")
        if self.error_code is not None and self.error_code not in WORKER_ERROR_CODES:
            raise WorkerProtocolError(f"unknown worker error code {self.error_code}")
        if self.state in {WorkerResultState.FAILED, WorkerResultState.CANCELED, WorkerResultState.TIMEOUT} and self.error_code is None:
            raise WorkerProtocolError(f"{self.state.value} worker result requires error_code")

    def encode(self) -> bytes:
        return canonical_json(
            {
                "api_version": WORKER_API_VERSION,
                "message_type": WorkerMessageType.RESULT.value,
                "request_key": self.request_key.as_dict(),
                "state": self.state.value,
                "result": self.result,
                "error_code": self.error_code,
            }
        )

    @classmethod
    def decode(cls, encoded: bytes) -> "WorkerResult":
        data = _decode(encoded)
        _require_type(data, WorkerMessageType.RESULT)
        try:
            state = WorkerResultState(str(data.get("state")))
        except ValueError as exc:
            raise WorkerProtocolError("invalid worker result state") from exc
        result_value = data.get("result")
        result = None if result_value is None else dict(_mapping(result_value, "result"))
        error_code = data.get("error_code")
        if error_code is not None and not isinstance(error_code, str):
            raise WorkerProtocolError("worker error_code must be a string or null")
        return cls(RequestKey.from_dict(data.get("request_key")), state, result, error_code)


@dataclass(frozen=True)
class WorkerHeartbeatMessage:
    worker_version: str
    sequence: int
    worker_state: str
    emitted_monotonic_ns: int
    active_request_key: RequestKey | None = None

    def __post_init__(self) -> None:
        if self.worker_version != WORKER_VERSION:
            raise WorkerProtocolError("unsupported worker version")
        checked_u32(self.sequence, "heartbeat sequence")
        checked_u64(self.emitted_monotonic_ns, "heartbeat timestamp")
        if self.worker_state not in {"STARTING", "READY", "RUNNING", "STOPPING"}:
            raise WorkerProtocolError("invalid worker heartbeat state")

    def encode(self) -> bytes:
        return canonical_json(
            {
                "api_version": WORKER_API_VERSION,
                "message_type": WorkerMessageType.HEARTBEAT.value,
                "worker_version": self.worker_version,
                "sequence": self.sequence,
                "worker_state": self.worker_state,
                "emitted_monotonic_ns": self.emitted_monotonic_ns,
                "active_request_key": self.active_request_key.as_dict() if self.active_request_key else None,
            }
        )

    @classmethod
    def decode(cls, encoded: bytes) -> "WorkerHeartbeatMessage":
        data = _decode(encoded)
        _require_type(data, WorkerMessageType.HEARTBEAT)
        active_value = data.get("active_request_key")
        return cls(
            str(data.get("worker_version")),
            checked_u32(data.get("sequence"), "heartbeat sequence"),
            str(data.get("worker_state")),
            checked_u64(data.get("emitted_monotonic_ns"), "heartbeat timestamp"),
            None if active_value is None else RequestKey.from_dict(active_value),
        )


@dataclass
class WorkerHeartbeat:
    """Supervisor-side observation of the latest validated heartbeat."""

    worker_version: str = WORKER_VERSION
    last_seen_ns: int = 0
    state: str = "STARTING"
    sequence: int | None = None

    def touch(self, message: WorkerHeartbeatMessage | None = None) -> None:
        if message is not None:
            if message.worker_version != self.worker_version:
                raise WorkerProtocolError("worker heartbeat version mismatch")
            if self.sequence is not None and message.sequence <= self.sequence:
                raise WorkerProtocolError("worker heartbeat sequence did not advance")
            self.sequence = message.sequence
            self.state = message.worker_state
        self.last_seen_ns = time.monotonic_ns()

    def is_alive(self, timeout_ms: int = 2000) -> bool:
        if timeout_ms <= 0:
            raise ValueError("heartbeat timeout must be positive")
        return self.last_seen_ns > 0 and time.monotonic_ns() - self.last_seen_ns <= timeout_ms * 1_000_000


ERROR_MAP = {
    "Full": "QUEUE_FULL",
    "TimeoutError": "WORKER_TIMEOUT",
    "DeadlineExceeded": "DEADLINE_EXCEEDED",
    "WorkerCanceled": "WORKER_CANCELED",
    "WorkerLost": "WORKER_LOST",
    "MemoryError": "WORKER_RESOURCE_EXHAUSTED",
    "WorkerProtocolError": "WORKER_PROTOCOL_ERROR",
}


def map_worker_error(error: BaseException) -> str:
    return ERROR_MAP.get(type(error).__name__, "WORKER_UNEXPECTED_ERROR")


def decode_worker_message(encoded: bytes) -> WorkerRequest | WorkerControl | WorkerResult | WorkerHeartbeatMessage:
    data = _decode(encoded)
    try:
        message_type = WorkerMessageType(str(data.get("message_type")))
    except ValueError as exc:
        raise WorkerProtocolError("invalid worker message_type") from exc
    decoders = {
        WorkerMessageType.JOB: WorkerRequest.decode,
        WorkerMessageType.CONTROL: WorkerControl.decode,
        WorkerMessageType.RESULT: WorkerResult.decode,
        WorkerMessageType.HEARTBEAT: WorkerHeartbeatMessage.decode,
    }
    return decoders[message_type](encoded)
