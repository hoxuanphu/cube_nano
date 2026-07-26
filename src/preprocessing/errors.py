"""Typed failures and state semantics for the preprocessing boundary."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping


class RunState(str, Enum):
    NEW = "NEW"
    VALIDATING = "VALIDATING"
    ADMITTED = "ADMITTED"
    PROCESSING = "PROCESSING"
    VERIFYING = "VERIFYING"
    COMPLETE = "COMPLETE"
    INVALID_INPUT = "INVALID_INPUT"
    UNTRUSTED_ARTIFACT = "UNTRUSTED_ARTIFACT"
    RESOURCE_REJECTED = "RESOURCE_REJECTED"
    CALIBRATION_ERROR = "CALIBRATION_ERROR"
    IO_FAULT = "IO_FAULT"
    RUNTIME_FAULT = "RUNTIME_FAULT"


class SafeAction(str, Enum):
    RETAIN_FOR_GROUND = "RETAIN_FOR_GROUND"


class FailureReason(str, Enum):
    INVALID_REQUEST = "INVALID_REQUEST"
    REQUEST_TYPE_INVALID = "REQUEST_TYPE_INVALID"
    CAPTURE_INCOMPLETE = "CAPTURE_INCOMPLETE"
    SOURCE_NOT_FOUND = "SOURCE_NOT_FOUND"
    SOURCE_FINGERPRINT_MISMATCH = "SOURCE_FINGERPRINT_MISMATCH"
    SCHEMA_MISMATCH = "SCHEMA_MISMATCH"
    PROFILE_INVALID = "PROFILE_INVALID"
    CALIBRATION_INVALID = "CALIBRATION_INVALID"
    CALIBRATION_UNSUPPORTED = "CALIBRATION_UNSUPPORTED"
    TRUST_REJECTED = "TRUST_REJECTED"
    SIGNATURE_INVALID = "SIGNATURE_INVALID"
    ISSUER_UNTRUSTED = "ISSUER_UNTRUSTED"
    KEY_UNTRUSTED = "KEY_UNTRUSTED"
    TRUST_EXPIRED = "TRUST_EXPIRED"
    TRUST_NOT_YET_VALID = "TRUST_NOT_YET_VALID"
    GENERATION_MISMATCH = "GENERATION_MISMATCH"
    FINGERPRINT_LINKAGE_MISMATCH = "FINGERPRINT_LINKAGE_MISMATCH"
    RESOURCE_PREFLIGHT = "RESOURCE_PREFLIGHT_REJECTED"
    RESOURCE_RUNTIME = "RESOURCE_RUNTIME_REJECTED"
    RESOURCE_OBSERVATION_UNAVAILABLE = "RESOURCE_OBSERVATION_UNAVAILABLE"
    CODEC_UNAVAILABLE = "CODEC_UNAVAILABLE"
    ARTIFACT_INCOMPLETE = "ARTIFACT_INCOMPLETE"
    ARTIFACT_CHECKSUM_MISMATCH = "ARTIFACT_CHECKSUM_MISMATCH"
    ARTIFACT_SCHEMA_INVALID = "ARTIFACT_SCHEMA_INVALID"
    IO_ERROR = "IO_ERROR"
    WARP_ERROR = "WARP_ERROR"
    NON_FINITE_OUTPUT = "NON_FINITE_OUTPUT"
    RUNTIME_FAULT = "RUNTIME_FAULT"
    PATCH_RESULT_INCOMPLETE = "PATCH_RESULT_INCOMPLETE"
    PATCH_RESULT_DUPLICATE = "PATCH_RESULT_DUPLICATE"
    PATCH_RESULT_MISSING = "PATCH_RESULT_MISSING"
    PATCH_RESULT_CHECKSUM_MISMATCH = "PATCH_RESULT_CHECKSUM_MISMATCH"
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"
    UNKNOWN = "UNKNOWN"


class PreprocessError(Exception):
    """Expected operational error that becomes a typed ``PreprocessFailure``."""

    def __init__(self, reason_code, message, *, state=None, provenance=None):
        self.reason_code = str(getattr(reason_code, "value", reason_code))
        self.state = state or RunState.RUNTIME_FAULT.value
        self.provenance = dict(provenance or {})
        super().__init__(message)


@dataclass(frozen=True)
class PreprocessFailure:
    """Fail-closed result returned for expected input, trust, resource, or I/O errors."""

    state: RunState | str
    reason_code: str
    safe_action: SafeAction | str = SafeAction.RETAIN_FOR_GROUND
    message: str = ""
    run_id: str | None = None
    provenance: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        state = RunState(self.state) if not isinstance(self.state, RunState) else self.state
        object.__setattr__(self, "state", state)
        action = (
            SafeAction(self.safe_action)
            if not isinstance(self.safe_action, SafeAction)
            else self.safe_action
        )
        object.__setattr__(self, "safe_action", action)
        object.__setattr__(self, "reason_code", str(getattr(self.reason_code, "value", self.reason_code)))
        object.__setattr__(self, "provenance", dict(self.provenance))

    @property
    def terminal_state(self):
        return self.state

    @property
    def is_failure(self):
        return True


_FAILURE_STATES = {
    RunState.INVALID_INPUT,
    RunState.UNTRUSTED_ARTIFACT,
    RunState.RESOURCE_REJECTED,
    RunState.CALIBRATION_ERROR,
    RunState.IO_FAULT,
    RunState.RUNTIME_FAULT,
}


class StateMachine:
    """Small explicit state machine used by both processing and artifact runs."""

    _TRANSITIONS = {
        RunState.NEW: {RunState.VALIDATING},
        RunState.VALIDATING: {RunState.ADMITTED, *_FAILURE_STATES},
        RunState.ADMITTED: {RunState.PROCESSING, *_FAILURE_STATES},
        RunState.PROCESSING: {RunState.VERIFYING, *_FAILURE_STATES},
        RunState.VERIFYING: {RunState.COMPLETE, *_FAILURE_STATES},
    }

    def __init__(self):
        self.state = RunState.NEW
        self.history = [self.state]

    def transition(self, state):
        state = RunState(state)
        allowed = self._TRANSITIONS.get(self.state, set())
        if state not in allowed:
            raise RuntimeError(f"Invalid preprocessing state transition {self.state.value} -> {state.value}")
        self.state = state
        self.history.append(state)
        return self.state

    @property
    def state_history(self):
        return tuple(self.history)
