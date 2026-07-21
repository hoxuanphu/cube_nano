"""Explicit allow-lists for command, job, product and transfer states."""

from __future__ import annotations

from collections import defaultdict

TRANSITIONS = {
    "command": {
        "RECEIVED": {"VALIDATED", "EXECUTION_FAILED"},
        "VALIDATED": {"COMMAND_REJECTED", "COMMAND_ACCEPTED", "EXECUTION_FAILED"},
        "COMMAND_ACCEPTED": {"DISPATCHED", "EXECUTED", "EXECUTION_FAILED"},
    },
    "job": {
        "QUEUED": {"RUNNING", "CANCEL_REQUESTED", "FAILED", "TIMEOUT"},
        "RUNNING": {"SUCCEEDED", "FAILED", "TIMEOUT", "CANCEL_REQUESTED"},
        "CANCEL_REQUESTED": {"CANCELED", "SUCCEEDED", "FAILED", "TIMEOUT"},
    },
    "product": {"STAGING": {"READY", "FAILED"}},
    "transfer": {
        "QUEUED": {"SENDING", "ABORTING", "CANCELED"},
        "SENDING": {"SEND_COMPLETED", "CANCEL_REQUESTED", "ABORTING"},
        "CANCEL_REQUESTED": {"CANCEL_DRAINING", "SEND_COMPLETED", "ABORTING"},
        "CANCEL_DRAINING": {"COOLDOWN", "ABORTING"},
        "ABORTING": {"COOLDOWN"},
        "COOLDOWN": {"SEND_FAILED", "CANCELED"},
    },
}


class InvalidTransition(RuntimeError):
    pass


class StateMachine:
    def __init__(self):
        self.states = {}
        self.history = defaultdict(list)

    def register(self, entity_id: str, kind: str, state: str) -> None:
        if kind not in TRANSITIONS:
            raise ValueError(f"unknown state machine kind {kind}")
        self.states[entity_id] = (kind, state)
        self.history[entity_id].append(state)

    def transition(self, entity_id: str, state: str, reason: str | None = None) -> None:
        kind, current = self.states[entity_id]
        if current not in TRANSITIONS[kind] or state not in TRANSITIONS[kind][current]:
            raise InvalidTransition(f"{kind} cannot transition {current} -> {state}")
        self.states[entity_id] = (kind, state)
        self.history[entity_id].append((state, reason))

    def state(self, entity_id: str) -> str:
        return self.states[entity_id][1]
