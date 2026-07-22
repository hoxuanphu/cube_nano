"""Crash-safe simulation run manifests and replay availability state."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping

from protocol.canonical import canonical_json, checked_u64, u64_to_json


class RunManifestError(RuntimeError):
    """Raised when a run manifest violates its lifecycle contract."""


class RunState(StrEnum):
    OPEN = "OPEN"
    FINAL = "FINAL"
    INCOMPLETE_CRASH = "INCOMPLETE_CRASH"
    INCOMPLETE_STORAGE = "INCOMPLETE_STORAGE"


class ReplayAvailability(StrEnum):
    PRESENT = "PRESENT"
    PINNED = "PINNED"
    EVICTED = "EVICTED"


def _utc_timestamp(epoch_seconds: int) -> str:
    if epoch_seconds < 0:
        raise ValueError("SOURCE_DATE_EPOCH must be non-negative")
    return datetime.fromtimestamp(epoch_seconds, UTC).isoformat().replace("+00:00", "Z")


def _hex_sha256(value: str | None, label: str) -> str | None:
    if value is None:
        return None
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise RunManifestError(f"{label} must be a lowercase SHA-256 hex digest")
    return value


def _normalize_u64(value: int | str, label: str) -> str:
    if isinstance(value, str):
        if len(value) != 16 or any(char not in "0123456789abcdef" for char in value):
            raise RunManifestError(f"{label} must be a 16-digit lowercase U64")
        checked_u64(int(value, 16), label)
        return value
    return u64_to_json(checked_u64(value, label))


@dataclass(frozen=True)
class ReplayRecord:
    state: ReplayAvailability
    sha256: str | None = None
    size_bytes: int | None = None
    revision: str | None = None

    def __post_init__(self) -> None:
        _hex_sha256(self.sha256, "replay sha256")
        if self.size_bytes is not None and self.size_bytes < 0:
            raise RunManifestError("replay size_bytes must be non-negative")
        if self.state in {ReplayAvailability.PRESENT, ReplayAvailability.PINNED}:
            if self.sha256 is None or self.size_bytes is None or not self.revision:
                raise RunManifestError(
                    "PRESENT/PINNED replay records require sha256, size_bytes and revision"
                )
        if self.state == ReplayAvailability.EVICTED and self.sha256 is not None:
            raise RunManifestError("EVICTED replay records must not claim replay bytes")

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "revision": self.revision,
        }


@dataclass(frozen=True)
class SimulationRunManifest:
    schema_version: int
    simulation_run_id: str
    state: RunState
    opened_at: str
    finalized_at: str | None
    release_id: str
    spacecraft_instance_id: str
    scoped_scene_ref: dict[str, Any]
    source_snapshot: dict[str, Any]
    config_revision: str
    model_revision: str
    deployment_profile_revision: str
    fault_profile_revision: str
    seed: str
    clock: dict[str, Any]
    command_set_sha256: str | None
    command_count: int
    replay: ReplayRecord
    incomplete_reason: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise RunManifestError("unsupported run manifest schema_version")
        _normalize_u64(self.simulation_run_id, "simulation_run_id")
        _normalize_u64(self.spacecraft_instance_id, "spacecraft_instance_id")
        _normalize_u64(self.seed, "seed")
        if not self.release_id or not self.config_revision or not self.model_revision:
            raise RunManifestError("release/config/model revisions are required")
        if self.command_count < 0:
            raise RunManifestError("command_count must be non-negative")
        _hex_sha256(self.command_set_sha256, "command_set_sha256")
        if self.state == RunState.FINAL:
            if self.finalized_at is None or self.command_set_sha256 is None:
                raise RunManifestError("FINAL run requires finalized_at and command_set_sha256")
            if self.incomplete_reason is not None:
                raise RunManifestError("FINAL run must not contain incomplete_reason")
        elif self.state == RunState.OPEN:
            if self.finalized_at is not None or self.incomplete_reason is not None:
                raise RunManifestError("OPEN run must not be finalized or incomplete")
        elif self.state in {RunState.INCOMPLETE_CRASH, RunState.INCOMPLETE_STORAGE}:
            if self.finalized_at is None or not self.incomplete_reason:
                raise RunManifestError("incomplete run requires finalized_at and a reason")

    @classmethod
    def open(
        cls,
        *,
        simulation_run_id: int | str,
        release_id: str,
        spacecraft_instance_id: int | str,
        scoped_scene_ref: Mapping[str, Any],
        source_snapshot: Mapping[str, Any],
        config_revision: str,
        model_revision: str,
        deployment_profile_revision: str,
        fault_profile_revision: str,
        seed: int | str,
        clock: Mapping[str, Any],
        opened_at: str | None = None,
    ) -> "SimulationRunManifest":
        timestamp = opened_at or _utc_timestamp(int(os.environ.get("SOURCE_DATE_EPOCH", "0")))
        return cls(
            schema_version=1,
            simulation_run_id=_normalize_u64(simulation_run_id, "simulation_run_id"),
            state=RunState.OPEN,
            opened_at=timestamp,
            finalized_at=None,
            release_id=release_id,
            spacecraft_instance_id=_normalize_u64(spacecraft_instance_id, "spacecraft_instance_id"),
            scoped_scene_ref=dict(scoped_scene_ref),
            source_snapshot=dict(source_snapshot),
            config_revision=config_revision,
            model_revision=model_revision,
            deployment_profile_revision=deployment_profile_revision,
            fault_profile_revision=fault_profile_revision,
            seed=_normalize_u64(seed, "seed"),
            clock=dict(clock),
            command_set_sha256=None,
            command_count=0,
            replay=ReplayRecord(ReplayAvailability.EVICTED, revision="open-v1"),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "simulation_run_id": self.simulation_run_id,
            "state": self.state.value,
            "opened_at": self.opened_at,
            "finalized_at": self.finalized_at,
            "release_id": self.release_id,
            "spacecraft_instance_id": self.spacecraft_instance_id,
            "scoped_scene_ref": self.scoped_scene_ref,
            "source_snapshot": self.source_snapshot,
            "config_revision": self.config_revision,
            "model_revision": self.model_revision,
            "deployment_profile_revision": self.deployment_profile_revision,
            "fault_profile_revision": self.fault_profile_revision,
            "seed": self.seed,
            "clock": self.clock,
            "command_set_sha256": self.command_set_sha256,
            "command_count": self.command_count,
            "replay": self.replay.as_dict(),
            "incomplete_reason": self.incomplete_reason,
        }

    def command_set_digest(self, commands: list[Mapping[str, Any]]) -> str:
        return hashlib.sha256(canonical_json([dict(command) for command in commands])).hexdigest()

    def finalize(
        self,
        *,
        commands: list[Mapping[str, Any]],
        replay_sha256: str,
        replay_size_bytes: int,
        replay_revision: str,
        replay_state: ReplayAvailability = ReplayAvailability.PRESENT,
        finalized_at: str | None = None,
    ) -> "SimulationRunManifest":
        if self.state != RunState.OPEN:
            raise RunManifestError("only OPEN runs may be finalized")
        return replace(
            self,
            state=RunState.FINAL,
            finalized_at=finalized_at or self.opened_at,
            command_set_sha256=self.command_set_digest(commands),
            command_count=len(commands),
            replay=ReplayRecord(replay_state, replay_sha256, replay_size_bytes, replay_revision),
        )

    def mark_incomplete(self, reason: str, *, storage: bool = False, finalized_at: str | None = None) -> "SimulationRunManifest":
        if self.state != RunState.OPEN:
            raise RunManifestError("only OPEN runs may become incomplete")
        if reason not in {"CRASH", "STORAGE_CAP"}:
            raise RunManifestError("incomplete reason must be CRASH or STORAGE_CAP")
        return replace(
            self,
            state=RunState.INCOMPLETE_STORAGE if storage else RunState.INCOMPLETE_CRASH,
            finalized_at=finalized_at or self.opened_at,
            replay=ReplayRecord(ReplayAvailability.EVICTED, revision="incomplete-v1"),
            incomplete_reason=reason,
        )

    def set_replay_state(self, state: ReplayAvailability) -> "SimulationRunManifest":
        if self.state != RunState.FINAL:
            raise RunManifestError("replay retention state is only mutable for FINAL runs")
        if state == ReplayAvailability.EVICTED:
            replay = ReplayRecord(state, revision=self.replay.revision)
        else:
            replay = ReplayRecord(state, self.replay.sha256, self.replay.size_bytes, self.replay.revision)
        return replace(self, replay=replay)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SimulationRunManifest":
        if not isinstance(value, Mapping):
            raise RunManifestError("run manifest must be an object")
        replay_value = value.get("replay")
        if not isinstance(replay_value, Mapping):
            raise RunManifestError("run manifest replay must be an object")
        try:
            state = RunState(str(value.get("state")))
            replay_state = ReplayAvailability(str(replay_value.get("state")))
        except ValueError as exc:
            raise RunManifestError("unknown run/replay state") from exc
        return cls(
            schema_version=int(value.get("schema_version", 0)),
            simulation_run_id=str(value.get("simulation_run_id", "")),
            state=state,
            opened_at=str(value.get("opened_at", "")),
            finalized_at=None if value.get("finalized_at") is None else str(value.get("finalized_at")),
            release_id=str(value.get("release_id", "")),
            spacecraft_instance_id=str(value.get("spacecraft_instance_id", "")),
            scoped_scene_ref=dict(value.get("scoped_scene_ref", {})),
            source_snapshot=dict(value.get("source_snapshot", {})),
            config_revision=str(value.get("config_revision", "")),
            model_revision=str(value.get("model_revision", "")),
            deployment_profile_revision=str(value.get("deployment_profile_revision", "")),
            fault_profile_revision=str(value.get("fault_profile_revision", "")),
            seed=str(value.get("seed", "")),
            clock=dict(value.get("clock", {})),
            command_set_sha256=value.get("command_set_sha256"),
            command_count=int(value.get("command_count", 0)),
            replay=ReplayRecord(
                replay_state,
                replay_value.get("sha256"),
                replay_value.get("size_bytes"),
                replay_value.get("revision"),
            ),
            incomplete_reason=value.get("incomplete_reason"),
        )


class AtomicRunManifestStore:
    """Persist a manifest using a same-directory fsync + atomic rename."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, manifest: SimulationRunManifest) -> None:
        payload = canonical_json(manifest.as_dict()) + b"\n"
        fd, temp_name = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=self.path.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, self.path)
            try:
                directory_fd = os.open(self.path.parent, os.O_RDONLY)
            except OSError:
                directory_fd = None
            if directory_fd is not None:
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        finally:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass

    def read(self) -> SimulationRunManifest:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RunManifestError(f"unable to read run manifest: {self.path}") from exc
        return SimulationRunManifest.from_dict(value)

    def recover_open(self) -> SimulationRunManifest | None:
        if not self.path.exists():
            return None
        manifest = self.read()
        if manifest.state != RunState.OPEN:
            return manifest
        recovered = manifest.mark_incomplete("CRASH")
        self.write(recovered)
        return recovered
