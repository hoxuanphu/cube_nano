"""Stable public request/result facade for the preprocessing module."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence

from .contracts import (
    PREPROCESS_ARTIFACT_SCHEMA_VERSION,
    CalibrationBundle,
    CaptureManifest,
    ComputeProfile,
    PreprocessingProfile,
    SourceDescriptor,
    TrustPolicy,
    normalize_digest,
)
from .errors import FailureReason, PreprocessError, PreprocessFailure, RunState, SafeAction, StateMachine


@dataclass(frozen=True)
class PreprocessRequest:
    """Immutable preprocessing-only request.

    There is intentionally no engine, model, normalization, patch, or tensor
    layout field in this request.  Those belong to the inference adapter.
    """

    source: Any = None
    capture_manifest: CaptureManifest | Mapping[str, Any] | None = None
    calibration_bundle: CalibrationBundle | Mapping[str, Any] | None = None
    preprocessing_profile: PreprocessingProfile | Mapping[str, Any] | None = None
    compute_profile: ComputeProfile | Mapping[str, Any] | None = None
    output_artifact_target: str | Path = ""
    run_id: str = ""
    correlation_id: str | None = None
    trust_policy: TrustPolicy = field(default_factory=TrustPolicy)
    verify_source_fingerprint: bool = True

    def __post_init__(self) -> None:
        if self.source is None:
            raise ValueError("source is required")
        object.__setattr__(self, "source", SourceDescriptor.from_value(self.source))
        if isinstance(self.capture_manifest, Mapping):
            object.__setattr__(self, "capture_manifest", CaptureManifest.from_mapping(self.capture_manifest))
        if not isinstance(self.capture_manifest, CaptureManifest):
            raise ValueError("capture_manifest must be a CaptureManifest or mapping")
        if isinstance(self.calibration_bundle, Mapping):
            object.__setattr__(self, "calibration_bundle", CalibrationBundle.from_mapping(self.calibration_bundle))
        if not isinstance(self.calibration_bundle, CalibrationBundle):
            raise ValueError("calibration_bundle must be a CalibrationBundle or mapping")
        if isinstance(self.preprocessing_profile, Mapping):
            object.__setattr__(self, "preprocessing_profile", PreprocessingProfile.from_mapping(self.preprocessing_profile))
        if not isinstance(self.preprocessing_profile, PreprocessingProfile):
            raise ValueError("preprocessing_profile must be a PreprocessingProfile or mapping")
        if isinstance(self.compute_profile, Mapping):
            object.__setattr__(self, "compute_profile", ComputeProfile.from_mapping(self.compute_profile))
        if not isinstance(self.compute_profile, ComputeProfile):
            raise ValueError("compute_profile must be a ComputeProfile or mapping")
        target = str(self.output_artifact_target).strip()
        if not target:
            raise ValueError("output_artifact_target is required")
        object.__setattr__(self, "output_artifact_target", Path(target))
        run_id = str(self.run_id).strip()
        correlation_id = None if self.correlation_id is None else str(self.correlation_id).strip()
        if not run_id and not correlation_id:
            raise ValueError("run_id or correlation_id is required")
        if not run_id:
            run_id = correlation_id
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "correlation_id", correlation_id or run_id)
        if not isinstance(self.trust_policy, TrustPolicy):
            raise ValueError("trust_policy must be a TrustPolicy")
        if not isinstance(self.verify_source_fingerprint, bool):
            raise ValueError("verify_source_fingerprint must be boolean")

    @property
    def source_descriptor(self) -> SourceDescriptor:
        return self.source

    @property
    def source_path(self) -> Path | None:
        return self.source.path

    @property
    def artifact_target(self) -> Path:
        return self.output_artifact_target


@dataclass(frozen=True)
class ArtifactOpenRequest:
    """Immutable request for verified artifact mode.

    Profile and calibration replacement values are intentionally absent: the
    artifact manifest remains authoritative and cannot be overwritten by a
    caller during open.
    """

    artifact_path: str | Path = ""
    expected_source_fingerprint: str | None = None
    expected_profile_fingerprint: str | None = None
    expected_calibration_fingerprint: str | None = None
    trust_policy: TrustPolicy = field(default_factory=TrustPolicy)
    compute_profile: ComputeProfile | Mapping[str, Any] | None = None
    run_id: str = ""
    correlation_id: str | None = None

    def __post_init__(self) -> None:
        path = str(self.artifact_path).strip()
        if not path:
            raise ValueError("artifact_path is required")
        object.__setattr__(self, "artifact_path", Path(path))
        for field_name in (
            "expected_source_fingerprint",
            "expected_profile_fingerprint",
            "expected_calibration_fingerprint",
        ):
            value = getattr(self, field_name)
            if value is None:
                raise ValueError(f"{field_name} is required")
            object.__setattr__(self, field_name, normalize_digest(value, field_name))
        if not isinstance(self.trust_policy, TrustPolicy):
            raise ValueError("trust_policy must be a TrustPolicy")
        if isinstance(self.compute_profile, Mapping):
            object.__setattr__(self, "compute_profile", ComputeProfile.from_mapping(self.compute_profile))
        if not isinstance(self.compute_profile, ComputeProfile):
            raise ValueError("compute_profile must be a ComputeProfile or mapping")
        run_id = str(self.run_id).strip()
        correlation_id = None if self.correlation_id is None else str(self.correlation_id).strip()
        if not run_id and not correlation_id:
            raise ValueError("run_id or correlation_id is required")
        object.__setattr__(self, "run_id", run_id or correlation_id)
        object.__setattr__(self, "correlation_id", correlation_id or run_id)


@dataclass(frozen=True)
class ArtifactManifest:
    """P0 artifact schema linking source/profile/calibration, without model data."""

    schema_version: int = PREPROCESS_ARTIFACT_SCHEMA_VERSION
    artifact_id: str = ""
    source_fingerprint: str = ""
    profile_fingerprint: str = ""
    calibration_fingerprint: str = ""
    output_shape: tuple[int, ...] = ()
    output_layout: str = ""
    output_dtype: str = ""
    validity_encoding: Mapping[str, Any] = field(default_factory=dict)
    reason_encoding: Mapping[str, Any] = field(default_factory=dict)
    mapping_ref: str = ""
    checksums: Mapping[str, str] = field(default_factory=dict)
    complete: bool = False
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != PREPROCESS_ARTIFACT_SCHEMA_VERSION:
            raise ValueError(f"artifact schema_version must be {PREPROCESS_ARTIFACT_SCHEMA_VERSION}")
        if not str(self.artifact_id).strip():
            raise ValueError("artifact_id is required")
        for field_name in ("source_fingerprint", "profile_fingerprint", "calibration_fingerprint"):
            object.__setattr__(self, field_name, normalize_digest(getattr(self, field_name), field_name))
        if not self.output_shape or any(int(value) <= 0 for value in self.output_shape):
            raise ValueError("artifact output_shape must contain positive dimensions")
        object.__setattr__(self, "output_shape", tuple(int(value) for value in self.output_shape))
        layout = str(self.output_layout).strip().upper()
        if not layout or len(layout) != len(self.output_shape):
            raise ValueError("artifact output_layout must match output_shape rank")
        object.__setattr__(self, "output_layout", layout)
        if not str(self.output_dtype).strip():
            raise ValueError("artifact output_dtype is required")
        object.__setattr__(self, "output_dtype", str(self.output_dtype))
        object.__setattr__(self, "validity_encoding", MappingProxyType(dict(self.validity_encoding)))
        object.__setattr__(self, "reason_encoding", MappingProxyType(dict(self.reason_encoding)))
        checksums = {}
        for name, digest in dict(self.checksums).items():
            key = str(name).strip()
            if not key or Path(key).name != key:
                raise ValueError("artifact checksum file names must be simple relative names")
            checksums[key] = normalize_digest(digest, f"checksums.{key}")
        object.__setattr__(self, "checksums", MappingProxyType(checksums))
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))
        mapping_ref = str(self.mapping_ref).strip()
        if not mapping_ref or Path(mapping_ref).name != mapping_ref:
            raise ValueError("artifact mapping_ref is required")
        object.__setattr__(self, "mapping_ref", mapping_ref)
        if not isinstance(self.complete, bool):
            raise ValueError("artifact complete must be boolean")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ArtifactManifest":
        if not isinstance(value, Mapping):
            raise ValueError("artifact manifest must be a mapping")
        return cls(
            schema_version=value.get("schema_version", 0),
            artifact_id=value.get("artifact_id", ""),
            source_fingerprint=value.get("source_fingerprint", ""),
            profile_fingerprint=value.get("profile_fingerprint", ""),
            calibration_fingerprint=value.get("calibration_fingerprint", ""),
            output_shape=value.get("output_shape", ()),
            output_layout=value.get("output_layout", ""),
            output_dtype=value.get("output_dtype", ""),
            validity_encoding=value.get("validity_encoding", {}),
            reason_encoding=value.get("reason_encoding", {}),
            mapping_ref=value.get("mapping_ref", ""),
            checksums=value.get("checksums", {}),
            complete=value.get("complete", False),
            provenance=value.get("provenance", {}),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "source_fingerprint": self.source_fingerprint,
            "profile_fingerprint": self.profile_fingerprint,
            "calibration_fingerprint": self.calibration_fingerprint,
            "output_shape": self.output_shape,
            "output_layout": self.output_layout,
            "output_dtype": self.output_dtype,
            "validity_encoding": self.validity_encoding,
            "reason_encoding": self.reason_encoding,
            "mapping_ref": self.mapping_ref,
            "checksums": self.checksums,
            "complete": self.complete,
            "provenance": self.provenance,
        }

    @classmethod
    def from_contracts(
        cls,
        *,
        artifact_id: str,
        capture_manifest: CaptureManifest,
        preprocessing_profile: PreprocessingProfile,
        calibration_bundle: CalibrationBundle,
        mapping_ref: str,
        complete: bool = False,
        provenance: Mapping[str, Any] | None = None,
    ) -> "ArtifactManifest":
        return cls(
            artifact_id=artifact_id,
            source_fingerprint=capture_manifest.source_fingerprint,
            profile_fingerprint=preprocessing_profile.fingerprint,
            calibration_fingerprint=calibration_bundle.fingerprint,
            output_shape=preprocessing_profile.output_shape,
            output_layout=preprocessing_profile.output_layout,
            output_dtype=preprocessing_profile.output_dtype,
            validity_encoding=preprocessing_profile.validity_encoding.to_mapping(),
            reason_encoding=preprocessing_profile.reason_encoding.to_mapping(),
            mapping_ref=mapping_ref,
            complete=complete,
            provenance=provenance or {},
        )


class ModelGridReader(Protocol):
    def read_block(self, row_start: int, row_end: int) -> Mapping[str, Any]:
        ...

    def __enter__(self) -> "ModelGridReader":
        ...

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        ...


@dataclass(frozen=True)
class PreprocessArtifact:
    """Verified artifact handle for the materialized preprocessing output."""

    artifact_path: Path | str = ""
    manifest: ArtifactManifest | Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        path = str(self.artifact_path).strip()
        if not path:
            raise ValueError("artifact_path is required")
        object.__setattr__(self, "artifact_path", Path(path))
        if isinstance(self.manifest, Mapping):
            object.__setattr__(self, "manifest", ArtifactManifest(**dict(self.manifest)))
        if not isinstance(self.manifest, ArtifactManifest):
            raise ValueError("manifest must be an ArtifactManifest or mapping")

    @property
    def is_failure(self) -> bool:
        return False

    def open(self) -> ModelGridReader | PreprocessFailure:
        machine = StateMachine()
        try:
            machine.transition(RunState.VALIDATING)
            machine.transition(RunState.ADMITTED)
            machine.transition(RunState.PROCESSING)
            machine.transition(RunState.VERIFYING)
            from .artifact_reader import open_artifact_reader

            reader = open_artifact_reader(self.artifact_path, expected_manifest=self.manifest)
            machine.transition(RunState.COMPLETE)
            return reader
        except PreprocessError as error:
            try:
                _transition_failure(machine, RunState(error.state))
            except RuntimeError:
                pass
            return _failure_from_error(error, request=self, machine=machine)


def _failure_from_error(error: PreprocessError, *, request: Any, machine: StateMachine) -> PreprocessFailure:
    provenance = dict(error.provenance)
    provenance.update(
        {
            "run_id": getattr(request, "run_id", None),
            "correlation_id": getattr(request, "correlation_id", None),
            "state_history": tuple(state.value for state in machine.state_history),
        }
    )
    return PreprocessFailure(
        state=error.state,
        reason_code=error.reason_code,
        safe_action=SafeAction.RETAIN_FOR_GROUND,
        message=str(error),
        run_id=getattr(request, "run_id", None),
        provenance=provenance,
    )


def _transition_failure(machine: StateMachine, state: RunState) -> None:
    if machine.state == state:
        return
    machine.transition(state)


def preprocess_capture(request: PreprocessRequest) -> PreprocessArtifact | PreprocessFailure:
    """Run source read, geometric warp, and atomic artifact materialization."""

    machine = StateMachine()
    if not isinstance(request, PreprocessRequest):
        return PreprocessFailure(
            state=RunState.INVALID_INPUT,
            reason_code=FailureReason.REQUEST_TYPE_INVALID,
            message="preprocess_capture requires a PreprocessRequest",
            provenance={"state_history": tuple(state.value for state in machine.state_history)},
        )
    try:
        machine.transition(RunState.VALIDATING)
        from .admission import ResourceAdmission, SystemResourceProbe
        from .resolver import ContractResolver

        resolved = ContractResolver(
            trust_policy=request.trust_policy,
            verify_source_fingerprint=request.verify_source_fingerprint,
        ).resolve(request)
        receipt = ResourceAdmission(resolved.compute_profile).preflight(resolved)
        machine.transition(RunState.ADMITTED)
        machine.transition(RunState.PROCESSING)
        from .artifact_writer import ArtifactWriter
        from .source_reader import open_source_reader
        from .transform_plan import TransformPlanner
        from .warp_backend import create_warp_backend

        snapshot = SystemResourceProbe().snapshot(request.artifact_target)
        admission = ResourceAdmission(resolved.compute_profile)
        runtime = admission.runtime_check(
            receipt,
            actual_peak_ram_bytes=receipt.estimate.ram_peak_bytes,
            free_disk_bytes=snapshot.free_disk_bytes,
            stage="runtime",
        )
        provenance = {
            "preflight": receipt.to_mapping(),
            "runtime": runtime.to_mapping(),
            "ram_observation": "conservative_preflight_upper_bound",
        }
        writer = None
        with open_source_reader(
            resolved.source,
            resolved.preprocessing_profile,
            compute_profile=resolved.compute_profile,
        ) as source_reader:
            planner = TransformPlanner(
                resolved.preprocessing_profile,
                resolved.calibration_bundle,
                source_reader.shape_yxc,
            )
            backend = create_warp_backend(
                resolved.preprocessing_profile,
                resolved.compute_profile.backend,
            )
            writer = ArtifactWriter(
                resolved,
                request.artifact_target,
                run_id=request.run_id,
                provenance=provenance,
            )
            try:
                for plan in planner.plan_strips(receipt.estimate.strip_rows):
                    source_block = source_reader.read_window(plan.source_window)
                    writer.write_block(
                        plan.output_row_start,
                        plan.output_row_end,
                        backend.warp(source_block, plan),
                    )
                machine.transition(RunState.VERIFYING)
                commit_snapshot = SystemResourceProbe().snapshot(request.artifact_target)
                commit = admission.before_publish(
                    receipt,
                    actual_peak_ram_bytes=receipt.estimate.ram_peak_bytes,
                    free_disk_bytes=commit_snapshot.free_disk_bytes,
                )
                provenance["commit"] = commit.to_mapping()
                artifact_path, manifest = writer.finalize()
            except BaseException:
                writer.abort()
                raise
        machine.transition(RunState.COMPLETE)
        return PreprocessArtifact(artifact_path=artifact_path, manifest=manifest)
    except PreprocessError as error:
        try:
            _transition_failure(machine, RunState(error.state))
        except RuntimeError as transition_error:
            return PreprocessFailure(
                state=RunState.RUNTIME_FAULT,
                reason_code=FailureReason.RUNTIME_FAULT,
                message=str(transition_error),
                run_id=request.run_id,
                provenance={"original_error": str(error), "state_history": tuple(state.value for state in machine.state_history)},
            )
        return _failure_from_error(error, request=request, machine=machine)


def open_preprocessed_artifact(request: ArtifactOpenRequest) -> PreprocessArtifact | PreprocessFailure:
    """Verify a complete artifact and return a reader-backed artifact handle."""

    machine = StateMachine()
    if not isinstance(request, ArtifactOpenRequest):
        return PreprocessFailure(
            state=RunState.INVALID_INPUT,
            reason_code=FailureReason.REQUEST_TYPE_INVALID,
            message="open_preprocessed_artifact requires an ArtifactOpenRequest",
            provenance={"state_history": tuple(state.value for state in machine.state_history)},
        )
    try:
        machine.transition(RunState.VALIDATING)
        machine.transition(RunState.ADMITTED)
        machine.transition(RunState.PROCESSING)
        from .artifact_reader import verify_artifact

        machine.transition(RunState.VERIFYING)
        manifest = verify_artifact(request.artifact_path, request=request)
        machine.transition(RunState.COMPLETE)
        return PreprocessArtifact(artifact_path=request.artifact_path, manifest=manifest)
    except PreprocessError as error:
        try:
            _transition_failure(machine, RunState(error.state))
        except RuntimeError as transition_error:
            return PreprocessFailure(
                state=RunState.RUNTIME_FAULT,
                reason_code=FailureReason.RUNTIME_FAULT,
                message=str(transition_error),
                run_id=request.run_id,
                provenance={
                    "original_error": str(error),
                    "state_history": tuple(state.value for state in machine.state_history),
                },
            )
        return _failure_from_error(error, request=request, machine=machine)


__all__ = [
    "ArtifactManifest",
    "ArtifactOpenRequest",
    "ModelGridReader",
    "PreprocessArtifact",
    "PreprocessRequest",
    "open_preprocessed_artifact",
    "preprocess_capture",
]
