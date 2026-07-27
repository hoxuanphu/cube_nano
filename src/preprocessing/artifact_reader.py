"""Verification and block reader for materialized preprocessing artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .artifact_writer import PAYLOAD_FILES, REQUIRED_ARTIFACT_FILES, sha256_path
from .contracts import (
    PREPROCESS_ARTIFACT_SCHEMA_VERSION,
    CalibrationBundle,
    CaptureManifest,
    PreprocessingProfile,
    TrustPolicy,
    canonical_json,
    normalize_digest,
)
from .errors import FailureReason, PreprocessError, RunState
from .resolver import verify_trust


def _artifact_error(
    reason: FailureReason,
    message: str,
    *,
    state: RunState = RunState.INVALID_INPUT,
    path: Path,
    provenance: Mapping[str, Any] | None = None,
) -> PreprocessError:
    details = {"artifact_path": str(path)}
    details.update(dict(provenance or {}))
    return PreprocessError(reason, message, state=state, provenance=details)


def _read_json(path: Path, *, field_name: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise _artifact_error(
            FailureReason.ARTIFACT_INCOMPLETE,
            f"artifact is missing {field_name}",
            path=path.parent,
            provenance={"missing_file": path.name},
        ) from exc
    except (OSError, UnicodeError) as exc:
        raise _artifact_error(
            FailureReason.IO_ERROR,
            f"unable to read artifact {field_name}",
            state=RunState.IO_FAULT,
            path=path.parent,
            provenance={"file": path.name, "error": str(exc)},
        ) from exc
    except json.JSONDecodeError as exc:
        raise _artifact_error(
            FailureReason.ARTIFACT_SCHEMA_INVALID,
            f"artifact {field_name} is not valid JSON",
            path=path.parent,
            provenance={"file": path.name, "error": str(exc)},
        ) from exc
    if not isinstance(value, Mapping):
        raise _artifact_error(
            FailureReason.ARTIFACT_SCHEMA_INVALID,
            f"artifact {field_name} must be a JSON object",
            path=path.parent,
            provenance={"file": path.name},
        )
    return value


def _validate_simple_file_set(path: Path) -> None:
    if not path.is_dir():
        raise _artifact_error(
            FailureReason.ARTIFACT_INCOMPLETE,
            f"artifact directory does not exist: {path}",
            path=path,
        )
    for name in REQUIRED_ARTIFACT_FILES:
        candidate = path / name
        if not candidate.is_file():
            raise _artifact_error(
                FailureReason.ARTIFACT_INCOMPLETE,
                f"artifact is missing required file: {name}",
                path=path,
                provenance={"missing_file": name},
            )


def _load_manifest(path: Path) -> tuple[Mapping[str, Any], Any]:
    _validate_simple_file_set(path)
    payload = _read_json(path / "manifest.json", field_name="manifest")
    if payload.get("status") != "complete" or payload.get("complete") is not True:
        raise _artifact_error(
            FailureReason.ARTIFACT_INCOMPLETE,
            "artifact manifest is not complete",
            path=path,
            provenance={"status": payload.get("status"), "complete": payload.get("complete")},
        )
    if payload.get("files") is not None:
        try:
            valid_file_set = set(payload.get("files", ())) == set(REQUIRED_ARTIFACT_FILES)
        except TypeError as exc:
            raise _artifact_error(
                FailureReason.ARTIFACT_SCHEMA_INVALID,
                "artifact manifest file set is invalid",
                path=path,
                provenance={"files": payload.get("files")},
            ) from exc
        if not valid_file_set:
            raise _artifact_error(
                FailureReason.ARTIFACT_SCHEMA_INVALID,
                "artifact manifest file set is invalid",
                path=path,
                provenance={"files": payload.get("files")},
            )
    checksums = payload.get("checksums")
    if not isinstance(checksums, Mapping) or set(checksums) != set(PAYLOAD_FILES):
        raise _artifact_error(
            FailureReason.ARTIFACT_SCHEMA_INVALID,
            "artifact manifest checksum set is invalid",
            path=path,
            provenance={"checksums": checksums},
        )
    try:
        manifest = _manifest_from_payload(payload)
    except (TypeError, ValueError, KeyError) as exc:
        raise _artifact_error(
            FailureReason.ARTIFACT_SCHEMA_INVALID,
            "artifact manifest does not match the published schema",
            path=path,
            provenance={"error": str(exc)},
        ) from exc
    for name, expected in manifest.checksums.items():
        actual = sha256_path(path / name)
        if actual != expected:
            raise _artifact_error(
                FailureReason.ARTIFACT_CHECKSUM_MISMATCH,
                f"artifact checksum mismatch for {name}",
                state=RunState.IO_FAULT,
                path=path,
                provenance={"file": name, "expected": expected, "actual": actual},
            )
    return payload, manifest


def _manifest_from_payload(payload: Mapping[str, Any]) -> Any:
    from .api import ArtifactManifest

    return ArtifactManifest.from_mapping(payload)


def _validate_metadata_impl(path: Path, manifest: Any) -> tuple[Mapping[str, Any], Mapping[str, Any], CaptureManifest, PreprocessingProfile, CalibrationBundle]:
    output = _read_json(path / "output.json", field_name="output metadata")
    preprocess = _read_json(path / "preprocess.json", field_name="preprocess metadata")
    expected_shape = tuple(int(value) for value in manifest.output_shape)
    if output.get("schema_version") != PREPROCESS_ARTIFACT_SCHEMA_VERSION:
        raise _artifact_error(
            FailureReason.ARTIFACT_SCHEMA_INVALID,
            "output metadata schema version is invalid",
            path=path,
        )
    if tuple(output.get("output_shape", ())) != expected_shape:
        raise _artifact_error(
            FailureReason.ARTIFACT_SCHEMA_INVALID,
            "output metadata shape does not match manifest",
            path=path,
        )
    if str(output.get("output_layout", "")).upper() != manifest.output_layout or str(output.get("dtype", "")) != manifest.output_dtype:
        raise _artifact_error(
            FailureReason.ARTIFACT_SCHEMA_INVALID,
            "output metadata layout or dtype does not match manifest",
            path=path,
        )
    if canonical_json(output.get("validity", {})) != canonical_json(manifest.validity_encoding):
        raise _artifact_error(
            FailureReason.ARTIFACT_SCHEMA_INVALID,
            "output validity encoding does not match manifest",
            path=path,
        )
    if canonical_json(output.get("reason", {})) != canonical_json(manifest.reason_encoding):
        raise _artifact_error(
            FailureReason.ARTIFACT_SCHEMA_INVALID,
            "output reason encoding does not match manifest",
            path=path,
        )
    if preprocess.get("schema_version") != PREPROCESS_ARTIFACT_SCHEMA_VERSION:
        raise _artifact_error(
            FailureReason.ARTIFACT_SCHEMA_INVALID,
            "preprocess metadata schema version is invalid",
            path=path,
        )
    try:
        output_fingerprint = normalize_digest(preprocess.get("output_fingerprint"), "output_fingerprint")
        raster_checksums = {name: manifest.checksums[name] for name in ("model-grid.tif", "validity.tif", "validity-reasons.tif", "mapping.npy")}
        expected_output_fingerprint = hashlib.sha256(canonical_json(raster_checksums).encode("utf-8")).hexdigest()
    except (TypeError, ValueError, KeyError) as exc:
        raise _artifact_error(
            FailureReason.ARTIFACT_SCHEMA_INVALID,
            "preprocess metadata output fingerprint is invalid",
            path=path,
            provenance={"error": str(exc)},
        ) from exc
    if output_fingerprint != expected_output_fingerprint:
        raise _artifact_error(
            FailureReason.ARTIFACT_CHECKSUM_MISMATCH,
            "preprocess metadata output fingerprint does not match payload",
            state=RunState.IO_FAULT,
            path=path,
            provenance={"expected": expected_output_fingerprint, "actual": output_fingerprint},
        )
    try:
        capture = CaptureManifest.from_mapping(preprocess["capture_manifest"])
        profile = PreprocessingProfile.from_mapping(preprocess["preprocessing_profile"])
        calibration = CalibrationBundle.from_mapping(preprocess["calibration_bundle"])
    except (KeyError, TypeError, ValueError) as exc:
        raise _artifact_error(
            FailureReason.ARTIFACT_SCHEMA_INVALID,
            "embedded preprocessing contracts are invalid",
            path=path,
            provenance={"error": str(exc)},
        ) from exc
    if preprocess.get("source_fingerprint") != capture.source_fingerprint:
        raise _artifact_error(
            FailureReason.ARTIFACT_SCHEMA_INVALID,
            "preprocess source fingerprint does not match capture manifest",
            path=path,
        )
    if (
        capture.source_fingerprint != manifest.source_fingerprint
        or profile.fingerprint != manifest.profile_fingerprint
        or calibration.fingerprint != manifest.calibration_fingerprint
    ):
        raise _artifact_error(
            FailureReason.ARTIFACT_SCHEMA_INVALID,
            "embedded contract fingerprints do not match manifest",
            path=path,
            provenance={
                "manifest_source": manifest.source_fingerprint,
                "embedded_source": capture.source_fingerprint,
                "manifest_profile": manifest.profile_fingerprint,
                "embedded_profile": profile.fingerprint,
                "manifest_calibration": manifest.calibration_fingerprint,
                "embedded_calibration": calibration.fingerprint,
            },
        )
    if (
        (capture.preprocessing_profile_digest is not None and capture.preprocessing_profile_digest != profile.fingerprint)
        or (capture.calibration_digest is not None and capture.calibration_digest != calibration.fingerprint)
    ):
        raise _artifact_error(
            FailureReason.ARTIFACT_SCHEMA_INVALID,
            "capture linkage does not match embedded profile and calibration",
            path=path,
        )
    if tuple(profile.output_shape) != expected_shape or profile.output_layout != manifest.output_layout or profile.output_dtype != manifest.output_dtype:
        raise _artifact_error(
            FailureReason.ARTIFACT_SCHEMA_INVALID,
            "embedded profile output contract does not match manifest",
            path=path,
        )
    mapping = preprocess.get("mapping")
    if not isinstance(mapping, Mapping) or mapping.get("ref") != manifest.mapping_ref:
        raise _artifact_error(
            FailureReason.ARTIFACT_SCHEMA_INVALID,
            "preprocess mapping reference is invalid",
            path=path,
        )
    if tuple(mapping.get("shape", ())) != (*profile.target_grid.shape, 2) or mapping.get("direction") != "target_to_source":
        raise _artifact_error(
            FailureReason.ARTIFACT_SCHEMA_INVALID,
            "preprocess mapping metadata is invalid",
            path=path,
        )
    return output, preprocess, capture, profile, calibration


def _validate_metadata(path: Path, manifest: Any) -> tuple[Mapping[str, Any], Mapping[str, Any], CaptureManifest, PreprocessingProfile, CalibrationBundle]:
    try:
        return _validate_metadata_impl(path, manifest)
    except PreprocessError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise _artifact_error(
            FailureReason.ARTIFACT_SCHEMA_INVALID,
            "artifact metadata contains invalid field types or values",
            path=path,
            provenance={"error": str(exc)},
        ) from exc


def _validate_payload_arrays(path: Path, manifest: Any, profile: PreprocessingProfile) -> None:
    try:
        import tifffile

        image = tifffile.memmap(path / "model-grid.tif", mode="r")
        validity = tifffile.memmap(path / "validity.tif", mode="r")
        reasons = tifffile.memmap(path / "validity-reasons.tif", mode="r")
    except (ImportError, OSError, ValueError, TypeError) as exc:
        raise _artifact_error(
            FailureReason.ARTIFACT_SCHEMA_INVALID,
            "artifact TIFF payload cannot be opened with the declared schema",
            path=path,
            provenance={"error": str(exc)},
        ) from exc
    try:
        expected_yx = (profile.target_grid.rows, profile.target_grid.cols)
        if tuple(image.shape) != tuple(profile.output_shape) or image.dtype != np.dtype(manifest.output_dtype):
            raise _artifact_error(
                FailureReason.ARTIFACT_SCHEMA_INVALID,
                "model-grid payload shape or dtype is invalid",
                path=path,
                provenance={"shape": image.shape, "dtype": image.dtype.str},
            )
        if tuple(validity.shape) != expected_yx or validity.dtype != np.dtype(profile.validity_encoding.dtype):
            raise _artifact_error(
                FailureReason.ARTIFACT_SCHEMA_INVALID,
                "validity payload shape or dtype is invalid",
                path=path,
            )
        if tuple(reasons.shape) != expected_yx or reasons.dtype != np.dtype(profile.reason_encoding.dtype):
            raise _artifact_error(
                FailureReason.ARTIFACT_SCHEMA_INVALID,
                "reason payload shape or dtype is invalid",
                path=path,
            )
    finally:
        for value in (image, validity, reasons):
            mmap = getattr(value, "_mmap", None)
            if mmap is not None:
                mmap.close()
    try:
        mapping = np.load(path / manifest.mapping_ref, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError, TypeError) as exc:
        raise _artifact_error(
            FailureReason.ARTIFACT_SCHEMA_INVALID,
            "artifact mapping payload cannot be opened",
            path=path,
            provenance={"error": str(exc)},
        ) from exc
    try:
        expected_mapping_shape = (profile.target_grid.rows, profile.target_grid.cols, 2)
        expected_mapping_dtype = np.dtype(profile.internal_numeric_precision)
        if tuple(mapping.shape) != expected_mapping_shape or mapping.dtype != expected_mapping_dtype:
            raise _artifact_error(
                FailureReason.ARTIFACT_SCHEMA_INVALID,
                "mapping payload shape or dtype is invalid",
                path=path,
                provenance={"shape": mapping.shape, "dtype": mapping.dtype.str},
            )
    finally:
        mmap = getattr(mapping, "_mmap", None)
        if mmap is not None:
            mmap.close()


def _verify_contracts(
    path: Path,
    capture: CaptureManifest,
    profile: PreprocessingProfile,
    calibration: CalibrationBundle,
    *,
    trust_policy: TrustPolicy | None,
) -> None:
    if trust_policy is not None:
        verify_trust(capture, "capture", trust_policy)
        verify_trust(profile, "profile", trust_policy)
        verify_trust(calibration, "calibration", trust_policy)


def verify_artifact(path: str | Path, *, request: Any) -> Any:
    """Verify a complete artifact against the caller's expected fingerprints."""

    artifact_path = Path(path)
    payload, manifest = _load_manifest(artifact_path)
    output, preprocess, capture, profile, calibration = _validate_metadata(artifact_path, manifest)
    _validate_payload_arrays(artifact_path, manifest, profile)
    _verify_contracts(
        artifact_path,
        capture,
        profile,
        calibration,
        trust_policy=request.trust_policy,
    )
    verify_trust(request.compute_profile, "compute", request.trust_policy)
    expected = {
        "source": request.expected_source_fingerprint,
        "profile": request.expected_profile_fingerprint,
        "calibration": request.expected_calibration_fingerprint,
    }
    actual = {
        "source": manifest.source_fingerprint,
        "profile": manifest.profile_fingerprint,
        "calibration": manifest.calibration_fingerprint,
    }
    for kind, expected_digest in expected.items():
        if expected_digest != actual[kind]:
            raise _artifact_error(
                FailureReason.FINGERPRINT_LINKAGE_MISMATCH,
                f"artifact {kind} fingerprint does not match the request",
                path=artifact_path,
                provenance={"kind": kind, "expected": expected_digest, "actual": actual[kind]},
            )
    try:
        payload_bytes = sum((artifact_path / name).stat().st_size for name in REQUIRED_ARTIFACT_FILES)
    except OSError as exc:
        raise _artifact_error(
            FailureReason.IO_ERROR,
            "unable to inspect artifact size",
            state=RunState.IO_FAULT,
            path=artifact_path,
            provenance={"error": str(exc)},
        ) from exc
    if payload_bytes > request.compute_profile.disk_budget_bytes:
        raise _artifact_error(
            FailureReason.RESOURCE_RUNTIME,
            "artifact exceeds the requested compute disk budget",
            state=RunState.RESOURCE_REJECTED,
            path=artifact_path,
            provenance={"artifact_bytes": payload_bytes, "disk_budget_bytes": request.compute_profile.disk_budget_bytes},
        )
    return manifest


def open_artifact_reader(path: str | Path, *, expected_manifest: Any | None = None) -> "PreprocessedArtifactReader":
    artifact_path = Path(path)
    _, manifest = _load_manifest(artifact_path)
    output, preprocess, capture, profile, calibration = _validate_metadata(artifact_path, manifest)
    _validate_payload_arrays(artifact_path, manifest, profile)
    if expected_manifest is not None and canonical_json(manifest.to_mapping()) != canonical_json(expected_manifest.to_mapping()):
        raise _artifact_error(
            FailureReason.ARTIFACT_SCHEMA_INVALID,
            "artifact manifest changed after it was verified",
            path=artifact_path,
        )
    return PreprocessedArtifactReader(artifact_path, manifest, profile)


class PreprocessedArtifactReader:
    """Block reader backed only by verified materialized artifact files."""

    def __init__(self, path: Path, manifest: Any, profile: PreprocessingProfile):
        self.path = path
        self.manifest = manifest
        self.profile = profile
        self._closed = False
        self._arrays: dict[str, Any] = {}
        try:
            import tifffile

            self._arrays["image"] = tifffile.memmap(path / "model-grid.tif", mode="r")
            self._arrays["validity"] = tifffile.memmap(path / "validity.tif", mode="r")
            self._arrays["reason"] = tifffile.memmap(path / "validity-reasons.tif", mode="r")
            self._arrays["mapping"] = np.load(path / manifest.mapping_ref, mmap_mode="r", allow_pickle=False)
        except (ImportError, OSError, ValueError, TypeError) as exc:
            self.close()
            raise _artifact_error(
                FailureReason.IO_ERROR,
                "unable to open verified artifact payload",
                state=RunState.IO_FAULT,
                path=path,
                provenance={"error": str(exc)},
            ) from exc

    def _image_slice(self, row_start: int, row_end: int) -> tuple[Any, ...]:
        if self.profile.output_layout == "YXC":
            return (slice(row_start, row_end), slice(None), slice(None))
        if self.profile.output_layout == "CYX":
            return (slice(None), slice(row_start, row_end), slice(None))
        if self.profile.output_layout == "YX":
            return (slice(row_start, row_end), slice(None))
        raise RuntimeError(f"unsupported artifact output layout {self.profile.output_layout}")

    def read_block(self, row_start: int, row_end: int) -> Mapping[str, Any]:
        if self._closed:
            raise RuntimeError("artifact reader is closed")
        if isinstance(row_start, bool) or isinstance(row_end, bool) or int(row_start) != row_start or int(row_end) != row_end:
            raise TypeError("artifact block bounds must be integers")
        row_start, row_end = int(row_start), int(row_end)
        target_rows = self.profile.target_grid.rows
        if row_start < 0 or row_end <= row_start or row_end > target_rows:
            raise ValueError("artifact block bounds are outside target grid")
        return {
            "image": np.asarray(self._arrays["image"][self._image_slice(row_start, row_end)]).copy(),
            "validity_yx": np.asarray(self._arrays["validity"][row_start:row_end, :]).copy(),
            "validity_reason_yx": np.asarray(self._arrays["reason"][row_start:row_end, :]).copy(),
            "mapping_yx": np.asarray(self._arrays["mapping"][row_start:row_end, :, :]).copy(),
            "mapping_ref": self.manifest.mapping_ref,
            "provenance": MappingProxyType(
                {
                    "artifact_path": str(self.path),
                    "artifact_id": self.manifest.artifact_id,
                    "output_row_start": row_start,
                    "output_row_end": row_end,
                }
            ),
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        arrays = self._arrays
        self._arrays = {}
        for value in arrays.values():
            mmap = getattr(value, "_mmap", None)
            if mmap is not None:
                mmap.close()

    def __enter__(self) -> "PreprocessedArtifactReader":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


__all__ = ["PreprocessedArtifactReader", "open_artifact_reader", "verify_artifact"]
