"""Atomic, profile-driven materialization of a ``PreprocessArtifact``."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contracts import PREPROCESS_ARTIFACT_SCHEMA_VERSION, canonical_json
from .errors import FailureReason, PreprocessError, RunState
from .warp_backend import WarpResult


PAYLOAD_FILES = (
    "model-grid.tif",
    "validity.tif",
    "validity-reasons.tif",
    "mapping.npy",
    "output.json",
    "preprocess.json",
)
REQUIRED_ARTIFACT_FILES = (*PAYLOAD_FILES, "manifest.json")


def sha256_path(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a closed artifact file without loading it into memory."""

    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as stream:
            while chunk := stream.read(chunk_size):
                digest.update(chunk)
    except OSError as exc:
        raise PreprocessError(
            FailureReason.IO_ERROR,
            f"unable to checksum artifact file: {path}",
            state=RunState.IO_FAULT,
            provenance={"path": str(path), "error": str(exc)},
        ) from exc
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    try:
        path.write_text(canonical_json(value) + "\n", encoding="utf-8")
    except OSError as exc:
        raise PreprocessError(
            FailureReason.IO_ERROR,
            f"unable to write artifact metadata: {path}",
            state=RunState.IO_FAULT,
            provenance={"path": str(path), "error": str(exc)},
        ) from exc


def _close_memmap(value: Any) -> None:
    if value is None:
        return
    flush = getattr(value, "flush", None)
    if callable(flush):
        flush()
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def _safe_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip(".-")
    return token[:48] or "run"


class ArtifactWriter:
    """Write one artifact into staging and publish it as one directory rename."""

    def __init__(
        self,
        resolved: Any,
        target: str | Path,
        *,
        run_id: str,
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        self.resolved = resolved
        self.profile = resolved.preprocessing_profile
        self.capture = resolved.capture_manifest
        self.calibration = resolved.calibration_bundle
        self.target = Path(target)
        self._closed = False
        self._finalized = False
        self._next_row = 0
        self._provenance = dict(provenance or {})
        self._arrays: dict[str, Any] = {}

        if self.target.exists():
            raise PreprocessError(
                FailureReason.IO_ERROR,
                f"artifact target already exists: {self.target}",
                state=RunState.IO_FAULT,
                provenance={"artifact_path": str(self.target)},
            )
        try:
            self.target.parent.mkdir(parents=True, exist_ok=True)
            staging_name = f".{self.target.name}.staging-{_safe_token(run_id)}-{uuid.uuid4().hex}"
            self.staging_path = self.target.parent / staging_name
            self.staging_path.mkdir()
            self._open_payload_arrays()
        except PreprocessError:
            self.abort()
            raise
        except (OSError, ValueError, TypeError) as exc:
            self.abort()
            raise PreprocessError(
                FailureReason.IO_ERROR,
                f"unable to create artifact staging area: {self.staging_path}",
                state=RunState.IO_FAULT,
                provenance={"artifact_path": str(self.target), "error": str(exc)},
            ) from exc

    @property
    def staging_dir(self) -> Path:
        return self.staging_path

    @property
    def logical_output_bytes(self) -> int:
        total = 0
        for name in PAYLOAD_FILES:
            path = self.staging_path / name
            if path.exists():
                total += path.stat().st_size
        return total

    def _open_payload_arrays(self) -> None:
        try:
            import tifffile
        except ImportError as exc:
            raise PreprocessError(
                FailureReason.CODEC_UNAVAILABLE,
                "writing TIFF artifacts requires tifffile",
                state=RunState.RESOURCE_REJECTED,
                provenance={"artifact_path": str(self.target)},
            ) from exc

        output_shape = tuple(int(value) for value in self.profile.output_shape)
        target_shape_yx = (self.profile.target_grid.rows, self.profile.target_grid.cols)
        try:
            self._arrays["image"] = tifffile.memmap(
                self.staging_path / "model-grid.tif",
                shape=output_shape,
                dtype=np.dtype(self.profile.output_dtype),
                mode="w+",
            )
            self._arrays["validity"] = tifffile.memmap(
                self.staging_path / "validity.tif",
                shape=target_shape_yx,
                dtype=np.dtype(self.profile.validity_encoding.dtype),
                mode="w+",
            )
            self._arrays["reason"] = tifffile.memmap(
                self.staging_path / "validity-reasons.tif",
                shape=target_shape_yx,
                dtype=np.dtype(self.profile.reason_encoding.dtype),
                mode="w+",
            )
            self._arrays["mapping"] = np.lib.format.open_memmap(
                self.staging_path / "mapping.npy",
                mode="w+",
                dtype=np.dtype(self.profile.internal_numeric_precision),
                shape=(*target_shape_yx, 2),
            )
        except (OSError, ValueError, TypeError) as exc:
            raise PreprocessError(
                FailureReason.IO_ERROR,
                "unable to allocate artifact payload files",
                state=RunState.IO_FAULT,
                provenance={"artifact_path": str(self.target), "error": str(exc)},
            ) from exc

    def _image_slice(self, row_start: int, row_end: int) -> tuple[Any, ...]:
        if self.profile.output_layout == "YXC":
            return (slice(row_start, row_end), slice(None), slice(None))
        if self.profile.output_layout == "CYX":
            return (slice(None), slice(row_start, row_end), slice(None))
        if self.profile.output_layout == "YX":
            return (slice(row_start, row_end), slice(None))
        raise PreprocessError(
            FailureReason.ARTIFACT_SCHEMA_INVALID,
            f"unsupported output layout {self.profile.output_layout}",
            state=RunState.INVALID_INPUT,
        )

    def write_block(self, row_start: int, row_end: int, result: WarpResult) -> None:
        """Write one contiguous output strip and its sidecars."""

        if self._closed or self._finalized:
            raise RuntimeError("artifact writer is closed")
        if isinstance(row_start, bool) or isinstance(row_end, bool) or int(row_start) != row_start or int(row_end) != row_end:
            raise ValueError("artifact block bounds must be integers")
        row_start, row_end = int(row_start), int(row_end)
        target_rows, target_cols = self.profile.target_grid.rows, self.profile.target_grid.cols
        if row_start != self._next_row:
            reason = FailureReason.PATCH_RESULT_DUPLICATE if row_start < self._next_row else FailureReason.ARTIFACT_INCOMPLETE
            raise PreprocessError(
                reason,
                f"artifact block starts at {row_start}, expected {self._next_row}",
                state=RunState.RUNTIME_FAULT,
                provenance={"expected_row_start": self._next_row, "actual_row_start": row_start},
            )
        if row_start < 0 or row_end <= row_start or row_end > target_rows:
            raise ValueError("artifact block bounds are outside target grid")
        rows = row_end - row_start
        image = np.asarray(result.image)
        expected_image_shape = tuple(int(value) for value in self.profile.output_shape)
        expected_image_shape = list(expected_image_shape)
        axis_y = self.profile.output_layout.index("Y")
        expected_image_shape[axis_y] = rows
        if tuple(image.shape) != tuple(expected_image_shape) or image.dtype != np.dtype(self.profile.output_dtype):
            raise PreprocessError(
                FailureReason.ARTIFACT_SCHEMA_INVALID,
                "warp image block does not match the output profile",
                state=RunState.RUNTIME_FAULT,
                provenance={"shape": image.shape, "dtype": image.dtype.str, "expected_shape": tuple(expected_image_shape)},
            )
        validity = np.asarray(result.validity_yx)
        reason_mask = np.asarray(result.validity_reason_yx)
        mapping = np.asarray(result.mapping_yx)
        expected_yx = (rows, target_cols)
        if validity.shape != expected_yx or validity.dtype != np.dtype(self.profile.validity_encoding.dtype):
            raise PreprocessError(
                FailureReason.ARTIFACT_SCHEMA_INVALID,
                "validity block does not match the output profile",
                state=RunState.RUNTIME_FAULT,
                provenance={"shape": validity.shape, "dtype": validity.dtype.str, "expected_shape": expected_yx},
            )
        if reason_mask.shape != expected_yx or reason_mask.dtype != np.dtype(self.profile.reason_encoding.dtype):
            raise PreprocessError(
                FailureReason.ARTIFACT_SCHEMA_INVALID,
                "reason block does not match the output profile",
                state=RunState.RUNTIME_FAULT,
                provenance={"shape": reason_mask.shape, "dtype": reason_mask.dtype.str, "expected_shape": expected_yx},
            )
        expected_mapping_dtype = np.dtype(self.profile.internal_numeric_precision)
        if mapping.shape != (*expected_yx, 2) or mapping.dtype != expected_mapping_dtype:
            raise PreprocessError(
                FailureReason.ARTIFACT_SCHEMA_INVALID,
                "mapping block does not match the output profile",
                state=RunState.RUNTIME_FAULT,
                provenance={"shape": mapping.shape, "dtype": mapping.dtype.str, "expected_shape": (*expected_yx, 2)},
            )
        try:
            self._arrays["image"][self._image_slice(row_start, row_end)] = image
            self._arrays["validity"][row_start:row_end, :] = validity
            self._arrays["reason"][row_start:row_end, :] = reason_mask
            self._arrays["mapping"][row_start:row_end, :, :] = mapping
        except (OSError, ValueError, TypeError) as exc:
            raise PreprocessError(
                FailureReason.IO_ERROR,
                "unable to write artifact output block",
                state=RunState.IO_FAULT,
                provenance={"row_start": row_start, "row_end": row_end, "error": str(exc)},
            ) from exc
        self._next_row = row_end

    def _close_payload_arrays(self) -> None:
        arrays = self._arrays
        self._arrays = {}
        for value in arrays.values():
            try:
                _close_memmap(value)
            except (OSError, ValueError) as exc:
                raise PreprocessError(
                    FailureReason.IO_ERROR,
                    "unable to close artifact payload file",
                    state=RunState.IO_FAULT,
                    provenance={"artifact_path": str(self.target), "error": str(exc)},
                ) from exc

    def _output_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": PREPROCESS_ARTIFACT_SCHEMA_VERSION,
            "output_shape": self.profile.output_shape,
            "output_layout": self.profile.output_layout,
            "axes": self.profile.output_layout,
            "dtype": self.profile.output_dtype,
            "channel_schema": self.profile.source_schema.channels,
            "validity": self.profile.validity_encoding.to_mapping(),
            "reason": self.profile.reason_encoding.to_mapping(),
            "files": {
                "image": "model-grid.tif",
                "validity": "validity.tif",
                "reason": "validity-reasons.tif",
                "mapping": "mapping.npy",
            },
        }

    def _preprocess_metadata(self, output_fingerprint: str) -> dict[str, Any]:
        profile = self.profile
        return {
            "schema_version": PREPROCESS_ARTIFACT_SCHEMA_VERSION,
            "source_fingerprint": self.capture.source_fingerprint,
            "output_fingerprint": output_fingerprint,
            "capture_manifest": self.capture.to_mapping(include_digest=True, include_trust=True),
            "preprocessing_profile": profile.to_mapping(include_digest=True, include_trust=True),
            "calibration_bundle": self.calibration.to_mapping(include_digest=True, include_trust=True),
            "source_grid": {
                "axes": profile.source_schema.axes,
                "shape": profile.source_schema.shape,
                "dtype": profile.source_schema.dtype,
                "channels": profile.source_schema.channels,
                "byte_order": profile.source_schema.byte_order,
                "nodata_semantics": profile.source_schema.nodata_semantics,
            },
            "model_grid": profile.target_grid.to_mapping(),
            "mapping": {
                "ref": "mapping.npy",
                "direction": "target_to_source",
                "coordinate_order": "YX",
                "components": ("row", "column"),
                "shape": (profile.target_grid.rows, profile.target_grid.cols, 2),
                "dtype": profile.internal_numeric_precision,
                "pixel_convention": profile.pixel_convention,
            },
            "source_footprint": {
                "source_roi": profile.source_roi.to_mapping(),
                "halo": profile.halo.to_mapping(),
                "coordinate_order": "YX",
            },
            "transform": {
                "type": profile.transform_type,
                "direction": profile.transform_direction,
                "pixel_convention": profile.pixel_convention,
            },
            "validity_policy": {
                "encoding": profile.validity_encoding.to_mapping(),
                "kernel": profile.validity_kernel,
                "support_threshold": profile.support_threshold,
                "border_policy": profile.border_policy,
            },
            "reason_policy": {
                "encoding": profile.reason_encoding.to_mapping(),
                "kernel": profile.reason_kernel,
            },
            "error_bound": {
                "internal_numeric_precision": profile.internal_numeric_precision,
                "calibration_quality": self.calibration.quality,
                "support_threshold": profile.support_threshold,
            },
            "provenance": self._provenance,
        }

    def _payload_checksums(self) -> dict[str, str]:
        missing = [name for name in PAYLOAD_FILES if not (self.staging_path / name).is_file()]
        if missing:
            raise PreprocessError(
                FailureReason.ARTIFACT_INCOMPLETE,
                "artifact staging bundle is missing payload files",
                state=RunState.RUNTIME_FAULT,
                provenance={"missing_files": tuple(missing)},
            )
        return {name: sha256_path(self.staging_path / name) for name in PAYLOAD_FILES}

    def _verify_staging(self, manifest_payload: Mapping[str, Any]) -> None:
        if manifest_payload.get("status") != "complete" or manifest_payload.get("complete") is not True:
            raise PreprocessError(
                FailureReason.ARTIFACT_INCOMPLETE,
                "artifact manifest is not complete",
                state=RunState.RUNTIME_FAULT,
            )
        checksums = manifest_payload.get("checksums")
        if not isinstance(checksums, Mapping) or set(checksums) != set(PAYLOAD_FILES):
            raise PreprocessError(
                FailureReason.ARTIFACT_SCHEMA_INVALID,
                "artifact manifest checksum set is invalid",
                state=RunState.RUNTIME_FAULT,
            )
        for name, expected in checksums.items():
            actual = sha256_path(self.staging_path / name)
            if actual != expected:
                raise PreprocessError(
                    FailureReason.ARTIFACT_CHECKSUM_MISMATCH,
                    f"staging checksum mismatch for {name}",
                    state=RunState.IO_FAULT,
                    provenance={"file": name, "expected": expected, "actual": actual},
                )

    def finalize(self) -> tuple[Path, Any]:
        """Close, checksum, verify, and atomically publish the artifact."""

        if self._closed or self._finalized:
            raise RuntimeError("artifact writer is closed")
        if self._next_row != self.profile.target_grid.rows:
            raise PreprocessError(
                FailureReason.ARTIFACT_INCOMPLETE,
                "artifact output rows are incomplete",
                state=RunState.RUNTIME_FAULT,
                provenance={"written_rows": self._next_row, "target_rows": self.profile.target_grid.rows},
            )
        try:
            self._close_payload_arrays()
            raster_checksums = {
                name: sha256_path(self.staging_path / name)
                for name in ("model-grid.tif", "validity.tif", "validity-reasons.tif", "mapping.npy")
            }
            output_fingerprint = hashlib.sha256(canonical_json(raster_checksums).encode("utf-8")).hexdigest()
            _write_json(self.staging_path / "output.json", self._output_metadata())
            _write_json(self.staging_path / "preprocess.json", self._preprocess_metadata(output_fingerprint))
            checksums = self._payload_checksums()
            artifact_id = f"{_safe_token(self.capture.capture_id)}-{self.capture.source_fingerprint[:16]}"
            provenance = dict(self._provenance)
            provenance.update(
                {
                    "status": "complete",
                    "output_fingerprint": output_fingerprint,
                    "payload_files": PAYLOAD_FILES,
                    "artifact_staging": "same_filesystem_directory_rename",
                }
            )
            manifest_payload = {
                "schema_version": PREPROCESS_ARTIFACT_SCHEMA_VERSION,
                "artifact_id": artifact_id,
                "source_fingerprint": self.capture.source_fingerprint,
                "profile_fingerprint": self.profile.fingerprint,
                "calibration_fingerprint": self.calibration.fingerprint,
                "output_shape": self.profile.output_shape,
                "output_layout": self.profile.output_layout,
                "output_dtype": self.profile.output_dtype,
                "validity_encoding": self.profile.validity_encoding.to_mapping(),
                "reason_encoding": self.profile.reason_encoding.to_mapping(),
                "mapping_ref": "mapping.npy",
                "checksums": checksums,
                "complete": True,
                "status": "complete",
                "files": REQUIRED_ARTIFACT_FILES,
                "provenance": provenance,
            }
            self._verify_staging(manifest_payload)
            from .api import ArtifactManifest

            manifest = ArtifactManifest.from_mapping(manifest_payload)
            _write_json(self.staging_path / "manifest.json", manifest_payload)
            os.replace(self.staging_path, self.target)
            self._finalized = True
            self._closed = True
        except PreprocessError:
            self.abort()
            raise
        except (OSError, ValueError, TypeError) as exc:
            self.abort()
            raise PreprocessError(
                FailureReason.IO_ERROR,
                f"unable to publish artifact: {self.target}",
                state=RunState.IO_FAULT,
                provenance={"artifact_path": str(self.target), "error": str(exc)},
            ) from exc

        return self.target, manifest

    def abort(self) -> None:
        if self._closed and not self.staging_path.exists():
            return
        self._closed = True
        arrays = self._arrays
        self._arrays = {}
        for value in arrays.values():
            try:
                _close_memmap(value)
            except (OSError, ValueError):
                pass
        if self.staging_path.exists():
            shutil.rmtree(self.staging_path, ignore_errors=True)

    def __enter__(self) -> "ArtifactWriter":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if exc_type is not None or not self._finalized:
            self.abort()


__all__ = [
    "ArtifactWriter",
    "PAYLOAD_FILES",
    "REQUIRED_ARTIFACT_FILES",
    "sha256_path",
]
