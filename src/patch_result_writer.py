"""Append-only patch-result artifact writer and verified reader.

Patch results are a model/inference artifact, separate from the geometric
preprocess artifact.  Records are written in row-major patch order, flushed
and fsynced on every append, and published only after the complete manifest
and payload checksums have been verified.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4

from preprocessing import FailureReason, PreprocessError, PreprocessFailure, RunState, SafeAction


PATCH_RESULT_SCHEMA_VERSION = 1
RECORDS_FILE = "records.jsonl"
HEADER_FILE = "header.json"
MANIFEST_FILE = "manifest.json"


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=_json_default,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: Any, field_name: str, *, required: bool = False) -> str:
    if value is None or str(value).strip() == "":
        if required:
            raise ValueError(f"{field_name} is required")
        return ""
    result = str(value).strip().lower()
    if len(result) != 64:
        raise ValueError(f"{field_name} must be a SHA-256 digest")
    try:
        bytes.fromhex(result)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a SHA-256 digest") from exc
    return result


def _text(value: Any, field_name: str, *, required: bool = True) -> str:
    result = "" if value is None else str(value).strip()
    if required and not result:
        raise ValueError(f"{field_name} is required")
    return result


def _pair(value: Any, field_name: str, *, allow_none: bool = False) -> tuple[int, int] | None:
    if value is None and allow_none:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
        raise ValueError(f"{field_name} must contain two positive integers")
    result = tuple(int(item) for item in value)
    if any(item <= 0 for item in result):
        raise ValueError(f"{field_name} must contain two positive integers")
    return result


def _window(value: Any) -> tuple[int, int, int, int]:
    if isinstance(value, Mapping):
        value = (
            value.get("row_start"),
            value.get("row_end"),
            value.get("col_start"),
            value.get("col_end"),
        )
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 4:
        raise ValueError("model_window must contain four integer bounds")
    result = tuple(int(item) for item in value)
    if result[0] < 0 or result[2] < 0 or result[1] <= result[0] or result[3] <= result[2]:
        raise ValueError("model_window bounds are invalid")
    return result


def _now_text() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class PatchResultArtifact:
    path: Path
    manifest: Mapping[str, Any]
    records: tuple[Mapping[str, Any], ...]

    @property
    def is_complete(self) -> bool:
        return bool(self.manifest.get("complete") is True and self.manifest.get("status") == "complete")

    @property
    def fingerprint(self) -> str:
        return str(self.manifest.get("result_fingerprint", ""))

    @property
    def source_fingerprint(self) -> str:
        return str(self.manifest.get("source_fingerprint", ""))

    def __enter__(self) -> "PatchResultArtifact":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        return None


class PatchResultWriter:
    """Write a complete row-major patch-result artifact.

    ``target`` is a directory containing ``records.jsonl``, ``header.json``
    and ``manifest.json`` after publication.  A staging directory is retained
    on an interrupted run so a caller can reopen with ``resume=True``.
    """

    def __init__(
        self,
        target: str | Path,
        *,
        capture_id: str,
        source_fingerprint: str,
        preprocessing_profile_id: str,
        engine_fingerprint: str,
        patch_grid_shape: Sequence[int] | None = None,
        patch_size: Sequence[int] | None = None,
        compute_profile_id: str | None = None,
        compute_profile_fingerprint: str | None = None,
        preprocessing_profile_fingerprint: str | None = None,
        input_spec_id: str | None = None,
        threshold: float | None = None,
        decision_policy_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        resume: bool = False,
    ) -> None:
        self.target = Path(target)
        self.capture_id = _text(capture_id, "capture_id")
        self.source_fingerprint = _digest(source_fingerprint, "source_fingerprint", required=True)
        self.preprocessing_profile_id = _text(preprocessing_profile_id, "preprocessing_profile_id")
        self.preprocessing_profile_fingerprint = _digest(
            preprocessing_profile_fingerprint,
            "preprocessing_profile_fingerprint",
        )
        self.engine_fingerprint = _digest(engine_fingerprint, "engine_fingerprint", required=True)
        self.compute_profile_id = _text(compute_profile_id, "compute_profile_id", required=False)
        self.compute_profile_fingerprint = _digest(compute_profile_fingerprint, "compute_profile_fingerprint")
        self.input_spec_id = _text(input_spec_id, "input_spec_id", required=False)
        self.patch_grid_shape = _pair(patch_grid_shape, "patch_grid_shape", allow_none=True)
        self.patch_size = _pair(patch_size, "patch_size", allow_none=True)
        if threshold is not None and not 0.0 <= float(threshold) <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        self.threshold = None if threshold is None else float(threshold)
        self.decision_policy_id = _text(decision_policy_id, "decision_policy_id", required=False)
        self.metadata = dict(metadata or {})
        self._closed = False
        self._finalized = False
        self._records: list[dict[str, Any]] = []
        self._coordinates: set[tuple[int, int]] = set()
        self._next_index = 0
        self.staging_path = self._resolve_staging(resume=resume)
        self.records_path = self.staging_path / RECORDS_FILE
        self.header_path = self.staging_path / HEADER_FILE
        self.manifest_path = self.staging_path / MANIFEST_FILE
        self.staging_path.mkdir(parents=True, exist_ok=True)
        if resume and self.records_path.exists():
            self._load_existing()
        else:
            self._write_header()
        self._stream = self.records_path.open("a", encoding="utf-8", newline="\n")

    def _resolve_staging(self, *, resume: bool) -> Path:
        self.target.parent.mkdir(parents=True, exist_ok=True)
        if self.target.exists() and not resume:
            raise FileExistsError(f"patch-result target already exists: {self.target}")
        candidates = sorted(self.target.parent.glob(f".{self.target.name}.staging-*"))
        if resume:
            if len(candidates) > 1:
                raise PreprocessError(
                    FailureReason.PATCH_RESULT_INCOMPLETE,
                    "multiple patch-result staging directories require operator selection",
                    state=RunState.IO_FAULT,
                    provenance={"candidates": tuple(str(item) for item in candidates)},
                )
            if candidates:
                return candidates[0]
        return self.target.parent / f".{self.target.name}.staging-{uuid4().hex}"

    def _header_payload(self) -> dict[str, Any]:
        return {
            "schema_version": PATCH_RESULT_SCHEMA_VERSION,
            "capture_id": self.capture_id,
            "source_fingerprint": self.source_fingerprint,
            "preprocessing_profile_id": self.preprocessing_profile_id,
            "preprocessing_profile_fingerprint": self.preprocessing_profile_fingerprint,
            "compute_profile_id": self.compute_profile_id,
            "compute_profile_fingerprint": self.compute_profile_fingerprint,
            "engine_fingerprint": self.engine_fingerprint,
            "input_spec_id": self.input_spec_id,
            "patch_grid_shape": self.patch_grid_shape,
            "patch_size": self.patch_size,
            "threshold": self.threshold,
            "decision_policy_id": self.decision_policy_id,
            "created_at": _now_text(),
            "metadata": self.metadata,
        }

    def _write_header(self) -> None:
        payload = _canonical_json(self._header_payload()) + "\n"
        self.header_path.write_text(payload, encoding="utf-8")

    def _load_existing(self) -> None:
        try:
            header = json.loads(self.header_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise PreprocessError(
                FailureReason.PATCH_RESULT_INCOMPLETE,
                "patch-result staging header is missing or invalid",
                state=RunState.IO_FAULT,
                provenance={"staging_path": str(self.staging_path), "error": str(exc)},
            ) from exc
        self._validate_header(header)
        try:
            with self.records_path.open("r", encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise PreprocessError(
                            FailureReason.PATCH_RESULT_CHECKSUM_MISMATCH,
                            "patch-result staging contains malformed JSON",
                            state=RunState.IO_FAULT,
                            provenance={"line": line_number, "error": str(exc)},
                        ) from exc
                    normalized = self._validate_record(record)
                    coordinate = (normalized["patch_row"], normalized["patch_col"])
                    if coordinate in self._coordinates:
                        raise PreprocessError(
                            FailureReason.PATCH_RESULT_DUPLICATE,
                            "duplicate patch record in staging",
                            state=RunState.RUNTIME_FAULT,
                            provenance={"coordinate": coordinate},
                        )
                    expected = self._expected_coordinate(self._next_index)
                    if expected is not None and coordinate != expected:
                        raise PreprocessError(
                            FailureReason.PATCH_RESULT_MISSING,
                            "patch-result staging has a missing or out-of-order record",
                            state=RunState.RUNTIME_FAULT,
                            provenance={"expected": expected, "actual": coordinate},
                        )
                    self._records.append(normalized)
                    self._coordinates.add(coordinate)
                    self._next_index += 1
        except OSError as exc:
            raise PreprocessError(
                FailureReason.IO_ERROR,
                "unable to read patch-result staging records",
                state=RunState.IO_FAULT,
                provenance={"error": str(exc)},
            ) from exc

    def _validate_header(self, header: Mapping[str, Any]) -> None:
        if header.get("schema_version") != PATCH_RESULT_SCHEMA_VERSION:
            raise PreprocessError(
                FailureReason.PATCH_RESULT_INCOMPLETE,
                "unsupported patch-result schema version",
                state=RunState.INVALID_INPUT,
            )
        expected = json.loads(_canonical_json(self._header_payload()))
        for key in (
            "capture_id",
            "source_fingerprint",
            "preprocessing_profile_id",
            "engine_fingerprint",
            "patch_grid_shape",
            "patch_size",
        ):
            if header.get(key) != expected.get(key):
                raise PreprocessError(
                    FailureReason.FINGERPRINT_LINKAGE_MISMATCH,
                    f"patch-result staging header field {key} does not match request",
                    state=RunState.INVALID_INPUT,
                    provenance={"expected": expected.get(key), "actual": header.get(key)},
                )

    def _expected_coordinate(self, index: int) -> tuple[int, int] | None:
        if self.patch_grid_shape is None:
            return None
        rows, cols = self.patch_grid_shape
        total = rows * cols
        if index >= total:
            return None
        return index // cols, index % cols

    def _validate_record(self, record: Any) -> dict[str, Any]:
        if hasattr(record, "__dataclass_fields__"):
            record = asdict(record)
        if not isinstance(record, Mapping):
            raise PreprocessError(
                FailureReason.PATCH_RESULT_INCOMPLETE,
                "patch result must be a mapping",
                state=RunState.INVALID_INPUT,
            )
        normalized = dict(record)
        try:
            normalized["patch_row"] = int(normalized["patch_row"])
            normalized["patch_col"] = int(normalized["patch_col"])
            if normalized["patch_row"] < 0 or normalized["patch_col"] < 0:
                raise ValueError("patch coordinates must be non-negative")
            normalized["model_window"] = _window(normalized["model_window"])
            normalized["valid_fraction"] = float(normalized["valid_fraction"])
            if not 0.0 <= normalized["valid_fraction"] <= 1.0:
                raise ValueError("valid_fraction must be between 0 and 1")
            normalized["inference_status"] = _text(normalized["inference_status"], "inference_status")
            normalized["source_mapping_ref"] = _text(normalized["source_mapping_ref"], "source_mapping_ref")
            summary = normalized.get("validity_reason_summary", {})
            if not isinstance(summary, Mapping):
                raise ValueError("validity_reason_summary must be a mapping")
            normalized["validity_reason_summary"] = {str(key): int(value) for key, value in summary.items()}
            if any(value < 0 for value in normalized["validity_reason_summary"].values()):
                raise ValueError("validity reason counts must be non-negative")
            normalized.setdefault("capture_id", self.capture_id)
            normalized.setdefault("preprocessing_profile_id", self.preprocessing_profile_id)
            normalized.setdefault("engine_fingerprint", self.engine_fingerprint)
            normalized.setdefault("timestamp", _now_text())
            if normalized["capture_id"] != self.capture_id:
                raise ValueError("record capture_id does not match writer")
            if normalized["preprocessing_profile_id"] != self.preprocessing_profile_id:
                raise ValueError("record preprocessing_profile_id does not match writer")
            if _digest(normalized["engine_fingerprint"], "record engine_fingerprint", required=True) != self.engine_fingerprint:
                raise ValueError("record engine_fingerprint does not match writer")
            if normalized.get("cloud_probability") is not None:
                normalized["cloud_probability"] = float(normalized["cloud_probability"])
                if not 0.0 <= normalized["cloud_probability"] <= 1.0:
                    raise ValueError("cloud_probability must be between 0 and 1")
        except (KeyError, TypeError, ValueError) as exc:
            raise PreprocessError(
                FailureReason.PATCH_RESULT_INCOMPLETE,
                f"invalid patch result record: {exc}",
                state=RunState.INVALID_INPUT,
            ) from exc
        return normalized

    @property
    def record_count(self) -> int:
        return len(self._records)

    def append(self, record: Mapping[str, Any] | Any) -> Mapping[str, Any]:
        if self._closed:
            raise RuntimeError("patch-result writer is closed")
        normalized = self._validate_record(record)
        coordinate = (normalized["patch_row"], normalized["patch_col"])
        if coordinate in self._coordinates:
            raise PreprocessError(
                FailureReason.PATCH_RESULT_DUPLICATE,
                "duplicate patch result coordinate",
                state=RunState.RUNTIME_FAULT,
                provenance={"coordinate": coordinate},
            )
        expected = self._expected_coordinate(self._next_index)
        if expected is not None and coordinate != expected:
            reason = FailureReason.PATCH_RESULT_MISSING if coordinate > expected else FailureReason.PATCH_RESULT_DUPLICATE
            raise PreprocessError(
                reason,
                "patch result coordinates must be appended in row-major order",
                state=RunState.RUNTIME_FAULT,
                provenance={"expected": expected, "actual": coordinate},
            )
        payload = _canonical_json(normalized) + "\n"
        try:
            self._stream.write(payload)
            self._stream.flush()
            os.fsync(self._stream.fileno())
        except OSError as exc:
            raise PreprocessError(
                FailureReason.IO_ERROR,
                "unable to append patch result",
                state=RunState.IO_FAULT,
                provenance={"error": str(exc), "coordinate": coordinate},
            ) from exc
        self._records.append(normalized)
        self._coordinates.add(coordinate)
        self._next_index += 1
        return MappingProxyType(dict(normalized))

    def _missing_coordinates(self) -> tuple[tuple[int, int], ...]:
        if self.patch_grid_shape is None:
            return ()
        rows, cols = self.patch_grid_shape
        return tuple(
            (row, col)
            for row in range(rows)
            for col in range(cols)
            if (row, col) not in self._coordinates
        )

    def finalize(self) -> PatchResultArtifact:
        if self._closed:
            raise RuntimeError("patch-result writer is closed")
        missing = self._missing_coordinates()
        if missing:
            raise PreprocessError(
                FailureReason.PATCH_RESULT_MISSING,
                "patch-result grid is missing records",
                state=RunState.RUNTIME_FAULT,
                provenance={"missing": missing},
            )
        if self.patch_grid_shape is not None and len(self._records) != self.patch_grid_shape[0] * self.patch_grid_shape[1]:
            raise PreprocessError(
                FailureReason.PATCH_RESULT_INCOMPLETE,
                "patch-result record count does not match patch grid",
                state=RunState.RUNTIME_FAULT,
                provenance={"record_count": len(self._records), "expected": self.patch_grid_shape[0] * self.patch_grid_shape[1]},
            )
        try:
            self._stream.flush()
            os.fsync(self._stream.fileno())
            self._stream.close()
            records_checksum = _sha256(self.records_path)
            header_checksum = _sha256(self.header_path)
            result_fingerprint = hashlib.sha256(
                _canonical_json(
                    {
                        "records": records_checksum,
                        "header": header_checksum,
                        "record_count": len(self._records),
                    }
                ).encode("utf-8")
            ).hexdigest()
            manifest = {
                "schema_version": PATCH_RESULT_SCHEMA_VERSION,
                "capture_id": self.capture_id,
                "source_fingerprint": self.source_fingerprint,
                "preprocessing_profile_id": self.preprocessing_profile_id,
                "preprocessing_profile_fingerprint": self.preprocessing_profile_fingerprint,
                "compute_profile_id": self.compute_profile_id,
                "compute_profile_fingerprint": self.compute_profile_fingerprint,
                "engine_fingerprint": self.engine_fingerprint,
                "input_spec_id": self.input_spec_id,
                "patch_grid_shape": self.patch_grid_shape,
                "patch_size": self.patch_size,
                "threshold": self.threshold,
                "decision_policy_id": self.decision_policy_id,
                "record_count": len(self._records),
                "checksums": {RECORDS_FILE: records_checksum, HEADER_FILE: header_checksum},
                "result_fingerprint": result_fingerprint,
                "complete": True,
                "status": "complete",
                "created_at": _now_text(),
            }
            temporary = self.staging_path / f".{MANIFEST_FILE}.{uuid4().hex}.tmp"
            temporary.write_text(_canonical_json(manifest) + "\n", encoding="utf-8")
            with temporary.open("r+b") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, self.manifest_path)
            if self.target.exists():
                raise FileExistsError(f"patch-result target already exists: {self.target}")
            os.replace(self.staging_path, self.target)
            self._closed = True
            self._finalized = True
            return PatchResultArtifact(self.target, MappingProxyType(manifest), tuple(MappingProxyType(item) for item in self._records))
        except PreprocessError:
            self.abort(remove_staging=False)
            raise
        except (OSError, ValueError, TypeError) as exc:
            self.abort(remove_staging=False)
            raise PreprocessError(
                FailureReason.IO_ERROR,
                "unable to publish patch-result artifact",
                state=RunState.IO_FAULT,
                provenance={"target": str(self.target), "error": str(exc)},
            ) from exc

    def abort(self, *, remove_staging: bool = False) -> None:
        if not self._closed:
            self._closed = True
            try:
                self._stream.close()
            except (AttributeError, OSError):
                pass
        if remove_staging and self.staging_path.exists():
            shutil.rmtree(self.staging_path, ignore_errors=True)

    def __enter__(self) -> "PatchResultWriter":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if exc_type is not None or not self._finalized:
            self.abort(remove_staging=False)


def _verify_records(path: Path, manifest: Mapping[str, Any], header: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    grid_shape = _pair(manifest.get("patch_grid_shape"), "patch_grid_shape", allow_none=True)
    expected = manifest.get("record_count")
    records: list[Mapping[str, Any]] = []
    coordinates: set[tuple[int, int]] = set()
    try:
        with (path / RECORDS_FILE).open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, Mapping):
                    raise ValueError("record is not an object")
                coordinate = (int(record["patch_row"]), int(record["patch_col"]))
                if coordinate in coordinates:
                    raise PreprocessError(
                        FailureReason.PATCH_RESULT_DUPLICATE,
                        "duplicate patch result record",
                        state=RunState.RUNTIME_FAULT,
                        provenance={"coordinate": coordinate, "line": line_number},
                    )
                index = len(records)
                if grid_shape is not None:
                    expected_coordinate = (index // grid_shape[1], index % grid_shape[1])
                    if coordinate != expected_coordinate:
                        raise PreprocessError(
                            FailureReason.PATCH_RESULT_MISSING,
                            "patch result contains missing or out-of-order records",
                            state=RunState.RUNTIME_FAULT,
                            provenance={"expected": expected_coordinate, "actual": coordinate},
                        )
                coordinates.add(coordinate)
                records.append(MappingProxyType(dict(record)))
    except PreprocessError:
        raise
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise PreprocessError(
            FailureReason.PATCH_RESULT_INCOMPLETE,
            "patch-result records are malformed",
            state=RunState.IO_FAULT,
            provenance={"error": str(exc)},
        ) from exc
    if expected != len(records):
        reason = FailureReason.PATCH_RESULT_MISSING if expected and len(records) < int(expected) else FailureReason.PATCH_RESULT_INCOMPLETE
        raise PreprocessError(
            reason,
            "patch-result record count does not match manifest",
            state=RunState.RUNTIME_FAULT,
            provenance={"expected": expected, "actual": len(records)},
        )
    if grid_shape is not None and len(records) != grid_shape[0] * grid_shape[1]:
        raise PreprocessError(
            FailureReason.PATCH_RESULT_MISSING,
            "patch-result grid is incomplete",
            state=RunState.RUNTIME_FAULT,
            provenance={"expected": grid_shape[0] * grid_shape[1], "actual": len(records)},
        )
    return tuple(records)


def verify_patch_results(path: str | Path) -> PatchResultArtifact:
    path = Path(path)
    if not path.is_dir():
        raise PreprocessError(
            FailureReason.PATCH_RESULT_INCOMPLETE,
            "patch-result artifact directory does not exist",
            state=RunState.IO_FAULT,
            provenance={"path": str(path)},
        )
    try:
        manifest = json.loads((path / MANIFEST_FILE).read_text(encoding="utf-8"))
        header = json.loads((path / HEADER_FILE).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreprocessError(
            FailureReason.PATCH_RESULT_INCOMPLETE,
            "patch-result manifest or header is missing/invalid",
            state=RunState.IO_FAULT,
            provenance={"path": str(path), "error": str(exc)},
        ) from exc
    if not isinstance(manifest, Mapping) or not isinstance(header, Mapping):
        raise PreprocessError(
            FailureReason.PATCH_RESULT_INCOMPLETE,
            "patch-result manifest and header must be JSON objects",
            state=RunState.RUNTIME_FAULT,
            provenance={"path": str(path)},
        )
    if manifest.get("schema_version") != PATCH_RESULT_SCHEMA_VERSION or manifest.get("status") != "complete" or manifest.get("complete") is not True:
        raise PreprocessError(
            FailureReason.PATCH_RESULT_INCOMPLETE,
            "patch-result manifest is not complete",
            state=RunState.RUNTIME_FAULT,
            provenance={"status": manifest.get("status"), "complete": manifest.get("complete")},
        )
    checksums = manifest.get("checksums")
    if not isinstance(checksums, Mapping):
        raise PreprocessError(
            FailureReason.PATCH_RESULT_INCOMPLETE,
            "patch-result manifest checksums are invalid",
            state=RunState.RUNTIME_FAULT,
        )
    for name in (RECORDS_FILE, HEADER_FILE):
        expected = checksums.get(name)
        try:
            actual = _sha256(path / name)
        except OSError as exc:
            raise PreprocessError(
                FailureReason.PATCH_RESULT_INCOMPLETE,
                f"patch-result payload file is missing: {name}",
                state=RunState.IO_FAULT,
                provenance={"file": name, "error": str(exc)},
            ) from exc
        if not expected or actual != expected:
            raise PreprocessError(
                FailureReason.PATCH_RESULT_CHECKSUM_MISMATCH,
                f"patch-result checksum mismatch for {name}",
                state=RunState.IO_FAULT,
                provenance={"file": name, "expected": expected},
            )
    for key in ("capture_id", "source_fingerprint", "preprocessing_profile_id", "engine_fingerprint"):
        if manifest.get(key) != header.get(key):
            raise PreprocessError(
                FailureReason.FINGERPRINT_LINKAGE_MISMATCH,
                f"patch-result header/manifest mismatch for {key}",
                state=RunState.INVALID_INPUT,
                provenance={"manifest": manifest.get(key), "header": header.get(key)},
            )
    records = _verify_records(path, manifest, header)
    return PatchResultArtifact(path, MappingProxyType(dict(manifest)), records)


def open_patch_results(path: str | Path) -> PatchResultArtifact | PreprocessFailure:
    try:
        return verify_patch_results(path)
    except PreprocessError as error:
        return PreprocessFailure(
            state=error.state,
            reason_code=error.reason_code,
            safe_action=SafeAction.RETAIN_FOR_GROUND,
            message=str(error),
            provenance=error.provenance,
        )


__all__ = [
    "HEADER_FILE",
    "MANIFEST_FILE",
    "PATCH_RESULT_SCHEMA_VERSION",
    "PatchResultArtifact",
    "PatchResultWriter",
    "RECORDS_FILE",
    "open_patch_results",
    "verify_patch_results",
]
