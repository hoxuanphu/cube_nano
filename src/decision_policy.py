"""Fail-closed decision policy for verified patch-result artifacts."""

from __future__ import annotations

import json
import hashlib
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from patch_result_writer import PatchResultArtifact, open_patch_results
from preprocessing import FailureReason, PreprocessFailure, SafeAction


DECISIONS = ("KEEP_FOR_DOWNLINK", "DELETE_CAPTURE", "RETAIN_FOR_GROUND")


def _now_text() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _window_area(record: Mapping[str, Any]) -> float:
    window = record.get("model_window")
    if isinstance(window, Mapping):
        row_start, row_end = int(window["row_start"]), int(window["row_end"])
        col_start, col_end = int(window["col_start"]), int(window["col_end"])
    else:
        row_start, row_end, col_start, col_end = (int(value) for value in window)
    return float(max(0, row_end - row_start) * max(0, col_end - col_start))


@dataclass(frozen=True)
class DecisionRecord:
    decision: str
    decision_reason: str
    source_fingerprint: str
    patch_result_fingerprint: str
    decision_policy_id: str
    cloud_coverage: float | None
    valid_area: float
    timestamp: str
    safe_action: str = SafeAction.RETAIN_FOR_GROUND.value
    reason_code: str | None = None
    preprocessing_profile_id: str | None = None
    engine_fingerprint: str | None = None
    georef_valid: bool = False
    provenance: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self) -> None:
        if self.decision not in DECISIONS:
            raise ValueError(f"decision must be one of {DECISIONS}")
        if self.safe_action != SafeAction.RETAIN_FOR_GROUND.value:
            raise ValueError("decision records use RETAIN_FOR_GROUND as the safe action")
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    @property
    def actionable(self) -> bool:
        return self.decision in {"KEEP_FOR_DOWNLINK", "DELETE_CAPTURE"}

    @property
    def fingerprint(self) -> str:
        payload = self.to_mapping()
        payload.pop("provenance", None)
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        ).hexdigest()

    def to_mapping(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "decision_reason": self.decision_reason,
            "source_fingerprint": self.source_fingerprint,
            "patch_result_fingerprint": self.patch_result_fingerprint,
            "decision_policy_id": self.decision_policy_id,
            "cloud_coverage": self.cloud_coverage,
            "valid_area": self.valid_area,
            "timestamp": self.timestamp,
            "safe_action": self.safe_action,
            "reason_code": self.reason_code,
            "preprocessing_profile_id": self.preprocessing_profile_id,
            "engine_fingerprint": self.engine_fingerprint,
            "georef_valid": self.georef_valid,
            "provenance": dict(self.provenance),
        }


class DecisionPolicy:
    """Create at most a safe, auditable decision from complete patch results.

    This class never deletes or mutates the source capture.  ``DELETE_CAPTURE``
    is only an instruction for the OBC/F' owner after it persists and checks
    the returned record and source fingerprint.
    """

    def __init__(
        self,
        *,
        policy_id: str = "patch-cloud-coverage-v1",
        cloud_coverage_threshold: float = 0.60,
        min_valid_area: float = 0.0,
        require_georef_valid: bool = True,
    ) -> None:
        if not str(policy_id).strip():
            raise ValueError("policy_id is required")
        if not 0.0 <= float(cloud_coverage_threshold) <= 1.0:
            raise ValueError("cloud_coverage_threshold must be between 0 and 1")
        if float(min_valid_area) < 0.0:
            raise ValueError("min_valid_area must be non-negative")
        if not isinstance(require_georef_valid, bool):
            raise ValueError("require_georef_valid must be boolean")
        self.policy_id = str(policy_id).strip()
        self.cloud_coverage_threshold = float(cloud_coverage_threshold)
        self.min_valid_area = float(min_valid_area)
        self.require_georef_valid = require_georef_valid

    def _record(
        self,
        *,
        decision: str,
        reason: str,
        result: PatchResultArtifact | None,
        source_fingerprint: str = "",
        cloud_coverage: float | None = None,
        valid_area: float = 0.0,
        reason_code: str | None = None,
        georef_valid: bool = False,
        provenance: Mapping[str, Any] | None = None,
    ) -> DecisionRecord:
        manifest = result.manifest if result is not None else {}
        return DecisionRecord(
            decision=decision,
            decision_reason=reason,
            source_fingerprint=source_fingerprint or str(manifest.get("source_fingerprint", "")),
            patch_result_fingerprint=result.fingerprint if result is not None else "",
            decision_policy_id=self.policy_id,
            cloud_coverage=cloud_coverage,
            valid_area=valid_area,
            timestamp=_now_text(),
            reason_code=reason_code,
            preprocessing_profile_id=str(manifest.get("preprocessing_profile_id", "")) or None,
            engine_fingerprint=str(manifest.get("engine_fingerprint", "")) or None,
            georef_valid=georef_valid,
            provenance=provenance or {},
        )

    def evaluate(
        self,
        patch_results: PatchResultArtifact | str | Path | PreprocessFailure,
        *,
        expected_source_fingerprint: str | None = None,
        expected_preprocessing_profile_id: str | None = None,
        expected_engine_fingerprint: str | None = None,
    ) -> DecisionRecord:
        """Evaluate results and return a safe decision record.

        Any incomplete, invalid, stale, or partially inferred result produces
        ``RETAIN_FOR_GROUND`` with a reason code instead of a deletion-like
        decision.
        """

        if isinstance(patch_results, PreprocessFailure):
            return self._record(
                decision="RETAIN_FOR_GROUND",
                reason="preprocess_or_inference_failure",
                result=None,
                source_fingerprint=expected_source_fingerprint or "",
                reason_code=patch_results.reason_code,
                provenance={"failure": patch_results.message, **patch_results.provenance},
            )
        if isinstance(patch_results, (str, Path)):
            opened = open_patch_results(patch_results)
            if isinstance(opened, PreprocessFailure):
                return self._record(
                    decision="RETAIN_FOR_GROUND",
                    reason="patch_result_not_verified",
                    result=None,
                    source_fingerprint=expected_source_fingerprint or "",
                    reason_code=opened.reason_code,
                    provenance={"failure": opened.message, **opened.provenance},
                )
            patch_results = opened
        if not isinstance(patch_results, PatchResultArtifact):
            raise TypeError("evaluate requires a PatchResultArtifact or artifact path")

        manifest = patch_results.manifest
        if expected_source_fingerprint and manifest.get("source_fingerprint") != expected_source_fingerprint:
            return self._record(
                decision="RETAIN_FOR_GROUND",
                reason="source_fingerprint_mismatch",
                result=patch_results,
                source_fingerprint=expected_source_fingerprint,
                reason_code=FailureReason.FINGERPRINT_LINKAGE_MISMATCH.value,
            )
        if expected_preprocessing_profile_id and manifest.get("preprocessing_profile_id") != expected_preprocessing_profile_id:
            return self._record(
                decision="RETAIN_FOR_GROUND",
                reason="preprocessing_profile_mismatch",
                result=patch_results,
                reason_code=FailureReason.FINGERPRINT_LINKAGE_MISMATCH.value,
            )
        if expected_engine_fingerprint and manifest.get("engine_fingerprint") != expected_engine_fingerprint:
            return self._record(
                decision="RETAIN_FOR_GROUND",
                reason="engine_fingerprint_mismatch",
                result=patch_results,
                reason_code=FailureReason.FINGERPRINT_LINKAGE_MISMATCH.value,
            )
        if not patch_results.is_complete:
            return self._record(
                decision="RETAIN_FOR_GROUND",
                reason="patch_result_manifest_incomplete",
                result=patch_results,
                reason_code=FailureReason.PATCH_RESULT_INCOMPLETE.value,
            )
        if not patch_results.records:
            return self._record(
                decision="RETAIN_FOR_GROUND",
                reason="patch_result_has_no_records",
                result=patch_results,
                reason_code=FailureReason.PATCH_RESULT_MISSING.value,
            )

        valid_area = 0.0
        cloud_area = 0.0
        georef_valid = True
        for record in patch_results.records:
            if str(record.get("inference_status", "")) != "valid":
                return self._record(
                    decision="RETAIN_FOR_GROUND",
                    reason="inference_not_complete_for_all_patches",
                    result=patch_results,
                    reason_code=FailureReason.PATCH_RESULT_INCOMPLETE.value,
                    provenance={"patch_row": record.get("patch_row"), "patch_col": record.get("patch_col")},
                )
            try:
                valid_fraction = float(record.get("valid_fraction"))
            except (TypeError, ValueError):
                valid_fraction = -1.0
            if not 0.0 <= valid_fraction <= 1.0:
                return self._record(
                    decision="RETAIN_FOR_GROUND",
                    reason="invalid_valid_fraction",
                    result=patch_results,
                    reason_code=FailureReason.PATCH_RESULT_INCOMPLETE.value,
                )
            if valid_fraction <= 0.0:
                return self._record(
                    decision="RETAIN_FOR_GROUND",
                    reason="patch_has_no_valid_area",
                    result=patch_results,
                    reason_code=FailureReason.PATCH_RESULT_INCOMPLETE.value,
                )
            raw_georef = record.get("georef_valid", False)
            if isinstance(raw_georef, str):
                raw_georef = raw_georef.strip().lower() == "true"
            georef_valid = georef_valid and bool(raw_georef)
            try:
                area = _window_area(record)
            except (KeyError, TypeError, ValueError):
                return self._record(
                    decision="RETAIN_FOR_GROUND",
                    reason="invalid_model_window",
                    result=patch_results,
                    reason_code=FailureReason.PATCH_RESULT_INCOMPLETE.value,
                    georef_valid=georef_valid,
                )
            probability = record.get("cloud_probability")
            try:
                probability = float(probability)
            except (TypeError, ValueError):
                probability = -1.0
            if not 0.0 <= probability <= 1.0:
                return self._record(
                    decision="RETAIN_FOR_GROUND",
                    reason="invalid_cloud_probability",
                    result=patch_results,
                    reason_code=FailureReason.PATCH_RESULT_INCOMPLETE.value,
                    georef_valid=georef_valid,
                )
            raw_label = record.get("cloud_label", "")
            if isinstance(raw_label, bool):
                label = "cloud" if raw_label else "clear"
            else:
                label = str(raw_label).lower()
            if label not in {"cloud", "clear"}:
                return self._record(
                    decision="RETAIN_FOR_GROUND",
                    reason="invalid_cloud_label",
                    result=patch_results,
                    reason_code=FailureReason.PATCH_RESULT_INCOMPLETE.value,
                    georef_valid=georef_valid,
                )
            weighted_area = area * valid_fraction
            valid_area += weighted_area
            if label == "cloud":
                cloud_area += weighted_area

        if valid_area < self.min_valid_area:
            return self._record(
                decision="RETAIN_FOR_GROUND",
                reason="valid_area_below_policy_minimum",
                result=patch_results,
                valid_area=valid_area,
                reason_code=FailureReason.PATCH_RESULT_INCOMPLETE.value,
                georef_valid=georef_valid,
            )
        if self.require_georef_valid and not georef_valid:
            return self._record(
                decision="RETAIN_FOR_GROUND",
                reason="georeferencing_quality_not_valid",
                result=patch_results,
                valid_area=valid_area,
                reason_code="GEOREF_INVALID",
                georef_valid=False,
            )
        coverage = None if valid_area <= 0.0 else float(cloud_area / valid_area)
        if coverage is None:
            return self._record(
                decision="RETAIN_FOR_GROUND",
                reason="valid_area_is_zero",
                result=patch_results,
                valid_area=valid_area,
                reason_code=FailureReason.PATCH_RESULT_INCOMPLETE.value,
                georef_valid=georef_valid,
            )
        if coverage >= self.cloud_coverage_threshold:
            decision = "DELETE_CAPTURE"
            reason = "cloud_coverage_at_or_above_threshold"
        else:
            decision = "KEEP_FOR_DOWNLINK"
            reason = "cloud_coverage_below_threshold"
        return self._record(
            decision=decision,
            reason=reason,
            result=patch_results,
            cloud_coverage=coverage,
            valid_area=valid_area,
            georef_valid=georef_valid,
            provenance={"cloud_area": cloud_area, "threshold": self.cloud_coverage_threshold},
        )

    decide = evaluate

    @staticmethod
    def persist(record: DecisionRecord, target: str | Path) -> Path:
        if not isinstance(record, DecisionRecord):
            raise TypeError("persist requires a DecisionRecord")
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            temporary.write_text(json.dumps(record.to_mapping(), sort_keys=True, indent=2, default=str) + "\n", encoding="utf-8")
            os.replace(temporary, target)
        except OSError:
            temporary.unlink(missing_ok=True)
            raise
        return target


__all__ = ["DECISIONS", "DecisionPolicy", "DecisionRecord"]
