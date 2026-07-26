"""Model-side adapter for verified preprocessing artifacts.

The preprocessing package owns geometry and spatial validity.  This module is
the only place where a model contract is applied: bands are reordered,
windows are padded, normalization is evaluated, and tensors are laid out for
an inference runtime.  It intentionally never opens a raw image or modifies
the materialized artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from input_contract import NormalizationSpec
from preprocessing import (
    FailureReason,
    ModelCompatibilityProfile,
    PreprocessArtifact,
    PreprocessError,
    PreprocessFailure,
    PreprocessingProfile,
    RunState,
    SafeAction,
)


def _now_text() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _digest(value: Any, field_name: str) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    if len(text) != 64:
        raise ValueError(f"{field_name} must be a SHA-256 digest")
    try:
        bytes.fromhex(text)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a SHA-256 digest") from exc
    return text


def _as_pair(value: Any, field_name: str) -> tuple[int, int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
        raise ValueError(f"{field_name} must contain two positive integers")
    result = tuple(int(item) for item in value)
    if any(item <= 0 for item in result):
        raise ValueError(f"{field_name} must contain two positive integers")
    return result


def _as_window(value: Any) -> tuple[int, int, int, int]:
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


class InferenceAdapterError(PreprocessError):
    """Expected adapter failure that must be handled as a safe result."""


@dataclass(frozen=True)
class CompatibilityReport:
    compatible: bool
    reasons: tuple[str, ...] = ()
    artifact_profile_fingerprint: str = ""
    engine_fingerprint: str = ""
    output_layout: str = ""
    output_dtype: str = ""
    spatial_semantics: str = ""
    profile_id: str = ""


@dataclass(frozen=True)
class AdaptedPatch:
    """One model window derived from an artifact block."""

    patch_row: int
    patch_col: int
    model_window: tuple[int, int, int, int]
    tensor: np.ndarray | None
    valid_fraction: float
    validity_reason_summary: Mapping[str, int]
    validity_yx: np.ndarray
    validity_reason_yx: np.ndarray
    source_mapping: np.ndarray
    source_mapping_ref: str
    inference_status: str
    reason_code: str | None = None

    @property
    def is_valid(self) -> bool:
        return self.inference_status == "ready" and self.tensor is not None


@dataclass(frozen=True)
class InferenceResult:
    """Deterministic patch-level inference output."""

    records: tuple[Mapping[str, Any], ...]
    status: str
    cloud_coverage: float | None
    valid_area: float
    processed_count: int
    valid_count: int
    invalid_count: int
    preprocessing_profile_id: str
    engine_fingerprint: str
    source_fingerprint: str
    patch_result: Any = None
    provenance: Mapping[str, Any] = MappingProxyType({})

    @property
    def is_failure(self) -> bool:
        return False


class InferenceAdapter:
    """Apply a :class:`ModelCompatibilityProfile` to a verified artifact.

    ``engine`` may be an already-created object exposing ``infer_batch`` or
    ``infer``.  ``engine_factory`` is preferred for production because it is
    invoked only after the artifact compatibility gate passes.
    """

    def __init__(
        self,
        model_profile: ModelCompatibilityProfile | Mapping[str, Any] | None = None,
        *,
        input_spec: Any = None,
        engine: Any = None,
        engine_factory: Callable[[], Any] | None = None,
        engine_fingerprint: str | None = None,
        runtime_fingerprint: str | None = None,
        accepted_output_layouts: Iterable[str] = ("YXC", "CYX", "YX"),
        accepted_output_dtypes: Iterable[str] | None = None,
        accepted_spatial_semantics: Iterable[str] | None = None,
        min_valid_fraction: float = 0.0,
        threshold: float = 0.5,
        pad_value: float = 0.0,
    ) -> None:
        if model_profile is not None and input_spec is not None:
            raise ValueError("provide model_profile or input_spec, not both")
        if input_spec is not None:
            model_profile = self._adapt_legacy_input_spec(
                input_spec,
                engine_fingerprint=engine_fingerprint,
                runtime_fingerprint=runtime_fingerprint,
            )
        if isinstance(model_profile, Mapping):
            model_profile = ModelCompatibilityProfile.from_mapping(model_profile)
        if not isinstance(model_profile, ModelCompatibilityProfile):
            raise ValueError("model_profile is required")
        if engine is not None and engine_factory is not None:
            raise ValueError("provide engine or engine_factory, not both")
        if not 0.0 <= float(min_valid_fraction) <= 1.0:
            raise ValueError("min_valid_fraction must be between 0 and 1")
        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        self.model_profile = model_profile
        self.engine = engine
        self.engine_factory = engine_factory
        self.engine_fingerprint = _digest(engine_fingerprint, "engine_fingerprint") if engine_fingerprint else model_profile.model_fingerprint
        if self.engine_fingerprint and self.engine_fingerprint != model_profile.model_fingerprint:
            raise ValueError("engine_fingerprint must match model_profile.model_fingerprint")
        self.runtime_fingerprint = (
            _digest(runtime_fingerprint, "runtime_fingerprint")
            if runtime_fingerprint
            else model_profile.runtime_fingerprint
        )
        if self.runtime_fingerprint != model_profile.runtime_fingerprint:
            raise ValueError("runtime_fingerprint must match model_profile.runtime_fingerprint")
        self.accepted_output_layouts = frozenset(str(item).upper() for item in accepted_output_layouts)
        self.accepted_output_dtypes = (
            frozenset(np.dtype(item).name for item in accepted_output_dtypes)
            if accepted_output_dtypes is not None
            else None
        )
        self.accepted_spatial_semantics = (
            frozenset(str(item) for item in accepted_spatial_semantics)
            if accepted_spatial_semantics is not None
            else None
        )
        self.min_valid_fraction = float(min_valid_fraction)
        self.threshold = float(threshold)
        self.pad_value = float(pad_value)
        self._normalizer: NormalizationSpec | None = None

    @staticmethod
    def _adapt_legacy_input_spec(input_spec: Any, *, engine_fingerprint: str | None, runtime_fingerprint: str | None):
        if not hasattr(input_spec, "to_model_compatibility_profile"):
            raise ValueError("input_spec must be a legacy EngineInputSpec")
        if getattr(input_spec, "engine_digest", None) is None:
            if engine_fingerprint is None:
                raise ValueError("legacy input_spec requires engine_fingerprint")
            from dataclasses import replace

            input_spec = replace(input_spec, engine_digest=engine_fingerprint)
        if runtime_fingerprint is None:
            raise ValueError("legacy input_spec requires runtime_fingerprint")
        return input_spec.to_model_compatibility_profile(runtime_fingerprint=runtime_fingerprint)

    def _compatibility_reasons(self, artifact: PreprocessArtifact, profile: PreprocessingProfile) -> list[str]:
        reasons = list(self.model_profile.accepts(profile))
        manifest = artifact.manifest
        if not manifest.complete:
            reasons.append("artifact manifest is not complete")
        if manifest.profile_fingerprint != profile.fingerprint:
            reasons.append("artifact profile fingerprint does not match profile metadata")
        if profile.output_layout not in self.accepted_output_layouts:
            reasons.append(f"unsupported artifact output layout: {profile.output_layout}")
        if self.accepted_output_dtypes is not None and profile.output_dtype not in self.accepted_output_dtypes:
            reasons.append(f"unsupported artifact output dtype: {profile.output_dtype}")
        semantics = profile.target_grid.spatial_semantics
        if self.accepted_spatial_semantics is not None and semantics not in self.accepted_spatial_semantics:
            reasons.append(f"unsupported artifact grid semantics: {semantics}")
        layout = self.model_profile.tensor_layout
        if "H" not in layout or "W" not in layout:
            reasons.append("tensor layout must contain H and W axes")
        if self.model_profile.batch_size <= 0:
            reasons.append("model batch size must be positive")
        if self.engine_fingerprint and self.engine_fingerprint != self.model_profile.model_fingerprint:
            reasons.append("engine fingerprint does not match model profile")
        return reasons

    def compatibility_report(self, artifact: PreprocessArtifact) -> CompatibilityReport:
        if not isinstance(artifact, PreprocessArtifact):
            raise TypeError("compatibility_report requires a PreprocessArtifact")
        reader = artifact.open()
        if isinstance(reader, PreprocessFailure):
            return CompatibilityReport(
                compatible=False,
                reasons=(reader.reason_code, reader.message),
                artifact_profile_fingerprint=artifact.manifest.profile_fingerprint,
                engine_fingerprint=self.engine_fingerprint,
            )
        try:
            profile = reader.profile
            reasons = self._compatibility_reasons(artifact, profile)
            return CompatibilityReport(
                compatible=not reasons,
                reasons=tuple(reasons),
                artifact_profile_fingerprint=profile.fingerprint,
                engine_fingerprint=self.engine_fingerprint,
                output_layout=profile.output_layout,
                output_dtype=profile.output_dtype,
                spatial_semantics=profile.target_grid.spatial_semantics,
                profile_id=profile.profile_id,
            )
        finally:
            reader.close()

    def check_compatibility(self, artifact: PreprocessArtifact) -> CompatibilityReport:
        report = self.compatibility_report(artifact)
        if not report.compatible:
            raise InferenceAdapterError(
                FailureReason.SCHEMA_MISMATCH,
                "preprocessed artifact is incompatible with the model contract: "
                + "; ".join(report.reasons),
                state=RunState.INVALID_INPUT,
                provenance={"compatibility": report},
            )
        return report

    @staticmethod
    def _to_hwc(image: np.ndarray, profile: PreprocessingProfile) -> np.ndarray:
        if profile.output_layout == "YXC":
            return np.asarray(image)
        if profile.output_layout == "CYX":
            return np.moveaxis(np.asarray(image), 0, -1)
        if profile.output_layout == "YX":
            return np.asarray(image)[..., None]
        raise InferenceAdapterError(
            FailureReason.SCHEMA_MISMATCH,
            f"unsupported artifact output layout {profile.output_layout}",
            state=RunState.INVALID_INPUT,
        )

    @staticmethod
    def _pad_array(array: np.ndarray, target_shape: tuple[int, int], policy: str, *, constant: Any) -> np.ndarray:
        height, width = array.shape[:2]
        pad_height = target_shape[0] - height
        pad_width = target_shape[1] - width
        if pad_height < 0 or pad_width < 0:
            raise ValueError("target patch shape is smaller than source window")
        if pad_height == 0 and pad_width == 0:
            return np.asarray(array)
        pad_spec = ((0, pad_height), (0, pad_width)) + ((0, 0),) * (array.ndim - 2)
        if policy == "constant":
            return np.pad(array, pad_spec, mode="constant", constant_values=constant)
        if policy in {"edge", "reflect"}:
            try:
                return np.pad(array, pad_spec, mode=policy)
            except ValueError as exc:
                raise InferenceAdapterError(
                    FailureReason.SCHEMA_MISMATCH,
                    f"unable to apply {policy} padding to artifact window",
                    state=RunState.INVALID_INPUT,
                    provenance={"error": str(exc)},
                ) from exc
        raise ValueError(f"unsupported padding policy {policy}")

    @staticmethod
    def _reason_summary(reason: np.ndarray, profile: PreprocessingProfile) -> Mapping[str, int]:
        result: dict[str, int] = {}
        reason_array = np.asarray(reason)
        for name, bit in profile.reason_encoding.bits.items():
            count = int(np.count_nonzero((reason_array & bit) != 0))
            if count:
                result[name] = count
        return MappingProxyType(result)

    def _normalization_spec(self, channels: int) -> NormalizationSpec:
        if self._normalizer is None:
            try:
                normalization = self.model_profile.normalization
                if isinstance(normalization, Mapping):
                    def thaw(value: Any) -> Any:
                        if isinstance(value, Mapping):
                            return {str(key): thaw(item) for key, item in value.items()}
                        if isinstance(value, tuple):
                            return [thaw(item) for item in value]
                        return value

                    normalization = thaw(normalization)
                self._normalizer = NormalizationSpec.from_value(normalization, channels)
            except (TypeError, ValueError) as exc:
                raise InferenceAdapterError(
                    FailureReason.SCHEMA_MISMATCH,
                    f"model normalization is invalid: {exc}",
                    state=RunState.INVALID_INPUT,
                ) from exc
        return self._normalizer

    def _tensor(self, image_hwc: np.ndarray, profile: PreprocessingProfile) -> np.ndarray:
        required = self.model_profile.required_band_order
        source_channels = tuple(profile.source_schema.channels)
        indices = [source_channels.index(name) for name in required]
        image_hwc = np.asarray(image_hwc)
        if image_hwc.shape[-1] != len(source_channels):
            raise InferenceAdapterError(
                FailureReason.SCHEMA_MISMATCH,
                "artifact channel count does not match profile schema",
                state=RunState.INVALID_INPUT,
            )
        image_hwc = image_hwc[..., indices]
        normalized = self._normalization_spec(len(required)).apply(image_hwc)
        patch_height, patch_width = self.model_profile.patch_size
        if normalized.shape[:2] != (patch_height, patch_width):
            raise InferenceAdapterError(
                FailureReason.SCHEMA_MISMATCH,
                "adapted patch does not match model patch size",
                state=RunState.INVALID_INPUT,
                provenance={"shape": normalized.shape, "expected": (patch_height, patch_width)},
            )
        channels = normalized.shape[-1]
        base = np.transpose(normalized, (2, 0, 1))[None, ...]
        axes = {"N": 0, "C": 1, "H": 2, "W": 3}
        try:
            permutation = tuple(axes[axis] for axis in self.model_profile.tensor_layout)
        except KeyError as exc:
            raise InferenceAdapterError(
                FailureReason.SCHEMA_MISMATCH,
                f"unsupported tensor layout {self.model_profile.tensor_layout}",
                state=RunState.INVALID_INPUT,
            ) from exc
        tensor = np.transpose(base, permutation)
        try:
            tensor = np.asarray(tensor, dtype=np.dtype(self.model_profile.tensor_dtype))
        except (TypeError, ValueError) as exc:
            raise InferenceAdapterError(
                FailureReason.SCHEMA_MISMATCH,
                "tensor dtype conversion failed",
                state=RunState.INVALID_INPUT,
                provenance={"dtype": self.model_profile.tensor_dtype, "error": str(exc)},
            ) from exc
        if not np.isfinite(tensor).all():
            raise InferenceAdapterError(
                FailureReason.NON_FINITE_OUTPUT,
                "normalization produced a non-finite tensor",
                state=RunState.RUNTIME_FAULT,
            )
        return np.ascontiguousarray(tensor)

    def _adapt_window(
        self,
        block: Mapping[str, Any],
        profile: PreprocessingProfile,
        *,
        patch_row: int,
        patch_col: int,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
    ) -> AdaptedPatch:
        patch_height, patch_width = self.model_profile.patch_size
        image = self._to_hwc(np.asarray(block["image"]), profile)
        validity = np.asarray(block["validity_yx"])
        reasons = np.asarray(block["validity_reason_yx"])
        mapping = np.asarray(block["mapping_yx"])
        full_area = patch_height * patch_width
        actual_area = max(0, row_end - row_start) * max(0, col_end - col_start)
        valid_value = profile.validity_encoding.valid_value
        valid_fraction = float(np.count_nonzero(validity == valid_value) / full_area)
        reason_summary = dict(self._reason_summary(reasons, profile))
        incomplete_edge = image.shape[0] != patch_height or image.shape[1] != patch_width
        policy = self.model_profile.padding_policy
        border_bit = profile.reason_encoding.bits.get("border", 0)
        if incomplete_edge and policy == "reject":
            if border_bit:
                reason_summary["border"] = reason_summary.get("border", 0) + int(full_area - actual_area)
            padded_validity = np.zeros((patch_height, patch_width), dtype=validity.dtype)
            padded_validity[: validity.shape[0], : validity.shape[1]] = validity
            padded_reasons = np.zeros((patch_height, patch_width), dtype=reasons.dtype)
            padded_reasons[: reasons.shape[0], : reasons.shape[1]] = reasons
            if border_bit:
                padded_reasons[validity.shape[0] :, :] |= border_bit
                padded_reasons[:, validity.shape[1] :] |= border_bit
            padded_mapping = np.full(
                (patch_height, patch_width, mapping.shape[-1]),
                np.nan,
                dtype=mapping.dtype,
            )
            padded_mapping[: mapping.shape[0], : mapping.shape[1], :] = mapping
            return AdaptedPatch(
                patch_row,
                patch_col,
                (row_start, row_end, col_start, col_end),
                None,
                valid_fraction,
                MappingProxyType(reason_summary),
                padded_validity,
                padded_reasons,
                padded_mapping,
                str(block.get("mapping_ref", "")),
                "invalid",
                "BORDER_PADDING_REJECTED",
            )

        if incomplete_edge:
            image = self._pad_array(image, (patch_height, patch_width), policy, constant=self.pad_value)
            padded_validity = self._pad_array(
                validity,
                (patch_height, patch_width),
                "constant",
                constant=profile.validity_encoding.invalid_value,
            )
            padded_reasons = self._pad_array(reasons, (patch_height, patch_width), "constant", constant=0)
            if border_bit:
                padded_reasons[validity.shape[0] :, :] |= border_bit
                padded_reasons[:, validity.shape[1] :] |= border_bit
            padded_mapping = np.full(
                (patch_height, patch_width, mapping.shape[-1]),
                np.nan,
                dtype=mapping.dtype,
            )
            padded_mapping[: mapping.shape[0], : mapping.shape[1], :] = mapping
            validity = padded_validity
            reasons = padded_reasons
            mapping = padded_mapping

        if valid_fraction <= 0.0 or valid_fraction < self.min_valid_fraction:
            return AdaptedPatch(
                patch_row,
                patch_col,
                (row_start, row_end, col_start, col_end),
                None,
                valid_fraction,
                MappingProxyType(reason_summary),
                validity,
                reasons,
                mapping,
                str(block.get("mapping_ref", "")),
                "invalid",
                "VALID_FRACTION_BELOW_THRESHOLD",
            )

        tensor = self._tensor(image, profile)
        return AdaptedPatch(
            patch_row,
            patch_col,
            (row_start, row_end, col_start, col_end),
            tensor,
            valid_fraction,
            MappingProxyType(reason_summary),
            validity,
            reasons,
            mapping,
            str(block.get("mapping_ref", "")),
            "ready",
            None,
        )

    def iter_patches(self, artifact: PreprocessArtifact) -> Iterable[AdaptedPatch]:
        """Yield row-major windows and masks from the verified artifact."""

        if not isinstance(artifact, PreprocessArtifact):
            raise TypeError("iter_patches requires a PreprocessArtifact")
        self.check_compatibility(artifact)
        reader = artifact.open()
        if isinstance(reader, PreprocessFailure):
            raise InferenceAdapterError(
                reader.reason_code,
                reader.message,
                state=reader.state,
                provenance=reader.provenance,
            )
        profile = reader.profile
        rows, cols = profile.target_grid.rows, profile.target_grid.cols
        patch_height, patch_width = self.model_profile.patch_size
        try:
            for patch_row, row_start in enumerate(range(0, rows, patch_height)):
                row_end = min(row_start + patch_height, rows)
                for patch_col, col_start in enumerate(range(0, cols, patch_width)):
                    col_end = min(col_start + patch_width, cols)
                    full_block = reader.read_block(row_start, row_end)
                    image = np.asarray(full_block["image"])
                    if profile.output_layout == "YXC":
                        image = image[:, col_start:col_end, :]
                    elif profile.output_layout == "CYX":
                        image = image[:, :, col_start:col_end]
                    else:
                        image = image[:, col_start:col_end]
                    block = {
                        "image": image,
                        "validity_yx": np.asarray(full_block["validity_yx"])[:, col_start:col_end],
                        "validity_reason_yx": np.asarray(full_block["validity_reason_yx"])[:, col_start:col_end],
                        "mapping_yx": np.asarray(full_block["mapping_yx"])[:, col_start:col_end, :],
                        "mapping_ref": full_block.get("mapping_ref", ""),
                    }
                    yield self._adapt_window(
                        block,
                        profile,
                        patch_row=patch_row,
                        patch_col=patch_col,
                        row_start=row_start,
                        row_end=row_end,
                        col_start=col_start,
                        col_end=col_end,
                    )
        finally:
            reader.close()

    def _resolve_engine(self) -> Any:
        if self.engine is None:
            if self.engine_factory is None:
                raise InferenceAdapterError(
                    FailureReason.INVALID_REQUEST,
                    "InferenceAdapter has no inference engine or engine_factory",
                    state=RunState.INVALID_INPUT,
                )
            try:
                self.engine = self.engine_factory()
            except (TimeoutError, ConnectionError, OSError, RuntimeError) as exc:
                raise InferenceAdapterError(
                    FailureReason.RUNTIME_FAULT,
                    "inference engine initialization failed",
                    state=RunState.RUNTIME_FAULT,
                    provenance={
                        "engine_fault_type": type(exc).__name__,
                        "engine_fault": str(exc),
                        "engine_operation": "initialize",
                    },
                ) from exc
        actual_fingerprint = getattr(self.engine, "engine_fingerprint", None)
        if actual_fingerprint is None:
            actual_fingerprint = getattr(self.engine, "fingerprint", None)
        if actual_fingerprint is not None and _digest(actual_fingerprint, "engine_fingerprint") != self.engine_fingerprint:
            raise InferenceAdapterError(
                FailureReason.FINGERPRINT_LINKAGE_MISMATCH,
                "inference engine fingerprint does not match the model profile",
                state=RunState.INVALID_INPUT,
                provenance={"expected": self.engine_fingerprint, "actual": str(actual_fingerprint)},
            )
        return self.engine

    @staticmethod
    def _infer_batch(engine: Any, tensors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        try:
            if hasattr(engine, "infer_batch"):
                result = engine.infer_batch(tensors)
            elif hasattr(engine, "infer"):
                predictions = []
                probabilities = []
                for tensor in tensors:
                    prediction, probability = engine.infer(tensor[None, ...])
                    predictions.append(prediction)
                    probabilities.append(probability)
                result = (predictions, probabilities)
            else:
                raise InferenceAdapterError(
                    FailureReason.INVALID_REQUEST,
                    "inference engine must expose infer_batch or infer",
                    state=RunState.INVALID_INPUT,
                )
        except InferenceAdapterError:
            raise
        except (TimeoutError, ConnectionError, OSError, RuntimeError) as exc:
            # TensorRT/CUDA timeout, context reset and device/transport faults
            # are operational failures. They must not escape as a partial run.
            raise InferenceAdapterError(
                FailureReason.RUNTIME_FAULT,
                "inference engine execution failed",
                state=RunState.RUNTIME_FAULT,
                provenance={
                    "engine_fault_type": type(exc).__name__,
                    "engine_fault": str(exc),
                },
            ) from exc
        if isinstance(result, Mapping):
            predictions = result.get("predictions", result.get("is_cloud"))
            probabilities = result.get("probabilities", result.get("probability"))
        else:
            try:
                predictions, probabilities = result
            except (TypeError, ValueError) as exc:
                raise InferenceAdapterError(
                    FailureReason.RUNTIME_FAULT,
                    "inference engine result must contain predictions and probabilities",
                    state=RunState.RUNTIME_FAULT,
                ) from exc
        predictions = np.asarray(predictions).reshape(-1)
        probabilities = np.asarray(probabilities, dtype=np.float32).reshape(-1)
        if len(predictions) != len(tensors) or len(probabilities) != len(tensors):
            raise InferenceAdapterError(
                FailureReason.RUNTIME_FAULT,
                "inference engine returned a result count different from the batch",
                state=RunState.RUNTIME_FAULT,
            )
        if not np.isfinite(probabilities).all():
            raise InferenceAdapterError(
                FailureReason.NON_FINITE_OUTPUT,
                "inference engine returned a non-finite probability",
                state=RunState.RUNTIME_FAULT,
            )
        if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
            raise InferenceAdapterError(
                FailureReason.RUNTIME_FAULT,
                "inference engine returned a probability outside [0, 1]",
                state=RunState.RUNTIME_FAULT,
            )
        return predictions.astype(bool), probabilities

    def run(self, artifact: PreprocessArtifact, *, patch_result_writer: Any = None) -> InferenceResult | PreprocessFailure:
        """Run the configured engine after compatibility validation.

        Expected operational failures are converted to ``PreprocessFailure``;
        programmer errors such as passing a path instead of an artifact remain
        ``TypeError`` so they cannot be silently treated as a valid run.
        """

        if not isinstance(artifact, PreprocessArtifact):
            raise TypeError("InferenceAdapter.run requires a PreprocessArtifact")
        try:
            report = self.check_compatibility(artifact)
            engine = self._resolve_engine()
            prepared: list[AdaptedPatch] = list(self.iter_patches(artifact))
            records: list[dict[str, Any]] = []
            ready = [patch for patch in prepared if patch.is_valid]
            predictions: dict[tuple[int, int], tuple[bool, float]] = {}
            batch_size = self.model_profile.batch_size
            for start in range(0, len(ready), batch_size):
                batch = ready[start : start + batch_size]
                tensors = np.concatenate([patch.tensor for patch in batch], axis=0)
                batch_predictions, batch_probabilities = self._infer_batch(engine, tensors)
                for patch, prediction, probability in zip(batch, batch_predictions, batch_probabilities):
                    predictions[(patch.patch_row, patch.patch_col)] = (bool(prediction), float(probability))

            valid_area = 0.0
            cloud_area = 0.0
            for patch in prepared:
                area = float((patch.model_window[1] - patch.model_window[0]) * (patch.model_window[3] - patch.model_window[2]))
                patch_valid_area = area * patch.valid_fraction
                if patch.is_valid:
                    valid_area += patch_valid_area
                    is_cloud, probability = predictions[(patch.patch_row, patch.patch_col)]
                    cloud_area += patch_valid_area if is_cloud else 0.0
                    status = "valid"
                    reason_code = None
                    cloud_label: str | None = "cloud" if is_cloud else "clear"
                    cloud_probability: float | None = probability
                else:
                    status = "invalid"
                    reason_code = patch.reason_code
                    cloud_label = None
                    cloud_probability = None
                record = {
                    "patch_row": patch.patch_row,
                    "patch_col": patch.patch_col,
                    "model_window": patch.model_window,
                    "valid_fraction": patch.valid_fraction,
                    "validity_reason_summary": dict(patch.validity_reason_summary),
                    "cloud_probability": cloud_probability,
                    "cloud_label": cloud_label,
                    "inference_status": status,
                    "source_mapping_ref": patch.source_mapping_ref,
                    "preprocessing_profile_id": report.profile_id,
                    "preprocessing_profile_fingerprint": artifact.manifest.profile_fingerprint,
                    "engine_fingerprint": self.engine_fingerprint,
                    "timestamp": _now_text(),
                    "reason_code": reason_code,
                    "threshold": self.threshold,
                    "georef_valid": True,
                }
                records.append(record)
                if patch_result_writer is not None:
                    patch_result_writer.append(record)
            coverage = None if valid_area <= 0.0 else float(cloud_area / valid_area)
            result = InferenceResult(
                records=tuple(MappingProxyType(record) for record in records),
                status="complete" if all(record["inference_status"] == "valid" for record in records) else "incomplete",
                cloud_coverage=coverage,
                valid_area=valid_area,
                processed_count=len(records),
                valid_count=sum(record["inference_status"] == "valid" for record in records),
                invalid_count=sum(record["inference_status"] != "valid" for record in records),
                preprocessing_profile_id=report.profile_id,
                engine_fingerprint=self.engine_fingerprint,
                source_fingerprint=artifact.manifest.source_fingerprint,
                patch_result=patch_result_writer,
                provenance=MappingProxyType({"artifact_path": str(artifact.artifact_path)}),
            )
            return result
        except InferenceAdapterError as error:
            return PreprocessFailure(
                state=error.state,
                reason_code=error.reason_code,
                safe_action=SafeAction.RETAIN_FOR_GROUND,
                message=str(error),
                provenance=error.provenance,
            )
        except PreprocessError as error:
            return PreprocessFailure(
                state=error.state,
                reason_code=error.reason_code,
                safe_action=SafeAction.RETAIN_FOR_GROUND,
                message=str(error),
                provenance=error.provenance,
            )

    infer_artifact = run


__all__ = [
    "AdaptedPatch",
    "CompatibilityReport",
    "InferenceAdapter",
    "InferenceAdapterError",
    "InferenceResult",
]
