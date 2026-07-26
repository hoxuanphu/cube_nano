"""CPU baseline warp backend driven only by preprocessing contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .contracts import PreprocessingProfile
from .errors import FailureReason, PreprocessError, RunState
from .source_reader import SourceBlock
from .transform_plan import TransformPlan, TransformPlanner
from .validity import ValidityBuilder


@dataclass(frozen=True)
class WarpResult:
    """Materialized result for one model-grid output strip."""

    image: np.ndarray
    validity_yx: np.ndarray
    validity_reason_yx: np.ndarray
    mapping_yx: np.ndarray
    support_fraction_yx: np.ndarray
    provenance: Mapping[str, Any]


def _cubic_weight(distance: np.ndarray, parameter: float = -0.5) -> np.ndarray:
    distance = np.abs(distance)
    first = (parameter + 2) * distance**3 - (parameter + 3) * distance**2 + 1
    second = parameter * distance**3 - 5 * parameter * distance**2 + 8 * parameter * distance - 4 * parameter
    return np.where(distance <= 1, first, np.where(distance < 2, second, 0.0))


class CPUWarpBackend:
    """Reference CPU implementation; no model or engine contract is imported."""

    def __init__(self, profile: PreprocessingProfile):
        self.profile = profile
        self.validity_builder = ValidityBuilder(profile)
        self._internal_dtype = np.dtype(profile.internal_numeric_precision)

    def _kernel_taps(self, plan: TransformPlan, kernel: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mapping = np.asarray(plan.mapping_yx, dtype=self._internal_dtype)
        y = mapping[..., 0] - plan.pixel_offset
        x = mapping[..., 1] - plan.pixel_offset
        finite = np.isfinite(y) & np.isfinite(x)
        safe_y = np.where(finite, y, 0.0)
        safe_x = np.where(finite, x, 0.0)
        if kernel == "nearest":
            rows = np.floor(safe_y + 0.5).astype(np.int64)[..., None]
            cols = np.floor(safe_x + 0.5).astype(np.int64)[..., None]
            weights = np.ones(rows.shape, dtype=self._internal_dtype)
            return rows, cols, weights * finite[..., None]
        if kernel in {"bilinear", "support"}:
            row0 = np.floor(safe_y).astype(np.int64)
            col0 = np.floor(safe_x).astype(np.int64)
            dy = safe_y - row0
            dx = safe_x - col0
            rows = np.stack((row0, row0, row0 + 1, row0 + 1), axis=-1)
            cols = np.stack((col0, col0 + 1, col0, col0 + 1), axis=-1)
            weights = np.stack(((1 - dy) * (1 - dx), (1 - dy) * dx, dy * (1 - dx), dy * dx), axis=-1)
            return rows, cols, weights * finite[..., None]
        if kernel == "bicubic":
            row0 = np.floor(safe_y).astype(np.int64)
            col0 = np.floor(safe_x).astype(np.int64)
            dy = safe_y - row0
            dx = safe_x - col0
            row_offsets = (-1, 0, 1, 2)
            col_offsets = (-1, 0, 1, 2)
            rows = np.stack([row0 + row_offset for row_offset in row_offsets for _ in col_offsets], axis=-1)
            cols = np.stack([col0 + col_offset for _ in row_offsets for col_offset in col_offsets], axis=-1)
            row_weights = [_cubic_weight(dy - row_offset) for row_offset in row_offsets]
            col_weights = [_cubic_weight(dx - col_offset) for col_offset in col_offsets]
            weights = np.stack(
                [row_weights[row_index] * col_weights[col_index] for row_index in range(4) for col_index in range(4)],
                axis=-1,
            )
            return rows, cols, weights * finite[..., None]
        raise ValueError(f"unsupported kernel '{kernel}'")

    @staticmethod
    def _reflect_indices(indices: np.ndarray, size: int) -> np.ndarray:
        if size <= 1:
            return np.zeros_like(indices)
        period = 2 * size - 2
        folded = np.mod(indices, period)
        return np.where(folded < size, folded, period - folded)

    def _prepare_tap_indices(
        self,
        rows: np.ndarray,
        cols: np.ndarray,
        plan: TransformPlan,
        block: SourceBlock,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        source_rows, source_cols = plan.source_shape_yx
        raw_in_source = (rows >= 0) & (rows < source_rows) & (cols >= 0) & (cols < source_cols)
        roi_row_start, roi_row_end, roi_col_start, roi_col_end = plan.source_roi_yx
        in_roi = (rows >= roi_row_start) & (rows < roi_row_end) & (cols >= roi_col_start) & (cols < roi_col_end)
        policy = self.profile.border_policy
        if policy == "edge":
            sample_rows = np.clip(rows, 0, source_rows - 1)
            sample_cols = np.clip(cols, 0, source_cols - 1)
            border = ~raw_in_source
        elif policy == "reflect":
            sample_rows = self._reflect_indices(rows, source_rows)
            sample_cols = self._reflect_indices(cols, source_cols)
            border = ~raw_in_source
        else:
            sample_rows = rows
            sample_cols = cols
            border = ~raw_in_source
        local_rows = sample_rows - block.row_start
        local_cols = sample_cols - block.col_start
        covered = (
            (local_rows >= 0)
            & (local_rows < block.image_yxc.shape[0])
            & (local_cols >= 0)
            & (local_cols < block.image_yxc.shape[1])
        )
        eligible = covered & in_roi
        if policy not in {"edge", "reflect"}:
            eligible &= raw_in_source
        return local_rows, local_cols, eligible, border

    def _resample_image(
        self,
        block: SourceBlock,
        rows: np.ndarray,
        cols: np.ndarray,
        weights: np.ndarray,
        eligible: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        height, width = rows.shape[:2]
        channels = block.image_yxc.shape[2]
        result = np.zeros((height, width, channels), dtype=self._internal_dtype)
        denominator = np.zeros((height, width), dtype=self._internal_dtype)
        for tap in range(rows.shape[-1]):
            local_rows = rows[..., tap]
            local_cols = cols[..., tap]
            safe_rows = np.clip(local_rows, 0, max(0, block.image_yxc.shape[0] - 1))
            safe_cols = np.clip(local_cols, 0, max(0, block.image_yxc.shape[1] - 1))
            source_valid = block.validity_yx[safe_rows, safe_cols] != self.profile.validity_encoding.invalid_value
            usable = eligible[..., tap] & source_valid
            weight = weights[..., tap].astype(self._internal_dtype, copy=False)
            denominator += np.where(usable, weight, 0.0)
            sample = block.image_yxc[safe_rows, safe_cols, :].astype(self._internal_dtype, copy=False)
            sample = np.where(usable[..., None], sample, 0.0)
            result += sample * weight[..., None]
        good = np.abs(denominator) > np.finfo(self._internal_dtype).eps
        result = np.divide(result, denominator[..., None], out=np.zeros_like(result), where=good[..., None])
        return result, good

    def _resample_validity(
        self,
        block: SourceBlock,
        rows: np.ndarray,
        cols: np.ndarray,
        weights: np.ndarray,
        eligible: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        height, width = rows.shape[:2]
        total = np.sum(np.abs(weights), axis=-1)
        supported = np.zeros((height, width), dtype=self._internal_dtype)
        reason = np.zeros((height, width), dtype=np.dtype(self.profile.reason_encoding.dtype))
        for tap in range(rows.shape[-1]):
            local_rows = rows[..., tap]
            local_cols = cols[..., tap]
            safe_rows = np.clip(local_rows, 0, max(0, block.image_yxc.shape[0] - 1))
            safe_cols = np.clip(local_cols, 0, max(0, block.image_yxc.shape[1] - 1))
            usable = eligible[..., tap]
            valid = block.validity_yx[safe_rows, safe_cols] != self.profile.validity_encoding.invalid_value
            magnitude = np.abs(weights[..., tap].astype(self._internal_dtype, copy=False))
            supported += np.where(usable & valid, magnitude, 0.0)
            reason |= np.where(usable & (magnitude > np.finfo(self._internal_dtype).eps), block.validity_reason_yx[safe_rows, safe_cols], 0)
        fraction = np.divide(supported, total, out=np.zeros_like(supported), where=total > 0)
        return fraction, reason

    def _cast_output(self, image: np.ndarray) -> np.ndarray:
        profile = self.profile
        finite = np.isfinite(image)
        if not np.all(finite):
            if profile.non_finite_policy == "reject":
                raise PreprocessError(
                    FailureReason.NON_FINITE_OUTPUT,
                    "warp produced non-finite output values",
                    state=RunState.RUNTIME_FAULT,
                    provenance={"non_finite_count": int(np.count_nonzero(~finite))},
                )
            image = np.where(finite, image, 0.0)
        if profile.clipping.mode == "range":
            image = np.clip(image, profile.clipping.lower, profile.clipping.upper)
        rounding = profile.rounding
        if rounding == "nearest_even":
            image = np.rint(image)
        elif rounding == "nearest_away":
            image = np.sign(image) * np.floor(np.abs(image) + 0.5)
        elif rounding == "floor":
            image = np.floor(image)
        elif rounding == "ceil":
            image = np.ceil(image)
        elif rounding == "truncate":
            image = np.trunc(image)
        dtype = np.dtype(profile.output_dtype)
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
        else:
            info = np.finfo(dtype)
        if np.any(image < info.min) or np.any(image > info.max):
            raise PreprocessError(
                FailureReason.WARP_ERROR,
                f"warp output is outside output dtype range {dtype}",
                state=RunState.RUNTIME_FAULT,
                provenance={"dtype": dtype.name, "min": float(np.min(image)), "max": float(np.max(image))},
            )
        return image.astype(dtype)

    def _layout_output(self, image_yxc: np.ndarray) -> np.ndarray:
        layout = self.profile.output_layout
        if layout == "YXC":
            return image_yxc
        if layout == "CYX":
            return np.transpose(image_yxc, (2, 0, 1))
        if layout == "YX":
            if image_yxc.shape[2] != 1:
                raise ValueError("YX output layout requires one source channel")
            return image_yxc[..., 0]
        raise ValueError(f"unsupported output layout '{layout}'")

    def warp(self, block: SourceBlock, plan: TransformPlan) -> WarpResult:
        if block.source_shape_yx != plan.source_shape_yx:
            raise ValueError("source block and transform plan source shapes do not match")
        window = plan.source_window
        if not window.empty and (
            block.row_start > window.row_start
            or block.row_end < window.row_end
            or block.col_start > window.col_start
            or block.col_end < window.col_end
        ):
            raise ValueError("source block does not cover the planned source window")
        image_rows, image_cols, image_weights = self._kernel_taps(plan, self.profile.image_kernel)
        image_local_rows, image_local_cols, image_eligible, image_border = self._prepare_tap_indices(
            image_rows, image_cols, plan, block
        )
        image, image_has_support = self._resample_image(
            block,
            image_local_rows,
            image_local_cols,
            image_weights,
            image_eligible,
        )
        validity_rows, validity_cols, validity_weights = self._kernel_taps(plan, self.profile.validity_kernel)
        validity_local_rows, validity_local_cols, validity_eligible, validity_border = self._prepare_tap_indices(
            validity_rows, validity_cols, plan, block
        )
        support_fraction, source_reason = self._resample_validity(
            block,
            validity_local_rows,
            validity_local_cols,
            validity_weights,
            validity_eligible,
        )
        reason_rows, reason_cols, reason_weights = self._kernel_taps(plan, self.profile.reason_kernel)
        reason_local_rows, reason_local_cols, reason_eligible, reason_border = self._prepare_tap_indices(
            reason_rows, reason_cols, plan, block
        )
        _, reason_from_kernel = self._resample_validity(
            block,
            reason_local_rows,
            reason_local_cols,
            reason_weights,
            reason_eligible,
        )
        reason = source_reason | reason_from_kernel
        epsilon = np.finfo(self._internal_dtype).eps
        physical_border = (
            (image_border & (np.abs(image_weights) > epsilon)).any(axis=-1)
            | (validity_border & (np.abs(validity_weights) > epsilon)).any(axis=-1)
            | (reason_border & (np.abs(reason_weights) > epsilon)).any(axis=-1)
        )
        outside_source = plan.outside_source_yx
        outside_roi = plan.outside_roi_yx | ~plan.finite_mapping_yx
        force_invalid = outside_roi.copy()
        if self.profile.border_policy in {"invalid", "constant"}:
            force_invalid |= outside_source
        insufficient = (support_fraction < self.profile.support_threshold) | ~image_has_support
        if self.profile.border_policy in {"invalid", "constant"}:
            insufficient |= physical_border
        masks = self.validity_builder.finalize(
            np.where(
                support_fraction >= self.profile.support_threshold,
                self.profile.validity_encoding.valid_value,
                self.profile.validity_encoding.invalid_value,
            ),
            reason,
            outside_mapping=outside_roi | (outside_source & (self.profile.border_policy in {"invalid", "constant"})),
            border=physical_border,
            insufficient_support=insufficient,
            force_invalid=force_invalid,
        )
        bad = ~np.isfinite(image).all(axis=2)
        if np.any(bad):
            if self.profile.non_finite_policy == "reject":
                raise PreprocessError(
                    FailureReason.NON_FINITE_OUTPUT,
                    "warp produced non-finite output values",
                    state=RunState.RUNTIME_FAULT,
                    provenance={"non_finite_pixels": int(np.count_nonzero(bad))},
                )
            image = np.where(bad[..., None], 0.0, image)
            masks = self.validity_builder.finalize(
                masks.validity_yx,
                masks.reason_yx,
                insufficient_support=bad,
                force_invalid=bad,
            )
        output = self._layout_output(self._cast_output(image))
        return WarpResult(
            image=output,
            validity_yx=masks.validity_yx,
            validity_reason_yx=masks.reason_yx,
            mapping_yx=np.asarray(plan.mapping_yx),
            support_fraction_yx=support_fraction.astype(self._internal_dtype),
            provenance={
                "backend": "cpu",
                "profile_fingerprint": plan.profile_fingerprint,
                "calibration_fingerprint": plan.calibration_fingerprint,
                "output_row_start": plan.output_row_start,
                "output_row_end": plan.output_row_end,
                "image_kernel": self.profile.image_kernel,
                "validity_kernel": self.profile.validity_kernel,
                "reason_kernel": self.profile.reason_kernel,
            },
        )


def create_warp_backend(profile: PreprocessingProfile, backend: str = "cpu") -> CPUWarpBackend:
    """Create only an admitted backend; GPU remains disabled until parity evidence exists."""

    normalized = str(backend).lower()
    if normalized != "cpu":
        raise PreprocessError(
            FailureReason.NOT_IMPLEMENTED,
            "GPU warp backend is disabled until CPU numeric parity and resource benchmarks are recorded",
            state=RunState.RESOURCE_REJECTED,
            provenance={"requested_backend": normalized, "required_gate": "P3-05"},
        )
    return CPUWarpBackend(profile)


__all__ = ["CPUWarpBackend", "WarpResult", "create_warp_backend"]
