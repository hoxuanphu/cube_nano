"""Profile-driven target-grid and source mapping planning."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterator, Mapping

import numpy as np

from .contracts import CalibrationBundle, PreprocessingProfile
from .errors import FailureReason, PreprocessError, RunState
from .source_reader import SourceWindow


def _integer_bounds(value: Any, field_name: str) -> tuple[int, int]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{field_name} must be a two-item sequence")
    result = tuple(int(item) for item in value)
    if any(int(item) != item for item in value):
        raise ValueError(f"{field_name} must contain integers")
    if result[1] <= result[0]:
        raise ValueError(f"{field_name} must have positive size")
    return result


@dataclass(frozen=True)
class TransformPlan:
    """Mapping and source admission information for one output strip."""

    output_row_start: int
    output_row_end: int
    target_shape_yx: tuple[int, int]
    source_shape_yx: tuple[int, int]
    source_roi_yx: tuple[int, int, int, int]
    source_window: SourceWindow
    mapping_yx: np.ndarray
    footprint_yx: np.ndarray
    finite_mapping_yx: np.ndarray
    outside_source_yx: np.ndarray
    outside_roi_yx: np.ndarray
    pixel_offset: float
    profile_fingerprint: str
    calibration_fingerprint: str
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        rows = self.output_row_end - self.output_row_start
        cols = self.target_shape_yx[1]
        if self.output_row_start < 0 or self.output_row_end <= self.output_row_start:
            raise ValueError("output strip bounds are invalid")
        if self.output_row_end > self.target_shape_yx[0]:
            raise ValueError("output strip exceeds target shape")
        mapping = np.asarray(self.mapping_yx)
        if mapping.shape != (rows, cols, 2):
            raise ValueError(f"mapping must have shape {(rows, cols, 2)}, got {mapping.shape}")
        footprint = np.asarray(self.footprint_yx)
        if footprint.shape != (rows, cols, 4):
            raise ValueError(f"footprint must have shape {(rows, cols, 4)}, got {footprint.shape}")
        for name in ("finite_mapping_yx", "outside_source_yx", "outside_roi_yx"):
            mask = np.asarray(getattr(self, name))
            if mask.shape != (rows, cols):
                raise ValueError(f"{name} must have shape {(rows, cols)}")
            mask.setflags(write=False)
            object.__setattr__(self, name, mask)
        mapping = mapping.copy()
        mapping.setflags(write=False)
        footprint = footprint.copy()
        footprint.setflags(write=False)
        object.__setattr__(self, "mapping_yx", mapping)
        object.__setattr__(self, "footprint_yx", footprint)
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    @property
    def output_shape_yx(self) -> tuple[int, int]:
        return (self.output_row_end - self.output_row_start, self.target_shape_yx[1])

    @property
    def outside_mapping_yx(self) -> np.ndarray:
        return self.outside_source_yx | self.outside_roi_yx | ~self.finite_mapping_yx


class TransformPlanner:
    """Build deterministic identity/affine plans from profile + calibration."""

    def __init__(
        self,
        profile: PreprocessingProfile,
        calibration: CalibrationBundle,
        source_shape_yx: tuple[int, int] | tuple[int, int, int],
    ):
        self.profile = profile
        self.calibration = calibration
        shape = tuple(int(value) for value in source_shape_yx)
        if len(shape) == 3:
            shape = shape[:2]
        if len(shape) != 2 or any(value <= 0 for value in shape):
            raise ValueError("source_shape_yx must contain two positive dimensions")
        if profile.source_schema.shape is not None:
            schema_shape = profile.source_schema.shape
            schema_yx = tuple(schema_shape[index] for index, axis in enumerate(profile.source_schema.axes) if axis in {"Y", "X"})
            if tuple(schema_yx) != shape:
                raise ValueError(f"source shape {shape} does not match profile source schema {schema_shape}")
        if calibration.transform_type != profile.transform_type:
            raise PreprocessError(
                FailureReason.CALIBRATION_UNSUPPORTED,
                "calibration transform type does not match preprocessing profile",
                state=RunState.CALIBRATION_ERROR,
                provenance={"profile_transform": profile.transform_type, "calibration_transform": calibration.transform_type},
            )
        if profile.transform_type not in {"identity", "affine"}:
            raise PreprocessError(
                FailureReason.CALIBRATION_UNSUPPORTED,
                f"transform type '{profile.transform_type}' is not implemented by the CPU baseline",
                state=RunState.CALIBRATION_ERROR,
                provenance={"transform_type": profile.transform_type},
            )
        if calibration.transform_direction != profile.transform_direction:
            raise PreprocessError(
                FailureReason.CALIBRATION_UNSUPPORTED,
                "calibration transform direction does not match preprocessing profile",
                state=RunState.CALIBRATION_ERROR,
            )
        if calibration.pixel_convention != profile.pixel_convention:
            raise PreprocessError(
                FailureReason.CALIBRATION_UNSUPPORTED,
                "calibration pixel convention does not match preprocessing profile",
                state=RunState.CALIBRATION_ERROR,
            )
        self.source_shape_yx = shape
        self.target_shape_yx = (profile.target_grid.rows, profile.target_grid.cols)
        self.pixel_offset = 0.5 if profile.pixel_convention == "center" else 0.0
        self.source_roi_yx = self._resolve_roi()
        self._matrix, self._offset = self._resolve_affine()

    def _resolve_roi(self) -> tuple[int, int, int, int]:
        roi = self.profile.source_roi
        if roi.mode == "full":
            return (0, self.source_shape_yx[0], 0, self.source_shape_yx[1])
        bounds = (roi.row_start, roi.row_end, roi.col_start, roi.col_end)
        if any(value is None for value in bounds):
            raise ValueError("window source ROI must define all bounds")
        row_start, row_end, col_start, col_end = (int(value) for value in bounds)
        if row_end > self.source_shape_yx[0] or col_end > self.source_shape_yx[1]:
            raise PreprocessError(
                FailureReason.SCHEMA_MISMATCH,
                "source ROI exceeds source shape",
                state=RunState.INVALID_INPUT,
                provenance={"source_shape_yx": self.source_shape_yx, "source_roi": bounds},
            )
        return (row_start, row_end, col_start, col_end)

    def _resolve_affine(self) -> tuple[np.ndarray, np.ndarray]:
        if self.profile.transform_type == "identity":
            matrix = np.eye(2, dtype=np.float64)
            offset = np.zeros(2, dtype=np.float64)
        else:
            parameters = dict(self.calibration.parameters)
            try:
                matrix = np.asarray(parameters["matrix"], dtype=np.float64)
                offset = np.asarray(parameters["offset"], dtype=np.float64)
            except KeyError as exc:
                raise PreprocessError(
                    FailureReason.CALIBRATION_INVALID,
                    "affine calibration requires matrix and offset parameters",
                    state=RunState.CALIBRATION_ERROR,
                ) from exc
            if matrix.shape != (2, 2) or offset.shape != (2,):
                raise PreprocessError(
                    FailureReason.CALIBRATION_INVALID,
                    "affine calibration matrix must be 2x2 and offset must have length 2",
                    state=RunState.CALIBRATION_ERROR,
                )
            determinant = float(np.linalg.det(matrix))
            if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(offset)) or not np.isfinite(determinant) or abs(determinant) <= 1e-12:
                raise PreprocessError(
                    FailureReason.CALIBRATION_INVALID,
                    "affine calibration contains non-finite or singular parameters",
                    state=RunState.CALIBRATION_ERROR,
                )
        return matrix, offset

    def _target_model_xy(self, row_start: int, row_end: int) -> np.ndarray:
        rows = np.arange(row_start, row_end, dtype=np.float64)[:, None]
        cols = np.arange(self.target_shape_yx[1], dtype=np.float64)[None, :]
        grid = self.profile.target_grid
        x = float(grid.origin[0]) + (cols + self.pixel_offset) * float(grid.resolution[0])
        y = float(grid.origin[1]) + (rows + self.pixel_offset) * float(grid.resolution[1])
        return np.stack(np.broadcast_arrays(x, y), axis=-1)

    def _map_model_to_source(self, model_xy: np.ndarray) -> np.ndarray:
        flat = model_xy.reshape(-1, 2)
        if self.calibration.transform_direction == "source_to_model":
            source_xy = np.linalg.solve(self._matrix, (flat - self._offset).T).T
        else:
            source_xy = flat @ self._matrix.T + self._offset
        return source_xy.reshape(model_xy.shape)

    @staticmethod
    def _kernel_indices(mapping: np.ndarray, kernel: str, pixel_offset: float) -> tuple[np.ndarray, np.ndarray]:
        y = mapping[..., 0] - pixel_offset
        x = mapping[..., 1] - pixel_offset
        finite = np.isfinite(y) & np.isfinite(x)
        safe_y = np.where(finite, y, 0.0)
        safe_x = np.where(finite, x, 0.0)
        if kernel == "nearest":
            return (
                np.floor(safe_y + 0.5).astype(np.int64)[..., None],
                np.floor(safe_x + 0.5).astype(np.int64)[..., None],
            )
        if kernel in {"bilinear", "support"}:
            y0 = np.floor(safe_y).astype(np.int64)
            x0 = np.floor(safe_x).astype(np.int64)
            return (
                np.stack((y0, y0, y0 + 1, y0 + 1), axis=-1),
                np.stack((x0, x0 + 1, x0, x0 + 1), axis=-1),
            )
        if kernel == "bicubic":
            y0 = np.floor(safe_y).astype(np.int64)
            x0 = np.floor(safe_x).astype(np.int64)
            y_offsets = np.arange(-1, 3, dtype=np.int64)
            x_offsets = np.arange(-1, 3, dtype=np.int64)
            return (
                np.stack([y0 + dy for dy in y_offsets for _ in x_offsets], axis=-1),
                np.stack([x0 + dx for _ in y_offsets for dx in x_offsets], axis=-1),
            )
        raise ValueError(f"unsupported kernel '{kernel}'")

    def _footprint(self, mapping: np.ndarray) -> np.ndarray:
        index_pairs = [
            self._kernel_indices(mapping, kernel, self.pixel_offset)
            for kernel in (self.profile.image_kernel, self.profile.validity_kernel, self.profile.reason_kernel)
        ]
        y_indices = np.concatenate([pair[0] for pair in index_pairs], axis=-1)
        x_indices = np.concatenate([pair[1] for pair in index_pairs], axis=-1)
        return np.stack(
            (
                np.min(y_indices, axis=-1),
                np.max(y_indices, axis=-1),
                np.min(x_indices, axis=-1),
                np.max(x_indices, axis=-1),
            ),
            axis=-1,
        ).astype(np.int64)

    def _source_window(self, footprint: np.ndarray) -> SourceWindow:
        rows, cols = self.source_shape_yx
        halo_rows = self.profile.halo.rows
        halo_cols = self.profile.halo.cols
        finite = np.isfinite(footprint).all(axis=-1)
        if not np.any(finite):
            return SourceWindow(0, 0, 0, 0)
        selected = footprint[finite]
        raw_row_start = int(np.min(selected[:, 0])) - halo_rows
        raw_row_end = int(np.max(selected[:, 1])) + 1 + halo_rows
        raw_col_start = int(np.min(selected[:, 2])) - halo_cols
        raw_col_end = int(np.max(selected[:, 3])) + 1 + halo_cols
        row_start = max(0, min(rows, raw_row_start))
        row_end = max(row_start, min(rows, raw_row_end))
        col_start = max(0, min(cols, raw_col_start))
        col_end = max(col_start, min(cols, raw_col_end))
        if row_start == row_end and rows:
            row_start = min(rows - 1, max(0, raw_row_start))
            row_end = row_start + 1
        if col_start == col_end and cols:
            col_start = min(cols - 1, max(0, raw_col_start))
            col_end = col_start + 1
        return SourceWindow(row_start, row_end, col_start, col_end)

    def plan_strip(self, output_row_start: int, output_row_end: int) -> TransformPlan:
        if isinstance(output_row_start, bool) or isinstance(output_row_end, bool):
            raise TypeError("output strip bounds must be integers")
        if int(output_row_start) != output_row_start or int(output_row_end) != output_row_end:
            raise TypeError("output strip bounds must be integers")
        output_row_start = int(output_row_start)
        output_row_end = int(output_row_end)
        if output_row_start < 0 or output_row_end <= output_row_start or output_row_end > self.target_shape_yx[0]:
            raise ValueError("output strip bounds are outside target grid")
        model_xy = self._target_model_xy(output_row_start, output_row_end)
        source_xy = self._map_model_to_source(model_xy)
        mapping_dtype = np.float32 if self.profile.internal_numeric_precision == "float32" else np.float64
        mapping = np.stack((source_xy[..., 1], source_xy[..., 0]), axis=-1).astype(mapping_dtype)
        finite = np.isfinite(mapping).all(axis=-1)
        source_row_min = self.pixel_offset
        source_row_max = self.source_shape_yx[0] - 1 + self.pixel_offset
        source_col_min = self.pixel_offset
        source_col_max = self.source_shape_yx[1] - 1 + self.pixel_offset
        roi_row_start, roi_row_end, roi_col_start, roi_col_end = self.source_roi_yx
        outside_source = finite & (
            (mapping[..., 0] < source_row_min)
            | (mapping[..., 0] > source_row_max)
            | (mapping[..., 1] < source_col_min)
            | (mapping[..., 1] > source_col_max)
        )
        outside_roi = finite & (
            (mapping[..., 0] < roi_row_start + self.pixel_offset)
            | (mapping[..., 0] > roi_row_end - 1 + self.pixel_offset)
            | (mapping[..., 1] < roi_col_start + self.pixel_offset)
            | (mapping[..., 1] > roi_col_end - 1 + self.pixel_offset)
        )
        footprint = self._footprint(mapping)
        window = self._source_window(footprint)
        return TransformPlan(
            output_row_start=output_row_start,
            output_row_end=output_row_end,
            target_shape_yx=self.target_shape_yx,
            source_shape_yx=self.source_shape_yx,
            source_roi_yx=self.source_roi_yx,
            source_window=window,
            mapping_yx=mapping,
            footprint_yx=footprint,
            finite_mapping_yx=finite,
            outside_source_yx=outside_source,
            outside_roi_yx=outside_roi,
            pixel_offset=self.pixel_offset,
            profile_fingerprint=self.profile.fingerprint,
            calibration_fingerprint=self.calibration.fingerprint,
            provenance={
                "transform_type": self.profile.transform_type,
                "transform_direction": self.profile.transform_direction,
                "target_grid": self.profile.target_grid.to_mapping(),
                "source_roi": self.profile.source_roi.to_mapping(),
                "halo": self.profile.halo.to_mapping(),
                "output_row_start": output_row_start,
                "output_row_end": output_row_end,
            },
        )

    def plan_strips(self, strip_rows: int) -> Iterator[TransformPlan]:
        if isinstance(strip_rows, bool) or int(strip_rows) != strip_rows or int(strip_rows) <= 0:
            raise ValueError("strip_rows must be a positive integer")
        for start in range(0, self.target_shape_yx[0], int(strip_rows)):
            yield self.plan_strip(start, min(self.target_shape_yx[0], start + int(strip_rows)))


__all__ = ["TransformPlan", "TransformPlanner"]
