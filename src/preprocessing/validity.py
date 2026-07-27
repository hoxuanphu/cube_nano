"""Validity and validity-reason mask construction for source blocks.

The builder keeps validity separate from raster samples.  It only interprets
NoData semantics declared by the preprocessing profile and combines explicit
quality masks supplied by a reader or warp backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import PreprocessingProfile
from .errors import FailureReason, PreprocessError, RunState


_NONE_KINDS = {"none", "absent", "not_applicable", "no_data"}


@dataclass(frozen=True)
class ValidityMasks:
    """Typed source/output masks in the encodings selected by a profile."""

    validity_yx: np.ndarray
    reason_yx: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.validity_yx.shape)  # type: ignore[return-value]


def _as_yxc(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 2:
        return image[..., None]
    if image.ndim == 3:
        return image
    raise ValueError(f"working image must be YX or YXC, got shape {image.shape}")


def _as_bool_mask(value: Any, shape: tuple[int, int], field_name: str) -> np.ndarray:
    if value is None:
        return np.zeros(shape, dtype=bool)
    array = np.asarray(value, dtype=bool)
    if array.ndim == 0:
        return np.full(shape, bool(array), dtype=bool)
    if tuple(array.shape) != shape:
        raise ValueError(f"{field_name} must have shape {shape}, got {array.shape}")
    return array


class ValidityBuilder:
    """Build source and propagated masks from a frozen profile contract."""

    def __init__(self, profile: PreprocessingProfile):
        self.profile = profile
        self.validity_encoding = profile.validity_encoding
        self.reason_encoding = profile.reason_encoding
        self._valid_dtype = np.dtype(self.validity_encoding.dtype)
        self._reason_dtype = np.dtype(self.reason_encoding.dtype)

    def _bit(self, name: str) -> int:
        try:
            return int(self.reason_encoding.bits[name])
        except KeyError as exc:
            raise PreprocessError(
                FailureReason.SCHEMA_MISMATCH,
                f"reason encoding does not define required bit '{name}'",
                state=RunState.INVALID_INPUT,
                provenance={"reason_name": name, "reason_encoding": self.reason_encoding.to_mapping()},
            ) from exc

    def _reason_array(self, reason_yx: Any, shape: tuple[int, int]) -> np.ndarray:
        if reason_yx is None:
            return np.zeros(shape, dtype=self._reason_dtype)
        array = np.asarray(reason_yx)
        if tuple(array.shape) != shape:
            raise ValueError(f"reason mask must have shape {shape}, got {array.shape}")
        if not np.issubdtype(array.dtype, np.integer):
            raise ValueError("reason mask must use an integer dtype")
        return array.astype(self._reason_dtype, copy=True)

    def add_reason(self, reason_yx: np.ndarray, name: str, mask: Any) -> np.ndarray:
        """Return a copy with ``name`` OR-ed wherever ``mask`` is true."""

        mask_array = _as_bool_mask(mask, tuple(reason_yx.shape), f"{name} mask")
        if not np.any(mask_array):
            return reason_yx
        result = reason_yx.copy()
        result[mask_array] |= np.asarray(self._bit(name), dtype=self._reason_dtype)
        return result

    def _nodata_mask(
        self,
        image_yxc: np.ndarray,
        *,
        external_nodata_mask: Any = None,
    ) -> np.ndarray:
        semantics = self.profile.source_schema.nodata_semantics
        if semantics is None:
            return np.zeros(image_yxc.shape[:2], dtype=bool)
        spec = dict(semantics)
        kind = str(spec.get("kind", spec.get("type", "none"))).lower().replace("-", "_")
        if kind in _NONE_KINDS:
            return np.zeros(image_yxc.shape[:2], dtype=bool)
        if kind in {"nan", "non_finite", "nonfinite"}:
            if not np.issubdtype(image_yxc.dtype, np.inexact):
                return np.zeros(image_yxc.shape[:2], dtype=bool)
            invalid = ~np.isfinite(image_yxc) if kind != "nan" else np.isnan(image_yxc)
            return np.any(invalid, axis=2)
        if kind in {"value", "values", "sentinel"}:
            configured = spec.get("value", spec.get("values", spec.get("sentinel")))
            if configured is None:
                raise ValueError("NoData value semantics require 'value' or 'values'")
            channels = self.profile.source_schema.channels
            invalid = np.zeros(image_yxc.shape[:2], dtype=bool)
            if isinstance(configured, Mapping):
                for index, channel in enumerate(channels):
                    if channel in configured:
                        invalid |= np.equal(image_yxc[..., index], configured[channel])
                return invalid
            if isinstance(configured, Sequence) and not isinstance(configured, (str, bytes)):
                values = tuple(configured)
                if len(values) == image_yxc.shape[2] and len(channels) == image_yxc.shape[2]:
                    for index, value in enumerate(values):
                        invalid |= np.equal(image_yxc[..., index], value)
                    return invalid
                return np.any(np.isin(image_yxc, values), axis=2)
            return np.any(np.equal(image_yxc, configured), axis=2)
        if kind in {"mask", "validity_mask"}:
            mask = external_nodata_mask if external_nodata_mask is not None else spec.get("mask")
            if mask is None:
                raise ValueError("NoData mask semantics require an external mask")
            return _as_bool_mask(mask, image_yxc.shape[:2], "NoData mask")
        if kind in {"range", "valid_range"}:
            lower = spec.get("lower")
            upper = spec.get("upper")
            if lower is None and upper is None:
                raise ValueError("NoData range semantics require lower or upper")
            invalid = np.zeros(image_yxc.shape[:2], dtype=bool)
            if lower is not None:
                invalid |= np.any(image_yxc < lower, axis=2)
            if upper is not None:
                invalid |= np.any(image_yxc > upper, axis=2)
            return invalid
        raise ValueError(f"unsupported NoData semantics kind '{kind}'")

    def _missing_mask(self, missing_channels: Any, shape: tuple[int, int]) -> np.ndarray:
        if missing_channels is None:
            return np.zeros(shape, dtype=bool)
        if isinstance(missing_channels, Mapping):
            known = set(self.profile.source_schema.channels)
            unknown = {str(name) for name in missing_channels} - known
            if unknown:
                raise ValueError(f"missing channel names are not in source schema: {sorted(unknown)}")
            result = np.zeros(shape, dtype=bool)
            for _, missing in missing_channels.items():
                value = np.asarray(missing)
                if value.ndim == 0:
                    if bool(value):
                        result[...] = True
                else:
                    result |= _as_bool_mask(value, shape, "missing channel mask")
            return result
        if isinstance(missing_channels, (str, bytes)):
            missing_channels = (missing_channels,)
        array = np.asarray(missing_channels)
        if array.dtype == bool and array.ndim == 1:
            if array.size != self.profile.source_schema.channel_count:
                raise ValueError("missing channel vector length does not match source schema")
            return np.full(shape, bool(np.any(array)), dtype=bool)
        if array.dtype == bool and array.ndim == 2:
            return _as_bool_mask(array, shape, "missing channel mask")
        if array.dtype == bool and array.ndim == 3:
            if array.shape[:2] != shape or array.shape[2] != self.profile.source_schema.channel_count:
                raise ValueError("missing channel mask must be YXC with source channel count")
            return np.any(array, axis=2)
        names = {str(name) for name in missing_channels}
        known = set(self.profile.source_schema.channels)
        unknown = names - known
        if unknown:
            raise ValueError(f"missing channel names are not in source schema: {sorted(unknown)}")
        return np.full(shape, bool(names), dtype=bool)

    def build_source(
        self,
        image_yxc: Any,
        *,
        missing_channels: Any = None,
        external_nodata_mask: Any = None,
    ) -> ValidityMasks:
        """Build masks for a source block without changing source samples."""

        image = _as_yxc(np.asarray(image_yxc))
        shape = tuple(image.shape[:2])
        if image.shape[2] != self.profile.source_schema.channel_count:
            raise ValueError(
                "source block channel count does not match profile: "
                f"{image.shape[2]} != {self.profile.source_schema.channel_count}"
            )
        reason = np.zeros(shape, dtype=self._reason_dtype)
        invalid = self._nodata_mask(image, external_nodata_mask=external_nodata_mask)
        reason = self.add_reason(reason, "source_nodata", invalid)
        missing = self._missing_mask(missing_channels, shape)
        reason = self.add_reason(reason, "missing_channel", missing)
        invalid |= missing
        validity = np.where(
            invalid,
            self.validity_encoding.invalid_value,
            self.validity_encoding.valid_value,
        ).astype(self._valid_dtype)
        return ValidityMasks(validity_yx=validity, reason_yx=reason)

    def finalize(
        self,
        base_validity_yx: Any,
        base_reason_yx: Any,
        *,
        outside_mapping: Any = None,
        border: Any = None,
        insufficient_support: Any = None,
        force_invalid: Any = None,
    ) -> ValidityMasks:
        """Combine source masks and geometric/support reasons for an output block."""

        base_validity = np.asarray(base_validity_yx) == self.validity_encoding.valid_value
        shape = tuple(base_validity.shape)
        if base_validity.ndim != 2:
            raise ValueError("validity mask must be YX")
        reason = self._reason_array(base_reason_yx, shape)
        invalid = ~base_validity
        for name, mask in (
            ("outside_mapping", outside_mapping),
            ("border", border),
            ("insufficient_support", insufficient_support),
        ):
            mask_array = _as_bool_mask(mask, shape, f"{name} mask")
            reason = self.add_reason(reason, name, mask_array)
            if name != "border":
                invalid |= mask_array
        if force_invalid is not None:
            invalid |= _as_bool_mask(force_invalid, shape, "force_invalid mask")
        validity = np.where(
            invalid,
            self.validity_encoding.invalid_value,
            self.validity_encoding.valid_value,
        ).astype(self._valid_dtype)
        return ValidityMasks(validity_yx=validity, reason_yx=reason)


__all__ = ["ValidityBuilder", "ValidityMasks"]
