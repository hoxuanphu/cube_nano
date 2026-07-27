"""Dataset-independent block readers for preprocessing input."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Protocol

import numpy as np

from .contracts import ComputeProfile, PreprocessingProfile, SourceDescriptor, SourceSchema
from .errors import FailureReason, PreprocessError, RunState
from .validity import ValidityBuilder


def _normalize_axes(axes: str) -> str:
    normalized = str(axes).upper().replace("S", "C")
    if len(set(normalized)) != len(normalized) or "Y" not in normalized or "X" not in normalized:
        raise ValueError(f"source axes must contain unique X and Y axes, got {axes!r}")
    if set(normalized) - {"X", "Y", "C"}:
        raise ValueError(f"unsupported source axes {axes!r}")
    return normalized


def _to_yxc(array: np.ndarray, axes: str) -> np.ndarray:
    axes = _normalize_axes(axes)
    array = np.asarray(array)
    if array.ndim != len(axes):
        raise ValueError(f"array rank {array.ndim} does not match source axes {axes}")
    if "C" not in axes:
        if any(size != 1 for axis, size in zip(axes, array.shape) if axis not in {"Y", "X"}):
            raise ValueError("non-spatial source axes must be singleton")
        spatial = np.transpose(array, [axes.index("Y"), axes.index("X")])
        return spatial[..., None]
    order = [axes.index(axis) for axis in ("Y", "X", "C")]
    return np.transpose(array, order)


def _validate_byte_order(dtype: np.dtype, declared: str) -> None:
    declared = str(declared).lower()
    if dtype.byteorder == "|" or not dtype.itemsize:
        return
    native = "little" if np.little_endian else "big"
    actual = native if dtype.byteorder in {"=", "|"} else ("little" if dtype.byteorder == "<" else "big")
    if declared not in {"native", "little", "big", "not_applicable"}:
        raise ValueError(f"unsupported declared byte order {declared!r}")
    expected = native if declared == "native" else declared
    if expected != "not_applicable" and expected != actual:
        raise ValueError(f"source byte order {actual} does not match declared {declared}")


@dataclass(frozen=True)
class SourceWindow:
    row_start: int
    row_end: int
    col_start: int
    col_end: int

    def __post_init__(self) -> None:
        values = (self.row_start, self.row_end, self.col_start, self.col_end)
        if any(isinstance(value, bool) or int(value) != value for value in values):
            raise ValueError("source window bounds must be integers")
        if min(values) < 0:
            raise ValueError("source window bounds must be non-negative")
        if self.row_end < self.row_start or self.col_end < self.col_start:
            raise ValueError("source window end must not precede start")

    @property
    def shape(self) -> tuple[int, int]:
        return (self.row_end - self.row_start, self.col_end - self.col_start)

    @property
    def empty(self) -> bool:
        return self.row_start == self.row_end or self.col_start == self.col_end


@dataclass(frozen=True)
class SourceBlock:
    """A source window plus its independently encoded quality masks."""

    image_yxc: np.ndarray
    validity_yx: np.ndarray
    validity_reason_yx: np.ndarray
    row_start: int
    col_start: int
    source_shape_yx: tuple[int, int]
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        image = np.asarray(self.image_yxc)
        if image.ndim != 3:
            raise ValueError("SourceBlock.image_yxc must be YXC")
        shape = tuple(image.shape[:2])
        if tuple(np.asarray(self.validity_yx).shape) != shape:
            raise ValueError("SourceBlock validity shape does not match image")
        if tuple(np.asarray(self.validity_reason_yx).shape) != shape:
            raise ValueError("SourceBlock reason shape does not match image")
        if self.row_start < 0 or self.col_start < 0:
            raise ValueError("SourceBlock origin must be non-negative")
        if self.row_start + shape[0] > self.source_shape_yx[0] or self.col_start + shape[1] > self.source_shape_yx[1]:
            raise ValueError("SourceBlock exceeds declared source shape")
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    @property
    def row_end(self) -> int:
        return self.row_start + int(self.image_yxc.shape[0])

    @property
    def col_end(self) -> int:
        return self.col_start + int(self.image_yxc.shape[1])


class SourceReader(Protocol):
    shape_yxc: tuple[int, int, int]
    dtype: np.dtype

    def read_rows(self, row_start: int, row_end: int) -> np.ndarray:
        ...

    def read_window(self, window: SourceWindow) -> SourceBlock:
        ...

    def physical_blocks(self, row_start: int, row_end: int) -> tuple[Any, ...]:
        ...

    def close(self) -> None:
        ...

    def __enter__(self) -> "SourceReader":
        ...

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        ...


class ArraySourceReader:
    """Reader for an already opened array, useful for adapters and golden tests."""

    def __init__(
        self,
        array: Any,
        profile: PreprocessingProfile,
        *,
        source_schema: SourceSchema | None = None,
        missing_channels: Any = None,
        external_nodata_mask: Any = None,
        provenance: Mapping[str, Any] | None = None,
    ):
        self.profile = profile
        self.source_schema = source_schema or profile.source_schema
        self._array = np.asarray(array)
        self._closed = False
        self._missing_channels = missing_channels
        self._external_nodata_mask = None if external_nodata_mask is None else np.asarray(external_nodata_mask, dtype=bool)
        self._validity_builder = ValidityBuilder(profile)
        self.provenance = MappingProxyType(dict(provenance or {"backend": "array"}))
        self._validate_source()

    def _validate_source(self) -> None:
        schema = self.source_schema
        axes = _normalize_axes(schema.axes)
        actual_dtype = np.dtype(self._array.dtype)
        expected_dtype = np.dtype(schema.dtype)
        if actual_dtype.newbyteorder("=") != expected_dtype.newbyteorder("="):
            raise ValueError(f"source dtype {self._array.dtype} does not match profile dtype {schema.dtype}")
        _validate_byte_order(self._array.dtype, schema.byte_order)
        if schema.shape is not None and tuple(self._array.shape) != tuple(schema.shape):
            raise ValueError(f"source shape {self._array.shape} does not match profile shape {schema.shape}")
        normalized = _to_yxc(self._array, axes)
        expected_channels = schema.channel_count
        if normalized.shape[2] != expected_channels:
            raise ValueError(
                f"source channel count {normalized.shape[2]} does not match profile channel count {expected_channels}"
            )
        self._array_yxc = normalized
        self.shape_yxc = tuple(int(value) for value in normalized.shape)
        self.dtype = np.dtype(normalized.dtype)
        if self._external_nodata_mask is not None and self._external_nodata_mask.shape != self.shape_yxc[:2]:
            raise ValueError("external NoData mask must match source YX shape")

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.shape_yxc

    def _validate_rows(self, row_start: int, row_end: int) -> None:
        if self._closed:
            raise RuntimeError("source reader is closed")
        if any(isinstance(value, bool) or int(value) != value for value in (row_start, row_end)):
            raise TypeError("row bounds must be integers")
        if row_start < 0 or row_end <= row_start or row_end > self.shape_yxc[0]:
            raise ValueError(f"invalid source row range [{row_start}, {row_end})")

    def _validate_window(self, window: SourceWindow) -> None:
        if self._closed:
            raise RuntimeError("source reader is closed")
        if window.row_end > self.shape_yxc[0] or window.col_end > self.shape_yxc[1]:
            raise ValueError("source window exceeds source shape")

    def read_rows(self, row_start: int, row_end: int) -> np.ndarray:
        self._validate_rows(row_start, row_end)
        return np.asarray(self._array_yxc[row_start:row_end, :, :])

    def read_window(self, window: SourceWindow) -> SourceBlock:
        self._validate_window(window)
        image = np.asarray(self._array_yxc[window.row_start:window.row_end, window.col_start:window.col_end, :])
        if self._external_nodata_mask is None:
            nodata = None
        else:
            nodata = self._external_nodata_mask[window.row_start:window.row_end, window.col_start:window.col_end]
        missing = self._missing_channels
        if missing is not None and not isinstance(missing, Mapping):
            missing_array = np.asarray(missing)
            if missing_array.ndim in {2, 3} and tuple(missing_array.shape[:2]) == self.shape_yxc[:2]:
                missing = missing_array[window.row_start:window.row_end, window.col_start:window.col_end, ...]
        masks = self._validity_builder.build_source(
            image,
            missing_channels=missing,
            external_nodata_mask=nodata,
        )
        return SourceBlock(
            image_yxc=image,
            validity_yx=masks.validity_yx,
            validity_reason_yx=masks.reason_yx,
            row_start=window.row_start,
            col_start=window.col_start,
            source_shape_yx=self.shape_yxc[:2],
            provenance=self.provenance,
        )

    def physical_blocks(self, row_start: int, row_end: int) -> tuple[Any, ...]:
        self._validate_rows(row_start, row_end)
        return ((row_start, row_end, 0, self.shape_yxc[1]),)

    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> "ArraySourceReader":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


def _require_bounded_full_decode(
    source: SourceDescriptor,
    compute_profile: ComputeProfile | None,
    decoded_bytes: int,
) -> None:
    if not source.compressed:
        return
    if compute_profile is None or not compute_profile.allow_compressed_full_decode:
        raise PreprocessError(
            FailureReason.CODEC_UNAVAILABLE,
            "compressed source requires an explicitly admitted bounded full-decode path",
            state=RunState.RESOURCE_REJECTED,
            provenance={"source_format": source.format},
        )
    bound = source.full_decode_bytes
    if bound is None or bound < decoded_bytes:
        raise PreprocessError(
            FailureReason.RESOURCE_PREFLIGHT,
            "compressed source has no proven full-decode memory bound",
            state=RunState.RESOURCE_REJECTED,
            provenance={"declared_full_decode_bytes": bound, "actual_decoded_bytes": decoded_bytes},
        )
    if compute_profile.max_full_decode_bytes is not None and bound > compute_profile.max_full_decode_bytes:
        raise PreprocessError(
            FailureReason.RESOURCE_PREFLIGHT,
            "compressed source full-decode bound exceeds compute profile",
            state=RunState.RESOURCE_REJECTED,
            provenance={
                "full_decode_bytes": bound,
                "max_full_decode_bytes": compute_profile.max_full_decode_bytes,
            },
        )


class FileSourceReader(ArraySourceReader):
    """Lazy-format file reader with bounded full-decode admission."""

    def __init__(
        self,
        source: SourceDescriptor,
        profile: PreprocessingProfile,
        *,
        compute_profile: ComputeProfile | None = None,
        array_key: str | None = None,
        missing_channels: Any = None,
        external_nodata_mask: Any = None,
    ):
        if source.path is None:
            raise ValueError("FileSourceReader requires a path-backed source descriptor")
        self.source = source
        self.path = Path(source.path)
        self._owner: Any = None
        self._full_decode = False
        try:
            array, provenance = self._load(array_key=array_key, compute_profile=compute_profile)
            super().__init__(
                array,
                profile,
                missing_channels=missing_channels,
                external_nodata_mask=external_nodata_mask,
                provenance=provenance,
            )
        except PreprocessError:
            self.close()
            raise
        except (OSError, ValueError, TypeError) as exc:
            self.close()
            raise PreprocessError(
                FailureReason.IO_ERROR,
                f"unable to open source file: {self.path}",
                state=RunState.IO_FAULT,
                provenance={"path": str(self.path), "error": str(exc)},
            ) from exc

    def _load(self, *, array_key: str | None, compute_profile: ComputeProfile | None) -> tuple[np.ndarray, Mapping[str, Any]]:
        suffix = self.path.suffix.lower()
        if suffix == ".npy":
            array = np.load(self.path, mmap_mode="r")
            self._owner = array
            return array, {"backend": "numpy", "path": str(self.path), "mode": "memmap"}
        if suffix == ".npz":
            compressed_source = replace(self.source, compressed=True)
            declared_bound = self.source.full_decode_bytes if self.source.full_decode_bytes is not None else -1
            _require_bounded_full_decode(compressed_source, compute_profile, int(declared_bound))
            with np.load(self.path, allow_pickle=False) as loaded:
                keys = list(loaded.files)
                if array_key is not None:
                    if array_key not in loaded:
                        raise ValueError(f"array key '{array_key}' is not present in {self.path}")
                    array = np.asarray(loaded[array_key])
                elif len(keys) == 1:
                    array = np.asarray(loaded[keys[0]])
                elif "image" in loaded:
                    array = np.asarray(loaded["image"])
                elif "arr_0" in loaded:
                    array = np.asarray(loaded["arr_0"])
                else:
                    raise ValueError(f"source archive {self.path} contains multiple arrays; select array_key")
            _require_bounded_full_decode(compressed_source, compute_profile, int(array.nbytes))
            self._full_decode = True
            return array, {"backend": "numpy", "path": str(self.path), "mode": "full_decode", "decoded_bytes": int(array.nbytes)}
        if suffix in {".tif", ".tiff"}:
            try:
                import tifffile
            except ImportError as exc:
                raise PreprocessError(
                    FailureReason.CODEC_UNAVAILABLE,
                    "reading TIFF sources requires tifffile",
                    state=RunState.RESOURCE_REJECTED,
                    provenance={"path": str(self.path)},
                ) from exc
            tif = tifffile.TiffFile(self.path)
            self._owner = tif
            if len(tif.series) != 1:
                raise ValueError("source TIFF must contain exactly one series")
            series = tif.series[0]
            axes = str(series.axes).replace("S", "C")
            try:
                array = tifffile.memmap(self.path, series=0)
                mode = "memmap"
            except (OSError, ValueError):
                decoded_bytes = math.prod(series.shape) * np.dtype(series.dtype).itemsize
                compressed_source = replace(self.source, compressed=True)
                _require_bounded_full_decode(compressed_source, compute_profile, int(decoded_bytes))
                array = np.asarray(series.asarray())
                mode = "full_decode"
                self._full_decode = True
            return array, {
                "backend": "tifffile",
                "path": str(self.path),
                "axes": axes,
                "mode": mode,
                "decoded_bytes": int(np.asarray(array).nbytes),
            }
        raise ValueError(f"unsupported source format '{suffix or self.source.format}'")

    def _validate_source(self) -> None:
        super()._validate_source()
        expected_axes = _normalize_axes(self.source_schema.axes)
        actual_axes = str(self.provenance.get("axes", expected_axes))
        if actual_axes and _normalize_axes(actual_axes) != expected_axes:
            raise ValueError(f"source axes {actual_axes} do not match profile axes {expected_axes}")

    def close(self) -> None:
        if getattr(self, "_closed", False):
            return
        super().close()
        owner = getattr(self, "_owner", None)
        if owner is not None and hasattr(owner, "close"):
            owner.close()
        self._owner = None


def open_source_reader(
    source: Any,
    profile: PreprocessingProfile,
    *,
    compute_profile: ComputeProfile | None = None,
    array_key: str | None = None,
    missing_channels: Any = None,
    external_nodata_mask: Any = None,
) -> ArraySourceReader | FileSourceReader:
    """Open an array or supported path without model-specific assumptions."""

    if isinstance(source, np.ndarray):
        return ArraySourceReader(
            source,
            profile,
            missing_channels=missing_channels,
            external_nodata_mask=external_nodata_mask,
        )
    descriptor = SourceDescriptor.from_value(source)
    return FileSourceReader(
        descriptor,
        profile,
        compute_profile=compute_profile,
        array_key=array_key,
        missing_channels=missing_channels,
        external_nodata_mask=external_nodata_mask,
    )


__all__ = [
    "ArraySourceReader",
    "FileSourceReader",
    "SourceBlock",
    "SourceReader",
    "SourceWindow",
    "open_source_reader",
]
