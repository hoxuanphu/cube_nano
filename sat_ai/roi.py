"""Memmap-only scene windows and scene-anchored ROI patch geometry."""

from __future__ import annotations

import hashlib
import json
import math
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import tifffile

from protocol.schemas import ROI


class SceneContractError(ValueError):
    """A scene or validity artifact cannot enter the mission runtime."""


FINGERPRINT_CACHE_CAPACITY = 64


@dataclass(frozen=True)
class VerifiedFileFingerprint:
    """A content digest bound to one immutable on-disk file identity."""

    path: Path
    sha256: str
    device: int
    inode: int
    size: int
    mtime_ns: int

    def matches(self, path: str | Path, expected_sha256: str) -> bool:
        candidate = Path(path).resolve()
        if candidate != self.path or expected_sha256 != self.sha256:
            return False
        try:
            value = candidate.stat()
        except OSError:
            return False
        return (
            int(getattr(value, "st_dev", 0)) == self.device
            and int(getattr(value, "st_ino", 0)) == self.inode
            and int(value.st_size) == self.size
            and int(value.st_mtime_ns) == self.mtime_ns
        )


_fingerprint_lock = threading.RLock()
_fingerprint_cache: OrderedDict[tuple[str, str, int, int, int, int], VerifiedFileFingerprint] = OrderedDict()
_binary_mask_cache: OrderedDict[tuple[str, str, int, int, int, int], bool] = OrderedDict()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _checked_sha256(value: object, label: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise SceneContractError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _fingerprint_key(path: Path, digest: str) -> tuple[str, str, int, int, int, int]:
    value = path.stat()
    return (
        str(path.resolve()),
        digest,
        int(getattr(value, "st_dev", 0)),
        int(getattr(value, "st_ino", 0)),
        int(value.st_size),
        int(value.st_mtime_ns),
    )


def _remember_lru(cache: OrderedDict, key: tuple, value: object) -> None:
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > FINGERPRINT_CACHE_CAPACITY:
        cache.popitem(last=False)


def verify_file_fingerprint(path: str | Path, expected_sha256: str, *, label: str = "source fingerprint") -> VerifiedFileFingerprint:
    """Verify once, then reuse a digest only while the file identity is stable."""

    source = Path(path).resolve()
    digest = _checked_sha256(expected_sha256, label)
    try:
        key = _fingerprint_key(source, digest)
    except OSError as exc:
        raise SceneContractError(f"{label} path is unavailable") from exc
    with _fingerprint_lock:
        cached = _fingerprint_cache.get(key)
        if cached is not None:
            _fingerprint_cache.move_to_end(key)
            return cached
    if _sha256_file(source) != digest:
        raise SceneContractError(f"{label} does not match source bytes")
    # Detect an in-place replacement during the verification pass.
    try:
        current_key = _fingerprint_key(source, digest)
    except OSError as exc:
        raise SceneContractError(f"{label} path changed during verification") from exc
    if current_key != key:
        raise SceneContractError(f"{label} path changed during verification")
    verified = VerifiedFileFingerprint(source, digest, *key[2:])
    with _fingerprint_lock:
        _remember_lru(_fingerprint_cache, key, verified)
    return verified


def _require_verified_fingerprint(
    path: Path,
    digest: str,
    verified: VerifiedFileFingerprint | None,
    *,
    label: str,
) -> VerifiedFileFingerprint:
    if verified is not None:
        if not verified.matches(path, digest):
            raise SceneContractError(f"{label} is not valid for the current file identity")
        return verified
    return verify_file_fingerprint(path, digest, label=label)


def _verify_binary_mask(mask: np.memmap, fingerprint: VerifiedFileFingerprint) -> None:
    key = (
        str(fingerprint.path),
        fingerprint.sha256,
        fingerprint.device,
        fingerprint.inode,
        fingerprint.size,
        fingerprint.mtime_ns,
    )
    with _fingerprint_lock:
        if _binary_mask_cache.get(key):
            _binary_mask_cache.move_to_end(key)
            return
    # A validity mask is a binary contract. Do this once at admission so an
    # unknown value cannot silently become invalid through ``== 1``.
    for row_start in range(0, mask.shape[0], 1024):
        values = np.unique(np.asarray(mask[row_start : row_start + 1024, :]))
        if not np.isin(values, np.array([0, 1], dtype=np.uint8)).all():
            raise SceneContractError("VALIDITY_MASK_NOT_BINARY")
    with _fingerprint_lock:
        _remember_lru(_binary_mask_cache, key, True)


@dataclass
class ReadMetrics:
    source_read_calls: int = 0
    validity_read_calls: int = 0
    logical_source_bytes_read: int = 0
    logical_validity_bytes_read: int = 0

    @property
    def logical_bytes_read(self) -> int:
        return self.logical_source_bytes_read + self.logical_validity_bytes_read

    def as_dict(self) -> dict[str, int]:
        return {
            "source_read_calls": self.source_read_calls,
            "validity_read_calls": self.validity_read_calls,
            "logical_source_bytes_read": self.logical_source_bytes_read,
            "logical_validity_bytes_read": self.logical_validity_bytes_read,
            "logical_bytes_read": self.logical_bytes_read,
        }


@dataclass(frozen=True)
class PatchWindow:
    x: int
    y: int
    width: int
    height: int
    scene_width: int
    scene_height: int
    roi_weight: int

    @property
    def x_end(self) -> int:
        return self.x + self.width

    @property
    def y_end(self) -> int:
        return self.y + self.height

    @property
    def pad_right(self) -> int:
        return max(0, self.width - self.scene_width)

    @property
    def pad_bottom(self) -> int:
        return max(0, self.height - self.scene_height)


@dataclass
class SceneWindow:
    path: Path
    source: np.ndarray
    input_spec_id: str
    band_order: tuple[str, ...]
    validity_kind: str
    validity_mask: np.ndarray | None = None
    nodata_values: tuple[int, ...] = ()
    nodata_any_band: bool = True
    metrics: ReadMetrics = field(default_factory=ReadMetrics)
    source_dtype: str = ""
    normalization_id: str = ""

    @property
    def height(self) -> int:
        return int(self.source.shape[0])

    @property
    def width(self) -> int:
        return int(self.source.shape[1])

    @property
    def channels(self) -> int:
        return int(self.source.shape[2])

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(item) for item in self.source.shape)

    def _validate_window(self, x: int, y: int, width: int, height: int) -> None:
        values = {"x": x, "y": y, "width": width, "height": height}
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values.values()):
            raise TypeError("scene window coordinates must be integers")
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise ValueError("scene window coordinates must be positive and in bounds")
        if x + width > self.width or y + height > self.height:
            raise ValueError(f"scene window [{x}, {y}, {width}, {height}] exceeds {self.shape}")

    def read_window(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        """Read only the requested X/Y window from the source memmap."""
        self._validate_window(x, y, width, height)
        self.metrics.source_read_calls += 1
        self.metrics.logical_source_bytes_read += width * height * self.channels * self.source.dtype.itemsize
        return np.asarray(self.source[y : y + height, x : x + width, :])

    def read_validity_window(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        *,
        source_window: np.ndarray | None = None,
    ) -> np.ndarray:
        self._validate_window(x, y, width, height)
        self.metrics.validity_read_calls += 1
        if self.validity_kind == "all_valid":
            return np.ones((height, width), dtype=bool)
        if self.validity_kind == "mask":
            assert self.validity_mask is not None
            self.metrics.logical_validity_bytes_read += width * height
            return np.asarray(self.validity_mask[y : y + height, x : x + width], dtype=np.uint8) == 1
        if self.validity_kind == "nodata_value":
            source = source_window if source_window is not None else self.read_window(x, y, width, height)
            comparisons = np.stack([source[:, :, index] == value for index, value in enumerate(self.nodata_values)], axis=2)
            invalid = np.any(comparisons, axis=2) if self.nodata_any_band else np.all(comparisons, axis=2)
            return ~invalid
        raise AssertionError(f"unknown validity kind {self.validity_kind}")

    def close(self) -> None:
        mmap = getattr(self.source, "_mmap", None)
        if mmap is not None and not mmap.closed:
            mmap.close()
        if self.validity_mask is not None:
            mmap = getattr(self.validity_mask, "_mmap", None)
            if mmap is not None and not mmap.closed:
                mmap.close()

    def __enter__(self) -> "SceneWindow":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.close()
        return False


def _load_sidecar(sidecar_path: Path) -> dict:
    try:
        value = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SceneContractError(f"scene sidecar not found: {sidecar_path}") from None
    except json.JSONDecodeError as exc:
        raise SceneContractError(f"invalid scene sidecar JSON: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise SceneContractError("scene sidecar schema_version must be 1")
    return value


def open_memmap_scene(
    source_path: str | Path,
    sidecar_path: str | Path,
    *,
    verify_source_fingerprint: bool = True,
    verified_source_fingerprint: VerifiedFileFingerprint | None = None,
) -> SceneWindow:
    """Open a single-series, single-level, uncompressed memmap-compatible TIFF."""
    source_path = Path(source_path).resolve()
    sidecar = _load_sidecar(Path(sidecar_path))
    fingerprint = sidecar.get("source_fingerprint", {})
    if not isinstance(fingerprint, dict) or fingerprint.get("algorithm") != "sha256":
        raise SceneContractError("source_fingerprint must be a SHA-256 object")
    source_digest = _checked_sha256(fingerprint.get("digest", ""), "source_fingerprint")
    if verify_source_fingerprint:
        _require_verified_fingerprint(
            source_path,
            source_digest,
            verified_source_fingerprint,
            label="source sidecar fingerprint",
        )
    elif verified_source_fingerprint is not None:
        _require_verified_fingerprint(
            source_path,
            source_digest,
            verified_source_fingerprint,
            label="source sidecar fingerprint",
        )
    try:
        source = tifffile.memmap(source_path, series=0, level=0, mode="r")
    except (OSError, ValueError, KeyError) as exc:
        raise SceneContractError("UNSUPPORTED_SCENE_FORMAT: source must be TIFF memmap-compatible") from exc
    if not isinstance(source, np.memmap) or source.ndim != 3 or source.shape[2] != 3:
        mmap = getattr(source, "_mmap", None)
        if mmap is not None:
            mmap.close()
        raise SceneContractError("scene must be a 3-channel YXC memmap")
    if source.dtype != np.dtype("uint16"):
        source._mmap.close()
        raise SceneContractError("scene source dtype must be uint16")
    if str(sidecar.get("axes")) != "YXC" or tuple(sidecar.get("shape", ())) != tuple(source.shape):
        source._mmap.close()
        raise SceneContractError("scene sidecar axes/shape does not match source")
    if tuple(str(item).lower() for item in sidecar.get("band_order", ())) != ("red", "green", "blue"):
        source._mmap.close()
        raise SceneContractError("scene sidecar band_order must be RGB")
    if str(sidecar.get("dtype")) != "uint16":
        source._mmap.close()
        raise SceneContractError("scene sidecar dtype must be uint16")
    validity = sidecar.get("validity", {"kind": "all_valid"})
    if not isinstance(validity, dict):
        source._mmap.close()
        raise SceneContractError("validity must be an object")
    kind = str(validity.get("kind"))
    if kind == "all_valid":
        return SceneWindow(
            source_path,
            source,
            str(sidecar.get("input_spec_id", "")),
            ("red", "green", "blue"),
            kind,
            source_dtype=str(sidecar.get("dtype", "")),
            normalization_id=_normalization_id(sidecar),
        )
    if kind == "nodata_value":
        values = tuple(int(item) for item in validity.get("values", ()))
        if len(values) != 3 or any(item < 0 or item > 65535 for item in values):
            source._mmap.close()
            raise SceneContractError("nodata_value validity must contain three uint16 values")
        rule = validity.get("rule", "any")
        if rule not in {"any", "all"}:
            source._mmap.close()
            raise SceneContractError("UNKNOWN_NODATA_RULE")
        return SceneWindow(
            source_path,
            source,
            str(sidecar.get("input_spec_id", "")),
            ("red", "green", "blue"),
            kind,
            nodata_values=values,
            nodata_any_band=rule == "any",
            source_dtype=str(sidecar.get("dtype", "")),
            normalization_id=_normalization_id(sidecar),
        )
    if kind != "mask":
        source._mmap.close()
        raise SceneContractError(f"unsupported validity kind {kind}")
    mask_path = (source_path.parent / str(validity.get("relative_path", ""))).resolve()
    if source_path.parent.resolve() not in mask_path.parents:
        source._mmap.close()
        raise SceneContractError("validity mask path escapes the scene package")
    try:
        mask = tifffile.memmap(mask_path, series=0, level=0, mode="r")
    except (OSError, ValueError, KeyError) as exc:
        source._mmap.close()
        raise SceneContractError("UNSUPPORTED_VALIDITY_MASK: mask must be memmap-compatible") from exc
    if not isinstance(mask, np.memmap) or mask.shape != source.shape[:2] or mask.dtype != np.dtype("uint8"):
        source._mmap.close()
        if isinstance(mask, np.memmap):
            mask._mmap.close()
        raise SceneContractError("UNSUPPORTED_VALIDITY_MASK: mask must be HxW uint8 memmap")
    if not mask.flags.c_contiguous:
        source._mmap.close()
        mask._mmap.close()
        raise SceneContractError("UNSUPPORTED_VALIDITY_MASK: mask must be contiguous")
    try:
        mask_digest = _checked_sha256(validity.get("sha256", ""), "validity mask SHA-256")
        mask_fingerprint = verify_file_fingerprint(
            mask_path,
            mask_digest,
            label="validity mask SHA-256",
        )
    except SceneContractError:
        source._mmap.close()
        mask._mmap.close()
        raise SceneContractError("UNSUPPORTED_VALIDITY_MASK: mask SHA-256 mismatch")
    try:
        _verify_binary_mask(mask, mask_fingerprint)
    except SceneContractError:
        source._mmap.close()
        mask._mmap.close()
        raise
    return SceneWindow(
        source_path,
        source,
        str(sidecar.get("input_spec_id", "")),
        ("red", "green", "blue"),
        kind,
        mask,
        source_dtype=str(sidecar.get("dtype", "")),
        normalization_id=_normalization_id(sidecar),
    )


def _normalization_id(sidecar: dict) -> str:
    value = sidecar.get("normalization", "")
    if isinstance(value, dict):
        return str(value.get("id", ""))
    return str(value)


def iter_patch_windows(scene_shape: tuple[int, int, int], roi: ROI, patch_size: int) -> Iterator[PatchWindow]:
    height, width, _ = scene_shape
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if roi.x_end > width or roi.y_end > height:
        raise ValueError("ROI exceeds scene bounds")
    start_x = (roi.x // patch_size) * patch_size
    start_y = (roi.y // patch_size) * patch_size
    for y in range(start_y, roi.y_end, patch_size):
        for x in range(start_x, roi.x_end, patch_size):
            scene_width = min(patch_size, width - x)
            scene_height = min(patch_size, height - y)
            intersection_width = max(0, min(x + scene_width, roi.x_end) - max(x, roi.x))
            intersection_height = max(0, min(y + scene_height, roi.y_end) - max(y, roi.y))
            yield PatchWindow(x, y, patch_size, patch_size, scene_width, scene_height, intersection_width * intersection_height)


def build_padded_patch(scene: SceneWindow, window: PatchWindow) -> tuple[np.ndarray, np.ndarray]:
    raw = scene.read_window(window.x, window.y, window.scene_width, window.scene_height)
    validity = scene.read_validity_window(
        window.x,
        window.y,
        window.scene_width,
        window.scene_height,
        source_window=raw,
    )
    patch = np.zeros((window.height, window.width, scene.channels), dtype=scene.source.dtype)
    valid_patch = np.zeros((window.height, window.width), dtype=bool)
    patch[: window.scene_height, : window.scene_width, :] = raw
    valid_patch[: window.scene_height, : window.scene_width] = validity
    return patch, valid_patch


class ProgressEmitter:
    def __init__(self, callback: Callable[[int, int, int], None] | None, *, min_delta_bp: int = 100, min_interval_ms: int = 1000, max_silence_ms: int = 5000):
        self.callback = callback
        self.min_delta_bp = min_delta_bp
        self.min_interval_ms = min_interval_ms
        self.max_silence_ms = max_silence_ms
        self._last_processed = 0
        self._last_elapsed_ms = 0

    def emit(self, processed: int, total: int, elapsed_ms: int, *, force: bool = False) -> None:
        if self.callback is None:
            return
        delta_bp = ((processed - self._last_processed) * 10000 // max(total, 1))
        due_to_time = elapsed_ms - self._last_elapsed_ms >= self.max_silence_ms
        due_to_interval = elapsed_ms - self._last_elapsed_ms >= self.min_interval_ms
        if force or (due_to_interval and delta_bp >= self.min_delta_bp) or due_to_time:
            self.callback(processed, total, elapsed_ms)
            self._last_processed = processed
            self._last_elapsed_ms = elapsed_ms
