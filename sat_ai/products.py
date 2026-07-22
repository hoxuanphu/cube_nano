"""Deterministic display artifacts, manifests and byte-canonical USTAR bundles."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import tifffile
from PIL import Image

from protocol.canonical import canonical_json
from protocol.schemas import ProductRef, RequestKey

from .roi import ROI, SceneWindow


IO_BUFFER_BYTES = 1024 * 1024


@dataclass(frozen=True)
class DisplayProfile:
    profile_id: str = "rgb-uint16-fixed-v1"
    black_point: int = 0
    white_point: int = 65535
    gamma_numerator: int = 1
    gamma_denominator: int = 1

    def validate(self) -> None:
        if not 0 <= self.black_point < self.white_point <= 65535:
            raise ValueError("display black/white points must be ordered uint16 values")
        if self.gamma_numerator <= 0 or self.gamma_denominator <= 0:
            raise ValueError("display gamma must be positive")


def tone_map_rgb(image: np.ndarray, profile: DisplayProfile = DisplayProfile()) -> np.ndarray:
    profile.validate()
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.dtype("uint16"):
        raise ValueError("tone mapping expects HWC uint16 RGB data")
    values = image.astype(np.float32)
    values = np.clip(
        (values - profile.black_point) / float(profile.white_point - profile.black_point),
        0.0,
        1.0,
    )
    gamma = profile.gamma_numerator / profile.gamma_denominator
    if gamma != 1.0:
        values = np.power(values, gamma, dtype=np.float32)
    return np.rint(values * 255.0).astype(np.uint8)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(IO_BUFFER_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _file_size(path: Path) -> int:
    size = path.stat().st_size
    if size < 0:
        raise ValueError(f"artifact has a negative size: {path}")
    return int(size)


def _write_tiff(path: Path, image: np.ndarray, axes: str) -> None:
    tifffile.imwrite(path, np.asarray(image), metadata={"axes": axes}, compression=None)


def _octal(value: int, width: int) -> bytes:
    encoded = format(value, "o").encode("ascii")
    if len(encoded) > width - 1:
        raise ValueError("USTAR numeric field overflow")
    return b"0" * (width - len(encoded) - 1) + encoded + b"\0"


def _ustar_header(name: str, size: int) -> bytes:
    encoded_name = name.encode("ascii")
    if not 0 < len(encoded_name) <= 100 or "\\" in name or name.startswith("/") or ".." in name.split("/"):
        raise ValueError("USTAR path must be a relative ASCII path of <=100 bytes")
    header = bytearray(512)
    header[0 : len(encoded_name)] = encoded_name
    header[100:108] = _octal(0o644, 8)
    header[108:116] = _octal(0, 8)
    header[116:124] = _octal(0, 8)
    header[124:136] = _octal(size, 12)
    header[136:148] = _octal(0, 12)
    header[148:156] = b"        "
    header[156] = ord("0")
    header[257:263] = b"ustar\0"
    header[263:265] = b"00"
    checksum = sum(header)
    header[148:156] = f"{checksum:06o}".encode("ascii") + b"\0 "
    return bytes(header)


def _normalized_ustar_entries(entries: Mapping[str, bytes | bytearray | memoryview | Path]) -> dict[str, bytes | Path]:
    normalized: dict[str, bytes | Path] = {}
    for path, data in entries.items():
        path = path.replace("\\", "/")
        if path in normalized:
            raise ValueError(f"duplicate USTAR path {path}")
        if isinstance(data, Path):
            if not data.is_file():
                raise FileNotFoundError(f"USTAR entry is not a regular file: {data}")
            normalized[path] = data
        elif isinstance(data, (bytes, bytearray, memoryview)):
            normalized[path] = bytes(data)
        else:
            raise TypeError("USTAR entries must be bytes or regular file paths")
    return normalized


def _entry_size(value: bytes | Path) -> int:
    return len(value) if isinstance(value, bytes) else _file_size(value)


def write_ustar(
    entries: Mapping[str, bytes | bytearray | memoryview | Path],
    destination: str | Path,
) -> tuple[int, str]:
    """Write a deterministic USTAR archive using only bounded file reads."""

    normalized = _normalized_ustar_entries(entries)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0

    def write(stream, data: bytes) -> None:
        nonlocal size
        stream.write(data)
        digest.update(data)
        size += len(data)

    try:
        with destination.open("xb") as output:
            for path in sorted(normalized):
                value = normalized[path]
                entry_size = _entry_size(value)
                write(output, _ustar_header(path, entry_size))
                if isinstance(value, bytes):
                    for offset in range(0, len(value), IO_BUFFER_BYTES):
                        write(output, value[offset : offset + IO_BUFFER_BYTES])
                else:
                    remaining = entry_size
                    with value.open("rb") as source:
                        while remaining:
                            chunk = source.read(min(IO_BUFFER_BYTES, remaining))
                            if not chunk:
                                raise RuntimeError(f"USTAR entry changed while reading: {value}")
                            write(output, chunk)
                            remaining -= len(chunk)
                        if source.read(1):
                            raise RuntimeError(f"USTAR entry changed while reading: {value}")
                write(output, b"\0" * ((-entry_size) % 512))
            write(output, b"\0" * 1024)
            output.flush()
            os.fsync(output.fileno())
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return size, digest.hexdigest()


def build_ustar(entries: dict[str, bytes]) -> bytes:
    """Compatibility helper for small in-memory callers and golden vectors."""

    with tempfile.TemporaryDirectory(prefix="ustar-compat-") as temporary:
        archive_path = Path(temporary) / "bundle.tar"
        write_ustar(entries, archive_path)
        result = bytearray()
        with archive_path.open("rb") as stream:
            while chunk := stream.read(IO_BUFFER_BYTES):
                result.extend(chunk)
        return bytes(result)


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".part", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            Path(temporary).unlink()
        except FileNotFoundError:
            pass


def cleanup_staging_products(output_directory: str | Path) -> list[str]:
    root = Path(output_directory)
    removed = []
    if not root.exists():
        return removed
    for path in root.glob("**/.staging-*"):
        if path.is_dir():
            shutil.rmtree(path)
            removed.append(str(path))
    return removed


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def build_products(
    result: dict[str, Any],
    scene: SceneWindow,
    output_directory: str | Path,
    product_ref: ProductRef,
    origin_request_key: RequestKey,
    *,
    source_sha256: str,
    display_profile: DisplayProfile = DisplayProfile(),
    created_at: str = "1970-01-01T00:00:00Z",
) -> dict[str, Any]:
    """Build and atomically publish one deterministic product bundle."""
    display_profile.validate()
    roi = ROI.from_dict(result["roi"])
    source_crop = scene.read_window(roi.x, roi.y, roi.width, roi.height)
    quicklook = tone_map_rgb(source_crop, display_profile)
    mask = np.asarray(result["cloud_mask"], dtype=np.uint8)
    if mask.shape != (roi.height, roi.width):
        raise ValueError("inference cloud_mask shape does not match ROI")
    validity = result.get("validity_mask")
    if validity is not None:
        validity = np.asarray(validity, dtype=np.uint8)
        if validity.shape != (roi.height, roi.width):
            raise ValueError("inference validity_mask shape does not match ROI")
        if not np.isin(np.unique(validity), (0, 1)).all():
            raise ValueError("inference validity_mask must contain only 0/1")
        if np.any((validity == 0) & (mask != 0)):
            raise ValueError("cloud_mask must be clear where validity_mask is invalid")
    root = Path(output_directory)
    product_directory = root / f"{product_ref.origin_boot_id:08x}" / f"{product_ref.product_id:08x}"
    product_directory.parent.mkdir(parents=True, exist_ok=True)
    if product_directory.exists():
        raise FileExistsError(f"product directory already exists: {product_directory}")
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".staging-{product_ref.product_id:08x}-",
            dir=product_directory.parent,
        )
    )
    try:
        quicklook_path = staging / "quicklook.webp"
        Image.fromarray(quicklook, mode="RGB").save(quicklook_path, format="WEBP", lossless=True, method=6)
        mask_path = staging / "cloud_mask.tif"
        _write_tiff(mask_path, mask, "YX")
        artifact_paths: dict[str, Path] = {
            "quicklook.webp": quicklook_path,
            "cloud_mask.tif": mask_path,
        }
        if validity is not None:
            validity_path = staging / "validity_mask.tif"
            _write_tiff(validity_path, validity, "YX")
            artifact_paths["validity_mask.tif"] = validity_path
        if result["science_decision"] == "ACCEPTED":
            crop_path = staging / "crop.tif"
            _write_tiff(crop_path, source_crop, "YXC")
            artifact_paths["crop.tif"] = crop_path
        artifact_list = [
            {"path": path, "size": _file_size(artifact_path), "sha256": _sha256_file(artifact_path)}
            for path, artifact_path in sorted(artifact_paths.items())
        ]
        manifest = {
            "schema_version": 1,
            "product_type": "ANALYSIS",
            "product_ref": product_ref.as_dict(),
            "origin_request_key": origin_request_key.as_dict(),
            "created_at": created_at,
            "source_sha256": source_sha256,
            "scene_ref": result.get("scene_ref"),
            "roi": result["roi"],
            "patch_count": result["patch_count"],
            "tiling_algorithm_id": result["tiling_algorithm_id"],
            "coverage_algorithm_id": result["coverage_algorithm_id"],
            "validity_algorithm_id": result["validity_algorithm_id"],
            "padding_algorithm_id": result["padding_algorithm_id"],
            "model_task": result.get("model_task", "patch_classification"),
            "model_release_id": result["model_release_id"],
            "model_sha256": result["model_sha256"],
            "assurance_level": result["science_status"],
            "input_spec_id": result["input_spec_id"],
            "config_snapshot": result["config_snapshot"],
            "threshold_mapping_id": result["threshold_mapping_id"],
            "threshold_lut_sha256": result["threshold_lut_sha256"],
            "model_threshold_bp": result["model_threshold_bp"],
            "coverage_limit_bp": result["coverage_limit_bp"],
            "science_decision": result["science_decision"],
            "display_profile": {
                "id": display_profile.profile_id,
                "black_point": display_profile.black_point,
                "white_point": display_profile.white_point,
                "gamma_numerator": display_profile.gamma_numerator,
                "gamma_denominator": display_profile.gamma_denominator,
            },
            "artifacts": artifact_list,
        }
        if result.get("cloud_positive_tile_area_ratio_bp") is not None:
            manifest["cloud_positive_tile_area_ratio_bp"] = result["cloud_positive_tile_area_ratio_bp"]
        for key in (
            "pixel_cloud_ratio_bp",
            "valid_pixel_ratio_bp",
            "decision_spec_id",
            "postprocess_id",
            "product_spec_id",
            "acceptance_profile_id",
            "target_id",
            "deployment_profile_id",
        ):
            if key in result:
                manifest[key] = result[key]
        manifest_bytes = canonical_json(manifest) + b"\n"
        _atomic_write(staging / "manifest.json", manifest_bytes)
        bundle_size, bundle_sha256 = write_ustar(
            {"manifest.json": staging / "manifest.json", **artifact_paths},
            staging / "bundle.tar",
        )
        _fsync_directory(staging)
        os.replace(staging, product_directory)
        _fsync_directory(product_directory.parent)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return {
        "product_ref": product_ref.as_dict(),
        "product_directory": str(product_directory),
        "manifest_sha256": _sha256_bytes(manifest_bytes),
        "bundle_sha256": bundle_sha256,
        "bundle_size": bundle_size,
        "artifacts": artifact_list,
    }
