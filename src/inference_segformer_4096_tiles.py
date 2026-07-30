"""SegFormer inference on a fixed 4096 x 4096 tiled representation.

The source RGB TIFF or NPY image is resampled to 4096 x 4096 with bilinear
interpolation, then inferred as a non-overlapping 4 x 4 grid of 1024 x 1024
tiles.  ``merged`` writes one 4096 x 4096 mask, while ``tiles`` writes the 16
individual mask TIFFs.

Source access reuses the large-image reader and resizes one destination strip
at a time. TIFF memory use therefore follows the selected streaming/cache mode.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


_SOURCE_ROOT = Path(__file__).resolve().parent
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from inference_segformer import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    MASK_CLOUD_VALUE,
    MODEL_CHANNELS,
    _normalization_scale,
    _normalize_patch,
    _predict_probability,
    _resolve_device,
    load_segformer,
)
from inference_segformer_large_image import (  # noqa: E402
    TIFF_EXTENSIONS,
    _StagedTiffOutput,
    _open_reader,
    _probe_image_shape,
)
from resource_guards import (  # noqa: E402
    DiskAllocation,
    require_disk_allocations,
    require_writable_parents,
)


RESIZED_IMAGE_SIZE = 4096
TILE_SIZE = 1024
OUTPUT_MODE_MERGED = "merged"
OUTPUT_MODE_TILES = "tiles"
OUTPUT_MODES = (OUTPUT_MODE_MERGED, OUTPUT_MODE_TILES)


@dataclass(frozen=True)
class _BilinearAxisPlan:
    """Precomputed source indices and weights for one resized dimension."""

    lower: np.ndarray
    upper: np.ndarray
    weight: np.ndarray


def _build_bilinear_axis_plan(source_length: int, target_length: int) -> _BilinearAxisPlan:
    if source_length <= 0 or target_length <= 0:
        raise ValueError("source_length and target_length must be positive")
    source_positions = (
        (np.arange(target_length, dtype=np.float64) + 0.5)
        * (float(source_length) / float(target_length))
        - 0.5
    )
    lower_unclipped = np.floor(source_positions).astype(np.int64)
    upper_unclipped = lower_unclipped + 1
    return _BilinearAxisPlan(
        lower=np.clip(lower_unclipped, 0, source_length - 1),
        upper=np.clip(upper_unclipped, 0, source_length - 1),
        weight=(source_positions - lower_unclipped).astype(np.float32),
    )


def _resize_row_bilinear(source_row: np.ndarray, plan: _BilinearAxisPlan) -> np.ndarray:
    """Resize an RGB source row horizontally to the plan's target width."""
    left = np.asarray(source_row[plan.lower], dtype=np.float32)
    right = np.asarray(source_row[plan.upper], dtype=np.float32)
    return left + (right - left) * plan.weight[:, np.newaxis]


def _read_resized_strip(
    reader,
    *,
    output_row_start: int,
    output_row_end: int,
    horizontal_plan: _BilinearAxisPlan,
    vertical_plan: _BilinearAxisPlan,
) -> np.ndarray:
    """Read a bilinearly resized RGB strip without materializing the source."""
    if output_row_start < 0 or output_row_end > len(vertical_plan.lower):
        raise ValueError("Invalid resized output row range")
    if output_row_end <= output_row_start:
        raise ValueError("A resized strip must contain at least one row")

    strip_height = output_row_end - output_row_start
    target_width = len(horizontal_plan.lower)
    strip = np.empty((strip_height, target_width, MODEL_CHANNELS), dtype=np.float32)
    row_cache: dict[int, np.ndarray] = {}

    def read_resized_source_row(source_row: int) -> np.ndarray:
        cached = row_cache.get(source_row)
        if cached is not None:
            return cached
        source = reader.read_rows(source_row, source_row + 1)[0]
        resized = _resize_row_bilinear(source, horizontal_plan)
        row_cache[source_row] = resized
        return resized

    for offset, output_row in enumerate(range(output_row_start, output_row_end)):
        lower = int(vertical_plan.lower[output_row])
        upper = int(vertical_plan.upper[output_row])
        top = read_resized_source_row(lower)
        bottom = read_resized_source_row(upper)
        vertical_weight = vertical_plan.weight[output_row]
        strip[offset] = top + (bottom - top) * vertical_weight

        # Source coordinates are monotonic. Keep only rows that may be used by
        # the next destination row instead of retaining a full strip cache.
        if output_row + 1 < output_row_end:
            next_lower = int(vertical_plan.lower[output_row + 1])
            next_upper = int(vertical_plan.upper[output_row + 1])
            keep = {next_lower, next_upper}
            for source_row in tuple(row_cache):
                if source_row not in keep:
                    del row_cache[source_row]

    return strip


def _write_tiff_atomically(destination: Path, array: np.ndarray) -> None:
    """Write a small mask tile through a same-filesystem temporary TIFF."""
    try:
        import tifffile
    except ImportError as exc:
        raise ImportError("Writing TIFF output requires tifffile") from exc

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.",
        suffix=destination.suffix,
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        tifffile.imwrite(temporary_path, array, metadata={"axes": "YX"}, compression=None)
        os.replace(temporary_path, destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _resolve_outputs(
    image_path: Path,
    output_mode: str,
    output_mask: str | Path | None,
    output_dir: str | Path | None,
) -> tuple[Path | None, Path | None]:
    if output_mode not in OUTPUT_MODES:
        raise ValueError(f"output_mode must be one of: {', '.join(OUTPUT_MODES)}")

    if output_mode == OUTPUT_MODE_MERGED:
        if output_dir is not None:
            raise ValueError("output_dir is only valid when output_mode='tiles'")
        mask_path = (
            Path(output_mask)
            if output_mask is not None
            else image_path.with_name(f"{image_path.stem}_segformer_4096_mask.tif")
        )
        if mask_path.suffix.lower() not in TIFF_EXTENSIONS:
            raise ValueError(f"output_mask must use a .tif or .tiff extension: {mask_path}")
        if mask_path.resolve(strict=False) == image_path.resolve(strict=False):
            raise ValueError("output_mask must not overwrite the input image")
        return mask_path, None

    if output_mask is not None:
        raise ValueError("output_mask is only valid when output_mode='merged'")
    tiles_path = (
        Path(output_dir)
        if output_dir is not None
        else image_path.with_name(f"{image_path.stem}_segformer_4096_mask_tiles")
    )
    if tiles_path.exists() and not tiles_path.is_dir():
        raise ValueError(f"output_dir must be a directory: {tiles_path}")
    return None, tiles_path


def _tile_paths(output_dir: Path, tiles_per_axis: int) -> dict[tuple[int, int], Path]:
    return {
        (tile_row, tile_column): output_dir / f"mask_r{tile_row:02d}_c{tile_column:02d}.tif"
        for tile_row in range(tiles_per_axis)
        for tile_column in range(tiles_per_axis)
    }


def infer_resized_4096_tiles(
    image_path: str | Path,
    *,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    output_mode: str = OUTPUT_MODE_MERGED,
    output_mask: str | Path | None = None,
    output_dir: str | Path | None = None,
    batch_size: int = 1,
    threshold: float = 0.5,
    device: str = "auto",
    input_scale: float | None = None,
    tiff_read_mode: str = "stream",
    tiff_cache_mode: str = "auto",
    tiff_cache_dir: str | Path | None = None,
    max_ram_cache_gib: float | str = 0.5,
    max_disk_cache_gib: float | str = 8.0,
    runtime_reserve_gib: float | str = 1.5,
    tiff_block_cache_mib: float | str = 64,
    channel_mapping: str | None = None,
    input_sidecar: str | Path | None = None,
    progress_every: int = 1,
    resize_size: int = RESIZED_IMAGE_SIZE,
    tile_size: int = TILE_SIZE,
    _filesystem_provider=None,
) -> dict[str, object]:
    """Resize an RGB image and run non-overlapping tiled SegFormer inference.

    The command-line entry point intentionally uses the required 4096 x 4096
    and 1024 x 1024 geometry. ``resize_size`` and ``tile_size`` are kept as
    keyword arguments to make the resampling implementation testable.
    """
    if resize_size <= 0:
        raise ValueError("resize_size must be positive")
    if tile_size <= 0 or resize_size % tile_size != 0:
        raise ValueError("tile_size must be positive and divide resize_size exactly")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if progress_every <= 0:
        raise ValueError("progress_every must be positive")

    image_path = Path(image_path)
    checkpoint_path = Path(checkpoint_path)
    output_mask_path, output_dir_path = _resolve_outputs(
        image_path,
        output_mode,
        output_mask,
        output_dir,
    )
    source_height, source_width, channels = _probe_image_shape(image_path)
    if channels != MODEL_CHANNELS:
        raise ValueError(f"Expected 3 RGB channels, got {channels}")

    tiles_per_axis = resize_size // tile_size
    tile_count = tiles_per_axis * tiles_per_axis
    tile_mask_bytes = tile_size * tile_size * np.dtype(np.uint8).itemsize
    output_tile_paths = (
        _tile_paths(output_dir_path, tiles_per_axis)
        if output_dir_path is not None
        else {}
    )
    if output_mask_path is not None:
        output_allocations = [
            DiskAllocation(
                output_mask_path,
                resize_size * resize_size * np.dtype(np.uint8).itemsize,
                "merged 4096 SegFormer mask TIFF",
            )
        ]
    else:
        output_allocations = [
            DiskAllocation(path, tile_mask_bytes, "individual 1024 SegFormer mask tile")
            for path in output_tile_paths.values()
        ]
    require_writable_parents(output_allocations)
    require_disk_allocations(output_allocations, provider=_filesystem_provider)

    resolved_device = _resolve_device(device)
    model = load_segformer(checkpoint_path, resolved_device)
    output_root = output_mask_path.parent if output_mask_path is not None else output_dir_path
    if output_root is None:
        raise RuntimeError("No output path was resolved")
    cache_dir = (
        Path(tiff_cache_dir)
        if tiff_cache_dir is not None
        else output_root / ".cube_nano-cache"
    )

    reader = _open_reader(
        image_path,
        tile_size=tile_size,
        batch_size=batch_size,
        tiff_read_mode=tiff_read_mode,
        tiff_cache_mode=tiff_cache_mode,
        tiff_cache_dir=cache_dir,
        max_ram_cache_gib=max_ram_cache_gib,
        max_disk_cache_gib=max_disk_cache_gib,
        runtime_reserve_gib=runtime_reserve_gib,
        tiff_block_cache_mib=tiff_block_cache_mib,
        output_allocations=output_allocations,
        channel_mapping=channel_mapping,
        input_sidecar=input_sidecar,
        filesystem_provider=_filesystem_provider,
    )
    with reader:
        reader_shape = tuple(int(value) for value in reader.shape)
        expected_shape = (source_height, source_width, channels)
        if reader_shape != expected_shape:
            raise ValueError(
                f"Input metadata changed while opening {image_path}: "
                f"expected {expected_shape}, got {reader_shape}"
            )

        scale = _normalization_scale(np.empty((), dtype=reader.dtype), input_scale)
        horizontal_plan = _build_bilinear_axis_plan(source_width, resize_size)
        vertical_plan = _build_bilinear_axis_plan(source_height, resize_size)
        mask_stage = (
            _StagedTiffOutput(output_mask_path, (resize_size, resize_size), np.uint8)
            if output_mask_path is not None
            else None
        )
        merged_mask = None
        if mask_stage is not None:
            try:
                merged_mask = mask_stage.open()
            except Exception:
                mask_stage.abort()
                raise
        elif output_dir_path is not None:
            output_dir_path.mkdir(parents=True, exist_ok=True)

        pending_patches: list[np.ndarray] = []
        pending_coordinates: list[tuple[int, int]] = []
        processed_tiles = 0
        cloud_pixel_count = 0
        started = time.perf_counter()

        def flush_batch() -> None:
            nonlocal cloud_pixel_count, processed_tiles
            if not pending_patches:
                return
            batch = np.stack(pending_patches, axis=0)
            probabilities = _predict_probability(model, batch, resolved_device, tile_size)
            for probability, (tile_row, tile_column) in zip(probabilities, pending_coordinates):
                tile_mask = np.where(
                    probability >= threshold,
                    MASK_CLOUD_VALUE,
                    0,
                ).astype(np.uint8)
                row_start = tile_row * tile_size
                column_start = tile_column * tile_size
                if merged_mask is not None:
                    merged_mask[
                        row_start : row_start + tile_size,
                        column_start : column_start + tile_size,
                    ] = tile_mask
                else:
                    _write_tiff_atomically(output_tile_paths[(tile_row, tile_column)], tile_mask)
                cloud_pixel_count += int(np.count_nonzero(tile_mask == MASK_CLOUD_VALUE))
                processed_tiles += 1
            pending_patches.clear()
            pending_coordinates.clear()
            if processed_tiles % progress_every == 0 or processed_tiles == tile_count:
                print(f"Processed {processed_tiles}/{tile_count} tiles", flush=True)

        try:
            for tile_row in range(tiles_per_axis):
                row_start = tile_row * tile_size
                resized_strip = _read_resized_strip(
                    reader,
                    output_row_start=row_start,
                    output_row_end=row_start + tile_size,
                    horizontal_plan=horizontal_plan,
                    vertical_plan=vertical_plan,
                )
                for tile_column in range(tiles_per_axis):
                    column_start = tile_column * tile_size
                    patch = resized_strip[
                        :,
                        column_start : column_start + tile_size,
                        :,
                    ]
                    normalized = _normalize_patch(patch, scale)
                    pending_patches.append(np.transpose(normalized, (2, 0, 1)))
                    pending_coordinates.append((tile_row, tile_column))
                    if len(pending_patches) >= batch_size:
                        flush_batch()
            flush_batch()
            if mask_stage is not None:
                mask_stage.commit()
        except Exception:
            if mask_stage is not None:
                mask_stage.abort()
            raise

        elapsed = time.perf_counter() - started
        return {
            "image": str(image_path),
            "checkpoint": str(checkpoint_path),
            "output_mode": output_mode,
            "mask": str(output_mask_path) if output_mask_path is not None else None,
            "mask_tiles": [str(path) for path in output_tile_paths.values()],
            "device": str(resolved_device),
            "image_shape": [source_height, source_width, channels],
            "resized_image_shape": [resize_size, resize_size, channels],
            "resize_method": "bilinear",
            "tile_size": tile_size,
            "tiles_per_axis": tiles_per_axis,
            "tile_count": tile_count,
            "batch_size": batch_size,
            "threshold": threshold,
            "input_scale": scale,
            "cloud_pixel_count": cloud_pixel_count,
            "cloud_ratio": float(cloud_pixel_count / (resize_size * resize_size)),
            "mask_bigtiff": mask_stage.bigtiff if mask_stage is not None else None,
            "reader_backend": getattr(reader, "backend", "unknown"),
            "reader_metrics": (
                reader.metrics.as_dict() if hasattr(reader, "metrics") else None
            ),
            "elapsed_seconds": round(elapsed, 3),
        }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Resize an RGB TIFF/NPY image to 4096 x 4096 and run SegFormer on "
            "sixteen 1024 x 1024 tiles"
        )
    )
    parser.add_argument("--image", required=True, help="Input RGB TIFF or NPY image")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument(
        "--output-mode",
        choices=OUTPUT_MODES,
        default=OUTPUT_MODE_MERGED,
        help="'merged' writes one 4096 x 4096 mask; 'tiles' writes 16 mask TIFFs",
    )
    parser.add_argument("--output-mask", default=None, help="Merged output uint8 TIFF/BigTIFF mask")
    parser.add_argument("--output-dir", default=None, help="Directory for individual mask tiles")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--input-scale", type=float, default=None)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--tiff-read-mode", choices=("auto", "stream", "full"), default="stream")
    parser.add_argument("--tiff-cache-mode", choices=("auto", "ram", "disk"), default="auto")
    parser.add_argument("--tiff-cache-dir", default=None)
    parser.add_argument("--max-ram-cache-gib", default="0.5")
    parser.add_argument("--max-disk-cache-gib", default="8.0")
    parser.add_argument("--runtime-reserve-gib", default="1.5")
    parser.add_argument("--tiff-block-cache-mib", default="64")
    parser.add_argument("--channel-mapping", default=None)
    parser.add_argument("--input-sidecar", default=None)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = infer_resized_4096_tiles(
        args.image,
        checkpoint_path=args.checkpoint,
        output_mode=args.output_mode,
        output_mask=args.output_mask,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        threshold=args.threshold,
        device=args.device,
        input_scale=args.input_scale,
        tiff_read_mode=args.tiff_read_mode,
        tiff_cache_mode=args.tiff_cache_mode,
        tiff_cache_dir=args.tiff_cache_dir,
        max_ram_cache_gib=args.max_ram_cache_gib,
        max_disk_cache_gib=args.max_disk_cache_gib,
        runtime_reserve_gib=args.runtime_reserve_gib,
        tiff_block_cache_mib=args.tiff_block_cache_mib,
        channel_mapping=args.channel_mapping,
        input_sidecar=args.input_sidecar,
        progress_every=args.progress_every,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
