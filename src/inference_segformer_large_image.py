"""Memory-bounded SegFormer inference for large RGB TIFF and NPY images.

Unlike :mod:`inference_segformer`, this entry point never creates a full-size
NumPy mask or probability array in RAM. Output pixels are written tile by tile
to staged TIFF/BigTIFF memmaps and are atomically published after inference.

TIFF input uses the repository's validated ``TiffReader``. ``--tiff-read-mode
stream`` is the safest setting for very large input because it requires a
memory-mappable TIFF. ``auto`` and ``full`` retain the repository's decoded
cache behaviour for compressed TIFF files.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch


_SOURCE_ROOT = Path(__file__).resolve().parent
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from inference_segformer import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_TILE_SIZE,
    MASK_CLOUD_VALUE,
    MODEL_CHANNELS,
    _normalize_patch,
    _normalization_scale,
    _predict_probability,
    _resolve_device,
    load_segformer,
)
from input_contract import legacy_input_spec  # noqa: E402
from resource_guards import (  # noqa: E402
    DiskAllocation,
    ReaderBudget,
    require_disk_allocations,
    require_writable_parents,
)
from tiff_reader import TiffReader, close_memmap  # noqa: E402


TIFF_EXTENSIONS = {".tif", ".tiff"}
NPY_EXTENSIONS = {".npy"}
_CLASSIC_TIFF_LIMIT = 4 * 1024**3
_BIGTIFF_SAFETY_MARGIN = 32 * 1024**2


class _NpyReader:
    """Session-scoped NPY memmap reader returning HWC RGB row strips."""

    backend = "npy-memmap"

    def __init__(self, path: Path) -> None:
        self.path = path
        self._array = np.load(path, mmap_mode="r", allow_pickle=False)
        if not isinstance(self._array, np.ndarray):
            raise ValueError(f"Expected a single NPY array in {path}")
        if self._array.ndim != 3:
            raise ValueError(
                f"Expected an RGB image with shape (H, W, 3) or (3, H, W), got "
                f"{self._array.shape} from {path}"
            )

        if self._array.shape[-1] == MODEL_CHANNELS:
            self._channel_first = False
            self.shape = tuple(int(value) for value in self._array.shape)
        elif self._array.shape[0] == MODEL_CHANNELS and self._array.shape[-1] != MODEL_CHANNELS:
            self._channel_first = True
            self.shape = (
                int(self._array.shape[1]),
                int(self._array.shape[2]),
                MODEL_CHANNELS,
            )
        else:
            raise ValueError(f"SegFormer expects exactly 3 RGB channels, got {self._array.shape} from {path}")
        self.dtype = self._array.dtype

    def __enter__(self) -> "_NpyReader":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.close()
        return False

    def read_rows(self, row_start: int, row_end: int) -> np.ndarray:
        if not 0 <= row_start < row_end <= self.shape[0]:
            raise ValueError(f"Invalid NPY row range [{row_start}, {row_end})")
        if self._channel_first:
            return np.moveaxis(self._array[:, row_start:row_end, :], 0, -1)
        return np.asarray(self._array[row_start:row_end, :, :])

    def close(self) -> None:
        if self._array is not None:
            close_memmap(self._array)
            self._array = None


class _StagedTiffOutput:
    """Write a single grayscale TIFF through a same-filesystem temporary file."""

    def __init__(self, destination: Path, shape: tuple[int, int], dtype: np.dtype) -> None:
        self.destination = destination
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self.temporary_path: Path | None = None
        self.array: np.memmap | None = None
        self._committed = False

    @property
    def nbytes(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64)) * self.dtype.itemsize

    @property
    def bigtiff(self) -> bool:
        return self.nbytes >= _CLASSIC_TIFF_LIMIT - _BIGTIFF_SAFETY_MARGIN

    def open(self) -> np.memmap:
        if self.array is not None:
            return self.array
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.destination.name}.",
            suffix=".tmp",
            dir=self.destination.parent,
        )
        os.close(descriptor)
        self.temporary_path = Path(temporary_name)
        self.temporary_path.unlink()
        try:
            import tifffile
        except ImportError as exc:
            raise ImportError("Writing TIFF output requires tifffile") from exc
        try:
            self.array = tifffile.memmap(
                self.temporary_path,
                shape=self.shape,
                dtype=self.dtype,
                mode="w+",
                bigtiff=self.bigtiff,
                metadata={"axes": "YX"},
            )
        except Exception:
            self.temporary_path.unlink(missing_ok=True)
            self.temporary_path = None
            raise
        return self.array

    def close(self) -> None:
        if self.array is not None:
            close_memmap(self.array)
            self.array = None

    def commit(self) -> None:
        if self.temporary_path is None:
            raise RuntimeError("Output has not been opened")
        self.close()
        os.replace(self.temporary_path, self.destination)
        self._committed = True
        self.temporary_path = None

    def abort(self) -> None:
        self.close()
        if not self._committed and self.temporary_path is not None:
            self.temporary_path.unlink(missing_ok=True)
            self.temporary_path = None


class _ProbabilityAccumulator:
    """Accumulate overlapping tile probabilities in disk-backed arrays."""

    def __init__(self, directory: Path, shape: tuple[int, int]) -> None:
        self.directory = directory
        self.shape = shape
        self.sum_path: Path | None = None
        self.weight_path: Path | None = None
        self.probability_sum: np.memmap | None = None
        self.weight: np.memmap | None = None

    @property
    def nbytes(self) -> int:
        pixels = int(np.prod(self.shape, dtype=np.int64))
        return pixels * (np.dtype(np.float32).itemsize + np.dtype(np.uint32).itemsize)

    def _open_array(self, prefix: str, dtype: np.dtype) -> tuple[Path, np.memmap]:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=prefix,
            suffix=".tmp",
            dir=self.directory,
        )
        os.close(descriptor)
        path = Path(temporary_name)
        try:
            array = np.memmap(path, mode="w+", dtype=dtype, shape=self.shape)
            array[...] = 0
            array.flush()
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return path, array

    def open(self) -> None:
        if self.probability_sum is not None or self.weight is not None:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        try:
            self.sum_path, self.probability_sum = self._open_array(
                ".segformer-probability-sum-",
                np.dtype(np.float32),
            )
            self.weight_path, self.weight = self._open_array(
                ".segformer-probability-weight-",
                np.dtype(np.uint32),
            )
        except Exception:
            self.cleanup()
            raise

    def add(
        self,
        probability: np.ndarray,
        row_start: int,
        row_end: int,
        column_start: int,
        column_end: int,
    ) -> None:
        if self.probability_sum is None or self.weight is None:
            raise RuntimeError("Probability accumulator is not open")
        sum_view = self.probability_sum[row_start:row_end, column_start:column_end]
        weight_view = self.weight[row_start:row_end, column_start:column_end]
        np.add(sum_view, probability, out=sum_view, casting="unsafe")
        np.add(weight_view, np.uint32(1), out=weight_view, casting="unsafe")

    def close(self) -> None:
        if self.probability_sum is not None:
            close_memmap(self.probability_sum)
            self.probability_sum = None
        if self.weight is not None:
            close_memmap(self.weight)
            self.weight = None

    def cleanup(self) -> None:
        self.close()
        if self.sum_path is not None:
            self.sum_path.unlink(missing_ok=True)
            self.sum_path = None
        if self.weight_path is not None:
            self.weight_path.unlink(missing_ok=True)
            self.weight_path = None


def _resolve_stride(tile_size: int, stride: int | None) -> int:
    resolved = tile_size // 2 if stride is None else int(stride)
    if resolved <= 0 or resolved > tile_size:
        raise ValueError("stride must be between 1 and tile_size")
    return resolved


def _window_starts(length: int, tile_size: int, stride: int) -> tuple[int, ...]:
    """Return starts with full coverage, shifting the last window to the edge."""
    if length <= tile_size:
        return (0,)
    last_start = length - tile_size
    starts = list(range(0, last_start + 1, stride))
    if starts[-1] != last_start:
        starts.append(last_start)
    return tuple(starts)


def _needs_bigtiff(nbytes: int) -> bool:
    """Return whether a TIFF allocation needs BigTIFF before header overhead."""
    return int(nbytes) >= _CLASSIC_TIFF_LIMIT - _BIGTIFF_SAFETY_MARGIN


def _validate_output_paths(
    image_path: Path,
    output_mask: Path,
    output_probability: Path | None,
) -> None:
    for path, label in ((output_mask, "output_mask"), (output_probability, "output_probability")):
        if path is None:
            continue
        if path.suffix.lower() not in TIFF_EXTENSIONS:
            raise ValueError(f"{label} must use a .tif or .tiff extension: {path}")
        if path.resolve(strict=False) == image_path.resolve(strict=False):
            raise ValueError(f"{label} must not overwrite the input image")
    if output_probability is not None and output_probability.resolve(strict=False) == output_mask.resolve(strict=False):
        raise ValueError("output_mask and output_probability must be different files")


def _probe_image_shape(image_path: Path) -> tuple[int, int, int]:
    """Read only enough input metadata to reserve output disk space."""
    suffix = image_path.suffix.lower()
    if suffix in NPY_EXTENSIONS:
        reader = _NpyReader(image_path)
        try:
            return reader.shape
        finally:
            reader.close()
    if suffix not in TIFF_EXTENSIONS:
        raise ValueError(
            "Large-image SegFormer inference supports TIFF and NPY input only. "
            "Convert PNG, JPEG, or NPZ input to an uncompressed TIFF or NPY first."
        )
    try:
        import tifffile
    except ImportError as exc:
        raise ImportError("Reading TIFF input requires tifffile") from exc

    with tifffile.TiffFile(image_path) as tiff:
        if len(tiff.series) != 1:
            raise ValueError("TIFF contains multiple series; select or split one RGB series first")
        levels = tuple(tiff.series[0].levels)
        if len(levels) != 1:
            raise ValueError("TIFF contains multiple pyramid levels; select or split one RGB level first")
        series = levels[0]
        axes = str(series.axes)
        shape = tuple(int(value) for value in series.shape)

    if axes.count("Y") != 1 or axes.count("X") != 1:
        raise ValueError(f"TIFF must contain one Y and one X axis, got axes {axes!r}")
    channel_axis = "C" if axes.count("C") == 1 else "S" if axes.count("S") == 1 else None
    if channel_axis is None:
        raise ValueError(f"TIFF must contain an RGB channel axis, got axes {axes!r}")
    for index, axis in enumerate(axes):
        if axis not in {"Y", "X", channel_axis} and shape[index] != 1:
            raise ValueError(f"TIFF has unsupported non-singleton axis {axis!r} in {axes!r}")
    height = shape[axes.index("Y")]
    width = shape[axes.index("X")]
    channels = shape[axes.index(channel_axis)]
    return height, width, channels


def _open_reader(
    image_path: Path,
    *,
    tile_size: int,
    batch_size: int,
    tiff_read_mode: str,
    tiff_cache_mode: str,
    tiff_cache_dir: Path,
    max_ram_cache_gib: float | str,
    max_disk_cache_gib: float | str,
    runtime_reserve_gib: float | str,
    tiff_block_cache_mib: float | str,
    output_allocations: list[DiskAllocation],
    channel_mapping: str | None,
    input_sidecar: str | Path | None,
    filesystem_provider=None,
):
    suffix = image_path.suffix.lower()
    if suffix in NPY_EXTENSIONS:
        return _NpyReader(image_path)
    if suffix not in TIFF_EXTENSIONS:
        raise ValueError(
            "Large-image SegFormer inference supports TIFF and NPY input only. "
            "Convert PNG, JPEG, or NPZ input to an uncompressed TIFF or NPY first."
        )

    budget = ReaderBudget.from_cli(
        max_ram_cache_gib=max_ram_cache_gib,
        max_disk_cache_gib=max_disk_cache_gib,
        runtime_reserve_gib=runtime_reserve_gib,
        tiff_block_cache_mib=tiff_block_cache_mib,
    )
    return TiffReader(
        image_path,
        legacy_input_spec(MODEL_CHANNELS, tile_size),
        read_mode=tiff_read_mode,
        cache_mode=tiff_cache_mode,
        budget=budget,
        cache_dir=tiff_cache_dir,
        patch_size=tile_size,
        batch_size=batch_size,
        channel_mapping=channel_mapping,
        input_sidecar=input_sidecar,
        filesystem_provider=filesystem_provider,
        disk_allocations=lambda _reader: output_allocations,
    )


def infer_large_image_streaming(
    image_path: str | Path,
    *,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    output_mask: str | Path | None = None,
    output_probability: str | Path | None = None,
    tile_size: int = DEFAULT_TILE_SIZE,
    stride: int | None = None,
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
    progress_every: int = 100,
    _filesystem_provider=None,
) -> dict[str, object]:
    """Run SegFormer in tiles while staging full-resolution outputs on disk.

    ``stream`` only accepts memory-mappable TIFFs, preventing an accidental
    full-image decode. Use ``auto`` when a compressed TIFF may use a bounded
    repository cache, after setting its cache and RAM budgets explicitly.
    Overlapping windows are blended in probability space before thresholding.
    """
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    stride = _resolve_stride(tile_size, stride)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if progress_every <= 0:
        raise ValueError("progress_every must be positive")

    image_path = Path(image_path)
    checkpoint_path = Path(checkpoint_path)
    output_mask = (
        Path(output_mask)
        if output_mask is not None
        else image_path.with_name(f"{image_path.stem}_segformer_large_mask.tif")
    )
    output_probability = Path(output_probability) if output_probability is not None else None
    _validate_output_paths(image_path, output_mask, output_probability)

    height, width, channels = _probe_image_shape(image_path)
    if channels != MODEL_CHANNELS:
        raise ValueError(f"Expected 3 RGB channels, got {channels}")
    accumulator = _ProbabilityAccumulator(output_mask.parent, (height, width))
    mask_bytes = height * width * np.dtype(np.uint8).itemsize
    output_allocations = [
        DiskAllocation(
            output_mask,
            mask_bytes,
            "streaming SegFormer mask TIFF",
        )
    ]
    output_allocations.extend(
        (
            DiskAllocation(
                output_mask.parent / f".{output_mask.name}.probability-sum.tmp",
                height * width * np.dtype(np.float32).itemsize,
                "sliding-window probability accumulation",
            ),
            DiskAllocation(
                output_mask.parent / f".{output_mask.name}.probability-weight.tmp",
                height * width * np.dtype(np.uint32).itemsize,
                "sliding-window probability weights",
            ),
        )
    )
    if output_probability is not None:
        output_allocations.append(
            DiskAllocation(
                output_probability,
                height * width * np.dtype(np.float32).itemsize,
                "streaming SegFormer probability TIFF",
            )
        )
    require_writable_parents(output_allocations)
    require_disk_allocations(output_allocations, provider=_filesystem_provider)

    resolved_device = _resolve_device(device)
    model = load_segformer(checkpoint_path, resolved_device)
    cache_dir = Path(tiff_cache_dir) if tiff_cache_dir is not None else output_mask.parent / ".cube_nano-cache"

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
        if reader_shape != (height, width, channels):
            raise ValueError(
                f"Input metadata changed while opening {image_path}: "
                f"expected {(height, width, channels)}, got {reader_shape}"
            )
        scale = _normalization_scale(np.empty((), dtype=reader.dtype), input_scale)
        mask_stage = _StagedTiffOutput(output_mask, (height, width), np.uint8)
        probability_stage = (
            _StagedTiffOutput(output_probability, (height, width), np.float32)
            if output_probability is not None
            else None
        )
        try:
            mask = mask_stage.open()
            probability_map = probability_stage.open() if probability_stage is not None else None
        except Exception:
            mask_stage.abort()
            if probability_stage is not None:
                probability_stage.abort()
            raise
        row_starts = _window_starts(height, tile_size, stride)
        column_starts = _window_starts(width, tile_size, stride)
        total_tiles = len(row_starts) * len(column_starts)
        processed_tiles = 0
        cloud_pixel_count = 0
        pending_patches: list[np.ndarray] = []
        pending_coords: list[tuple[int, int, int, int]] = []
        started = time.perf_counter()
        try:
            accumulator.open()
        except Exception:
            mask_stage.abort()
            if probability_stage is not None:
                probability_stage.abort()
            raise

        def flush_batch() -> None:
            nonlocal processed_tiles
            if not pending_patches:
                return
            batch = np.stack(pending_patches, axis=0)
            probabilities = _predict_probability(model, batch, resolved_device, tile_size)
            for probability, (row_start, row_end, column_start, column_end) in zip(
                probabilities, pending_coords
            ):
                tile_height = row_end - row_start
                tile_width = column_end - column_start
                probability = probability[:tile_height, :tile_width]
                accumulator.add(
                    probability,
                    row_start,
                    row_end,
                    column_start,
                    column_end,
                )
                processed_tiles += 1
            pending_patches.clear()
            pending_coords.clear()
            if processed_tiles % progress_every == 0 or processed_tiles == total_tiles:
                print(f"Processed {processed_tiles}/{total_tiles} tiles", flush=True)

        try:
            for row_start in row_starts:
                row_end = min(row_start + tile_size, height)
                strip = reader.read_rows(row_start, row_end)
                for column_start in column_starts:
                    column_end = min(column_start + tile_size, width)
                    patch = np.zeros((tile_size, tile_size, MODEL_CHANNELS), dtype=reader.dtype)
                    patch[: row_end - row_start, : column_end - column_start] = strip[
                        :,
                        column_start:column_end,
                        :,
                    ]
                    patch = _normalize_patch(patch, scale)
                    pending_patches.append(np.transpose(patch, (2, 0, 1)))
                    pending_coords.append((row_start, row_end, column_start, column_end))
                    if len(pending_patches) >= batch_size:
                        flush_batch()
            flush_batch()

            if accumulator.probability_sum is None or accumulator.weight is None:
                raise RuntimeError("Probability accumulator is not available after inference")
            for row_start in range(0, height, tile_size):
                row_end = min(row_start + tile_size, height)
                for column_start in range(0, width, tile_size):
                    column_end = min(column_start + tile_size, width)
                    probability_sum = accumulator.probability_sum[
                        row_start:row_end,
                        column_start:column_end,
                    ]
                    weight = accumulator.weight[row_start:row_end, column_start:column_end]
                    if np.any(weight == 0):
                        raise RuntimeError("Sliding-window inference left pixels without probability coverage")
                    probability = np.empty(probability_sum.shape, dtype=np.float32)
                    np.divide(probability_sum, weight, out=probability)
                    is_cloud = probability >= threshold
                    mask[row_start:row_end, column_start:column_end] = np.where(
                        is_cloud,
                        MASK_CLOUD_VALUE,
                        0,
                    ).astype(np.uint8)
                    if probability_map is not None:
                        probability_map[row_start:row_end, column_start:column_end] = probability
                    cloud_pixel_count += int(np.count_nonzero(is_cloud))
            accumulator.close()
            mask_stage.commit()
            if probability_stage is not None:
                probability_stage.commit()
        except Exception:
            mask_stage.abort()
            if probability_stage is not None:
                probability_stage.abort()
            raise
        finally:
            accumulator.cleanup()

        elapsed = time.perf_counter() - started
        result = {
            "image": str(image_path),
            "checkpoint": str(checkpoint_path),
            "mask": str(output_mask),
            "probability": str(output_probability) if output_probability else None,
            "device": str(resolved_device),
            "image_shape": [height, width, channels],
            "tile_size": tile_size,
            "stride": stride,
            "overlap": tile_size - stride,
            "batch_size": batch_size,
            "tile_count": total_tiles,
            "threshold": threshold,
            "input_scale": scale,
            "cloud_pixel_count": cloud_pixel_count,
            "cloud_ratio": float(cloud_pixel_count / (height * width)),
            "mask_bigtiff": mask_stage.bigtiff,
            "probability_bigtiff": probability_stage.bigtiff if probability_stage else None,
            "reader_backend": getattr(reader, "backend", "unknown"),
            "reader_metrics": (
                reader.metrics.as_dict() if hasattr(reader, "metrics") else None
            ),
            "elapsed_seconds": round(elapsed, 3),
        }
        return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run memory-bounded SegFormer-B0 inference on a large RGB TIFF or NPY image"
    )
    parser.add_argument("--image", required=True, help="Input RGB TIFF or NPY image")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output-mask", default=None, help="Output uint8 TIFF or BigTIFF mask")
    parser.add_argument("--output-probability", default=None, help="Optional float32 TIFF or BigTIFF probability map")
    parser.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE)
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Sliding-window stride; defaults to half the tile size for 50%% overlap",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--input-scale", type=float, default=None)
    parser.add_argument("--progress-every", type=int, default=100)
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
    result = infer_large_image_streaming(
        args.image,
        checkpoint_path=args.checkpoint,
        output_mask=args.output_mask,
        output_probability=args.output_probability,
        tile_size=args.tile_size,
        stride=args.stride,
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
