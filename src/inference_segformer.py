"""Standalone large-image inference for the local SegFormer-B0 checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from torch.nn import functional as F


if __package__ in {None, ""}:
    _repository_root = Path(__file__).resolve().parents[1]
    if str(_repository_root) not in sys.path:
        sys.path.insert(0, str(_repository_root))

try:
    from models.segformer_b0 import get_segformer_b0
except ModuleNotFoundError:
    from src.models.segformer_b0 import get_segformer_b0


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = _REPOSITORY_ROOT / "checkpoints" / "segformer_b0_rgb_research_baseline.pth"
DEFAULT_TILE_SIZE = 384
MODEL_CHANNELS = 3
MASK_CLOUD_VALUE = 255


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return resolved


def _load_state_dict(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    # PyTorch 2.6+ defaults to weights_only=True. Some training metadata
    # contains pathlib.PosixPath, which cannot be instantiated on Windows.
    # It is metadata only, so deserialize it as the platform-neutral equivalent
    # without weakening the restricted unpickler for arbitrary classes.
    with torch.serialization.safe_globals(
        [(PurePosixPath, "pathlib.PosixPath")]
    ):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict):
        checkpoint = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint must contain a state dictionary: {checkpoint_path}")
    if checkpoint and all(str(key).startswith("module.") for key in checkpoint):
        checkpoint = {
            str(key).removeprefix("module."): value
            for key, value in checkpoint.items()
        }
    return checkpoint


def load_segformer(checkpoint_path: str | Path, device: torch.device) -> torch.nn.Module:
    """Create the local SegFormer-B0 and load a training checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"SegFormer checkpoint not found: {checkpoint_path}")
    model = get_segformer_b0().to(device)
    model.load_state_dict(_load_state_dict(checkpoint_path))
    model.eval()
    return model


def _select_array(mapping: np.lib.npyio.NpzFile, array_key: str | None, path: Path) -> np.ndarray:
    if array_key:
        if array_key not in mapping:
            available = ", ".join(mapping.files)
            raise KeyError(f"Array key '{array_key}' not found in {path}; available: {available}")
        return np.asarray(mapping[array_key])
    if len(mapping.files) == 1:
        return np.asarray(mapping[mapping.files[0]])
    for preferred in ("image", "arr_0", "data"):
        if preferred in mapping.files:
            return np.asarray(mapping[preferred])
    available = ", ".join(mapping.files)
    raise ValueError(f"{path} contains multiple arrays; use --array-key. Available: {available}")


def _canonicalize_rgb(array: np.ndarray, path: Path) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim != 3:
        raise ValueError(f"Expected an RGB image with shape (H, W, 3) or (3, H, W), got {array.shape} from {path}")
    if array.shape[-1] == MODEL_CHANNELS:
        return array
    if array.shape[0] == MODEL_CHANNELS and array.shape[-1] != MODEL_CHANNELS:
        return np.moveaxis(array, 0, -1)
    raise ValueError(f"SegFormer expects exactly 3 RGB channels, got {array.shape} from {path}")


def load_image(path: str | Path, array_key: str | None = None) -> np.ndarray:
    """Load an image as HWC RGB, using a memmap when an uncompressed TIFF allows it."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        try:
            import tifffile
        except ImportError as exc:
            raise ImportError("Reading TIFF input requires tifffile") from exc
        try:
            array = tifffile.memmap(path)
        except (OSError, RuntimeError, ValueError):
            array = tifffile.imread(path)
    elif suffix == ".npy":
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            array = _select_array(archive, array_key, path)
    elif suffix in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError("Reading raster input requires Pillow") from exc
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"))
    else:
        raise ValueError("Supported input formats are TIFF, PNG, JPEG, BMP, WEBP, NPY and NPZ")
    return _canonicalize_rgb(array, path)


def _normalization_scale(array: np.ndarray, input_scale: float | None) -> float:
    if input_scale is not None:
        if not np.isfinite(input_scale) or input_scale <= 0:
            raise ValueError("input_scale must be a finite positive number")
        return float(input_scale)
    if np.issubdtype(array.dtype, np.integer):
        return float(np.iinfo(array.dtype).max)
    return 1.0


def _normalize_patch(patch: np.ndarray, scale: float) -> np.ndarray:
    normalized = patch.astype(np.float32, copy=False) / np.float32(scale)
    return np.clip(normalized, 0.0, 1.0)


def _predict_probability(
    model: torch.nn.Module,
    batch: np.ndarray,
    device: torch.device,
    tile_size: int,
) -> np.ndarray:
    tensor = torch.from_numpy(np.ascontiguousarray(batch)).to(device)
    with torch.inference_mode():
        logits = model(tensor)
        if logits.ndim != 4 or logits.shape[1] != 2:
            raise ValueError(f"SegFormer returned unexpected logits shape: {tuple(logits.shape)}")
        probabilities = F.softmax(logits.float(), dim=1)[:, 1:2]
        probabilities = F.interpolate(
            probabilities,
            size=(tile_size, tile_size),
            mode="bilinear",
            align_corners=False,
        )
    return probabilities[:, 0].detach().cpu().numpy().astype(np.float32, copy=False)


def _write_tiff(path: Path, array: np.ndarray) -> None:
    try:
        import tifffile
    except ImportError as exc:
        raise ImportError("Writing TIFF output requires tifffile") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, array, metadata={"axes": "YX"}, compression=None)


def infer_large_image(
    image_path: str | Path,
    *,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    output_mask: str | Path | None = None,
    output_probability: str | Path | None = None,
    tile_size: int = DEFAULT_TILE_SIZE,
    batch_size: int = 1,
    threshold: float = 0.5,
    device: str = "auto",
    array_key: str | None = None,
    input_scale: float | None = None,
) -> dict[str, object]:
    """Run tiled SegFormer inference and write a full-size cloud mask."""
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    image_path = Path(image_path)
    checkpoint_path = Path(checkpoint_path)
    output_mask = Path(output_mask) if output_mask else image_path.with_name(f"{image_path.stem}_segformer_mask.tif")
    output_probability = Path(output_probability) if output_probability else None
    resolved_device = _resolve_device(device)
    image = load_image(image_path, array_key=array_key)
    height, width, channels = image.shape
    if channels != MODEL_CHANNELS:
        raise ValueError(f"Expected 3 RGB channels, got {channels}")
    scale = _normalization_scale(image, input_scale)
    model = load_segformer(checkpoint_path, resolved_device)

    cloud_mask = np.zeros((height, width), dtype=np.uint8)
    probability_map = (
        np.zeros((height, width), dtype=np.float32)
        if output_probability is not None
        else None
    )
    pending_patches: list[np.ndarray] = []
    pending_coords: list[tuple[int, int, int, int]] = []
    total_tiles = ((height + tile_size - 1) // tile_size) * ((width + tile_size - 1) // tile_size)
    processed_tiles = 0
    started = time.perf_counter()

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
            cloud_mask[row_start:row_end, column_start:column_end] = np.where(
                probability >= threshold,
                MASK_CLOUD_VALUE,
                0,
            ).astype(np.uint8)
            if probability_map is not None:
                probability_map[row_start:row_end, column_start:column_end] = probability
            processed_tiles += 1
        pending_patches.clear()
        pending_coords.clear()
        print(f"Processed {processed_tiles}/{total_tiles} tiles", flush=True)

    for row_start in range(0, height, tile_size):
        row_end = min(row_start + tile_size, height)
        for column_start in range(0, width, tile_size):
            column_end = min(column_start + tile_size, width)
            patch = np.zeros((tile_size, tile_size, MODEL_CHANNELS), dtype=image.dtype)
            patch[: row_end - row_start, : column_end - column_start] = image[
                row_start:row_end,
                column_start:column_end,
                :,
            ]
            patch = _normalize_patch(patch, scale)
            pending_patches.append(np.transpose(patch, (2, 0, 1)))
            pending_coords.append((row_start, row_end, column_start, column_end))
            if len(pending_patches) >= batch_size:
                flush_batch()
        flush_batch()

    _write_tiff(output_mask, cloud_mask)
    if output_probability is not None and probability_map is not None:
        _write_tiff(output_probability, probability_map)

    elapsed = time.perf_counter() - started
    result = {
        "image": str(image_path),
        "checkpoint": str(checkpoint_path),
        "mask": str(output_mask),
        "probability": str(output_probability) if output_probability else None,
        "device": str(resolved_device),
        "image_shape": [height, width, channels],
        "tile_size": tile_size,
        "batch_size": batch_size,
        "tile_count": total_tiles,
        "threshold": threshold,
        "input_scale": scale,
        "cloud_pixel_count": int(np.count_nonzero(cloud_mask == MASK_CLOUD_VALUE)),
        "cloud_ratio": float(np.count_nonzero(cloud_mask == MASK_CLOUD_VALUE) / cloud_mask.size),
        "elapsed_seconds": round(elapsed, 3),
    }
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run standalone SegFormer-B0 inference on a large RGB image")
    parser.add_argument("--image", required=True, help="Input RGB image: TIFF, PNG, JPEG, NPY or NPZ")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output-mask", default=None, help="Output uint8 cloud mask TIFF")
    parser.add_argument("--output-probability", default=None, help="Optional float32 cloud-probability TIFF")
    parser.add_argument(
        "--tile-size",
        type=int,
        default=DEFAULT_TILE_SIZE,
        help="Inference tile size; default 384 matches the native-size training run",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--array-key", default=None, help="Array key for a multi-array NPZ file")
    parser.add_argument(
        "--input-scale",
        type=float,
        default=None,
        help="Optional source scale; defaults to integer dtype max or 1.0 for float input",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = infer_large_image(
        args.image,
        checkpoint_path=args.checkpoint,
        output_mask=args.output_mask,
        output_probability=args.output_probability,
        tile_size=args.tile_size,
        batch_size=args.batch_size,
        threshold=args.threshold,
        device=args.device,
        array_key=args.array_key,
        input_scale=args.input_scale,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
