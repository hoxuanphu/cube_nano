import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from models.mobilenetv3 import get_cloud_model


def normalize_image(image, source="input"):
    """Convert a numeric image to float32 in the [0, 1] range."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = image.astype(np.float32)
        if not np.isfinite(image).all():
            raise ValueError(f"{source} contains NaN or infinite values")
        if image.size and image.max() > 1.0:
            scale = 65535.0 if image.max() > 255.0 else 255.0
            image = image / scale
        return np.clip(image, 0.0, 1.0)

    if np.issubdtype(image.dtype, np.integer):
        scale = float(np.iinfo(image.dtype).max)
        return np.clip(image.astype(np.float32) / scale, 0.0, 1.0)

    raise ValueError(f"Unsupported image dtype {image.dtype} from {source}")


def load_image(path):
    """Load an HWC or CHW image from NPY, TIFF, or a Pillow-supported file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Input image not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path, allow_pickle=False)
    if suffix in {".tif", ".tiff"}:
        import tifffile

        return tifffile.imread(path)
    return np.asarray(Image.open(path))


def prepare_input(image, channels=4, patch_size=256, source="input"):
    """Normalize, center-crop, and convert one image to a BCHW tensor."""
    image = np.asarray(image)
    if image.ndim == 2:
        image = image[:, :, None]
    if image.ndim != 3:
        raise ValueError(
            f"Expected a 2D image or a 3D HWC/CHW array, got {image.shape} from {source}"
        )

    # NPY and remote-sensing TIFF files can be stored as either HWC or CHW.
    if image.shape[-1] not in (1, 2, 3, 4) and image.shape[0] in (1, 2, 3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] not in (1, 2, 3, 4):
        raise ValueError(
            f"Cannot determine channel axis for shape {image.shape} from {source}; "
            "expected HWC or CHW with 1 to 4 channels"
        )

    height, width, input_channels = image.shape
    if height < patch_size or width < patch_size:
        raise ValueError(
            f"patch_size={patch_size} exceeds image shape ({height}, {width}) from {source}"
        )

    image = normalize_image(image, source=source)
    row = (height - patch_size) // 2
    col = (width - patch_size) // 2
    image = image[row : row + patch_size, col : col + patch_size]

    if input_channels < channels:
        padding = np.zeros(
            (patch_size, patch_size, channels - input_channels), dtype=np.float32
        )
        image = np.concatenate([image, padding], axis=2)
    elif input_channels > channels:
        image = image[:, :, :channels]

    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
    return tensor.unsqueeze(0)


def load_model(model_path, channels, device, allow_untrained=False):
    model = get_cloud_model(pretrained=False, num_channels=channels)
    if os.path.isfile(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        if not isinstance(checkpoint, dict):
            raise ValueError(f"Unsupported checkpoint format in {model_path}")

        if checkpoint and all(key.startswith("module.") for key in checkpoint):
            checkpoint = {key.removeprefix("module."): value for key, value in checkpoint.items()}
        model.load_state_dict(checkpoint)
    elif not allow_untrained:
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}. "
            "Use --allow-untrained only for a pipeline smoke test."
        )

    model.to(device)
    model.eval()
    return model


def predict(model, input_tensor, device, threshold=0.5):
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be between 0 and 1, got {threshold}")

    with torch.inference_mode():
        logit = model(input_tensor.to(device)).reshape(-1)[0]
        probability = torch.sigmoid(logit).item()

    return {
        "class": "cloud" if probability >= threshold else "clear",
        "is_cloud": probability >= threshold,
        "probability": probability,
        "threshold": threshold,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one cloud/clear prediction with a PyTorch checkpoint."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", help="Path to a .npy, TIFF, or image file")
    input_group.add_argument(
        "--random-input",
        action="store_true",
        help="Use a random patch for a pipeline smoke test",
    )
    parser.add_argument(
        "--model-path", default="checkpoints/best_model.pth", help="PyTorch checkpoint"
    )
    parser.add_argument("--channels", type=int, default=4, choices=(3, 4))
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument(
        "--allow-untrained",
        action="store_true",
        help="Allow a random-weight model when the checkpoint does not exist",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.patch_size <= 0:
        raise ValueError(f"patch_size must be greater than zero, got {args.patch_size}")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device_name = (
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)

    if args.random_input:
        rng = np.random.default_rng(42)
        image = rng.random(
            (args.patch_size, args.patch_size, args.channels), dtype=np.float32
        )
        source = "random input"
    else:
        image = load_image(args.input)
        source = args.input

    input_tensor = prepare_input(
        image,
        channels=args.channels,
        patch_size=args.patch_size,
        source=source,
    )
    model = load_model(
        args.model_path,
        channels=args.channels,
        device=device,
        allow_untrained=args.allow_untrained,
    )
    result = predict(model, input_tensor, device=device, threshold=args.threshold)
    result.update(
        {
            "input": source,
            "model_path": args.model_path,
            "device": str(device),
            "checkpoint_loaded": os.path.isfile(args.model_path),
        }
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
