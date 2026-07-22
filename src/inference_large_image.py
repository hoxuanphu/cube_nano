"""Run sliding-window inference on large images with a PyTorch checkpoint."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from inference import load_model

LEGACY_DEV_ONLY = True
from inference_large_image_trt import (
    SUPPORTED_IMAGE_EXTENSIONS,
    process_large_image as _process_large_image_backend,
)


def resolve_device(device: str | torch.device) -> torch.device:
    """Resolve an explicit or automatic PyTorch device selection."""
    if isinstance(device, torch.device):
        resolved = device
    elif device == "auto":
        resolved = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        resolved = torch.device(device)

    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return resolved


class CloudTorchInfer:
    """Batch inference adapter matching the large-image backend contract."""

    def __init__(
        self,
        checkpoint_path,
        *,
        channels=3,
        patch_size=256,
        threshold=0.5,
        input_spec=None,
        device="auto",
        allow_untrained=False,
    ):
        if channels not in (3, 4):
            raise ValueError("channels must be 3 or 4")
        if patch_size <= 0:
            raise ValueError("patch_size must be greater than zero")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        if input_spec is not None:
            if input_spec.channels != channels:
                raise ValueError(
                    f"Input specification has {input_spec.channels} channels, expected {channels}"
                )
            if input_spec.patch_size != patch_size:
                raise ValueError(
                    f"Input specification patch size is {input_spec.patch_size}, "
                    f"expected {patch_size}"
                )

        self.channels = channels
        self.patch_size = patch_size
        self.threshold = threshold
        self.device = resolve_device(device)
        self.model = load_model(
            checkpoint_path,
            channels=channels,
            device=self.device,
            allow_untrained=allow_untrained,
        )
        self.model.to(self.device)
        self.model.eval()

    def _prepare_batch(self, batch) -> torch.Tensor:
        batch = np.asarray(batch)
        expected_tail = (self.channels, self.patch_size, self.patch_size)
        if batch.ndim != 4 or tuple(batch.shape[1:]) != expected_tail:
            raise ValueError(
                f"Expected batch shape (N, {self.channels}, {self.patch_size}, "
                f"{self.patch_size}), got {batch.shape}"
            )
        if not np.issubdtype(batch.dtype, np.number):
            raise ValueError(f"Unsupported batch dtype {batch.dtype}")
        batch = np.ascontiguousarray(batch, dtype=np.float32)
        if not np.all(np.isfinite(batch)):
            raise ValueError("Inference batch contains NaN or infinite values")
        return torch.from_numpy(batch).to(self.device)

    def infer_batch(self, batch):
        tensor = self._prepare_batch(batch)
        with torch.inference_mode():
            logits = self.model(tensor).reshape(-1)
            if logits.numel() != tensor.shape[0]:
                raise ValueError(
                    f"Model returned {logits.numel()} logits for batch size {tensor.shape[0]}"
                )
            probabilities_tensor = torch.sigmoid(logits)

        probabilities = (
            probabilities_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        )
        predictions = probabilities >= self.threshold
        return predictions, probabilities

    def infer(self, batch):
        predictions, probabilities = self.infer_batch(batch)
        if len(predictions) != 1:
            raise ValueError("infer expects a batch containing exactly one patch")
        return bool(predictions[0]), float(probabilities[0])


def process_large_image(
    large_image_path,
    model_path="checkpoints/best_model.pth",
    out_mask="cloud_mask_output.tif",
    patch_size=256,
    channels=3,
    batch_size=1,
    array_key=None,
    threshold=0.5,
    device="auto",
    allow_untrained=False,
    mask_cache=None,
    cloud_coverage_threshold=0.60,
    discard_cloudy=False,
    tiff_read_mode="auto",
    tiff_cache_mode="auto",
    max_ram_cache_gib=0.5,
    max_disk_cache_gib=8.0,
    runtime_reserve_gib=1.5,
    tiff_block_cache_mib=64,
    tiff_cache_dir=None,
    tiff_series=None,
    tiff_level=None,
    channel_mapping=None,
    input_sidecar=None,
    _memory_provider=None,
    _filesystem_provider=None,
    _torch_infer_factory=None,
):
    """Classify every patch in a large image and write a full-size cloud mask."""
    resolved_device = resolve_device(device)
    torch_infer_factory = _torch_infer_factory or CloudTorchInfer

    def backend_factory(checkpoint_path, **kwargs):
        return torch_infer_factory(
            checkpoint_path,
            device=resolved_device,
            allow_untrained=allow_untrained,
            **kwargs,
        )

    result = _process_large_image_backend(
        large_image_path,
        model_path,
        out_mask=out_mask,
        patch_size=patch_size,
        channels=channels,
        batch_size=batch_size,
        array_key=array_key,
        threshold=threshold,
        mask_cache=mask_cache,
        cloud_coverage_threshold=cloud_coverage_threshold,
        discard_cloudy=discard_cloudy,
        tiff_read_mode=tiff_read_mode,
        tiff_cache_mode=tiff_cache_mode,
        max_ram_cache_gib=max_ram_cache_gib,
        max_disk_cache_gib=max_disk_cache_gib,
        runtime_reserve_gib=runtime_reserve_gib,
        tiff_block_cache_mib=tiff_block_cache_mib,
        tiff_cache_dir=tiff_cache_dir,
        tiff_series=tiff_series,
        tiff_level=tiff_level,
        channel_mapping=channel_mapping,
        input_sidecar=input_sidecar,
        _memory_provider=_memory_provider,
        _filesystem_provider=_filesystem_provider,
        _trt_infer_factory=backend_factory,
        _backend_name="PyTorch",
    )
    result.update(
        {
            "model_path": str(model_path),
            "device": str(resolved_device),
            "checkpoint_loaded": os.path.isfile(model_path),
        }
    )
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run PyTorch cloud inference on a large multi-channel image."
    )
    parser.add_argument(
        "--image",
        required=True,
        help=(
            "Path to a 3- or 4-channel image. Supported extensions: "
            + ", ".join(SUPPORTED_IMAGE_EXTENSIONS)
        ),
    )
    parser.add_argument(
        "--model-path",
        default="checkpoints/best_model.pth",
        help="Path to a PyTorch checkpoint.",
    )
    parser.add_argument(
        "--out-mask",
        "--out_mask",
        dest="out_mask",
        default="cloud_mask_output.tif",
        help="Output cloud-mask TIFF path.",
    )
    parser.add_argument("--patch-size", "--patch_size", dest="patch_size", type=int, default=256)
    parser.add_argument("--channels", type=int, choices=(3, 4), default=3)
    parser.add_argument("--legacy", action="store_true", help="Explicitly enable the legacy 4-channel development path")
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--allow-untrained", action="store_true")
    parser.add_argument("--array-key", "--array_key", dest="array_key", default=None)
    parser.add_argument("--mask-cache", "--mask_cache", dest="mask_cache", default=None)
    parser.add_argument(
        "--cloud-coverage-threshold",
        "--cloud_coverage_threshold",
        dest="cloud_coverage_threshold",
        type=float,
        default=0.60,
    )
    parser.add_argument("--discard-cloudy", action="store_true")
    parser.add_argument(
        "--tiff-read-mode",
        "--tiff_read_mode",
        dest="tiff_read_mode",
        choices=("auto", "stream", "full"),
        default="auto",
    )
    parser.add_argument(
        "--tiff-cache-mode",
        "--tiff_cache_mode",
        dest="tiff_cache_mode",
        choices=("auto", "ram", "disk"),
        default="auto",
    )
    parser.add_argument(
        "--max-ram-cache-gib",
        "--max_ram_cache_gib",
        dest="max_ram_cache_gib",
    )
    parser.add_argument(
        "--max-disk-cache-gib",
        "--max_disk_cache_gib",
        dest="max_disk_cache_gib",
    )
    parser.add_argument("--runtime-reserve-gib", "--runtime_reserve_gib", dest="runtime_reserve_gib", default="1.5")
    parser.add_argument("--tiff-block-cache-mib", "--tiff_block_cache_mib", dest="tiff_block_cache_mib", default="64")
    parser.add_argument("--tiff-cache-dir", "--tiff_cache_dir", dest="tiff_cache_dir")
    parser.add_argument("--tiff-series", "--tiff_series", dest="tiff_series", type=int)
    parser.add_argument("--tiff-level", "--tiff_level", dest="tiff_level", type=int)
    parser.add_argument("--channel-mapping", "--channel_mapping", dest="channel_mapping")
    parser.add_argument("--input-sidecar", "--input_sidecar", dest="input_sidecar")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.channels == 4 and not args.legacy:
        raise RuntimeError("legacy 4-channel inference requires --legacy")
    if args.tiff_read_mode == "stream":
        if args.tiff_cache_mode != "auto":
            parser.error("--tiff-read-mode=stream requires --tiff-cache-mode=auto")
        if args.max_ram_cache_gib is not None or args.max_disk_cache_gib is not None:
            parser.error(
                "stream mode does not accept explicit decoded-cache size options"
            )

    result = process_large_image(
        args.image,
        model_path=args.model_path,
        out_mask=args.out_mask,
        patch_size=args.patch_size,
        channels=args.channels,
        batch_size=args.batch_size,
        array_key=args.array_key,
        threshold=args.threshold,
        device=args.device,
        allow_untrained=args.allow_untrained,
        mask_cache=args.mask_cache,
        cloud_coverage_threshold=args.cloud_coverage_threshold,
        discard_cloudy=args.discard_cloudy,
        tiff_read_mode=args.tiff_read_mode,
        tiff_cache_mode=args.tiff_cache_mode,
        max_ram_cache_gib=(
            "0.5" if args.max_ram_cache_gib is None else args.max_ram_cache_gib
        ),
        max_disk_cache_gib=(
            "8.0" if args.max_disk_cache_gib is None else args.max_disk_cache_gib
        ),
        runtime_reserve_gib=args.runtime_reserve_gib,
        tiff_block_cache_mib=args.tiff_block_cache_mib,
        tiff_cache_dir=args.tiff_cache_dir,
        tiff_series=args.tiff_series,
        tiff_level=args.tiff_level,
        channel_mapping=args.channel_mapping,
        input_sidecar=args.input_sidecar,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
