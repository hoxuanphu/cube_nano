"""Production model loader bound to the released satellite model package."""

from __future__ import annotations

import os
from pathlib import Path

import torch

from src.models.mobilenetv3 import get_cloud_model
from src.models.segformer_b0 import get_segformer_b0


def _model_for_task(model_task: str, channels: int):
    if model_task == "patch_classification":
        return get_cloud_model(pretrained=False, num_channels=channels)
    if model_task == "semantic_cloud_segmentation":
        return get_segformer_b0(pretrained=False, num_classes=2, num_channels=channels)
    raise ValueError(f"unsupported model task: {model_task}")


def load_model(
    model_path: str | Path,
    channels: int,
    device: torch.device,
    *,
    allow_untrained: bool = False,
    model_task: str = "patch_classification",
):
    if channels != 3:
        raise ValueError("production model runtime accepts exactly 3 RGB channels")
    model = _model_for_task(model_task, channels)
    path = Path(model_path)
    if path.is_file():
        checkpoint = torch.load(path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        if not isinstance(checkpoint, dict):
            raise ValueError(f"Unsupported checkpoint format in {path}")
        if checkpoint and all(key.startswith("module.") for key in checkpoint):
            checkpoint = {key.removeprefix("module."): value for key, value in checkpoint.items()}
        model.load_state_dict(checkpoint)
    elif not allow_untrained:
        raise FileNotFoundError(f"Model checkpoint not found: {path}")
    model.to(device)
    model.eval()
    return model

