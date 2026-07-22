"""Reproducible SegFormer-B0 training entry point."""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

# Allow both ``python src/train_segmentation.py`` and ``python -m src.train_segmentation``.
if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[1]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

try:
    from data.segmentation_dataset import SegmentationDataset
    from losses import masked_segmentation_loss
    from models.segformer_b0 import SEGFORMER_IMPLEMENTATION_ID, get_segformer_b0
except ModuleNotFoundError:  # Package invocation: python -m src.train_segmentation
    from src.data.segmentation_dataset import SegmentationDataset
    from src.losses import masked_segmentation_loss
    from src.models.segformer_b0 import SEGFORMER_IMPLEMENTATION_ID, get_segformer_b0


@dataclass(frozen=True)
class SegmentationTrainingConfig:
    epochs: int = 50
    learning_rate: float = 6e-5
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    early_stopping_patience: int = 12
    cross_entropy_weight: float = 1.0
    dice_weight: float = 1.0
    dice_epsilon: float = 1e-6
    ignore_index: int = 255
    seed: int = 42
    use_amp: bool = True
    batch_size: int = 1
    preserve_native_size: bool = True

    def validate(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0 or self.warmup_epochs < 0:
            raise ValueError("training epoch and batch settings are invalid")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("training optimizer settings are invalid")
        if self.ignore_index != 255:
            raise ValueError("ignore_index is pinned to 255")
        if self.preserve_native_size and self.batch_size != 1:
            raise ValueError("native-size SegFormer training requires batch_size=1")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _batch_valid_count(batch: dict[str, torch.Tensor], ignore_index: int) -> int:
    validity = batch["validity_mask"].to(torch.bool)
    target = batch["mask"]
    return int(torch.count_nonzero(validity & (target != ignore_index)).item())


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: SegmentationTrainingConfig,
    *,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> dict[str, float | int]:
    model.train()
    total_loss = 0.0
    total_valid = 0
    optimizer_steps = 0
    skipped_invalid = 0
    amp_enabled = bool(scaler is not None and scaler.is_enabled())
    for batch in loader:
        valid_count = _batch_valid_count(batch, config.ignore_index)
        if valid_count == 0:
            skipped_invalid += 1
            continue
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["mask"].to(device, non_blocking=True)
        validity = batch["validity_mask"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)
            loss, _ = masked_segmentation_loss(
                logits,
                targets,
                validity_mask=validity,
                ignore_index=config.ignore_index,
                cross_entropy_weight=config.cross_entropy_weight,
                dice_weight=config.dice_weight,
                epsilon=config.dice_epsilon,
            )
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite segmentation loss")
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.detach().cpu()) * valid_count
        total_valid += valid_count
        optimizer_steps += 1
    return {
        "loss": total_loss / max(total_valid, 1),
        "valid_pixels": total_valid,
        "optimizer_steps": optimizer_steps,
        "skipped_all_invalid_batches": skipped_invalid,
    }


@torch.no_grad()
def evaluate_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: SegmentationTrainingConfig,
) -> dict[str, float | int]:
    model.eval()
    total_loss = 0.0
    total_valid = 0
    skipped_invalid = 0
    for batch in loader:
        valid_count = _batch_valid_count(batch, config.ignore_index)
        if valid_count == 0:
            skipped_invalid += 1
            continue
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["mask"].to(device, non_blocking=True)
        validity = batch["validity_mask"].to(device, non_blocking=True)
        logits = model(images)
        loss, _ = masked_segmentation_loss(
            logits,
            targets,
            validity_mask=validity,
            ignore_index=config.ignore_index,
            cross_entropy_weight=config.cross_entropy_weight,
            dice_weight=config.dice_weight,
            epsilon=config.dice_epsilon,
        )
        total_loss += float(loss.cpu()) * valid_count
        total_valid += valid_count
    return {
        "loss": total_loss / max(total_valid, 1),
        "valid_pixels": total_valid,
        "skipped_all_invalid_batches": skipped_invalid,
    }


def build_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    scaler: torch.cuda.amp.GradScaler | None,
    epoch: int,
    global_step: int,
    best_metric: float,
    config: SegmentationTrainingConfig,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": None if scaler is None else scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_metric": best_metric,
        "training_config": asdict(config),
        "implementation_id": SEGFORMER_IMPLEMENTATION_ID,
        "metadata": dict(metadata),
    }


def train(
    train_dir: str | Path,
    validation_dir: str | Path,
    output_path: str | Path,
    *,
    config: SegmentationTrainingConfig | None = None,
    device: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = config or SegmentationTrainingConfig()
    config.validate()
    set_seed(config.seed)
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_dataset = SegmentationDataset(
        train_dir,
        is_train=True,
        preserve_native_size=config.preserve_native_size,
    )
    validation_dataset = SegmentationDataset(
        validation_dir,
        is_train=False,
        preserve_native_size=config.preserve_native_size,
    )
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=0)
    model = get_segformer_b0().to(selected_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(config.epochs - config.warmup_epochs, 1),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp and selected_device.type == "cuda")
    best_loss = float("inf")
    best_state: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    global_step = 0
    stale_epochs = 0
    for epoch in range(config.epochs):
        train_metrics = train_one_epoch(model, train_loader, optimizer, selected_device, config, scaler=scaler)
        validation_metrics = evaluate_loss(model, validation_loader, selected_device, config)
        if epoch >= config.warmup_epochs:
            scheduler.step()
        global_step += int(train_metrics["optimizer_steps"])
        record = {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}
        history.append(record)
        val_loss = float(validation_metrics["loss"])
        if int(validation_metrics["valid_pixels"]) > 0 and val_loss < best_loss:
            best_loss = val_loss
            stale_epochs = 0
            best_state = build_checkpoint(
                model,
                optimizer,
                scheduler,
                scaler=scaler,
                epoch=epoch,
                global_step=global_step,
                best_metric=-best_loss,
                config=config,
                metadata=metadata or {},
            )
        else:
            stale_epochs += 1
            if stale_epochs >= config.early_stopping_patience:
                break
    if best_state is None:
        raise RuntimeError("validation produced no valid pixels; no checkpoint was created")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, output)
    report = {
        "implementation_id": SEGFORMER_IMPLEMENTATION_ID,
        "checkpoint_path": str(output),
        "best_validation_loss": best_loss,
        "history": history,
        "metadata": metadata or {},
        "training_config": asdict(config),
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the SegFormer-B0 cloud segmenter")
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--validation-dir", required=True)
    parser.add_argument("--output", default="checkpoints/segformer_b0_rgb_r1.pth")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=6e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--tile-training",
        action="store_true",
        help="Use legacy 256x256 training tiles instead of native source dimensions",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    report = train(
        args.train_dir,
        args.validation_dir,
        args.output,
        config=SegmentationTrainingConfig(
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            seed=args.seed,
            preserve_native_size=not args.tile_training,
        ),
        device=args.device,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
