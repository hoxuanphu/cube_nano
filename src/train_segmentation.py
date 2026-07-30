"""Reproducible SegFormer-B0 training entry point."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from functools import partial
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
    from data.segmentation_dataset import SegmentationDataset, collate_segmentation_batch
    from losses import masked_segmentation_loss
    from models.segformer_b0 import (
        SEGFORMER_IMPLEMENTATION_ID,
        get_segformer_b0,
        load_segformer_mit_b0_encoder,
    )
except ModuleNotFoundError:  # Package invocation: python -m src.train_segmentation
    from src.data.segmentation_dataset import SegmentationDataset, collate_segmentation_batch
    from src.losses import masked_segmentation_loss
    from src.models.segformer_b0 import (
        SEGFORMER_IMPLEMENTATION_ID,
        get_segformer_b0,
        load_segformer_mit_b0_encoder,
    )


@dataclass(frozen=True)
class SegmentationTrainingConfig:
    epochs: int = 50
    learning_rate: float = 6e-5
    encoder_learning_rate: float | None = None
    decoder_learning_rate: float | None = None
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    warmup_start_factor: float = 0.0
    scheduler_start_epoch: int | None = None
    lr_plateau_patience: int = 5
    lr_plateau_factor: float = 0.5
    min_learning_rate: float = 1e-7
    early_stopping_patience: int = 12
    cross_entropy_weight: float = 1.0
    dice_weight: float = 1.0
    dice_epsilon: float = 1e-6
    ignore_index: int = 255
    seed: int = 42
    use_amp: bool = True
    batch_size: int = 1
    preserve_native_size: bool = True
    pretrained_encoder_path: str | None = None
    record_epoch_metrics: bool = False
    metric_threshold_bp: int = 5000
    max_false_clear_rate: float = 0.05
    checkpoint_selection_metric: str = "validation_loss"

    def validate(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0 or self.warmup_epochs < 0:
            raise ValueError("training epoch and batch settings are invalid")
        if self.lr_plateau_patience < 0 or not 0 < self.lr_plateau_factor < 1:
            raise ValueError("learning-rate plateau settings are invalid")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("training optimizer settings are invalid")
        for name, value in (
            ("encoder_learning_rate", self.encoder_learning_rate),
            ("decoder_learning_rate", self.decoder_learning_rate),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when configured")
        if not 0.0 <= self.warmup_start_factor <= 1.0:
            raise ValueError("warmup_start_factor must be in [0, 1]")
        if self.scheduler_start_epoch is not None and self.scheduler_start_epoch < 0:
            raise ValueError("scheduler_start_epoch must be non-negative when configured")
        if self.min_learning_rate < 0 or self.min_learning_rate >= self.learning_rate:
            raise ValueError("minimum learning rate must be non-negative and below the initial rate")
        if any(
            self.min_learning_rate >= value
            for value in (self.encoder_learning_rate, self.decoder_learning_rate)
            if value is not None
        ):
            raise ValueError("minimum learning rate must be below every configured parameter-group LR")
        if not 0 <= self.metric_threshold_bp <= 10000:
            raise ValueError("metric_threshold_bp must be in [0, 10000]")
        if not 0.0 <= self.max_false_clear_rate <= 1.0:
            raise ValueError("max_false_clear_rate must be in [0, 1]")
        if self.checkpoint_selection_metric not in {
            "validation_loss",
            "validation_cloud_dice_constrained_false_clear",
        }:
            raise ValueError("unsupported checkpoint_selection_metric")
        if self.ignore_index != 255:
            raise ValueError("ignore_index is pinned to 255")


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


def _batch_padding_counts(batch: dict[str, Any]) -> tuple[int, int]:
    """Return collate-introduced padding and total batch pixels.

    Invalid source pixels are deliberately not counted as padding. The collate
    function supplies the pre-pad sample shapes when native-size batching is
    active, which keeps the two concepts separate in experiment evidence.
    """

    mask = batch["mask"]
    total_pixels = int(mask.numel())
    sample_shapes = batch.get("sample_shapes")
    if sample_shapes is None:
        return 0, total_pixels
    source_pixels = sum(int(height) * int(width) for height, width in sample_shapes)
    if source_pixels < 0 or source_pixels > total_pixels:
        raise ValueError("invalid collated sample shapes")
    return total_pixels - source_pixels, total_pixels


def _aggregate_epoch_metrics(
    *,
    total_loss: float,
    total_cross_entropy: float,
    total_soft_dice: float,
    total_valid: int,
    optimizer_steps: int,
    skipped_invalid: int,
    padding_pixels: int,
    batch_pixels: int,
    sample_count: int,
    started_at: float,
) -> dict[str, float | int]:
    elapsed_seconds = max(time.perf_counter() - started_at, 1e-9)
    return {
        "loss": total_loss / max(total_valid, 1),
        "cross_entropy": total_cross_entropy / max(total_valid, 1),
        "soft_dice": total_soft_dice / max(total_valid, 1),
        "valid_pixels": total_valid,
        "padding_pixels": padding_pixels,
        "batch_pixels": batch_pixels,
        "padding_ratio": padding_pixels / max(batch_pixels, 1),
        "optimizer_steps": optimizer_steps,
        "skipped_all_invalid_batches": skipped_invalid,
        "samples": sample_count,
        "elapsed_seconds": elapsed_seconds,
        "iterations_per_second": optimizer_steps / elapsed_seconds,
        "samples_per_second": sample_count / elapsed_seconds,
        "throughput_samples_per_second": sample_count / elapsed_seconds,
        "valid_pixels_per_second": total_valid / elapsed_seconds,
    }


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
    started_at = time.perf_counter()
    total_loss = 0.0
    total_cross_entropy = 0.0
    total_soft_dice = 0.0
    total_valid = 0
    optimizer_steps = 0
    skipped_invalid = 0
    padding_pixels = 0
    batch_pixels = 0
    sample_count = 0
    amp_enabled = bool(scaler is not None and scaler.is_enabled())
    for batch in loader:
        batch_padding_pixels, current_batch_pixels = _batch_padding_counts(batch)
        padding_pixels += batch_padding_pixels
        batch_pixels += current_batch_pixels
        sample_count += int(batch["mask"].shape[0])
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
            loss, loss_parts = masked_segmentation_loss(
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
        total_cross_entropy += float(loss_parts["cross_entropy"].detach().cpu()) * valid_count
        total_soft_dice += float(loss_parts["soft_dice"].detach().cpu()) * valid_count
        total_valid += valid_count
        optimizer_steps += 1
    return _aggregate_epoch_metrics(
        total_loss=total_loss,
        total_cross_entropy=total_cross_entropy,
        total_soft_dice=total_soft_dice,
        total_valid=total_valid,
        optimizer_steps=optimizer_steps,
        skipped_invalid=skipped_invalid,
        padding_pixels=padding_pixels,
        batch_pixels=batch_pixels,
        sample_count=sample_count,
        started_at=started_at,
    )


@torch.no_grad()
def evaluate_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: SegmentationTrainingConfig,
) -> dict[str, float | int]:
    model.eval()
    started_at = time.perf_counter()
    total_loss = 0.0
    total_cross_entropy = 0.0
    total_soft_dice = 0.0
    total_valid = 0
    skipped_invalid = 0
    padding_pixels = 0
    batch_pixels = 0
    sample_count = 0
    for batch in loader:
        batch_padding_pixels, current_batch_pixels = _batch_padding_counts(batch)
        padding_pixels += batch_padding_pixels
        batch_pixels += current_batch_pixels
        sample_count += int(batch["mask"].shape[0])
        valid_count = _batch_valid_count(batch, config.ignore_index)
        if valid_count == 0:
            skipped_invalid += 1
            continue
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["mask"].to(device, non_blocking=True)
        validity = batch["validity_mask"].to(device, non_blocking=True)
        logits = model(images)
        loss, loss_parts = masked_segmentation_loss(
            logits,
            targets,
            validity_mask=validity,
            ignore_index=config.ignore_index,
            cross_entropy_weight=config.cross_entropy_weight,
            dice_weight=config.dice_weight,
            epsilon=config.dice_epsilon,
        )
        total_loss += float(loss.cpu()) * valid_count
        total_cross_entropy += float(loss_parts["cross_entropy"].cpu()) * valid_count
        total_soft_dice += float(loss_parts["soft_dice"].cpu()) * valid_count
        total_valid += valid_count
    return _aggregate_epoch_metrics(
        total_loss=total_loss,
        total_cross_entropy=total_cross_entropy,
        total_soft_dice=total_soft_dice,
        total_valid=total_valid,
        optimizer_steps=0,
        skipped_invalid=skipped_invalid,
        padding_pixels=padding_pixels,
        batch_pixels=batch_pixels,
        sample_count=sample_count,
        started_at=started_at,
    )


def build_optimizer(
    model: torch.nn.Module,
    config: SegmentationTrainingConfig,
) -> torch.optim.AdamW:
    """Create explicit MiT encoder and segmentation-decoder parameter groups."""

    encoder_parameters: list[torch.nn.Parameter] = []
    decoder_parameters: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(("patch_embeds.", "blocks.")):
            encoder_parameters.append(parameter)
        else:
            decoder_parameters.append(parameter)
    groups: list[dict[str, Any]] = []
    if encoder_parameters:
        encoder_lr = config.encoder_learning_rate or config.learning_rate
        groups.append(
            {
                "params": encoder_parameters,
                "lr": encoder_lr,
                "base_learning_rate": encoder_lr,
                "group_name": "encoder",
            }
        )
    if decoder_parameters:
        decoder_lr = config.decoder_learning_rate or config.learning_rate
        groups.append(
            {
                "params": decoder_parameters,
                "lr": decoder_lr,
                "base_learning_rate": decoder_lr,
                "group_name": "decoder",
            }
        )
    if not groups:
        raise ValueError("model has no trainable parameters")
    return torch.optim.AdamW(groups, weight_decay=config.weight_decay)


def linear_warmup_factor(
    epoch: int,
    warmup_epochs: int,
    start_factor: float,
) -> float:
    if epoch < 0 or warmup_epochs < 0 or not 0.0 <= start_factor <= 1.0:
        raise ValueError("invalid linear warm-up settings")
    if warmup_epochs == 0 or epoch >= warmup_epochs:
        return 1.0
    progress = (epoch + 1) / warmup_epochs
    return float(start_factor + (1.0 - start_factor) * progress)


def apply_linear_warmup(
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    warmup_epochs: int,
    start_factor: float,
) -> float:
    factor = linear_warmup_factor(epoch, warmup_epochs, start_factor)
    if epoch < warmup_epochs:
        for group in optimizer.param_groups:
            group["lr"] = float(group["base_learning_rate"]) * factor
    return factor


def learning_rates_by_group(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    return {
        str(group.get("group_name", f"group_{index}")): float(group["lr"])
        for index, group in enumerate(optimizer.param_groups)
    }


def _selection_summary(
    validation_metrics: dict[str, Any],
    config: SegmentationTrainingConfig,
) -> tuple[float, dict[str, Any]]:
    if config.checkpoint_selection_metric == "validation_loss":
        loss = float(validation_metrics["loss"])
        return -loss, {
            "metric": "validation_loss",
            "value": loss,
            "constraint_satisfied": True,
        }
    segmentation = validation_metrics.get("segmentation")
    if not isinstance(segmentation, dict):
        raise RuntimeError("segmentation metrics are required for Dice-based checkpoint selection")
    dice = float(segmentation["cloud_dice"])
    false_clear = float(segmentation["false_clear_rate"])
    constraint_satisfied = false_clear <= config.max_false_clear_rate
    # Any constraint-satisfying Dice beats a violating epoch. Among violations,
    # retain the lowest false-clear rate so the run remains diagnosable.
    score = dice if constraint_satisfied else -false_clear
    return score, {
        "metric": "validation_cloud_dice_constrained_false_clear",
        "value": dice,
        "false_clear_rate": false_clear,
        "max_false_clear_rate": config.max_false_clear_rate,
        "constraint_satisfied": constraint_satisfied,
    }


def _epoch_segmentation_metrics(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    threshold_bp: int,
    bootstrap_seed: int,
) -> dict[str, float | int]:
    """Collect fixed-threshold validation metrics without using the test split."""

    try:
        from eval_segmentation import evaluate_loader
    except ModuleNotFoundError:
        from src.eval_segmentation import evaluate_loader

    report = evaluate_loader(
        model,
        loader,
        device=device,
        threshold_bp=threshold_bp,
        # Per-epoch bootstrap intervals do not affect checkpoint selection and
        # would inflate every training report. Final P0/P1 reports use 1000.
        bootstrap_samples=1,
        bootstrap_seed=bootstrap_seed,
    )
    metrics = report["metrics"]
    macro = report["macro_scene_metrics"]
    coverage = report["coverage_metrics"]
    return {
        "threshold_bp": int(threshold_bp),
        "cloud_iou": float(metrics["cloud_iou"]),
        "cloud_dice": float(metrics["cloud_dice"]),
        "cloud_precision": float(metrics["cloud_precision"]),
        "cloud_recall": float(metrics["cloud_recall"]),
        "false_clear_rate": float(metrics["false_clear_rate"]),
        "boundary_f1": float(metrics["boundary_f1"]),
        "macro_cloud_iou": float(macro["macro_cloud_iou"]),
        "macro_cloud_dice": float(macro["macro_cloud_dice"]),
        "macro_cloud_precision": float(macro["macro_cloud_precision"]),
        "macro_cloud_recall": float(macro["macro_cloud_recall"]),
        "macro_false_clear_rate": float(macro["macro_false_clear_rate"]),
        "macro_boundary_f1": float(macro["macro_boundary_f1"]),
        "scene_count": int(macro["scene_count"]),
        "coverage_bias_bp": coverage["coverage_bias_bp"],
        "coverage_mae_bp": coverage["coverage_mae_bp"],
        "coverage_rmse_bp": coverage["coverage_rmse_bp"],
        "coverage_p95_abs_error_bp": coverage["coverage_p95_abs_error_bp"],
    }


def build_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
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
    if (
        config.checkpoint_selection_metric == "validation_cloud_dice_constrained_false_clear"
        and not config.record_epoch_metrics
    ):
        raise ValueError("Dice-based checkpoint selection requires record_epoch_metrics=True")
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
    native_collate = (
        partial(collate_segmentation_batch, ignore_index=config.ignore_index)
        if config.preserve_native_size
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=native_collate,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=native_collate,
    )
    model = get_segformer_b0()
    pretrained_report = None
    if config.pretrained_encoder_path:
        pretrained_report = load_segformer_mit_b0_encoder(model, config.pretrained_encoder_path)
    model = model.to(selected_device)
    optimizer = build_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.lr_plateau_factor,
        patience=config.lr_plateau_patience,
        threshold=0.0,
        threshold_mode="rel",
        min_lr=config.min_learning_rate,
        eps=1e-12,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp and selected_device.type == "cuda")
    best_loss = float("inf")
    best_score = float("-inf")
    best_selection: dict[str, Any] | None = None
    best_epoch: int | None = None
    best_state: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    global_step = 0
    stale_epochs = 0
    scheduler_start_epoch = (
        config.warmup_epochs
        if config.scheduler_start_epoch is None
        else config.scheduler_start_epoch
    )
    for epoch in range(config.epochs):
        warmup_factor = apply_linear_warmup(
            optimizer,
            epoch=epoch,
            warmup_epochs=config.warmup_epochs,
            start_factor=config.warmup_start_factor,
        )
        epoch_learning_rates = learning_rates_by_group(optimizer)
        train_metrics = train_one_epoch(model, train_loader, optimizer, selected_device, config, scaler=scaler)
        validation_metrics = evaluate_loss(model, validation_loader, selected_device, config)
        if config.record_epoch_metrics:
            validation_metrics["segmentation"] = _epoch_segmentation_metrics(
                model,
                validation_loader,
                device=selected_device,
                threshold_bp=config.metric_threshold_bp,
                bootstrap_seed=config.seed,
            )
        selection_score, selection = _selection_summary(validation_metrics, config)
        if epoch >= scheduler_start_epoch:
            scheduler.step(float(validation_metrics["loss"]))
        global_step += int(train_metrics["optimizer_steps"])
        record = {
            "epoch": epoch,
            "learning_rate": float(next(iter(epoch_learning_rates.values()))),
            "learning_rates": epoch_learning_rates,
            "next_learning_rates": learning_rates_by_group(optimizer),
            "warmup_factor": warmup_factor,
            "optimizer_step": global_step,
            "train": train_metrics,
            "validation": validation_metrics,
            "checkpoint_selection": selection,
        }
        history.append(record)
        val_loss = float(validation_metrics["loss"])
        if int(validation_metrics["valid_pixels"]) > 0 and selection_score > best_score:
            best_loss = val_loss
            best_score = selection_score
            best_selection = selection
            best_epoch = epoch
            stale_epochs = 0
            best_state = build_checkpoint(
                model,
                optimizer,
                scheduler,
                scaler=scaler,
                epoch=epoch,
                global_step=global_step,
                best_metric=best_score,
                config=config,
                metadata={
                    **(metadata or {}),
                    "checkpoint_selection": selection,
                    "checkpoint_selection_epoch": epoch,
                },
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
        "best_validation_score": best_score,
        "best_epoch": best_epoch,
        "best_checkpoint_selection": best_selection,
        "history": history,
        "metadata": metadata or {},
        "training_config": asdict(config),
        "pretrained_encoder": pretrained_report,
    }
    serialized_report = json.dumps(report, indent=2, sort_keys=True, default=str)
    output.with_suffix(".json").write_text(serialized_report, encoding="utf-8")
    return json.loads(serialized_report)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the SegFormer-B0 cloud segmenter")
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--validation-dir", required=True)
    parser.add_argument("--output", default="checkpoints/segformer_b0_rgb_r1.pth")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=6e-5)
    parser.add_argument("--encoder-learning-rate", type=float, default=None)
    parser.add_argument("--decoder-learning-rate", type=float, default=None)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--warmup-start-factor", type=float, default=0.0)
    parser.add_argument("--scheduler-start-epoch", type=int, default=None)
    parser.add_argument("--lr-plateau-patience", type=int, default=5)
    parser.add_argument("--lr-plateau-factor", type=float, default=0.5)
    parser.add_argument("--min-learning-rate", type=float, default=1e-7)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--record-epoch-metrics", action="store_true")
    parser.add_argument("--metric-threshold-bp", type=int, default=5000)
    parser.add_argument("--max-false-clear-rate", type=float, default=0.05)
    parser.add_argument(
        "--checkpoint-selection-metric",
        choices=("validation_loss", "validation_cloud_dice_constrained_false_clear"),
        default="validation_loss",
    )
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
            encoder_learning_rate=args.encoder_learning_rate,
            decoder_learning_rate=args.decoder_learning_rate,
            warmup_epochs=args.warmup_epochs,
            warmup_start_factor=args.warmup_start_factor,
            scheduler_start_epoch=args.scheduler_start_epoch,
            lr_plateau_patience=args.lr_plateau_patience,
            lr_plateau_factor=args.lr_plateau_factor,
            min_learning_rate=args.min_learning_rate,
            batch_size=args.batch_size,
            seed=args.seed,
            preserve_native_size=not args.tile_training,
            record_epoch_metrics=args.record_epoch_metrics,
            metric_threshold_bp=args.metric_threshold_bp,
            max_false_clear_rate=args.max_false_clear_rate,
            checkpoint_selection_metric=args.checkpoint_selection_metric,
        ),
        device=args.device,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
