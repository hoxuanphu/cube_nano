"""Validation threshold selection and scene-level segmentation metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

# Allow both ``python src/eval_segmentation.py`` and ``python -m src.eval_segmentation``.
if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[1]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

from sat_ai.segmentation import cloud_probabilities_from_logits

try:
    from data.segmentation_dataset import SegmentationDataset
    from models.segformer_b0 import get_segformer_b0
except ModuleNotFoundError:  # Package invocation: python -m src.eval_segmentation
    from src.data.segmentation_dataset import SegmentationDataset
    from src.models.segformer_b0 import get_segformer_b0


def confusion_counts(
    cloud_probability: np.ndarray,
    target: np.ndarray,
    validity_mask: np.ndarray,
    threshold_bp: int,
) -> dict[str, int]:
    probability = np.asarray(cloud_probability, dtype=np.float32)
    target = np.asarray(target)
    valid = np.asarray(validity_mask, dtype=bool) & (target != 255)
    if probability.shape != target.shape or target.shape != valid.shape:
        raise ValueError("probability, target and validity shapes must match")
    if not np.isfinite(probability).all() or not 0 <= int(threshold_bp) <= 10000:
        raise ValueError("probability must be finite and threshold must be in [0, 10000]")
    prediction = probability >= np.float32(int(threshold_bp) / 10000.0)
    positive = target == 1
    negative = target == 0
    return {
        "tp": int(np.count_nonzero(valid & prediction & positive)),
        "tn": int(np.count_nonzero(valid & ~prediction & negative)),
        "fp": int(np.count_nonzero(valid & prediction & negative)),
        "fn": int(np.count_nonzero(valid & ~prediction & positive)),
        "valid_pixels": int(np.count_nonzero(valid)),
    }


def _safe(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def metrics_from_counts(counts: dict[str, int]) -> dict[str, float | int]:
    tp, tn, fp, fn = (int(counts[key]) for key in ("tp", "tn", "fp", "fn"))
    return {
        **counts,
        "cloud_iou": _safe(tp, tp + fp + fn),
        "cloud_dice": _safe(2 * tp, 2 * tp + fp + fn),
        "cloud_precision": _safe(tp, tp + fp),
        "cloud_recall": _safe(tp, tp + fn),
        "false_clear_rate": _safe(fn, tp + fn),
        "accuracy": _safe(tp + tn, tp + tn + fp + fn),
    }


def segmentation_metrics(
    cloud_probability: np.ndarray,
    target: np.ndarray,
    validity_mask: np.ndarray,
    threshold_bp: int,
) -> dict[str, float | int]:
    return metrics_from_counts(
        confusion_counts(cloud_probability, target, validity_mask, threshold_bp)
    )


def scene_macro_metrics(
    cloud_probability: np.ndarray,
    target: np.ndarray,
    validity_mask: np.ndarray,
    scene_ids: Iterable[str],
    threshold_bp: int,
) -> dict[str, float | int]:
    """Aggregate metrics per scene so large scenes do not dominate the report."""

    probabilities = np.asarray(cloud_probability)
    targets = np.asarray(target)
    validities = np.asarray(validity_mask)
    scene_ids = tuple(str(value) for value in scene_ids)
    if probabilities.ndim != 3 or len(scene_ids) != probabilities.shape[0]:
        raise ValueError("scene IDs must contain one value per prediction tile")
    grouped: dict[str, list[int]] = {}
    for index, scene_id in enumerate(scene_ids):
        grouped.setdefault(scene_id, []).append(index)
    reports = []
    for indices in grouped.values():
        reports.append(
            segmentation_metrics(
                probabilities[indices],
                targets[indices],
                validities[indices],
                threshold_bp,
            )
        )
    keys = ("cloud_iou", "cloud_dice", "cloud_precision", "cloud_recall", "false_clear_rate")
    result: dict[str, float | int] = {"scene_count": len(reports)}
    for key in keys:
        result[f"macro_{key}"] = float(np.mean([float(report[key]) for report in reports]))
    return result


def select_pixel_threshold(
    cloud_probability: np.ndarray,
    target: np.ndarray,
    validity_mask: np.ndarray,
    *,
    candidates_bp: Iterable[int] = range(1000, 10000, 100),
    max_false_clear_rate: float = 0.05,
) -> tuple[int, dict[str, float | int]]:
    """Select only from validation predictions, constrained by false-clear."""

    if not 0.0 <= max_false_clear_rate <= 1.0:
        raise ValueError("max_false_clear_rate must be in [0, 1]")
    candidates = tuple(int(value) for value in candidates_bp)
    if not candidates or any(value < 0 or value > 10000 for value in candidates):
        raise ValueError("threshold candidates must be basis points in [0, 10000]")
    eligible: list[tuple[int, dict[str, float | int]]] = []
    all_metrics: list[tuple[int, dict[str, float | int]]] = []
    for threshold_bp in candidates:
        metric = segmentation_metrics(cloud_probability, target, validity_mask, threshold_bp)
        all_metrics.append((threshold_bp, metric))
        if float(metric["false_clear_rate"]) <= max_false_clear_rate:
            eligible.append((threshold_bp, metric))
    pool = eligible or all_metrics
    threshold, metric = max(
        pool,
        key=lambda item: (
            float(item[1]["cloud_dice"]),
            float(item[1]["cloud_iou"]),
            -item[0],
        ),
    )
    return threshold, metric


def _eroded(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    return (
        padded[1:-1, 1:-1]
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )


def _dilated(mask: np.ndarray, radius: int) -> np.ndarray:
    result = np.asarray(mask, dtype=bool)
    for _ in range(max(0, radius)):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        result = (
            padded[1:-1, 1:-1]
            | padded[:-2, 1:-1]
            | padded[2:, 1:-1]
            | padded[1:-1, :-2]
            | padded[1:-1, 2:]
        )
    return result


def boundary_f1(prediction: np.ndarray, target: np.ndarray, *, tolerance_pixels: int = 2) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if prediction.shape != target.shape or tolerance_pixels < 0:
        raise ValueError("boundary inputs or tolerance are invalid")
    prediction_boundary = prediction & ~_eroded(prediction)
    target_boundary = target & ~_eroded(target)
    predicted_count = int(np.count_nonzero(prediction_boundary))
    target_count = int(np.count_nonzero(target_boundary))
    matched_prediction = int(np.count_nonzero(prediction_boundary & _dilated(target_boundary, tolerance_pixels)))
    matched_target = int(np.count_nonzero(target_boundary & _dilated(prediction_boundary, tolerance_pixels)))
    precision = _safe(matched_prediction, predicted_count)
    recall = _safe(matched_target, target_count)
    return _safe(2 * precision * recall, precision + recall)


def coverage_ratio_bp(cloud_mask: np.ndarray, validity_mask: np.ndarray) -> int:
    cloud = np.asarray(cloud_mask, dtype=bool)
    valid = np.asarray(validity_mask, dtype=bool)
    if cloud.shape != valid.shape:
        raise ValueError("cloud and validity masks must have matching shapes")
    valid_count = int(np.count_nonzero(valid))
    if valid_count <= 0:
        raise ValueError("coverage requires at least one valid pixel")
    cloud_count = int(np.count_nonzero(cloud & valid))
    return (cloud_count * 10000) // valid_count


def coverage_error_metrics(
    predicted_masks: Iterable[np.ndarray],
    target_masks: Iterable[np.ndarray],
    validity_masks: Iterable[np.ndarray],
) -> dict[str, float | int]:
    errors: list[int] = []
    for predicted, target, validity in zip(predicted_masks, target_masks, validity_masks):
        predicted_bp = coverage_ratio_bp(predicted, validity)
        target_bp = coverage_ratio_bp(target, validity)
        errors.append(predicted_bp - target_bp)
    if not errors:
        raise ValueError("coverage metric input is empty")
    absolute = np.abs(np.asarray(errors, dtype=np.float64))
    return {
        "scene_count": len(errors),
        "coverage_bias_bp": float(np.mean(errors)),
        "coverage_mae_bp": float(np.mean(absolute)),
        "coverage_rmse_bp": float(np.sqrt(np.mean(np.square(errors)))),
        "coverage_p95_abs_error_bp": float(np.percentile(absolute, 95)),
    }


def bootstrap_scene_metric(values: Iterable[float], *, samples: int = 1000, seed: int = 42) -> dict[str, float]:
    values = np.asarray(tuple(float(value) for value in values), dtype=np.float64)
    if values.size == 0 or samples <= 0:
        raise ValueError("bootstrap values and sample count must be non-empty/positive")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, values.size, size=(samples, values.size))
    means = values[draws].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "lower_95": float(np.percentile(means, 2.5)),
        "upper_95": float(np.percentile(means, 97.5)),
        "samples": samples,
        "seed": seed,
    }


def _group_scene_indices(scene_ids: Iterable[str], expected_count: int) -> dict[str, list[int]]:
    values = tuple(str(value) for value in scene_ids)
    if len(values) != expected_count:
        raise ValueError("scene IDs must contain one value per prediction sample")
    grouped: dict[str, list[int]] = {}
    for index, scene_id in enumerate(values):
        grouped.setdefault(scene_id, []).append(index)
    return grouped


def _scene_reports(
    probabilities: np.ndarray,
    targets: np.ndarray,
    validities: np.ndarray,
    scene_ids: Iterable[str],
    threshold_bp: int,
) -> list[dict[str, float | int | str | None]]:
    reports: list[dict[str, float | int | str | None]] = []
    threshold = np.float32(threshold_bp / 10000.0)
    for scene_id, indices in _group_scene_indices(scene_ids, probabilities.shape[0]).items():
        probability = probabilities[indices]
        target = targets[indices]
        validity = validities[indices]
        valid = np.asarray(validity, dtype=bool) & (target != 255)
        metric = segmentation_metrics(probability, target, validity, threshold_bp)
        boundary_values = [
            boundary_f1(
                (sample_probability >= threshold) & sample_valid,
                (sample_target == 1) & sample_valid,
                tolerance_pixels=2,
            )
            for sample_probability, sample_target, sample_valid in zip(probability, target, valid)
        ]
        report: dict[str, float | int | str | None] = {
            "scene_id": scene_id,
            **metric,
            "boundary_f1": float(np.mean(boundary_values)),
        }
        if int(metric["valid_pixels"]) > 0:
            report["predicted_coverage_bp"] = coverage_ratio_bp(probability >= threshold, valid)
            report["target_coverage_bp"] = coverage_ratio_bp(target == 1, valid)
        else:
            report["predicted_coverage_bp"] = None
            report["target_coverage_bp"] = None
        reports.append(report)
    return reports


def _coverage_summary(scene_reports: Iterable[dict[str, float | int | str | None]]) -> dict[str, float | int | None]:
    reports = tuple(scene_reports)
    errors = [
        int(report["predicted_coverage_bp"]) - int(report["target_coverage_bp"])
        for report in reports
        if report["predicted_coverage_bp"] is not None and report["target_coverage_bp"] is not None
    ]
    excluded = len(reports) - len(errors)
    if not errors:
        return {
            "scene_count": 0,
            "excluded_all_invalid_scenes": excluded,
            "coverage_bias_bp": None,
            "coverage_mae_bp": None,
            "coverage_rmse_bp": None,
            "coverage_p95_abs_error_bp": None,
        }
    absolute = np.abs(np.asarray(errors, dtype=np.float64))
    return {
        "scene_count": len(errors),
        "excluded_all_invalid_scenes": excluded,
        "coverage_bias_bp": float(np.mean(errors)),
        "coverage_mae_bp": float(np.mean(absolute)),
        "coverage_rmse_bp": float(np.sqrt(np.mean(np.square(errors)))),
        "coverage_p95_abs_error_bp": float(np.percentile(absolute, 95)),
    }


def evaluate_predictions(
    cloud_probability: np.ndarray,
    target: np.ndarray,
    validity_mask: np.ndarray,
    scene_ids: Iterable[str],
    *,
    threshold_bp: int,
    source_pixel_count: int | None = None,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 42,
) -> dict:
    """Build reproducible pixel and scene-level evidence for a locked threshold."""

    probabilities = np.asarray(cloud_probability, dtype=np.float32)
    targets = np.asarray(target)
    validities = np.asarray(validity_mask, dtype=bool)
    scene_values = tuple(str(value) for value in scene_ids)
    if (
        probabilities.ndim != 3
        or probabilities.shape[0] == 0
        or probabilities.shape != targets.shape
        or targets.shape != validities.shape
    ):
        raise ValueError("prediction, target and validity arrays must be matching [N, H, W] tensors")
    if source_pixel_count is None:
        source_pixel_count = int(sum(np.asarray(item).size for item in targets))
    if source_pixel_count <= 0:
        raise ValueError("source_pixel_count must be positive")
    reports = _scene_reports(probabilities, targets, validities, scene_values, threshold_bp)
    metrics = segmentation_metrics(probabilities, targets, validities, threshold_bp)
    metrics["boundary_f1"] = float(np.mean([float(report["boundary_f1"]) for report in reports]))
    macro = scene_macro_metrics(probabilities, targets, validities, scene_values, threshold_bp)
    macro["macro_boundary_f1"] = float(np.mean([float(report["boundary_f1"]) for report in reports]))
    bootstrap_keys = (
        "cloud_iou",
        "cloud_dice",
        "cloud_precision",
        "cloud_recall",
        "false_clear_rate",
        "boundary_f1",
    )
    return {
        "schema_version": 1,
        "threshold_bp": int(threshold_bp),
        "metrics": metrics,
        "macro_scene_metrics": macro,
        "coverage_metrics": _coverage_summary(reports),
        "scene_bootstrap": {
            key: bootstrap_scene_metric(
                (float(report[key]) for report in reports),
                samples=bootstrap_samples,
                seed=bootstrap_seed,
            )
            for key in bootstrap_keys
        },
        "scene_metrics": reports,
        "scene_ids": [str(report["scene_id"]) for report in reports],
        "source_pixel_count": int(source_pixel_count),
        "valid_pixel_ratio": float(np.count_nonzero(validities) / source_pixel_count),
    }


def calibrate_validation_predictions(
    cloud_probability: np.ndarray,
    target: np.ndarray,
    validity_mask: np.ndarray,
    scene_ids: Iterable[str],
    *,
    candidates_bp: Iterable[int] = range(1000, 10000, 100),
    max_false_clear_rate: float = 0.05,
    source_pixel_count: int | None = None,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 42,
) -> dict:
    """Select a pixel threshold from validation predictions and record the evidence."""

    candidate_values = tuple(int(value) for value in candidates_bp)
    threshold_bp, selection_metrics = select_pixel_threshold(
        cloud_probability,
        target,
        validity_mask,
        candidates_bp=candidate_values,
        max_false_clear_rate=max_false_clear_rate,
    )
    report = evaluate_predictions(
        cloud_probability,
        target,
        validity_mask,
        scene_ids,
        threshold_bp=threshold_bp,
        source_pixel_count=source_pixel_count,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    report["threshold_selection"] = {
        "dataset_role": "validation",
        "selection_metric": "validation-false-clear-constrained-dice",
        "max_false_clear_rate": float(max_false_clear_rate),
        "candidates_bp": list(candidate_values),
        "selected_metrics": selection_metrics,
    }
    return report


@torch.no_grad()
def collect_predictions(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    validities: list[np.ndarray] = []
    scene_ids: list[str] = []
    source_pixel_count = 0
    for batch in loader:
        images = batch["image"].to(device)
        logits = model(images)
        batch_probability = cloud_probabilities_from_logits(
            logits,
            batch["mask"].shape[-2:],
        )[:, 1].cpu().numpy()
        batch_target = batch["mask"].numpy()
        batch_validity = batch["validity_mask"].numpy()
        probabilities.extend(batch_probability)
        targets.extend(batch_target)
        validities.extend(batch_validity)
        scene_ids.extend(str(value) for value in batch["scene_id"])
    if not probabilities:
        raise ValueError("prediction loader is empty")
    if len(probabilities) != len(targets) or len(targets) != len(validities):
        raise AssertionError("prediction collection lost sample alignment")
    source_pixel_count = sum(int(target.size) for target in targets)
    max_height = max(int(target.shape[0]) for target in targets)
    max_width = max(int(target.shape[1]) for target in targets)
    probability_stack = np.zeros((len(probabilities), max_height, max_width), dtype=np.float32)
    target_stack = np.full((len(targets), max_height, max_width), 255, dtype=np.int64)
    validity_stack = np.zeros((len(validities), max_height, max_width), dtype=bool)
    for index, (probability, target, validity) in enumerate(zip(probabilities, targets, validities)):
        if probability.shape != target.shape or target.shape != validity.shape:
            raise ValueError("prediction, target and validity sample shapes must match")
        height, width = target.shape
        probability_stack[index, :height, :width] = probability
        target_stack[index, :height, :width] = target
        validity_stack[index, :height, :width] = validity
    return probability_stack, target_stack, validity_stack, scene_ids, source_pixel_count


def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    threshold_bp: int,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 42,
) -> dict:
    probabilities, targets, validities, scene_ids, source_pixel_count = collect_predictions(model, loader, device)
    return evaluate_predictions(
        probabilities,
        targets,
        validities,
        scene_ids,
        threshold_bp=threshold_bp,
        source_pixel_count=source_pixel_count,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )


def calibrate_validation_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    candidates_bp: Iterable[int] = range(1000, 10000, 100),
    max_false_clear_rate: float = 0.05,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 42,
) -> dict:
    probabilities, targets, validities, scene_ids, source_pixel_count = collect_predictions(model, loader, device)
    return calibrate_validation_predictions(
        probabilities,
        targets,
        validities,
        scene_ids,
        candidates_bp=candidates_bp,
        max_false_clear_rate=max_false_clear_rate,
        source_pixel_count=source_pixel_count,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a SegFormer-B0 cloud segmenter")
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--threshold-bp", type=int, default=5000)
    parser.add_argument("--dataset-role", choices=("validation", "test"), default="test")
    parser.add_argument(
        "--select-threshold",
        action="store_true",
        help="Select a pixel threshold from validation predictions; requires --dataset-role validation",
    )
    parser.add_argument("--max-false-clear-rate", type=float, default=0.05)
    parser.add_argument("--threshold-start-bp", type=int, default=1000)
    parser.add_argument("--threshold-stop-bp", type=int, default=10000)
    parser.add_argument("--threshold-step-bp", type=int, default=100)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="results/segmentation_eval.json")
    args = parser.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset = SegmentationDataset(args.test_dir, is_train=False, preserve_native_size=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model = get_segformer_b0().to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    model.load_state_dict(checkpoint)
    if args.select_threshold:
        if args.dataset_role != "validation":
            parser.error("--select-threshold requires --dataset-role validation")
        if args.threshold_step_bp <= 0 or args.threshold_stop_bp <= args.threshold_start_bp:
            parser.error("threshold sweep requires positive step and stop greater than start")
        report = calibrate_validation_loader(
            model,
            loader,
            device=device,
            candidates_bp=range(args.threshold_start_bp, args.threshold_stop_bp, args.threshold_step_bp),
            max_false_clear_rate=args.max_false_clear_rate,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
    else:
        report = evaluate_loader(
            model,
            loader,
            device=device,
            threshold_bp=args.threshold_bp,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
        report["dataset_role"] = args.dataset_role
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
