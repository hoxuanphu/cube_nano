"""Auditing and reproducible ablation orchestration for cloud segmentation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import traceback
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[1]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

from src.data.segmentation_dataset import SegmentationDataset, collate_segmentation_batch
from src.eval_segmentation import (
    bootstrap_paired_scene_metric,
    calibrate_validation_loader,
)
from src.models.segformer_b0 import get_segformer_b0
from src.train_segmentation import SegmentationTrainingConfig, train


DEFAULT_WEIGHTS = (1.0, 1.2, 1.5, 2.0, 3.0)
DEFAULT_SEEDS = (42, 123, 456)
PRIMARY_METRIC_KEYS = (
    "cloud_recall",
    "false_clear_rate",
    "cloud_precision",
    "cloud_dice",
    "cloud_iou",
)
SUMMARY_METRIC_KEYS = PRIMARY_METRIC_KEYS + (
    "boundary_f1",
    "coverage_bias_bp",
    "coverage_mae_bp",
    "coverage_rmse_bp",
    "coverage_p95_abs_error_bp",
    "train_loss",
    "train_cross_entropy",
    "train_soft_dice",
    "validation_loss",
    "validation_cross_entropy",
    "validation_soft_dice",
    "convergence_epoch",
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _git_commit() -> str:
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _dataset_fingerprint(dataset_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(dataset_dir.rglob("*.npy")):
        relative = str(path.relative_to(dataset_dir)).replace("\\", "/").encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _lineage_manifest(data_dir: Path) -> dict[str, Any] | None:
    candidates = (
        data_dir / "scene_split_lineage.json",
        data_dir.parent / "scene_split_lineage.json",
        data_dir.parent.parent / "scene_split_lineage.json",
    )
    for path in candidates:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    return None


def audit_dataset(data_dir: str | Path, *, dataset_role: str = "evaluation") -> dict[str, Any]:
    """Count only valid target pixels and retain scene-level evidence."""

    if dataset_role not in {"training", "validation", "test", "evaluation"}:
        raise ValueError("dataset_role must be training, validation, test, or evaluation")
    root = Path(data_dir)
    dataset = SegmentationDataset(root, is_train=dataset_role == "training", preserve_native_size=True)
    totals = {"surface": 0, "cloud": 0, "valid_pixels": 0, "ignored_pixels": 0}
    scenes: dict[str, dict[str, int]] = defaultdict(
        lambda: {"surface": 0, "cloud": 0, "valid_pixels": 0, "ignored_pixels": 0}
    )
    for record in dataset.records:
        mask = np.load(record.mask_path, allow_pickle=False)
        validity = (
            np.load(record.validity_path, allow_pickle=False)
            if record.validity_path is not None
            else np.ones(mask.shape, dtype=np.uint8)
        ).astype(bool)
        if mask.shape != validity.shape:
            raise ValueError(f"mask/validity shape mismatch: {record.mask_path.name}")
        valid = validity & (mask != dataset.ignore_index)
        if not np.isin(mask[valid], (0, 1)).all():
            raise ValueError(f"valid target contains values other than 0/1: {record.mask_path}")
        counts = {
            "surface": int(np.count_nonzero(valid & (mask == 0))),
            "cloud": int(np.count_nonzero(valid & (mask == 1))),
            "valid_pixels": int(np.count_nonzero(valid)),
            "ignored_pixels": int(mask.size - np.count_nonzero(valid)),
        }
        for key, value in counts.items():
            totals[key] += value
            scenes[record.scene_id][key] += value

    valid_pixels = totals["valid_pixels"]
    surface = totals["surface"]
    cloud = totals["cloud"]
    return {
        "schema_version": 1,
        "dataset_role": dataset_role,
        "dataset_dir": str(root.resolve()),
        "dataset_fingerprint": _dataset_fingerprint(root),
        "sample_count": len(dataset.records),
        "scene_count": len(scenes),
        "class_mapping": {"surface": 0, "cloud": 1, "ignore_index": dataset.ignore_index},
        "counts": totals,
        "ratios": {
            "surface": float(surface / valid_pixels) if valid_pixels else 0.0,
            "cloud": float(cloud / valid_pixels) if valid_pixels else 0.0,
            "surface_to_cloud": float(surface / cloud) if cloud else math.inf,
        },
        "scene_counts": dict(sorted(scenes.items())),
        "split_lineage": _lineage_manifest(root),
        "method": "count valid & (target != 255); validity=False and target=255 excluded",
    }


def _load_checkpoint_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    model = get_segformer_b0().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state)
    return model


def _validation_loader(data_dir: Path, config: SegmentationTrainingConfig) -> DataLoader:
    dataset = SegmentationDataset(data_dir, is_train=False, preserve_native_size=config.preserve_native_size)
    collate = collate_segmentation_batch if config.preserve_native_size else None
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate)


def _best_epoch(history: Iterable[dict[str, Any]]) -> int | None:
    values = tuple(history)
    if not values:
        return None
    return int(min(values, key=lambda item: float(item["validation"]["loss"]))["epoch"])


def _best_history_record(history: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = tuple(history)
    if not values:
        raise ValueError("training history is empty")
    return min(values, key=lambda item: float(item["validation"]["loss"]))


def _run_one(
    train_dir: Path,
    validation_dir: Path,
    run_dir: Path,
    *,
    cloud_class_weight: float,
    seed: int,
    epochs: int,
    device: str | None,
    max_false_clear_rate: float,
    bootstrap_samples: int,
) -> dict[str, Any]:
    run_id = f"weight-{cloud_class_weight:g}_seed-{seed}"
    config = SegmentationTrainingConfig(
        epochs=epochs,
        seed=seed,
        cloud_class_weight=cloud_class_weight,
        cross_entropy_weight=1.0,
        dice_weight=1.0,
        preserve_native_size=True,
    )
    checkpoint_path = run_dir / "checkpoint.pth"
    metadata = {
        "experiment_id": "segmentation-asymmetric-penalty-v1",
        "run_id": run_id,
        "dataset_roles": {"train": str(train_dir.resolve()), "validation": str(validation_dir.resolve())},
        "threshold_policy": {
            "selected_from": "validation",
            "max_false_clear_rate": max_false_clear_rate,
        },
    }
    report = train(
        train_dir,
        validation_dir,
        checkpoint_path,
        config=config,
        device=device,
        metadata=metadata,
    )
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = _load_checkpoint_model(checkpoint_path, selected_device)
    validation_report = calibrate_validation_loader(
        model,
        _validation_loader(validation_dir, config),
        device=selected_device,
        max_false_clear_rate=max_false_clear_rate,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=seed,
    )
    validation_report.update(
        {
            "dataset_role": "validation",
            "run_id": run_id,
            "cloud_class_weight": cloud_class_weight,
            "seed": seed,
            "checkpoint_path": str(checkpoint_path),
            "convergence_epoch": _best_epoch(report["history"]),
            "training_report_path": str(run_dir / "training_report.json"),
            "provenance": {
                "git_commit": report["metadata"].get("git_commit", "unknown"),
                "implementation_id": report["implementation_id"],
                "dataset_fingerprint": {
                    "train": _dataset_fingerprint(train_dir),
                    "validation": _dataset_fingerprint(validation_dir),
                },
            },
            "training_config": asdict(config),
        }
    )
    _write_json(run_dir / "training_report.json", report)
    _write_json(run_dir / "validation_report.json", validation_report)
    best_record = _best_history_record(report["history"])
    coverage = validation_report["coverage_metrics"]
    return {
        "run_id": run_id,
        "status": "DONE",
        "cloud_class_weight": cloud_class_weight,
        "seed": seed,
        "checkpoint": str(checkpoint_path),
        "training_report": str(run_dir / "training_report.json"),
        "validation_report": str(run_dir / "validation_report.json"),
        "convergence_epoch": validation_report["convergence_epoch"],
        "threshold_bp": validation_report["threshold_bp"],
        "metrics": validation_report["metrics"],
        "coverage_metrics": coverage,
        "loss_components": {
            "train": best_record["train"],
            "validation": best_record["validation"],
        },
    }


def _aggregate(successful_runs: list[dict[str, Any]], output_dir: Path, *, bootstrap_samples: int) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in successful_runs:
        grouped[str(run["cloud_class_weight"])].append(run)
    summary: dict[str, Any] = {"schema_version": 1, "by_weight": {}, "paired_comparisons": []}
    for weight, runs in sorted(grouped.items(), key=lambda item: float(item[0])):
        metrics = {}
        for key in SUMMARY_METRIC_KEYS:
            if key in PRIMARY_METRIC_KEYS or key == "boundary_f1":
                values = [float(run["metrics"][key]) for run in runs]
            elif key.startswith("coverage_"):
                values = [float(run["coverage_metrics"][key]) for run in runs]
            elif key.startswith("train_"):
                values = [float(run["loss_components"]["train"][key.removeprefix("train_")]) for run in runs]
            elif key.startswith("validation_"):
                values = [float(run["loss_components"]["validation"][key.removeprefix("validation_")]) for run in runs]
            else:
                values = [float(run[key]) for run in runs]
            metrics[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "count": len(values),
            }
        summary["by_weight"][weight] = {"run_count": len(runs), "metrics": metrics, "runs": runs}

    baseline = {int(run["seed"]): run for run in grouped.get("1.0", [])}
    for weight, runs in sorted(grouped.items(), key=lambda item: float(item[0])):
        if float(weight) == 1.0:
            continue
        for run in runs:
            baseline_run = baseline.get(int(run["seed"]))
            if baseline_run is None:
                continue
            left = json.loads(Path(baseline_run["validation_report"]).read_text(encoding="utf-8"))
            right = json.loads(Path(run["validation_report"]).read_text(encoding="utf-8"))
            left_scene = {str(item["scene_id"]): item for item in left["scene_metrics"]}
            right_scene = {str(item["scene_id"]): item for item in right["scene_metrics"]}
            comparison = {"weight": float(weight), "seed": int(run["seed"]), "baseline_weight": 1.0, "metrics": {}}
            for key in PRIMARY_METRIC_KEYS:
                comparison["metrics"][key] = bootstrap_paired_scene_metric(
                    {scene: float(item[key]) for scene, item in left_scene.items()},
                    {scene: float(item[key]) for scene, item in right_scene.items()},
                    samples=bootstrap_samples,
                    seed=int(run["seed"]),
                )
            summary["paired_comparisons"].append(comparison)
    _write_json(output_dir / "ablation_summary.json", summary)
    return summary


def run_ablation(
    train_dir: str | Path,
    validation_dir: str | Path,
    output_dir: str | Path,
    *,
    weights: Iterable[float] = DEFAULT_WEIGHTS,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    epochs: int = 50,
    device: str | None = None,
    max_false_clear_rate: float = 0.05,
    bootstrap_samples: int = 1000,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "READY",
        "experiment_id": "segmentation-asymmetric-penalty-v1",
        "git_commit": _git_commit(),
        "weights": [float(value) for value in weights],
        "seeds": [int(value) for value in seeds],
        "fixed_loss_config": {"cross_entropy_weight": 1.0, "dice_weight": 1.0},
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "acceptance_criteria": {
            "profile_id": "segformer-b0-rgb-acceptance-v1",
            "max_false_clear_rate": max_false_clear_rate,
            "min_cloud_iou": 0.75,
            "min_cloud_dice": 0.85,
            "min_cloud_recall": 0.90,
            "max_coverage_mae_bp": 500,
            "max_coverage_p95_abs_error_bp": 1000,
            "pending_stakeholder_approval": {
                "min_cloud_precision": None,
                "max_cloud_dice_drop_vs_baseline": None,
            },
            "selection_priority": ["cloud_recall", "false_clear_rate", "lower_weight_if_indistinguishable"],
            "source": "sat_ai/acceptance_profile.yaml; precision and relative-Dice limits remain unapproved",
        },
        "runs": [],
    }
    train_root = Path(train_dir)
    validation_root = Path(validation_dir)
    try:
        train_audit = audit_dataset(train_root, dataset_role="training")
        validation_audit = audit_dataset(validation_root, dataset_role="validation")
    except Exception as exc:
        manifest.update(
            {
                "status": "BLOCKED",
                "blocker": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "condition_to_unlock": "provide frozen processed train/validation segmentation directories",
                },
            }
        )
        _write_json(output / "ablation_manifest.json", manifest)
        return manifest
    manifest["dataset_audit"] = {"train": train_audit, "validation": validation_audit}

    for weight in manifest["weights"]:
        for seed in manifest["seeds"]:
            run_dir = output / f"weight-{weight:g}_seed-{seed}"
            try:
                result = _run_one(
                    train_root,
                    validation_root,
                    run_dir,
                    cloud_class_weight=weight,
                    seed=seed,
                    epochs=epochs,
                    device=device,
                    max_false_clear_rate=max_false_clear_rate,
                    bootstrap_samples=bootstrap_samples,
                )
            except Exception as exc:
                result = {
                    "run_id": f"weight-{weight:g}_seed-{seed}",
                    "status": "FAILED",
                    "cloud_class_weight": weight,
                    "seed": seed,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                _write_json(run_dir / "run_failure.json", result)
            manifest["runs"].append(result)
            _write_json(output / "ablation_manifest.json", manifest)

    successful = [run for run in manifest["runs"] if run["status"] == "DONE"]
    failed = [run for run in manifest["runs"] if run["status"] != "DONE"]
    manifest["status"] = "DONE" if not failed else "PARTIAL_FAILURE"
    if successful:
        _aggregate(successful, output, bootstrap_samples=bootstrap_samples)
    _write_json(output / "ablation_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and run asymmetric cloud-loss experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--data-dir", required=True)
    audit_parser.add_argument("--output", required=True)
    audit_parser.add_argument(
        "--dataset-role",
        choices=("training", "validation", "test", "evaluation"),
        default="evaluation",
    )
    run_parser = subparsers.add_parser("ablation")
    run_parser.add_argument("--train-dir", required=True)
    run_parser.add_argument("--validation-dir", required=True)
    run_parser.add_argument("--output-dir", required=True)
    run_parser.add_argument("--epochs", type=int, default=50)
    run_parser.add_argument("--device", default=None)
    run_parser.add_argument("--bootstrap-samples", type=int, default=1000)
    args = parser.parse_args()
    if args.command == "audit":
        _write_json(Path(args.output), audit_dataset(args.data_dir, dataset_role=args.dataset_role))
        return
    manifest = run_ablation(
        args.train_dir,
        args.validation_dir,
        args.output_dir,
        epochs=args.epochs,
        device=args.device,
        bootstrap_samples=args.bootstrap_samples,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if manifest["status"] in {"BLOCKED", "PARTIAL_FAILURE"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
