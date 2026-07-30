"""Aggregate validation-only SegFormer experiment evidence across training seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np


_METRICS = (
    "cloud_iou",
    "cloud_dice",
    "cloud_precision",
    "cloud_recall",
    "false_clear_rate",
    "false_positive_rate",
    "boundary_f1",
)


def _seed_from_report(report: dict) -> int | None:
    cache = report.get("_cube_nano_cache", {})
    if not isinstance(cache, dict) or cache.get("seed") is None:
        return None
    return int(cache["seed"])


def _bootstrap_multi_seed_scene_metric(
    reports: Iterable[dict],
    metric: str,
    *,
    samples: int,
    seed: int,
) -> dict[str, float | int]:
    scene_values = [
        np.asarray([float(scene[metric]) for scene in report["scene_metrics"]], dtype=np.float64)
        for report in reports
    ]
    if not scene_values or any(values.size == 0 for values in scene_values) or samples <= 0:
        raise ValueError("reports need non-empty scene metrics and positive bootstrap samples")
    rng = np.random.default_rng(seed)
    draws = []
    for values in scene_values:
        indices = rng.integers(0, values.size, size=(samples, values.size))
        draws.append(values[indices].mean(axis=1))
    means = np.asarray(draws).mean(axis=0)
    point_estimate = float(np.mean([values.mean() for values in scene_values]))
    return {
        "mean": point_estimate,
        "lower_95": float(np.percentile(means, 2.5)),
        "upper_95": float(np.percentile(means, 97.5)),
        "samples": int(samples),
        "bootstrap_seed": int(seed),
        "seed_count": len(scene_values),
    }


def aggregate_validation_seed_reports(
    reports: Iterable[dict],
    *,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 42,
) -> dict:
    """Summarize three-seed validation evidence without consuming a test split."""

    values = list(reports)
    if not values:
        raise ValueError("at least one validation report is required")
    for report in values:
        if report.get("threshold_selection", {}).get("dataset_role") != "validation":
            raise ValueError("only validation calibration reports can be aggregated")
        if not isinstance(report.get("scene_metrics"), list) or not report["scene_metrics"]:
            raise ValueError("validation report has no scene metrics")
    seeds = [_seed_from_report(report) for report in values]
    if len([seed for seed in seeds if seed is not None]) != len(set(seed for seed in seeds if seed is not None)):
        raise ValueError("each aggregated report must use a distinct training seed")

    seed_reports = []
    for report, training_seed in zip(values, seeds):
        seed_reports.append(
            {
                "seed": training_seed,
                "threshold_bp": int(report["threshold_bp"]),
                "micro_metrics": {key: float(report["metrics"][key]) for key in _METRICS},
                "macro_metrics": {
                    key: float(report["macro_scene_metrics"][f"macro_{key}"])
                    for key in _METRICS
                },
            }
        )

    summary: dict[str, dict[str, float]] = {}
    for level, prefix in (("micro_metrics", ""), ("macro_metrics", "macro_")):
        for metric in _METRICS:
            series = np.asarray([report[level][metric] for report in seed_reports], dtype=np.float64)
            summary[f"{prefix}{metric}"] = {
                "mean": float(series.mean()),
                "median": float(np.median(series)),
                "minimum": float(series.min()),
                "maximum": float(series.max()),
            }

    return {
        "schema_version": 1,
        "dataset_role": "validation",
        "selection_metric": "validation_cloud_dice_constrained_false_clear",
        "seed_count": len(values),
        "seeds": seeds,
        "seed_reports": seed_reports,
        "seed_summary": summary,
        "scene_bootstrap_across_seeds": {
            metric: _bootstrap_multi_seed_scene_metric(
                values,
                metric,
                samples=bootstrap_samples,
                seed=bootstrap_seed,
            )
            for metric in _METRICS
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate SegFormer validation reports across seeds")
    parser.add_argument("--report", action="append", required=True, help="Validation calibration report; repeat per seed")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    args = parser.parse_args()
    reports = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.report]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    aggregate = aggregate_validation_seed_reports(
        reports,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    output.write_text(json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
