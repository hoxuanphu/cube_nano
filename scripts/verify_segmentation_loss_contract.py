"""Run dependency-light checks for the asymmetric segmentation-loss contract."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.eval_segmentation import bootstrap_paired_scene_metric, calibrate_validation_predictions
from src.losses import masked_segmentation_loss, soft_dice_loss
from src.segmentation_experiment import DEFAULT_SEEDS, DEFAULT_WEIGHTS
from src.train_segmentation import SegmentationTrainingConfig


def _check_loss_contract() -> dict[str, str]:
    logits = torch.tensor([[[[5.0, 5.0]], [[-5.0, -5.0]]]], requires_grad=True)
    target = torch.tensor([[[1, 0]]], dtype=torch.long)
    validity = torch.ones_like(target, dtype=torch.bool)
    baseline, _ = masked_segmentation_loss(
        logits,
        target,
        validity_mask=validity,
        cloud_class_weight=1.0,
        dice_weight=0.0,
    )
    weighted, _ = masked_segmentation_loss(
        logits,
        target,
        validity_mask=validity,
        cloud_class_weight=2.0,
        dice_weight=0.0,
    )
    assert float(weighted.detach()) > float(baseline.detach()), "cloud miss was not penalized more strongly"
    weighted.backward()
    assert torch.isfinite(logits.grad).all(), "weighted loss gradient is non-finite"

    reference_logits = torch.tensor([[[[3.0, -2.0]], [[-3.0, 2.0]]]])
    ignore_target = torch.tensor([[[255, 1]]], dtype=torch.long)
    changed_target = torch.tensor([[[0, 1]]], dtype=torch.long)
    ignore_loss, _ = masked_segmentation_loss(
        reference_logits,
        ignore_target,
        validity_mask=torch.tensor([[[False, True]]]),
        dice_weight=0.0,
    )
    changed_loss, _ = masked_segmentation_loss(
        reference_logits,
        changed_target,
        validity_mask=torch.tensor([[[False, True]]]),
        dice_weight=0.0,
    )
    torch.testing.assert_close(ignore_loss, changed_loss)

    invalid_target = torch.tensor([[[1, 0]]], dtype=torch.long)
    invalid_mask = torch.tensor([[[False, True]]])
    clear_invalid, _ = masked_segmentation_loss(
        reference_logits, invalid_target, validity_mask=invalid_mask, dice_weight=0.0
    )
    cloud_invalid, _ = masked_segmentation_loss(
        reference_logits,
        torch.tensor([[[0, 0]]]),
        validity_mask=invalid_mask,
        dice_weight=0.0,
    )
    torch.testing.assert_close(clear_invalid, cloud_invalid)

    all_invalid_logits = torch.zeros((1, 2, 2, 2), requires_grad=True)
    all_invalid_target = torch.full((1, 2, 2), 255, dtype=torch.long)
    zero, _ = masked_segmentation_loss(
        all_invalid_logits,
        all_invalid_target,
        validity_mask=torch.zeros_like(all_invalid_target, dtype=torch.bool),
    )
    assert float(zero.detach()) == 0.0
    zero.backward()
    assert float(all_invalid_logits.grad.abs().sum()) == 0.0

    compatibility_logits = torch.tensor(
        [[[[1.0, -1.0]], [[-1.0, 1.0]]]], requires_grad=True
    )
    compatibility_target = torch.tensor([[[0, 1]]], dtype=torch.long)
    compatibility_validity = torch.ones_like(compatibility_target, dtype=torch.bool)
    actual, _ = masked_segmentation_loss(
        compatibility_logits,
        compatibility_target,
        validity_mask=compatibility_validity,
    )
    expected = F.cross_entropy(compatibility_logits, compatibility_target) + soft_dice_loss(
        compatibility_logits,
        compatibility_target,
        validity_mask=compatibility_validity,
    )
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(compatibility_logits.grad).all()
    return {"status": "PASS", "checks": "weighted, ignore, validity, all-invalid, backward, baseline"}


def _check_evaluation_contract() -> dict[str, str | float | int]:
    paired = bootstrap_paired_scene_metric(
        {"scene-a": 0.5, "scene-b": 0.7},
        {"scene-a": 0.6, "scene-b": 0.8},
        samples=100,
        seed=42,
    )
    assert paired["scene_count"] == 2
    assert abs(float(paired["mean_difference"]) - 0.1) < 1e-12
    try:
        bootstrap_paired_scene_metric({"scene-a": 0.5}, {"scene-b": 0.6})
    except ValueError:
        pass
    else:
        raise AssertionError("paired bootstrap accepted different scene IDs")

    probabilities = np.array([[[0.1, 0.9], [0.8, 0.2]]], dtype=np.float32)
    target = np.array([[[0, 1], [1, 0]]], dtype=np.int64)
    validity = np.ones_like(target, dtype=bool)
    report = calibrate_validation_predictions(
        probabilities,
        target,
        validity,
        ["scene-a"],
        max_false_clear_rate=0.0,
        bootstrap_samples=10,
    )
    assert report["threshold_selection"]["dataset_role"] == "validation"
    return {
        "status": "PASS",
        "paired_scene_count": int(paired["scene_count"]),
        "calibrated_threshold_bp": int(report["threshold_bp"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="artifacts/segmentation_loss_asymmetric_penalty/l1_l2_smoke.json",
    )
    args = parser.parse_args()
    config = SegmentationTrainingConfig(cloud_class_weight=1.5)
    config.validate()
    result = {
        "schema_version": 1,
        "status": "PASS",
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "class_mapping": {"surface": 0, "cloud": 1, "ignore_index": 255},
        "fixed_loss_config": {"cross_entropy_weight": 1.0, "dice_weight": 1.0},
        "ablation_weights": list(DEFAULT_WEIGHTS),
        "ablation_seeds": list(DEFAULT_SEEDS),
        "loss_contract": _check_loss_contract(),
        "evaluation_contract": _check_evaluation_contract(),
        "pytest_status": "not installed; equivalent assertions executed by this verifier",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
