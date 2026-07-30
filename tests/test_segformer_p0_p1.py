from __future__ import annotations

import numpy as np
import torch

from src.data.segmentation_dataset import collate_segmentation_batch
from src.eval_segmentation import build_error_atlas, calibrate_validation_predictions
from src.models.segformer_b0 import get_segformer_b0
from src.segformer_experiment_report import aggregate_validation_seed_reports
from src.train_segmentation import (
    SegmentationTrainingConfig,
    apply_linear_warmup,
    build_optimizer,
    train_one_epoch,
)


def test_p1_parameter_groups_and_linear_warmup_use_independent_lrs():
    model = get_segformer_b0()
    config = SegmentationTrainingConfig(
        learning_rate=6e-5,
        encoder_learning_rate=1e-5,
        decoder_learning_rate=1e-4,
        warmup_epochs=3,
        warmup_start_factor=0.0,
    )
    optimizer = build_optimizer(model, config)

    assert {group["group_name"] for group in optimizer.param_groups} == {"encoder", "decoder"}
    expected = ((1 / 3, 1 / 3), (2 / 3, 2 / 3), (1.0, 1.0))
    for epoch, (encoder_factor, decoder_factor) in enumerate(expected):
        assert apply_linear_warmup(
            optimizer,
            epoch=epoch,
            warmup_epochs=config.warmup_epochs,
            start_factor=config.warmup_start_factor,
        ) == encoder_factor
        learning_rates = {group["group_name"]: group["lr"] for group in optimizer.param_groups}
        assert learning_rates["encoder"] == config.encoder_learning_rate * encoder_factor
        assert learning_rates["decoder"] == config.decoder_learning_rate * decoder_factor


def test_p1_native_padding_is_separate_from_invalid_pixels_in_epoch_metrics():
    samples = [
        {
            "image": torch.zeros((3, 2, 4), dtype=torch.float32),
            "mask": torch.zeros((2, 4), dtype=torch.long),
            "validity_mask": torch.ones((2, 4), dtype=torch.bool),
            "scene_id": "scene_a",
            "tile_coordinates": (0, 0, 4, 2),
        },
        {
            "image": torch.zeros((3, 4, 5), dtype=torch.float32),
            "mask": torch.zeros((4, 5), dtype=torch.long),
            "validity_mask": torch.ones((4, 5), dtype=torch.bool),
            "scene_id": "scene_b",
            "tile_coordinates": (0, 0, 5, 4),
        },
    ]
    batch = collate_segmentation_batch(samples)
    model = torch.nn.Conv2d(3, 2, kernel_size=1)
    metrics = train_one_epoch(
        model,
        [batch],
        torch.optim.AdamW(model.parameters(), lr=1e-3),
        torch.device("cpu"),
        SegmentationTrainingConfig(use_amp=False),
    )

    assert batch["sample_shapes"] == [(2, 4), (4, 5)]
    assert metrics["batch_pixels"] == 40
    assert metrics["padding_pixels"] == 12
    assert metrics["padding_ratio"] == 0.3
    assert metrics["valid_pixels"] == 28


def _validation_report() -> dict:
    probabilities = np.array(
        [
            [[0.10, 0.90], [0.10, 0.90]],
            [[0.10, 0.90], [0.10, 0.90]],
        ],
        dtype=np.float32,
    )
    targets = np.array(
        [
            [[0, 1], [0, 1]],
            [[0, 1], [0, 1]],
        ],
        dtype=np.int64,
    )
    return calibrate_validation_predictions(
        probabilities,
        targets,
        np.ones_like(targets, dtype=bool),
        ["scene_a", "scene_b"],
        candidates_bp=(0, 5000, 10000),
        max_false_clear_rate=0.0,
        refine_step_bp=25,
        bootstrap_samples=8,
    )


def test_p0_calibration_refines_a_coarse_validation_sweep():
    report = _validation_report()
    selection = report["threshold_selection"]

    assert selection["coarse_candidates_bp"] == [0, 5000, 10000]
    assert selection["coarse_threshold_bp"] == 5000
    assert selection["refine_step_bp"] == 25
    assert selection["refined_candidates_bp"]
    assert 0 <= report["threshold_bp"] <= 10000


def test_p0_error_atlas_ranks_each_required_scene_failure_mode():
    report = {
        "dataset_role": "validation",
        "threshold_bp": 5000,
        "scene_metrics": [
            {"scene_id": "fn", "false_clear_rate": 0.9, "fn": 9, "false_positive_rate": 0.1, "fp": 1, "boundary_f1": 0.8, "valid_pixels": 10},
            {"scene_id": "fp", "false_clear_rate": 0.1, "fn": 1, "false_positive_rate": 0.8, "fp": 8, "boundary_f1": 0.7, "valid_pixels": 10},
            {"scene_id": "edge", "false_clear_rate": 0.1, "fn": 1, "false_positive_rate": 0.1, "fp": 1, "boundary_f1": 0.2, "valid_pixels": 10},
        ],
    }
    atlas = build_error_atlas(report, top_k=2)

    assert atlas["rankings"]["highest_false_clear_rate"][0]["scene_id"] == "fn"
    assert atlas["rankings"]["highest_false_positive_rate"][0]["scene_id"] == "fp"
    assert atlas["rankings"]["lowest_boundary_f1"][0]["scene_id"] == "edge"


def test_p0_seed_aggregate_keeps_micro_macro_and_scene_bootstrap_evidence():
    first = _validation_report()
    second = _validation_report()
    first["_cube_nano_cache"] = {"seed": 42}
    second["_cube_nano_cache"] = {"seed": 43}

    aggregate = aggregate_validation_seed_reports(
        [first, second],
        bootstrap_samples=16,
        bootstrap_seed=7,
    )

    assert aggregate["dataset_role"] == "validation"
    assert aggregate["seeds"] == [42, 43]
    assert aggregate["seed_summary"]["cloud_dice"]["median"] == 1.0
    assert aggregate["seed_summary"]["macro_cloud_dice"]["median"] == 1.0
    assert aggregate["scene_bootstrap_across_seeds"]["boundary_f1"]["samples"] == 16
