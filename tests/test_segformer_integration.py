from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile
import torch

from gds.product_store import ProductManifest
from flight.deployment import validate_segmentation_release_contracts
from protocol.schemas import ConfigSnapshot, ProductRef, ROI, RequestKey
from sat_ai.contracts import AcceptanceProfile, TargetDeploymentSpec
from sat_ai.inference import InferenceConfig, InsufficientValidData, infer_region
from sat_ai.manifest import ModelManifest
from sat_ai.products import build_products
from sat_ai.roi import open_memmap_scene
from sat_ai.segmentation import cloud_probabilities_from_logits, postprocess_segmentation_logits
from sat_ai.threshold_lut import ThresholdLUT
from src.data.preprocess_95cloud import decode_ground_truth, process_scene
from src.data.segmentation_dataset import SegmentationDataset
from src.eval_segmentation import calibrate_validation_predictions, evaluate_loader
from src.losses import masked_segmentation_loss
from src.models.segformer_b0 import get_segformer_b0
from src.train_segmentation import SegmentationTrainingConfig, train_one_epoch


ROOT = Path(__file__).resolve().parents[1]


def _sidecar(source: Path, *, input_spec_id: str, validity: dict | None = None) -> Path:
    sidecar = source.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_fingerprint": {
                    "algorithm": "sha256",
                    "digest": hashlib.sha256(source.read_bytes()).hexdigest(),
                },
                "axes": "YXC",
                "shape": list(tifffile.imread(source).shape),
                "band_order": ["red", "green", "blue"],
                "dtype": "uint16",
                "input_spec_id": input_spec_id,
                "validity": validity or {"kind": "all_valid"},
            }
        ),
        encoding="utf-8",
    )
    return sidecar


def _runtime(logits: np.ndarray):
    manifest = ModelManifest.from_file(ROOT / "sat_ai" / "segformer_model_manifest.yaml")
    return SimpleNamespace(
        manifest=manifest,
        infer_outputs=lambda batch: np.repeat(logits, len(batch), axis=0),
    )


def _config(*, model_threshold_bp: int = 5000, coverage_limit_bp: int = 6000) -> InferenceConfig:
    manifest = ModelManifest.from_file(ROOT / "sat_ai" / "segformer_model_manifest.yaml")
    lut = ThresholdLUT.from_file(
        ROOT / "protocol" / "golden_vectors" / "threshold_lut.bin",
        manifest.threshold_lut_sha256,
    )
    return InferenceConfig(ConfigSnapshot(0, 0, model_threshold_bp, coverage_limit_bp), lut)


def test_manifest_migration_and_segmentation_contracts():
    classifier = ModelManifest.from_file(ROOT / "sat_ai" / "model_manifest.yaml")
    segmentation = ModelManifest.from_file(ROOT / "sat_ai" / "segformer_model_manifest.yaml")
    assert classifier.model_task == "patch_classification"
    assert segmentation.model_task == "semantic_cloud_segmentation"
    assert segmentation.output_spec.shape == (1, 2, 64, 64)
    assert segmentation.product_spec.cloud_value != segmentation.product_spec.valid_value


def test_segformer_release_gate_requires_validated_bound_contracts():
    manifest = ModelManifest.from_file(ROOT / "sat_ai" / "segformer_model_manifest.yaml")
    acceptance = AcceptanceProfile.from_file(ROOT / "sat_ai" / "acceptance_profile.yaml")
    target = TargetDeploymentSpec.from_file(ROOT / "sat_ai" / "target_deployment_spec.yaml")
    profile = {
        "target_id": target.target_id,
        "batch_size": 1,
        "deployable": True,
        "ready": True,
    }
    with pytest.raises(ValueError, match="validated"):
        validate_segmentation_release_contracts(manifest, profile, acceptance, target)

    with pytest.raises(ValueError, match="calibration_id"):
        replace(manifest, assurance_level="validated").validate()

    decision = replace(manifest.decision_spec, calibration_id="validation-calibration-v1")
    validated_manifest = replace(manifest, assurance_level="validated", decision_spec=decision)
    validated_manifest.validate()
    with pytest.raises(ValueError, match="approved"):
        validate_segmentation_release_contracts(validated_manifest, profile, acceptance, target)

    validate_segmentation_release_contracts(
        validated_manifest,
        profile,
        replace(acceptance, status="approved"),
        target,
    )


def test_ground_truth_decoder_rejects_unaudited_values_and_preserves_invalidity():
    cloud, valid = decode_ground_truth(np.array([[0, 255], [99, 1]], dtype=np.uint8), [99])
    np.testing.assert_array_equal(cloud, [[0, 1], [0, 1]])
    np.testing.assert_array_equal(valid, [[1, 1], [0, 1]])
    try:
        decode_ground_truth(np.array([[0, 7]], dtype=np.uint8))
    except ValueError as exc:
        assert "unsupported ground-truth values" in str(exc)
    else:
        raise AssertionError("unaudited ground-truth encoding was accepted")


def test_softmax_postprocess_never_marks_invalid_pixels_as_cloud():
    logits = np.zeros((1, 2, 64, 64), dtype=np.float32)
    logits[:, 1] = 10.0
    validity = np.ones((1, 256, 256), dtype=bool)
    validity[:, 0, 0] = False
    result = postprocess_segmentation_logits(
        logits,
        validity,
        target_size=(256, 256),
        threshold_bp=5000,
    )
    assert result.cloud_mask[0, 0, 0] == 0
    assert result.validity_mask[0, 0, 0] == 0
    assert result.cloud_mask[0, 1, 1] == 255


def test_loss_all_invalid_batch_is_zero_and_does_not_step():
    logits = torch.zeros((1, 2, 4, 4), requires_grad=True)
    target = torch.full((1, 4, 4), 255, dtype=torch.long)
    validity = torch.zeros((1, 4, 4), dtype=torch.bool)
    loss, parts = masked_segmentation_loss(logits, target, validity_mask=validity)
    assert float(loss.detach()) == 0.0
    assert float(parts["soft_dice"].detach()) == 0.0
    loss.backward()
    assert float(logits.grad.abs().sum()) == 0.0


def test_cross_entropy_excludes_pixels_marked_invalid_by_validity_mask():
    logits = torch.tensor([[[[3.0, -2.0]], [[-3.0, 2.0]]]])
    validity = torch.tensor([[[False, True]]])
    clear_invalid = torch.tensor([[[0, 1]]])
    cloud_invalid = torch.tensor([[[1, 1]]])
    clear_loss, _ = masked_segmentation_loss(
        logits,
        clear_invalid,
        validity_mask=validity,
        dice_weight=0.0,
    )
    cloud_loss, _ = masked_segmentation_loss(
        logits,
        cloud_invalid,
        validity_mask=validity,
        dice_weight=0.0,
    )
    assert torch.equal(clear_loss, cloud_loss)


def test_masked_loss_upsamples_logits_to_native_target_size():
    logits = torch.zeros((1, 2, 75, 95), requires_grad=True)
    target = torch.zeros((1, 300, 380), dtype=torch.long)
    target[:, 80:220, 120:300] = 1
    validity = torch.ones((1, 300, 380), dtype=torch.bool)
    loss, _ = masked_segmentation_loss(logits, target, validity_mask=validity)
    assert torch.isfinite(loss)
    loss.backward()
    assert float(logits.grad.abs().sum()) > 0.0


def test_train_epoch_accepts_native_size_targets():
    model = torch.nn.Conv2d(3, 2, kernel_size=4, stride=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = {
        "image": torch.zeros((1, 3, 300, 380), dtype=torch.float32),
        "mask": torch.zeros((1, 300, 380), dtype=torch.long),
        "validity_mask": torch.ones((1, 300, 380), dtype=torch.bool),
    }
    metrics = train_one_epoch(
        model,
        [batch],
        optimizer,
        torch.device("cpu"),
        SegmentationTrainingConfig(batch_size=1),
    )
    assert metrics["optimizer_steps"] == 1
    assert metrics["valid_pixels"] == 300 * 380


def test_segmentation_runtime_stitches_edge_and_uses_valid_pixel_denominator(tmp_path):
    source = tmp_path / "scene.tif"
    tifffile.imwrite(source, np.zeros((300, 300, 3), dtype=np.uint16), metadata={"axes": "YXC"}, compression=None)
    sidecar = _sidecar(source, input_spec_id="segformer-rgb-dtype-range-v1")
    logits = np.zeros((1, 2, 64, 64), dtype=np.float32)
    logits[:, 1] = 10.0
    runtime = _runtime(logits)
    with open_memmap_scene(source, sidecar) as scene:
        result = infer_region(scene, ROI(0, 0, 300, 300), runtime, _config())
        assert result["patch_count"] == 4
        assert result["analyzed_area"] == 90000
        assert result["pixel_cloud_ratio_bp"] == 10000
        assert result["valid_pixel_ratio_bp"] == 10000
        assert np.all(result["validity_mask"] == 1)
        summary = build_products(
            result,
            scene,
            tmp_path / "products",
            ProductRef(1, 1, 1),
            RequestKey(1, 1),
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        )
    product_dir = Path(summary["product_directory"])
    assert (product_dir / "cloud_mask.tif").is_file()
    assert (product_dir / "validity_mask.tif").is_file()
    manifest = json.loads((product_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["model_task"] == "semantic_cloud_segmentation"
    assert manifest["pixel_cloud_ratio_bp"] == 10000
    assert {item["path"] for item in manifest["artifacts"]} >= {"cloud_mask.tif", "validity_mask.tif"}
    ProductManifest.from_bytes((product_dir / "manifest.json").read_bytes())


@pytest.mark.parametrize(
    ("model_threshold_bp", "coverage_limit_bp"),
    ((4900, 6000), (5000, 8000)),
)
def test_segmentation_runtime_rejects_mutable_decision_values(
    tmp_path,
    model_threshold_bp,
    coverage_limit_bp,
):
    source = tmp_path / "scene.tif"
    tifffile.imwrite(source, np.zeros((256, 256, 3), dtype=np.uint16), metadata={"axes": "YXC"})
    sidecar = _sidecar(source, input_spec_id="segformer-rgb-dtype-range-v1")
    runtime = _runtime(np.zeros((1, 2, 64, 64), dtype=np.float32))
    with open_memmap_scene(source, sidecar) as scene:
        with pytest.raises(ValueError, match="DECISION_SPEC_MISMATCH"):
            infer_region(
                scene,
                ROI(0, 0, 256, 256),
                runtime,
                _config(
                    model_threshold_bp=model_threshold_bp,
                    coverage_limit_bp=coverage_limit_bp,
                ),
            )


def test_segmentation_dataset_returns_ignore_target_for_padding(tmp_path):
    for directory in ("clear", "masks"):
        (tmp_path / directory).mkdir()
    np.save(tmp_path / "clear" / "scene_a_p0.npy", np.zeros((300, 300, 3), dtype=np.uint16))
    np.save(tmp_path / "masks" / "scene_a_p0.npy", np.zeros((300, 300), dtype=np.uint8))
    dataset = SegmentationDataset(tmp_path, is_train=False)
    assert len(dataset) == 4
    sample = dataset[3]
    assert sample["image"].shape == (3, 256, 256)
    assert sample["mask"].shape == (256, 256)
    assert sample["scene_id"] == "scene_a"
    assert int((sample["mask"] == 255).sum()) > 0
    assert int(sample["validity_mask"].sum()) == 44 * 44


def test_native_dataset_and_model_keep_non_square_source_dimensions(tmp_path):
    for directory in ("clear", "masks", "validity"):
        (tmp_path / directory).mkdir()
    image = np.zeros((300, 380, 3), dtype=np.uint16)
    mask = np.zeros((300, 380), dtype=np.uint8)
    mask[80:220, 120:300] = 1
    validity = np.ones((300, 380), dtype=np.uint8)
    np.save(tmp_path / "clear" / "scene_a_p0.npy", image)
    np.save(tmp_path / "masks" / "scene_a_p0.npy", mask)
    np.save(tmp_path / "validity" / "scene_a_p0.npy", validity)

    dataset = SegmentationDataset(tmp_path, is_train=False, preserve_native_size=True)
    sample = dataset[0]
    assert sample["image"].shape == (3, 300, 380)
    assert sample["mask"].shape == (300, 380)
    assert sample["tile_coordinates"] == (0, 0, 380, 300)

    model = get_segformer_b0().eval()
    with torch.inference_mode():
        logits = model(sample["image"].unsqueeze(0))
    assert logits.shape == (1, 2, 75, 95)


def test_native_preprocess_saves_one_full_scene_pair(tmp_path):
    raw = tmp_path / "raw"
    output = tmp_path / "processed"
    image = np.arange(300 * 380, dtype=np.uint16).reshape(300, 380)
    ground_truth = np.zeros((300, 380), dtype=np.uint8)
    ground_truth[80:220, 120:300] = 255
    scene_files = {}
    for channel in ("red", "green", "blue"):
        path = raw / f"train_{channel}" / f"{channel}_scene_a.TIF"
        path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(path, image)
        scene_files[channel] = path
    gt_path = raw / "train_gt" / "gt_scene_a.TIF"
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(gt_path, ground_truth)
    scene_files["gt"] = gt_path

    assert process_scene(
        "scene_a",
        raw,
        output,
        patch_size=None,
        channels=3,
        scene_files=scene_files,
    ) == 1
    saved = np.load(output / "cloud" / "scene_a_p0.npy")
    saved_mask = np.load(output / "masks" / "scene_a_p0.npy")
    assert saved.shape == (300, 380, 3)
    assert saved_mask.shape == (300, 380)


def test_evaluation_probability_helper_matches_runtime_postprocess():
    logits = np.zeros((1, 2, 64, 64), dtype=np.float32)
    logits[:, 1, :, :32] = 4.0
    logits[:, 1, :, 32:] = -4.0
    validity = np.ones((1, 256, 256), dtype=bool)
    runtime = postprocess_segmentation_logits(
        logits,
        validity,
        target_size=(256, 256),
        threshold_bp=5000,
    )
    evaluation = cloud_probabilities_from_logits(torch.from_numpy(logits), (256, 256))[:, 1].numpy()
    np.testing.assert_allclose(evaluation, runtime.cloud_probability)


def test_evaluator_aggregates_native_scenes_with_different_spatial_sizes():
    model = torch.nn.Conv2d(3, 2, kernel_size=4, stride=4)
    loader = [
        {
            "image": torch.zeros((1, 3, 300, 380), dtype=torch.float32),
            "mask": torch.zeros((1, 300, 380), dtype=torch.long),
            "validity_mask": torch.ones((1, 300, 380), dtype=torch.bool),
            "scene_id": ["scene_a"],
        },
        {
            "image": torch.zeros((1, 3, 320, 384), dtype=torch.float32),
            "mask": torch.zeros((1, 320, 384), dtype=torch.long),
            "validity_mask": torch.ones((1, 320, 384), dtype=torch.bool),
            "scene_id": ["scene_b"],
        },
    ]
    report = evaluate_loader(model, loader, device=torch.device("cpu"), threshold_bp=5000)
    assert report["macro_scene_metrics"]["scene_count"] == 2
    assert report["coverage_metrics"]["scene_count"] == 2
    assert report["scene_bootstrap"]["cloud_dice"]["samples"] == 1000
    assert report["valid_pixel_ratio"] == 1.0


def test_validation_calibration_records_scene_level_evidence():
    probabilities = np.array(
        [
            [[0.30, 0.70], [0.30, 0.70]],
            [[0.30, 0.70], [0.30, 0.70]],
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
    validities = np.ones_like(targets, dtype=bool)
    report = calibrate_validation_predictions(
        probabilities,
        targets,
        validities,
        ["scene_a", "scene_b"],
        candidates_bp=(2500, 5000, 7500),
        max_false_clear_rate=0.0,
        bootstrap_samples=32,
    )
    assert report["threshold_bp"] == 5000
    assert report["threshold_selection"]["dataset_role"] == "validation"
    assert report["threshold_selection"]["selection_metric"] == "validation-false-clear-constrained-dice"
    assert report["coverage_metrics"]["coverage_mae_bp"] == 0.0
    assert len(report["scene_metrics"]) == 2
    assert report["scene_bootstrap"]["cloud_iou"]["samples"] == 32
