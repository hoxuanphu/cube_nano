import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from decision_policy import DecisionPolicy  # noqa: E402
from inference_adapter import InferenceAdapter  # noqa: E402
from patch_result_writer import (  # noqa: E402
    PatchResultWriter,
    open_patch_results,
)
from preprocessing import (  # noqa: E402
    CalibrationBundle,
    CaptureManifest,
    ComputeProfile,
    FailureReason,
    PreprocessFailure,
    PreprocessRequest,
    PreprocessingProfile,
    TrustPolicy,
    preprocess_capture,
)
from preprocessed_inference import (  # noqa: E402
    InferenceProcessRequest,
    PreprocessedInference,
    process_capture,
)


def _request(tmp_path, shape=(5, 6)):
    source = tmp_path / "source.npy"
    image = np.arange(shape[0] * shape[1] * 3, dtype=np.uint16).reshape(*shape, 3)
    np.save(source, image)
    profile = PreprocessingProfile.identity_fixture(shape=shape)
    calibration = CalibrationBundle.identity_fixture()
    capture = CaptureManifest(
        capture_id="p5-capture",
        completion_marker=True,
        source_fingerprint=hashlib.sha256(source.read_bytes()).hexdigest(),
        sensor_product="fixture-sensor",
        dimensions=(*shape, 3),
        source_layout="YXC",
        source_dtype="uint16",
        source_byte_order="native",
        channel_schema=("red", "green", "blue"),
        nodata_semantics={"kind": "none"},
        source_path=str(source),
        preprocessing_profile_digest=profile.fingerprint,
        calibration_digest=calibration.fingerprint,
    )
    compute = ComputeProfile(
        compute_profile_id="p5-cpu",
        profile_version="1",
        backend="cpu",
        ram_budget_bytes=8_000_000,
        disk_budget_bytes=8_000_000,
        os_obc_reserve_bytes=1_000,
        safety_margin_bytes=100,
        max_strip_rows=3,
        queue_depth=1,
        inflight_strips=1,
        temporary_directory=str(tmp_path),
        thermal_policy="fixture",
    )
    return PreprocessRequest(
        source=source,
        capture_manifest=capture,
        calibration_bundle=calibration,
        preprocessing_profile=profile,
        compute_profile=compute,
        output_artifact_target=tmp_path / "preprocessed.artifact",
        run_id="p5-run",
        trust_policy=TrustPolicy.development(),
    )


def _model_profile():
    from preprocessing import ModelCompatibilityProfile

    return ModelCompatibilityProfile(
        profile_id="fixture-model",
        profile_version="1",
        model_fingerprint="a" * 64,
        required_band_order=("blue", "red", "green"),
        tensor_layout="NCHW",
        tensor_dtype="float32",
        patch_size=(4, 4),
        batch_size=2,
        padding_policy="constant",
        normalization={"id": "identity-v1", "kind": "identity"},
        runtime_fingerprint="b" * 64,
    )


def test_adapter_reorders_bands_pads_edges_and_runs_after_gate(tmp_path):
    artifact = preprocess_capture(_request(tmp_path))
    assert not isinstance(artifact, PreprocessFailure)
    model = _model_profile()
    calls = []

    class FakeEngine:
        def infer_batch(self, batch):
            calls.append(np.array(batch, copy=True))
            probability = np.clip(batch.mean(axis=(1, 2, 3)) / 1000.0, 0.0, 1.0).astype(np.float32)
            return probability > 0.5, probability

    adapter = InferenceAdapter(
        model,
        engine_factory=lambda: FakeEngine(),
        min_valid_fraction=0.0,
    )
    result = adapter.run(artifact)
    assert result.status == "complete"
    assert result.processed_count == 4
    assert result.invalid_count == 0
    assert len(calls) == 2
    assert calls[0].shape == (2, 3, 4, 4)
    # First channel is blue, not red; source values are not min/max-scaled by
    # the identity normalization.
    assert calls[0][0, 0, 0, 0] == 2
    # Edge padding is invalid and therefore does not become a clear/cloud
    # classification; the record still carries its validity fraction.
    edge = result.records[-1]
    assert edge["valid_fraction"] == pytest.approx(2 / 16)


def test_compatibility_failure_does_not_create_engine(tmp_path):
    artifact = preprocess_capture(_request(tmp_path))
    assert not isinstance(artifact, PreprocessFailure)
    model = _model_profile()
    created = []
    bad_model = type(model)(
        **{
            **model.to_mapping(include_digest=False),
            "required_band_order": ("nir", "red", "green"),
        }
    )
    adapter = InferenceAdapter(bad_model, engine_factory=lambda: created.append(True))
    result = adapter.run(artifact)
    assert isinstance(result, PreprocessFailure)
    assert result.reason_code == FailureReason.SCHEMA_MISMATCH.value
    assert created == []


def test_raw_and_artifact_modes_converge_without_private_reader_imports(tmp_path):
    request = _request(tmp_path)
    artifact = preprocess_capture(request)
    assert not isinstance(artifact, PreprocessFailure)
    from dataclasses import replace

    raw_request = replace(request, output_artifact_target=tmp_path / "raw-preprocessed.artifact")
    adapter = InferenceAdapter(_model_profile(), engine=type("Engine", (), {
        "infer_batch": lambda self, batch: (np.zeros(len(batch), dtype=bool), np.zeros(len(batch), dtype=np.float32)),
    })())
    inference = PreprocessedInference(adapter)
    raw_result = process_capture(InferenceProcessRequest(preprocessing=raw_request), inference=inference)
    artifact_result = process_capture(
        InferenceProcessRequest(
            artifact=__import__("preprocessing").ArtifactOpenRequest(
                artifact_path=artifact.artifact_path,
                expected_source_fingerprint=artifact.manifest.source_fingerprint,
                expected_profile_fingerprint=artifact.manifest.profile_fingerprint,
                expected_calibration_fingerprint=request.calibration_bundle.fingerprint,
                trust_policy=TrustPolicy.development(),
                compute_profile=request.compute_profile,
                run_id="artifact-open",
            )
        ),
        inference=inference,
    )
    assert raw_result.processed_count == artifact_result.processed_count
    tree = ast.parse((SRC / "preprocessed_inference.py").read_text(encoding="utf-8"))
    imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not any(name in " ".join(imports) for name in ("tiff_reader", "PIL", "tifffile", "rasterio", "source_reader"))


def _record(row, col, *, status="valid", cloud="clear", valid_fraction=1.0, engine="c" * 64):
    return {
        "patch_row": row,
        "patch_col": col,
        "model_window": (row * 2, row * 2 + 2, col * 2, col * 2 + 2),
        "valid_fraction": valid_fraction,
        "validity_reason_summary": {},
        "cloud_probability": 0.9 if cloud == "cloud" else 0.1,
        "cloud_label": cloud if status == "valid" else None,
        "inference_status": status,
        "source_mapping_ref": "mapping.npy",
        "preprocessing_profile_id": "profile.fixture",
        "engine_fingerprint": engine,
        "timestamp": "2026-07-24T00:00:00Z",
        "georef_valid": True,
    }


def test_patch_result_writer_resume_checksum_and_decision(tmp_path):
    target = tmp_path / "capture.patch-results"
    kwargs = dict(
        capture_id="capture-1",
        source_fingerprint="d" * 64,
        preprocessing_profile_id="profile.fixture",
        engine_fingerprint="c" * 64,
        patch_grid_shape=(2, 2),
        patch_size=(2, 2),
    )
    writer = PatchResultWriter(target, **kwargs)
    writer.append(_record(0, 0))
    writer.append(_record(0, 1, cloud="cloud"))
    writer.abort()
    resumed = PatchResultWriter(target, resume=True, **kwargs)
    resumed.append(_record(1, 0))
    resumed.append(_record(1, 1))
    artifact = resumed.finalize()
    opened = open_patch_results(target)
    assert opened.fingerprint == artifact.fingerprint
    decision = DecisionPolicy(cloud_coverage_threshold=0.60).evaluate(opened)
    assert decision.decision == "KEEP_FOR_DOWNLINK"
    assert decision.cloud_coverage == pytest.approx(0.25)


def test_patch_result_duplicate_missing_checksum_and_invalid_status_are_fail_closed(tmp_path):
    target = tmp_path / "results"
    kwargs = dict(
        capture_id="capture-2",
        source_fingerprint="e" * 64,
        preprocessing_profile_id="profile.fixture",
        engine_fingerprint="f" * 64,
        patch_grid_shape=(1, 2),
        patch_size=(2, 2),
    )
    writer = PatchResultWriter(target, **kwargs)
    writer.append(_record(0, 0, engine="f" * 64))
    with pytest.raises(Exception) as duplicate:
        writer.append(_record(0, 0, engine="f" * 64))
    assert getattr(duplicate.value, "reason_code", "") == FailureReason.PATCH_RESULT_DUPLICATE.value
    writer.abort()
    resumed = PatchResultWriter(target, resume=True, **kwargs)
    resumed.append(_record(0, 1, status="invalid", engine="f" * 64))
    artifact = resumed.finalize()
    decision = DecisionPolicy().evaluate(artifact)
    assert decision.decision == "RETAIN_FOR_GROUND"
    (target / "records.jsonl").write_text((target / "records.jsonl").read_text() + "\n", encoding="utf-8")
    invalid = open_patch_results(target)
    assert isinstance(invalid, PreprocessFailure)
    assert invalid.reason_code == FailureReason.PATCH_RESULT_CHECKSUM_MISMATCH.value
    # A checksum-restored but incomplete artifact remains fail-closed.
    (target / "records.jsonl").write_bytes((target / "records.jsonl").read_bytes()[:-1])
    (target / "manifest.json").write_text(json.dumps(dict(artifact.manifest)), encoding="utf-8")
    opened = open_patch_results(target)
    if not isinstance(opened, PreprocessFailure):
        decision = DecisionPolicy().evaluate(opened)
        assert decision.decision == "RETAIN_FOR_GROUND"
