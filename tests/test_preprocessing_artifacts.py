import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from preprocessing import (  # noqa: E402
    ArtifactOpenRequest,
    CalibrationBundle,
    CaptureManifest,
    ComputeProfile,
    FailureReason,
    PreprocessFailure,
    PreprocessRequest,
    PreprocessingProfile,
    TrustPolicy,
    open_preprocessed_artifact,
    preprocess_capture,
)


def _make_request(tmp_path: Path, *, target_name: str = "result.artifact"):
    source = tmp_path / "capture.npy"
    image = np.arange(4 * 5 * 3, dtype=np.uint16).reshape(4, 5, 3)
    np.save(source, image)
    profile = PreprocessingProfile.identity_fixture(shape=(4, 5))
    calibration = CalibrationBundle.identity_fixture()
    capture = CaptureManifest(
        capture_id="capture-artifact",
        completion_marker=True,
        source_fingerprint=hashlib.sha256(source.read_bytes()).hexdigest(),
        sensor_product="fixture-sensor",
        dimensions=(4, 5, 3),
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
        compute_profile_id="fixture-cpu",
        profile_version="1",
        backend="cpu",
        ram_budget_bytes=2_000_000,
        disk_budget_bytes=2_000_000,
        os_obc_reserve_bytes=1_000,
        safety_margin_bytes=100,
        max_strip_rows=2,
        queue_depth=1,
        inflight_strips=1,
        temporary_directory=str(tmp_path),
        thermal_policy="fixture-unbounded",
    )
    request = PreprocessRequest(
        source=source,
        capture_manifest=capture,
        calibration_bundle=calibration,
        preprocessing_profile=profile,
        compute_profile=compute,
        output_artifact_target=tmp_path / target_name,
        run_id="artifact-run",
        trust_policy=TrustPolicy.development(),
    )
    return request, source, capture, profile, calibration, compute


def _open_request(path: Path, capture, profile, calibration, compute, *, source=None):
    return ArtifactOpenRequest(
        artifact_path=path,
        expected_source_fingerprint=source or capture.source_fingerprint,
        expected_profile_fingerprint=profile.fingerprint,
        expected_calibration_fingerprint=calibration.fingerprint,
        trust_policy=TrustPolicy.development(),
        compute_profile=compute,
        run_id="artifact-open",
    )


def _run_request(tmp_path: Path):
    request, source, capture, profile, calibration, compute = _make_request(tmp_path)
    artifact = preprocess_capture(request)
    assert not isinstance(artifact, PreprocessFailure)
    return artifact, source, capture, profile, calibration, compute


def test_artifact_metadata_stays_preprocessing_only(tmp_path):
    artifact, _, _, _, _, _ = _run_request(tmp_path)
    preprocess = json.loads((artifact.artifact_path / "preprocess.json").read_text(encoding="utf-8"))
    output = json.loads((artifact.artifact_path / "output.json").read_text(encoding="utf-8"))
    encoded = json.dumps({"preprocess": preprocess, "output": output}, sort_keys=True)
    for forbidden in ("patch_size", "normalization", "tensor_dtype", "engine", "model_fingerprint"):
        assert forbidden not in encoded
    assert preprocess["mapping"]["direction"] == "target_to_source"
    assert preprocess["source_fingerprint"]
    assert preprocess["output_fingerprint"]


def test_checksum_mismatch_is_rejected_and_source_is_retained(tmp_path):
    artifact, source, capture, profile, calibration, compute = _run_request(tmp_path)
    raster = artifact.artifact_path / "model-grid.tif"
    payload = bytearray(raster.read_bytes())
    payload[-1] ^= 0x01
    raster.write_bytes(payload)

    result = open_preprocessed_artifact(
        _open_request(artifact.artifact_path, capture, profile, calibration, compute)
    )
    assert isinstance(result, PreprocessFailure)
    assert result.reason_code == FailureReason.ARTIFACT_CHECKSUM_MISMATCH.value
    assert source.is_file()


def test_incomplete_or_non_complete_manifest_cannot_open(tmp_path):
    request, source, capture, profile, calibration, compute = _make_request(tmp_path)
    artifact_dir = request.artifact_target
    artifact_dir.mkdir()
    result = open_preprocessed_artifact(
        _open_request(artifact_dir, capture, profile, calibration, compute)
    )
    assert isinstance(result, PreprocessFailure)
    assert result.reason_code == FailureReason.ARTIFACT_INCOMPLETE.value
    assert source.is_file()


def test_manifest_status_is_fail_closed_even_when_payload_files_exist(tmp_path):
    artifact, _, capture, profile, calibration, compute = _run_request(tmp_path)
    manifest_path = artifact.artifact_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "writing"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = open_preprocessed_artifact(
        _open_request(artifact.artifact_path, capture, profile, calibration, compute)
    )
    assert isinstance(result, PreprocessFailure)
    assert result.reason_code == FailureReason.ARTIFACT_INCOMPLETE.value


def test_artifact_expected_fingerprint_mismatch_is_rejected(tmp_path):
    artifact, _, capture, profile, calibration, compute = _run_request(tmp_path)
    result = open_preprocessed_artifact(
        _open_request(
            artifact.artifact_path,
            capture,
            profile,
            calibration,
            compute,
            source="0" * 64,
        )
    )
    assert isinstance(result, PreprocessFailure)
    assert result.reason_code == FailureReason.FINGERPRINT_LINKAGE_MISMATCH.value


def test_atomic_publish_io_failure_leaves_no_target_or_staging(tmp_path, monkeypatch):
    request, source, _, _, _, _ = _make_request(tmp_path)

    def fail_replace(*args, **kwargs):
        raise OSError("injected rename failure")

    monkeypatch.setattr("preprocessing.artifact_writer.os.replace", fail_replace)
    result = preprocess_capture(request)

    assert isinstance(result, PreprocessFailure)
    assert result.reason_code == FailureReason.IO_ERROR.value
    assert result.safe_action.value == "RETAIN_FOR_GROUND"
    assert source.is_file()
    assert not request.artifact_target.exists()
    assert not list(tmp_path.glob(".result.artifact.staging-*"))


def test_disk_full_during_finalize_is_not_published(tmp_path, monkeypatch):
    request, source, _, _, _, _ = _make_request(tmp_path)

    def fail_close(_writer):
        raise OSError("simulated disk full")

    monkeypatch.setattr(
        "preprocessing.artifact_writer.ArtifactWriter._close_payload_arrays",
        fail_close,
    )
    result = preprocess_capture(request)

    assert isinstance(result, PreprocessFailure)
    assert result.reason_code == FailureReason.IO_ERROR.value
    assert source.is_file()
    assert not request.artifact_target.exists()
    assert not list(tmp_path.glob(".result.artifact.staging-*"))
