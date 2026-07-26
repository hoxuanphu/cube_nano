import hashlib
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from preprocessing import (  # noqa: E402
    ArtifactOpenRequest,
    CalibrationBundle,
    CaptureManifest,
    ComputeProfile,
    ContractResolver,
    FailureReason,
    PreprocessError,
    PreprocessFailure,
    PreprocessRequest,
    PreprocessingProfile,
    ResourceAdmission,
    RunState,
    StaticResourceProbe,
    TrustMetadata,
    TrustPolicy,
    attach_trust,
    open_preprocessed_artifact,
    preprocess_capture,
)


NOW = "2026-07-24T00:00:00Z"
EXPIRES = "2026-07-25T00:00:00Z"


def _make_request(tmp_path, *, trust_policy=None, compute=None, source_value=None, verify_source=True):
    source = tmp_path / "capture.npy"
    image = np.arange(4 * 5 * 3, dtype=np.uint16).reshape(4, 5, 3)
    np.save(source, image)
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    profile = PreprocessingProfile.identity_fixture(shape=(4, 5))
    calibration = CalibrationBundle.identity_fixture()
    capture = CaptureManifest(
        capture_id="capture-1",
        completion_marker=True,
        source_fingerprint=source_digest,
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
    compute = compute or ComputeProfile(
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
    return PreprocessRequest(
        source=source_value if source_value is not None else source,
        capture_manifest=capture,
        calibration_bundle=calibration,
        preprocessing_profile=profile,
        compute_profile=compute,
        output_artifact_target=tmp_path / "result.artifact",
        run_id="run-1",
        trust_policy=trust_policy or TrustPolicy.development(),
        verify_source_fingerprint=verify_source,
    ), source, capture, profile, calibration, compute


def _trusted_request(tmp_path):
    base_request, source, capture, profile, calibration, compute = _make_request(tmp_path)
    key = b"fixture-secret"
    profile = attach_trust(
        profile,
        issuer="ground",
        key_id="key-1",
        key=key,
        generation=1,
        issued_at=NOW,
        expires_at=EXPIRES,
    )
    calibration = attach_trust(
        calibration,
        issuer="ground",
        key_id="key-1",
        key=key,
        generation=1,
        issued_at=NOW,
        expires_at=EXPIRES,
    )
    capture = replace(
        capture,
        preprocessing_profile_digest=profile.fingerprint,
        calibration_digest=calibration.fingerprint,
        digest=None,
    )
    capture = attach_trust(
        capture,
        issuer="ground",
        key_id="key-1",
        key=key,
        generation=1,
        issued_at=NOW,
        expires_at=EXPIRES,
    )
    compute = attach_trust(
        compute,
        issuer="ground",
        key_id="key-1",
        key=key,
        generation=1,
        issued_at=NOW,
        expires_at=EXPIRES,
    )
    policy = TrustPolicy(
        require_signature=True,
        trusted_issuers=("ground",),
        trusted_keys={"key-1": key},
        expected_generations={"capture": 1, "profile": 1, "calibration": 1, "compute": 1},
        now=NOW,
    )
    request = replace(
        base_request,
        capture_manifest=capture,
        calibration_bundle=calibration,
        preprocessing_profile=profile,
        compute_profile=compute,
        trust_policy=policy,
    )
    return request, source, capture, profile, calibration, compute


def test_trust_gate_verifies_signature_generation_and_expiry(tmp_path):
    request, _, _, profile, _, _ = _trusted_request(tmp_path)
    resolved = ContractResolver(trust_policy=request.trust_policy).resolve(request)
    assert resolved.preprocessing_profile.fingerprint == profile.fingerprint

    tampered_trust = replace(profile.trust, signature="0" * 64)
    tampered = replace(profile, trust=tampered_trust, digest=None)
    with pytest.raises(PreprocessError) as error:
        ContractResolver(trust_policy=request.trust_policy).resolve(replace(request, preprocessing_profile=tampered))
    assert error.value.state == RunState.UNTRUSTED_ARTIFACT
    assert error.value.reason_code == FailureReason.SIGNATURE_INVALID.value

    expired = replace(request.trust_policy, now="2026-07-26T00:00:00Z")
    with pytest.raises(PreprocessError) as error:
        ContractResolver(trust_policy=expired).resolve(request)
    assert error.value.reason_code == FailureReason.TRUST_EXPIRED.value


def test_untrusted_input_is_rejected_before_source_stat_or_hash(tmp_path):
    request, _, _, _, _, _ = _make_request(
        tmp_path,
        trust_policy=TrustPolicy(),
        source_value=tmp_path / "does-not-exist.bin",
    )
    with pytest.raises(PreprocessError) as error:
        ContractResolver().resolve(request)
    assert error.value.state == RunState.UNTRUSTED_ARTIFACT
    assert error.value.reason_code == FailureReason.TRUST_REJECTED.value


def test_source_fingerprint_and_calibration_linkage_are_checked_after_trust(tmp_path):
    request, source, _, _, _, _ = _make_request(tmp_path)
    source.write_bytes(b"changed")
    with pytest.raises(PreprocessError) as error:
        ContractResolver(trust_policy=TrustPolicy.development()).resolve(request)
    assert error.value.reason_code == FailureReason.SOURCE_FINGERPRINT_MISMATCH.value


def test_two_tier_resource_admission_is_conservative_and_fail_closed(tmp_path):
    request, _, _, _, _, compute = _make_request(tmp_path)
    resolved = ContractResolver(trust_policy=TrustPolicy.development()).resolve(request)
    admission = ResourceAdmission(compute, probe=StaticResourceProbe(available_ram_bytes=10_000_000, free_disk_bytes=10_000_000))
    receipt = admission.preflight(resolved)
    assert receipt.estimate.ram_peak_bytes <= compute.ram_budget_bytes
    assert receipt.estimate.disk_required_bytes <= compute.disk_budget_bytes

    runtime = admission.runtime_check(
        receipt,
        actual_peak_ram_bytes=receipt.estimate.ram_peak_bytes,
        free_disk_bytes=receipt.estimate.disk_required_bytes,
    )
    assert runtime.stage == "runtime"

    with pytest.raises(PreprocessError) as error:
        admission.before_publish(
            receipt,
            actual_peak_ram_bytes=compute.ram_budget_bytes + 1,
            free_disk_bytes=compute.disk_budget_bytes,
        )
    assert error.value.reason_code == FailureReason.RESOURCE_RUNTIME.value

    with pytest.raises(PreprocessError) as error:
        admission.runtime_check(
            receipt,
            actual_peak_ram_bytes=receipt.estimate.ram_peak_bytes,
            free_disk_bytes=receipt.estimate.disk_required_bytes - 1,
        )
    assert error.value.reason_code == FailureReason.RESOURCE_RUNTIME.value


def test_runtime_disk_check_applies_preflight_staging_multiplier(tmp_path):
    request, _, _, _, _, compute = _make_request(tmp_path)
    compute = replace(compute, artifact_staging_multiplier=2.0, digest=None)
    request = replace(request, compute_profile=compute)
    resolved = ContractResolver(trust_policy=TrustPolicy.development()).resolve(request)
    admission = ResourceAdmission(compute)
    receipt = admission.preflight(resolved)
    estimate = receipt.estimate
    headroom = estimate.disk_components["checksum_headroom"] + estimate.disk_components["safety_margin"]
    old_runtime_requirement = (
        estimate.disk_components["temporary"] + estimate.output_bytes + headroom
    )
    assert estimate.disk_components["artifact_staging"] == 2 * estimate.output_bytes
    assert old_runtime_requirement < estimate.disk_required_bytes

    with pytest.raises(PreprocessError) as error:
        admission.runtime_check(
            receipt,
            actual_peak_ram_bytes=estimate.ram_peak_bytes,
            free_disk_bytes=old_runtime_requirement,
        )
    assert error.value.reason_code == FailureReason.RESOURCE_RUNTIME.value

    runtime = admission.runtime_check(
        receipt,
        actual_peak_ram_bytes=estimate.ram_peak_bytes,
        free_disk_bytes=estimate.disk_required_bytes,
    )
    assert runtime.actual_disk_required_bytes == estimate.disk_required_bytes


def test_compressed_source_requires_proven_bound(tmp_path):
    request, _, _, _, _, compute = _make_request(tmp_path)
    compressed_source = replace(
        request.source,
        compressed=True,
        full_decode_bytes=None,
    )
    request = replace(request, source=compressed_source)
    resolved = ContractResolver(trust_policy=TrustPolicy.development()).resolve(request)
    with pytest.raises(PreprocessError) as error:
        ResourceAdmission(compute).preflight(resolved)
    assert error.value.reason_code == FailureReason.CODEC_UNAVAILABLE.value


def test_public_facade_returns_typed_failure_and_state_history(tmp_path):
    request, source, capture, profile, calibration, compute = _make_request(tmp_path)
    result = preprocess_capture(request)
    assert not isinstance(result, PreprocessFailure)
    assert result.manifest.complete is True
    assert result.artifact_path.is_dir()
    assert set(result.manifest.checksums) == {
        "model-grid.tif",
        "validity.tif",
        "validity-reasons.tif",
        "mapping.npy",
        "output.json",
        "preprocess.json",
    }

    reopened = open_preprocessed_artifact(
        ArtifactOpenRequest(
            artifact_path=result.artifact_path,
            expected_source_fingerprint=capture.source_fingerprint,
            expected_profile_fingerprint=profile.fingerprint,
            expected_calibration_fingerprint=calibration.fingerprint,
            trust_policy=request.trust_policy,
            compute_profile=compute,
            run_id="open-run-1",
        )
    )
    assert not isinstance(reopened, PreprocessFailure)
    with reopened.open() as reader:
        block = reader.read_block(0, 4)
    np.testing.assert_array_equal(block["image"], np.load(source))
    np.testing.assert_array_equal(block["validity_yx"], np.ones((4, 5), dtype=np.uint8))
    np.testing.assert_array_equal(block["validity_reason_yx"], np.zeros((4, 5), dtype=np.uint16))
    expected_mapping = np.stack(
        np.meshgrid(
            np.arange(4, dtype=np.float32) + 0.5,
            np.arange(5, dtype=np.float32) + 0.5,
            indexing="ij",
        ),
        axis=-1,
    )
    np.testing.assert_array_equal(block["mapping_yx"], expected_mapping)


def test_public_facade_does_not_swallow_internal_value_error(tmp_path, monkeypatch):
    request, _, _, _, _, _ = _make_request(tmp_path)

    def raise_invariant_error(self, value):
        raise ValueError("internal invariant failure")

    monkeypatch.setattr("preprocessing.resolver.ContractResolver.resolve", raise_invariant_error)
    with pytest.raises(ValueError, match="internal invariant failure"):
        preprocess_capture(request)
