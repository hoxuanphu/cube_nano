import dataclasses
import hashlib
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from preprocessing import (  # noqa: E402
    ArtifactManifest,
    CalibrationBundle,
    CaptureManifest,
    ComputeProfile,
    ModelCompatibilityProfile,
    PreprocessRequest,
    PreprocessingProfile,
    TrustPolicy,
)


def test_profile_fixtures_are_deterministic_and_model_independent():
    first = PreprocessingProfile.identity_fixture(shape=(4, 5))
    second = PreprocessingProfile.identity_fixture(shape=(4, 5))
    affine = PreprocessingProfile.affine_fixture(shape=(4, 5))

    assert first.fingerprint == second.fingerprint
    assert first.to_mapping() == second.to_mapping()
    assert first.transform_type == "identity"
    assert affine.transform_type == "affine"
    assert first.output_shape == (4, 5, 3)
    assert "normalization" not in first.to_mapping()
    assert "patch_size" not in first.to_mapping()
    assert "tensor_layout" not in first.to_mapping()
    with pytest.raises(TypeError):
        first.source_schema.nodata_semantics["new"] = "value"


def test_model_contract_is_separate_from_preprocessing_contract():
    profile_fields = {field.name for field in dataclasses.fields(PreprocessRequest)}
    assert "engine_input_spec" not in profile_fields
    assert "model_compatibility_profile" not in profile_fields

    model = ModelCompatibilityProfile(
        profile_id="fixture-model",
        profile_version="1",
        model_fingerprint="0" * 64,
        required_band_order=("red", "green", "blue"),
        tensor_layout="NCHW",
        tensor_dtype="float32",
        patch_size=(4, 4),
        batch_size=1,
        padding_policy="reject",
        normalization="dtype-range-v1",
        runtime_fingerprint="1" * 64,
    )
    assert model.fingerprint
    assert model.accepts(PreprocessingProfile.identity_fixture(shape=(4, 4))) == ()


def test_legacy_engine_spec_has_explicit_migration_adapter():
    from input_contract import EngineInputSpec as LegacyEngineInputSpec, legacy_input_spec

    legacy = dataclasses.replace(
        legacy_input_spec(channels=3, patch_size=4),
        engine_digest="a" * 64,
    )
    converted = legacy.to_model_compatibility_profile(runtime_fingerprint="b" * 64)

    assert isinstance(converted, ModelCompatibilityProfile)
    assert converted.profile_id == legacy.input_spec_id
    assert converted.required_band_order == legacy.band_order
    assert converted.patch_size == (4, 4)
    assert converted.tensor_layout == "NCHW"
    assert LegacyEngineInputSpec is not ModelCompatibilityProfile


def test_preprocessing_facade_does_not_export_ambiguous_engine_input_spec():
    import preprocessing

    assert "EngineInputSpec" not in preprocessing.__all__
    assert not hasattr(preprocessing, "EngineInputSpec")


def test_artifact_schema_links_only_spatial_preprocessing_contracts():
    profile = PreprocessingProfile.identity_fixture(shape=(4, 5))
    calibration = CalibrationBundle.identity_fixture()
    source_digest = hashlib.sha256(b"source").hexdigest()
    capture = CaptureManifest(
        capture_id="capture-1",
        completion_marker=True,
        source_fingerprint=source_digest,
        sensor_product="fixture-sensor",
        dimensions=(4, 5, 3),
        source_layout="YXC",
        source_dtype="uint16",
        channel_schema=("red", "green", "blue"),
        preprocessing_profile_digest=profile.fingerprint,
        calibration_digest=calibration.fingerprint,
    )
    manifest = ArtifactManifest.from_contracts(
        artifact_id="capture-1.artifact",
        capture_manifest=capture,
        preprocessing_profile=profile,
        calibration_bundle=calibration,
        mapping_ref="mapping.json",
    )
    assert manifest.profile_fingerprint == profile.fingerprint
    assert manifest.calibration_fingerprint == calibration.fingerprint
    assert "model" not in manifest.provenance


def test_clean_process_import_uses_public_package_without_working_directory(tmp_path):
    command = [
        sys.executable,
        "-c",
        "import preprocessing; print(preprocessing.preprocess_capture.__name__)",
    ]
    environment = {"PYTHONPATH": str(SRC)}
    result = subprocess.run(command, cwd=tmp_path, env=environment, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "preprocess_capture"
