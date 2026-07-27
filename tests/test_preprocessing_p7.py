"""P7 validation campaigns for geometry, validity and fail-closed recovery."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from decision_policy import DecisionPolicy  # noqa: E402
from inference_adapter import InferenceAdapter  # noqa: E402
from patch_result_writer import PatchResultWriter  # noqa: E402
from preprocessing import (  # noqa: E402
    ArtifactOpenRequest,
    CalibrationBundle,
    CaptureManifest,
    ClippingPolicy,
    ComputeProfile,
    FailureReason,
    ModelCompatibilityProfile,
    PreprocessError,
    PreprocessFailure,
    PreprocessRequest,
    PreprocessingProfile,
    ResourceAdmission,
    RunState,
    StaticResourceProbe,
    TargetGrid,
    TrustPolicy,
    open_preprocessed_artifact,
    preprocess_capture,
)
from preprocessing.source_reader import ArraySourceReader, SourceWindow  # noqa: E402
from preprocessing.transform_plan import TransformPlanner  # noqa: E402
from preprocessing.warp_backend import CPUWarpBackend  # noqa: E402


def _profile(*, shape=(4, 5), channels=("red", "green", "blue"), dtype="uint16", **changes):
    profile = PreprocessingProfile.identity_fixture(shape=shape, channels=channels, dtype=dtype)
    if changes:
        profile = replace(profile, **changes, digest=None)
    return profile


def _affine_profile(*, shape=(4, 5), channels=("red", "green", "blue"), dtype="uint16", **changes):
    profile = PreprocessingProfile.affine_fixture(shape=shape, channels=channels, dtype=dtype)
    if changes:
        profile = replace(profile, **changes, digest=None)
    return profile


def _warp(profile, calibration, image, *, row_start=0, row_end=None, reader_kwargs=None):
    reader = ArraySourceReader(image, profile, **(reader_kwargs or {}))
    planner = TransformPlanner(profile, calibration, reader.shape_yxc)
    plan = planner.plan_strip(row_start, row_end or profile.target_grid.rows)
    return CPUWarpBackend(profile).warp(reader.read_window(plan.source_window), plan), plan


@pytest.mark.parametrize("dtype", ["uint8", "uint16", "float32"])
@pytest.mark.parametrize("layout", ["YXC", "CYX"])
def test_p7_golden_identity_preserves_dtype_and_layout(dtype, layout):
    profile = _profile(shape=(3, 4), dtype=dtype, output_layout=layout)
    image = np.arange(3 * 4 * 3, dtype=np.dtype(dtype)).reshape(3, 4, 3)
    result, plan = _warp(profile, CalibrationBundle.identity_fixture(), image)

    expected = image if layout == "YXC" else np.moveaxis(image, -1, 0)
    np.testing.assert_array_equal(result.image, expected)
    assert result.image.dtype == np.dtype(dtype)
    assert result.image.shape == profile.output_shape
    assert result.provenance["profile_fingerprint"] == profile.fingerprint
    assert result.provenance["calibration_fingerprint"] == CalibrationBundle.identity_fixture().fingerprint
    np.testing.assert_allclose(result.mapping_yx[0, 0], (0.5, 0.5))
    assert plan.provenance["target_grid"] == profile.target_grid.to_mapping()


def test_p7_golden_single_band_yx_layout_and_target_grid():
    profile = _profile(shape=(4, 5), channels=("gray",), dtype="uint16")
    profile = replace(
        profile,
        source_schema=replace(profile.source_schema, axes="YX", shape=(4, 5), channels=()),
        output_layout="YX",
        digest=None,
    )
    target = TargetGrid(
        shape=(2, 3),
        extent=(0.0, 0.0, 3.0, 2.0),
        resolution=(1.0, 1.0),
        origin=(0.0, 0.0),
        axes="YX",
        pixel_convention="center",
        spatial_semantics="p7-target-grid",
    )
    profile = replace(profile, target_grid=target, digest=None)
    image = np.arange(4 * 5, dtype=np.uint16).reshape(4, 5)
    result, plan = _warp(profile, CalibrationBundle.identity_fixture(), image)

    np.testing.assert_array_equal(result.image, image[:2, :3])
    assert result.image.shape == (2, 3)
    assert result.image.dtype == np.dtype("uint16")
    np.testing.assert_allclose(
        result.mapping_yx,
        np.array(
            [
                [[0.5, 0.5], [0.5, 1.5], [0.5, 2.5]],
                [[1.5, 0.5], [1.5, 1.5], [1.5, 2.5]],
            ],
            dtype=np.float32,
        ),
    )
    assert plan.provenance["target_grid"]["axes"] == "YX"


@pytest.mark.parametrize("kernel", ["nearest", "bilinear", "bicubic", "support"])
@pytest.mark.parametrize("border_policy", ["invalid", "constant", "edge", "reflect"])
def test_p7_golden_kernel_border_campaign_is_deterministic(kernel, border_policy):
    base = _affine_profile(
        shape=(6, 7),
        channels=("gray",),
        dtype="float32",
        image_kernel=kernel,
        validity_kernel="nearest",
        reason_kernel="nearest",
        border_policy=border_policy,
        support_threshold=0.5,
        halo=replace(_profile(shape=(6, 7), channels=("gray",), dtype="float32").halo, rows=1, cols=1),
    )
    calibration = CalibrationBundle.affine_fixture(offset=(0.75, 0.75))
    image = np.arange(6 * 7, dtype=np.float32).reshape(6, 7, 1)
    first, _ = _warp(base, calibration, image)
    second, _ = _warp(base, calibration, image)

    assert first.image.shape == base.output_shape
    assert first.image.dtype == np.dtype("float32")
    assert np.isfinite(first.image).all()
    np.testing.assert_array_equal(first.image, second.image)
    np.testing.assert_array_equal(first.validity_yx, second.validity_yx)
    np.testing.assert_array_equal(first.validity_reason_yx, second.validity_reason_yx)
    np.testing.assert_array_equal(first.mapping_yx, second.mapping_yx)
    if border_policy in {"invalid", "constant"}:
        assert np.any(first.validity_yx == base.validity_encoding.invalid_value)
    assert np.any((first.validity_reason_yx & base.reason_encoding.bits["border"]) != 0)


def test_p7_golden_affine_bilinear_oracle_and_strip_invariance():
    profile = _affine_profile(
        shape=(2, 2),
        channels=("gray",),
        dtype="float32",
        output_layout="CYX",
        image_kernel="bilinear",
        validity_kernel="nearest",
        reason_kernel="nearest",
        border_policy="edge",
        support_threshold=0.0,
        rounding="none",
    )
    profile = replace(
        profile,
        target_grid=TargetGrid(
            shape=(1, 1),
            extent=(0.0, 0.0, 1.0, 1.0),
            resolution=(1.0, 1.0),
            origin=(0.0, 0.0),
            axes="XY",
            pixel_convention="center",
            spatial_semantics="p7-oracle-grid",
        ),
        digest=None,
    )
    image = np.array([[[0.0], [10.0]], [[20.0], [30.0]]], dtype=np.float32)
    calibration = CalibrationBundle.affine_fixture(offset=(-0.25, -0.25))
    result, plan = _warp(profile, calibration, image)

    # Model-center (0.5, 0.5) maps to source (0.25, 0.25):
    # 0.75*0.75*0 + 0.75*0.25*10 + 0.25*0.75*20 + 0.25*0.25*30 = 7.5.
    assert result.image[0, 0, 0] == pytest.approx(7.5)
    np.testing.assert_allclose(plan.mapping_yx[0, 0], (0.75, 0.75))

    reader = ArraySourceReader(image, profile)
    planner = TransformPlanner(profile, calibration, reader.shape_yxc)
    one = CPUWarpBackend(profile).warp(
        reader.read_window(planner.plan_strip(0, 1).source_window),
        planner.plan_strip(0, 1),
    )
    np.testing.assert_array_equal(one.image, result.image)


@pytest.mark.parametrize(
    ("rounding", "expected"),
    [
        ("nearest_even", 2),
        ("nearest_away", 3),
        ("floor", 2),
        ("ceil", 3),
        ("truncate", 2),
    ],
)
def test_p7_golden_rounding_and_clipping_matrix(rounding, expected):
    base = _profile(shape=(1, 3), channels=("gray",), dtype="float32")
    profile = replace(
        base,
        output_dtype="uint8",
        clipping=ClippingPolicy(mode="range", lower=0, upper=255),
        rounding=rounding,
        digest=None,
    )
    image = np.array([[[2.5], [300.0], [-4.0]]], dtype=np.float32)
    result, _ = _warp(profile, CalibrationBundle.identity_fixture(), image)
    assert result.image.dtype == np.dtype("uint8")
    assert result.image[0, 0, 0] == expected
    assert result.image[0, 1, 0] == 255
    assert result.image[0, 2, 0] == 0


@pytest.mark.parametrize("source_dtype", ["uint8", "uint16", "float32"])
@pytest.mark.parametrize("output_dtype", ["uint8", "uint16", "float32"])
def test_p7_golden_dtype_conversion_is_profile_driven(source_dtype, output_dtype):
    profile = _profile(shape=(2, 3), channels=("gray",), dtype=source_dtype, output_dtype=output_dtype)
    values = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32).reshape(2, 3, 1)
    info = np.iinfo(np.dtype(source_dtype)) if np.issubdtype(np.dtype(source_dtype), np.integer) else None
    image = values.astype(np.dtype(source_dtype))
    if info is not None:
        image = np.mod(image, info.max).astype(np.dtype(source_dtype))
    result, _ = _warp(profile, CalibrationBundle.identity_fixture(), image)

    assert result.image.dtype == np.dtype(output_dtype)
    np.testing.assert_array_equal(result.image[..., 0], image[..., 0].astype(np.dtype(output_dtype)))


def test_p7_lut_without_a_calibration_bundle_is_fail_closed():
    base = _profile(shape=(2, 2), channels=("gray",), dtype="float32")
    profile = replace(
        base,
        profile_id="p7-lut-boundary",
        transform_type="lut",
        calibration_selector=replace(
            base.calibration_selector,
            calibration_id="p7-lut",
            allowed_transform_types=("lut",),
        ),
        digest=None,
    )
    calibration = CalibrationBundle(
        calibration_id="p7-lut",
        version="1",
        sensor_product="fixture-sensor",
        transform_type="lut",
        transform_direction="source_to_model",
        pixel_convention="center",
        parameters={"mapping_ref": "missing-lut-artifact"},
        quality={"max_residual": 0.0},
    )
    image = np.ones((2, 2, 1), dtype=np.float32)
    with pytest.raises(PreprocessError) as error:
        _warp(profile, calibration, image)
    assert error.value.reason_code == FailureReason.CALIBRATION_UNSUPPORTED.value
    assert error.value.state == RunState.CALIBRATION_ERROR


def test_p7_validity_reasons_preserve_multiple_source_causes():
    profile = _affine_profile(
        shape=(3, 3),
        channels=("gray",),
        dtype="uint16",
    )
    profile = replace(
        profile,
        source_schema=replace(profile.source_schema, nodata_semantics={"kind": "value", "value": 0}),
        digest=None,
    )
    image = np.ones((3, 3, 1), dtype=np.uint16)
    image[1, 1, 0] = 0
    missing = np.zeros((3, 3), dtype=bool)
    missing[1, 1] = True
    reader = ArraySourceReader(image, profile, missing_channels=missing)
    block = reader.read_window(SourceWindow(0, 3, 0, 3))
    bits = profile.reason_encoding.bits

    assert block.validity_yx[1, 1] == profile.validity_encoding.invalid_value
    assert block.validity_reason_yx[1, 1] == bits["source_nodata"] | bits["missing_channel"]
    assert np.all(block.validity_reason_yx[image[..., 0] != 0] == 0)


def test_p7_validity_reasons_propagate_geometry_border_and_support():
    profile = _affine_profile(
        shape=(3, 3),
        channels=("gray",),
        dtype="uint16",
        image_kernel="bilinear",
        validity_kernel="nearest",
        reason_kernel="nearest",
        border_policy="invalid",
        support_threshold=0.75,
    )
    calibration = CalibrationBundle.affine_fixture(offset=(0.25, 0.25))
    image = np.full((3, 3, 1), 10, dtype=np.uint16)
    image[1, 1, 0] = 0
    missing = np.zeros((3, 3), dtype=bool)
    missing[1, 2] = True
    profile = replace(
        profile,
        source_schema=replace(profile.source_schema, nodata_semantics={"kind": "value", "value": 0}),
        digest=None,
    )
    result, _ = _warp(profile, calibration, image, reader_kwargs={"missing_channels": missing})
    bits = profile.reason_encoding.bits

    assert np.any((result.validity_reason_yx & bits["border"]) != 0)
    assert np.any((result.validity_reason_yx & bits["outside_mapping"]) != 0)
    assert np.any((result.validity_reason_yx & bits["insufficient_support"]) != 0)
    assert np.any((result.validity_reason_yx & bits["source_nodata"]) != 0)
    assert np.any((result.validity_reason_yx & bits["missing_channel"]) != 0)
    assert np.all(
        (result.validity_yx == profile.validity_encoding.invalid_value)
        | (result.validity_yx == profile.validity_encoding.valid_value)
    )


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_p7_non_finite_policy_is_fail_closed_or_invalid(value):
    image = np.array([[[value]]], dtype=np.float32)
    reject = _profile(shape=(1, 1), channels=("gray",), dtype="float32")
    with pytest.raises(PreprocessError) as rejected:
        _warp(reject, CalibrationBundle.identity_fixture(), image)
    assert getattr(rejected.value, "reason_code", None) == FailureReason.NON_FINITE_OUTPUT.value

    replace_profile = replace(reject, non_finite_policy="replace", digest=None)
    result, _ = _warp(replace_profile, CalibrationBundle.identity_fixture(), image)
    assert result.image[0, 0, 0] == 0
    assert result.validity_yx[0, 0] == replace_profile.validity_encoding.invalid_value
    assert result.validity_reason_yx[0, 0] & replace_profile.reason_encoding.bits["insufficient_support"]


def test_p7_cast_overflow_without_clipping_is_rejected():
    profile = _profile(shape=(1, 1), channels=("gray",), dtype="float32", output_dtype="uint8")
    image = np.array([[[300.0]]], dtype=np.float32)
    with pytest.raises(PreprocessError) as error:
        _warp(profile, CalibrationBundle.identity_fixture(), image)
    assert getattr(error.value, "reason_code", None) == FailureReason.WARP_ERROR.value


def _capture_request(tmp_path, *, profile=None, calibration=None, compute=None, source_descriptor=None):
    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    profile = profile or _profile(shape=(3, 4), dtype="uint16")
    calibration = calibration or CalibrationBundle.identity_fixture()
    source = tmp_path / "p7-source.npy"
    np.save(source, np.arange(3 * 4 * profile.source_schema.channel_count, dtype=np.uint16).reshape(3, 4, -1))
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    capture = CaptureManifest(
        capture_id="p7-capture",
        completion_marker=True,
        source_fingerprint=source_digest,
        sensor_product="fixture-sensor",
        dimensions=(3, 4, profile.source_schema.channel_count),
        source_layout="YXC",
        source_dtype="uint16",
        source_byte_order="native",
        channel_schema=profile.source_schema.channels,
        nodata_semantics=profile.source_schema.nodata_semantics,
        source_path=str(source),
        preprocessing_profile_digest=profile.fingerprint,
        calibration_digest=calibration.fingerprint,
    )
    compute = compute or ComputeProfile(
        compute_profile_id="p7-cpu",
        profile_version="1",
        backend="cpu",
        ram_budget_bytes=8_000_000,
        disk_budget_bytes=8_000_000,
        os_obc_reserve_bytes=1_000,
        safety_margin_bytes=100,
        max_strip_rows=2,
        queue_depth=1,
        inflight_strips=1,
        temporary_directory=str(tmp_path),
        checksum_headroom_bytes=100,
        thermal_policy="fixture",
    )
    request = PreprocessRequest(
        source=source_descriptor if source_descriptor is not None else source,
        capture_manifest=capture,
        calibration_bundle=calibration,
        preprocessing_profile=profile,
        compute_profile=compute,
        output_artifact_target=tmp_path / "p7-artifact",
        run_id="p7-run",
        trust_policy=TrustPolicy.development(),
    )
    return request, source, capture, profile, calibration, compute


def test_p7_fault_injection_calibration_and_resource_admission(tmp_path):
    affine = PreprocessingProfile.affine_fixture(shape=(3, 4), dtype="uint16")
    singular = CalibrationBundle.affine_fixture(matrix=((1.0, 0.0), (2.0, 0.0)))
    request, _, _, _, _, _ = _capture_request(tmp_path, profile=affine, calibration=singular)
    result = preprocess_capture(request)
    assert isinstance(result, PreprocessFailure)
    assert result.state == RunState.CALIBRATION_ERROR
    assert result.reason_code == FailureReason.CALIBRATION_INVALID.value
    assert result.safe_action.value == "RETAIN_FOR_GROUND"
    assert not request.artifact_target.exists()

    valid_request, _, _, _, _, _ = _capture_request(tmp_path / "resources")
    from preprocessing.resolver import ContractResolver

    resolved = ContractResolver(trust_policy=TrustPolicy.development()).resolve(valid_request)
    receipt = ResourceAdmission(valid_request.compute_profile).preflight(resolved)
    estimate = receipt.estimate
    with pytest.raises(PreprocessError) as ram_error:
        ResourceAdmission(
            valid_request.compute_profile,
            probe=StaticResourceProbe(available_ram_bytes=estimate.ram_peak_bytes - 1, free_disk_bytes=estimate.disk_required_bytes),
        ).preflight(resolved)
    assert getattr(ram_error.value, "reason_code", None) == FailureReason.RESOURCE_PREFLIGHT.value
    with pytest.raises(PreprocessError) as disk_error:
        ResourceAdmission(valid_request.compute_profile).runtime_check(
            receipt,
            actual_peak_ram_bytes=estimate.ram_peak_bytes,
            free_disk_bytes=estimate.disk_required_bytes - 1,
            stage="p7-disk-runtime",
        )
    assert getattr(disk_error.value, "reason_code", None) == FailureReason.RESOURCE_RUNTIME.value


def test_p7_fault_injection_trust_and_codec_fail_closed(tmp_path):
    request, source, _, _, _, _ = _capture_request(tmp_path)
    untrusted = replace(request, trust_policy=TrustPolicy())
    result = preprocess_capture(untrusted)
    assert isinstance(result, PreprocessFailure)
    assert result.state == RunState.UNTRUSTED_ARTIFACT
    assert result.reason_code == FailureReason.TRUST_REJECTED.value
    assert source.exists()

    compressed = replace(
        request,
        source={
            "path": source,
            "format": "npz",
            "compressed": True,
            "full_decode_bytes": None,
        },
    )
    result = preprocess_capture(compressed)
    assert isinstance(result, PreprocessFailure)
    assert result.state == RunState.RESOURCE_REJECTED
    assert result.reason_code == FailureReason.CODEC_UNAVAILABLE.value
    assert source.exists()


@pytest.mark.parametrize("fault", [TimeoutError("TensorRT timeout"), RuntimeError("CUDA context reset")])
def test_p7_fault_injection_engine_timeout_and_reset_are_typed(tmp_path, fault):
    request, _, _, _, _, _ = _capture_request(tmp_path)
    artifact = preprocess_capture(request)
    assert not isinstance(artifact, PreprocessFailure)
    model = ModelCompatibilityProfile(
        profile_id="p7-model",
        profile_version="1",
        model_fingerprint="a" * 64,
        required_band_order=("red", "green", "blue"),
        tensor_layout="NCHW",
        tensor_dtype="float32",
        patch_size=(2, 2),
        batch_size=1,
        padding_policy="constant",
        normalization={"id": "identity-v1", "kind": "identity"},
        runtime_fingerprint="b" * 64,
    )

    class FailingEngine:
        def infer_batch(self, _batch):
            raise fault

    result = InferenceAdapter(model, engine=FailingEngine()).run(artifact)
    assert isinstance(result, PreprocessFailure)
    assert result.state == RunState.RUNTIME_FAULT
    assert result.reason_code == FailureReason.RUNTIME_FAULT.value
    assert result.safe_action.value == "RETAIN_FOR_GROUND"
    assert result.provenance["engine_fault_type"] == type(fault).__name__


@pytest.mark.parametrize("fault", [TimeoutError("engine initialization timeout"), RuntimeError("CUDA context reset")])
def test_p7_fault_injection_engine_factory_faults_are_typed(tmp_path, fault):
    request, _, _, _, _, _ = _capture_request(tmp_path)
    artifact = preprocess_capture(request)
    assert not isinstance(artifact, PreprocessFailure)
    model = ModelCompatibilityProfile(
        profile_id="p7-model-factory",
        profile_version="1",
        model_fingerprint="f" * 64,
        required_band_order=("red", "green", "blue"),
        tensor_layout="NCHW",
        tensor_dtype="float32",
        patch_size=(2, 2),
        batch_size=1,
        padding_policy="constant",
        normalization={"id": "identity-v1", "kind": "identity"},
        runtime_fingerprint="e" * 64,
    )

    def failing_factory():
        raise fault

    result = InferenceAdapter(model, engine_factory=failing_factory).run(artifact)
    assert isinstance(result, PreprocessFailure)
    assert result.state == RunState.RUNTIME_FAULT
    assert result.reason_code == FailureReason.RUNTIME_FAULT.value
    assert result.provenance["engine_operation"] == "initialize"


@pytest.mark.parametrize("manifest_mutation", ["checksum", "partial"])
def test_p7_fault_injection_checksum_and_partial_manifest_are_unopenable(tmp_path, manifest_mutation):
    request, _, capture, profile, calibration, compute = _capture_request(tmp_path)
    artifact = preprocess_capture(request)
    assert not isinstance(artifact, PreprocessFailure)
    if manifest_mutation == "checksum":
        payload = artifact.artifact_path / "model-grid.tif"
        payload.write_bytes(payload.read_bytes() + b"tamper")
    else:
        manifest_path = artifact.artifact_path / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["complete"] = False
        manifest["status"] = "writing"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    opened = open_preprocessed_artifact(
        ArtifactOpenRequest(
            artifact_path=artifact.artifact_path,
            expected_source_fingerprint=capture.source_fingerprint,
            expected_profile_fingerprint=profile.fingerprint,
            expected_calibration_fingerprint=calibration.fingerprint,
            trust_policy=request.trust_policy,
            compute_profile=compute,
            run_id="p7-open",
        )
    )
    assert isinstance(opened, PreprocessFailure)
    expected = (
        FailureReason.ARTIFACT_CHECKSUM_MISMATCH.value
        if manifest_mutation == "checksum"
        else FailureReason.ARTIFACT_INCOMPLETE.value
    )
    assert opened.reason_code == expected


def test_p7_shadow_decision_requires_georef_and_never_deletes_source(tmp_path):
    # Exercise the policy gate with a complete patch-result whose georef is
    # explicitly invalid. The policy emits a safe record; OBC/F' owns action.
    source_fingerprint = "c" * 64
    engine_fingerprint = "d" * 64
    writer = PatchResultWriter(
        tmp_path / "p7-shadow.patch-results",
        capture_id="p7-shadow",
        source_fingerprint=source_fingerprint,
        preprocessing_profile_id="p7-profile",
        engine_fingerprint=engine_fingerprint,
        patch_grid_shape=(1, 1),
        patch_size=(2, 2),
    )
    writer.append(
        {
            "patch_row": 0,
            "patch_col": 0,
            "model_window": (0, 2, 0, 2),
            "valid_fraction": 1.0,
            "validity_reason_summary": {},
            "cloud_probability": 0.99,
            "cloud_label": "cloud",
            "inference_status": "valid",
            "source_mapping_ref": "mapping.npy",
            "georef_valid": False,
        }
    )
    result = writer.finalize()
    decision = DecisionPolicy(cloud_coverage_threshold=0.1).evaluate(result)
    assert decision.decision == "RETAIN_FOR_GROUND"
    assert decision.reason_code == "GEOREF_INVALID"
    assert decision.safe_action == "RETAIN_FOR_GROUND"
    assert not decision.actionable
    assert not (tmp_path / "capture-deleted").exists()
