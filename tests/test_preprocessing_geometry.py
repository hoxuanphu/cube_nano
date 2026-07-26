import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from preprocessing import CalibrationBundle, ClippingPolicy, ComputeProfile, PreprocessingProfile  # noqa: E402
from preprocessing.errors import FailureReason, PreprocessError, RunState  # noqa: E402
from preprocessing.source_reader import ArraySourceReader, SourceWindow, open_source_reader  # noqa: E402
from preprocessing.transform_plan import TransformPlanner  # noqa: E402
from preprocessing.validity import ValidityBuilder  # noqa: E402
from preprocessing.warp_backend import CPUWarpBackend, create_warp_backend  # noqa: E402


def _profile(*, shape=(6, 7), dtype="uint16", **changes):
    profile = PreprocessingProfile.identity_fixture(shape=shape, dtype=dtype)
    if changes:
        profile = replace(profile, **changes, digest=None)
    return profile


def _warp(profile, calibration, image, *, row_start=0, row_end=None):
    reader = ArraySourceReader(image, profile)
    planner = TransformPlanner(profile, calibration, reader.shape_yxc)
    plan = planner.plan_strip(row_start, row_end or profile.target_grid.rows)
    block = reader.read_window(plan.source_window)
    return CPUWarpBackend(profile).warp(block, plan), plan


def test_source_reader_preserves_values_dtype_axes_and_window_masks():
    profile = _profile(shape=(3, 4))
    image_yxc = np.arange(3 * 4 * 3, dtype=np.uint16).reshape(3, 4, 3)
    image_cyx = np.moveaxis(image_yxc, -1, 0)
    source_schema = replace(
        profile.source_schema,
        axes="CYX",
        shape=image_cyx.shape,
        byte_order="native",
    )
    profile = replace(profile, source_schema=source_schema, digest=None)

    reader = ArraySourceReader(image_cyx, profile)
    assert reader.read_rows(0, 3).dtype == np.dtype("uint16")
    assert np.array_equal(reader.read_rows(0, 3), image_yxc)
    block = reader.read_window(SourceWindow(1, 3, 1, 4))
    assert np.array_equal(block.image_yxc, image_yxc[1:3, 1:4])
    assert np.all(block.validity_yx == profile.validity_encoding.valid_value)
    assert np.all(block.validity_reason_yx == 0)


def test_source_reader_accepts_declared_non_native_byte_order():
    profile = _profile(shape=(2, 2))
    source_schema = replace(profile.source_schema, byte_order="big")
    profile = replace(profile, source_schema=source_schema, digest=None)
    image = np.arange(2 * 2 * 3, dtype=">u2").reshape(2, 2, 3)
    reader = ArraySourceReader(image, profile)
    block = reader.read_window(SourceWindow(0, 2, 0, 2))
    assert block.image_yxc.dtype.byteorder == ">"
    assert np.array_equal(block.image_yxc, image)


def test_nodata_and_missing_channel_reasons_are_or_combined():
    source_schema = replace(
        _profile(shape=(2, 3)).source_schema,
        nodata_semantics={"kind": "value", "value": 0},
    )
    profile = replace(_profile(shape=(2, 3)), source_schema=source_schema, digest=None)
    image = np.ones((2, 3, 3), dtype=np.uint16)
    image[0, 0, 0] = 0
    image[0, 2, 0] = 0
    missing = np.zeros((2, 3), dtype=bool)
    missing[0, 1] = True
    missing[0, 2] = True

    block = ArraySourceReader(image, profile, missing_channels=missing).read_window(SourceWindow(0, 2, 0, 3))
    bits = profile.reason_encoding.bits
    assert block.validity_yx[0, 0] == profile.validity_encoding.invalid_value
    assert block.validity_yx[0, 1] == profile.validity_encoding.invalid_value
    assert block.validity_reason_yx[0, 0] == bits["source_nodata"]
    assert block.validity_reason_yx[0, 1] == bits["missing_channel"]
    assert block.validity_reason_yx[0, 2] == bits["source_nodata"] | bits["missing_channel"]


def test_validity_builder_does_not_infer_invalidity_from_color_without_nodata_contract():
    profile = _profile(shape=(2, 2))
    image = np.zeros((2, 2, 3), dtype=np.uint16)
    masks = ValidityBuilder(profile).build_source(image)
    assert np.all(masks.validity_yx == profile.validity_encoding.valid_value)
    assert np.all(masks.reason_yx == 0)


def test_identity_warp_is_exact_and_uses_profile_output_boundary():
    profile = _profile(shape=(4, 5))
    image = np.arange(4 * 5 * 3, dtype=np.uint16).reshape(4, 5, 3)
    result, plan = _warp(profile, CalibrationBundle.identity_fixture(), image)

    assert np.array_equal(result.image, image)
    assert result.image.dtype == np.dtype(profile.output_dtype)
    assert np.all(result.validity_yx == profile.validity_encoding.valid_value)
    assert np.all(result.validity_reason_yx == 0)
    assert np.allclose(result.mapping_yx[0, 0], (0.5, 0.5))
    assert np.allclose(result.mapping_yx[-1, -1], (3.5, 4.5))
    assert plan.source_window == SourceWindow(0, 4, 0, 5)


def test_affine_mapping_and_invalid_border_are_deterministic():
    profile = PreprocessingProfile.affine_fixture(shape=(4, 5), dtype="uint16")
    calibration = CalibrationBundle.affine_fixture(offset=(0.25, 0.25))
    image = np.arange(4 * 5 * 3, dtype=np.uint16).reshape(4, 5, 3)
    result, plan = _warp(profile, calibration, image)

    assert np.allclose(plan.mapping_yx[0, 0], (0.25, 0.25))
    assert result.validity_yx[0, 0] == profile.validity_encoding.invalid_value
    assert result.validity_reason_yx[0, 0] & profile.reason_encoding.bits["outside_mapping"]
    assert result.validity_yx[-1, -1] == profile.validity_encoding.valid_value


def test_strip_size_and_halo_do_not_change_warp_result():
    profile = replace(
        PreprocessingProfile.affine_fixture(shape=(8, 9), dtype="uint16"),
        image_kernel="bilinear",
        validity_kernel="bicubic",
        reason_kernel="support",
        halo=replace(PreprocessingProfile.affine_fixture(shape=(8, 9), dtype="uint16").halo, rows=1, cols=1),
        digest=None,
    )
    calibration = CalibrationBundle.affine_fixture(offset=(0.25, 0.25))
    image = np.arange(8 * 9 * 3, dtype=np.uint16).reshape(8, 9, 3)
    reader = ArraySourceReader(image, profile)
    planner = TransformPlanner(profile, calibration, reader.shape_yxc)
    backend = CPUWarpBackend(profile)

    full_plan = planner.plan_strip(0, 8)
    full = backend.warp(reader.read_window(full_plan.source_window), full_plan)
    chunks = []
    for plan in planner.plan_strips(3):
        chunks.append(backend.warp(reader.read_window(plan.source_window), plan))
    chunk_image = np.concatenate([chunk.image for chunk in chunks], axis=0)
    chunk_validity = np.concatenate([chunk.validity_yx for chunk in chunks], axis=0)
    chunk_reasons = np.concatenate([chunk.validity_reason_yx for chunk in chunks], axis=0)
    chunk_mapping = np.concatenate([chunk.mapping_yx for chunk in chunks], axis=0)

    assert np.array_equal(chunk_image, full.image)
    assert np.array_equal(chunk_validity, full.validity_yx)
    assert np.array_equal(chunk_reasons, full.validity_reason_yx)
    assert np.allclose(chunk_mapping, full.mapping_yx)


def test_profile_driven_rounding_clipping_and_layout():
    base = PreprocessingProfile.identity_fixture(shape=(1, 4), channels=("gray",), dtype="float32")
    profile = replace(
        base,
        output_dtype="uint8",
        output_layout="CYX",
        clipping=ClippingPolicy(mode="range", lower=0, upper=255),
        rounding="nearest_even",
        digest=None,
    )
    image = np.array([[[1.4], [2.5], [300.0], [-2.0]]], dtype=np.float32)
    result, _ = _warp(profile, CalibrationBundle.identity_fixture(), image)
    assert result.image.dtype == np.dtype("uint8")
    assert result.image.shape == (1, 1, 4)
    assert np.array_equal(result.image[0, 0], np.array([1, 2, 255, 0], dtype=np.uint8))


def test_non_finite_policy_rejects_or_replaces_at_warp_boundary():
    reject_profile = _profile(shape=(1, 1), dtype="float32")
    image = np.array([[[np.nan, 1.0, 2.0]]], dtype=np.float32)
    with pytest.raises(PreprocessError) as error:
        _warp(reject_profile, CalibrationBundle.identity_fixture(), image)
    assert error.value.reason_code == FailureReason.NON_FINITE_OUTPUT.value
    assert error.value.state == RunState.RUNTIME_FAULT

    replace_profile = replace(reject_profile, non_finite_policy="replace", digest=None)
    result, _ = _warp(replace_profile, CalibrationBundle.identity_fixture(), image)
    assert result.image[0, 0, 0] == 0
    assert result.validity_yx[0, 0] == replace_profile.validity_encoding.invalid_value


def test_compressed_full_decode_requires_explicit_bounded_admission(tmp_path):
    path = tmp_path / "capture.npz"
    image = np.ones((2, 2, 3), dtype=np.uint16)
    np.savez_compressed(path, image=image)
    profile = _profile(shape=(2, 2))

    with pytest.raises(PreprocessError) as error:
        open_source_reader({"path": path, "compressed": True}, profile)
    assert error.value.reason_code == FailureReason.CODEC_UNAVAILABLE.value

    compute = ComputeProfile(
        compute_profile_id="fixture-compressed",
        profile_version="1",
        backend="cpu",
        ram_budget_bytes=1_000_000,
        disk_budget_bytes=1_000_000,
        os_obc_reserve_bytes=1_000,
        safety_margin_bytes=1_000,
        max_strip_rows=2,
        queue_depth=1,
        inflight_strips=1,
        temporary_directory=str(tmp_path),
        allow_compressed_full_decode=True,
        max_full_decode_bytes=int(image.nbytes),
        thermal_policy="fixture",
    )
    reader = open_source_reader(
        {"path": path, "compressed": True, "full_decode_bytes": int(image.nbytes)},
        profile,
        compute_profile=compute,
    )
    with reader:
        assert np.array_equal(reader.read_rows(0, 2), image)


def test_gpu_backend_is_explicitly_disabled_until_p3_05_evidence():
    profile = _profile(shape=(2, 2))
    with pytest.raises(PreprocessError) as error:
        create_warp_backend(profile, backend="gpu")
    assert error.value.reason_code == FailureReason.NOT_IMPLEMENTED.value
    assert error.value.state == RunState.RESOURCE_REJECTED
