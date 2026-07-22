"""Versioned contracts shared by training, inference and product publishing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


def _required(mapping: Mapping[str, Any], key: str, label: str) -> Any:
    if key not in mapping:
        raise ValueError(f"{label} is missing {key}")
    return mapping[key]


def _tuple_of_strings(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{label} must be a non-empty list")
    return tuple(str(item) for item in value)


def _tuple_of_ints(value: Any, label: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{label} must be a non-empty list")
    return tuple(int(item) for item in value)


def _basis_points(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer basis-point value")
    result = int(value)
    if not 0 <= result <= 10000:
        raise ValueError(f"{label} must be in [0, 10000]")
    return result


@dataclass(frozen=True)
class ModelOutputSpec:
    """Graph output contract, before resize, probability and product encoding."""

    kind: str
    name: str
    shape: tuple[int | None, ...]
    logical_dtype: str
    physical_dtype: str
    class_axis: int
    classes: tuple[str, ...]
    output_stride: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ModelOutputSpec":
        if not isinstance(value, Mapping):
            raise ValueError("model output must be an object")
        raw_shape = _required(value, "shape", "model output")
        if not isinstance(raw_shape, list) or not raw_shape:
            raise ValueError("model output shape must be a non-empty list")
        shape = tuple(None if item is None else int(item) for item in raw_shape)
        result = cls(
            kind=str(_required(value, "kind", "model output")),
            name=str(_required(value, "name", "model output")),
            shape=shape,
            logical_dtype=str(_required(value, "logical_dtype", "model output")),
            physical_dtype=str(_required(value, "physical_dtype", "model output")),
            class_axis=int(_required(value, "class_axis", "model output")),
            classes=_tuple_of_strings(_required(value, "classes", "model output"), "model output classes"),
            output_stride=int(_required(value, "output_stride", "model output")),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.kind != "semantic_logits":
            raise ValueError("SegFormer output kind must be semantic_logits")
        if self.name != "logits":
            raise ValueError("SegFormer output name must be logits")
        if self.shape != (1, 2, 64, 64):
            raise ValueError("SegFormer MVP output shape must be [1, 2, 64, 64]")
        if self.logical_dtype != "float32":
            raise ValueError("SegFormer logical output dtype must be float32")
        if self.class_axis != 1 or self.classes != ("clear", "cloud"):
            raise ValueError("SegFormer class axis/mapping must be axis 1 with clear/cloud")
        if self.output_stride != 4:
            raise ValueError("SegFormer MVP output stride must be 4")

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "shape": list(self.shape),
            "logical_dtype": self.logical_dtype,
            "physical_dtype": self.physical_dtype,
            "class_axis": self.class_axis,
            "classes": list(self.classes),
            "output_stride": self.output_stride,
        }


@dataclass(frozen=True)
class PostprocessSpec:
    postprocess_id: str
    resize_mode: str
    resize_target: str
    align_corners: bool
    probability_kind: str
    cloud_class_index: int
    threshold_source: str
    invalid_policy: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PostprocessSpec":
        if not isinstance(value, Mapping):
            raise ValueError("postprocess must be an object")
        resize = _required(value, "resize", "postprocess")
        probability = _required(value, "probability", "postprocess")
        if not isinstance(resize, Mapping) or not isinstance(probability, Mapping):
            raise ValueError("postprocess resize/probability must be objects")
        result = cls(
            postprocess_id=str(_required(value, "postprocess_id", "postprocess")),
            resize_mode=str(_required(resize, "mode", "postprocess resize")),
            resize_target=str(_required(resize, "target", "postprocess resize")),
            align_corners=bool(_required(resize, "align_corners", "postprocess resize")),
            probability_kind=str(_required(probability, "kind", "postprocess probability")),
            cloud_class_index=int(_required(probability, "cloud_class_index", "postprocess probability")),
            threshold_source=str(_required(value, "threshold_source", "postprocess")),
            invalid_policy=str(_required(value, "invalid_policy", "postprocess")),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.resize_mode != "bilinear" or self.resize_target != "input_spatial_shape":
            raise ValueError("SegFormer postprocess must use bilinear input-sized resize")
        if self.align_corners:
            raise ValueError("SegFormer postprocess align_corners must be false")
        if self.probability_kind != "softmax" or self.cloud_class_index != 1:
            raise ValueError("SegFormer probability contract must use softmax cloud class 1")
        if self.threshold_source != "decision_spec.pixel_cloud_probability_threshold_bp":
            raise ValueError("postprocess threshold source must be DecisionSpec")
        if self.invalid_policy != "exclude-from-mask-metrics-and-coverage":
            raise ValueError("postprocess invalid policy is not the pinned MVP policy")

    def as_dict(self) -> dict[str, Any]:
        return {
            "postprocess_id": self.postprocess_id,
            "resize": {
                "mode": self.resize_mode,
                "target": self.resize_target,
                "align_corners": self.align_corners,
            },
            "probability": {
                "kind": self.probability_kind,
                "cloud_class_index": self.cloud_class_index,
            },
            "threshold_source": self.threshold_source,
            "invalid_policy": self.invalid_policy,
        }


@dataclass(frozen=True)
class ProductSpec:
    product_spec_id: str
    cloud_mask_dtype: str
    clear_value: int
    cloud_value: int
    validity_mask_dtype: str
    invalid_value: int
    valid_value: int
    cloud_mask_artifact: str
    validity_mask_artifact: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ProductSpec":
        if not isinstance(value, Mapping):
            raise ValueError("product spec must be an object")
        cloud_mask = _required(value, "cloud_mask", "product spec")
        validity = _required(value, "validity_mask", "product spec")
        if not isinstance(cloud_mask, Mapping) or not isinstance(validity, Mapping):
            raise ValueError("product cloud_mask/validity_mask must be objects")
        result = cls(
            product_spec_id=str(_required(value, "product_spec_id", "product spec")),
            cloud_mask_dtype=str(_required(cloud_mask, "dtype", "cloud mask")),
            clear_value=int(_required(cloud_mask, "clear_value", "cloud mask")),
            cloud_value=int(_required(cloud_mask, "cloud_value", "cloud mask")),
            validity_mask_dtype=str(_required(validity, "dtype", "validity mask")),
            invalid_value=int(_required(validity, "invalid_value", "validity mask")),
            valid_value=int(_required(validity, "valid_value", "validity mask")),
            cloud_mask_artifact=str(_required(cloud_mask, "artifact", "cloud mask")),
            validity_mask_artifact=str(_required(validity, "artifact", "validity mask")),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.cloud_mask_dtype != "uint8" or (self.clear_value, self.cloud_value) != (0, 255):
            raise ValueError("cloud mask product must be uint8 with clear=0/cloud=255")
        if self.validity_mask_dtype != "uint8" or (self.invalid_value, self.valid_value) != (0, 1):
            raise ValueError("validity mask product must be uint8 with invalid=0/valid=1")
        if self.cloud_mask_artifact == self.validity_mask_artifact:
            raise ValueError("cloud and validity masks must be separate artifacts")

    def as_dict(self) -> dict[str, Any]:
        return {
            "product_spec_id": self.product_spec_id,
            "cloud_mask": {
                "dtype": self.cloud_mask_dtype,
                "clear_value": self.clear_value,
                "cloud_value": self.cloud_value,
                "artifact": self.cloud_mask_artifact,
            },
            "validity_mask": {
                "dtype": self.validity_mask_dtype,
                "invalid_value": self.invalid_value,
                "valid_value": self.valid_value,
                "artifact": self.validity_mask_artifact,
            },
        }


@dataclass(frozen=True)
class DecisionSpec:
    decision_spec_id: str
    pixel_cloud_probability_threshold_bp: int
    coverage_limit_bp: int
    threshold_selection_metric: str
    false_clear_constraint_bp: int
    calibration_id: str
    min_valid_pixel_ratio_bp: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DecisionSpec":
        if not isinstance(value, Mapping):
            raise ValueError("decision spec must be an object")
        result = cls(
            decision_spec_id=str(_required(value, "decision_spec_id", "decision spec")),
            pixel_cloud_probability_threshold_bp=_basis_points(
                _required(value, "pixel_cloud_probability_threshold_bp", "decision spec"),
                "pixel_cloud_probability_threshold_bp",
            ),
            coverage_limit_bp=_basis_points(_required(value, "coverage_limit_bp", "decision spec"), "coverage_limit_bp"),
            threshold_selection_metric=str(_required(value, "threshold_selection_metric", "decision spec")),
            false_clear_constraint_bp=_basis_points(
                _required(value, "false_clear_constraint_bp", "decision spec"),
                "false_clear_constraint_bp",
            ),
            calibration_id=str(_required(value, "calibration_id", "decision spec")),
            min_valid_pixel_ratio_bp=_basis_points(
                value.get("min_valid_pixel_ratio_bp", 9500),
                "min_valid_pixel_ratio_bp",
            ),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if not self.decision_spec_id:
            raise ValueError("decision_spec_id must not be empty")
        if not self.threshold_selection_metric:
            raise ValueError("threshold_selection_metric must not be empty")
        if not self.calibration_id:
            raise ValueError("calibration_id must not be empty")

    def matches_config(self, pixel_threshold_bp: int, coverage_limit_bp: int) -> bool:
        """Return whether mutable command values match this immutable release decision."""

        return (
            pixel_threshold_bp == self.pixel_cloud_probability_threshold_bp
            and coverage_limit_bp == self.coverage_limit_bp
        )

    def require_config(self, pixel_threshold_bp: int, coverage_limit_bp: int) -> None:
        if not self.matches_config(pixel_threshold_bp, coverage_limit_bp):
            raise ValueError("DECISION_SPEC_MISMATCH")

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision_spec_id": self.decision_spec_id,
            "pixel_cloud_probability_threshold_bp": self.pixel_cloud_probability_threshold_bp,
            "coverage_limit_bp": self.coverage_limit_bp,
            "threshold_selection_metric": self.threshold_selection_metric,
            "false_clear_constraint_bp": self.false_clear_constraint_bp,
            "calibration_id": self.calibration_id,
            "min_valid_pixel_ratio_bp": self.min_valid_pixel_ratio_bp,
        }


@dataclass(frozen=True)
class AcceptanceProfile:
    profile_id: str
    status: str
    quality: dict[str, Any]
    decision: dict[str, Any]
    parity: dict[str, Any]
    runtime: dict[str, Any]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AcceptanceProfile":
        if not isinstance(value, Mapping):
            raise ValueError("acceptance profile must be an object")
        result = cls(
            profile_id=str(_required(value, "profile_id", "acceptance profile")),
            status=str(_required(value, "status", "acceptance profile")),
            quality=dict(_required(value, "quality", "acceptance profile")),
            decision=dict(_required(value, "decision", "acceptance profile")),
            parity=dict(_required(value, "parity", "acceptance profile")),
            runtime=dict(_required(value, "runtime", "acceptance profile")),
        )
        result.validate()
        return result

    @classmethod
    def from_file(cls, path: str | Path) -> "AcceptanceProfile":
        value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.from_mapping(value)

    def validate(self) -> None:
        required = {
            "quality": (
                "min_cloud_iou",
                "min_cloud_dice",
                "min_cloud_recall",
                "max_false_clear_rate",
                "max_coverage_mae_bp",
                "max_coverage_p95_abs_error_bp",
                "boundary_f1_tolerance_pixels",
                "min_boundary_f1",
            ),
            "decision": ("max_false_accept_cloudy_scene_rate", "max_false_reject_useful_scene_rate"),
            "parity": ("pytorch_onnx", "pytorch_tensorrt_fp16"),
            "runtime": (
                "max_cold_start_ms",
                "max_warm_p95_ms_per_tile",
                "max_end_to_end_deadline_miss_rate",
                "max_peak_rss_bytes",
                "min_valid_pixel_ratio",
            ),
        }
        sections = {"quality": self.quality, "decision": self.decision, "parity": self.parity, "runtime": self.runtime}
        for section, keys in required.items():
            for key in keys:
                if key not in sections[section] or sections[section][key] is None:
                    raise ValueError(f"acceptance profile is missing {section}.{key}")
        if not self.profile_id:
            raise ValueError("acceptance profile_id must not be empty")
        if self.status not in {"engineering-baseline", "approved"}:
            raise ValueError("acceptance profile status must be engineering-baseline or approved")

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "status": self.status,
            "quality": dict(self.quality),
            "decision": dict(self.decision),
            "parity": dict(self.parity),
            "runtime": dict(self.runtime),
        }


@dataclass(frozen=True)
class TargetDeploymentSpec:
    target_id: str
    hardware_sku: str
    os: str
    l4t: str
    cuda: str
    tensorrt: str
    precision: str
    power_mode: str
    clock_policy: str
    memory_budget_bytes: int
    batch_size: int
    input_shape: tuple[int, ...]
    onnx_opset: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TargetDeploymentSpec":
        if not isinstance(value, Mapping):
            raise ValueError("target deployment spec must be an object")
        result = cls(
            target_id=str(_required(value, "target_id", "target deployment spec")),
            hardware_sku=str(_required(value, "hardware_sku", "target deployment spec")),
            os=str(_required(value, "os", "target deployment spec")),
            l4t=str(_required(value, "l4t", "target deployment spec")),
            cuda=str(_required(value, "cuda", "target deployment spec")),
            tensorrt=str(_required(value, "tensorrt", "target deployment spec")),
            precision=str(_required(value, "precision", "target deployment spec")),
            power_mode=str(_required(value, "power_mode", "target deployment spec")),
            clock_policy=str(_required(value, "clock_policy", "target deployment spec")),
            memory_budget_bytes=int(_required(value, "memory_budget_bytes", "target deployment spec")),
            batch_size=int(_required(value, "batch_size", "target deployment spec")),
            input_shape=tuple(int(item) for item in _required(value, "input_shape", "target deployment spec")),
            onnx_opset=int(_required(value, "onnx_opset", "target deployment spec")),
        )
        result.validate()
        return result

    @classmethod
    def from_file(cls, path: str | Path) -> "TargetDeploymentSpec":
        value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.from_mapping(value)

    def validate(self) -> None:
        if not self.target_id or not self.hardware_sku or not self.os:
            raise ValueError("target deployment identity fields must not be empty")
        if self.memory_budget_bytes <= 0 or self.batch_size != 1:
            raise ValueError("target memory budget must be positive and MVP batch size must be 1")
        if self.input_shape != (1, 3, 256, 256) or self.onnx_opset < 13:
            raise ValueError("target input/opset does not match the pinned SegFormer MVP")

    def as_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "hardware_sku": self.hardware_sku,
            "os": self.os,
            "l4t": self.l4t,
            "cuda": self.cuda,
            "tensorrt": self.tensorrt,
            "precision": self.precision,
            "power_mode": self.power_mode,
            "clock_policy": self.clock_policy,
            "memory_budget_bytes": self.memory_budget_bytes,
            "batch_size": self.batch_size,
            "input_shape": list(self.input_shape),
            "onnx_opset": self.onnx_opset,
        }


def load_yaml_contract(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"contract file must contain an object: {path}")
    return value
