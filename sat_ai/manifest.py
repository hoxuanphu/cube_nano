"""Immutable checkpoint/InputSpec manifest validation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .contracts import DecisionSpec, ModelOutputSpec, PostprocessSpec, ProductSpec


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _require(mapping: dict[str, Any], key: str, label: str) -> Any:
    if key not in mapping:
        raise ValueError(f"{label} is missing {key}")
    return mapping[key]


@dataclass(frozen=True)
class InputSpec:
    input_spec_id: str
    channels: int
    band_order: tuple[str, ...]
    patch_size: int
    source_dtype: str
    tensor_dtype: str
    tensor_layout: str
    input_shape: tuple[int | None, int, int, int]
    normalization_id: str
    normalization_kind: str
    integer_scale: int
    padding_id: str
    padding_kind: str
    padding_value_space: str
    padding_values: tuple[int, ...]

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "InputSpec":
        if not isinstance(value, dict):
            raise ValueError("input_spec must be an object")
        normalization = _require(value, "normalization", "input_spec")
        padding = _require(value, "padding", "input_spec")
        if not isinstance(normalization, dict) or not isinstance(padding, dict):
            raise ValueError("input_spec normalization and padding must be objects")
        raw_shape = _require(value, "input_shape", "input_spec")
        if not isinstance(raw_shape, list) or len(raw_shape) != 4:
            raise ValueError("input_spec.input_shape must be [null, channels, height, width]")
        shape = tuple(None if item is None else int(item) for item in raw_shape)
        result = cls(
            input_spec_id=str(_require(value, "input_spec_id", "input_spec")),
            channels=int(_require(value, "channels", "input_spec")),
            band_order=tuple(str(item).lower() for item in _require(value, "band_order", "input_spec")),
            patch_size=int(_require(value, "patch_size", "input_spec")),
            source_dtype=str(_require(value, "source_dtype", "input_spec")),
            tensor_dtype=str(_require(value, "tensor_dtype", "input_spec")),
            tensor_layout=str(_require(value, "tensor_layout", "input_spec")),
            input_shape=shape,
            normalization_id=str(_require(normalization, "id", "normalization")),
            normalization_kind=str(_require(normalization, "kind", "normalization")),
            integer_scale=int(_require(normalization, "integer_scale", "normalization")),
            padding_id=str(_require(padding, "id", "padding")),
            padding_kind=str(_require(padding, "kind", "padding")),
            padding_value_space=str(_require(padding, "value_space", "padding")),
            padding_values=tuple(int(item) for item in _require(padding, "values", "padding")),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.channels != 3 or self.band_order != ("red", "green", "blue"):
            raise ValueError("the released MVP checkpoint is RGB with canonical band order")
        if self.patch_size != 256 or self.input_shape != (None, 3, 256, 256):
            raise ValueError("MVP InputSpec shape must be [null, 3, 256, 256]")
        if self.source_dtype != "uint16" or self.tensor_dtype != "float32":
            raise ValueError("MVP source/tensor dtype must be uint16/float32")
        if self.tensor_layout != "NCHW":
            raise ValueError("MVP tensor layout must be NCHW")
        if self.normalization_id not in {
            "legacy-dtype-range-v1",
            "segformer-rgb-dtype-range-v1",
        } or self.normalization_kind != "dtype-range":
            raise ValueError("MVP normalization must be a pinned dtype-range contract")
        if self.integer_scale != 65535:
            raise ValueError("uint16 normalization scale must be 65535")
        if self.padding_kind != "constant" or self.padding_value_space != "source":
            raise ValueError("MVP uses constant source-space padding")
        if self.padding_values != (0, 0, 0):
            raise ValueError("MVP padding values must be [0, 0, 0]")

    def normalize(self, patch: np.ndarray) -> np.ndarray:
        patch = np.asarray(patch)
        if patch.ndim != 3 or patch.shape[-1] != self.channels:
            raise ValueError(f"expected HWC RGB data, got {patch.shape}")
        if patch.dtype != np.dtype(self.source_dtype):
            raise ValueError(f"expected {self.source_dtype} source data, got {patch.dtype}")
        return patch.astype(np.float32) / np.float32(self.integer_scale)


@dataclass(frozen=True)
class ModelManifest:
    schema_version: int
    model_id: str
    model_release_id: str
    checkpoint_sha256: str
    framework: str
    assurance_level: str
    assurance_profile_id: str
    threshold_mapping_id: str
    threshold_lut_sha256: str
    input_spec: InputSpec
    supported_domains: dict[str, tuple[str, ...]]
    model_task: str = "patch_classification"
    output_spec: ModelOutputSpec | None = None
    postprocess_spec: PostprocessSpec | None = None
    product_spec: ProductSpec | None = None
    decision_spec: DecisionSpec | None = None
    acceptance_profile_id: str | None = None

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "ModelManifest":
        if not isinstance(value, dict):
            raise ValueError("model manifest must be an object")
        input_spec = InputSpec.from_mapping(_require(value, "input_spec", "model manifest"))
        output = _require(value, "output", "model manifest")
        domains = value.get("supported_domains", {})
        if not isinstance(output, dict) or not isinstance(domains, dict):
            raise ValueError("manifest output/domains must be objects")
        digest = str(_require(value, "checkpoint_sha256", "model manifest")).lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("checkpoint_sha256 must be a SHA-256 hex digest")
        schema_version = int(_require(value, "schema_version", "model manifest"))
        model_task = str(value.get("model_task", "patch_classification"))
        output_spec = None
        postprocess_spec = None
        product_spec = None
        decision_spec = None
        if model_task == "semantic_cloud_segmentation":
            output_spec = ModelOutputSpec.from_mapping(_require(output, "model_output", "manifest output"))
            postprocess_spec = PostprocessSpec.from_mapping(
                _require(value, "postprocess", "model manifest")
            )
            product_spec = ProductSpec.from_mapping(_require(value, "product", "model manifest"))
            decision_spec = DecisionSpec.from_mapping(
                _require(value, "decision_spec", "model manifest")
            )
        result = cls(
            schema_version=schema_version,
            model_id=str(_require(value, "model_id", "model manifest")),
            model_release_id=str(_require(value, "model_release_id", "model manifest")),
            checkpoint_sha256=digest,
            framework=str(_require(value, "framework", "model manifest")),
            assurance_level=str(_require(value, "assurance_level", "model manifest")),
            assurance_profile_id=str(_require(value, "assurance_profile_id", "model manifest")),
            threshold_mapping_id=str(output.get("threshold_mapping_id", "")),
            threshold_lut_sha256=str(output.get("threshold_lut_sha256", "")),
            input_spec=input_spec,
            supported_domains={
                str(key): tuple(str(item) for item in (items or []))
                for key, items in domains.items()
            },
            model_task=model_task,
            output_spec=output_spec,
            postprocess_spec=postprocess_spec,
            product_spec=product_spec,
            decision_spec=decision_spec,
            acceptance_profile_id=(
                None
                if value.get("acceptance_profile_id") is None
                else str(value["acceptance_profile_id"])
            ),
        )
        result.validate()
        return result

    @classmethod
    def from_file(cls, path: str | Path) -> "ModelManifest":
        path = Path(path)
        try:
            value = yaml.safe_load(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise FileNotFoundError(f"model manifest not found: {path}") from None
        except yaml.YAMLError as exc:
            raise ValueError(f"invalid model manifest YAML: {exc}") from exc
        return cls.from_mapping(value)

    def validate(self) -> None:
        if self.schema_version not in {1, 2}:
            raise ValueError("model manifest schema_version must be 1 or 2")
        if self.framework != "pytorch":
            raise ValueError("MVP model framework must be pytorch")
        if self.model_task not in {"patch_classification", "semantic_cloud_segmentation"}:
            raise ValueError(f"unsupported model task: {self.model_task}")
        if self.schema_version == 1 and self.model_task != "patch_classification":
            raise ValueError("schema_version 1 is reserved for patch_classification")
        if self.model_task == "patch_classification":
            if self.assurance_level != "demo_non_validated":
                raise ValueError("MVP classifier assurance level is demo_non_validated")
            if self.threshold_mapping_id != "logit-bp-f32-lut-v1":
                raise ValueError("MVP classifier threshold mapping is logit-bp-f32-lut-v1")
            if len(self.threshold_lut_sha256) != 64:
                raise ValueError("classifier threshold_lut_sha256 must be a SHA-256 digest")
        else:
            if self.schema_version != 2:
                raise ValueError("SegFormer manifest must use schema_version 2")
            if self.assurance_level not in {"pilot_non_validated", "demo_non_validated", "validated"}:
                raise ValueError("SegFormer assurance level is invalid")
            if not all(
                spec is not None
                for spec in (self.output_spec, self.postprocess_spec, self.product_spec, self.decision_spec)
            ):
                raise ValueError("SegFormer manifest requires all output/product/decision contracts")
            if not self.acceptance_profile_id:
                raise ValueError("SegFormer manifest requires acceptance_profile_id")
            if self.assurance_profile_id != self.acceptance_profile_id:
                raise ValueError("SegFormer assurance and acceptance profile IDs must match")
            assert self.decision_spec is not None
            if self.assurance_level == "validated" and self.decision_spec.calibration_id == "none":
                raise ValueError("validated SegFormer manifest requires a non-default calibration_id")
            if self.threshold_mapping_id and len(self.threshold_lut_sha256) != 64:
                raise ValueError("SegFormer threshold LUT reference must be a SHA-256 digest")

    def domain_status(self, domain: dict[str, Any] | None) -> str:
        """Return VERIFIED, DOMAIN_UNVERIFIED, or DOMAIN_MISMATCH.

        Empty allow-lists in the released manifest deliberately do not claim
        support for every sensor. They produce an explicit unverified result;
        a populated allow-list still rejects a known incompatible value.
        """
        if domain is None or not isinstance(domain, dict):
            return "DOMAIN_UNVERIFIED"
        mapping = {
            "sensor_ids": "sensor_id",
            "platform_ids": "platform_id",
            "product_types": "product_type",
            "processing_levels": "processing_level",
        }
        constrained = False
        for allowed_key, domain_key in mapping.items():
            allowed = tuple(self.supported_domains.get(allowed_key, ()))
            if not allowed:
                continue
            constrained = True
            actual = domain.get(domain_key)
            if actual is None or str(actual) not in allowed:
                return "DOMAIN_MISMATCH"
        return "VERIFIED" if constrained else "DOMAIN_UNVERIFIED"

    def input_contract(self) -> dict[str, Any]:
        result = {
            "input_spec_id": self.input_spec.input_spec_id,
            "channels": self.input_spec.channels,
            "band_order": list(self.input_spec.band_order),
            "patch_size": self.input_spec.patch_size,
            "source_dtype": self.input_spec.source_dtype,
            "normalization_id": self.input_spec.normalization_id,
        }
        if self.model_task == "semantic_cloud_segmentation":
            assert self.output_spec is not None
            result.update(
                tensor_dtype=self.input_spec.tensor_dtype,
                tensor_layout=self.input_spec.tensor_layout,
                input_shape=list(self.input_spec.input_shape),
                output_spec=self.output_spec.as_dict(),
            )
        return result

    def verify_checkpoint(self, checkpoint_path: str | Path) -> None:
        checkpoint_path = Path(checkpoint_path)
        actual = sha256_file(checkpoint_path)
        if actual != self.checkpoint_sha256:
            raise ValueError(
                f"checkpoint SHA-256 mismatch: manifest={self.checkpoint_sha256}, actual={actual}"
            )
        try:
            import torch

            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        except Exception as exc:
            raise ValueError(f"unable to inspect checkpoint: {exc}") from exc
        if isinstance(checkpoint, dict):
            checkpoint = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
        if not isinstance(checkpoint, dict):
            raise ValueError("checkpoint must contain a state dictionary")
        if self.model_task == "patch_classification":
            first_conv = next(
                (value for key, value in checkpoint.items() if key.endswith("features.0.0.weight")),
                None,
            )
            if first_conv is None or len(first_conv.shape) != 4 or first_conv.shape[1] != self.input_spec.channels:
                raise ValueError("checkpoint first convolution does not match InputSpec channels")
            if tuple(first_conv.shape[2:]) != (3, 3):
                raise ValueError("checkpoint first convolution shape is not the released 3x3 layer")
            return
        first_embed = next(
            (value for key, value in checkpoint.items() if key.endswith("patch_embeds.0.proj.weight")),
            None,
        )
        if first_embed is None or len(first_embed.shape) != 4 or first_embed.shape[1] != self.input_spec.channels:
            raise ValueError("SegFormer first patch embedding does not match InputSpec channels")
        if tuple(first_embed.shape[2:]) != (7, 7):
            raise ValueError("SegFormer first patch embedding must be 7x7")


def load_model_manifest(path: str | Path, checkpoint_path: str | Path | None = None) -> ModelManifest:
    manifest = ModelManifest.from_file(path)
    if checkpoint_path is not None:
        manifest.verify_checkpoint(checkpoint_path)
    return manifest
