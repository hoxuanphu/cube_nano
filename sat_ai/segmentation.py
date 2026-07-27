"""SegFormer logits postprocessing and pixel-valid product semantics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from .contracts import ModelOutputSpec, PostprocessSpec, ProductSpec


@dataclass(frozen=True)
class SegmentationTile:
    cloud_probability: np.ndarray
    cloud_mask: np.ndarray
    validity_mask: np.ndarray


def cloud_probabilities_from_logits(logits: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
    """Apply the release postprocess order: softmax first, then bilinear resize."""

    if logits.ndim != 4 or logits.shape[1] != 2:
        raise ValueError(f"expected segmentation logits [N, 2, h, w], got {tuple(logits.shape)}")
    if len(target_size) != 2 or any(int(value) <= 0 for value in target_size):
        raise ValueError("target_size must contain two positive spatial dimensions")
    probabilities = F.softmax(logits.float(), dim=1)
    return F.interpolate(
        probabilities,
        size=target_size,
        mode="bilinear",
        align_corners=False,
    )


def postprocess_segmentation_logits(
    logits: np.ndarray,
    validity_mask: np.ndarray,
    *,
    target_size: tuple[int, int],
    threshold_bp: int,
    output_spec: ModelOutputSpec | None = None,
    postprocess_spec: PostprocessSpec | None = None,
    product_spec: ProductSpec | None = None,
) -> SegmentationTile:
    """Resize logits once, threshold cloud probability, then apply validity."""

    logits = np.asarray(logits, dtype=np.float32)
    if logits.ndim != 4 or logits.shape[1] != 2:
        raise ValueError(f"expected segmentation logits [N, 2, h, w], got {logits.shape}")
    if logits.shape[0] <= 0 or not np.isfinite(logits).all():
        raise ValueError("segmentation logits must be finite and non-empty")
    validity = np.asarray(validity_mask, dtype=bool)
    if validity.ndim == 2:
        validity = validity[None, ...]
    if validity.shape != (logits.shape[0], target_size[0], target_size[1]):
        raise ValueError(
            "segmentation validity shape must match batch and target spatial shape: "
            f"validity={validity.shape}, expected={(logits.shape[0], *target_size)}"
        )
    if not isinstance(threshold_bp, int) or isinstance(threshold_bp, bool) or not 0 <= threshold_bp <= 10000:
        raise ValueError("threshold_bp must be an integer in [0, 10000]")
    if output_spec is not None:
        output_spec.validate()
        if tuple(logits.shape) != output_spec.shape and logits.shape[0] == 1:
            raise ValueError(f"logits shape {logits.shape} does not match {output_spec.shape}")
    if postprocess_spec is not None:
        postprocess_spec.validate()
    if product_spec is not None:
        product_spec.validate()

    tensor = torch.from_numpy(np.ascontiguousarray(logits))
    probabilities = cloud_probabilities_from_logits(tensor, target_size)[:, 1]
    probability_array = probabilities.numpy()
    threshold = np.float32(threshold_bp / 10000.0)
    cloud = probability_array >= threshold
    cloud = np.logical_and(cloud, validity)
    cloud_value = 255 if product_spec is None else product_spec.cloud_value
    valid_value = 1 if product_spec is None else product_spec.valid_value
    invalid_value = 0 if product_spec is None else product_spec.invalid_value
    return SegmentationTile(
        cloud_probability=probability_array.astype(np.float32, copy=False),
        cloud_mask=np.where(cloud, cloud_value, 0).astype(np.uint8),
        validity_mask=np.where(validity, valid_value, invalid_value).astype(np.uint8),
    )
