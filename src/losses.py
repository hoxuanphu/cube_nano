"""Masked semantic segmentation losses used by the SegFormer training loop."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def _valid_pixels(target: Tensor, validity_mask: Tensor | None, ignore_index: int) -> Tensor:
    valid = target != ignore_index
    if validity_mask is not None:
        if validity_mask.shape != target.shape:
            raise ValueError("validity_mask shape must match target shape")
        valid = valid & validity_mask.to(torch.bool)
    return valid


def resize_logits_to_target(logits: Tensor, target: Tensor) -> Tensor:
    """Upsample decoder logits to the native target size without resizing labels."""

    if logits.ndim != 4 or logits.shape[1] != 2:
        raise ValueError(f"expected logits [B, 2, H, W], got {tuple(logits.shape)}")
    if target.ndim != 3 or target.shape[0] != logits.shape[0]:
        raise ValueError("segmentation target must be [B, H, W] with the same batch size as logits")
    if target.shape[-2:] == logits.shape[-2:]:
        return logits
    return F.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)


def soft_dice_loss(
    logits: Tensor,
    target: Tensor,
    *,
    validity_mask: Tensor | None = None,
    ignore_index: int = 255,
    epsilon: float = 1e-6,
) -> Tensor:
    """Cloud-class soft Dice over valid pixels in the whole mini-batch."""

    logits = resize_logits_to_target(logits, target)
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    valid = _valid_pixels(target, validity_mask, ignore_index)
    if not torch.any(valid):
        return logits.sum() * 0.0
    probability = torch.softmax(logits.float(), dim=1)[:, 1]
    cloud_target = (target == 1).to(probability.dtype)
    valid_float = valid.to(probability.dtype)
    probability = probability * valid_float
    cloud_target = cloud_target * valid_float
    intersection = (probability * cloud_target).sum()
    denominator = probability.sum() + cloud_target.sum()
    dice = (2.0 * intersection + epsilon) / (denominator + epsilon)
    return 1.0 - dice


def masked_segmentation_loss(
    logits: Tensor,
    target: Tensor,
    *,
    validity_mask: Tensor | None = None,
    ignore_index: int = 255,
    cross_entropy_weight: float = 1.0,
    dice_weight: float = 1.0,
    epsilon: float = 1e-6,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Return CE + Dice and its components, safely handling all-invalid batches."""

    logits = resize_logits_to_target(logits, target)
    valid = _valid_pixels(target, validity_mask, ignore_index)
    if cross_entropy_weight < 0 or dice_weight < 0 or cross_entropy_weight + dice_weight <= 0:
        raise ValueError("loss weights must be non-negative and not both zero")
    if torch.any(valid):
        cross_entropy_target = torch.where(
            valid,
            target,
            torch.full_like(target, ignore_index),
        )
        cross_entropy = F.cross_entropy(logits, cross_entropy_target, ignore_index=ignore_index)
    else:
        cross_entropy = logits.sum() * 0.0
    dice = soft_dice_loss(
        logits,
        target,
        validity_mask=validity_mask,
        ignore_index=ignore_index,
        epsilon=epsilon,
    )
    total = cross_entropy_weight * cross_entropy + dice_weight * dice
    return total, {"cross_entropy": cross_entropy, "soft_dice": dice}


class SoftDiceLoss(torch.nn.Module):
    def __init__(self, *, ignore_index: int = 255, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.epsilon = epsilon

    def forward(self, logits: Tensor, target: Tensor, validity_mask: Tensor | None = None) -> Tensor:
        return soft_dice_loss(
            logits,
            target,
            validity_mask=validity_mask,
            ignore_index=self.ignore_index,
            epsilon=self.epsilon,
        )
