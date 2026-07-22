"""Offline-generated float32 logit threshold LUT and integer decisions."""

from __future__ import annotations

import hashlib
import math
import struct
from decimal import Decimal, getcontext
from pathlib import Path

from protocol.canonical import MAX_U64, checked_u16

LUT_ID = "logit-bp-f32-lut-v1"
LUT_ENTRIES = 10001


class ThresholdLUT:
    def __init__(self, raw: bytes, *, lut_id: str = LUT_ID):
        if lut_id != LUT_ID:
            raise ValueError("unsupported threshold LUT ID")
        if len(raw) != LUT_ENTRIES * 4:
            raise ValueError("threshold LUT must contain exactly 10001 float32 entries")
        self.raw = bytes(raw)
        self.lut_id = lut_id
        self.sha256 = hashlib.sha256(self.raw).hexdigest()

    @classmethod
    def from_file(cls, path: str | Path, expected_sha256: str | None = None) -> "ThresholdLUT":
        lut = cls(Path(path).read_bytes())
        if expected_sha256 is not None and lut.sha256 != expected_sha256:
            raise ValueError("threshold LUT SHA-256 mismatch")
        return lut

    def threshold(self, threshold_bp: int) -> float:
        checked_u16(threshold_bp, "threshold_bp")
        if threshold_bp > 10000:
            raise ValueError("threshold_bp must be in [0, 10000]")
        return struct.unpack(">f", self.raw[threshold_bp * 4 : threshold_bp * 4 + 4])[0]

    def classify(self, logit: float, threshold_bp: int) -> bool:
        if not math.isfinite(float(logit)):
            raise ValueError("non-finite model logit")
        checked_u16(threshold_bp, "threshold_bp")
        if threshold_bp > 10000:
            raise ValueError("threshold_bp must be in [0, 10000]")
        if threshold_bp == 0:
            return True
        if threshold_bp == 10000:
            return False
        return float(logit) >= self.threshold(threshold_bp)


def generate_threshold_lut() -> ThresholdLUT:
    getcontext().prec = 96
    values = []
    for basis_points in range(LUT_ENTRIES):
        if basis_points == 0:
            value = float("-inf")
        elif basis_points == 10000:
            value = float("inf")
        else:
            ratio = Decimal(basis_points) / Decimal(10000 - basis_points)
            value = float(ratio.ln())
        values.append(struct.pack(">f", value))
    return ThresholdLUT(b"".join(values))


def coverage_ratio_bp(cloud_positive_area: int, analyzed_area: int) -> int:
    if not 0 <= cloud_positive_area <= analyzed_area:
        raise ValueError("cloud_positive_area must be within analyzed_area")
    if analyzed_area <= 0:
        raise ValueError("analyzed_area must be positive")
    if cloud_positive_area > MAX_U64 // 10000:
        raise OverflowError("cloud area multiplication exceeds U64")
    return (cloud_positive_area * 10000) // analyzed_area


def coverage_accepted(cloud_positive_area: int, analyzed_area: int, coverage_limit_bp: int) -> bool:
    checked_u16(coverage_limit_bp, "coverage_limit_bp")
    if coverage_limit_bp > 10000:
        raise ValueError("coverage_limit_bp must be in [0, 10000]")
    if analyzed_area <= 0 or not 0 <= cloud_positive_area <= analyzed_area:
        raise ValueError("invalid coverage areas")
    if cloud_positive_area > MAX_U64 // 10000 or coverage_limit_bp * analyzed_area > MAX_U64:
        raise OverflowError("coverage comparison exceeds U64")
    return cloud_positive_area * 10000 < coverage_limit_bp * analyzed_area
