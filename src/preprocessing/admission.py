"""Two-tier resource admission for preprocessing runs."""

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .contracts import ComputeProfile
from .errors import FailureReason, PreprocessError, RunState
from .resolver import ResolvedContracts


def _sum_components(components: Mapping[str, int]) -> int:
    return sum(int(value) for value in components.values())


def _byte_count(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a non-negative integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be a non-negative integer") from error
    if result < 0 or result != value:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return result


def _scaled_bytes(byte_count: int, multiplier: float) -> int:
    """Apply the same conservative staging rule at every admission tier."""

    return math.ceil(_byte_count(byte_count, "byte_count") * multiplier)


@dataclass(frozen=True)
class ResourceEstimate:
    """Conservative upper bounds calculated before decoder/backend allocation."""

    ram_components: Mapping[str, int]
    disk_components: Mapping[str, int]
    output_bytes: int
    target_shape: tuple[int, ...]
    strip_rows: int
    compressed_full_decode: bool = False

    def __post_init__(self) -> None:
        ram = {str(key): max(0, int(value)) for key, value in dict(self.ram_components).items()}
        disk = {str(key): max(0, int(value)) for key, value in dict(self.disk_components).items()}
        object.__setattr__(self, "ram_components", MappingProxyType(ram))
        object.__setattr__(self, "disk_components", MappingProxyType(disk))
        object.__setattr__(self, "output_bytes", max(0, int(self.output_bytes)))
        object.__setattr__(self, "target_shape", tuple(int(value) for value in self.target_shape))
        object.__setattr__(self, "strip_rows", int(self.strip_rows))

    @property
    def ram_peak_bytes(self) -> int:
        return _sum_components(self.ram_components)

    @property
    def disk_required_bytes(self) -> int:
        return _sum_components(self.disk_components)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "ram_components": dict(self.ram_components),
            "disk_components": dict(self.disk_components),
            "ram_peak_bytes": self.ram_peak_bytes,
            "disk_required_bytes": self.disk_required_bytes,
            "output_bytes": self.output_bytes,
            "target_shape": self.target_shape,
            "strip_rows": self.strip_rows,
            "compressed_full_decode": self.compressed_full_decode,
        }


@dataclass(frozen=True)
class ResourceSnapshot:
    available_ram_bytes: int | None = None
    free_disk_bytes: int | None = None


class SystemResourceProbe:
    """Best-effort host probe; runtime admission remains fail-closed if unavailable."""

    def snapshot(self, path: str | Path) -> ResourceSnapshot:
        available_ram = None
        try:
            import psutil  # type: ignore

            available_ram = int(psutil.virtual_memory().available)
        except (ImportError, OSError, AttributeError):
            pass
        disk_path = Path(path)
        while not disk_path.exists() and disk_path != disk_path.parent:
            disk_path = disk_path.parent
        try:
            free_disk = int(shutil.disk_usage(disk_path).free)
        except OSError:
            free_disk = None
        return ResourceSnapshot(available_ram_bytes=available_ram, free_disk_bytes=free_disk)


@dataclass(frozen=True)
class StaticResourceProbe:
    """Deterministic probe for tests and deployment admission evidence."""

    available_ram_bytes: int | None
    free_disk_bytes: int | None

    def snapshot(self, path: str | Path) -> ResourceSnapshot:
        return ResourceSnapshot(self.available_ram_bytes, self.free_disk_bytes)


@dataclass(frozen=True)
class AdmissionReceipt:
    estimate: ResourceEstimate
    compute_profile_id: str
    output_target: Path
    reservation_id: str

    def to_mapping(self) -> dict[str, Any]:
        return {
            "estimate": self.estimate.to_mapping(),
            "compute_profile_id": self.compute_profile_id,
            "output_target": str(self.output_target),
            "reservation_id": self.reservation_id,
        }


@dataclass(frozen=True)
class RuntimeAdmission:
    receipt: AdmissionReceipt
    stage: str
    actual_peak_ram_bytes: int
    actual_disk_required_bytes: int
    observed_free_disk_bytes: int

    def to_mapping(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "actual_peak_ram_bytes": self.actual_peak_ram_bytes,
            "actual_disk_required_bytes": self.actual_disk_required_bytes,
            "observed_free_disk_bytes": self.observed_free_disk_bytes,
            "receipt": self.receipt.to_mapping(),
        }


class ResourceAdmission:
    """Perform preflight and runtime/commit checks without allocating resources."""

    def __init__(self, compute_profile: ComputeProfile, *, probe: Any | None = None):
        self.compute_profile = compute_profile
        self.probe = probe

    def estimate(self, resolved: ResolvedContracts) -> ResourceEstimate:
        profile = resolved.preprocessing_profile
        capture = resolved.capture_manifest
        source = resolved.source
        compute = self.compute_profile
        layout = capture.source_layout
        source_rows = capture.dimensions[layout.index("Y")]
        source_cols = capture.dimensions[layout.index("X")]
        channels = profile.output_channels
        input_bytes = np.dtype(capture.source_dtype).itemsize
        output_bytes_per_sample = np.dtype(profile.output_dtype).itemsize
        mask_bytes = np.dtype(profile.validity_encoding.dtype).itemsize
        reason_bytes = np.dtype(profile.reason_encoding.dtype).itemsize
        precision_bytes = np.dtype(np.float32 if profile.internal_numeric_precision == "float32" else np.float64).itemsize
        strip_rows = min(profile.target_grid.rows, compute.max_strip_rows)
        target_cols = profile.target_grid.cols
        source_strip_rows = min(source_rows, strip_rows + 2 * profile.halo.rows)
        decoder_strip = source_cols * source_strip_rows * channels * input_bytes
        decoder_allocation = decoder_strip * compute.decoder_workers
        float_warp = target_cols * strip_rows * channels * precision_bytes
        validity = target_cols * strip_rows * mask_bytes
        reason = target_cols * strip_rows * reason_bytes
        mapping = target_cols * strip_rows * 2 * precision_bytes
        per_strip = decoder_strip + float_warp + validity + reason + mapping
        queue = per_strip * compute.queue_depth * compute.inflight_strips
        full_decode = 0
        if source.compressed:
            if not compute.allow_compressed_full_decode:
                raise PreprocessError(
                    FailureReason.CODEC_UNAVAILABLE,
                    "compressed source requires an explicitly admitted bounded full-decode path",
                    state=RunState.RESOURCE_REJECTED,
                    provenance={"source_format": source.format},
                )
            if source.full_decode_bytes is None:
                raise PreprocessError(
                    FailureReason.RESOURCE_PREFLIGHT,
                    "compressed source has no proven full-decode memory bound",
                    state=RunState.RESOURCE_REJECTED,
                    provenance={"source_format": source.format},
                )
            if compute.max_full_decode_bytes is not None and source.full_decode_bytes > compute.max_full_decode_bytes:
                raise PreprocessError(
                    FailureReason.RESOURCE_PREFLIGHT,
                    "compressed source full-decode bound exceeds compute profile",
                    state=RunState.RESOURCE_REJECTED,
                    provenance={
                        "full_decode_bytes": source.full_decode_bytes,
                        "max_full_decode_bytes": compute.max_full_decode_bytes,
                    },
                )
            full_decode = source.full_decode_bytes
        ram_components = {
            "os_obc_reserve": compute.os_obc_reserve_bytes,
            "decoder_source_strip_halo": decoder_allocation,
            "float_warp_working": float_warp,
            "validity_buffer": validity,
            "reason_buffer": reason,
            "mapping_buffer": mapping,
            "bounded_queue": queue,
            "compressed_full_decode": full_decode,
            "safety_margin": compute.safety_margin_bytes,
        }
        target_pixels = profile.target_grid.rows * profile.target_grid.cols
        image_output = target_pixels * channels * output_bytes_per_sample
        validity_output = target_pixels * mask_bytes
        reason_output = target_pixels * reason_bytes
        mapping_output = target_pixels * 2 * precision_bytes
        output_bytes = image_output + validity_output + reason_output + mapping_output
        staging = _scaled_bytes(output_bytes, compute.artifact_staging_multiplier)
        temporary = int(source.metadata.get("temporary_bytes", 0)) if source.metadata else 0
        disk_components = {
            "temporary": max(0, temporary),
            "artifact_staging": staging,
            "checksum_headroom": compute.checksum_headroom_bytes,
            "safety_margin": compute.safety_margin_bytes,
        }
        return ResourceEstimate(
            ram_components=ram_components,
            disk_components=disk_components,
            output_bytes=output_bytes,
            target_shape=profile.output_shape,
            strip_rows=strip_rows,
            compressed_full_decode=bool(source.compressed),
        )

    def preflight(self, resolved: ResolvedContracts) -> AdmissionReceipt:
        estimate = self.estimate(resolved)
        compute = self.compute_profile
        if estimate.ram_peak_bytes > compute.ram_budget_bytes:
            raise PreprocessError(
                FailureReason.RESOURCE_PREFLIGHT,
                "preflight RAM upper bound exceeds compute profile",
                state=RunState.RESOURCE_REJECTED,
                provenance={"estimate": estimate.to_mapping(), "ram_budget_bytes": compute.ram_budget_bytes},
            )
        if estimate.disk_required_bytes > compute.disk_budget_bytes:
            raise PreprocessError(
                FailureReason.RESOURCE_PREFLIGHT,
                "preflight disk upper bound exceeds compute profile",
                state=RunState.RESOURCE_REJECTED,
                provenance={"estimate": estimate.to_mapping(), "disk_budget_bytes": compute.disk_budget_bytes},
            )
        if self.probe is not None:
            snapshot = self.probe.snapshot(resolved.output_artifact_target)
            if snapshot.available_ram_bytes is not None and snapshot.available_ram_bytes < estimate.ram_peak_bytes:
                raise PreprocessError(
                    FailureReason.RESOURCE_PREFLIGHT,
                    "available RAM is below the conservative preflight bound",
                    state=RunState.RESOURCE_REJECTED,
                    provenance={"available_ram_bytes": snapshot.available_ram_bytes, "required_ram_bytes": estimate.ram_peak_bytes},
                )
            if snapshot.free_disk_bytes is not None and snapshot.free_disk_bytes < estimate.disk_required_bytes:
                raise PreprocessError(
                    FailureReason.RESOURCE_PREFLIGHT,
                    "free disk is below the conservative preflight bound",
                    state=RunState.RESOURCE_REJECTED,
                    provenance={"free_disk_bytes": snapshot.free_disk_bytes, "required_disk_bytes": estimate.disk_required_bytes},
                )
        reservation_id = f"{compute.compute_profile_id}:{resolved.capture_manifest.capture_id}:{resolved.preprocessing_profile.fingerprint[:16]}"
        return AdmissionReceipt(
            estimate=estimate,
            compute_profile_id=compute.compute_profile_id,
            output_target=resolved.output_artifact_target,
            reservation_id=reservation_id,
        )

    admit_preflight = preflight

    def runtime_check(
        self,
        receipt: AdmissionReceipt,
        *,
        actual_peak_ram_bytes: int | None = None,
        free_disk_bytes: int | None = None,
        temporary_bytes: int | None = None,
        result_bytes: int | None = None,
        headroom_bytes: int | None = None,
        stage: str = "runtime",
    ) -> RuntimeAdmission:
        """Check measured runtime usage before allocation or publication.

        ``result_bytes`` is the logical materialized output size.  The
        configured staging multiplier is applied here and in ``estimate()``.
        """

        if actual_peak_ram_bytes is None and self.probe is not None:
            # A host's available RAM is useful for preflight, but is not a
            # measured peak allocation after backend initialization.
            actual_peak_ram_bytes = None
        if free_disk_bytes is None and self.probe is not None:
            free_disk_bytes = self.probe.snapshot(receipt.output_target).free_disk_bytes
        if actual_peak_ram_bytes is None or free_disk_bytes is None:
            raise PreprocessError(
                FailureReason.RESOURCE_OBSERVATION_UNAVAILABLE,
                "runtime admission requires measured peak RAM and free disk observations",
                state=RunState.RESOURCE_REJECTED,
                provenance={"stage": stage},
            )
        actual_peak_ram_bytes = _byte_count(actual_peak_ram_bytes, "actual_peak_ram_bytes")
        free_disk_bytes = _byte_count(free_disk_bytes, "free_disk_bytes")
        if temporary_bytes is None:
            temporary_bytes = receipt.estimate.disk_components.get("temporary", 0)
        if result_bytes is None:
            result_bytes = receipt.estimate.output_bytes
        if headroom_bytes is None:
            headroom_bytes = (
                receipt.estimate.disk_components.get("checksum_headroom", 0)
                + receipt.estimate.disk_components.get("safety_margin", 0)
            )
        # ``result_bytes`` is the logical materialized output size.  Apply the
        # same staging multiplier used by preflight so runtime cannot reserve
        # less disk than the conservative estimate.
        staged_result_bytes = _scaled_bytes(result_bytes, self.compute_profile.artifact_staging_multiplier)
        temporary_bytes = _byte_count(temporary_bytes, "temporary_bytes")
        headroom_bytes = _byte_count(headroom_bytes, "headroom_bytes")
        actual_disk_required = temporary_bytes + staged_result_bytes + headroom_bytes
        if actual_peak_ram_bytes > self.compute_profile.ram_budget_bytes:
            raise PreprocessError(
                FailureReason.RESOURCE_RUNTIME,
                "runtime RAM peak exceeds compute profile",
                state=RunState.RESOURCE_REJECTED,
                provenance={
                    "stage": stage,
                    "actual_peak_ram_bytes": actual_peak_ram_bytes,
                    "ram_budget_bytes": self.compute_profile.ram_budget_bytes,
                },
            )
        if actual_disk_required > self.compute_profile.disk_budget_bytes or actual_disk_required > free_disk_bytes:
            raise PreprocessError(
                FailureReason.RESOURCE_RUNTIME,
                "runtime disk requirement exceeds budget or free space",
                state=RunState.RESOURCE_REJECTED,
                provenance={
                    "stage": stage,
                    "actual_disk_required_bytes": actual_disk_required,
                    "result_bytes": int(result_bytes),
                    "artifact_staging_bytes": staged_result_bytes,
                    "disk_budget_bytes": self.compute_profile.disk_budget_bytes,
                    "free_disk_bytes": free_disk_bytes,
                },
            )
        return RuntimeAdmission(
            receipt=receipt,
            stage=stage,
            actual_peak_ram_bytes=actual_peak_ram_bytes,
            actual_disk_required_bytes=actual_disk_required,
            observed_free_disk_bytes=free_disk_bytes,
        )

    def check_runtime(self, receipt: AdmissionReceipt, **kwargs: Any) -> RuntimeAdmission:
        return self.runtime_check(receipt, **kwargs)

    def before_publish(self, receipt: AdmissionReceipt, **kwargs: Any) -> RuntimeAdmission:
        kwargs.setdefault("stage", "commit")
        return self.runtime_check(receipt, **kwargs)


__all__ = [
    "AdmissionReceipt",
    "ResourceAdmission",
    "ResourceEstimate",
    "ResourceSnapshot",
    "RuntimeAdmission",
    "StaticResourceProbe",
    "SystemResourceProbe",
]
