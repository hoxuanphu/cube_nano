"""Contract resolution and trust verification for preprocessing runs."""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Mapping

from .contracts import (
    CalibrationBundle,
    CaptureManifest,
    ComputeProfile,
    PreprocessingProfile,
    SourceDescriptor,
    TrustMetadata,
    TrustPolicy,
    canonical_json,
)
from .errors import FailureReason, PreprocessError, RunState


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a source file without invoking an image decoder."""

    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as source:
            while chunk := source.read(chunk_size):
                digest.update(chunk)
    except OSError as exc:
        raise PreprocessError(
            FailureReason.IO_ERROR,
            f"unable to hash source file: {path}",
            state=RunState.IO_FAULT,
            provenance={"path": str(path), "error": str(exc)},
        ) from exc
    return digest.hexdigest()


def _trust_failure(reason: FailureReason, message: str, contract: Any, kind: str) -> PreprocessError:
    return PreprocessError(
        reason,
        message,
        state=RunState.UNTRUSTED_ARTIFACT,
        provenance={"artifact_kind": kind, "fingerprint": getattr(contract, "fingerprint", None)},
    )


def verify_trust(contract: Any, kind: str, policy: TrustPolicy) -> None:
    """Verify a contract's issuer, key, generation, validity window and signature."""

    trust: TrustMetadata | None = getattr(contract, "trust", None)
    if not policy.require_signature:
        return
    if trust is None:
        raise _trust_failure(FailureReason.TRUST_REJECTED, f"unsigned {kind} artifact is not trusted", contract, kind)
    if policy.trusted_issuers and trust.issuer not in policy.trusted_issuers:
        raise _trust_failure(
            FailureReason.ISSUER_UNTRUSTED,
            f"issuer '{trust.issuer}' is not trusted for {kind}",
            contract,
            kind,
        )
    key = policy.trusted_keys.get(trust.key_id)
    if key is None:
        raise _trust_failure(
            FailureReason.KEY_UNTRUSTED,
            f"key '{trust.key_id}' is not trusted for {kind}",
            contract,
            kind,
        )
    expected_generation = policy.expected_generations.get(kind)
    if expected_generation is not None and trust.generation != expected_generation:
        raise _trust_failure(
            FailureReason.GENERATION_MISMATCH,
            f"{kind} generation {trust.generation} does not match expected {expected_generation}",
            contract,
            kind,
        )
    now = policy.verification_time()
    skew = timedelta(seconds=policy.clock_skew_seconds)
    issued = trust.issued_at
    expires = trust.expires_at
    from .contracts import _parse_datetime  # local import keeps helper private to contract module

    issued_at = _parse_datetime(issued, f"{kind}.trust.issued_at")
    expires_at = _parse_datetime(expires, f"{kind}.trust.expires_at")
    if issued_at > now + skew:
        raise _trust_failure(FailureReason.TRUST_NOT_YET_VALID, f"{kind} trust envelope is not yet valid", contract, kind)
    if expires_at < now - skew:
        raise _trust_failure(FailureReason.TRUST_EXPIRED, f"{kind} trust envelope has expired", contract, kind)
    expected_signature = hmac.new(key, contract.fingerprint.encode("ascii"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected_signature, trust.signature):
        raise _trust_failure(FailureReason.SIGNATURE_INVALID, f"{kind} signature does not match its fingerprint", contract, kind)


def _invalid(message: str, *, reason: FailureReason = FailureReason.INVALID_REQUEST, provenance: Mapping[str, Any] | None = None) -> PreprocessError:
    return PreprocessError(reason, message, state=RunState.INVALID_INPUT, provenance=provenance)


@dataclass(frozen=True)
class ResolvedContracts:
    """Validated, trusted contracts ready for resource preflight."""

    source: SourceDescriptor
    capture_manifest: CaptureManifest
    preprocessing_profile: PreprocessingProfile
    calibration_bundle: CalibrationBundle
    compute_profile: ComputeProfile
    output_artifact_target: Path
    provenance: Mapping[str, Any]


class ContractResolver:
    """Resolve request contracts before decoder, backend, or engine allocation."""

    def __init__(self, *, trust_policy: TrustPolicy | None = None, verify_source_fingerprint: bool = True):
        self.trust_policy = trust_policy
        self.verify_source_fingerprint = verify_source_fingerprint

    def resolve(self, request: Any) -> ResolvedContracts:
        from .api import PreprocessRequest

        if not isinstance(request, PreprocessRequest):
            raise _invalid("preprocess_capture requires a PreprocessRequest", reason=FailureReason.REQUEST_TYPE_INVALID)
        try:
            capture = request.capture_manifest
            profile = request.preprocessing_profile
            calibration = request.calibration_bundle
            compute = request.compute_profile
            source = SourceDescriptor.from_value(request.source)
            if not isinstance(capture, CaptureManifest):
                capture = CaptureManifest.from_mapping(capture)
            if not isinstance(profile, PreprocessingProfile):
                profile = PreprocessingProfile.from_mapping(profile)
            if not isinstance(calibration, CalibrationBundle):
                calibration = CalibrationBundle.from_mapping(calibration)
            if not isinstance(compute, ComputeProfile):
                compute = ComputeProfile.from_mapping(compute)
        except (TypeError, ValueError, KeyError) as exc:
            raise _invalid(f"invalid preprocessing contract: {exc}", reason=FailureReason.SCHEMA_MISMATCH) from exc

        # Trust is evaluated before source existence, stat, hashing, decoding, or allocation.
        policy = self.trust_policy or request.trust_policy or TrustPolicy()
        for kind, contract in (
            ("capture", capture),
            ("profile", profile),
            ("calibration", calibration),
            ("compute", compute),
        ):
            verify_trust(contract, kind, policy)

        if not capture.completion_marker:
            raise PreprocessError(
                FailureReason.CAPTURE_INCOMPLETE,
                "capture completion marker is false",
                state=RunState.INVALID_INPUT,
                provenance={"capture_id": capture.capture_id},
            )
        self._check_compatibility(capture, profile, calibration)
        self._check_source_descriptor(source, capture)

        if source.path is not None:
            if not source.path.is_file():
                raise PreprocessError(
                    FailureReason.SOURCE_NOT_FOUND,
                    f"source file does not exist: {source.path}",
                    state=RunState.INVALID_INPUT,
                    provenance={"source_path": str(source.path)},
                )
            if source.size_bytes is not None:
                try:
                    actual_size = source.path.stat().st_size
                except OSError as exc:
                    raise PreprocessError(
                        FailureReason.IO_ERROR,
                        f"unable to stat source file: {source.path}",
                        state=RunState.IO_FAULT,
                        provenance={"source_path": str(source.path), "error": str(exc)},
                    ) from exc
                if actual_size != source.size_bytes:
                    raise PreprocessError(
                        FailureReason.SOURCE_FINGERPRINT_MISMATCH,
                        "source descriptor size does not match the source file",
                        state=RunState.INVALID_INPUT,
                        provenance={"expected_size_bytes": source.size_bytes, "actual_size_bytes": actual_size},
                    )
            if self.verify_source_fingerprint and getattr(request, "verify_source_fingerprint", True):
                actual_digest = sha256_file(source.path)
                if not hmac.compare_digest(actual_digest, capture.source_fingerprint):
                    raise PreprocessError(
                        FailureReason.SOURCE_FINGERPRINT_MISMATCH,
                        "capture source_fingerprint does not match the source file",
                        state=RunState.INVALID_INPUT,
                        provenance={"expected": capture.source_fingerprint, "actual": actual_digest},
                    )
        elif self.verify_source_fingerprint and getattr(request, "verify_source_fingerprint", True):
            descriptor_digest = source.metadata.get("source_fingerprint")
            if descriptor_digest is None or str(descriptor_digest).lower() != capture.source_fingerprint:
                raise PreprocessError(
                    FailureReason.SOURCE_FINGERPRINT_MISMATCH,
                    "non-file source descriptor does not carry the expected fingerprint",
                    state=RunState.INVALID_INPUT,
                    provenance={"expected": capture.source_fingerprint},
                )

        output_target = Path(request.output_artifact_target)
        if not str(output_target).strip():
            raise _invalid("output_artifact_target must not be empty")
        return ResolvedContracts(
            source=source,
            capture_manifest=capture,
            preprocessing_profile=profile,
            calibration_bundle=calibration,
            compute_profile=compute,
            output_artifact_target=output_target,
            provenance={
                "capture_id": capture.capture_id,
                "profile_id": profile.profile_id,
                "profile_fingerprint": profile.fingerprint,
                "calibration_fingerprint": calibration.fingerprint,
                "compute_profile_id": compute.compute_profile_id,
            },
        )

    @staticmethod
    def _check_source_descriptor(source: SourceDescriptor, capture: CaptureManifest) -> None:
        if capture.source_path and source.path:
            try:
                if Path(capture.source_path).resolve() != source.path.resolve():
                    raise PreprocessError(
                        FailureReason.FINGERPRINT_LINKAGE_MISMATCH,
                        "capture source_path does not match request source",
                        state=RunState.INVALID_INPUT,
                        provenance={"manifest_source_path": capture.source_path, "request_source_path": str(source.path)},
                    )
            except OSError:
                if Path(capture.source_path) != source.path:
                    raise PreprocessError(
                        FailureReason.FINGERPRINT_LINKAGE_MISMATCH,
                        "capture source_path does not match request source",
                        state=RunState.INVALID_INPUT,
                    )

    @staticmethod
    def _check_compatibility(capture: CaptureManifest, profile: PreprocessingProfile, calibration: CalibrationBundle) -> None:
        source_schema = profile.source_schema
        if source_schema.axes != capture.source_layout:
            raise _invalid(
                "capture source layout does not match preprocessing profile",
                reason=FailureReason.SCHEMA_MISMATCH,
                provenance={"capture_layout": capture.source_layout, "profile_layout": source_schema.axes},
            )
        if source_schema.shape is not None and source_schema.shape != capture.dimensions:
            raise _invalid(
                "capture dimensions do not match preprocessing profile",
                reason=FailureReason.SCHEMA_MISMATCH,
                provenance={"capture_dimensions": capture.dimensions, "profile_shape": source_schema.shape},
            )
        if source_schema.dtype != capture.source_dtype:
            raise _invalid(
                "capture dtype does not match preprocessing profile",
                reason=FailureReason.SCHEMA_MISMATCH,
                provenance={"capture_dtype": capture.source_dtype, "profile_dtype": source_schema.dtype},
            )
        if source_schema.byte_order != capture.source_byte_order:
            raise _invalid(
                "capture byte order does not match preprocessing profile",
                reason=FailureReason.SCHEMA_MISMATCH,
                provenance={"capture_byte_order": capture.source_byte_order, "profile_byte_order": source_schema.byte_order},
            )
        if tuple(source_schema.channels) != tuple(capture.channel_schema):
            raise _invalid(
                "capture channel schema does not match preprocessing profile",
                reason=FailureReason.SCHEMA_MISMATCH,
                provenance={"capture_channels": capture.channel_schema, "profile_channels": source_schema.channels},
            )
        if canonical_json(source_schema.nodata_semantics) != canonical_json(capture.nodata_semantics):
            raise _invalid(
                "capture NoData semantics do not match preprocessing profile",
                reason=FailureReason.SCHEMA_MISMATCH,
                provenance={
                    "capture_nodata": capture.nodata_semantics,
                    "profile_nodata": source_schema.nodata_semantics,
                },
            )
        selector = profile.calibration_selector
        if capture.sensor_product != selector.sensor_product or calibration.sensor_product != selector.sensor_product:
            raise PreprocessError(
                FailureReason.CALIBRATION_UNSUPPORTED,
                "sensor product is not compatible with the calibration selector",
                state=RunState.CALIBRATION_ERROR,
                provenance={"capture_sensor": capture.sensor_product, "selector_sensor": selector.sensor_product},
            )
        if calibration.calibration_id != selector.calibration_id or calibration.version != selector.version:
            raise PreprocessError(
                FailureReason.CALIBRATION_UNSUPPORTED,
                "calibration id/version does not match the preprocessing selector",
                state=RunState.CALIBRATION_ERROR,
                provenance={
                    "expected_calibration_id": selector.calibration_id,
                    "actual_calibration_id": calibration.calibration_id,
                    "expected_version": selector.version,
                    "actual_version": calibration.version,
                },
            )
        if calibration.transform_type != profile.transform_type or calibration.transform_type not in selector.allowed_transform_types:
            raise PreprocessError(
                FailureReason.CALIBRATION_UNSUPPORTED,
                "calibration transform type is not accepted by the profile",
                state=RunState.CALIBRATION_ERROR,
                provenance={"profile_transform": profile.transform_type, "calibration_transform": calibration.transform_type},
            )
        if calibration.transform_direction != profile.transform_direction:
            raise PreprocessError(
                FailureReason.CALIBRATION_UNSUPPORTED,
                "calibration transform direction does not match profile",
                state=RunState.CALIBRATION_ERROR,
                provenance={"profile_direction": profile.transform_direction, "calibration_direction": calibration.transform_direction},
            )
        if calibration.pixel_convention != profile.pixel_convention:
            raise PreprocessError(
                FailureReason.CALIBRATION_UNSUPPORTED,
                "calibration pixel convention does not match profile",
                state=RunState.CALIBRATION_ERROR,
                provenance={"profile_convention": profile.pixel_convention, "calibration_convention": calibration.pixel_convention},
            )
        for metric, limit in selector.quality_limits.items():
            if metric not in calibration.quality or calibration.quality[metric] > limit:
                raise PreprocessError(
                    FailureReason.CALIBRATION_UNSUPPORTED,
                    f"calibration quality metric '{metric}' is outside the profile limit",
                    state=RunState.CALIBRATION_ERROR,
                    provenance={"metric": metric, "limit": limit, "actual": calibration.quality.get(metric)},
                )
        if capture.preprocessing_profile_digest and capture.preprocessing_profile_digest != profile.fingerprint:
            raise PreprocessError(
                FailureReason.FINGERPRINT_LINKAGE_MISMATCH,
                "capture is linked to a different preprocessing profile",
                state=RunState.INVALID_INPUT,
                provenance={"expected": capture.preprocessing_profile_digest, "actual": profile.fingerprint},
            )
        if capture.calibration_digest and capture.calibration_digest != calibration.fingerprint:
            raise PreprocessError(
                FailureReason.FINGERPRINT_LINKAGE_MISMATCH,
                "capture is linked to a different calibration bundle",
                state=RunState.INVALID_INPUT,
                provenance={"expected": capture.calibration_digest, "actual": calibration.fingerprint},
            )


def resolve_contracts(request: Any, *, trust_policy: TrustPolicy | None = None) -> ResolvedContracts:
    return ContractResolver(trust_policy=trust_policy).resolve(request)


__all__ = ["ContractResolver", "ResolvedContracts", "resolve_contracts", "sha256_file", "verify_trust"]
