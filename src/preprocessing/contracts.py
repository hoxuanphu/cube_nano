"""Immutable contracts for the dataset/model independent preprocessing boundary.

This module deliberately contains no image decoder, GPU, TensorRT, or filesystem
allocation code.  The contracts are serializable and their fingerprints are
computed from a canonical representation that excludes the trust envelope.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = 1
PREPROCESSING_API_VERSION = "1.0"
PREPROCESS_ARTIFACT_SCHEMA_VERSION = 1


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_freeze(item) for item in value), key=repr))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    """Return the deterministic JSON representation used for fingerprints."""

    return json.dumps(
        _thaw(_freeze(value)),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _content_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def normalize_digest(value: Any, field_name: str = "digest") -> str:
    if isinstance(value, Mapping):
        algorithm = str(value.get("algorithm", "")).lower()
        if algorithm != "sha256":
            raise ValueError(f"{field_name}.algorithm must be 'sha256'")
        value = value.get("digest")
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field_name} must be a 64-character SHA-256 digest")
    try:
        bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must contain hexadecimal characters") from exc
    return value.lower()


def _text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive integer") from exc
    if result <= 0 or result != value:
        raise ValueError(f"{field_name} must be a positive integer")
    return result


def _non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a non-negative integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a non-negative integer") from exc
    if result < 0 or result != value:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return result


def _finite_float(value: Any, field_name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be a finite number")
    return result


def _number_tuple(value: Any, length: int, field_name: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{field_name} must contain exactly {length} numbers")
    return tuple(_finite_float(item, f"{field_name}[{index}]") for index, item in enumerate(value))


def _int_tuple(value: Any, field_name: str, *, length: int | None = None) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or (length is not None and len(value) != length):
        expected = f" exactly {length}" if length is not None else ""
        raise ValueError(f"{field_name} must contain{expected} integers")
    result = tuple(_positive_int(item, f"{field_name}[{index}]") for index, item in enumerate(value))
    if not result:
        raise ValueError(f"{field_name} must not be empty")
    return result


def _normalize_dtype(value: Any, field_name: str) -> str:
    if not isinstance(value, (str, np.dtype, type)):
        raise ValueError(f"{field_name} must be a numeric dtype")
    try:
        dtype = np.dtype(value)
    except TypeError as exc:
        raise ValueError(f"{field_name} must be a numeric dtype") from exc
    if dtype.kind not in "biufc":
        raise ValueError(f"{field_name} must be an integer or floating-point dtype")
    return dtype.name


def _normalize_choice(value: Any, field_name: str, choices: Mapping[str, str]) -> str:
    normalized = _text(value, field_name).lower().replace("-", "_").replace(" ", "_")
    normalized = choices.get(normalized, normalized)
    if normalized not in set(choices.values()):
        allowed = ", ".join(sorted(set(choices.values())))
        raise ValueError(f"{field_name} must be one of: {allowed}")
    return normalized


def _parse_datetime(value: Any, field_name: str) -> datetime:
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, str):
        text_value = value.strip()
        if text_value.endswith("Z"):
            text_value = text_value[:-1] + "+00:00"
        try:
            result = datetime.fromisoformat(text_value)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be an ISO-8601 datetime") from exc
    else:
        raise ValueError(f"{field_name} must be an ISO-8601 datetime")
    if result.tzinfo is None:
        raise ValueError(f"{field_name} must include a timezone")
    return result.astimezone(timezone.utc)


def _datetime_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class TrustMetadata:
    """Signed trust envelope attached to a contract artifact."""

    issuer: str = ""
    key_id: str = ""
    generation: int = 0
    issued_at: str | datetime | None = None
    expires_at: str | datetime | None = None
    signature: str = ""
    algorithm: str = "hmac-sha256"

    def __post_init__(self) -> None:
        object.__setattr__(self, "issuer", _text(self.issuer, "trust.issuer"))
        object.__setattr__(self, "key_id", _text(self.key_id, "trust.key_id"))
        object.__setattr__(self, "generation", _positive_int(self.generation, "trust.generation"))
        issued = _parse_datetime(self.issued_at, "trust.issued_at")
        expires = _parse_datetime(self.expires_at, "trust.expires_at")
        if expires <= issued:
            raise ValueError("trust.expires_at must be later than trust.issued_at")
        object.__setattr__(self, "issued_at", _datetime_text(issued))
        object.__setattr__(self, "expires_at", _datetime_text(expires))
        algorithm = str(self.algorithm).lower().replace("_", "-")
        if algorithm != "hmac-sha256":
            raise ValueError("trust.algorithm must be 'hmac-sha256'")
        object.__setattr__(self, "algorithm", algorithm)
        signature = str(self.signature).lower()
        if len(signature) != 64:
            raise ValueError("trust.signature must be a 64-character hexadecimal HMAC")
        try:
            bytes.fromhex(signature)
        except ValueError as exc:
            raise ValueError("trust.signature must contain hexadecimal characters") from exc
        object.__setattr__(self, "signature", signature)

    @classmethod
    def for_digest(
        cls,
        digest: str,
        *,
        issuer: str,
        key_id: str,
        key: bytes | str,
        generation: int,
        issued_at: str | datetime,
        expires_at: str | datetime,
    ) -> "TrustMetadata":
        digest = normalize_digest(digest, "signed digest")
        key_bytes = key.encode("utf-8") if isinstance(key, str) else bytes(key)
        signature = hmac.new(key_bytes, digest.encode("ascii"), hashlib.sha256).hexdigest()
        return cls(
            issuer=issuer,
            key_id=key_id,
            generation=generation,
            issued_at=issued_at,
            expires_at=expires_at,
            signature=signature,
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "issuer": self.issuer,
            "key_id": self.key_id,
            "generation": self.generation,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "signature": self.signature,
            "algorithm": self.algorithm,
        }


@dataclass(frozen=True)
class TrustPolicy:
    """Verification policy. Secure-by-default; use ``development`` explicitly for fixtures."""

    require_signature: bool = True
    trusted_issuers: tuple[str, ...] = ()
    trusted_keys: Mapping[str, bytes | str] = field(default_factory=dict)
    expected_generations: Mapping[str, int] = field(default_factory=dict)
    now: str | datetime | None = None
    clock_skew_seconds: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.require_signature, bool):
            raise ValueError("trust policy require_signature must be boolean")
        issuers = tuple(_text(value, "trust policy issuer") for value in self.trusted_issuers)
        object.__setattr__(self, "trusted_issuers", issuers)
        keys = {}
        for key_id, key in dict(self.trusted_keys).items():
            normalized_id = _text(key_id, "trust policy key_id")
            key_bytes = key.encode("utf-8") if isinstance(key, str) else bytes(key)
            if not key_bytes:
                raise ValueError(f"trust policy key '{normalized_id}' must not be empty")
            keys[normalized_id] = key_bytes
        object.__setattr__(self, "trusted_keys", MappingProxyType(keys))
        generations = {}
        for kind, generation in dict(self.expected_generations).items():
            generations[_text(kind, "trust policy artifact kind")] = _positive_int(
                generation, f"trust policy generation for {kind}"
            )
        object.__setattr__(self, "expected_generations", MappingProxyType(generations))
        if self.now is not None:
            object.__setattr__(self, "now", _datetime_text(_parse_datetime(self.now, "trust policy now")))
        object.__setattr__(self, "clock_skew_seconds", _non_negative_int(self.clock_skew_seconds, "clock_skew_seconds"))

    @classmethod
    def development(cls) -> "TrustPolicy":
        return cls(require_signature=False)

    def verification_time(self) -> datetime:
        return _parse_datetime(self.now, "trust policy now") if self.now else datetime.now(timezone.utc)


@dataclass(frozen=True)
class SourceSchema:
    axes: str = ""
    shape: tuple[int, ...] | None = None
    dtype: str = ""
    channels: tuple[str, ...] = ()
    representation: str = ""
    byte_order: str = ""
    nodata_semantics: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        axes = _text(self.axes, "source_schema.axes").upper()
        if len(set(axes)) != len(axes) or "Y" not in axes or "X" not in axes:
            raise ValueError("source_schema.axes must contain unique X and Y axes")
        object.__setattr__(self, "axes", axes)
        if self.shape is not None:
            shape = _int_tuple(self.shape, "source_schema.shape")
            if len(shape) != len(axes):
                raise ValueError("source_schema.shape rank must match source_schema.axes")
            object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "dtype", _normalize_dtype(self.dtype, "source_schema.dtype"))
        channels = tuple(_text(channel, "source_schema channel") for channel in self.channels)
        if "C" in axes:
            if not channels:
                raise ValueError("source_schema.channels is required when axes contains C")
            if self.shape is not None and self.shape[axes.index("C")] != len(channels):
                raise ValueError("source_schema channel count does not match shape")
        elif channels:
            raise ValueError("source_schema.channels requires a C axis")
        if len(set(channels)) != len(channels):
            raise ValueError("source_schema.channels must be unique")
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "representation", _text(self.representation, "source_schema.representation"))
        byte_order = _text(self.byte_order, "source_schema.byte_order").lower()
        if byte_order not in {"little", "big", "native", "not_applicable"}:
            raise ValueError("source_schema.byte_order must be little, big, native, or not_applicable")
        object.__setattr__(self, "byte_order", byte_order)
        if self.nodata_semantics is not None and not isinstance(self.nodata_semantics, Mapping):
            raise ValueError("source_schema.nodata_semantics must be a mapping or None")
        object.__setattr__(self, "nodata_semantics", _freeze(self.nodata_semantics) if self.nodata_semantics is not None else None)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SourceSchema":
        if not isinstance(value, Mapping):
            raise ValueError("source_schema must be a mapping")
        channels = value.get("channels", value.get("channel_schema", ()))
        if channels and isinstance(channels[0], Mapping):
            channels = tuple(item.get("name", item.get("role", "")) for item in channels)
        return cls(
            axes=value.get("axes", ""),
            shape=value.get("shape"),
            dtype=value.get("dtype", ""),
            channels=tuple(channels or ()),
            representation=value.get("representation", value.get("source_representation", "")),
            byte_order=value.get("byte_order", ""),
            nodata_semantics=value.get("nodata_semantics", value.get("nodata")),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "axes": self.axes,
            "shape": self.shape,
            "dtype": self.dtype,
            "channels": self.channels,
            "representation": self.representation,
            "byte_order": self.byte_order,
            "nodata_semantics": self.nodata_semantics,
        }

    @property
    def channel_count(self) -> int:
        return len(self.channels) if self.channels else 1


@dataclass(frozen=True)
class CalibrationSelector:
    sensor_product: str = ""
    calibration_id: str = ""
    version: str = ""
    allowed_transform_types: tuple[str, ...] = ()
    quality_limits: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sensor_product", _text(self.sensor_product, "calibration_selector.sensor_product"))
        object.__setattr__(self, "calibration_id", _text(self.calibration_id, "calibration_selector.calibration_id"))
        object.__setattr__(self, "version", _text(self.version, "calibration_selector.version"))
        choices = {"identity": "identity", "affine": "affine", "lut": "lut", "brown_conrady": "brown_conrady"}
        transform_types = tuple(
            _normalize_choice(item, "calibration_selector.transform_type", choices)
            for item in self.allowed_transform_types
        )
        if not transform_types:
            raise ValueError("calibration_selector.allowed_transform_types must not be empty")
        if len(set(transform_types)) != len(transform_types):
            raise ValueError("calibration_selector.allowed_transform_types must be unique")
        object.__setattr__(self, "allowed_transform_types", transform_types)
        limits = {}
        for key, value in dict(self.quality_limits).items():
            limits[_text(key, "calibration quality limit")] = _finite_float(value, f"calibration_selector.quality_limits.{key}")
        object.__setattr__(self, "quality_limits", MappingProxyType(limits))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CalibrationSelector":
        if not isinstance(value, Mapping):
            raise ValueError("calibration_selector must be a mapping")
        return cls(
            sensor_product=value.get("sensor_product", value.get("sensor", "")),
            calibration_id=value.get("calibration_id", value.get("id", "")),
            version=value.get("version", ""),
            allowed_transform_types=tuple(value.get("allowed_transform_types", value.get("transform_types", ()))),
            quality_limits=value.get("quality_limits", {}),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "sensor_product": self.sensor_product,
            "calibration_id": self.calibration_id,
            "version": self.version,
            "allowed_transform_types": self.allowed_transform_types,
            "quality_limits": self.quality_limits,
        }


@dataclass(frozen=True)
class CalibrationBundle:
    schema_version: int = SCHEMA_VERSION
    calibration_id: str = ""
    version: str = ""
    sensor_product: str = ""
    transform_type: str = ""
    transform_direction: str = ""
    pixel_convention: str = ""
    parameters: Mapping[str, Any] = field(default_factory=dict)
    quality: Mapping[str, float] = field(default_factory=dict)
    trust: TrustMetadata | None = None
    digest: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"calibration schema_version must be {SCHEMA_VERSION}")
        object.__setattr__(self, "calibration_id", _text(self.calibration_id, "calibration_id"))
        object.__setattr__(self, "version", _text(self.version, "calibration version"))
        object.__setattr__(self, "sensor_product", _text(self.sensor_product, "calibration sensor_product"))
        choices = {"identity": "identity", "affine": "affine", "lut": "lut", "brown_conrady": "brown_conrady"}
        object.__setattr__(self, "transform_type", _normalize_choice(self.transform_type, "calibration transform_type", choices))
        directions = {"source_to_model": "source_to_model", "model_to_source": "model_to_source"}
        object.__setattr__(self, "transform_direction", _normalize_choice(self.transform_direction, "calibration transform_direction", directions))
        conventions = {"center": "center", "corner": "corner"}
        object.__setattr__(self, "pixel_convention", _normalize_choice(self.pixel_convention, "calibration pixel_convention", conventions))
        if not isinstance(self.parameters, Mapping):
            raise ValueError("calibration parameters must be a mapping")
        object.__setattr__(self, "parameters", _freeze(self.parameters))
        quality = {str(key): _finite_float(value, f"calibration quality.{key}") for key, value in dict(self.quality).items()}
        object.__setattr__(self, "quality", MappingProxyType(quality))
        if self.trust is not None and not isinstance(self.trust, TrustMetadata):
            object.__setattr__(self, "trust", TrustMetadata(**dict(self.trust)))
        expected = self._calculate_digest()
        if self.digest is not None and normalize_digest(self.digest, "calibration digest") != expected:
            raise ValueError("calibration digest does not match canonical content")
        object.__setattr__(self, "digest", expected)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CalibrationBundle":
        if not isinstance(value, Mapping):
            raise ValueError("calibration bundle must be a mapping")
        return cls(
            schema_version=value.get("schema_version", SCHEMA_VERSION),
            calibration_id=value.get("calibration_id", value.get("id", "")),
            version=value.get("version", ""),
            sensor_product=value.get("sensor_product", value.get("sensor", "")),
            transform_type=value.get("transform_type", ""),
            transform_direction=value.get("transform_direction", ""),
            pixel_convention=value.get("pixel_convention", ""),
            parameters=value.get("parameters", {}),
            quality=value.get("quality", {}),
            trust=_trust_from_value(value.get("trust")),
            digest=value.get("digest"),
        )

    @classmethod
    def identity_fixture(cls, *, sensor_product: str = "fixture-sensor", calibration_id: str = "fixture-identity") -> "CalibrationBundle":
        return cls(
            calibration_id=calibration_id,
            version="1",
            sensor_product=sensor_product,
            transform_type="identity",
            transform_direction="source_to_model",
            pixel_convention="center",
            parameters={"matrix": [[1.0, 0.0], [0.0, 1.0]], "offset": [0.0, 0.0]},
            quality={"max_residual": 0.0},
        )

    @classmethod
    def affine_fixture(
        cls,
        *,
        matrix: Sequence[Sequence[float]] = ((1.0, 0.0), (0.0, 1.0)),
        offset: Sequence[float] = (0.0, 0.0),
        sensor_product: str = "fixture-sensor",
        calibration_id: str = "fixture-affine",
    ) -> "CalibrationBundle":
        matrix_value = tuple(_number_tuple(row, 2, "affine matrix row") for row in matrix)
        if len(matrix_value) != 2:
            raise ValueError("affine matrix must be 2x2")
        return cls(
            calibration_id=calibration_id,
            version="1",
            sensor_product=sensor_product,
            transform_type="affine",
            transform_direction="source_to_model",
            pixel_convention="center",
            parameters={"matrix": matrix_value, "offset": _number_tuple(offset, 2, "affine offset")},
            quality={"max_residual": 0.0},
        )

    def _calculate_digest(self) -> str:
        return _content_digest(self.to_mapping(include_digest=False, include_trust=False))

    @property
    def fingerprint(self) -> str:
        return self.digest

    def to_mapping(self, *, include_digest: bool = True, include_trust: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": self.schema_version,
            "calibration_id": self.calibration_id,
            "version": self.version,
            "sensor_product": self.sensor_product,
            "transform_type": self.transform_type,
            "transform_direction": self.transform_direction,
            "pixel_convention": self.pixel_convention,
            "parameters": self.parameters,
            "quality": self.quality,
        }
        if include_trust and self.trust is not None:
            result["trust"] = self.trust.to_mapping()
        if include_digest:
            result["digest"] = self.digest
        return result


@dataclass(frozen=True)
class CaptureManifest:
    schema_version: int = SCHEMA_VERSION
    capture_id: str = ""
    completion_marker: bool = False
    source_fingerprint: str = ""
    sensor_product: str = ""
    dimensions: tuple[int, ...] = ()
    source_layout: str = ""
    source_dtype: str = ""
    source_byte_order: str = "native"
    channel_schema: tuple[str, ...] = ()
    nodata_semantics: Mapping[str, Any] | None = None
    source_path: str | None = None
    preprocessing_profile_digest: str | None = None
    calibration_digest: str | None = None
    trust: TrustMetadata | None = None
    digest: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"capture manifest schema_version must be {SCHEMA_VERSION}")
        object.__setattr__(self, "capture_id", _text(self.capture_id, "capture_id"))
        if not isinstance(self.completion_marker, bool):
            raise ValueError("capture completion_marker must be boolean")
        object.__setattr__(self, "source_fingerprint", normalize_digest(self.source_fingerprint, "source_fingerprint"))
        object.__setattr__(self, "sensor_product", _text(self.sensor_product, "capture sensor_product"))
        object.__setattr__(self, "dimensions", _int_tuple(self.dimensions, "capture dimensions"))
        layout = _text(self.source_layout, "capture source_layout").upper()
        if len(set(layout)) != len(layout) or "X" not in layout or "Y" not in layout:
            raise ValueError("capture source_layout must contain unique X and Y axes")
        object.__setattr__(self, "source_layout", layout)
        object.__setattr__(self, "source_dtype", _normalize_dtype(self.source_dtype, "capture source_dtype"))
        byte_order = _text(self.source_byte_order, "capture source_byte_order").lower()
        if byte_order not in {"little", "big", "native", "not_applicable"}:
            raise ValueError("capture source_byte_order must be little, big, native, or not_applicable")
        object.__setattr__(self, "source_byte_order", byte_order)
        channels = tuple(_text(value, "capture channel") for value in self.channel_schema)
        if "C" in layout and not channels:
            raise ValueError("capture channel_schema is required when source_layout contains C")
        if "C" in layout and self.dimensions[layout.index("C")] != len(channels):
            raise ValueError("capture channel_schema count does not match dimensions")
        if len(set(channels)) != len(channels):
            raise ValueError("capture channel_schema must be unique")
        object.__setattr__(self, "channel_schema", channels)
        if self.nodata_semantics is not None and not isinstance(self.nodata_semantics, Mapping):
            raise ValueError("capture nodata_semantics must be a mapping or None")
        object.__setattr__(self, "nodata_semantics", _freeze(self.nodata_semantics) if self.nodata_semantics is not None else None)
        if self.source_path is not None:
            object.__setattr__(self, "source_path", str(Path(self.source_path)))
        for field_name in ("preprocessing_profile_digest", "calibration_digest"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, normalize_digest(value, field_name))
        if self.trust is not None and not isinstance(self.trust, TrustMetadata):
            object.__setattr__(self, "trust", TrustMetadata(**dict(self.trust)))
        expected = self._calculate_digest()
        if self.digest is not None and normalize_digest(self.digest, "capture digest") != expected:
            raise ValueError("capture digest does not match canonical content")
        object.__setattr__(self, "digest", expected)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CaptureManifest":
        if not isinstance(value, Mapping):
            raise ValueError("capture manifest must be a mapping")
        return cls(
            schema_version=value.get("schema_version", SCHEMA_VERSION),
            capture_id=value.get("capture_id", value.get("id", "")),
            completion_marker=value.get("completion_marker", value.get("complete", False)),
            source_fingerprint=value.get("source_fingerprint", ""),
            sensor_product=value.get("sensor_product", value.get("sensor", "")),
            dimensions=value.get("dimensions", value.get("shape", ())),
            source_layout=value.get("source_layout", value.get("axes", "")),
            source_dtype=value.get("source_dtype", value.get("dtype", "")),
            source_byte_order=value.get("source_byte_order", value.get("byte_order", "native")),
            channel_schema=value.get("channel_schema", value.get("channels", ())),
            nodata_semantics=value.get("nodata_semantics", value.get("nodata")),
            source_path=value.get("source_path"),
            preprocessing_profile_digest=value.get("preprocessing_profile_digest"),
            calibration_digest=value.get("calibration_digest"),
            trust=_trust_from_value(value.get("trust")),
            digest=value.get("digest"),
        )

    def _calculate_digest(self) -> str:
        return _content_digest(self.to_mapping(include_digest=False, include_trust=False))

    @property
    def fingerprint(self) -> str:
        return self.digest

    def to_mapping(self, *, include_digest: bool = True, include_trust: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": self.schema_version,
            "capture_id": self.capture_id,
            "completion_marker": self.completion_marker,
            "source_fingerprint": self.source_fingerprint,
            "sensor_product": self.sensor_product,
            "dimensions": self.dimensions,
            "source_layout": self.source_layout,
            "source_dtype": self.source_dtype,
            "source_byte_order": self.source_byte_order,
            "channel_schema": self.channel_schema,
            "nodata_semantics": self.nodata_semantics,
            "source_path": self.source_path,
            "preprocessing_profile_digest": self.preprocessing_profile_digest,
            "calibration_digest": self.calibration_digest,
        }
        if include_trust and self.trust is not None:
            result["trust"] = self.trust.to_mapping()
        if include_digest:
            result["digest"] = self.digest
        return result


@dataclass(frozen=True)
class TargetGrid:
    shape: tuple[int, int] | None = None
    extent: tuple[float, float, float, float] | None = None
    resolution: tuple[float, float] | None = None
    origin: tuple[float, float] | None = None
    axes: str = ""
    pixel_convention: str = ""
    spatial_semantics: str = ""

    def __post_init__(self) -> None:
        shape = None
        if self.shape is not None:
            raw_shape = _int_tuple(self.shape, "target_grid.shape", length=2)
            shape = (raw_shape[0], raw_shape[1])
        extent = None if self.extent is None else _number_tuple(self.extent, 4, "target_grid.extent")
        resolution = None if self.resolution is None else _number_tuple(self.resolution, 2, "target_grid.resolution")
        if extent is None or resolution is None:
            if shape is None:
                raise ValueError("target_grid requires shape or both extent and resolution")
        if resolution is not None and (resolution[0] <= 0 or resolution[1] <= 0):
            raise ValueError("target_grid.resolution values must be positive")
        if extent is not None and (extent[2] <= extent[0] or extent[3] <= extent[1]):
            raise ValueError("target_grid.extent must have positive width and height")
        if extent is not None and resolution is not None:
            computed = (
                (extent[3] - extent[1]) / resolution[1],
                (extent[2] - extent[0]) / resolution[0],
            )
            rounded = (round(computed[0]), round(computed[1]))
            if any(not math.isclose(computed[index], rounded[index], rel_tol=0, abs_tol=1e-9) for index in range(2)):
                raise ValueError("target_grid extent/resolution do not produce an integer shape")
            if shape is None:
                shape = rounded
            elif shape != rounded:
                raise ValueError("target_grid shape does not match extent/resolution")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "extent", extent)
        object.__setattr__(self, "resolution", resolution)
        if self.origin is None:
            raise ValueError("target_grid.origin is required")
        object.__setattr__(self, "origin", _number_tuple(self.origin, 2, "target_grid.origin"))
        axes = _text(self.axes, "target_grid.axes").upper()
        if axes not in {"XY", "YX"}:
            raise ValueError("target_grid.axes must be XY or YX")
        object.__setattr__(self, "axes", axes)
        conventions = {"center": "center", "corner": "corner"}
        object.__setattr__(self, "pixel_convention", _normalize_choice(self.pixel_convention, "target_grid.pixel_convention", conventions))
        object.__setattr__(self, "spatial_semantics", _text(self.spatial_semantics, "target_grid.spatial_semantics"))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TargetGrid":
        if not isinstance(value, Mapping):
            raise ValueError("target_grid must be a mapping")
        return cls(
            shape=value.get("shape", value.get("size")),
            extent=value.get("extent"),
            resolution=value.get("resolution", value.get("gsd")),
            origin=value.get("origin"),
            axes=value.get("axes", ""),
            pixel_convention=value.get("pixel_convention", ""),
            spatial_semantics=value.get("spatial_semantics", value.get("semantics", "")),
        )

    @property
    def rows(self) -> int:
        return self.shape[0]

    @property
    def cols(self) -> int:
        return self.shape[1]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "shape": self.shape,
            "extent": self.extent,
            "resolution": self.resolution,
            "origin": self.origin,
            "axes": self.axes,
            "pixel_convention": self.pixel_convention,
            "spatial_semantics": self.spatial_semantics,
        }


@dataclass(frozen=True)
class SourceROI:
    mode: str = ""
    row_start: int | None = None
    row_end: int | None = None
    col_start: int | None = None
    col_end: int | None = None

    def __post_init__(self) -> None:
        mode = str(self.mode).lower().replace("-", "_")
        if mode not in {"full", "window"}:
            raise ValueError("source_roi.mode must be full or window")
        object.__setattr__(self, "mode", mode)
        values = (self.row_start, self.row_end, self.col_start, self.col_end)
        if mode == "window":
            if any(value is None for value in values):
                raise ValueError("window source_roi requires all bounds")
            bounds = tuple(_non_negative_int(value, "source_roi bound") for value in values)
            if bounds[1] <= bounds[0] or bounds[3] <= bounds[2]:
                raise ValueError("source_roi window bounds must have positive size")
            object.__setattr__(self, "row_start", bounds[0])
            object.__setattr__(self, "row_end", bounds[1])
            object.__setattr__(self, "col_start", bounds[2])
            object.__setattr__(self, "col_end", bounds[3])
        elif any(value is not None for value in values):
            raise ValueError("full source_roi must not define bounds")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SourceROI":
        if not isinstance(value, Mapping):
            raise ValueError("source_roi must be a mapping")
        return cls(
            mode=value.get("mode", ""),
            row_start=value.get("row_start"),
            row_end=value.get("row_end"),
            col_start=value.get("col_start"),
            col_end=value.get("col_end"),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "row_start": self.row_start,
            "row_end": self.row_end,
            "col_start": self.col_start,
            "col_end": self.col_end,
        }


@dataclass(frozen=True)
class Halo:
    rows: int | None = None
    cols: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "rows", _non_negative_int(self.rows, "halo.rows"))
        object.__setattr__(self, "cols", _non_negative_int(self.cols, "halo.cols"))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Halo":
        if not isinstance(value, Mapping):
            raise ValueError("halo must be a mapping")
        return cls(rows=value.get("rows"), cols=value.get("cols"))

    def to_mapping(self) -> dict[str, int]:
        return {"rows": self.rows, "cols": self.cols}


@dataclass(frozen=True)
class MaskEncoding:
    dtype: str = ""
    valid_value: int = 1
    invalid_value: int = 0
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"mask encoding schema_version must be {SCHEMA_VERSION}")
        dtype = _normalize_dtype(self.dtype, "mask encoding dtype")
        if not np.issubdtype(np.dtype(dtype), np.unsignedinteger):
            raise ValueError("mask encoding dtype must be unsigned integer")
        object.__setattr__(self, "dtype", dtype)
        valid = _non_negative_int(self.valid_value, "mask valid_value")
        invalid = _non_negative_int(self.invalid_value, "mask invalid_value")
        if valid == invalid:
            raise ValueError("mask valid_value and invalid_value must differ")
        maximum = int(np.iinfo(np.dtype(dtype)).max)
        if valid > maximum or invalid > maximum:
            raise ValueError("mask values do not fit mask encoding dtype")
        object.__setattr__(self, "valid_value", valid)
        object.__setattr__(self, "invalid_value", invalid)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "MaskEncoding":
        if not isinstance(value, Mapping):
            raise ValueError("validity_encoding must be a mapping")
        return cls(
            dtype=value.get("dtype", ""),
            valid_value=value.get("valid_value", value.get("valid", 1)),
            invalid_value=value.get("invalid_value", value.get("invalid", 0)),
            schema_version=value.get("schema_version", SCHEMA_VERSION),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "dtype": self.dtype,
            "valid_value": self.valid_value,
            "invalid_value": self.invalid_value,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True)
class ReasonEncoding:
    dtype: str = ""
    bits: Mapping[str, int] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"reason encoding schema_version must be {SCHEMA_VERSION}")
        dtype = _normalize_dtype(self.dtype, "reason encoding dtype")
        if not np.issubdtype(np.dtype(dtype), np.unsignedinteger):
            raise ValueError("reason encoding dtype must be unsigned integer")
        object.__setattr__(self, "dtype", dtype)
        if not self.bits:
            raise ValueError("reason encoding bits must not be empty")
        normalized = {}
        maximum = int(np.iinfo(np.dtype(dtype)).max)
        for name, value in dict(self.bits).items():
            key = _text(name, "reason bit name")
            bit = _positive_int(value, f"reason bit {key}")
            if bit > maximum or bit & (bit - 1):
                raise ValueError(f"reason bit {key} must be one unsigned power of two")
            normalized[key] = bit
        if len(set(normalized.values())) != len(normalized):
            raise ValueError("reason encoding bit values must be unique")
        object.__setattr__(self, "bits", MappingProxyType(normalized))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ReasonEncoding":
        if not isinstance(value, Mapping):
            raise ValueError("reason_encoding must be a mapping")
        return cls(
            dtype=value.get("dtype", ""),
            bits=value.get("bits", value.get("bit_layout", {})),
            schema_version=value.get("schema_version", SCHEMA_VERSION),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {"dtype": self.dtype, "bits": self.bits, "schema_version": self.schema_version}


@dataclass(frozen=True)
class ClippingPolicy:
    mode: str = ""
    lower: float | None = None
    upper: float | None = None

    def __post_init__(self) -> None:
        mode = str(self.mode).lower().replace("-", "_")
        if mode not in {"none", "range"}:
            raise ValueError("clipping.mode must be none or range")
        object.__setattr__(self, "mode", mode)
        if mode == "none":
            if self.lower is not None or self.upper is not None:
                raise ValueError("clipping bounds are not allowed when mode is none")
        else:
            if self.lower is None or self.upper is None:
                raise ValueError("range clipping requires lower and upper")
            lower = _finite_float(self.lower, "clipping.lower")
            upper = _finite_float(self.upper, "clipping.upper")
            if upper < lower:
                raise ValueError("clipping.upper must be >= clipping.lower")
            object.__setattr__(self, "lower", lower)
            object.__setattr__(self, "upper", upper)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ClippingPolicy":
        if not isinstance(value, Mapping):
            raise ValueError("clipping must be a mapping")
        return cls(mode=value.get("mode", ""), lower=value.get("lower"), upper=value.get("upper"))

    def to_mapping(self) -> dict[str, Any]:
        return {"mode": self.mode, "lower": self.lower, "upper": self.upper}


@dataclass(frozen=True)
class PreprocessingProfile:
    schema_version: int = SCHEMA_VERSION
    profile_id: str = ""
    profile_version: str = ""
    implementation_version: str = ""
    source_schema: SourceSchema | Mapping[str, Any] | None = None
    calibration_selector: CalibrationSelector | Mapping[str, Any] | None = None
    transform_direction: str = ""
    transform_type: str = ""
    pixel_convention: str = ""
    target_grid: TargetGrid | Mapping[str, Any] | None = None
    source_roi: SourceROI | Mapping[str, Any] | None = None
    halo: Halo | Mapping[str, Any] | None = None
    border_policy: str = ""
    image_kernel: str = ""
    validity_kernel: str = ""
    reason_kernel: str = ""
    support_threshold: float | None = None
    internal_numeric_precision: str = ""
    output_layout: str = ""
    output_dtype: str = ""
    rounding: str = ""
    clipping: ClippingPolicy | Mapping[str, Any] | None = None
    non_finite_policy: str = ""
    validity_encoding: MaskEncoding | Mapping[str, Any] | None = None
    reason_encoding: ReasonEncoding | Mapping[str, Any] | None = None
    digest: str | None = None
    trust: TrustMetadata | None = None

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"preprocessing profile schema_version must be {SCHEMA_VERSION}")
        object.__setattr__(self, "profile_id", _text(self.profile_id, "profile_id"))
        object.__setattr__(self, "profile_version", _text(self.profile_version, "profile_version"))
        object.__setattr__(self, "implementation_version", _text(self.implementation_version, "implementation_version"))
        source_schema = self.source_schema if isinstance(self.source_schema, SourceSchema) else SourceSchema.from_mapping(self.source_schema or {})
        selector = self.calibration_selector if isinstance(self.calibration_selector, CalibrationSelector) else CalibrationSelector.from_mapping(self.calibration_selector or {})
        target_grid = self.target_grid if isinstance(self.target_grid, TargetGrid) else TargetGrid.from_mapping(self.target_grid or {})
        source_roi = self.source_roi if isinstance(self.source_roi, SourceROI) else SourceROI.from_mapping(self.source_roi or {})
        halo = self.halo if isinstance(self.halo, Halo) else Halo.from_mapping(self.halo or {})
        clipping = self.clipping if isinstance(self.clipping, ClippingPolicy) else ClippingPolicy.from_mapping(self.clipping or {})
        validity_encoding = self.validity_encoding if isinstance(self.validity_encoding, MaskEncoding) else MaskEncoding.from_mapping(self.validity_encoding or {})
        reason_encoding = self.reason_encoding if isinstance(self.reason_encoding, ReasonEncoding) else ReasonEncoding.from_mapping(self.reason_encoding or {})
        object.__setattr__(self, "source_schema", source_schema)
        object.__setattr__(self, "calibration_selector", selector)
        object.__setattr__(self, "target_grid", target_grid)
        object.__setattr__(self, "source_roi", source_roi)
        object.__setattr__(self, "halo", halo)
        object.__setattr__(self, "clipping", clipping)
        object.__setattr__(self, "validity_encoding", validity_encoding)
        object.__setattr__(self, "reason_encoding", reason_encoding)
        directions = {"source_to_model": "source_to_model", "model_to_source": "model_to_source"}
        object.__setattr__(self, "transform_direction", _normalize_choice(self.transform_direction, "transform_direction", directions))
        transforms = {"identity": "identity", "affine": "affine", "lut": "lut", "brown_conrady": "brown_conrady"}
        object.__setattr__(self, "transform_type", _normalize_choice(self.transform_type, "transform_type", transforms))
        conventions = {"center": "center", "corner": "corner"}
        object.__setattr__(self, "pixel_convention", _normalize_choice(self.pixel_convention, "pixel_convention", conventions))
        if target_grid.pixel_convention != self.pixel_convention:
            raise ValueError("profile and target_grid pixel_convention must match")
        borders = {"invalid": "invalid", "constant": "constant", "edge": "edge", "reflect": "reflect"}
        object.__setattr__(self, "border_policy", _normalize_choice(self.border_policy, "border_policy", borders))
        kernels = {"nearest": "nearest", "bilinear": "bilinear", "bicubic": "bicubic", "support": "support"}
        object.__setattr__(self, "image_kernel", _normalize_choice(self.image_kernel, "image_kernel", kernels))
        object.__setattr__(self, "validity_kernel", _normalize_choice(self.validity_kernel, "validity_kernel", kernels))
        object.__setattr__(self, "reason_kernel", _normalize_choice(self.reason_kernel, "reason_kernel", kernels))
        if self.support_threshold is None:
            raise ValueError("support_threshold is required")
        support = _finite_float(self.support_threshold, "support_threshold")
        if not 0.0 <= support <= 1.0:
            raise ValueError("support_threshold must be between 0 and 1")
        object.__setattr__(self, "support_threshold", support)
        precision = _normalize_choice(
            self.internal_numeric_precision,
            "internal_numeric_precision",
            {"float32": "float32", "float64": "float64"},
        )
        object.__setattr__(self, "internal_numeric_precision", precision)
        layout = _text(self.output_layout, "output_layout").upper()
        if len(set(layout)) != len(layout) or set(layout) - {"X", "Y", "C"} or "X" not in layout or "Y" not in layout:
            raise ValueError("output_layout must contain unique X and Y axes")
        if "C" in layout and len(source_schema.channels) == 0:
            raise ValueError("output_layout with C requires source_schema.channels")
        if source_schema.channels and "C" not in layout:
            raise ValueError("output_layout must retain the source channel axis")
        object.__setattr__(self, "output_layout", layout)
        output_dtype = _normalize_dtype(self.output_dtype, "output_dtype")
        object.__setattr__(self, "output_dtype", output_dtype)
        roundings = {
            "none": "none",
            "nearest_even": "nearest_even",
            "nearest_away": "nearest_away",
            "floor": "floor",
            "ceil": "ceil",
            "truncate": "truncate",
        }
        rounding = _normalize_choice(self.rounding, "rounding", roundings)
        if np.issubdtype(np.dtype(output_dtype), np.integer) and rounding == "none":
            raise ValueError("integer output_dtype requires an explicit rounding policy")
        object.__setattr__(self, "rounding", rounding)
        non_finite = _normalize_choice(
            self.non_finite_policy,
            "non_finite_policy",
            {"reject": "reject", "replace": "replace"},
        )
        object.__setattr__(self, "non_finite_policy", non_finite)
        if self.trust is not None and not isinstance(self.trust, TrustMetadata):
            object.__setattr__(self, "trust", TrustMetadata(**dict(self.trust)))
        expected = self._calculate_digest()
        if self.digest is not None and normalize_digest(self.digest, "profile digest") != expected:
            raise ValueError("profile digest does not match canonical content")
        object.__setattr__(self, "digest", expected)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PreprocessingProfile":
        if not isinstance(value, Mapping):
            raise ValueError("preprocessing profile must be a mapping")
        resampling = value.get("resampling", {})
        if not isinstance(resampling, Mapping):
            raise ValueError("profile resampling must be a mapping")
        return cls(
            schema_version=value.get("schema_version", SCHEMA_VERSION),
            profile_id=value.get("profile_id", value.get("id", "")),
            profile_version=value.get("profile_version", value.get("version", "")),
            implementation_version=value.get("implementation_version", ""),
            source_schema=value.get("source_schema", value.get("source_representation")),
            calibration_selector=value.get("calibration_selector"),
            transform_direction=value.get("transform_direction", ""),
            transform_type=value.get("transform_type", ""),
            pixel_convention=value.get("pixel_convention", ""),
            target_grid=value.get("target_grid"),
            source_roi=value.get("source_roi"),
            halo=value.get("halo"),
            border_policy=value.get("border_policy", ""),
            image_kernel=value.get("image_kernel", resampling.get("image_kernel", "")),
            validity_kernel=value.get("validity_kernel", resampling.get("validity_kernel", "")),
            reason_kernel=value.get("reason_kernel", resampling.get("reason_kernel", "")),
            support_threshold=value.get("support_threshold", resampling.get("support_threshold")),
            internal_numeric_precision=value.get("internal_numeric_precision", ""),
            output_layout=value.get("output_layout", ""),
            output_dtype=value.get("output_dtype", ""),
            rounding=value.get("rounding", ""),
            clipping=value.get("clipping"),
            non_finite_policy=value.get("non_finite_policy", ""),
            validity_encoding=value.get("validity_encoding"),
            reason_encoding=value.get("reason_encoding"),
            digest=value.get("digest"),
            trust=_trust_from_value(value.get("trust")),
        )

    @classmethod
    def identity_fixture(
        cls,
        *,
        shape: tuple[int, int] = (16, 16),
        channels: Sequence[str] = ("red", "green", "blue"),
        dtype: str = "uint16",
        profile_id: str = "fixture.identity",
        sensor_product: str = "fixture-sensor",
    ) -> "PreprocessingProfile":
        rows, cols = _int_tuple(shape, "fixture shape", length=2)
        channel_names = tuple(channels)
        source = SourceSchema(
            axes="YXC",
            shape=(rows, cols, len(channel_names)),
            dtype=dtype,
            channels=channel_names,
            representation="raster",
            byte_order="native",
            nodata_semantics={"kind": "none"},
        )
        target = TargetGrid(
            shape=(rows, cols),
            extent=(0.0, 0.0, float(cols), float(rows)),
            resolution=(1.0, 1.0),
            origin=(0.0, 0.0),
            axes="XY",
            pixel_convention="center",
            spatial_semantics="fixture-grid",
        )
        return cls(
            profile_id=profile_id,
            profile_version="1",
            implementation_version="reference-1",
            source_schema=source,
            calibration_selector=CalibrationSelector(
                sensor_product=sensor_product,
                calibration_id="fixture-identity",
                version="1",
                allowed_transform_types=("identity",),
                quality_limits={"max_residual": 1.0},
            ),
            transform_direction="source_to_model",
            transform_type="identity",
            pixel_convention="center",
            target_grid=target,
            source_roi=SourceROI(mode="full"),
            halo=Halo(rows=0, cols=0),
            border_policy="invalid",
            image_kernel="nearest",
            validity_kernel="nearest",
            reason_kernel="nearest",
            support_threshold=1.0,
            internal_numeric_precision="float32",
            output_layout="YXC",
            output_dtype=dtype,
            rounding="nearest_even",
            clipping=ClippingPolicy(mode="none"),
            non_finite_policy="reject",
            validity_encoding=MaskEncoding(dtype="uint8", valid_value=1, invalid_value=0),
            reason_encoding=ReasonEncoding(
                dtype="uint16",
                bits={
                    "source_nodata": 1,
                    "missing_channel": 2,
                    "outside_mapping": 4,
                    "border": 8,
                    "insufficient_support": 16,
                },
            ),
        )

    @classmethod
    def affine_fixture(cls, **kwargs: Any) -> "PreprocessingProfile":
        profile = cls.identity_fixture(**kwargs)
        selector = replace(profile.calibration_selector, allowed_transform_types=("affine",), calibration_id="fixture-affine")
        return replace(profile, profile_id="fixture.affine", transform_type="affine", calibration_selector=selector, digest=None)

    def _calculate_digest(self) -> str:
        return _content_digest(self.to_mapping(include_digest=False, include_trust=False))

    @property
    def fingerprint(self) -> str:
        return self.digest

    @property
    def output_channels(self) -> int:
        return self.source_schema.channel_count

    @property
    def output_shape(self) -> tuple[int, ...]:
        spatial = {"Y": self.target_grid.rows, "X": self.target_grid.cols, "C": self.output_channels}
        return tuple(spatial[axis] for axis in self.output_layout)

    def to_mapping(self, *, include_digest: bool = True, include_trust: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "implementation_version": self.implementation_version,
            "source_schema": self.source_schema.to_mapping(),
            "calibration_selector": self.calibration_selector.to_mapping(),
            "transform_direction": self.transform_direction,
            "transform_type": self.transform_type,
            "pixel_convention": self.pixel_convention,
            "target_grid": self.target_grid.to_mapping(),
            "source_roi": self.source_roi.to_mapping(),
            "halo": self.halo.to_mapping(),
            "border_policy": self.border_policy,
            "image_kernel": self.image_kernel,
            "validity_kernel": self.validity_kernel,
            "reason_kernel": self.reason_kernel,
            "support_threshold": self.support_threshold,
            "internal_numeric_precision": self.internal_numeric_precision,
            "output_layout": self.output_layout,
            "output_dtype": self.output_dtype,
            "rounding": self.rounding,
            "clipping": self.clipping.to_mapping(),
            "non_finite_policy": self.non_finite_policy,
            "validity_encoding": self.validity_encoding.to_mapping(),
            "reason_encoding": self.reason_encoding.to_mapping(),
        }
        if include_trust and self.trust is not None:
            result["trust"] = self.trust.to_mapping()
        if include_digest:
            result["digest"] = self.digest
        return result


@dataclass(frozen=True)
class ModelCompatibilityProfile:
    """Model-side contract kept separate from ``PreprocessingProfile``."""

    schema_version: int = SCHEMA_VERSION
    profile_id: str = ""
    profile_version: str = ""
    model_fingerprint: str = ""
    required_band_order: tuple[str, ...] = ()
    tensor_layout: str = ""
    tensor_dtype: str = ""
    patch_size: tuple[int, int] = ()
    batch_size: int = 0
    padding_policy: str = ""
    normalization: str | Mapping[str, Any] = ""
    runtime_fingerprint: str = ""
    trust: TrustMetadata | None = None
    digest: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"model compatibility schema_version must be {SCHEMA_VERSION}")
        object.__setattr__(self, "profile_id", _text(self.profile_id, "model profile_id"))
        object.__setattr__(self, "profile_version", _text(self.profile_version, "model profile_version"))
        object.__setattr__(self, "model_fingerprint", normalize_digest(self.model_fingerprint, "model_fingerprint"))
        bands = tuple(_text(value, "required band") for value in self.required_band_order)
        if not bands or len(set(bands)) != len(bands):
            raise ValueError("required_band_order must be non-empty and unique")
        object.__setattr__(self, "required_band_order", bands)
        layout = _text(self.tensor_layout, "tensor_layout").upper()
        if len(set(layout)) != len(layout) or "N" not in layout or "C" not in layout:
            raise ValueError("tensor_layout must contain unique N and C axes")
        object.__setattr__(self, "tensor_layout", layout)
        object.__setattr__(self, "tensor_dtype", _normalize_dtype(self.tensor_dtype, "tensor_dtype"))
        patch = _int_tuple(self.patch_size, "patch_size", length=2)
        object.__setattr__(self, "patch_size", (patch[0], patch[1]))
        object.__setattr__(self, "batch_size", _positive_int(self.batch_size, "batch_size"))
        padding = _text(self.padding_policy, "padding_policy").lower().replace("-", "_")
        if padding not in {"reject", "constant", "edge", "reflect"}:
            raise ValueError("padding_policy must be reject, constant, edge, or reflect")
        object.__setattr__(self, "padding_policy", padding)
        if isinstance(self.normalization, Mapping):
            normalization = _freeze(self.normalization)
        else:
            normalization = _text(self.normalization, "normalization")
        object.__setattr__(self, "normalization", normalization)
        object.__setattr__(self, "runtime_fingerprint", normalize_digest(self.runtime_fingerprint, "runtime_fingerprint"))
        if self.trust is not None and not isinstance(self.trust, TrustMetadata):
            object.__setattr__(self, "trust", TrustMetadata(**dict(self.trust)))
        expected = self._calculate_digest()
        if self.digest is not None and normalize_digest(self.digest, "model profile digest") != expected:
            raise ValueError("model profile digest does not match canonical content")
        object.__setattr__(self, "digest", expected)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ModelCompatibilityProfile":
        if not isinstance(value, Mapping):
            raise ValueError("model compatibility profile must be a mapping")
        return cls(
            schema_version=value.get("schema_version", SCHEMA_VERSION),
            profile_id=value.get("profile_id", value.get("id", "")),
            profile_version=value.get("profile_version", value.get("version", "")),
            model_fingerprint=value.get("model_fingerprint", value.get("engine_fingerprint", "")),
            required_band_order=value.get("required_band_order", value.get("band_order", ())),
            tensor_layout=value.get("tensor_layout", ""),
            tensor_dtype=value.get("tensor_dtype", value.get("input_dtype", "")),
            patch_size=value.get("patch_size", ()),
            batch_size=value.get("batch_size", 0),
            padding_policy=value.get("padding_policy", ""),
            normalization=value.get("normalization", ""),
            runtime_fingerprint=value.get("runtime_fingerprint", ""),
            trust=_trust_from_value(value.get("trust")),
            digest=value.get("digest"),
        )

    def _calculate_digest(self) -> str:
        return _content_digest(self.to_mapping(include_digest=False, include_trust=False))

    @property
    def fingerprint(self) -> str:
        return self.digest

    def accepts(self, profile: PreprocessingProfile) -> tuple[str, ...]:
        reasons = []
        missing = set(self.required_band_order) - set(profile.source_schema.channels)
        if missing:
            reasons.append(f"missing bands: {sorted(missing)}")
        if self.tensor_layout.replace("N", "").replace("C", "") and "H" not in self.tensor_layout and "W" not in self.tensor_layout:
            reasons.append("tensor layout has no spatial axes")
        return tuple(reasons)

    def to_mapping(self, *, include_digest: bool = True, include_trust: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "model_fingerprint": self.model_fingerprint,
            "required_band_order": self.required_band_order,
            "tensor_layout": self.tensor_layout,
            "tensor_dtype": self.tensor_dtype,
            "patch_size": self.patch_size,
            "batch_size": self.batch_size,
            "padding_policy": self.padding_policy,
            "normalization": self.normalization,
            "runtime_fingerprint": self.runtime_fingerprint,
        }
        if include_trust and self.trust is not None:
            result["trust"] = self.trust.to_mapping()
        if include_digest:
            result["digest"] = self.digest
        return result


# The legacy ``src/input_contract.py`` module also defines ``EngineInputSpec``
# with a different schema.  Keep ``ModelCompatibilityProfile`` as the
# canonical model-side name here instead of exporting an ambiguous alias.


@dataclass(frozen=True)
class ComputeProfile:
    schema_version: int = SCHEMA_VERSION
    compute_profile_id: str = ""
    profile_version: str = ""
    backend: str = ""
    ram_budget_bytes: int = 0
    disk_budget_bytes: int = 0
    os_obc_reserve_bytes: int = 0
    safety_margin_bytes: int = 0
    max_strip_rows: int = 0
    queue_depth: int = 0
    inflight_strips: int = 0
    decoder_workers: int = 1
    temporary_directory: str = ""
    allow_compressed_full_decode: bool = False
    max_full_decode_bytes: int | None = None
    artifact_staging_multiplier: float = 1.0
    checksum_headroom_bytes: int = 0
    thermal_policy: str = ""
    trust: TrustMetadata | None = None
    digest: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"compute profile schema_version must be {SCHEMA_VERSION}")
        object.__setattr__(self, "compute_profile_id", _text(self.compute_profile_id, "compute_profile_id"))
        object.__setattr__(self, "profile_version", _text(self.profile_version, "compute profile_version"))
        object.__setattr__(self, "backend", _text(self.backend, "compute backend").lower())
        for field_name in ("ram_budget_bytes", "disk_budget_bytes", "max_strip_rows", "queue_depth", "inflight_strips"):
            object.__setattr__(self, field_name, _positive_int(getattr(self, field_name), field_name))
        for field_name in ("os_obc_reserve_bytes", "safety_margin_bytes", "checksum_headroom_bytes"):
            object.__setattr__(self, field_name, _non_negative_int(getattr(self, field_name), field_name))
        object.__setattr__(self, "decoder_workers", _positive_int(self.decoder_workers, "decoder_workers"))
        object.__setattr__(self, "temporary_directory", str(Path(_text(self.temporary_directory, "temporary_directory"))))
        if not isinstance(self.allow_compressed_full_decode, bool):
            raise ValueError("allow_compressed_full_decode must be boolean")
        if self.max_full_decode_bytes is not None:
            object.__setattr__(self, "max_full_decode_bytes", _positive_int(self.max_full_decode_bytes, "max_full_decode_bytes"))
        multiplier = _finite_float(self.artifact_staging_multiplier, "artifact_staging_multiplier")
        if multiplier < 1.0:
            raise ValueError("artifact_staging_multiplier must be at least 1")
        object.__setattr__(self, "artifact_staging_multiplier", multiplier)
        object.__setattr__(self, "thermal_policy", _text(self.thermal_policy, "thermal_policy"))
        if self.trust is not None and not isinstance(self.trust, TrustMetadata):
            object.__setattr__(self, "trust", TrustMetadata(**dict(self.trust)))
        expected = self._calculate_digest()
        if self.digest is not None and normalize_digest(self.digest, "compute profile digest") != expected:
            raise ValueError("compute profile digest does not match canonical content")
        object.__setattr__(self, "digest", expected)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ComputeProfile":
        if not isinstance(value, Mapping):
            raise ValueError("compute profile must be a mapping")
        return cls(
            schema_version=value.get("schema_version", SCHEMA_VERSION),
            compute_profile_id=value.get("compute_profile_id", value.get("id", "")),
            profile_version=value.get("profile_version", value.get("version", "")),
            backend=value.get("backend", ""),
            ram_budget_bytes=value.get("ram_budget_bytes", value.get("ram_budget", 0)),
            disk_budget_bytes=value.get("disk_budget_bytes", value.get("disk_budget", 0)),
            os_obc_reserve_bytes=value.get("os_obc_reserve_bytes", 0),
            safety_margin_bytes=value.get("safety_margin_bytes", 0),
            max_strip_rows=value.get("max_strip_rows", value.get("strip_rows", 0)),
            queue_depth=value.get("queue_depth", 0),
            inflight_strips=value.get("inflight_strips", value.get("max_inflight_strips", 0)),
            decoder_workers=value.get("decoder_workers", 1),
            temporary_directory=value.get("temporary_directory", value.get("temp_dir", "")),
            allow_compressed_full_decode=value.get("allow_compressed_full_decode", False),
            max_full_decode_bytes=value.get("max_full_decode_bytes"),
            artifact_staging_multiplier=value.get("artifact_staging_multiplier", 1.0),
            checksum_headroom_bytes=value.get("checksum_headroom_bytes", 0),
            thermal_policy=value.get("thermal_policy", ""),
            trust=_trust_from_value(value.get("trust")),
            digest=value.get("digest"),
        )

    def _calculate_digest(self) -> str:
        return _content_digest(self.to_mapping(include_digest=False, include_trust=False))

    @property
    def fingerprint(self) -> str:
        return self.digest

    def to_mapping(self, *, include_digest: bool = True, include_trust: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": self.schema_version,
            "compute_profile_id": self.compute_profile_id,
            "profile_version": self.profile_version,
            "backend": self.backend,
            "ram_budget_bytes": self.ram_budget_bytes,
            "disk_budget_bytes": self.disk_budget_bytes,
            "os_obc_reserve_bytes": self.os_obc_reserve_bytes,
            "safety_margin_bytes": self.safety_margin_bytes,
            "max_strip_rows": self.max_strip_rows,
            "queue_depth": self.queue_depth,
            "inflight_strips": self.inflight_strips,
            "decoder_workers": self.decoder_workers,
            "temporary_directory": self.temporary_directory,
            "allow_compressed_full_decode": self.allow_compressed_full_decode,
            "max_full_decode_bytes": self.max_full_decode_bytes,
            "artifact_staging_multiplier": self.artifact_staging_multiplier,
            "checksum_headroom_bytes": self.checksum_headroom_bytes,
            "thermal_policy": self.thermal_policy,
        }
        if include_trust and self.trust is not None:
            result["trust"] = self.trust.to_mapping()
        if include_digest:
            result["digest"] = self.digest
        return result


@dataclass(frozen=True)
class SourceDescriptor:
    path: Path | None = None
    format: str = ""
    size_bytes: int | None = None
    compressed: bool = False
    full_decode_bytes: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.path is not None:
            object.__setattr__(self, "path", Path(self.path))
        if self.size_bytes is not None:
            object.__setattr__(self, "size_bytes", _positive_int(self.size_bytes, "source size_bytes"))
        if not isinstance(self.compressed, bool):
            raise ValueError("source compressed must be boolean")
        if self.full_decode_bytes is not None:
            object.__setattr__(self, "full_decode_bytes", _positive_int(self.full_decode_bytes, "source full_decode_bytes"))
        if self.path is None and self.size_bytes is None:
            raise ValueError("source descriptor requires path or size_bytes")
        object.__setattr__(self, "format", _text(self.format, "source format") if self.format else "unknown")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("source metadata must be a mapping")
        metadata = dict(self.metadata)
        if "temporary_bytes" in metadata:
            metadata["temporary_bytes"] = _non_negative_int(metadata["temporary_bytes"], "source metadata.temporary_bytes")
        if "source_fingerprint" in metadata:
            metadata["source_fingerprint"] = normalize_digest(metadata["source_fingerprint"], "source metadata.source_fingerprint")
        object.__setattr__(self, "metadata", _freeze(metadata))

    @classmethod
    def from_value(cls, value: Any) -> "SourceDescriptor":
        if isinstance(value, cls):
            return value
        if isinstance(value, (str, Path)):
            return cls(path=Path(value), format=Path(value).suffix.lstrip(".") or "unknown")
        if isinstance(value, Mapping):
            return cls(
                path=value.get("path", value.get("source_path")),
                format=value.get("format", value.get("source_format", "unknown")),
                size_bytes=value.get("size_bytes"),
                compressed=value.get("compressed", False),
                full_decode_bytes=value.get("full_decode_bytes"),
                metadata=value.get("metadata", {}),
            )
        raise ValueError("source must be a path or source descriptor mapping")


def _trust_from_value(value: Any) -> TrustMetadata | None:
    if value is None:
        return None
    if isinstance(value, TrustMetadata):
        return value
    if isinstance(value, Mapping):
        return TrustMetadata(**dict(value))
    raise ValueError("trust must be a TrustMetadata or mapping")


def attach_trust(
    contract: Any,
    *,
    issuer: str,
    key_id: str,
    key: bytes | str,
    generation: int,
    issued_at: str | datetime,
    expires_at: str | datetime,
) -> Any:
    """Return a copy of a contract with an HMAC trust envelope attached."""

    trust = TrustMetadata.for_digest(
        contract.fingerprint,
        issuer=issuer,
        key_id=key_id,
        key=key,
        generation=generation,
        issued_at=issued_at,
        expires_at=expires_at,
    )
    return replace(contract, trust=trust, digest=None)


__all__ = [
    "CalibrationBundle",
    "CalibrationSelector",
    "CaptureManifest",
    "ClippingPolicy",
    "ComputeProfile",
    "Halo",
    "MaskEncoding",
    "ModelCompatibilityProfile",
    "PREPROCESS_ARTIFACT_SCHEMA_VERSION",
    "PREPROCESSING_API_VERSION",
    "PreprocessingProfile",
    "ReasonEncoding",
    "SCHEMA_VERSION",
    "SourceDescriptor",
    "SourceROI",
    "SourceSchema",
    "TargetGrid",
    "TrustMetadata",
    "TrustPolicy",
    "attach_trust",
    "canonical_json",
    "normalize_digest",
]
