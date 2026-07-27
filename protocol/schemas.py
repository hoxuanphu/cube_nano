"""Mission command and identity schemas for the MVP wire contract."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum, IntEnum
from typing import Any, Mapping

from .canonical import (
    canonical_json,
    checked_add_u64,
    checked_u16,
    checked_u32,
    checked_u64,
    jcs_canonical_json,
    u64_from_json,
    u64_to_bytes,
    u64_to_json,
)

PROTOCOL_SCHEMA_VERSION = 1
_HTTP_RFC3339 = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})T"
    r"(?P<time>\d{2}:\d{2}:\d{2})"
    r"(?P<fraction>\.\d{1,6})?"
    r"(?P<zone>Z|[+-]\d{2}:\d{2})$"
)


class CommandOpcode(IntEnum):
    CLOUD_SET_CONFIG = 0x00010001
    SCENE_REQUEST_CATALOG = 0x00010002
    SCENE_REQUEST_PREVIEW = 0x00010003
    SCENE_ANALYZE = 0x00010004
    ROI_REQUEST = 0x00010005
    JOB_GET_STATUS = 0x00010006
    JOB_CANCEL = 0x00010007
    PRODUCT_REQUEST_DOWNLINK = 0x00010008
    PRODUCT_CANCEL_DOWNLINK = 0x00010009


class ErrorCode(str, Enum):
    TARGET_INSTANCE_MISMATCH = "TARGET_INSTANCE_MISMATCH"
    CONFIG_REVISION_MISMATCH = "CONFIG_REVISION_MISMATCH"
    CONFIG_SNAPSHOT_MISMATCH = "CONFIG_SNAPSHOT_MISMATCH"
    CATALOG_EPOCH_MISMATCH = "CATALOG_EPOCH_MISMATCH"
    SCENE_REVISION_MISMATCH = "SCENE_REVISION_MISMATCH"
    INVALID_ROI = "INVALID_ROI"
    QUEUE_FULL = "QUEUE_FULL"
    DUPLICATE_REQUEST_CONFLICT = "DUPLICATE_REQUEST_CONFLICT"
    DUPLICATE_REQUEST_RETIRED = "DUPLICATE_REQUEST_RETIRED"
    TARGET_RETIRED = "TARGET_RETIRED"
    PRODUCT_TARGET_INSTANCE_MISMATCH = "PRODUCT_TARGET_INSTANCE_MISMATCH"


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


@dataclass(frozen=True, order=True)
class RequestKey:
    ground_instance_id: int
    request_id: int

    def __post_init__(self) -> None:
        checked_u64(self.ground_instance_id, "ground_instance_id")
        checked_u32(self.request_id, "request_id")

    def as_wire(self) -> bytes:
        return u64_to_bytes(self.ground_instance_id) + self.request_id.to_bytes(4, "big")

    def as_dict(self) -> dict[str, Any]:
        return {
            "ground_instance_id": u64_to_json(self.ground_instance_id),
            "request_id": self.request_id,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "RequestKey":
        data = _require_mapping(value, "request_key")
        return cls(
            u64_from_json(data.get("ground_instance_id"), "ground_instance_id"),
            checked_u32(data.get("request_id"), "request_id"),
        )


@dataclass(frozen=True, order=True)
class SceneRef:
    catalog_epoch: int
    scene_id: int
    scene_revision: int

    def __post_init__(self) -> None:
        checked_u32(self.catalog_epoch, "catalog_epoch")
        checked_u32(self.scene_id, "scene_id")
        checked_u32(self.scene_revision, "scene_revision")

    def as_dict(self) -> dict[str, int]:
        return {
            "catalog_epoch": self.catalog_epoch,
            "scene_id": self.scene_id,
            "scene_revision": self.scene_revision,
        }

    def as_wire(self) -> bytes:
        return b"".join(
            value.to_bytes(4, "big")
            for value in (self.catalog_epoch, self.scene_id, self.scene_revision)
        )

    @classmethod
    def from_dict(cls, value: Any) -> "SceneRef":
        data = _require_mapping(value, "scene_ref")
        return cls(
            checked_u32(data.get("catalog_epoch"), "catalog_epoch"),
            checked_u32(data.get("scene_id"), "scene_id"),
            checked_u32(data.get("scene_revision"), "scene_revision"),
        )


@dataclass(frozen=True)
class ScopedSceneRef:
    spacecraft_instance_id: int
    scene_ref: SceneRef

    def __post_init__(self) -> None:
        checked_u64(self.spacecraft_instance_id, "spacecraft_instance_id")

    def as_dict(self) -> dict[str, Any]:
        return {
            "spacecraft_instance_id": u64_to_json(self.spacecraft_instance_id),
            "scene_ref": self.scene_ref.as_dict(),
        }


@dataclass(frozen=True)
class ProductRef:
    spacecraft_instance_id: int
    origin_boot_id: int
    product_id: int

    def __post_init__(self) -> None:
        checked_u64(self.spacecraft_instance_id, "spacecraft_instance_id")
        checked_u32(self.origin_boot_id, "origin_boot_id")
        checked_u32(self.product_id, "product_id")

    def as_dict(self) -> dict[str, Any]:
        return {
            "spacecraft_instance_id": u64_to_json(self.spacecraft_instance_id),
            "origin_boot_id": self.origin_boot_id,
            "product_id": self.product_id,
        }

    def as_wire(self) -> bytes:
        return (
            u64_to_bytes(self.spacecraft_instance_id)
            + self.origin_boot_id.to_bytes(4, "big")
            + self.product_id.to_bytes(4, "big")
        )

    @classmethod
    def from_dict(cls, value: Any) -> "ProductRef":
        data = _require_mapping(value, "product_ref")
        return cls(
            u64_from_json(data.get("spacecraft_instance_id"), "spacecraft_instance_id"),
            checked_u32(data.get("origin_boot_id"), "origin_boot_id"),
            checked_u32(data.get("product_id"), "product_id"),
        )


@dataclass(frozen=True)
class ROI:
    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        for name, value in (
            ("x", self.x),
            ("y", self.y),
            ("width", self.width),
            ("height", self.height),
        ):
            checked_u32(value, name)
        if self.width == 0 or self.height == 0:
            raise ValueError("ROI width and height must be positive")
        checked_add_u64(self.x, self.width, "ROI x end")
        checked_add_u64(self.y, self.height, "ROI y end")

    @property
    def x_end(self) -> int:
        return self.x + self.width

    @property
    def y_end(self) -> int:
        return self.y + self.height

    def as_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}

    def as_wire(self) -> bytes:
        return b"".join(
            value.to_bytes(4, "big") for value in (self.x, self.y, self.width, self.height)
        )

    @classmethod
    def from_dict(cls, value: Any) -> "ROI":
        data = _require_mapping(value, "roi")
        return cls(
            checked_u32(data.get("x"), "x"),
            checked_u32(data.get("y"), "y"),
            checked_u32(data.get("width"), "width"),
            checked_u32(data.get("height"), "height"),
        )


@dataclass(frozen=True)
class ConfigSnapshot:
    epoch: int
    revision: int
    model_threshold_bp: int
    coverage_limit_bp: int

    def __post_init__(self) -> None:
        checked_u32(self.epoch, "config_epoch")
        checked_u32(self.revision, "config_revision")
        checked_u16(self.model_threshold_bp, "model_threshold_bp")
        checked_u16(self.coverage_limit_bp, "coverage_limit_bp")
        if self.model_threshold_bp > 10000 or self.coverage_limit_bp > 10000:
            raise ValueError("threshold basis points must be in [0, 10000]")

    def as_dict(self) -> dict[str, int]:
        return {
            "config_epoch": self.epoch,
            "config_revision": self.revision,
            "model_threshold_bp": self.model_threshold_bp,
            "coverage_limit_bp": self.coverage_limit_bp,
        }

    def as_wire(self) -> bytes:
        return (
            self.epoch.to_bytes(4, "big")
            + self.revision.to_bytes(4, "big")
            + self.model_threshold_bp.to_bytes(2, "big")
            + self.coverage_limit_bp.to_bytes(2, "big")
        )


def _read_u16(data: bytes, offset: int, label: str) -> tuple[int, int]:
    end = offset + 2
    if end > len(data):
        raise ValueError(f"truncated command field {label}")
    return int.from_bytes(data[offset:end], "big"), end


def _read_u32(data: bytes, offset: int, label: str) -> tuple[int, int]:
    end = offset + 4
    if end > len(data):
        raise ValueError(f"truncated command field {label}")
    return int.from_bytes(data[offset:end], "big"), end


def _read_u64(data: bytes, offset: int, label: str) -> tuple[int, int]:
    end = offset + 8
    if end > len(data):
        raise ValueError(f"truncated command field {label}")
    return int.from_bytes(data[offset:end], "big"), end


@dataclass(frozen=True)
class Command:
    opcode: CommandOpcode
    target_spacecraft_instance_id: int
    request_key: RequestKey
    payload: dict[str, Any]

    def __post_init__(self) -> None:
        checked_u64(self.target_spacecraft_instance_id, "target_spacecraft_instance_id")
        if not isinstance(self.payload, dict):
            raise TypeError("command payload must be a dictionary")

    def semantic_dict(self) -> dict[str, Any]:
        return {
            "opcode": int(self.opcode),
            "target_spacecraft_instance_id": u64_to_json(
                self.target_spacecraft_instance_id
            ),
            "request_key": self.request_key.as_dict(),
            "payload": self.payload,
        }

    def argument_bytes(self) -> bytes:
        return (
            u64_to_bytes(self.target_spacecraft_instance_id)
            + self.request_key.as_wire()
            + _encode_payload(self.opcode, self.payload)
        )

    def wire_bytes(self) -> bytes:
        return int(self.opcode).to_bytes(4, "big") + self.argument_bytes()


def _config_from_payload(payload: Mapping[str, Any]) -> tuple[int, int, int, int]:
    return (
        checked_u32(payload.get("expected_config_epoch"), "expected_config_epoch"),
        checked_u32(payload.get("expected_config_revision"), "expected_config_revision"),
        checked_u16(payload.get("model_threshold_bp"), "model_threshold_bp"),
        checked_u16(payload.get("coverage_limit_bp"), "coverage_limit_bp"),
    )


def _encode_payload(opcode: CommandOpcode, payload: Mapping[str, Any]) -> bytes:
    if opcode == CommandOpcode.CLOUD_SET_CONFIG:
        epoch, revision, model_bp, coverage_bp = _config_from_payload(payload)
        return (
            epoch.to_bytes(4, "big")
            + revision.to_bytes(4, "big")
            + model_bp.to_bytes(2, "big")
            + coverage_bp.to_bytes(2, "big")
        )
    if opcode == CommandOpcode.SCENE_REQUEST_CATALOG:
        return b""
    if opcode in {
        CommandOpcode.SCENE_REQUEST_PREVIEW,
        CommandOpcode.SCENE_ANALYZE,
        CommandOpcode.ROI_REQUEST,
    }:
        scene = SceneRef.from_dict(payload.get("scene_ref"))
        result = scene.as_wire()
        if opcode == CommandOpcode.SCENE_REQUEST_PREVIEW:
            return result
        epoch, revision, model_bp, coverage_bp = _config_from_payload(payload)
        result += epoch.to_bytes(4, "big") + revision.to_bytes(4, "big")
        if opcode == CommandOpcode.ROI_REQUEST:
            result += ROI.from_dict(payload.get("roi")).as_wire()
        return result + model_bp.to_bytes(2, "big") + coverage_bp.to_bytes(2, "big")
    if opcode in {CommandOpcode.JOB_GET_STATUS, CommandOpcode.JOB_CANCEL}:
        return RequestKey.from_dict(payload.get("target_request_key")).as_wire()
    if opcode == CommandOpcode.PRODUCT_REQUEST_DOWNLINK:
        return (
            RequestKey.from_dict(payload.get("origin_request_key")).as_wire()
            + ProductRef.from_dict(payload.get("product_ref")).as_wire()
        )
    if opcode == CommandOpcode.PRODUCT_CANCEL_DOWNLINK:
        transfer_id = checked_u32(payload.get("transfer_id"), "transfer_id")
        return ProductRef.from_dict(payload.get("product_ref")).as_wire() + transfer_id.to_bytes(4, "big")
    raise ValueError(f"unsupported command opcode {opcode}")


def encode_command(command: Command) -> bytes:
    return command.wire_bytes()


def decode_command(data: bytes) -> Command:
    data = bytes(data)
    opcode_value, offset = _read_u32(data, 0, "opcode")
    try:
        opcode = CommandOpcode(opcode_value)
    except ValueError as exc:
        raise ValueError(f"unknown command opcode 0x{opcode_value:08x}") from exc
    target, offset = _read_u64(data, offset, "target_spacecraft_instance_id")
    ground, offset = _read_u64(data, offset, "ground_instance_id")
    request_id, offset = _read_u32(data, offset, "request_id")
    request_key = RequestKey(ground, request_id)
    payload: dict[str, Any] = {}
    if opcode == CommandOpcode.CLOUD_SET_CONFIG:
        epoch, offset = _read_u32(data, offset, "expected_config_epoch")
        revision, offset = _read_u32(data, offset, "expected_config_revision")
        model_bp, offset = _read_u16(data, offset, "model_threshold_bp")
        coverage_bp, offset = _read_u16(data, offset, "coverage_limit_bp")
        payload.update(
            expected_config_epoch=epoch,
            expected_config_revision=revision,
            model_threshold_bp=model_bp,
            coverage_limit_bp=coverage_bp,
        )
    elif opcode == CommandOpcode.SCENE_REQUEST_CATALOG:
        pass
    elif opcode in {
        CommandOpcode.SCENE_REQUEST_PREVIEW,
        CommandOpcode.SCENE_ANALYZE,
        CommandOpcode.ROI_REQUEST,
    }:
        values = []
        for label in ("catalog_epoch", "scene_id", "scene_revision"):
            value, offset = _read_u32(data, offset, label)
            values.append(value)
        payload["scene_ref"] = SceneRef(*values).as_dict()
        if opcode != CommandOpcode.SCENE_REQUEST_PREVIEW:
            epoch, offset = _read_u32(data, offset, "expected_config_epoch")
            revision, offset = _read_u32(data, offset, "expected_config_revision")
            payload.update(expected_config_epoch=epoch, expected_config_revision=revision)
            if opcode == CommandOpcode.ROI_REQUEST:
                roi_values = []
                for label in ("x", "y", "width", "height"):
                    value, offset = _read_u32(data, offset, label)
                    roi_values.append(value)
                payload["roi"] = ROI(*roi_values).as_dict()
            model_bp, offset = _read_u16(data, offset, "model_threshold_bp")
            coverage_bp, offset = _read_u16(data, offset, "coverage_limit_bp")
            payload.update(model_threshold_bp=model_bp, coverage_limit_bp=coverage_bp)
    elif opcode in {CommandOpcode.JOB_GET_STATUS, CommandOpcode.JOB_CANCEL}:
        ground, offset = _read_u64(data, offset, "target_ground_instance_id")
        request_id, offset = _read_u32(data, offset, "target_request_id")
        payload["target_request_key"] = RequestKey(ground, request_id).as_dict()
    elif opcode == CommandOpcode.PRODUCT_REQUEST_DOWNLINK:
        ground, offset = _read_u64(data, offset, "origin_ground_instance_id")
        request_id, offset = _read_u32(data, offset, "origin_request_id")
        payload["origin_request_key"] = RequestKey(ground, request_id).as_dict()
        instance, offset = _read_u64(data, offset, "product_spacecraft_instance_id")
        boot, offset = _read_u32(data, offset, "origin_boot_id")
        product_id, offset = _read_u32(data, offset, "product_id")
        payload["product_ref"] = ProductRef(instance, boot, product_id).as_dict()
    elif opcode == CommandOpcode.PRODUCT_CANCEL_DOWNLINK:
        instance, offset = _read_u64(data, offset, "product_spacecraft_instance_id")
        boot, offset = _read_u32(data, offset, "origin_boot_id")
        product_id, offset = _read_u32(data, offset, "product_id")
        transfer_id, offset = _read_u32(data, offset, "transfer_id")
        payload["product_ref"] = ProductRef(instance, boot, product_id).as_dict()
        payload["transfer_id"] = transfer_id
    if offset != len(data):
        raise ValueError(f"trailing bytes in {opcode.name} command")
    return Command(opcode, target, request_key, payload)


def mission_digest(command: Command) -> str:
    arguments = command.argument_bytes()
    data = (
        b"mission-command-v1\0"
        + PROTOCOL_SCHEMA_VERSION.to_bytes(2, "big")
        + int(command.opcode).to_bytes(4, "big")
        + len(arguments).to_bytes(4, "big")
        + arguments
    )
    return hashlib.sha256(data).hexdigest()


def _normalize_http_expiry(value: Any) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError("expires_at datetime must be timezone-aware")
        parsed = value.astimezone(UTC)
    else:
        if not isinstance(value, str) or _HTTP_RFC3339.fullmatch(value) is None:
            raise ValueError(
                "expires_at must be an RFC 3339 timestamp with an explicit timezone"
            )
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
        except ValueError as exc:
            raise ValueError("expires_at is not a valid timestamp") from exc
    base = parsed.strftime("%Y-%m-%dT%H:%M:%S")
    if parsed.microsecond:
        return f"{base}.{parsed.microsecond:06d}".rstrip("0") + "Z"
    return f"{base}Z"


def normalize_http_idempotency_body(body: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize clock-independent fields before RFC 8785 serialization."""

    if not isinstance(body, Mapping):
        raise TypeError("HTTP semantic body must be an object")
    normalized = dict(body)
    delivery_mode = normalized.get("delivery_mode", "immediate")
    if delivery_mode not in {"immediate", "next_contact"}:
        raise ValueError("invalid delivery_mode")
    normalized["delivery_mode"] = delivery_mode
    if "expires_at" not in normalized:
        normalized["expires_at"] = "DEFAULT"
    else:
        normalized["expires_at"] = _normalize_http_expiry(normalized["expires_at"])
    return normalized


def http_idempotency_digest(body: Mapping[str, Any]) -> str:
    """Digest a validated semantic HTTP body using RFC 8785 JCS."""

    normalized = normalize_http_idempotency_body(body)
    return hashlib.sha256(jcs_canonical_json(normalized)).hexdigest()
