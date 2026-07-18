import hashlib
import hmac
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SUPPORTED_BAND_ROLES = ("red", "green", "blue", "nir")


def _load_json_object(path, label):
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"{label} not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _normalization_id(value):
    if isinstance(value, str) and value:
        return value
    if isinstance(value, dict) and isinstance(value.get("id"), str) and value["id"]:
        return value["id"]
    raise ValueError("normalization must be a non-empty ID or an object with a non-empty 'id'")


def parse_channel_mapping(value):
    if value is None:
        return None
    if isinstance(value, dict):
        items = value.items()
    elif isinstance(value, str):
        if not value.strip():
            raise ValueError("channel_mapping must not be empty")
        parsed = []
        for entry in value.split(","):
            role, separator, index = entry.partition("=")
            if not separator:
                raise ValueError(
                    "channel_mapping entries must use role=index, for example red=0,green=1,blue=2"
                )
            parsed.append((role.strip(), index.strip()))
        items = parsed
    else:
        raise TypeError("channel_mapping must be a string, mapping, or None")

    result = {}
    for raw_role, raw_index in items:
        role = str(raw_role).strip().lower()
        if role not in SUPPORTED_BAND_ROLES:
            raise ValueError(f"Unsupported channel role '{raw_role}'")
        if role in result:
            raise ValueError(f"Duplicate channel role '{role}'")
        try:
            index = int(raw_index)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Channel index for '{role}' must be an integer") from exc
        if index < 0:
            raise ValueError(f"Channel index for '{role}' must be non-negative")
        result[role] = index

    if len(set(result.values())) != len(result):
        raise ValueError("channel_mapping must not contain duplicate physical indexes")
    return result


def _fingerprint_digest(payload, label):
    if isinstance(payload, str):
        return payload.lower()
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a SHA-256 object")
    if str(payload.get("algorithm", "")).lower() != "sha256":
        raise ValueError(f"{label}.algorithm must be 'sha256'")
    digest = payload.get("digest")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"{label}.digest must be a 64-character SHA-256 digest")
    return digest.lower()


@dataclass(frozen=True)
class InputSidecar:
    schema_version: int
    source_digest: str
    axes: str
    shape: tuple
    band_order: tuple
    dtype: np.dtype
    input_spec_id: str
    normalization_id: str

    @property
    def role_to_index(self):
        return {role: index for index, role in enumerate(self.band_order)}


def load_input_sidecar(path, source_path, verify_fingerprint=True):
    payload = _load_json_object(path, "input sidecar")
    if payload.get("schema_version") != 1:
        raise ValueError("input sidecar schema_version must be 1")

    source_digest = _fingerprint_digest(payload.get("source_fingerprint"), "source_fingerprint")
    if verify_fingerprint:
        actual_digest = sha256_file(source_path)
        if not hmac.compare_digest(source_digest, actual_digest):
            raise ValueError("input sidecar source_fingerprint does not match the TIFF input")

    axes = payload.get("axes")
    if not isinstance(axes, str) or not axes:
        raise ValueError("input sidecar axes must be a non-empty string")

    raw_shape = payload.get("shape")
    if not isinstance(raw_shape, list) or not raw_shape:
        raise ValueError("input sidecar shape must be a non-empty list")
    try:
        shape = tuple(int(value) for value in raw_shape)
    except (TypeError, ValueError) as exc:
        raise ValueError("input sidecar shape values must be integers") from exc
    if any(value <= 0 for value in shape):
        raise ValueError("input sidecar shape values must be positive")

    raw_band_order = payload.get("band_order")
    if not isinstance(raw_band_order, list) or not raw_band_order:
        raise ValueError("input sidecar band_order must be a non-empty list")
    band_order = tuple(str(role).lower() for role in raw_band_order)
    if len(set(band_order)) != len(band_order):
        raise ValueError("input sidecar band_order must not contain duplicate roles")
    unsupported = set(band_order) - set(SUPPORTED_BAND_ROLES)
    if unsupported:
        raise ValueError(f"Unsupported sidecar band roles: {sorted(unsupported)}")

    try:
        dtype = np.dtype(payload.get("dtype"))
    except TypeError as exc:
        raise ValueError("input sidecar dtype is invalid") from exc

    input_spec_id = payload.get("input_spec_id")
    if not isinstance(input_spec_id, str) or not input_spec_id:
        raise ValueError("input sidecar input_spec_id must be a non-empty string")

    return InputSidecar(
        schema_version=1,
        source_digest=source_digest,
        axes=axes,
        shape=shape,
        band_order=band_order,
        dtype=dtype,
        input_spec_id=input_spec_id,
        normalization_id=_normalization_id(payload.get("normalization")),
    )


def _channel_values(payload, name, channels, default):
    value = payload.get(name, default)
    if np.isscalar(value):
        values = (float(value),) * channels
    else:
        try:
            values = tuple(float(item) for item in value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"normalization.{name} must contain numbers") from exc
    if len(values) != channels:
        raise ValueError(f"normalization.{name} must contain {channels} values")
    if not all(np.isfinite(item) for item in values):
        raise ValueError(f"normalization.{name} values must be finite")
    return values


@dataclass(frozen=True)
class NormalizationSpec:
    id: str
    kind: str
    scale: tuple
    offset: tuple
    mean: tuple
    std: tuple
    clip_min: float | None = None
    clip_max: float | None = None

    @classmethod
    def from_value(cls, value, channels):
        if isinstance(value, str):
            payload = {"id": value, "kind": value}
        elif isinstance(value, dict):
            payload = value
        else:
            raise ValueError("normalization must be an ID or object")

        normalization_id = _normalization_id(payload)
        kind = str(payload.get("kind", normalization_id)).lower().replace("_", "-")
        aliases = {
            "dtype-range-v1": "dtype-range",
            "legacy-dtype-range-v1": "dtype-range",
            "identity-v1": "identity",
            "scale-offset-v1": "scale-offset",
            "standardize-v1": "standardize",
        }
        kind = aliases.get(kind, kind)
        if kind not in {"dtype-range", "identity", "scale-offset", "standardize"}:
            raise ValueError(f"Unsupported normalization kind '{kind}'")

        scale = _channel_values(payload, "scale", channels, 1.0)
        offset = _channel_values(payload, "offset", channels, 0.0)
        mean = _channel_values(payload, "mean", channels, 0.0)
        std = _channel_values(payload, "std", channels, 1.0)
        if any(item == 0 for item in std):
            raise ValueError("normalization.std values must be non-zero")

        clip = payload.get("clip")
        clip_min = clip_max = None
        if clip is not None:
            if not isinstance(clip, (list, tuple)) or len(clip) != 2:
                raise ValueError("normalization.clip must be [minimum, maximum]")
            clip_min, clip_max = (float(clip[0]), float(clip[1]))
            if not np.isfinite(clip_min) or not np.isfinite(clip_max) or clip_min >= clip_max:
                raise ValueError("normalization.clip must contain finite increasing values")

        return cls(
            id=normalization_id,
            kind=kind,
            scale=scale,
            offset=offset,
            mean=mean,
            std=std,
            clip_min=clip_min,
            clip_max=clip_max,
        )

    def apply(self, patch):
        patch = np.asarray(patch)
        if patch.ndim != 3 or patch.shape[-1] != len(self.scale):
            raise ValueError(
                f"Normalization expects HWC data with {len(self.scale)} channels, got {patch.shape}"
            )
        original_dtype = patch.dtype
        result = patch.astype(np.float32)

        if self.kind == "dtype-range":
            if np.issubdtype(original_dtype, np.integer):
                limits = np.iinfo(original_dtype)
                result = (result - float(limits.min)) / float(limits.max - limits.min)
            elif np.issubdtype(original_dtype, np.floating):
                if not np.all(np.isfinite(result)):
                    raise ValueError("Float input contains NaN or infinite values")
                if result.size and (float(result.min()) < 0.0 or float(result.max()) > 1.0):
                    raise ValueError(
                        "dtype-range normalization requires float input already in [0, 1]"
                    )
            else:
                raise ValueError(f"Unsupported image dtype {original_dtype}")
        elif self.kind in {"scale-offset", "standardize"}:
            result = result * np.asarray(self.scale, dtype=np.float32)
            result = result + np.asarray(self.offset, dtype=np.float32)
        elif self.kind != "identity":
            raise AssertionError(f"Unhandled normalization kind {self.kind}")

        if self.kind == "standardize":
            result = result - np.asarray(self.mean, dtype=np.float32)
            result = result / np.asarray(self.std, dtype=np.float32)

        if self.clip_min is not None:
            result = np.clip(result, self.clip_min, self.clip_max)
        if not np.all(np.isfinite(result)):
            raise ValueError("Normalization produced NaN or infinite values")
        return np.asarray(result, dtype=np.float32)


@dataclass(frozen=True)
class EngineInputSpec:
    schema_version: int
    input_spec_id: str
    band_order: tuple
    normalization: NormalizationSpec
    input_shape: tuple
    input_dtype: np.dtype
    engine_digest: str | None = None
    manifest_path: Path | None = None

    @property
    def channels(self):
        return self.input_shape[1]

    @property
    def patch_size(self):
        if self.input_shape[2] != self.input_shape[3]:
            raise ValueError(f"Engine input must use square patches, got {self.input_shape[2:]}")
        return self.input_shape[2]


def load_engine_manifest(path, engine_path=None, verify_fingerprint=True):
    payload = _load_json_object(path, "engine manifest")
    if payload.get("schema_version") != 1:
        raise ValueError("engine manifest schema_version must be 1")

    input_spec_payload = payload.get("input_spec")
    if input_spec_payload is None:
        input_spec_payload = payload
    if not isinstance(input_spec_payload, dict):
        raise ValueError("engine manifest input_spec must be an object")

    input_shape_raw = payload.get("input_shape", input_spec_payload.get("input_shape"))
    if not isinstance(input_shape_raw, list) or len(input_shape_raw) != 4:
        raise ValueError("engine manifest input_shape must be [batch, channels, height, width]")
    try:
        input_shape = tuple(int(value) for value in input_shape_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("engine manifest input_shape values must be integers") from exc
    if any(value <= 0 for value in input_shape):
        raise ValueError("engine manifest input_shape must be fixed and positive")

    raw_band_order = input_spec_payload.get("band_order")
    if not isinstance(raw_band_order, list):
        raise ValueError("engine manifest band_order must be a list")
    band_order = tuple(str(role).lower() for role in raw_band_order)
    expected_roles = ("red", "green", "blue") if input_shape[1] == 3 else (
        "red",
        "green",
        "blue",
        "nir",
    )
    if input_shape[1] not in (3, 4) or band_order != expected_roles:
        raise ValueError(
            f"engine manifest band_order must be canonical for {input_shape[1]} channels"
        )

    input_spec_id = input_spec_payload.get("input_spec_id", input_spec_payload.get("id"))
    if not isinstance(input_spec_id, str) or not input_spec_id:
        raise ValueError("engine manifest input_spec_id must be a non-empty string")

    engine_digest = None
    fingerprint = payload.get("engine_fingerprint")
    if fingerprint is not None:
        engine_digest = _fingerprint_digest(fingerprint, "engine_fingerprint")
    elif engine_path is not None:
        raise ValueError("engine manifest must contain engine_fingerprint")

    if engine_path is not None and verify_fingerprint:
        actual_digest = sha256_file(engine_path)
        if not hmac.compare_digest(engine_digest, actual_digest):
            raise ValueError("engine manifest fingerprint does not match the TensorRT engine")

    try:
        input_dtype = np.dtype(payload.get("input_dtype", input_spec_payload.get("input_dtype")))
    except TypeError as exc:
        raise ValueError("engine manifest input_dtype is invalid") from exc

    normalization = NormalizationSpec.from_value(
        input_spec_payload.get("normalization"),
        input_shape[1],
    )
    return EngineInputSpec(
        schema_version=1,
        input_spec_id=input_spec_id,
        band_order=band_order,
        normalization=normalization,
        input_shape=input_shape,
        input_dtype=input_dtype,
        engine_digest=engine_digest,
        manifest_path=Path(path),
    )


def legacy_input_spec(channels, patch_size, input_dtype=np.float32):
    if channels == 3:
        band_order = ("red", "green", "blue")
    elif channels == 4:
        band_order = ("red", "green", "blue", "nir")
    else:
        raise ValueError("Legacy input contract supports exactly 3 or 4 channels")
    normalization = NormalizationSpec.from_value("legacy-dtype-range-v1", channels)
    return EngineInputSpec(
        schema_version=1,
        input_spec_id="legacy-dtype-range-v1",
        band_order=band_order,
        normalization=normalization,
        input_shape=(1, channels, patch_size, patch_size),
        input_dtype=np.dtype(input_dtype),
    )
