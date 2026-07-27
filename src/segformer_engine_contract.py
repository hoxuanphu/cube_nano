"""Pinned TensorRT engine artifact contract for SegFormer-B0."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class SegFormerEngineManifest:
    schema_version: int
    engine_sha256: str
    model_release_id: str
    target_id: str
    precision: str
    input_shape: tuple[int, ...]
    input_dtype: str
    output_shape: tuple[int, ...]
    output_dtype: str
    builder_flags: tuple[str, ...]
    plugin_hashes: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SegFormerEngineManifest":
        if not isinstance(value, Mapping):
            raise ValueError("SegFormer engine manifest must be an object")
        result = cls(
            schema_version=int(value.get("schema_version")),
            engine_sha256=str(value.get("engine_sha256", "")).lower(),
            model_release_id=str(value.get("model_release_id", "")),
            target_id=str(value.get("target_id", "")),
            precision=str(value.get("precision", "")),
            input_shape=tuple(int(item) for item in value.get("input_shape", ())),
            input_dtype=str(value.get("input_dtype", "")),
            output_shape=tuple(int(item) for item in value.get("output_shape", ())),
            output_dtype=str(value.get("output_dtype", "")),
            builder_flags=tuple(str(item) for item in value.get("builder_flags", ())),
            plugin_hashes=tuple(str(item) for item in value.get("plugin_hashes", ())),
        )
        result.validate()
        return result

    @classmethod
    def from_file(cls, path: str | Path) -> "SegFormerEngineManifest":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_mapping(value)

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError("SegFormer engine manifest schema_version must be 1")
        if len(self.engine_sha256) != 64 or any(char not in "0123456789abcdef" for char in self.engine_sha256):
            raise ValueError("engine_sha256 must be a SHA-256 digest")
        if not self.model_release_id or not self.target_id:
            raise ValueError("engine manifest model/target identity must not be empty")
        if self.precision != "fp16":
            raise ValueError("SegFormer TensorRT MVP engine precision must be fp16")
        if self.input_shape != (1, 3, 256, 256) or self.output_shape != (1, 2, 64, 64):
            raise ValueError("SegFormer engine input/output shape is not the pinned MVP contract")
        if self.input_dtype != "float32" or self.output_dtype not in {"float16", "float32"}:
            raise ValueError("SegFormer engine input/output physical dtype is unsupported")

    def verify_engine(self, path: str | Path) -> None:
        actual = _sha256(Path(path))
        if actual != self.engine_sha256:
            raise ValueError(f"TensorRT engine SHA-256 mismatch: expected={self.engine_sha256}, actual={actual}")
