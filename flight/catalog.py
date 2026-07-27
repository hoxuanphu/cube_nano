"""Satellite-authoritative immutable scene catalog."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from protocol.canonical import canonical_json
from protocol.schemas import SceneRef


@dataclass(frozen=True)
class SceneRecord:
    scene_ref: SceneRef
    path: Path
    sidecar_path: Path
    source_sha256: str
    shape: tuple[int, int, int]
    capability: str = "VERIFIED"
    domain: dict[str, str] | None = None
    sidecar_sha256: str | None = None
    catalog_path: str | None = None
    catalog_sidecar_path: str | None = None

    def as_dict(self) -> dict[str, Any]:
        sidecar_digest = self.sidecar_sha256
        if sidecar_digest is None:
            sidecar_digest = hashlib.sha256(self.sidecar_path.read_bytes()).hexdigest()
        return {
            "scene_ref": self.scene_ref.as_dict(),
            "path": self.catalog_path if self.catalog_path is not None else self.path.as_posix(),
            "sidecar_path": (
                self.catalog_sidecar_path
                if self.catalog_sidecar_path is not None
                else self.sidecar_path.as_posix()
            ),
            "source_sha256": self.source_sha256,
            "sidecar_sha256": sidecar_digest,
            "shape": list(self.shape),
            "capability": self.capability,
            "domain": self.domain or {},
        }


class SceneCatalog:
    def __init__(self, epoch: int, revision: int, scenes: list[SceneRecord], *, snapshot_sha256: str | None = None):
        self.epoch = epoch
        self.revision = revision
        self.scenes = {record.scene_ref.scene_id: record for record in scenes}
        payload = {
            "catalog_epoch": epoch,
            "catalog_revision": revision,
            "scenes": [record.as_dict() for record in sorted(scenes, key=lambda item: (item.scene_ref.scene_id, item.scene_ref.scene_revision))],
        }
        self.snapshot_payload = canonical_json(payload)
        self.snapshot_sha256 = hashlib.sha256(self.snapshot_payload).hexdigest()
        if snapshot_sha256 is not None and snapshot_sha256 != self.snapshot_sha256:
            raise ValueError("catalog snapshot SHA-256 mismatch")

    @classmethod
    def from_file(cls, path: str | Path) -> "SceneCatalog":
        path = Path(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise FileNotFoundError(f"scene catalog not found: {path}") from None
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError("scene catalog schema_version must be 1")
        root = path.parent
        scenes = []
        for item in payload.get("scenes", []):
            scene_ref = SceneRef.from_dict(item.get("scene_ref"))
            source = (root / str(item["path"])).resolve()
            sidecar = (root / str(item["sidecar_path"])).resolve()
            scenes.append(
                SceneRecord(
                    scene_ref,
                    source,
                    sidecar,
                    str(item["source_sha256"]).lower(),
                    tuple(int(value) for value in item["shape"]),
                    str(item.get("capability", "VERIFIED")),
                    dict(item.get("domain", {})),
                    None if item.get("sidecar_sha256") is None else str(item["sidecar_sha256"]).lower(),
                    str(item["path"]),
                    str(item["sidecar_path"]),
                )
            )
        return cls(int(payload["catalog_epoch"]), int(payload["catalog_revision"]), scenes, snapshot_sha256=payload.get("snapshot_sha256"))

    def get(self, scene_ref: SceneRef) -> SceneRecord:
        if scene_ref.catalog_epoch != self.epoch:
            raise ValueError("CATALOG_EPOCH_MISMATCH")
        record = self.scenes.get(scene_ref.scene_id)
        if record is None:
            raise ValueError("SCENE_NOT_FOUND")
        if record.scene_ref.scene_revision != scene_ref.scene_revision:
            raise ValueError("SCENE_REVISION_MISMATCH")
        if record.capability != "VERIFIED":
            raise ValueError(f"SCENE_{record.capability}")
        return record

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "catalog_epoch": self.epoch,
            "catalog_revision": self.revision,
            "snapshot_sha256": self.snapshot_sha256,
            "scenes": [record.as_dict() for record in sorted(self.scenes.values(), key=lambda item: item.scene_ref.scene_id)],
        }

    def bundle_bytes(self) -> bytes:
        """Return the deterministic catalog snapshot sent over the TM file path."""

        return canonical_json(self.snapshot()) + b"\n"

    @classmethod
    def from_bundle_bytes(cls, data: bytes) -> "SceneCatalog":
        try:
            payload = json.loads(bytes(data).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("catalog bundle is not valid UTF-8 JSON") from exc
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError("catalog bundle schema_version must be 1")
        # The snapshot hash is defined over the canonical payload without the
        # hash field itself.  This keeps the authority digest independent of
        # the transport bundle and manifest bytes.
        scenes = payload.get("scenes")
        if not isinstance(scenes, list):
            raise ValueError("catalog bundle scenes must be an array")
        unsigned = {
            "catalog_epoch": payload.get("catalog_epoch"),
            "catalog_revision": payload.get("catalog_revision"),
            "scenes": scenes,
        }
        expected = hashlib.sha256(canonical_json(unsigned)).hexdigest()
        if str(payload.get("snapshot_sha256", "")).lower() != expected:
            raise ValueError("catalog snapshot SHA-256 mismatch")
        return cls(
            int(payload["catalog_epoch"]),
            int(payload["catalog_revision"]),
            [
                SceneRecord(
                    SceneRef.from_dict(item.get("scene_ref")),
                    Path(str(item["path"])),
                    Path(str(item["sidecar_path"])),
                    str(item["source_sha256"]).lower(),
                    tuple(int(value) for value in item["shape"]),
                    str(item.get("capability", "VERIFIED")),
                    dict(item.get("domain", {})),
                    str(item["sidecar_sha256"]).lower(),
                    str(item["path"]),
                    str(item["sidecar_path"]),
                )
                for item in scenes
            ],
            snapshot_sha256=expected,
        )
