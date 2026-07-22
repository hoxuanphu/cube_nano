"""Instance-scoped verified replicas of the satellite scene catalog."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from protocol.canonical import canonical_json, checked_u32, checked_u64, u64_to_json
from protocol.schemas import SceneRef, ScopedSceneRef

from .u64 import encode_sqlite_u64
from .writer import MutationPriority, SQLiteWriter


class CatalogError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class CatalogScene:
    scene_ref: SceneRef
    source_sha256: str
    sidecar_sha256: str
    shape: tuple[int, int, int]
    capability: str
    domain: dict[str, Any]
    metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "scene_ref": self.scene_ref.as_dict(),
            "source_sha256": self.source_sha256,
            "sidecar_sha256": self.sidecar_sha256,
            "shape": list(self.shape),
            "capability": self.capability,
            "domain": self.domain,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class CatalogBundle:
    epoch: int
    revision: int
    snapshot_sha256: str
    scenes: tuple[CatalogScene, ...]
    bundle_sha256: str
    raw_bytes: bytes

    @classmethod
    def parse(cls, data: bytes) -> "CatalogBundle":
        raw = bytes(data)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CatalogError("CATALOG_DECODE_ERROR", "catalog bundle is not UTF-8 JSON") from exc
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise CatalogError("CATALOG_SCHEMA_ERROR", "catalog schema_version must be 1")
        if canonical_json(payload) + b"\n" != raw:
            raise CatalogError("CATALOG_NON_CANONICAL", "catalog bundle is not canonical JSON")
        epoch = checked_u32(payload.get("catalog_epoch"), "catalog_epoch")
        revision = checked_u32(payload.get("catalog_revision"), "catalog_revision")
        raw_scenes = payload.get("scenes")
        if not isinstance(raw_scenes, list):
            raise CatalogError("CATALOG_SCHEMA_ERROR", "catalog scenes must be an array")
        unsigned = {"catalog_epoch": epoch, "catalog_revision": revision, "scenes": raw_scenes}
        snapshot_sha = hashlib.sha256(canonical_json(unsigned)).hexdigest()
        if str(payload.get("snapshot_sha256", "")).lower() != snapshot_sha:
            raise CatalogError("CATALOG_CHECKSUM_FAILED", "catalog snapshot SHA-256 mismatch")
        scenes: list[CatalogScene] = []
        seen: set[tuple[int, int]] = set()
        for item in raw_scenes:
            if not isinstance(item, Mapping):
                raise CatalogError("CATALOG_SCHEMA_ERROR", "catalog scene entry must be an object")
            ref = SceneRef.from_dict(item.get("scene_ref"))
            if ref.catalog_epoch != epoch:
                raise CatalogError("CATALOG_EPOCH_MISMATCH", "scene epoch differs from catalog epoch")
            identity = (ref.scene_id, ref.scene_revision)
            if identity in seen:
                raise CatalogError("CATALOG_DUPLICATE_SCENE", "catalog contains duplicate scene identity")
            seen.add(identity)
            source_sha = str(item.get("source_sha256", "")).lower()
            sidecar_sha = str(item.get("sidecar_sha256", "")).lower()
            if not _is_sha256(source_sha) or not _is_sha256(sidecar_sha):
                raise CatalogError("CATALOG_SCHEMA_ERROR", "scene source/sidecar hashes must be SHA-256")
            shape = tuple(int(value) for value in item.get("shape", ()))
            if len(shape) != 3 or any(value <= 0 for value in shape):
                raise CatalogError("CATALOG_SCHEMA_ERROR", "scene shape must be positive HWC")
            capability = str(item.get("capability", "UNSUPPORTED")).upper()
            if capability not in {"VERIFIED", "UNSUPPORTED", "INVALID"}:
                raise CatalogError("CATALOG_SCHEMA_ERROR", "unknown scene capability")
            domain = item.get("domain", {})
            if not isinstance(domain, Mapping):
                raise CatalogError("CATALOG_SCHEMA_ERROR", "scene domain must be an object")
            metadata = dict(item.get("metadata", {}))
            # Paths from the satellite are descriptive only.  GDS never uses
            # them as a local filesystem path.
            scenes.append(CatalogScene(ref, source_sha, sidecar_sha, shape, capability, dict(domain), metadata))
        scenes.sort(key=lambda item: (item.scene_ref.scene_id, item.scene_ref.scene_revision))
        return cls(epoch, revision, snapshot_sha, tuple(scenes), hashlib.sha256(raw).hexdigest(), raw)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "catalog_epoch": self.epoch,
            "catalog_revision": self.revision,
            "snapshot_sha256": self.snapshot_sha256,
            "scenes": [scene.as_dict() for scene in self.scenes],
            "bundle_sha256": self.bundle_sha256,
        }


@dataclass(frozen=True)
class CatalogStatus:
    spacecraft_instance_id: int
    catalog_epoch: int | None
    catalog_revision: int | None
    snapshot_sha256: str | None
    synced: bool
    stale: bool
    scene_count: int
    source_boot_id: int | None
    link_session_id: int | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "spacecraft_instance_id": u64_to_json(self.spacecraft_instance_id),
            "catalog_epoch": self.catalog_epoch,
            "catalog_revision": self.catalog_revision,
            "snapshot_sha256": self.snapshot_sha256,
            "synced": self.synced,
            "stale": self.stale,
            "scene_count": self.scene_count,
            "source_boot_id": self.source_boot_id,
            "link_session_id": None if self.link_session_id is None else u64_to_json(self.link_session_id),
        }


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


class CatalogReplicaStore:
    """Stage and atomically activate a complete verified catalog snapshot."""

    def __init__(self, writer: SQLiteWriter):
        self.writer = writer

    def activate(
        self,
        spacecraft_instance_id: int,
        bundle: CatalogBundle | bytes,
        *,
        source_boot_id: int | None = None,
        link_session_id: int | None = None,
        received_at_us: int = 0,
    ) -> CatalogStatus:
        instance = checked_u64(spacecraft_instance_id, "spacecraft_instance_id")
        parsed = bundle if isinstance(bundle, CatalogBundle) else CatalogBundle.parse(bundle)
        if source_boot_id is not None:
            checked_u32(source_boot_id, "source_boot_id")
        if link_session_id is not None:
            checked_u64(link_session_id, "link_session_id")
        if received_at_us < 0:
            raise ValueError("received_at_us must be non-negative")

        def mutation(connection):
            instance_blob = encode_sqlite_u64(instance)
            existing = connection.execute(
                "SELECT snapshot_sha256,state,is_active FROM catalog_snapshots "
                "WHERE source_spacecraft_instance_id=? AND catalog_epoch=? AND catalog_revision=?",
                (instance_blob, parsed.epoch, parsed.revision),
            ).fetchone()
            if existing is not None and bytes(existing[0]).hex() == parsed.snapshot_sha256 and str(existing[1]) == "VERIFIED" and int(existing[2]) == 1:
                return self._status_in_transaction(connection, instance)
            connection.execute(
                "DELETE FROM scenes WHERE source_spacecraft_instance_id=? AND catalog_epoch=? AND catalog_revision=?",
                (instance_blob, parsed.epoch, parsed.revision),
            )
            connection.execute(
                "DELETE FROM catalog_snapshots WHERE source_spacecraft_instance_id=? AND catalog_epoch=? AND catalog_revision=?",
                (instance_blob, parsed.epoch, parsed.revision),
            )
            connection.execute(
                "INSERT INTO catalog_snapshots(source_spacecraft_instance_id,catalog_epoch,catalog_revision,snapshot_sha256,state,manifest_json,synced_at_us,source_boot_id,source_link_session_id,is_active,verified_at_us) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    instance_blob,
                    parsed.epoch,
                    parsed.revision,
                    bytes.fromhex(parsed.snapshot_sha256),
                    "STAGING",
                    parsed.raw_bytes.decode("utf-8"),
                    received_at_us,
                    source_boot_id,
                    None if link_session_id is None else encode_sqlite_u64(link_session_id),
                    0,
                    None,
                ),
            )
            for scene in parsed.scenes:
                metadata = dict(scene.metadata)
                metadata.update({"domain": scene.domain, "shape": list(scene.shape)})
                connection.execute(
                    "INSERT INTO scenes(source_spacecraft_instance_id,catalog_epoch,scene_id,scene_revision,catalog_revision,source_sha256,sidecar_sha256,metadata_json,state,source_stat_json,sidecar_stat_json,invalid_reason,ingested_at_us,active_preview_generation) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,0)",
                    (
                        instance_blob,
                        parsed.epoch,
                        scene.scene_ref.scene_id,
                        scene.scene_ref.scene_revision,
                        parsed.revision,
                        bytes.fromhex(scene.source_sha256),
                        bytes.fromhex(scene.sidecar_sha256),
                        json.dumps(metadata, sort_keys=True, separators=(",", ":")),
                        "ACTIVE" if scene.capability == "VERIFIED" else scene.capability,
                        None,
                        None,
                        None if scene.capability != "INVALID" else "CATALOG_INVALID",
                        received_at_us,
                    ),
                )
            connection.execute(
                "UPDATE catalog_snapshots SET state='RETIRED',is_active=0,retired_at_us=? WHERE source_spacecraft_instance_id=? AND is_active=1",
                (received_at_us, instance_blob),
            )
            connection.execute(
                "UPDATE catalog_snapshots SET state='VERIFIED',is_active=1,verified_at_us=? WHERE source_spacecraft_instance_id=? AND catalog_epoch=? AND catalog_revision=?",
                (received_at_us, instance_blob, parsed.epoch, parsed.revision),
            )
            return self._status_in_transaction(connection, instance)

        return self.writer.mutate("activate_catalog_snapshot", mutation, priority=MutationPriority.HIGH)

    def mark_stale(self, spacecraft_instance_id: int, *, at_us: int = 0) -> None:
        instance = checked_u64(spacecraft_instance_id, "spacecraft_instance_id")
        self.writer.mutate(
            "mark_catalog_stale",
            lambda connection: connection.execute(
                "UPDATE catalog_snapshots SET state='RETIRED',is_active=0,retired_at_us=? WHERE source_spacecraft_instance_id=? AND is_active=1",
                (at_us, encode_sqlite_u64(instance)),
            ),
            priority=MutationPriority.HIGH,
        )

    def status(self, spacecraft_instance_id: int) -> CatalogStatus:
        instance = checked_u64(spacecraft_instance_id, "spacecraft_instance_id")
        with self.writer.reader() as connection:
            return self._status_in_transaction(connection, instance)

    @staticmethod
    def _status_in_transaction(connection, instance: int) -> CatalogStatus:
        row = connection.execute(
            "SELECT c.catalog_epoch,c.catalog_revision,c.snapshot_sha256,c.state,c.source_boot_id,c.source_link_session_id,COUNT(s.scene_id) "
            "FROM catalog_snapshots c LEFT JOIN scenes s ON s.source_spacecraft_instance_id=c.source_spacecraft_instance_id AND s.catalog_epoch=c.catalog_epoch AND s.catalog_revision=c.catalog_revision "
            "WHERE c.source_spacecraft_instance_id=? AND c.is_active=1 GROUP BY c.catalog_epoch,c.catalog_revision",
            (encode_sqlite_u64(instance),),
        ).fetchone()
        if row is None:
            return CatalogStatus(instance, None, None, None, False, True, 0, None, None)
        return CatalogStatus(
            instance,
            int(row[0]),
            int(row[1]),
            bytes(row[2]).hex(),
            str(row[3]) == "VERIFIED",
            str(row[3]) != "VERIFIED",
            int(row[6]),
            None if row[4] is None else int(row[4]),
            None if row[5] is None else int.from_bytes(bytes(row[5]), "big"),
        )

    def list_scenes(self, spacecraft_instance_id: int, *, limit: int = 100, after_scene_id: int | None = None) -> tuple[tuple[CatalogScene, ...], int | None, CatalogStatus]:
        instance = checked_u64(spacecraft_instance_id, "spacecraft_instance_id")
        if not 1 <= limit <= 1_000:
            raise ValueError("limit must be in [1, 1000]")
        status = self.status(instance)
        if status.catalog_epoch is None or status.catalog_revision is None:
            return (), None, status
        with self.writer.reader() as connection:
            params: list[Any] = [encode_sqlite_u64(instance), status.catalog_epoch, status.catalog_revision]
            where = "s.source_spacecraft_instance_id=? AND s.catalog_epoch=? AND s.catalog_revision=?"
            if after_scene_id is not None:
                checked_u32(after_scene_id, "after_scene_id")
                where += " AND s.scene_id>?"
                params.append(after_scene_id)
            rows = connection.execute(
                "SELECT s.scene_id,s.scene_revision,s.source_sha256,s.sidecar_sha256,s.metadata_json,s.state FROM scenes s WHERE "
                + where + " ORDER BY s.scene_id LIMIT ?",
                (*params, limit),
            ).fetchall()
        scenes = []
        for row in rows:
            metadata = json.loads(str(row[4]))
            shape = tuple(int(item) for item in metadata.get("shape", ()))
            domain = dict(metadata.pop("domain", {}))
            capability = "VERIFIED" if str(row[5]) == "ACTIVE" else str(row[5])
            scenes.append(CatalogScene(SceneRef(status.catalog_epoch, int(row[0]), int(row[1])), bytes(row[2]).hex(), bytes(row[3]).hex(), shape, capability, domain, metadata))
        next_cursor = scenes[-1].scene_ref.scene_id if len(scenes) == limit else None
        return tuple(scenes), next_cursor, status

    def get_scene(self, scoped_ref: ScopedSceneRef) -> CatalogScene:
        instance = checked_u64(scoped_ref.spacecraft_instance_id, "spacecraft_instance_id")
        ref = scoped_ref.scene_ref
        with self.writer.reader() as connection:
            row = connection.execute(
                "SELECT s.scene_revision,s.source_sha256,s.sidecar_sha256,s.metadata_json,s.state FROM scenes s JOIN catalog_snapshots c ON c.source_spacecraft_instance_id=s.source_spacecraft_instance_id AND c.catalog_epoch=s.catalog_epoch AND c.catalog_revision=s.catalog_revision WHERE s.source_spacecraft_instance_id=? AND s.catalog_epoch=? AND s.scene_id=? AND s.scene_revision=? AND c.is_active=1",
                (encode_sqlite_u64(instance), ref.catalog_epoch, ref.scene_id, ref.scene_revision),
            ).fetchone()
        if row is None:
            status = self.status(instance)
            if status.catalog_epoch != ref.catalog_epoch:
                raise CatalogError("CATALOG_EPOCH_MISMATCH", "scene references an inactive catalog epoch")
            raise CatalogError("SCENE_NOT_FOUND", "scene is not in the active catalog")
        metadata = json.loads(str(row[3]))
        shape = tuple(int(item) for item in metadata.pop("shape", ()))
        domain = dict(metadata.pop("domain", {}))
        capability = "VERIFIED" if str(row[4]) == "ACTIVE" else str(row[4])
        return CatalogScene(ref, bytes(row[1]).hex(), bytes(row[2]).hex(), shape, capability, domain, metadata)
