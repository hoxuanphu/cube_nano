"""Preview quicklook products and ProductRef compare-and-swap pointers."""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from protocol.canonical import canonical_json, checked_u32
from protocol.file_packet import cfdp_checksum
from protocol.schemas import ProductRef, RequestKey, SceneRef, ScopedSceneRef
from sat_ai.products import build_ustar

from .catalog import CatalogError, CatalogReplicaStore
from .product_store import ProductManifest, ArtifactDescriptor, ProductStore, verify_bundle
from .u64 import encode_sqlite_u64
from .writer import MutationPriority, SQLiteWriter


class PreviewError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class PreviewProduct:
    product_ref: ProductRef
    scene_ref: SceneRef
    quicklook_sha256: str
    bundle_sha256: str
    catalog_revision: int
    etag: str


def _encode_image(image_bytes: bytes, *, format_name: str = "WEBP") -> bytes:
    try:
        image = Image.open(io.BytesIO(bytes(image_bytes)))
        image.load()
    except (OSError, ValueError) as exc:
        raise PreviewError("INVALID_PREVIEW_IMAGE", "quicklook is not a readable image") from exc
    output = io.BytesIO()
    image.save(output, format=format_name, lossless=True if format_name == "WEBP" else False)
    return output.getvalue()


def build_preview_bundle(
    quicklook: bytes,
    *,
    product_ref: ProductRef,
    origin_request_key: RequestKey,
    scene_ref: SceneRef,
    source_sha256: str,
    display_profile: dict[str, Any] | None = None,
) -> bytes:
    quicklook = _encode_image(quicklook)
    artifact = ArtifactDescriptor("quicklook.webp", len(quicklook), hashlib.sha256(quicklook).hexdigest())
    manifest = ProductManifest(
        "PREVIEW",
        product_ref,
        origin_request_key,
        (artifact,),
        {
            "scene_ref": scene_ref.as_dict(),
            "source_sha256": source_sha256,
            "display_profile": display_profile or {"id": "server-preview-v1"},
        },
    )
    return build_ustar({"manifest.json": manifest.to_bytes(), "quicklook.webp": quicklook})


class PreviewService:
    def __init__(self, writer: SQLiteWriter, catalog: CatalogReplicaStore, product_store: ProductStore):
        self.writer = writer
        self.catalog = catalog
        self.product_store = product_store

    def publish_preview(
        self,
        scoped_scene: ScopedSceneRef,
        *,
        product_ref: ProductRef,
        origin_request_key: RequestKey,
        quicklook: bytes,
        display_profile: dict[str, Any] | None = None,
        received_at_us: int = 0,
        expected_preview_generation: int | None = None,
    ) -> PreviewProduct:
        if product_ref.spacecraft_instance_id != scoped_scene.spacecraft_instance_id:
            raise PreviewError("PRODUCT_TARGET_INSTANCE_MISMATCH", "preview ProductRef is outside the scene instance")
        scene = self.catalog.get_scene(scoped_scene)
        bundle = build_preview_bundle(quicklook, product_ref=product_ref, origin_request_key=origin_request_key, scene_ref=scene.scene_ref, source_sha256=scene.source_sha256, display_profile=display_profile)
        bundle_sha = hashlib.sha256(bundle).hexdigest()
        verified = verify_bundle(bundle, expected_bundle_sha256=bundle_sha, expected_file_checksum=cfdp_checksum(bundle), expected_product_ref=product_ref)
        self.product_store.publish(verified)
        self._activate_pointer(scoped_scene, product_ref, received_at_us, expected_preview_generation)
        return PreviewProduct(product_ref, scene.scene_ref, hashlib.sha256(_extract_quicklook(verified.extracted_root)).hexdigest(), bundle_sha, self.catalog.status(scoped_scene.spacecraft_instance_id).catalog_revision or 0, bundle_sha)

    def _activate_pointer(self, scoped_scene: ScopedSceneRef, product_ref: ProductRef, at_us: int, expected_generation: int | None) -> None:
        ref = scoped_scene.scene_ref
        params: list[Any] = [
            encode_sqlite_u64(product_ref.spacecraft_instance_id), product_ref.origin_boot_id, product_ref.product_id,
            encode_sqlite_u64(scoped_scene.spacecraft_instance_id), ref.catalog_epoch, ref.scene_id, ref.scene_revision,
        ]
        predicate = ""
        if expected_generation is not None:
            checked_u32(expected_generation, "expected_preview_generation")
            predicate = " AND active_preview_generation=?"
            params.append(expected_generation)
        def mutation(connection):
            row = connection.execute("SELECT state FROM scenes WHERE source_spacecraft_instance_id=? AND catalog_epoch=? AND scene_id=? AND scene_revision=?", (encode_sqlite_u64(scoped_scene.spacecraft_instance_id), ref.catalog_epoch, ref.scene_id, ref.scene_revision)).fetchone()
            if row is None or str(row[0]) != "ACTIVE":
                raise PreviewError("SCENE_NOT_VERIFIED", "preview pointer target is not a verified scene")
            cursor = connection.execute(
                "UPDATE scenes SET active_preview_spacecraft_instance_id=?,active_preview_origin_boot_id=?,active_preview_product_id=?,active_preview_generation=active_preview_generation+1 WHERE source_spacecraft_instance_id=? AND catalog_epoch=? AND scene_id=? AND scene_revision=? AND state='ACTIVE'" + predicate,
                tuple(params),
            )
            if cursor.rowcount != 1:
                raise PreviewError("PREVIEW_POINTER_CONFLICT", "active preview CAS failed")
        self.writer.mutate("activate_preview_product", mutation, priority=MutationPriority.HIGH)

    def active_preview(self, scoped_scene: ScopedSceneRef) -> ProductRef | None:
        with self.writer.reader() as connection:
            row = connection.execute("SELECT active_preview_spacecraft_instance_id,active_preview_origin_boot_id,active_preview_product_id FROM scenes WHERE source_spacecraft_instance_id=? AND catalog_epoch=? AND scene_id=? AND scene_revision=?", (encode_sqlite_u64(scoped_scene.spacecraft_instance_id), scoped_scene.scene_ref.catalog_epoch, scoped_scene.scene_ref.scene_id, scoped_scene.scene_ref.scene_revision)).fetchone()
        if row is None or row[0] is None:
            return None
        return ProductRef(int.from_bytes(bytes(row[0]), "big"), int(row[1]), int(row[2]))

    def tile(self, product_ref: ProductRef, z: int, x: int, y: int, *, tile_size: int = 256) -> tuple[bytes, str]:
        if not 0 <= z <= 12 or x < 0 or y < 0:
            raise PreviewError("INVALID_TILE", "tile coordinates are outside the supported range")
        if tile_size != 256:
            raise PreviewError("INVALID_TILE", "MVP tile size is fixed at 256")
        product = self.product_store.get(product_ref)
        if product is None or product["state"] != "PUBLISHED" or not product.get("local_path"):
            raise PreviewError("PRODUCT_NOT_AVAILABLE", "preview product is not available")
        path = Path(str(product["local_path"])) / "quicklook.webp"
        if not path.is_file() or self.product_store.root not in path.resolve().parents:
            raise PreviewError("PRODUCT_NOT_AVAILABLE", "quicklook artifact is missing")
        with Image.open(path) as image:
            image = image.convert("RGB")
            scale = 1 << z
            width = tile_size * scale
            height = tile_size * scale
            image.thumbnail((width, height), Image.Resampling.LANCZOS)
            tile = Image.new("RGB", (width, height))
            tile.paste(image, (0, 0))
            left, top = x * tile_size, y * tile_size
            if left >= width or top >= height:
                raise PreviewError("TILE_NOT_FOUND", "tile is outside the quicklook pyramid")
            tile = tile.crop((left, top, min(left + tile_size, width), min(top + tile_size, height)))
            if tile.size != (tile_size, tile_size):
                padded = Image.new("RGB", (tile_size, tile_size))
                padded.paste(tile, (0, 0))
                tile = padded
            output = io.BytesIO()
            tile.save(output, format="WEBP", lossless=True, method=6)
        content = output.getvalue()
        return content, hashlib.sha256(content).hexdigest()


def _extract_quicklook(root: Path) -> bytes:
    path = root / "quicklook.webp"
    if not path.is_file():
        raise PreviewError("PREVIEW_ARTIFACT_MISSING", "verified preview has no quicklook")
    return path.read_bytes()
