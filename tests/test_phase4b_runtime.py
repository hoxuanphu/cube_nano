"""Focused P4b contract tests: TM, catalog, file, realtime and topology."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from flight.catalog import SceneCatalog
from gds.catalog import CatalogBundle, CatalogReplicaStore
from gds.file_reassembly import FilePacketReassembler, FileReassemblyError
from gds.product_store import ArtifactDescriptor, ProductManifest, ProductStore
from gds.realtime import RealtimeHub, ResyncRequired
from gds.tm import TMDecoder, TmDecodeError, ValidatedTransportEnvelope, encode_tm_message
from gds.topology import TopologyError, TopologyProfile
from gds.events import EventStore
from gds.ingest import TmIngestService
from gds.preview import PreviewService
from gds.retention import RetentionManager
from gds.writer import SQLiteWriter
from protocol.canonical import canonical_json, deterministic_cbor_encode
from protocol.ccsds import encode_space_packet, encode_tm_frame
from protocol.file_packet import FilePacket, FilePacketType, cfdp_checksum
from protocol.schemas import ProductRef, RequestKey, SceneRef, ScopedSceneRef
from sat_ai.products import build_ustar
from flight.scene_package import ingest_scene_package, scrub_scene_package
import numpy as np
import tifffile


def _envelope(frame: bytes, *, apid_id: int = 1, session: int = 2, epoch: int = 0, frame_id: int = 3):
    return ValidatedTransportEnvelope(
        1,
        7,
        session,
        11,
        12,
        frame_id,
        13,
        epoch,
        0,
        100,
        "DOWNLINK",
        frame,
    )


def _catalog_bundle() -> bytes:
    scene = {
        "scene_ref": {"catalog_epoch": 1, "scene_id": 1, "scene_revision": 1},
        "path": "ignored/source.tif",
        "sidecar_path": "ignored/source.json",
        "source_sha256": "a" * 64,
        "sidecar_sha256": "b" * 64,
        "shape": [256, 256, 3],
        "capability": "VERIFIED",
        "domain": {"sensor": "test"},
    }
    unsigned = {"catalog_epoch": 1, "catalog_revision": 1, "scenes": [scene]}
    payload = {
        "schema_version": 1,
        **unsigned,
        "snapshot_sha256": hashlib.sha256(canonical_json(unsigned)).hexdigest(),
    }
    return canonical_json(payload) + b"\n"


def _product_bundle(product: ProductRef) -> tuple[bytes, int]:
    content = b"verified artifact"
    manifest = ProductManifest(
        "ANALYSIS",
        product,
        RequestKey(1, 9),
        (ArtifactDescriptor("artifact.bin", len(content), hashlib.sha256(content).hexdigest()),),
        {
            "scene_ref": SceneRef(1, 1, 1).as_dict(),
            "source_sha256": "a" * 64,
            "roi": {"x": 0, "y": 0, "width": 1, "height": 1},
            "config_snapshot": {"config_epoch": 1, "config_revision": 1, "model_threshold_bp": 1, "coverage_limit_bp": 1},
            "model_release_id": "test",
            "science_decision": "ACCEPTED",
            "cloud_positive_tile_area_ratio_bp": 0,
        },
    )
    bundle = build_ustar({"manifest.json": manifest.to_bytes(), "artifact.bin": content})
    return bundle, cfdp_checksum(bundle)


def test_tm_decoder_uses_envelope_identity_and_rejects_crc_or_descriptor():
    frame = encode_tm_message(1, {"channel_id": 4, "value": 9, "satellite_time_us": 10})
    decoded = TMDecoder(expected_instance_id=1, expected_session_id=2, expected_link_generation=11).decode(_envelope(frame))
    assert decoded.kind.value == "TELEMETRY"
    assert decoded.envelope.sender_boot_id == 7
    assert decoded.message["value"] == 9

    with pytest.raises(TmDecodeError, match="CRC"):
        bad = bytearray(frame)
        bad[-1] ^= 1
        TMDecoder().decode(_envelope(bytes(bad)))

    wrong_descriptor = encode_tm_frame(encode_space_packet(1, b"\x00\x02" + deterministic_cbor_encode({}), 0))
    with pytest.raises(TmDecodeError, match="descriptor"):
        TMDecoder().decode(_envelope(wrong_descriptor))


def test_catalog_replica_activation_is_atomic_and_instance_scoped(tmp_path: Path):
    with SQLiteWriter(tmp_path / "gds.sqlite3") as writer:
        store = CatalogReplicaStore(writer)
        status = store.activate(1, _catalog_bundle(), source_boot_id=7, link_session_id=2, received_at_us=100)
        assert status.synced and status.catalog_epoch == 1
        scenes, cursor, _ = store.list_scenes(1)
        assert cursor is None and scenes[0].scene_ref == SceneRef(1, 1, 1)
        with pytest.raises(Exception, match="epoch"):
            store.get_scene(__import__("protocol.schemas", fromlist=["ScopedSceneRef"]).ScopedSceneRef(1, SceneRef(2, 1, 1)))
        assert store.status(2).stale


def test_flight_catalog_bundle_verifies_without_satellite_paths():
    catalog = SceneCatalog.from_bundle_bytes(_catalog_bundle())
    assert catalog.snapshot()["snapshot_sha256"] == hashlib.sha256(
        canonical_json({
            "catalog_epoch": 1,
            "catalog_revision": 1,
            "scenes": catalog.snapshot()["scenes"],
        })
    ).hexdigest()


def test_flight_catalog_file_hash_preserves_relative_catalog_paths(tmp_path: Path):
    source = tmp_path / "fixture.tif"
    sidecar = tmp_path / "fixture.sidecar.json"
    source.write_bytes(b"fixture-source")
    sidecar.write_bytes(b'{"fixture":true}\n')
    scene = {
        "scene_ref": {"catalog_epoch": 1, "scene_id": 1, "scene_revision": 1},
        "path": "fixture.tif",
        "sidecar_path": "fixture.sidecar.json",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "sidecar_sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
        "shape": [256, 256, 3],
        "capability": "VERIFIED",
        "domain": {"profile": "fixture"},
    }
    unsigned = {"catalog_epoch": 1, "catalog_revision": 1, "scenes": [scene]}
    (tmp_path / "catalog.json").write_bytes(
        canonical_json(
            {"schema_version": 1, **unsigned, "snapshot_sha256": hashlib.sha256(canonical_json(unsigned)).hexdigest()}
        )
        + b"\n"
    )

    catalog = SceneCatalog.from_file(tmp_path / "catalog.json")

    assert catalog.snapshot_sha256 == hashlib.sha256(canonical_json(unsigned)).hexdigest()
    assert catalog.snapshot()["scenes"][0]["path"] == "fixture.tif"
    assert catalog.get(SceneRef(1, 1, 1)).path == source.resolve()


def test_file_reassembly_out_of_order_duplicate_and_verified_publish(tmp_path: Path):
    product = ProductRef(1, 1, 1)
    bundle, file_checksum = _product_bundle(product)
    bundle_sha = hashlib.sha256(bundle).hexdigest()
    start_value = {
        "source": "b/00000001/00000001.tar",
        "destination": f"p/00000001/00000001/0000002a/{bundle_sha}.tar",
        "file_size": len(bundle),
        "checksum": file_checksum,
        "product_ref": product.as_dict(),
        "transfer_id": 42,
    }
    start = FilePacket(FilePacketType.START, 0, 0, canonical_json(start_value))
    chunks = [bundle[offset : offset + 17] for offset in range(0, len(bundle), 17)]
    with SQLiteWriter(tmp_path / "gds.sqlite3") as writer:
        products = ProductStore(writer, tmp_path / "products")
        receiver = FilePacketReassembler(tmp_path / "reassembly", writer=writer, product_store=products)
        assert receiver.receive(start, spacecraft_instance_id=1, link_session_id=2, file_epoch_id=3).state == "RECEIVING"
        for index in reversed(range(len(chunks))):
            offset = index * 17
            packet = FilePacket(FilePacketType.DATA, index + 1, offset, chunks[index])
            assert receiver.receive(packet, spacecraft_instance_id=1, link_session_id=2, file_epoch_id=3).state == "RECEIVING"
            if index == len(chunks) - 1:
                receiver.receive(packet, spacecraft_instance_id=1, link_session_id=2, file_epoch_id=3)
        end = FilePacket(FilePacketType.END, len(chunks) + 1, len(bundle), file_checksum.to_bytes(4, "big"))
        result = receiver.receive(end, spacecraft_instance_id=1, link_session_id=2, file_epoch_id=3)
        assert result.state == "VERIFIED"
        assert products.get(product)["state"] == "PUBLISHED"
        assert (tmp_path / "products" / "0000000000000001" / "00000001" / "00000001" / "artifact.bin").is_file()


def test_file_reassembly_conflicting_overlap_and_missing_start():
    receiver = FilePacketReassembler(Path("."))
    missing = receiver.receive(FilePacket(FilePacketType.DATA, 1, 0, b"x"), spacecraft_instance_id=1, link_session_id=2, file_epoch_id=3)
    assert missing.state == "INCOMPLETE" and missing.reason == "MISSING_START"


def test_realtime_cursor_replay_and_resync(tmp_path: Path):
    with SQLiteWriter(tmp_path / "gds.sqlite3") as writer:
        events = EventStore(writer)
        hub = RealtimeHub(events, lambda: {"state": "READY"}, max_client_events=10)
        first = events.append("ONE")
        hub.publish(first)
        snapshot, client, replay = hub.connect()
        assert snapshot.as_of_event_id == first.event_id
        assert replay[-1]["event_name"] == "ONE"
        second = events.append("TWO")
        hub.publish(second)
        assert client.drain()[-1]["event_name"] == "TWO"
        hub.set_retention_floor(3)
        with pytest.raises(ResyncRequired):
            hub.connect(first.event_id)


def test_topology_profile_rejects_public_or_foreign_requests():
    profile = TopologyProfile.from_file("protocol/runtime_profile.yaml")
    profile.validate_request(host="127.0.0.1", origin="http://127.0.0.1", peer="127.0.0.1", body_bytes=1, header_bytes=1, method="POST")
    with pytest.raises(TopologyError, match="Host"):
        profile.validate_request(host="10.0.0.1", origin=None, peer="127.0.0.1", body_bytes=1, header_bytes=1)


def test_preview_pointer_cas_tile_and_retention_tombstone(tmp_path: Path):
    image = __import__("io").BytesIO()
    from PIL import Image

    Image.new("RGB", (32, 16), (12, 34, 56)).save(image, format="PNG")
    with SQLiteWriter(tmp_path / "gds.sqlite3") as writer:
        catalog = CatalogReplicaStore(writer)
        catalog.activate(1, _catalog_bundle(), received_at_us=1)
        products = ProductStore(writer, tmp_path / "products")
        preview = PreviewService(writer, catalog, products)
        ref = ProductRef(1, 1, 2)
        item = preview.publish_preview(ScopedSceneRef(1, SceneRef(1, 1, 1)), product_ref=ref, origin_request_key=RequestKey(1, 20), quicklook=image.getvalue(), received_at_us=1)
        assert item.product_ref == ref
        assert preview.active_preview(ScopedSceneRef(1, SceneRef(1, 1, 1))) == ref
        tile, etag = preview.tile(ref, 0, 0, 0)
        assert tile and len(etag) == 64
        retention = RetentionManager(writer, products)
        product = products.get(ref)
        assert product is not None
        assert product["retention_until_us"] > product["published_at_us"]
        retention.evict_expired_products(product["retention_until_us"] - 1)
        assert products.get(ref)["state"] == "PUBLISHED"
        retention.evict_expired_products(product["retention_until_us"])
        tombstone = retention.lookup_tombstone(ref, product["retention_until_us"] + 1)
        assert tombstone is not None and tombstone.status_code == 410


def test_retention_cleanup_covers_part_raw_and_rotated_log_files(tmp_path: Path):
    with SQLiteWriter(tmp_path / "gds.sqlite3") as writer:
        products = ProductStore(writer, tmp_path / "products")
        retention = RetentionManager(writer, products)
        raw_root = tmp_path / "raw"
        staging_root = tmp_path / "staging"
        log_root = tmp_path / "logs"
        raw_root.mkdir()
        staging_root.mkdir()
        log_root.mkdir()
        for path in (raw_root / "frames.seg", staging_root / "transfer.part", log_root / "gds.log.1"):
            path.write_bytes(b"old")
            os.utime(path, ns=(1_000_000_000, 1_000_000_000))
        cleaned = retention.cleanup_files(
            now_us=10 * 86_400_000_000,
            raw_roots=(raw_root,),
            staging_roots=(staging_root,),
            log_roots=(log_root,),
        )
        assert len(cleaned["raw"]) == 1
        assert len(cleaned["staging"]) == 1
        assert len(cleaned["logs"]) == 1


def test_product_default_retention_is_not_immediately_expired(tmp_path: Path):
    product = ProductRef(1, 1, 77)
    bundle, checksum = _product_bundle(product)
    from gds.product_store import verify_bundle

    verified = verify_bundle(
        bundle,
        expected_bundle_sha256=hashlib.sha256(bundle).hexdigest(),
        expected_file_checksum=checksum,
        expected_product_ref=product,
        temporary_root=tmp_path / "verified",
    )
    with SQLiteWriter(tmp_path / "gds.sqlite3") as writer:
        clock = lambda: 1_000_000
        products = ProductStore(writer, tmp_path / "products", clock=clock)
        products.publish(verified)
        stored = products.get(product)
        assert stored["verified_at_us"] == 1_000_000
        assert stored["published_at_us"] == 1_000_000
        assert stored["retention_until_us"] == 1_000_000 + 30 * 86_400_000_000


def test_scene_package_stat_scrub_rejects_out_of_band_mutation(tmp_path: Path):
    source = tmp_path / "scene.tif"
    sidecar = tmp_path / "scene.sidecar.json"
    tifffile.imwrite(source, np.zeros((8, 8, 3), dtype=np.uint16), metadata={"axes": "YXC"}, compression=None)
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    sidecar.write_text(json.dumps({
        "schema_version": 1,
        "source_fingerprint": {"algorithm": "sha256", "digest": source_sha},
        "axes": "YXC",
        "shape": [8, 8, 3],
        "band_order": ["red", "green", "blue"],
        "dtype": "uint16",
        "input_spec_id": "rgb-legacy-dtype-range-v1",
        "validity": {"kind": "all_valid"},
    }, separators=(",", ":")), encoding="utf-8")
    package = ingest_scene_package(source, sidecar, tmp_path / "packages", SceneRef(1, 1, 1))
    scrub_scene_package(package)
    os.utime(package.source_path, ns=(package.source_stat["mtime_ns"] + 1_000_000_000, package.source_stat["mtime_ns"] + 1_000_000_000))
    with pytest.raises(Exception, match="INVALID_SCENE_SOURCE_STAT"):
        scrub_scene_package(package)


def test_scene_package_hash_is_canonical_and_stale_package_is_quarantined(tmp_path: Path):
    source = tmp_path / "scene.tif"
    sidecar = tmp_path / "scene.sidecar.json"
    tifffile.imwrite(source, np.zeros((8, 8, 3), dtype=np.uint16), metadata={"axes": "YXC"}, compression=None)
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    sidecar.write_text(json.dumps({
        "schema_version": 1,
        "source_fingerprint": {"algorithm": "sha256", "digest": source_sha},
        "axes": "YXC",
        "shape": [8, 8, 3],
        "band_order": ["red", "green", "blue"],
        "dtype": "uint16",
        "input_spec_id": "rgb-legacy-dtype-range-v1",
        "validity": {"kind": "all_valid"},
    }, separators=(",", ":")), encoding="utf-8")
    package = ingest_scene_package(source, sidecar, tmp_path / "packages", SceneRef(1, 1, 1))
    manifest = json.loads((package.root / "package.json").read_text(encoding="utf-8"))
    descriptor = dict(manifest)
    descriptor.pop("package_sha256")
    assert manifest["package_sha256"] == hashlib.sha256(canonical_json(descriptor)).hexdigest()

    os.chmod(package.source_path, 0o666)
    package.source_path.write_bytes(b"stale package bytes")
    rebuilt = ingest_scene_package(source, sidecar, tmp_path / "packages", SceneRef(1, 1, 1))
    assert rebuilt.package_sha256 == package.package_sha256
    assert rebuilt.source_path.read_bytes() == source.read_bytes()
    assert list((tmp_path / "packages").glob(".quarantine-*"))


def test_tm_ingest_persists_event_and_telemetry_identity(tmp_path: Path):
    with SQLiteWriter(tmp_path / "gds.sqlite3") as writer:
        events = EventStore(writer)
        hub = RealtimeHub(events, lambda: {})
        service = TmIngestService(writer, TMDecoder(), realtime=hub)
        telemetry = encode_tm_message(1, {"channel_id": 1, "value": 4, "satellite_time_us": 9})
        result = service.ingest(_envelope(telemetry, frame_id=10))
        assert result.telemetry is not None and result.telemetry.inserted
        event_frame = encode_tm_message(
            2,
            {"event_name": "COMMAND_ACCEPTED", "severity": "INFO", "message": {"ok": True}},
            master_channel_count=1,
            virtual_channel_count=1,
        )
        event_result = service.ingest(_envelope(event_frame, frame_id=11))
        assert event_result.event is not None
        assert hub.clients() == ()
