"""Regression coverage for follow-up realtime and bounded-streaming fixes."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import tifffile
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from flight.file_downlink import FileDownlinkCoordinator
from gds.events import EventRecord
from gds.file_reassembly import FilePacketReassembler
from gds.http_app import create_app
from gds.product_store import (
    ArtifactDescriptor,
    ProductManifest,
    ProductStore,
    stream_file_integrity,
    verify_bundle,
)
from gds.realtime import RESYNC_CLOSE_CODE, RealtimeHub, ResyncRequired
from gds.topology import TopologyProfile
from gds.writer import SQLiteWriter
from protocol.canonical import canonical_json
from protocol.file_packet import MAX_FILE_DATA_PER_FRAME, FilePacket, FilePacketType
from protocol.schemas import ProductRef, RequestKey, SceneRef
from sat_ai.products import write_ustar
from sat_ai.roi import SceneContractError, open_memmap_scene


ROOT = Path(__file__).resolve().parents[1]


class _EventStore:
    def __init__(self, records: tuple[EventRecord, ...] = ()):
        self.records = records

    def latest_event_id(self) -> int:
        return self.records[-1].event_id if self.records else 0

    def list_events(self, *, after_event_id: int | None = None, limit: int = 100):
        records = tuple(record for record in self.records if after_event_id is None or record.event_id > after_event_id)
        return records[:limit], None


def _event(event_id: int, *, message: object = None) -> EventRecord:
    return EventRecord(event_id, "FOLLOWUP", "INFO", message, datetime(2026, 7, 21, tzinfo=UTC))


def _analysis_manifest(product: ProductRef, artifact: ArtifactDescriptor) -> ProductManifest:
    return ProductManifest(
        "ANALYSIS",
        product,
        RequestKey(7, 1),
        (artifact,),
        {
            "scene_ref": SceneRef(1, 1, 1).as_dict(),
            "source_sha256": "a" * 64,
            "roi": {"x": 0, "y": 0, "width": 1, "height": 1},
            "config_snapshot": {
                "config_epoch": 0,
                "config_revision": 0,
                "model_threshold_bp": 5000,
                "coverage_limit_bp": 6000,
            },
            "model_release_id": "followup-test",
            "science_decision": "ACCEPTED",
            "cloud_positive_tile_area_ratio_bp": 0,
        },
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(64 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_scene_sidecar(source: Path, sidecar: Path, *, validity: dict[str, object] | None = None) -> None:
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_fingerprint": {"algorithm": "sha256", "digest": _sha256(source)},
                "axes": "YXC",
                "shape": [8, 8, 3],
                "band_order": ["red", "green", "blue"],
                "dtype": "uint16",
                "input_spec_id": "rgb-legacy-dtype-range-v1",
                "validity": validity or {"kind": "all_valid"},
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )


def test_realtime_hex_cursor_boundary_stale_ahead_and_slow_client_contract():
    store = _EventStore((_event(0xAE), _event(0xAF)))
    hub = RealtimeHub(store, lambda: {"state": "READY"}, max_client_events=2, max_client_bytes=4096)
    snapshot, client, replay = hub.connect("00000000000000ae")
    assert snapshot.as_of_event_id == 0xAF
    assert [item["event_id"] for item in replay] == ["00000000000000af"]

    # A delayed publisher must not repeat the event already included in replay.
    hub.publish(_event(0xAF))
    assert client.drain() == ()
    live = _event(0xB0)
    hub.publish(live)
    assert [item["event_id"] for item in client.drain()] == ["00000000000000b0"]

    hub.set_retention_floor(0xAF)
    with pytest.raises(ResyncRequired):
        hub.connect("00000000000000ad")
    with pytest.raises(ResyncRequired):
        hub.connect("00000000000000b1")

    slow_hub = RealtimeHub(_EventStore(), lambda: {}, max_client_events=1, max_client_bytes=4096)
    _, slow_client, _ = slow_hub.connect()
    slow_hub.publish(_event(1))
    slow_hub.publish(_event(2))
    assert slow_client.closed and slow_client.close_code == RESYNC_CLOSE_CODE
    assert slow_client.take_terminal_envelope() == {
        "type": "error",
        "error": "RESYNC_REQUIRED",
        "message": "realtime client exceeded its bounded replay buffer",
    }
    assert slow_client.take_terminal_envelope() is None
    assert slow_hub.clients() == ()


def test_realtime_http_sends_resync_envelope_before_documented_close():
    profile = TopologyProfile.from_file(ROOT / "protocol" / "runtime_profile.yaml")

    def client_for(hub: RealtimeHub, events: _EventStore) -> TestClient:
        mission = SimpleNamespace(
            topology=profile,
            gds=SimpleNamespace(realtime=hub, events=events),
            snapshot=lambda: {"runtime": {"browser_gds": "CONNECTED"}},
        )
        return TestClient(
            create_app(ROOT, service=mission),
            base_url="http://127.0.0.1:8000",
            client=("127.0.0.1", 41001),
        )

    retained = _EventStore((_event(1), _event(2)))
    retained_hub = RealtimeHub(retained, lambda: {}, max_client_events=1, max_client_bytes=4096)
    retained_hub.set_retention_floor(2)
    with client_for(retained_hub, retained).websocket_connect(
        "/ws/telemetry?last_event_id=0000000000000000",
        headers={"host": "127.0.0.1", "origin": "http://127.0.0.1:8000"},
    ) as socket:
        assert socket.receive_json()["error"] == "RESYNC_REQUIRED"
        with pytest.raises(WebSocketDisconnect) as closed:
            socket.receive_json()
        assert closed.value.code == RESYNC_CLOSE_CODE

    live = _EventStore()
    slow_hub = RealtimeHub(live, lambda: {}, max_client_events=1, max_client_bytes=4096)
    with client_for(slow_hub, live).websocket_connect(
        "/ws/telemetry",
        headers={"host": "127.0.0.1", "origin": "http://127.0.0.1:8000"},
    ) as socket:
        assert socket.receive_json()["type"] == "snapshot"
        slow_hub.publish(_event(3))
        slow_hub.publish(_event(4))
        assert socket.receive_json()["error"] == "RESYNC_REQUIRED"
        with pytest.raises(WebSocketDisconnect) as closed:
            socket.receive_json()
        assert closed.value.code == RESYNC_CLOSE_CODE


def test_scene_fingerprint_cache_reuses_only_unchanged_verified_source(tmp_path: Path):
    source = tmp_path / "scene.tif"
    sidecar = tmp_path / "scene.json"
    tifffile.imwrite(source, np.zeros((8, 8, 3), dtype=np.uint16), metadata={"axes": "YXC"}, compression=None)
    _write_scene_sidecar(source, sidecar)

    from sat_ai import roi as roi_module

    with mock.patch.object(roi_module, "_sha256_file", wraps=roi_module._sha256_file) as hashed:
        with open_memmap_scene(source, sidecar) as scene:
            assert scene.shape == (8, 8, 3)
        with open_memmap_scene(source, sidecar) as scene:
            assert scene.shape == (8, 8, 3)
        assert hashed.call_count == 1

        tifffile.imwrite(source, np.ones((8, 8, 3), dtype=np.uint16), metadata={"axes": "YXC"}, compression=None)
        changed = source.stat()
        os.utime(source, ns=(changed.st_atime_ns, changed.st_mtime_ns + 1_000_000_000))
        with pytest.raises(SceneContractError, match="fingerprint"):
            open_memmap_scene(source, sidecar)
        assert hashed.call_count == 2


def test_mask_and_unknown_nodata_continue_to_fail_closed(tmp_path: Path):
    source = tmp_path / "scene.tif"
    sidecar = tmp_path / "scene.json"
    mask = tmp_path / "validity.tif"
    tifffile.imwrite(source, np.zeros((8, 8, 3), dtype=np.uint16), metadata={"axes": "YXC"}, compression=None)
    invalid_mask = np.ones((8, 8), dtype=np.uint8)
    invalid_mask[0, 0] = 2
    tifffile.imwrite(mask, invalid_mask, metadata={"axes": "YX"}, compression=None)
    _write_scene_sidecar(source, sidecar, validity={"kind": "mask", "relative_path": mask.name, "sha256": _sha256(mask)})
    with pytest.raises(SceneContractError, match="VALIDITY_MASK_NOT_BINARY"):
        open_memmap_scene(source, sidecar)

    _write_scene_sidecar(source, sidecar, validity={"kind": "nodata_value", "values": [0, 0, 0], "rule": "unknown"})
    with pytest.raises(SceneContractError, match="UNKNOWN_NODATA_RULE"):
        open_memmap_scene(source, sidecar)


def test_streamed_bundle_verify_publish_and_downlink(tmp_path: Path):
    checksum_vector = tmp_path / "checksum-vector.bin"
    checksum_vector.write_bytes(b"\x01\x02\x03\x04\x05")
    assert stream_file_integrity(checksum_vector, buffer_bytes=3).cfdp_checksum == 0x06020304

    product = ProductRef(1, 1, 9)
    artifact = tmp_path / "artifact.bin"
    with artifact.open("wb") as stream:
        for _ in range(48):
            stream.write(bytes(range(256)) * 256)
    descriptor = ArtifactDescriptor(artifact.name, artifact.stat().st_size, _sha256(artifact))
    manifest = _analysis_manifest(product, descriptor)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(manifest.to_bytes())
    bundle_path = tmp_path / "bundle.tar"
    bundle_size, bundle_sha = write_ustar({"manifest.json": manifest_path, artifact.name: artifact}, bundle_path)
    integrity = stream_file_integrity(bundle_path, buffer_bytes=257)
    assert integrity.size == bundle_size
    assert integrity.sha256 == bundle_sha

    verified = verify_bundle(
        bundle_path,
        expected_bundle_sha256=bundle_sha,
        expected_file_checksum=integrity.cfdp_checksum,
        expected_product_ref=product,
        temporary_root=tmp_path / "verified",
    )
    assert verified.bundle_path != bundle_path.resolve()
    assert verified.bundle_path.is_file()
    assert verified.bundle_size == bundle_size
    bundle_path.write_bytes(b"source mutation after verification must not alter the snapshot")
    with SQLiteWriter(tmp_path / "gds.sqlite3") as writer:
        store = ProductStore(writer, tmp_path / "products")
        summary = store.publish(verified)
        final_bundle = Path(summary["product_directory"]) / "bundle.tar"
        assert stream_file_integrity(final_bundle).sha256 == bundle_sha

        coordinator = FileDownlinkCoordinator(cooldown_ticks=1)
        coordinator.start(77, product, final_bundle)
        packets = list(coordinator.packets(77))
        start = json.loads(packets[0].payload)
        assert start["file_size"] == bundle_size and start["checksum"] == integrity.cfdp_checksum
        data = b"".join(packet.payload for packet in packets if packet.packet_type is FilePacketType.DATA)
        assert hashlib.sha256(data).hexdigest() == bundle_sha
        assert max(len(packet.payload) for packet in packets if packet.packet_type is FilePacketType.DATA) <= MAX_FILE_DATA_PER_FRAME


def test_reassembly_verifies_from_part_file_without_whole_file_read(tmp_path: Path):
    product = ProductRef(1, 1, 10)
    artifact = tmp_path / "small.bin"
    artifact.write_bytes(b"bounded-file-reassembly" * 100)
    descriptor = ArtifactDescriptor(artifact.name, artifact.stat().st_size, _sha256(artifact))
    manifest = _analysis_manifest(product, descriptor)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(manifest.to_bytes())
    bundle_path = tmp_path / "small.tar"
    bundle_size, bundle_sha = write_ustar({"manifest.json": manifest_path, artifact.name: artifact}, bundle_path)
    integrity = stream_file_integrity(bundle_path)

    start_payload = {
        "source": "b/00000001/0000000a.tar",
        "destination": f"p/00000001/0000000a/0000004e/{bundle_sha}.tar",
        "file_size": bundle_size,
        "checksum": integrity.cfdp_checksum,
        "product_ref": product.as_dict(),
        "transfer_id": 78,
    }
    with SQLiteWriter(tmp_path / "gds.sqlite3") as writer:
        store = ProductStore(writer, tmp_path / "products")
        receiver = FilePacketReassembler(tmp_path / "reassembly", writer=writer, product_store=store)
        assert receiver.receive(
            FilePacket(FilePacketType.START, 0, 0, canonical_json(start_payload)),
            spacecraft_instance_id=1,
            link_session_id=1,
            file_epoch_id=1,
        ).state == "RECEIVING"
        with bundle_path.open("rb") as stream:
            offset = 0
            sequence = 1
            while chunk := stream.read(MAX_FILE_DATA_PER_FRAME):
                receiver.receive(
                    FilePacket(FilePacketType.DATA, sequence, offset, chunk),
                    spacecraft_instance_id=1,
                    link_session_id=1,
                    file_epoch_id=1,
                )
                offset += len(chunk)
                sequence += 1
        with mock.patch.object(Path, "read_bytes", side_effect=AssertionError("whole-file read is forbidden")):
            result = receiver.receive(
                FilePacket(FilePacketType.END, sequence, bundle_size, integrity.cfdp_checksum.to_bytes(4, "big")),
                spacecraft_instance_id=1,
                link_session_id=1,
                file_epoch_id=1,
            )
        assert result.state == "VERIFIED"
