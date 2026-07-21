"""Phase 6 recovery evidence for FilePacket loss, retry, cancel and restart."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from gds.file_reassembly import FilePacketReassembler
from gds.product_store import ArtifactDescriptor, ProductManifest, ProductStore
from gds.writer import SQLiteWriter
from protocol.canonical import canonical_json
from protocol.file_packet import FilePacket, FilePacketType, cfdp_checksum
from protocol.schemas import ProductRef, RequestKey, SceneRef
from sat_ai.products import build_ustar


def _bundle(product: ProductRef) -> tuple[bytes, int]:
    content = b"phase6-recovery-artifact"
    manifest = ProductManifest(
        "ANALYSIS",
        product,
        RequestKey(7, 1),
        (ArtifactDescriptor("artifact.bin", len(content), hashlib.sha256(content).hexdigest()),),
        {
            "scene_ref": SceneRef(1, 1, 1).as_dict(),
            "source_sha256": "a" * 64,
            "roi": {"x": 0, "y": 0, "width": 1, "height": 1},
            "config_snapshot": {"config_epoch": 0, "config_revision": 0, "model_threshold_bp": 5000, "coverage_limit_bp": 6000},
            "model_release_id": "phase6-test",
            "science_decision": "ACCEPTED",
            "cloud_positive_tile_area_ratio_bp": 0,
        },
    )
    value = build_ustar({"manifest.json": manifest.to_bytes(), "artifact.bin": content})
    return value, cfdp_checksum(value)


def _start(product: ProductRef, bundle: bytes, checksum: int, transfer_id: int) -> FilePacket:
    bundle_sha = hashlib.sha256(bundle).hexdigest()
    payload = {
        "source": f"b/{product.origin_boot_id:08x}/{product.product_id:08x}.tar",
        "destination": f"p/{product.origin_boot_id:08x}/{product.product_id:08x}/{transfer_id:08x}/{bundle_sha}.tar",
        "file_size": len(bundle),
        "checksum": checksum,
        "product_ref": product.as_dict(),
        "transfer_id": transfer_id,
    }
    return FilePacket(FilePacketType.START, 0, 0, canonical_json(payload))


def test_file_loss_cancel_retry_and_out_of_order_publish(tmp_path: Path):
    product = ProductRef(1, 1, 9)
    bundle, checksum = _bundle(product)
    chunks = [bundle[offset : offset + 23] for offset in range(0, len(bundle), 23)]
    with SQLiteWriter(tmp_path / "gds.sqlite3") as writer:
        products = ProductStore(writer, tmp_path / "products")
        receiver = FilePacketReassembler(tmp_path / "reassembly", writer=writer, product_store=products)
        assert receiver.receive(_start(product, bundle, checksum, 1), spacecraft_instance_id=1, link_session_id=3, file_epoch_id=1, origin_request_key=RequestKey(7, 1)).state == "RECEIVING"
        receiver.receive(FilePacket(FilePacketType.DATA, 1, 0, chunks[0]), spacecraft_instance_id=1, link_session_id=3, file_epoch_id=1)
        canceled = receiver.receive(FilePacket(FilePacketType.CANCEL, 2, 0, b"retry"), spacecraft_instance_id=1, link_session_id=3, file_epoch_id=1)
        assert canceled.state == "CANCELED"

        assert receiver.receive(_start(product, bundle, checksum, 2), spacecraft_instance_id=1, link_session_id=3, file_epoch_id=2).state == "RECEIVING"
        for index in reversed(range(len(chunks))):
            receiver.receive(FilePacket(FilePacketType.DATA, index + 1, index * 23, chunks[index]), spacecraft_instance_id=1, link_session_id=3, file_epoch_id=2)
        end = receiver.receive(FilePacket(FilePacketType.END, len(chunks) + 1, len(bundle), checksum.to_bytes(4, "big")), spacecraft_instance_id=1, link_session_id=3, file_epoch_id=2)
        assert end.state == "VERIFIED"
        assert products.get(product)["state"] == "PUBLISHED"


def test_file_reassembly_recovers_durable_receiving_state_after_restart(tmp_path: Path):
    product = ProductRef(1, 2, 3)
    bundle, checksum = _bundle(product)
    chunks = [bundle[offset : offset + 31] for offset in range(0, len(bundle), 31)]
    with SQLiteWriter(tmp_path / "gds.sqlite3") as writer:
        products = ProductStore(writer, tmp_path / "products")
        first = FilePacketReassembler(tmp_path / "reassembly", writer=writer, product_store=products)
        assert first.receive(_start(product, bundle, checksum, 3), spacecraft_instance_id=1, link_session_id=4, file_epoch_id=8).state == "RECEIVING"
        first.receive(FilePacket(FilePacketType.DATA, 1, 0, chunks[0]), spacecraft_instance_id=1, link_session_id=4, file_epoch_id=8)

        restarted = FilePacketReassembler(tmp_path / "reassembly", writer=writer, product_store=products)
        for index in range(1, len(chunks)):
            restarted.receive(FilePacket(FilePacketType.DATA, index + 1, index * 31, chunks[index]), spacecraft_instance_id=1, link_session_id=4, file_epoch_id=8)
        result = restarted.receive(FilePacket(FilePacketType.END, len(chunks) + 1, len(bundle), checksum.to_bytes(4, "big")), spacecraft_instance_id=1, link_session_id=4, file_epoch_id=8)
        assert result.state == "VERIFIED"
        assert products.get(product)["state"] == "PUBLISHED"
