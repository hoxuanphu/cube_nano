"""Run the P4b scene -> TC -> TM -> FilePacket round trip without web UI."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flight.satellite_simulator import SatelliteSimulator
from gds.local_sil import LocalSilRuntime
from gds.tm import ValidatedTransportEnvelope
from protocol.canonical import canonical_json
from protocol.ccsds import TcTypeBdFrame, SpacePacket, encode_space_packet
from protocol.schemas import Command, CommandOpcode, ProductRef, RequestKey, ROI, SceneRef, encode_command


def _tc(simulator: SatelliteSimulator, command: Command, sequence: int) -> dict:
    packet = encode_space_packet(0, encode_command(command), sequence, packet_type=1)
    frame = TcTypeBdFrame(68, 0, sequence & 0xFF, SpacePacket.decode(packet)).encode()
    return simulator.receive_tc_frame(frame)


def _ack_to_gds(satellite: SatelliteSimulator, gds: LocalSilRuntime, result: dict, *, frame_id: int, session_id: int, run_id: int) -> None:
    frame = satellite.payload.encode_ack_tm_frame(result)
    envelope = ValidatedTransportEnvelope(
        1,
        satellite.payload.journal.boot_id,
        session_id,
        1,
        run_id,
        frame_id,
        frame_id,
        0,
        0,
        int(time.time_ns() // 1_000),
        "DOWNLINK",
        frame,
    )
    gds.ingest_tm(envelope)


def run(root: Path) -> dict:
    with tempfile.TemporaryDirectory(prefix="p4b-satellite-state-") as satellite_state, tempfile.TemporaryDirectory(prefix="p4b-ground-state-") as ground_state:
        satellite = SatelliteSimulator(root, state_directory=satellite_state, device="cpu")
        gds = LocalSilRuntime(root, state_directory=ground_state)
        product_path: Path | None = None
        try:
            instance = satellite.payload.profile.spacecraft_instance_id
            run_id = 0x5044425F524F554E
            session_id = 1
            # The repository already contains development products.  Allocate
            # this fixture's durable IDs from a high, run-local range so the
            # fixture never aliases an existing artifact directory.
            with satellite.payload.journal.transaction() as connection:
                connection.execute("UPDATE meta SET value='1879048192' WHERE key='next_product_id'")
                connection.execute("UPDATE meta SET value='1879048192' WHERE key='next_transfer_id'")
            catalog_key = RequestKey(0x1000000000000001, 1)
            catalog_command = Command(CommandOpcode.SCENE_REQUEST_CATALOG, instance, catalog_key, {})
            catalog_result = _tc(satellite, catalog_command, 1)
            catalog_bundle = canonical_json(catalog_result["catalog"]) + b"\n"
            catalog_status = gds.catalog.activate(instance, catalog_bundle, source_boot_id=satellite.payload.journal.boot_id, link_session_id=session_id, received_at_us=int(time.time_ns() // 1_000))
            _ack_to_gds(satellite, gds, catalog_result, frame_id=1, session_id=session_id, run_id=run_id)

            config = satellite.payload.journal.current_config()
            analysis_key = RequestKey(0x1000000000000001, 2)
            analysis_command = Command(
                CommandOpcode.ROI_REQUEST,
                instance,
                analysis_key,
                {
                    "scene_ref": SceneRef(1, 1, 1).as_dict(),
                    "roi": ROI(0, 0, 256, 256).as_dict(),
                    "expected_config_epoch": config.epoch,
                    "expected_config_revision": config.revision,
                    "model_threshold_bp": config.model_threshold_bp,
                    "coverage_limit_bp": config.coverage_limit_bp,
                },
            )
            analysis_result = _tc(satellite, analysis_command, 2)
            _ack_to_gds(satellite, gds, analysis_result, frame_id=2, session_id=session_id, run_id=run_id)
            satellite.payload.wait_for_jobs(60)
            job = satellite.payload.journal.get_job(analysis_key)
            if job is None or str(job["state"]) != "SUCCEEDED":
                raise RuntimeError(f"analysis did not succeed: {None if job is None else (job['state'], job['error_code'], job['result_json'])}")
            product = ProductRef.from_dict(json.loads(job["product_ref_json"]))
            product_path = root / "data" / "satellite" / "products" / f"{product.origin_boot_id:08x}" / f"{product.product_id:08x}"

            downlink_key = RequestKey(0x1000000000000001, 3)
            downlink_command = Command(
                CommandOpcode.PRODUCT_REQUEST_DOWNLINK,
                instance,
                downlink_key,
                {"origin_request_key": analysis_key.as_dict(), "product_ref": product.as_dict()},
            )
            downlink_result = _tc(satellite, downlink_command, 3)
            _ack_to_gds(satellite, gds, downlink_result, frame_id=3, session_id=session_id, run_id=run_id)
            frames = satellite.payload.drain_downlink(int(downlink_result["transfer_id"]))
            active = satellite.payload.file_downlink.active
            assert active is not None
            file_epoch = active.attempt_epoch
            for index, frame in enumerate(frames, start=4):
                envelope = ValidatedTransportEnvelope(
                    instance,
                    satellite.payload.journal.boot_id,
                    session_id,
                    1,
                    run_id,
                    index,
                    index,
                    file_epoch,
                    0,
                    int(time.time_ns() // 1_000),
                    "DOWNLINK",
                    frame,
                )
                result = gds.ingest_tm(envelope)
                if result.error_code is not None:
                    raise RuntimeError(f"TM/file ingest failed: {result.error_code}")
            ground_product = gds.product_store.get(product)
            if ground_product is None or ground_product["state"] != "PUBLISHED":
                raise RuntimeError(f"ground product was not published: {ground_product}")
            return {
                "status": "PASS",
                "spacecraft_instance_id": f"{instance:016x}",
                "request_keys": [catalog_key.as_dict(), analysis_key.as_dict(), downlink_key.as_dict()],
                "catalog": catalog_status.as_dict(),
                "analysis": {"stage": analysis_result.get("stage"), "job_state": job["state"]},
                "downlink": {"stage": downlink_result.get("stage"), "frame_count": len(frames), "transfer_id": downlink_result.get("transfer_id")},
                "product": {"product_ref": product.as_dict(), "state": ground_product["state"], "bundle_sha256": ground_product["bundle_sha256"]},
                "shared_volume_bypass": False,
            }
        finally:
            satellite.close()
            gds.close()
            if product_path is not None and product_path.is_dir():
                shutil.rmtree(product_path, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(run(args.root.resolve()), ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
