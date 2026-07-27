import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from flight.journal import SatelliteJournal  # noqa: E402
from flight.mission_com_scheduler import MissionComScheduler, QueueKind  # noqa: E402
from flight.mission_udp_adapter import MissionUdpAdapter  # noqa: E402
from protocol.canonical import (  # noqa: E402
    deterministic_cbor_decode,
    deterministic_cbor_encode,
    u64_from_json,
    u64_to_bytes,
    u64_to_json,
)
from protocol.ccsds import TcTypeBdFrame, crc16_ccitt, decode_space_packet, decode_tm_frame, encode_space_packet, encode_tm_frame  # noqa: E402
from protocol.file_packet import FilePacket, FilePacketType, decode_file_packet, encode_file_packet  # noqa: E402
from protocol.generate_vectors import build_vectors  # noqa: E402
from protocol.messages import PacketDescriptor, decode_application_message, encode_tm_application  # noqa: E402
from protocol.profile import MissionProfile  # noqa: E402
from protocol.schemas import (  # noqa: E402
    Command,
    CommandOpcode,
    ROI,
    RequestKey,
    SceneRef,
    decode_command,
    encode_command,
)
from sat_ai.products import build_ustar  # noqa: E402
from sat_ai.roi import open_memmap_scene  # noqa: E402


class MissionContractTests(unittest.TestCase):
    def test_mission_profile_freezes_worker_and_scheduler_bounds(self):
        profile = MissionProfile.from_file(ROOT / "protocol" / "mission_profile.yaml")
        self.assertEqual(profile.max_pending_jobs, 4)
        self.assertEqual(
            (profile.ack_mailbox_capacity, profile.control_queue_capacity, profile.file_queue_capacity),
            (32, 64, 16),
        )
        self.assertEqual(
            (profile.worker_heartbeat_interval_ms, profile.worker_heartbeat_timeout_ms),
            (1000, 5000),
        )

    def test_build_manifest_hashes_release_artifacts(self):
        manifest = json.loads((ROOT / "build_manifest.json").read_text(encoding="utf-8"))
        for relative_path, expected_sha256 in manifest["artifacts"].items():
            with self.subTest(path=relative_path):
                self.assertEqual(
                    hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest(),
                    expected_sha256,
                )

    def test_u64_boundaries_and_strict_json(self):
        for value in (0, 2**53 - 1, 2**53, 2**63 - 1, 2**63, 2**64 - 1):
            self.assertEqual(int.from_bytes(u64_to_bytes(value), "big"), value)
            self.assertEqual(u64_from_json(u64_to_json(value)), value)
        for value in ("000000000000000A", "0x0000000000000001", "1", 1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                u64_from_json(value)

    def test_deterministic_cbor_sorts_maps_and_round_trips_u64(self):
        first = deterministic_cbor_encode({"b": 2, "a": 1, 0: 2**64 - 1})
        second = deterministic_cbor_encode({0: 2**64 - 1, "a": 1, "b": 2})
        self.assertEqual(first, second)
        self.assertEqual(deterministic_cbor_decode(first)[0], 2**64 - 1)

    def test_command_round_trip_preserves_scene_anchored_roi(self):
        command = Command(
            CommandOpcode.ROI_REQUEST,
            1,
            RequestKey(2, 3),
            {
                "scene_ref": SceneRef(4, 5, 6).as_dict(),
                "roi": ROI(1, 2, 256, 512).as_dict(),
                "expected_config_epoch": 0,
                "expected_config_revision": 1,
                "model_threshold_bp": 5000,
                "coverage_limit_bp": 6000,
            },
        )
        self.assertEqual(decode_command(encode_command(command)), command)

    def test_tm_frame_has_fixed_size_crc_and_rollover_values(self):
        packet = encode_tm_application(1, PacketDescriptor.TELEMETRY, {"ok": True}, 16383)
        frame = encode_tm_frame(packet, master_channel_count=255, virtual_channel_count=255)
        self.assertEqual(len(frame), 1024)
        decoded = decode_tm_frame(frame)
        self.assertEqual(decoded.packet.sequence_count, 16383)
        corrupted = bytearray(frame)
        corrupted[20] ^= 0x01
        with self.assertRaisesRegex(ValueError, "CRC"):
            decode_tm_frame(corrupted)

    def test_tc_type_bd_sets_bypass_and_scid_bits(self):
        packet = decode_space_packet(encode_space_packet(0, b"command", 1, packet_type=1))
        encoded = TcTypeBdFrame(68, 0, 255, packet).encode()
        self.assertEqual(encoded[0], 0x20)
        self.assertEqual(encoded[1], 68)
        decoded = TcTypeBdFrame.decode(encoded)
        self.assertEqual(decoded.spacecraft_id, 68)
        self.assertEqual(decoded.sequence_number, 255)
        invalid = bytearray(encoded)
        invalid[0] &= ~0x20
        invalid[-2:] = crc16_ccitt(invalid[:-2]).to_bytes(2, "big")
        with self.assertRaisesRegex(ValueError, "Type-BD"):
            TcTypeBdFrame.decode(invalid)

    def test_file_packet_990_boundary(self):
        self.assertEqual(len(encode_file_packet(FilePacket(FilePacketType.DATA, 1, 0, b"x" * 990))), 1003)
        with self.assertRaises(ValueError):
            FilePacket(FilePacketType.DATA, 1, 0, b"x" * 991)

    def test_committed_golden_vectors_are_complete_and_regenerable(self):
        committed = json.loads((ROOT / "protocol" / "golden_vectors" / "vectors.json").read_text(encoding="utf-8"))
        self.assertEqual(committed, build_vectors())
        self.assertEqual([item["sequence_count"] for item in committed["space_packet_sequences"]], [16382, 16383, 0, 1])
        self.assertEqual([item["sequence_number"] for item in committed["tc_type_bd_sequences"]], [254, 255, 0, 1])
        for item in committed["tc_type_bd_sequences"]:
            tc = TcTypeBdFrame.decode(bytes.fromhex(item["hex"]))
            self.assertEqual(tc.packet.packet_type, 1)
            self.assertEqual(tc.packet.apid, 0)
        self.assertEqual(
            [(item["apid"], item["descriptor"]) for item in committed["descriptor_apid_mapping"]],
            [(0, 0), (1, 1), (2, 2), (3, 3)],
        )
        for item in committed["descriptor_apid_mapping"]:
            packet = decode_space_packet(bytes.fromhex(item["space_packet_hex"]))
            message = decode_application_message(packet.payload)
            self.assertEqual(packet.apid, item["apid"])
            self.assertEqual(int(message.descriptor), item["descriptor"])
        for item in committed["file_packets"]:
            packet = decode_file_packet(bytes.fromhex(item["hex"]))
            self.assertEqual(packet.packet_type.name, item["packet_type"])
            self.assertEqual(len(packet.payload), item["payload_bytes"])

    def test_ustar_is_byte_deterministic(self):
        entries = {"z.txt": b"z", "a.txt": b"a"}
        self.assertEqual(build_ustar(entries), build_ustar(dict(reversed(list(entries.items())))))

    def test_scheduler_completion_waits_for_status_and_return(self):
        scheduler = MissionComScheduler()
        scheduler.enqueue_ack(b"one")
        scheduler.enqueue_control(b"two")
        adapter = MissionUdpAdapter(scheduler)
        first = adapter.send_next()
        self.assertIsNotNone(first)
        adapter.receive_status("OK")
        self.assertEqual(scheduler.state.value, "IN_FLIGHT")
        self.assertIsNone(adapter.send_next())
        adapter.receive_return()
        self.assertEqual(scheduler.state.value, "READY")
        second = adapter.send_next()
        self.assertIsNotNone(second)
        self.assertNotEqual(first.item_id, second.item_id)

    def test_scheduler_preserves_tm_channel_order_across_priority_queues(self):
        scheduler = MissionComScheduler()
        # ACK normally outranks FILE, but it cannot overtake a file START that
        # already owns the preceding durable MCFC/VCFC allocation.
        scheduler.enqueue_file(b"file-start", ordering_key=12)
        scheduler.enqueue_ack(b"later-event", ordering_key=13)

        first = scheduler.poll()
        self.assertIsNotNone(first)
        self.assertEqual(first.kind, QueueKind.FILE)
        self.assertEqual(first.ordering_key, 12)
        scheduler.mark_status(first.item_id, "LINK_CONSUMED")
        scheduler.mark_upstream_return(first.item_id)

        second = scheduler.poll()
        self.assertIsNotNone(second)
        self.assertEqual(second.kind, QueueKind.ACK)
        self.assertEqual(second.ordering_key, 13)

    def test_completion_callback_can_poll_reentrantly(self):
        scheduler = MissionComScheduler()
        scheduler.enqueue_control(b"next")
        adapter = MissionUdpAdapter(scheduler)
        polled = []

        def on_complete(item, status):
            polled.append(adapter.send_next())

        scheduler.enqueue_ack(b"first", on_complete)
        first = adapter.send_next()
        adapter.receive_return()
        adapter.receive_status("OK")
        self.assertEqual(len(polled), 1)
        self.assertIsNotNone(polled[0])
        self.assertIsNotNone(adapter.gate)

    def test_session_reset_completes_inflight_frame_once_and_blocks_next(self):
        scheduler = MissionComScheduler()
        completions = []
        scheduler.enqueue_file(b"frame", lambda item, status: completions.append(status))
        adapter = MissionUdpAdapter(scheduler)
        first = adapter.send_next()
        adapter.reset()
        self.assertEqual(completions, ["SESSION_RESET"])
        self.assertEqual(scheduler.state.value, "NOT_READY")
        self.assertIsNone(scheduler.current)
        adapter.receive_return()
        adapter.receive_status("SUCCESS")
        self.assertEqual(completions, ["SESSION_RESET"])
        self.assertIsNone(adapter.send_next())

    def test_memmap_scene_window_rejects_compressed_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.tif"
            tifffile.imwrite(source, np.zeros((8, 8, 3), dtype=np.uint16), compression="deflate", metadata={"axes": "YXC"})
            sidecar = root / "source.json"
            sidecar.write_text(json.dumps({"schema_version": 1, "source_fingerprint": {"algorithm": "sha256", "digest": hashlib.sha256(source.read_bytes()).hexdigest()}, "axes": "YXC", "shape": [8, 8, 3], "band_order": ["red", "green", "blue"], "dtype": "uint16", "validity": {"kind": "all_valid"}}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "UNSUPPORTED_SCENE_FORMAT"):
                open_memmap_scene(source, sidecar)

    def test_journal_replays_same_digest_and_rejects_conflict(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = SatelliteJournal(Path(directory) / "sat.sqlite3", 1)
            key = RequestKey(2, 3)
            journal.record_command(key, 1, "digest-a", {"x": 1}, "EXECUTED", {"stage": "EXECUTED"})
            self.assertEqual(journal.lookup_request(key, "digest-a")[0], "DUPLICATE")
            self.assertEqual(journal.lookup_request(key, "digest-b")[0], "CONFLICT")
            journal.compact_request(key)
            self.assertEqual(journal.lookup_request(key, "digest-a")[0], "RETIRED")
            journal.close()
