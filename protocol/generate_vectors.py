"""Generate the byte-level vector inventory for the pinned local profile."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from protocol.canonical import canonical_json, u64_to_bytes, u64_to_json
from protocol.ccsds import TcTypeBdFrame, SpacePacket, encode_space_packet, encode_tm_frame
from protocol.file_packet import FilePacket, FilePacketType, cfdp_checksum, encode_file_packet
from protocol.messages import PacketDescriptor, encode_application_message, encode_tm_application
from protocol.schemas import Command, CommandOpcode, RequestKey, encode_command


def build_vectors() -> dict:
    packet_sequences = [
        {"sequence_count": count, "hex": encode_space_packet(1, b"\x00\x01", count).hex()}
        for count in (16382, 16383, 0, 1)
    ]
    tm_counters = [
        {"master_channel_count": count, "virtual_channel_count": count, "hex": encode_tm_frame(encode_tm_application(1, PacketDescriptor.TELEMETRY, {"counter": count}, count), master_channel_count=count, virtual_channel_count=count).hex()}
        for count in (254, 255, 0, 1)
    ]
    data_990 = encode_file_packet(FilePacket(FilePacketType.DATA, 1, 0, b"A" * 990))
    command = Command(
        CommandOpcode.SCENE_REQUEST_CATALOG,
        1,
        RequestKey(2**63, 0xFFFFFFFE),
        {},
    )
    tc_space_packet = SpacePacket(0, 16383, encode_command(command), packet_type=1)
    tc_frames = [
        {
            "sequence_number": count,
            "hex": TcTypeBdFrame(68, 0, count, tc_space_packet).encode().hex(),
        }
        for count in (254, 255, 0, 1)
    ]
    descriptor_vectors = []
    for apid, descriptor in (
        (0, PacketDescriptor.COMMAND),
        (1, PacketDescriptor.TELEMETRY),
        (2, PacketDescriptor.EVENT_ACK),
        (3, PacketDescriptor.FILE),
    ):
        body = {"apid": apid, "descriptor": int(descriptor)}
        application = encode_application_message(descriptor, body)
        descriptor_vectors.append(
            {
                "apid": apid,
                "descriptor": int(descriptor),
                "application_hex": application.hex(),
                "space_packet_hex": encode_space_packet(apid, application, apid).hex(),
            }
        )
    file_packets = [
        FilePacket(FilePacketType.START, 0, 0, b"source\0destination\0"),
        FilePacket(FilePacketType.DATA, 1, 0, b"A" * 990),
        FilePacket(FilePacketType.END, 2, 990, cfdp_checksum(b"A" * 990).to_bytes(4, "big")),
        FilePacket(FilePacketType.CANCEL, 2, 990, b"MISSION_CANCEL"),
    ]
    return {
        "schema_version": 1,
        "profile_id": "local_sil-fprime-v4.1.0-stock-apid",
        "u64": [
            {"value": value, "json": u64_to_json(value), "wire_hex": u64_to_bytes(value).hex()}
            for value in (0, 2**53 - 1, 2**53, 2**63 - 1, 2**63, 2**64 - 1)
        ],
        "space_packet_sequences": packet_sequences,
        "tc_type_bd_sequences": tc_frames,
        "tm_counters": tm_counters,
        "descriptor_apid_mapping": descriptor_vectors,
        "file_packets": [
            {
                "packet_type": packet.packet_type.name,
                "sequence_index": packet.sequence_index,
                "offset": packet.offset,
                "payload_bytes": len(packet.payload),
                "hex": encode_file_packet(packet).hex(),
            }
            for packet in file_packets
        ],
        "file_data_boundary": {
            "accepted_payload_bytes": 990,
            "encoded_descriptor_filepacket_bytes": len(data_990),
            "rejected_payload_bytes": 991,
        },
        "cfdp_checksum": [
            {"offset": 0, "data_hex": value.hex(), "checksum": cfdp_checksum(value)}
            for value in (b"", b"A", b"ABCDE", bytes(range(17)))
        ]
        + [
            {
                "offset": 1,
                "file_size": 7,
                "data_hex": b"ABCDE".hex(),
                "checksum": cfdp_checksum(b"ABCDE", offset=1, file_size=7),
            }
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("protocol/golden_vectors/vectors.json"))
    args = parser.parse_args()
    encoded = canonical_json(build_vectors()) + b"\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(encoded)
    print(json.dumps({"sha256": hashlib.sha256(encoded).hexdigest(), "path": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
