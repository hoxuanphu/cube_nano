"""F Prime descriptor payloads carried in CCSDS Space Packets."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

from .canonical import deterministic_cbor_decode, deterministic_cbor_encode
from .ccsds import SpacePacket, encode_space_packet


class PacketDescriptor(IntEnum):
    COMMAND = 0x0000
    TELEMETRY = 0x0001
    EVENT_ACK = 0x0002
    FILE = 0x0003


@dataclass(frozen=True)
class ApplicationMessage:
    descriptor: PacketDescriptor
    body: dict[str, Any]


def encode_application_message(descriptor: PacketDescriptor, body: dict[str, Any]) -> bytes:
    return int(descriptor).to_bytes(2, "big") + deterministic_cbor_encode(body)


def decode_application_message(payload: bytes) -> ApplicationMessage:
    payload = bytes(payload)
    if len(payload) < 3:
        raise ValueError("application payload is shorter than its descriptor")
    try:
        descriptor = PacketDescriptor(int.from_bytes(payload[:2], "big"))
    except ValueError as exc:
        raise ValueError("unknown F Prime packet descriptor") from exc
    body = deterministic_cbor_decode(payload[2:])
    if not isinstance(body, dict):
        raise ValueError("application message body must be a CBOR map")
    return ApplicationMessage(descriptor, body)


def encode_tm_application(
    apid: int,
    descriptor: PacketDescriptor,
    body: dict[str, Any],
    sequence_count: int,
) -> bytes:
    payload = encode_application_message(descriptor, body)
    return encode_space_packet(apid, payload, sequence_count)


def decode_tm_application(packet_bytes: bytes) -> tuple[SpacePacket, ApplicationMessage]:
    packet = SpacePacket.decode(packet_bytes)
    return packet, decode_application_message(packet.payload)
