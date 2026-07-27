"""F Prime FilePacket-compatible MVP framing and checksum helpers."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum

MAX_FILE_DATA_PER_FRAME = 990
FILE_PACKET_HEADER_SIZE = 11
PACKET_DESCRIPTOR_SIZE = 2


class FilePacketType(IntEnum):
    START = 1
    DATA = 2
    END = 3
    CANCEL = 4


@dataclass(frozen=True)
class FilePacket:
    packet_type: FilePacketType
    sequence_index: int
    offset: int
    payload: bytes = b""

    def __post_init__(self) -> None:
        if not 0 <= self.sequence_index <= 0xFFFFFFFF:
            raise ValueError("FilePacket sequenceIndex must fit U32")
        if not 0 <= self.offset <= 0xFFFFFFFF:
            raise ValueError("FilePacket offset must fit U32")
        if self.packet_type == FilePacketType.DATA and len(self.payload) > MAX_FILE_DATA_PER_FRAME:
            raise ValueError("FilePacket DATA exceeds the 990-byte TM frame boundary")
        if self.packet_type != FilePacketType.DATA and len(self.payload) > 0xFFFF:
            raise ValueError("FilePacket control payload is too large")

    def encode(self) -> bytes:
        return struct.pack(">BIIH", int(self.packet_type), self.sequence_index, self.offset, len(self.payload)) + self.payload

    @classmethod
    def decode(cls, data: bytes) -> "FilePacket":
        data = bytes(data)
        if len(data) < FILE_PACKET_HEADER_SIZE:
            raise ValueError("FilePacket is shorter than its 11-byte header")
        packet_type, sequence, offset, length = struct.unpack(">BIIH", data[:11])
        try:
            packet_type = FilePacketType(packet_type)
        except ValueError as exc:
            raise ValueError("unknown FilePacket type") from exc
        if len(data) != FILE_PACKET_HEADER_SIZE + length:
            raise ValueError("FilePacket payload length mismatch")
        return cls(packet_type, sequence, offset, data[11:])


def encode_file_packet(packet: FilePacket, descriptor: int = 0x0003) -> bytes:
    if not 0 <= descriptor <= 0xFFFF:
        raise ValueError("FilePacket descriptor must fit U16")
    return descriptor.to_bytes(PACKET_DESCRIPTOR_SIZE, "big") + packet.encode()


def decode_file_packet(data: bytes, expected_descriptor: int = 0x0003) -> FilePacket:
    data = bytes(data)
    if len(data) < PACKET_DESCRIPTOR_SIZE + FILE_PACKET_HEADER_SIZE:
        raise ValueError("descriptor/FilePacket payload is too short")
    descriptor = int.from_bytes(data[:2], "big")
    if descriptor != expected_descriptor:
        raise ValueError(f"unexpected F Prime packet descriptor 0x{descriptor:04x}")
    return FilePacket.decode(data[2:])


def cfdp_checksum(data: bytes, offset: int = 0, file_size: int | None = None) -> int:
    """Return the F Prime/CFDP additive checksum for a file or range.

    ``offset`` and ``file_size`` are useful for golden tests of non-aligned
    ranges. Missing bytes in a partial range are treated as zero, while a full
    file (the normal path) is grouped into absolute four-byte big-endian words.
    """

    data = bytes(data)
    if offset < 0 or offset > 0xFFFFFFFF:
        raise ValueError("checksum offset must be non-negative U32")
    if file_size is None:
        file_size = offset + len(data)
    if file_size < offset + len(data):
        raise ValueError("file_size cannot be smaller than the supplied range")
    total = 0
    for word_start in range((offset // 4) * 4, file_size, 4):
        word = bytearray(4)
        for index in range(4):
            absolute = word_start + index
            source_index = absolute - offset
            if 0 <= source_index < len(data):
                word[index] = data[source_index]
        total = (total + int.from_bytes(word, "big")) & 0xFFFFFFFF
    return total
