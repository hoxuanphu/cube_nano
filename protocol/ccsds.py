"""Small, byte-oriented CCSDS profile codec for the local SIL.

This is intentionally limited to the profile declared in mission_profile.yaml:
Space Packets, TC Type-BD, and one packet per fixed-size TM frame. It does not
claim COP-1/FARM/CLCW or channel coding support.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

from .canonical import checked_u16, checked_u32

SPACE_PACKET_HEADER_SIZE = 6
TM_PRIMARY_HEADER_SIZE = 6
TM_FECF_SIZE = 2
TM_FRAME_SIZE = 1024
TM_DATA_FIELD_SIZE = TM_FRAME_SIZE - TM_PRIMARY_HEADER_SIZE - TM_FECF_SIZE
IDLE_APID = 0x7FF


def crc16_ccitt(data: bytes, initial: int = 0xFFFF) -> int:
    crc = checked_u16(initial, "CRC initial")
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


@dataclass(frozen=True)
class SpacePacket:
    apid: int
    sequence_count: int
    payload: bytes
    packet_type: int = 0
    secondary_header_present: bool = False
    sequence_flags: int = 3

    def __post_init__(self) -> None:
        if not 0 <= self.apid <= 0x7FF:
            raise ValueError("Space Packet APID must fit 11 bits")
        if not 0 <= self.sequence_count <= 0x3FFF:
            raise ValueError("Space Packet sequence count must fit 14 bits")
        if self.packet_type not in (0, 1):
            raise ValueError("Space Packet type must be 0 or 1")
        if self.sequence_flags not in (0, 1, 2, 3):
            raise ValueError("Space Packet sequence flags must fit 2 bits")
        if not isinstance(self.payload, bytes):
            raise TypeError("Space Packet payload must be bytes")
        if not self.payload:
            raise ValueError("MVP Space Packets contain at least one data byte")
        if len(self.payload) > 0x10000:
            raise ValueError("Space Packet payload is too large")

    def encode(self) -> bytes:
        first = (
            (self.packet_type << 12)
            | (int(self.secondary_header_present) << 11)
            | self.apid
        )
        second = (self.sequence_flags << 14) | self.sequence_count
        packet_data_length = len(self.payload) - 1
        return struct.pack(">HHH", first, second, packet_data_length) + self.payload

    @classmethod
    def decode(cls, data: bytes) -> "SpacePacket":
        data = bytes(data)
        if len(data) < SPACE_PACKET_HEADER_SIZE + 1:
            raise ValueError("Space Packet is shorter than its primary header")
        first, second, packet_data_length = struct.unpack(">HHH", data[:6])
        if first >> 13:
            raise ValueError("unsupported Space Packet version")
        expected = SPACE_PACKET_HEADER_SIZE + packet_data_length + 1
        if len(data) != expected:
            raise ValueError(
                f"Space Packet length field expects {expected} bytes, got {len(data)}"
            )
        return cls(
            apid=first & 0x7FF,
            sequence_count=second & 0x3FFF,
            payload=data[6:],
            packet_type=(first >> 12) & 1,
            secondary_header_present=bool((first >> 11) & 1),
            sequence_flags=(second >> 14) & 3,
        )


def encode_space_packet(
    apid: int,
    payload: bytes,
    sequence_count: int,
    *,
    packet_type: int = 0,
    secondary_header_present: bool = False,
    sequence_flags: int = 3,
) -> bytes:
    return SpacePacket(
        apid=apid,
        sequence_count=sequence_count,
        payload=bytes(payload),
        packet_type=packet_type,
        secondary_header_present=secondary_header_present,
        sequence_flags=sequence_flags,
    ).encode()


def decode_space_packet(data: bytes) -> SpacePacket:
    return SpacePacket.decode(data)


@dataclass(frozen=True)
class TmFrame:
    spacecraft_id: int
    virtual_channel_id: int
    master_channel_count: int
    virtual_channel_count: int
    packet: SpacePacket
    segment_length_id: int = 3
    ocf: bytes = b""

    def __post_init__(self) -> None:
        if not 0 <= self.spacecraft_id <= 0x3FF:
            raise ValueError("TM spacecraft ID must fit 10 bits")
        if not 0 <= self.virtual_channel_id <= 7:
            raise ValueError("TM VCID must fit 3 bits")
        if not 0 <= self.master_channel_count <= 0xFF:
            raise ValueError("TM MCFC must fit 8 bits")
        if not 0 <= self.virtual_channel_count <= 0xFF:
            raise ValueError("TM VCFC must fit 8 bits")
        if self.segment_length_id not in (0, 1, 2, 3):
            raise ValueError("TM segment length ID must fit 2 bits")
        if self.ocf:
            raise ValueError("MVP TM frames do not contain an OCF")

    def encode(self, frame_size: int = TM_FRAME_SIZE) -> bytes:
        if frame_size != TM_FRAME_SIZE:
            raise ValueError("MVP TM frame size is fixed at 1024 bytes")
        packet_bytes = self.packet.encode()
        if len(packet_bytes) > TM_DATA_FIELD_SIZE:
            raise ValueError("Space Packet does not fit in TM data field")
        remaining = TM_DATA_FIELD_SIZE - len(packet_bytes)
        if remaining == 0:
            filler = b""
        elif remaining >= SPACE_PACKET_HEADER_SIZE + 1:
            filler = encode_space_packet(IDLE_APID, b"\x00" * (remaining - 6), 0)
        else:
            raise ValueError("TM data field has a non-representable idle remainder")
        first = (self.spacecraft_id << 4) | (self.virtual_channel_id << 1)
        pointer = (self.segment_length_id << 11) | 0
        header = struct.pack(">HBBH", first, self.master_channel_count, self.virtual_channel_count, pointer)
        body = header + packet_bytes + filler
        fecf = crc16_ccitt(body).to_bytes(2, "big")
        result = body + fecf
        if len(result) != frame_size:
            raise AssertionError(f"TM encoder produced {len(result)} bytes")
        return result

    @classmethod
    def decode(cls, data: bytes, frame_size: int = TM_FRAME_SIZE) -> "TmFrame":
        data = bytes(data)
        if len(data) != frame_size:
            raise ValueError(f"TM frame must contain exactly {frame_size} bytes")
        expected_crc = int.from_bytes(data[-2:], "big")
        actual_crc = crc16_ccitt(data[:-2])
        if actual_crc != expected_crc:
            raise ValueError("TM FECF/CRC mismatch")
        first, mcfc, vcfc, pointer = struct.unpack(">HBBH", data[:6])
        if first >> 14:
            raise ValueError("unsupported TM transfer-frame version")
        if pointer & 0x7FF:
            raise ValueError("MVP TM first-header pointer must be zero")
        if len(data) < 6 + SPACE_PACKET_HEADER_SIZE:
            raise ValueError("TM frame has no complete first Space Packet header")
        packet_end = 6 + SPACE_PACKET_HEADER_SIZE + int.from_bytes(data[10:12], "big") + 1
        if packet_end > frame_size - 2:
            raise ValueError("TM first Space Packet exceeds frame data field")
        packet = SpacePacket.decode(data[6:packet_end])
        return cls(
            spacecraft_id=(first >> 4) & 0x3FF,
            virtual_channel_id=(first >> 1) & 0x7,
            master_channel_count=mcfc,
            virtual_channel_count=vcfc,
            packet=packet,
            segment_length_id=(pointer >> 11) & 0x3,
        )


def encode_tm_frame(
    packet: bytes | SpacePacket,
    *,
    spacecraft_id: int = 68,
    virtual_channel_id: int = 0,
    master_channel_count: int = 0,
    virtual_channel_count: int = 0,
) -> bytes:
    parsed = packet if isinstance(packet, SpacePacket) else SpacePacket.decode(packet)
    return TmFrame(
        spacecraft_id=spacecraft_id,
        virtual_channel_id=virtual_channel_id,
        master_channel_count=master_channel_count,
        virtual_channel_count=virtual_channel_count,
        packet=parsed,
    ).encode()


def decode_tm_frame(data: bytes) -> TmFrame:
    return TmFrame.decode(data)


@dataclass(frozen=True)
class TcTypeBdFrame:
    spacecraft_id: int
    virtual_channel_id: int
    sequence_number: int
    packet: SpacePacket

    HEADER_SIZE = 5

    def encode(self, frame_size: int | None = None) -> bytes:
        packet_bytes = self.packet.encode()
        if not 0 <= self.spacecraft_id <= 0x3FF:
            raise ValueError("TC spacecraft ID must fit 10 bits")
        if not 0 <= self.virtual_channel_id <= 0x3F:
            raise ValueError("TC VCID must fit 6 bits")
        if not 0 <= self.sequence_number <= 0xFF:
            raise ValueError("TC frame sequence must fit 8 bits")
        if frame_size is None:
            frame_size = self.HEADER_SIZE + len(packet_bytes) + 2
        if frame_size < self.HEADER_SIZE + len(packet_bytes) + 2 or frame_size - 1 > 0x3FF:
            raise ValueError("TC frame is too small for its Space Packet")
        data_field_size = frame_size - self.HEADER_SIZE - 2
        # Type-BD primary header: TFVN=0, bypass=1, control=0, reserved=0,
        # SCID(10), VCID(6), frame length(10), then frame sequence.
        frame_length = frame_size - 1
        header = bytes([
            0x20 | ((self.spacecraft_id >> 8) & 0x03),
            self.spacecraft_id & 0xFF,
            (self.virtual_channel_id << 2) | ((frame_length >> 8) & 0x03),
            frame_length & 0xFF,
            self.sequence_number,
        ])
        body = header + packet_bytes + b"\x00" * (data_field_size - len(packet_bytes))
        return body + crc16_ccitt(body).to_bytes(2, "big")

    @classmethod
    def decode(cls, data: bytes) -> "TcTypeBdFrame":
        data = bytes(data)
        if len(data) < cls.HEADER_SIZE + 2:
            raise ValueError("TC frame is too short")
        if crc16_ccitt(data[:-2]) != int.from_bytes(data[-2:], "big"):
            raise ValueError("TC FECF/CRC mismatch")
        if data[0] >> 6:
            raise ValueError("unsupported TC transfer-frame version")
        if not data[0] & 0x20 or data[0] & 0x10:
            raise ValueError("TC frame is not a Type-BD data frame")
        if data[0] & 0x0C:
            raise ValueError("TC transfer-frame reserved bits must be zero")
        frame_length = ((data[2] & 0x03) << 8) | data[3]
        if frame_length + 1 != len(data):
            raise ValueError("TC frame data-field length mismatch")
        packet_end = 5 + SPACE_PACKET_HEADER_SIZE + int.from_bytes(data[9:11], "big") + 1
        packet = SpacePacket.decode(data[5:packet_end])
        return cls(
            spacecraft_id=((data[0] & 0x03) << 8) | data[1],
            virtual_channel_id=(data[2] >> 2) & 0x3F,
            sequence_number=data[4],
            packet=packet,
        )
