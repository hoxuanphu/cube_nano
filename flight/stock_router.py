"""Stock APID router shape for the F Prime v4.1.0 MVP mapping."""

from __future__ import annotations

from dataclasses import dataclass

from protocol.ccsds import SpacePacket
from protocol.schemas import Command, decode_command


@dataclass(frozen=True)
class RouteResult:
    accepted: bool
    apid: int
    command: Command | None = None
    error_code: str | None = None


class StockApidRouter:
    def __init__(
        self,
        command_apid: int = 0,
        *,
        expected_packet_type: int = 1,
        expected_secondary_header_present: bool = False,
        expected_sequence_flags: int = 3,
    ):
        self.command_apid = command_apid
        self.expected_packet_type = expected_packet_type
        self.expected_secondary_header_present = expected_secondary_header_present
        self.expected_sequence_flags = expected_sequence_flags

    def route_tc(self, packet_bytes: bytes) -> RouteResult:
        try:
            packet = SpacePacket.decode(packet_bytes)
        except ValueError as exc:
            return RouteResult(False, -1, error_code="INVALID_PACKET")
        if packet.packet_type != self.expected_packet_type:
            return RouteResult(False, packet.apid, error_code="PACKET_TYPE_MISMATCH")
        if packet.secondary_header_present != self.expected_secondary_header_present:
            return RouteResult(False, packet.apid, error_code="SECONDARY_HEADER_MISMATCH")
        if packet.sequence_flags != self.expected_sequence_flags:
            return RouteResult(False, packet.apid, error_code="SEQUENCE_FLAGS_MISMATCH")
        if packet.apid != self.command_apid:
            return RouteResult(False, packet.apid, error_code="INVALID_APID")
        try:
            command = decode_command(packet.payload)
        except ValueError as exc:
            return RouteResult(False, packet.apid, error_code="INVALID_COMMAND")
        return RouteResult(True, packet.apid, command=command)
