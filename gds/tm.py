"""Validated TM transport envelopes and stock APID/descriptor decoding."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import IntEnum, StrEnum
from typing import Any, Mapping

from protocol.canonical import canonical_json, checked_u16, checked_u32, checked_u64, deterministic_cbor_encode, u64_to_json
from protocol.ccsds import TmFrame, decode_tm_frame
from protocol.file_packet import FilePacket, decode_file_packet
from protocol.messages import PacketDescriptor, decode_application_message, encode_application_message

from link_sim.transport import Direction, SidebandEnvelope

from .u64 import decode_sqlite_u64, encode_sqlite_u64
from .writer import MutationPriority, SQLiteWriter


class TmDecodeError(ValueError):
    """A transfer frame or application descriptor failed validation."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class TmPacketKind(StrEnum):
    TELEMETRY = "TELEMETRY"
    EVENT_ACK = "EVENT_ACK"
    FILE = "FILE"


@dataclass(frozen=True)
class ValidatedTransportEnvelope:
    """Identity captured at transport validation time.

    The decoder never looks up a mutable current-link object to populate these
    fields.  A receiver may compare an envelope with a binding snapshot, but
    the resulting record remains self-contained for replay and audit.
    """

    source_spacecraft_instance_id: int
    sender_boot_id: int
    link_session_id: int
    link_generation: int
    simulation_run_id: int
    link_frame_id: int
    sender_frame_id: int
    file_epoch_id: int
    copy_index: int
    received_at_us: int
    direction: str
    frame_bytes: bytes
    fault: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        checked_u64(self.source_spacecraft_instance_id, "source_spacecraft_instance_id")
        checked_u32(self.sender_boot_id, "sender_boot_id")
        checked_u64(self.link_session_id, "link_session_id")
        checked_u64(self.link_generation, "link_generation")
        checked_u64(self.simulation_run_id, "simulation_run_id")
        checked_u64(self.link_frame_id, "link_frame_id")
        checked_u64(self.sender_frame_id, "sender_frame_id")
        checked_u64(self.file_epoch_id, "file_epoch_id")
        checked_u32(self.copy_index, "copy_index")
        if self.received_at_us < 0:
            raise ValueError("received_at_us must be non-negative")
        if self.direction not in {"UPLINK", "DOWNLINK"}:
            raise ValueError("transport direction must be UPLINK or DOWNLINK")
        if not isinstance(self.frame_bytes, bytes) or not self.frame_bytes:
            raise ValueError("frame_bytes must be non-empty bytes")

    @classmethod
    def from_sideband(
        cls,
        sideband: SidebandEnvelope,
        frame_bytes: bytes,
        *,
        received_at_us: int,
        simulation_run_id: int,
        copy_index: int = 0,
        link_generation: int = 0,
        fault: Mapping[str, Any] | None = None,
    ) -> "ValidatedTransportEnvelope":
        try:
            sideband.validate_egress()
        except (TypeError, ValueError) as exc:
            raise TmDecodeError("INVALID_TRANSPORT_ENVELOPE", str(exc)) from exc
        if len(frame_bytes) != sideband.frame_length:
            raise TmDecodeError("FRAME_LENGTH_MISMATCH", "sideband frame length does not match bytes")
        if sideband.direction is not Direction.EGRESS:
            raise TmDecodeError("INVALID_DIRECTION", "TM decoder accepts egress frames only")
        return cls(
            sideband.spacecraft_instance_id,
            sideband.sender_boot_id,
            sideband.link_session_id,
            link_generation,
            simulation_run_id,
            sideband.link_frame_id,
            sideband.sender_frame_id,
            sideband.file_epoch_id,
            copy_index,
            received_at_us,
            "DOWNLINK",
            bytes(frame_bytes),
            fault,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_spacecraft_instance_id": u64_to_json(self.source_spacecraft_instance_id),
            "sender_boot_id": self.sender_boot_id,
            "link_session_id": u64_to_json(self.link_session_id),
            "link_generation": u64_to_json(self.link_generation),
            "simulation_run_id": u64_to_json(self.simulation_run_id),
            "link_frame_id": u64_to_json(self.link_frame_id),
            "sender_frame_id": u64_to_json(self.sender_frame_id),
            "file_epoch_id": u64_to_json(self.file_epoch_id),
            "copy_index": self.copy_index,
            "received_at_us": self.received_at_us,
            "direction": self.direction,
            "fault": dict(self.fault or {}),
        }


@dataclass(frozen=True)
class DecodedTmPacket:
    envelope: ValidatedTransportEnvelope
    frame: TmFrame
    apid: int
    descriptor: int
    kind: TmPacketKind
    application_payload: bytes
    message: dict[str, Any] | None = None
    file_packet: FilePacket | None = None

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "transport": self.envelope.as_dict(),
            "spacecraft_id": self.frame.spacecraft_id,
            "virtual_channel_id": self.frame.virtual_channel_id,
            "master_channel_count": self.frame.master_channel_count,
            "virtual_channel_count": self.frame.virtual_channel_count,
            "apid": self.apid,
            "descriptor": self.descriptor,
            "kind": self.kind.value,
            "space_packet_sequence": self.frame.packet.sequence_count,
        }
        if self.message is not None:
            result["message"] = self.message
        if self.file_packet is not None:
            result["file_packet"] = {
                "type": self.file_packet.packet_type.name,
                "sequence_index": self.file_packet.sequence_index,
                "offset": self.file_packet.offset,
                "payload_length": len(self.file_packet.payload),
            }
        return result


class TmCounterStatus(StrEnum):
    BASELINE = "BASELINE"
    IN_ORDER = "IN_ORDER"
    ROLLOVER = "ROLLOVER"
    GAP = "GAP"
    DUPLICATE = "DUPLICATE"
    STALE_GENERATION = "STALE_GENERATION"
    STALE_SESSION = "STALE_SESSION"
    STALE_COUNTER = "STALE_COUNTER"
    COUNTER_CONFLICT = "COUNTER_CONFLICT"

    @property
    def accepted(self) -> bool:
        return self in {
            TmCounterStatus.BASELINE,
            TmCounterStatus.IN_ORDER,
            TmCounterStatus.ROLLOVER,
            TmCounterStatus.GAP,
        }


@dataclass(frozen=True)
class TmCounterObservation:
    status: TmCounterStatus
    master_gap: int = 0
    virtual_gap: int = 0
    packet_gap: int = 0
    file_gap: int = 0

    @property
    def accepted(self) -> bool:
        return self.status.accepted


class TmCounterLedger:
    """Persist TM counter baselines before decoded payloads are allowed to act."""

    def __init__(self, writer: SQLiteWriter) -> None:
        self.writer = writer

    @staticmethod
    def _advance(previous: int, current: int, modulus: int) -> tuple[str, int]:
        if current == previous:
            return "DUPLICATE", 0
        delta = (current - previous) % modulus
        if delta == 1:
            return ("ROLLOVER" if previous == modulus - 1 and current == 0 else "IN_ORDER"), 0
        # More than half a modulus is unambiguously an old/stale counter for
        # this profile.  The ambiguous half-modulus point fails closed too.
        if 1 < delta < modulus // 2:
            return "GAP", delta - 1
        return "STALE", 0

    @staticmethod
    def _increment_epoch(value: int, *, label: str) -> int:
        if value >= 0xFFFFFFFF:
            raise RuntimeError(f"{label} epoch exhausted; rebind the spacecraft instance")
        return value + 1

    @staticmethod
    def _observation_from_row(row: sqlite3.Row) -> TmCounterObservation:
        return TmCounterObservation(
            TmCounterStatus(str(row["status"])),
            int(row["master_gap"]),
            int(row["virtual_gap"]),
            int(row["packet_gap"]),
            int(row["file_gap"]),
        )

    def observe(self, decoded: DecodedTmPacket) -> TmCounterObservation:
        if not isinstance(decoded, DecodedTmPacket):
            raise TypeError("decoded must be a DecodedTmPacket")
        envelope = decoded.envelope
        if envelope.direction != "DOWNLINK":
            raise ValueError("TM counters accept DOWNLINK frames only")
        frame_sha256 = hashlib.sha256(envelope.frame_bytes).digest()
        file_scope = envelope.file_epoch_id if decoded.kind is TmPacketKind.FILE else 0
        file_sequence = (
            None
            if decoded.file_packet is None
            else int(decoded.file_packet.sequence_index)
        )

        def insert_observation(
            connection: sqlite3.Connection,
            observation: TmCounterObservation,
        ) -> None:
            connection.execute(
                "INSERT INTO tm_counter_observations("
                "source_spacecraft_instance_id,simulation_run_id,link_generation,"
                "link_session_id,link_frame_id,copy_index,apid,frame_sha256,status,"
                "master_gap,virtual_gap,packet_gap,file_gap,received_at_us) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    encode_sqlite_u64(envelope.source_spacecraft_instance_id),
                    encode_sqlite_u64(envelope.simulation_run_id),
                    encode_sqlite_u64(envelope.link_generation),
                    encode_sqlite_u64(envelope.link_session_id),
                    encode_sqlite_u64(envelope.link_frame_id),
                    envelope.copy_index,
                    decoded.apid,
                    frame_sha256,
                    observation.status.value,
                    observation.master_gap,
                    observation.virtual_gap,
                    observation.packet_gap,
                    observation.file_gap,
                    envelope.received_at_us,
                ),
            )

        def mutation(connection: sqlite3.Connection) -> TmCounterObservation:
            previous_observation = connection.execute(
                "SELECT frame_sha256,status,master_gap,virtual_gap,packet_gap,file_gap "
                "FROM tm_counter_observations WHERE source_spacecraft_instance_id=? "
                "AND simulation_run_id=? AND link_frame_id=? AND copy_index=?",
                (
                    encode_sqlite_u64(envelope.source_spacecraft_instance_id),
                    encode_sqlite_u64(envelope.simulation_run_id),
                    encode_sqlite_u64(envelope.link_frame_id),
                    envelope.copy_index,
                ),
            ).fetchone()
            if previous_observation is not None:
                if bytes(previous_observation["frame_sha256"]) != frame_sha256:
                    return TmCounterObservation(TmCounterStatus.COUNTER_CONFLICT)
                return TmCounterObservation(TmCounterStatus.DUPLICATE)

            generation_row = connection.execute(
                "SELECT active_link_generation,active_link_session_id "
                "FROM tm_source_generations WHERE source_spacecraft_instance_id=?",
                (encode_sqlite_u64(envelope.source_spacecraft_instance_id),),
            ).fetchone()
            if generation_row is None:
                connection.execute(
                    "INSERT INTO tm_source_generations("
                    "source_spacecraft_instance_id,active_link_generation,"
                    "active_link_session_id,updated_at_us) VALUES(?,?,?,?)",
                    (
                        encode_sqlite_u64(envelope.source_spacecraft_instance_id),
                        encode_sqlite_u64(envelope.link_generation),
                        encode_sqlite_u64(envelope.link_session_id),
                        envelope.received_at_us,
                    ),
                )
            else:
                active_generation = decode_sqlite_u64(
                    generation_row["active_link_generation"],
                    "active_link_generation",
                )
                active_session = decode_sqlite_u64(
                    generation_row["active_link_session_id"],
                    "active_link_session_id",
                )
                if envelope.link_generation < active_generation:
                    observation = TmCounterObservation(TmCounterStatus.STALE_GENERATION)
                    insert_observation(connection, observation)
                    return observation
                if (
                    envelope.link_generation == active_generation
                    and envelope.link_session_id != active_session
                ):
                    observation = TmCounterObservation(TmCounterStatus.STALE_SESSION)
                    insert_observation(connection, observation)
                    return observation
                if envelope.link_generation > active_generation:
                    connection.execute(
                        "UPDATE tm_source_generations SET active_link_generation=?,"
                        "active_link_session_id=?,updated_at_us=? "
                        "WHERE source_spacecraft_instance_id=?",
                        (
                            encode_sqlite_u64(envelope.link_generation),
                            encode_sqlite_u64(envelope.link_session_id),
                            envelope.received_at_us,
                            encode_sqlite_u64(envelope.source_spacecraft_instance_id),
                        ),
                    )

            channel_params = (
                encode_sqlite_u64(envelope.source_spacecraft_instance_id),
                encode_sqlite_u64(envelope.link_generation),
                encode_sqlite_u64(envelope.link_session_id),
                envelope.sender_boot_id,
                decoded.frame.virtual_channel_id,
            )
            channel_state = connection.execute(
                "SELECT * FROM tm_channel_counter_states "
                "WHERE source_spacecraft_instance_id=? AND link_generation=? "
                "AND link_session_id=? AND sender_boot_id=? "
                "AND virtual_channel_id=?",
                channel_params,
            ).fetchone()
            if channel_state is None:
                master_relation = virtual_relation = "BASELINE"
                master_gap = virtual_gap = 0
                master_epoch = virtual_epoch = 0
            else:
                master_relation, master_gap = self._advance(
                    int(channel_state["last_master_channel_count"]),
                    decoded.frame.master_channel_count,
                    256,
                )
                virtual_relation, virtual_gap = self._advance(
                    int(channel_state["last_virtual_channel_count"]),
                    decoded.frame.virtual_channel_count,
                    256,
                )
                master_epoch = int(channel_state["master_epoch"])
                virtual_epoch = int(channel_state["virtual_epoch"])

            packet_params = (*channel_params, decoded.apid)
            packet_state = connection.execute(
                "SELECT * FROM tm_packet_counter_states "
                "WHERE source_spacecraft_instance_id=? AND link_generation=? "
                "AND link_session_id=? AND sender_boot_id=? "
                "AND virtual_channel_id=? AND apid=?",
                packet_params,
            ).fetchone()
            if packet_state is None:
                packet_relation = "BASELINE"
                packet_gap = 0
                packet_epoch = 0
            else:
                packet_relation, packet_gap = self._advance(
                    int(packet_state["last_packet_sequence"]),
                    decoded.frame.packet.sequence_count,
                    16_384,
                )
                packet_epoch = int(packet_state["packet_epoch"])

            relations = (master_relation, virtual_relation, packet_relation)
            if all(relation == "DUPLICATE" for relation in relations):
                observation = TmCounterObservation(TmCounterStatus.DUPLICATE)
                insert_observation(connection, observation)
                return observation
            if "STALE" in relations or "DUPLICATE" in relations:
                observation = TmCounterObservation(TmCounterStatus.STALE_COUNTER)
                insert_observation(connection, observation)
                return observation

            if "GAP" in relations:
                status = TmCounterStatus.GAP
            elif "ROLLOVER" in relations:
                status = TmCounterStatus.ROLLOVER
            elif all(relation == "BASELINE" for relation in relations):
                status = TmCounterStatus.BASELINE
            else:
                status = TmCounterStatus.IN_ORDER

            if master_relation == "ROLLOVER":
                master_epoch = self._increment_epoch(master_epoch, label="TM master counter")
            if virtual_relation == "ROLLOVER":
                virtual_epoch = self._increment_epoch(virtual_epoch, label="TM virtual counter")
            if packet_relation == "ROLLOVER":
                packet_epoch = self._increment_epoch(packet_epoch, label="TM packet counter")

            if channel_state is None:
                connection.execute(
                    "INSERT INTO tm_channel_counter_states("
                    "source_spacecraft_instance_id,link_generation,link_session_id,"
                    "sender_boot_id,virtual_channel_id,last_master_channel_count,"
                    "last_virtual_channel_count,master_epoch,virtual_epoch,"
                    "last_link_frame_id,updated_at_us) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        *channel_params,
                        decoded.frame.master_channel_count,
                        decoded.frame.virtual_channel_count,
                        master_epoch,
                        virtual_epoch,
                        encode_sqlite_u64(envelope.link_frame_id),
                        envelope.received_at_us,
                    ),
                )
            else:
                connection.execute(
                    "UPDATE tm_channel_counter_states SET last_master_channel_count=?,"
                    "last_virtual_channel_count=?,master_epoch=?,virtual_epoch=?,"
                    "last_link_frame_id=?,updated_at_us=? "
                    "WHERE source_spacecraft_instance_id=? AND link_generation=? "
                    "AND link_session_id=? AND sender_boot_id=? "
                    "AND virtual_channel_id=?",
                    (
                        decoded.frame.master_channel_count,
                        decoded.frame.virtual_channel_count,
                        master_epoch,
                        virtual_epoch,
                        encode_sqlite_u64(envelope.link_frame_id),
                        envelope.received_at_us,
                        *channel_params,
                    ),
                )

            if packet_state is None:
                connection.execute(
                    "INSERT INTO tm_packet_counter_states("
                    "source_spacecraft_instance_id,link_generation,link_session_id,"
                    "sender_boot_id,virtual_channel_id,apid,last_packet_sequence,"
                    "packet_epoch,last_link_frame_id,updated_at_us) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (
                        *packet_params,
                        decoded.frame.packet.sequence_count,
                        packet_epoch,
                        encode_sqlite_u64(envelope.link_frame_id),
                        envelope.received_at_us,
                    ),
                )
            else:
                connection.execute(
                    "UPDATE tm_packet_counter_states SET last_packet_sequence=?,"
                    "packet_epoch=?,last_link_frame_id=?,updated_at_us=? "
                    "WHERE source_spacecraft_instance_id=? AND link_generation=? "
                    "AND link_session_id=? AND sender_boot_id=? "
                    "AND virtual_channel_id=? AND apid=?",
                    (
                        decoded.frame.packet.sequence_count,
                        packet_epoch,
                        encode_sqlite_u64(envelope.link_frame_id),
                        envelope.received_at_us,
                        *packet_params,
                    ),
                )

            file_gap = 0
            if file_sequence is not None:
                file_state_params = (*packet_params, encode_sqlite_u64(file_scope))
                file_state = connection.execute(
                    "SELECT last_file_sequence,file_epoch FROM tm_counter_states "
                    "WHERE source_spacecraft_instance_id=? AND link_generation=? "
                    "AND link_session_id=? AND sender_boot_id=? "
                    "AND virtual_channel_id=? AND apid=? AND file_epoch_id=?",
                    file_state_params,
                ).fetchone()
                if file_state is None:
                    next_file_sequence = file_sequence
                    file_epoch = 0
                    connection.execute(
                        "INSERT INTO tm_counter_states("
                        "source_spacecraft_instance_id,link_generation,link_session_id,"
                        "sender_boot_id,virtual_channel_id,apid,file_epoch_id,"
                        "last_master_channel_count,last_virtual_channel_count,"
                        "last_packet_sequence,last_file_sequence,master_epoch,virtual_epoch,"
                        "packet_epoch,file_epoch,last_link_frame_id,updated_at_us) "
                        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            *file_state_params,
                            decoded.frame.master_channel_count,
                            decoded.frame.virtual_channel_count,
                            decoded.frame.packet.sequence_count,
                            next_file_sequence,
                            master_epoch,
                            virtual_epoch,
                            packet_epoch,
                            file_epoch,
                            encode_sqlite_u64(envelope.link_frame_id),
                            envelope.received_at_us,
                        ),
                    )
                else:
                    previous_file_sequence = file_state["last_file_sequence"]
                    next_file_sequence = previous_file_sequence
                    if previous_file_sequence is None:
                        next_file_sequence = file_sequence
                    elif file_sequence > int(previous_file_sequence):
                        file_gap = max(0, file_sequence - int(previous_file_sequence) - 1)
                        next_file_sequence = file_sequence
                    # FilePacket DATA may arrive out of order.  Retain its
                    # high-watermark separately from APID packet sequencing.
                    connection.execute(
                        "UPDATE tm_counter_states SET last_master_channel_count=?,"
                        "last_virtual_channel_count=?,last_packet_sequence=?,"
                        "last_file_sequence=?,master_epoch=?,virtual_epoch=?,"
                        "packet_epoch=?,last_link_frame_id=?,updated_at_us=? "
                        "WHERE source_spacecraft_instance_id=? AND link_generation=? "
                        "AND link_session_id=? AND sender_boot_id=? "
                        "AND virtual_channel_id=? AND apid=? AND file_epoch_id=?",
                        (
                            decoded.frame.master_channel_count,
                            decoded.frame.virtual_channel_count,
                            decoded.frame.packet.sequence_count,
                            next_file_sequence,
                            master_epoch,
                            virtual_epoch,
                            packet_epoch,
                            encode_sqlite_u64(envelope.link_frame_id),
                            envelope.received_at_us,
                            *file_state_params,
                        ),
                    )

            observation = TmCounterObservation(
                status,
                master_gap,
                virtual_gap,
                packet_gap,
                file_gap,
            )
            insert_observation(connection, observation)
            return observation

        return self.writer.mutate(
            "observe_tm_counters",
            mutation,
            priority=MutationPriority.HIGH,
        )


class TMDecoder:
    """Decode only the pinned MVP TM profile (APID 1/2/3, VC0, CRC)."""

    DESCRIPTORS = {1: 0x0001, 2: 0x0002, 3: 0x0003}
    KINDS = {1: TmPacketKind.TELEMETRY, 2: TmPacketKind.EVENT_ACK, 3: TmPacketKind.FILE}

    def __init__(
        self,
        *,
        spacecraft_id: int = 68,
        expected_instance_id: int | None = None,
        expected_boot_id: int | None = None,
        expected_session_id: int | None = None,
        expected_link_generation: int | None = None,
    ) -> None:
        self.spacecraft_id = checked_u16(spacecraft_id, "spacecraft_id")
        if self.spacecraft_id > 0x3FF:
            raise ValueError("spacecraft_id must fit 10 bits")
        self.expected_instance_id = (
            None if expected_instance_id is None else checked_u64(expected_instance_id, "expected_instance_id")
        )
        self.expected_boot_id = (
            None if expected_boot_id is None else checked_u32(expected_boot_id, "expected_boot_id")
        )
        self.expected_session_id = (
            None if expected_session_id is None else checked_u64(expected_session_id, "expected_session_id")
        )
        self.expected_link_generation = (
            None if expected_link_generation is None else checked_u64(expected_link_generation, "expected_link_generation")
        )

    def decode(self, envelope: ValidatedTransportEnvelope) -> DecodedTmPacket:
        if not isinstance(envelope, ValidatedTransportEnvelope):
            raise TypeError("TM decoder requires a validated transport envelope")
        if self.expected_instance_id is not None and envelope.source_spacecraft_instance_id != self.expected_instance_id:
            raise TmDecodeError("SOURCE_INSTANCE_MISMATCH", "TM source instance does not match binding snapshot")
        if self.expected_boot_id is not None and envelope.sender_boot_id != self.expected_boot_id:
            raise TmDecodeError("SOURCE_BOOT_MISMATCH", "TM sender boot does not match binding snapshot")
        if self.expected_session_id is not None and envelope.link_session_id != self.expected_session_id:
            raise TmDecodeError("LINK_SESSION_MISMATCH", "TM link session does not match binding snapshot")
        if self.expected_link_generation is not None and envelope.link_generation != self.expected_link_generation:
            raise TmDecodeError("LINK_GENERATION_MISMATCH", "TM link generation does not match binding snapshot")
        try:
            frame = decode_tm_frame(envelope.frame_bytes)
        except (TypeError, ValueError) as exc:
            message = str(exc)
            code = "CRC_INVALID" if "CRC" in message else "INVALID_TM_FRAME"
            raise TmDecodeError(code, message) from exc
        if frame.spacecraft_id != self.spacecraft_id:
            raise TmDecodeError("SPACECRAFT_ID_MISMATCH", "TM SCID does not match mission profile")
        if frame.virtual_channel_id != 0:
            raise TmDecodeError("VCID_MISMATCH", "MVP TM decoder accepts VC0 only")
        packet = frame.packet
        expected_descriptor = self.DESCRIPTORS.get(packet.apid)
        if expected_descriptor is None:
            raise TmDecodeError("UNKNOWN_APID", f"unsupported TM APID {packet.apid}")
        if len(packet.payload) < 2:
            raise TmDecodeError("MISSING_DESCRIPTOR", "TM application payload has no F Prime descriptor")
        descriptor = int.from_bytes(packet.payload[:2], "big")
        if descriptor != expected_descriptor:
            raise TmDecodeError("DESCRIPTOR_MISMATCH", f"APID {packet.apid} has descriptor 0x{descriptor:04x}")
        application_payload = packet.payload[2:]
        kind = self.KINDS[packet.apid]
        file_packet = None
        message = None
        if kind is TmPacketKind.FILE:
            try:
                file_packet = decode_file_packet(packet.payload)
            except (TypeError, ValueError) as exc:
                raise TmDecodeError("INVALID_FILE_PACKET", str(exc)) from exc
        else:
            message = _decode_message(packet.payload)
        return DecodedTmPacket(envelope, frame, packet.apid, descriptor, kind, application_payload, message, file_packet)


def _decode_message(payload: bytes) -> dict[str, Any]:
    try:
        parsed = decode_application_message(payload)
        if deterministic_cbor_encode(parsed.body) != bytes(payload[2:]):
            raise TmDecodeError("NON_CANONICAL_APPLICATION_PAYLOAD", "TM CBOR payload is not deterministic")
        return parsed.body
    except TmDecodeError:
        raise
    except (TypeError, ValueError):
        pass
    # JSON is retained as a migration-friendly local fixture encoding.  The
    # stock F Prime reference path above is the canonical runtime path.
    payload = payload[2:]
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TmDecodeError("INVALID_APPLICATION_PAYLOAD", "TM payload is not canonical JSON") from exc
    if not isinstance(value, dict):
        raise TmDecodeError("INVALID_APPLICATION_PAYLOAD", "TM payload must be a JSON object")
    if canonical_json(value) != bytes(payload):
        raise TmDecodeError("NON_CANONICAL_APPLICATION_PAYLOAD", "TM payload is not canonical JSON")
    return value


def encode_tm_message(
    apid: int,
    message: Mapping[str, Any],
    *,
    sequence_count: int = 0,
    spacecraft_id: int = 68,
    virtual_channel_id: int = 0,
    master_channel_count: int = 0,
    virtual_channel_count: int = 0,
) -> bytes:
    """Encode deterministic telemetry/event JSON for local SIL fixtures."""

    if apid not in TMDecoder.DESCRIPTORS or apid == 3:
        raise ValueError("encode_tm_message supports telemetry/event APIDs 1 and 2")
    descriptor = PacketDescriptor(TMDecoder.DESCRIPTORS[apid])
    payload = encode_application_message(descriptor, dict(message))
    from protocol.ccsds import encode_space_packet, encode_tm_frame

    return encode_tm_frame(
        encode_space_packet(apid, payload, sequence_count),
        spacecraft_id=spacecraft_id,
        virtual_channel_id=virtual_channel_id,
        master_channel_count=master_channel_count,
        virtual_channel_count=virtual_channel_count,
    )


def utc_now_us() -> int:
    return int(datetime.now(UTC).timestamp() * 1_000_000)
