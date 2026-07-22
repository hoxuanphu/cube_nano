"""TM ingest orchestration: decode, persist frame metadata, and fan out by APID."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from protocol.canonical import checked_u32, u64_to_json
from protocol.schemas import RequestKey

from .events import EventRecord, EventStore
from .file_reassembly import FilePacketReassembler, ReassemblyResult
from .ledger import AtomicCommandLedger, LedgerIntegrityError
from .metrics import MetricsRegistry
from .outbox import AckResult, OutboxService
from .realtime import RealtimeHub
from .telemetry import TelemetryIngestResult, TelemetrySample, TelemetryStore
from .tm import (
    DecodedTmPacket,
    TMDecoder,
    TmCounterLedger,
    TmCounterObservation,
    TmDecodeError,
    TmPacketKind,
    ValidatedTransportEnvelope,
)
from .u64 import encode_sqlite_u64
from .writer import MutationPriority, SQLiteWriter


@dataclass(frozen=True)
class TmIngestResult:
    decoded: DecodedTmPacket | None
    telemetry: TelemetryIngestResult | None = None
    event: EventRecord | None = None
    file: ReassemblyResult | None = None
    counter: TmCounterObservation | None = None
    ack: AckResult | None = None
    error_code: str | None = None


class TmIngestService:
    def __init__(
        self,
        writer: SQLiteWriter,
        decoder: TMDecoder,
        *,
        telemetry: TelemetryStore | None = None,
        events: EventStore | None = None,
        reassembler: FilePacketReassembler | None = None,
        metrics: MetricsRegistry | None = None,
        realtime: RealtimeHub | None = None,
        outbox: OutboxService | None = None,
        ledger: AtomicCommandLedger | None = None,
        counters: TmCounterLedger | None = None,
    ) -> None:
        self.writer = writer
        self.decoder = decoder
        self.telemetry = telemetry or TelemetryStore(writer)
        self.events = events or EventStore(writer)
        self.reassembler = reassembler
        self.metrics = metrics or MetricsRegistry()
        self.realtime = realtime
        self.outbox = outbox
        self.ledger = ledger
        self.counters = counters or TmCounterLedger(writer)

    def ingest(self, envelope: ValidatedTransportEnvelope) -> TmIngestResult:
        try:
            decoded = self.decoder.decode(envelope)
        except TmDecodeError as exc:
            self.metrics.inc("gds_tm_decode_errors_total", labels={"code": exc.code})
            self._record_invalid_frame(envelope, exc.code)
            return TmIngestResult(None, error_code=exc.code)
        counter = self.counters.observe(decoded)
        self.metrics.inc(
            "gds_tm_counter_observations_total",
            labels={"status": counter.status.value},
        )
        self.metrics.inc("gds_tm_frames_received_total", labels={"apid": decoded.apid})
        self._record_frame(decoded)
        if not counter.accepted:
            return TmIngestResult(decoded, counter=counter, error_code=counter.status.value)
        if decoded.kind is TmPacketKind.TELEMETRY:
            result = self._ingest_telemetry(decoded)
            return TmIngestResult(
                result.decoded,
                telemetry=result.telemetry,
                event=result.event,
                file=result.file,
                counter=counter,
                ack=result.ack,
                error_code=result.error_code,
            )
        if decoded.kind is TmPacketKind.EVENT_ACK:
            result = self._ingest_event(decoded)
            return TmIngestResult(
                result.decoded,
                telemetry=result.telemetry,
                event=result.event,
                file=result.file,
                counter=counter,
                ack=result.ack,
                error_code=result.error_code,
            )
        if self.reassembler is None:
            return TmIngestResult(decoded, counter=counter, error_code="FILE_REASSEMBLER_UNAVAILABLE")
        try:
            result = self.reassembler.receive(
                decoded.file_packet,
                spacecraft_instance_id=envelope.source_spacecraft_instance_id,
                link_session_id=envelope.link_session_id,
                file_epoch_id=envelope.file_epoch_id,
                received_at_us=envelope.received_at_us,
            )
        except (TypeError, ValueError) as exc:
            code = getattr(exc, "code", "FILE_REASSEMBLY_ERROR")
            self.metrics.inc("gds_file_reassembly_errors_total", labels={"code": code})
            return TmIngestResult(decoded, counter=counter, error_code=code)
        if self.ledger is not None and result.product_ref is not None and result.transfer_id is not None:
            transfer_state = {
                "RECEIVING": "RECEIVING",
                "VERIFIED": "VERIFIED",
                "CANCELED": "CANCELED",
                "INCOMPLETE": "FAILED",
                "CHECKSUM_FAILED": "FAILED",
            }.get(result.state)
            if transfer_state is not None:
                try:
                    linked_request_key = self.ledger.update_product_downlink_file_state(
                        result.product_ref,
                        transfer_id=result.transfer_id,
                        transfer_state=transfer_state,
                    )
                except (TypeError, ValueError, LedgerIntegrityError):
                    self.metrics.inc(
                        "gds_file_transfer_correlation_total",
                        labels={"result": "REJECTED"},
                    )
                else:
                    self.metrics.inc(
                        "gds_file_transfer_correlation_total",
                        labels={
                            "result": (
                                "ATTACHED"
                                if linked_request_key is not None
                                else "DEFERRED"
                            )
                        },
                    )
        self.metrics.set("gds_file_bytes_received", result.bytes_received)
        return TmIngestResult(decoded, file=result, counter=counter)

    def _ingest_telemetry(self, decoded: DecodedTmPacket) -> TmIngestResult:
        assert decoded.message is not None
        message = decoded.message
        try:
            channel_id = checked_u32(message.get("channel_id"), "channel_id")
            sample = TelemetrySample(
                source_spacecraft_instance_id=decoded.envelope.source_spacecraft_instance_id,
                source_boot_id=decoded.envelope.sender_boot_id,
                simulation_run_id=decoded.envelope.simulation_run_id,
                direction=decoded.envelope.direction,
                link_session_id=decoded.envelope.link_session_id,
                link_frame_id=decoded.envelope.link_frame_id,
                copy_index=decoded.envelope.copy_index,
                sample_ordinal=int(message.get("sample_ordinal", 0)),
                apid=decoded.apid,
                channel_id=channel_id,
                received_at_us=decoded.envelope.received_at_us,
                raw_value=decoded.application_payload,
                decoded_value=message.get("value"),
                satellite_time_us=message.get("satellite_time_us"),
            )
            stored = self.telemetry.ingest(sample)
            return TmIngestResult(decoded, telemetry=stored)
        except (TypeError, ValueError) as exc:
            code = getattr(exc, "error_code", "TELEMETRY_SCHEMA_ERROR")
            self.metrics.inc("gds_tm_payload_errors_total", labels={"code": code})
            return TmIngestResult(decoded, error_code=code)

    def _ingest_event(self, decoded: DecodedTmPacket) -> TmIngestResult:
        assert decoded.message is not None
        message = decoded.message
        try:
            request_key = None
            if message.get("request_key") is not None:
                request_key = RequestKey.from_dict(message["request_key"])
            record = self.events.append(
                str(message.get("event_name", "MISSION_EVENT")),
                severity=str(message.get("severity", "INFO")),
                message=message.get("message", message),
                source_spacecraft_instance_id=decoded.envelope.source_spacecraft_instance_id,
                source_boot_id=decoded.envelope.sender_boot_id,
                request_key=request_key,
                dictionary_version=message.get("dictionary_version"),
            )
            if self.realtime is not None:
                self.realtime.publish(record)
            ack = None
            if self.outbox is not None and request_key is not None:
                stage = str(message.get("stage", "")).upper()
                reason_value = message.get("error_code")
                reason = None if reason_value is None else str(reason_value)
                success = reason is None and not stage.endswith("REJECTED") and stage not in {
                    "REJECTED",
                    "FAILED",
                }
                try:
                    ack = self.outbox.ingest_correlated_tm(
                        request_key,
                        source_spacecraft_instance_id=decoded.envelope.source_spacecraft_instance_id,
                        link_generation=decoded.envelope.link_generation,
                        link_session_id=decoded.envelope.link_session_id,
                        success=success,
                        reason=reason,
                        result=message,
                    )
                except KeyError:
                    # Satellite journal replay can legitimately outlive GDS
                    # command retention.  Preserve its event but never create
                    # command state from an unknown RequestKey.
                    self.metrics.inc(
                        "gds_tm_payload_errors_total",
                        labels={"code": "UNKNOWN_REQUEST_KEY"},
                    )
                if self.ledger is not None:
                    transfer_id = message.get("transfer_id")
                    if transfer_id is not None:
                        try:
                            self.ledger.update_product_downlink_transfer(
                                request_key,
                                transfer_id=int(transfer_id),
                                transfer_state=(
                                    "DISPATCHED" if success else "FAILED"
                                ),
                            )
                        except (TypeError, ValueError, LedgerIntegrityError):
                            self.metrics.inc(
                                "gds_tm_payload_errors_total",
                                labels={"code": "TRANSFER_ID_SCHEMA_ERROR"},
                            )
            return TmIngestResult(decoded, event=record, ack=ack)
        except (TypeError, ValueError) as exc:
            code = getattr(exc, "error_code", "EVENT_SCHEMA_ERROR")
            self.metrics.inc("gds_tm_payload_errors_total", labels={"code": code})
            return TmIngestResult(decoded, error_code=code)

    def _record_frame(self, decoded: DecodedTmPacket) -> None:
        envelope = decoded.envelope
        try:
            self.writer.mutate(
                "record_tm_link_frame",
                lambda connection: connection.execute(
                    "INSERT OR IGNORE INTO link_frames(simulation_run_id,direction,link_frame_id,copy_index,source_spacecraft_instance_id,target_spacecraft_instance_id,link_session_id,apid,vcid,frame_sequence,crc_valid,fault_json,segment_path,segment_offset,segment_length,received_at_us) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (encode_sqlite_u64(envelope.simulation_run_id), envelope.direction, encode_sqlite_u64(envelope.link_frame_id), envelope.copy_index, encode_sqlite_u64(envelope.source_spacecraft_instance_id), None, encode_sqlite_u64(envelope.link_session_id), decoded.apid, decoded.frame.virtual_channel_id, decoded.frame.virtual_channel_count, 1, json.dumps(dict(envelope.fault or {}), sort_keys=True, separators=(",", ":")), "inline://validated-tm", 0, len(envelope.frame_bytes), envelope.received_at_us),
                ),
                priority=MutationPriority.LOW,
            )
        except Exception as exc:
            self.metrics.inc(
                "gds_tm_frame_metadata_dropped_total",
                labels={"reason": type(exc).__name__},
            )

    def _record_invalid_frame(self, envelope: ValidatedTransportEnvelope, code: str) -> None:
        self.metrics.inc("gds_tm_invalid_frames_total", labels={"code": code})
        try:
            self.writer.mutate(
                "record_invalid_tm_frame",
                lambda connection: connection.execute(
                    "INSERT OR IGNORE INTO link_frames(simulation_run_id,direction,link_frame_id,copy_index,source_spacecraft_instance_id,target_spacecraft_instance_id,link_session_id,apid,vcid,frame_sequence,crc_valid,fault_json,segment_path,segment_offset,segment_length,received_at_us) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (encode_sqlite_u64(envelope.simulation_run_id), envelope.direction, encode_sqlite_u64(envelope.link_frame_id), envelope.copy_index, encode_sqlite_u64(envelope.source_spacecraft_instance_id), None, encode_sqlite_u64(envelope.link_session_id), None, None, None, 0, json.dumps({"decode_error": code}, sort_keys=True), "inline://invalid-tm", 0, len(envelope.frame_bytes), envelope.received_at_us),
                ),
                priority=MutationPriority.LOW,
            )
        except Exception:
            # Invalid UDP input must never recursively block the high-priority
            # writer path; the metric remains the durable observable.
            return
