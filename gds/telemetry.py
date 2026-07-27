"""Telemetry deduplication and one-minute numeric rollups."""

from __future__ import annotations

import hashlib
import math
import sqlite3
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Sequence

from protocol.canonical import checked_u16, checked_u32, checked_u64, u64_to_json

from .audit import AuditStore, _json_value
from .idempotency import datetime_to_unix_us
from .writer import MutationPriority, SQLiteWriter
from .u64 import encode_sqlite_u64
from .u64 import decode_sqlite_u64


TELEMETRY_BUCKET_US = 60_000_000


@dataclass(frozen=True)
class TelemetrySample:
    source_spacecraft_instance_id: int
    source_boot_id: int
    simulation_run_id: int
    direction: str
    link_session_id: int
    link_frame_id: int
    copy_index: int
    sample_ordinal: int
    apid: int
    channel_id: int
    received_at_us: int
    raw_value: bytes
    decoded_value: Any
    satellite_time_us: int | None = None
    source_retention_revision: int = 0


@dataclass(frozen=True)
class TelemetryIngestResult:
    inserted: bool
    duplicate: bool
    key: tuple[int, int, str, int, int, int]


class TelemetryConflictError(RuntimeError):
    error_code = "TELEMETRY_DEDUPE_CONFLICT"

    def __init__(
        self,
        sample: TelemetrySample,
        *,
        existing_sha256: str,
        incoming_sha256: str,
    ) -> None:
        super().__init__("telemetry dedupe key already contains different bytes")
        self.sample = sample
        self.existing_sha256 = existing_sha256
        self.incoming_sha256 = incoming_sha256


class TelemetryStore:
    def __init__(
        self,
        writer: SQLiteWriter,
        *,
        audit: AuditStore | None = None,
    ) -> None:
        self.writer = writer
        self.audit = audit or AuditStore(writer)

    @staticmethod
    def _normalize(sample: TelemetrySample) -> TelemetrySample:
        if not isinstance(sample, TelemetrySample):
            raise TypeError("sample must be a TelemetrySample")
        direction = sample.direction.upper() if isinstance(sample.direction, str) else ""
        if direction not in {"UPLINK", "DOWNLINK"}:
            raise ValueError("telemetry direction must be UPLINK or DOWNLINK")
        if sample.received_at_us < 0 or sample.source_retention_revision < 0:
            raise ValueError("telemetry timestamps/revision must be non-negative")
        raw_value = bytes(sample.raw_value)
        if not raw_value:
            raise ValueError("telemetry raw_value must not be empty")
        if isinstance(sample.decoded_value, float) and not math.isfinite(sample.decoded_value):
            raise ValueError("telemetry decoded value must be finite")
        return TelemetrySample(
            checked_u64(sample.source_spacecraft_instance_id, "source_spacecraft_instance_id"),
            checked_u32(sample.source_boot_id, "source_boot_id"),
            checked_u64(sample.simulation_run_id, "simulation_run_id"),
            direction,
            checked_u64(sample.link_session_id, "link_session_id"),
            checked_u64(sample.link_frame_id, "link_frame_id"),
            checked_u32(sample.copy_index, "copy_index"),
            checked_u32(sample.sample_ordinal, "sample_ordinal"),
            checked_u16(sample.apid, "apid"),
            checked_u32(sample.channel_id, "channel_id"),
            int(sample.received_at_us),
            raw_value,
            sample.decoded_value,
            None if sample.satellite_time_us is None else int(sample.satellite_time_us),
            int(sample.source_retention_revision),
        )

    @staticmethod
    def _key(sample: TelemetrySample) -> tuple[int, int, str, int, int, int]:
        return (
            sample.source_spacecraft_instance_id,
            sample.simulation_run_id,
            sample.direction,
            sample.link_frame_id,
            sample.copy_index,
            sample.sample_ordinal,
        )

    @classmethod
    def _ingest_in_transaction(
        cls,
        connection: sqlite3.Connection,
        sample: TelemetrySample,
    ) -> TelemetryIngestResult:
        sample = cls._normalize(sample)
        instance_blob = encode_sqlite_u64(sample.source_spacecraft_instance_id)
        run_blob = encode_sqlite_u64(sample.simulation_run_id)
        frame_blob = encode_sqlite_u64(sample.link_frame_id)
        raw_hash = hashlib.sha256(sample.raw_value).digest()
        decoded_json = _json_value(sample.decoded_value)
        existing = connection.execute(
            "SELECT raw_value,payload_sha256,decoded_value_json FROM telemetry_samples "
            "WHERE source_spacecraft_instance_id=? AND simulation_run_id=? "
            "AND direction=? AND link_frame_id=? AND copy_index=? AND sample_ordinal=?",
            (
                instance_blob,
                run_blob,
                sample.direction,
                frame_blob,
                sample.copy_index,
                sample.sample_ordinal,
            ),
        ).fetchone()
        if existing is not None:
            if (
                bytes(existing[0]) == sample.raw_value
                and bytes(existing[1]) == raw_hash
                and str(existing[2]) == decoded_json
            ):
                return TelemetryIngestResult(False, True, cls._key(sample))
            raise TelemetryConflictError(
                sample,
                existing_sha256=bytes(existing[1]).hex(),
                incoming_sha256=raw_hash.hex(),
            )

        connection.execute(
            "INSERT INTO telemetry_samples("
            "source_spacecraft_instance_id,source_boot_id,simulation_run_id,direction,"
            "link_session_id,link_frame_id,copy_index,sample_ordinal,apid,channel_id,"
            "satellite_time_us,received_at_us,raw_value,decoded_value_json,payload_sha256) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                instance_blob,
                sample.source_boot_id,
                run_blob,
                sample.direction,
                encode_sqlite_u64(sample.link_session_id),
                frame_blob,
                sample.copy_index,
                sample.sample_ordinal,
                sample.apid,
                sample.channel_id,
                sample.satellite_time_us,
                sample.received_at_us,
                sample.raw_value,
                decoded_json,
                raw_hash,
            ),
        )
        bucket = (sample.received_at_us // TELEMETRY_BUCKET_US) * TELEMETRY_BUCKET_US
        numeric = (
            float(sample.decoded_value)
            if isinstance(sample.decoded_value, (int, float))
            and not isinstance(sample.decoded_value, bool)
            else None
        )
        rollup = connection.execute(
            "SELECT sample_count,min_value,max_value,mean_value "
            "FROM telemetry_rollups WHERE source_spacecraft_instance_id=? "
            "AND channel_id=? AND bucket_start_us=?",
            (instance_blob, sample.channel_id, bucket),
        ).fetchone()
        if rollup is None:
            count = 1
            min_value = numeric
            max_value = numeric
            mean_value = numeric
            connection.execute(
                "INSERT INTO telemetry_rollups("
                "source_spacecraft_instance_id,channel_id,bucket_start_us,sample_count,"
                "min_value,max_value,mean_value,last_value_json,source_retention_revision) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    instance_blob,
                    sample.channel_id,
                    bucket,
                    count,
                    min_value,
                    max_value,
                    mean_value,
                    decoded_json,
                    sample.source_retention_revision,
                ),
            )
        else:
            old_count = int(rollup[0])
            count = old_count + 1
            min_value = rollup[1]
            max_value = rollup[2]
            mean_value = rollup[3]
            if numeric is not None:
                min_value = numeric if min_value is None else min(float(min_value), numeric)
                max_value = numeric if max_value is None else max(float(max_value), numeric)
                mean_value = numeric if mean_value is None else (
                    (float(mean_value) * old_count + numeric) / count
                )
            connection.execute(
                "UPDATE telemetry_rollups SET sample_count=?,min_value=?,max_value=?,"
                "mean_value=?,last_value_json=?,source_retention_revision=MAX("
                "source_retention_revision,?) WHERE source_spacecraft_instance_id=? "
                "AND channel_id=? AND bucket_start_us=?",
                (
                    count,
                    min_value,
                    max_value,
                    mean_value,
                    decoded_json,
                    sample.source_retention_revision,
                    instance_blob,
                    sample.channel_id,
                    bucket,
                ),
            )
        return TelemetryIngestResult(True, False, cls._key(sample))

    def ingest(self, sample: TelemetrySample | None = None, **fields: Any) -> TelemetryIngestResult:
        if sample is None:
            if "received_at" in fields and "received_at_us" not in fields:
                value = fields.pop("received_at")
                if not isinstance(value, datetime) or value.tzinfo is None:
                    raise ValueError("received_at must be timezone-aware")
                fields["received_at_us"] = datetime_to_unix_us(value)
            sample = TelemetrySample(**fields)
        try:
            return self.writer.mutate(
                "ingest_telemetry_sample",
                lambda connection: self._ingest_in_transaction(connection, sample),
                priority=MutationPriority.LOW,
            )
        except TelemetryConflictError as exc:
            self.audit.append(
                principal="telemetry",
                action="TELEMETRY_DEDUPE_CONFLICT",
                target_type="telemetry_sample",
                target_identity={
                    "source_spacecraft_instance_id": u64_to_json(
                        exc.sample.source_spacecraft_instance_id
                    ),
                    "simulation_run_id": u64_to_json(exc.sample.simulation_run_id),
                    "direction": exc.sample.direction,
                    "link_frame_id": u64_to_json(exc.sample.link_frame_id),
                    "copy_index": exc.sample.copy_index,
                    "sample_ordinal": exc.sample.sample_ordinal,
                },
                new_value={
                    "existing_sha256": exc.existing_sha256,
                    "incoming_sha256": exc.incoming_sha256,
                },
            )
            raise

    def ingest_batch(self, samples: Sequence[TelemetrySample]) -> tuple[TelemetryIngestResult, ...]:
        if not 1 <= len(samples) <= 100:
            raise ValueError("telemetry batch must contain 1..100 samples")
        normalized = tuple(self._normalize(sample) for sample in samples)
        try:
            return self.writer.mutate(
                "ingest_telemetry_batch",
                lambda connection: tuple(
                    self._ingest_in_transaction(connection, item) for item in normalized
                ),
                priority=MutationPriority.LOW,
            )
        except TelemetryConflictError as exc:
            self.audit.append(
                principal="telemetry",
                action="TELEMETRY_DEDUPE_CONFLICT",
                target_type="telemetry_sample",
                target_identity={"key": list(self._key(exc.sample))},
                new_value={
                    "existing_sha256": exc.existing_sha256,
                    "incoming_sha256": exc.incoming_sha256,
                },
            )
            raise

    def latest_for_instance(self, spacecraft_instance_id: int, *, limit: int = 100) -> tuple[dict[str, Any], ...]:
        """Read the bounded latest telemetry window for API snapshots."""
        checked_u64(spacecraft_instance_id, "spacecraft_instance_id")
        if isinstance(limit, bool) or not 1 <= limit <= 1_000:
            raise ValueError("limit must be in [1, 1000]")
        with self.writer.reader() as connection:
            rows = connection.execute(
                "SELECT source_spacecraft_instance_id,source_boot_id,simulation_run_id,direction,"
                "link_session_id,link_frame_id,copy_index,sample_ordinal,apid,channel_id,"
                "satellite_time_us,received_at_us,decoded_value_json FROM telemetry_samples "
                "WHERE source_spacecraft_instance_id=? ORDER BY received_at_us DESC,link_frame_id DESC LIMIT ?",
                (encode_sqlite_u64(spacecraft_instance_id), limit),
            ).fetchall()
        return tuple(
            {
                "source_spacecraft_instance_id": f"{decode_sqlite_u64(row[0]):016x}",
                "source_boot_id": int(row[1]),
                "simulation_run_id": f"{decode_sqlite_u64(row[2]):016x}",
                "direction": str(row[3]),
                "link_session_id": f"{decode_sqlite_u64(row[4]):016x}",
                "link_frame_id": f"{decode_sqlite_u64(row[5]):016x}",
                "copy_index": int(row[6]),
                "sample_ordinal": int(row[7]),
                "apid": int(row[8]),
                "channel_id": int(row[9]),
                "satellite_time_us": row[10],
                "received_at_us": int(row[11]),
                "value": json.loads(str(row[12])),
            }
            for row in rows
        )
