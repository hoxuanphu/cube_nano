"""Durable onboard journal for command idempotency and restart reconciliation."""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from protocol.canonical import canonical_json, checked_u32, checked_u64, u64_to_bytes
from protocol.schemas import CommandOpcode, ConfigSnapshot, ProductRef, RequestKey


logger = logging.getLogger(__name__)


TERMINAL_JOB_STATES = frozenset({"SUCCEEDED", "FAILED", "TIMEOUT", "CANCELED"})


@dataclass(frozen=True)
class TmFrameCounters:
    """One durable allocation for a TM packet and its transfer-frame counts."""

    apid: int
    virtual_channel_id: int
    packet_sequence: int
    packet_sequence_epoch: int
    master_channel_count: int
    master_channel_epoch: int
    virtual_channel_count: int
    virtual_channel_epoch: int


def _json(value: Any) -> str:
    return canonical_json(value).decode("utf-8")


def _load_json(value: str | None, default: Any = None) -> Any:
    return default if value is None else json.loads(value)


class SatelliteJournal:
    """Single-writer SQLite journal with explicit transaction boundaries."""

    def __init__(self, path: str | Path, spacecraft_instance_id: int):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.spacecraft_instance_id = checked_u64(spacecraft_instance_id, "spacecraft_instance_id")
        self._lock = threading.RLock()
        self.connection = sqlite3.connect(self.path, timeout=5.0, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute("PRAGMA busy_timeout=5000")
        self._create_schema()
        self.boot_id = self._increment_boot_id()

    def _create_schema(self) -> None:
        with self.connection:
            self.connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS config (
                    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                    epoch INTEGER NOT NULL,
                    revision INTEGER NOT NULL,
                    model_threshold_bp INTEGER NOT NULL,
                    coverage_limit_bp INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS commands (
                    ground_instance_id BLOB NOT NULL,
                    request_id INTEGER NOT NULL,
                    opcode INTEGER NOT NULL,
                    digest TEXT NOT NULL,
                    state TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    result_json TEXT,
                    created_at_ns INTEGER NOT NULL,
                    PRIMARY KEY (ground_instance_id, request_id)
                );
                CREATE TABLE IF NOT EXISTS retired_request_ranges (
                    ground_instance_id BLOB NOT NULL,
                    start_request_id INTEGER NOT NULL,
                    end_request_id INTEGER NOT NULL,
                    PRIMARY KEY (ground_instance_id, start_request_id)
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    ground_instance_id BLOB NOT NULL,
                    request_id INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    scene_json TEXT NOT NULL,
                    roi_json TEXT,
                    config_json TEXT NOT NULL,
                    snapshot_json TEXT NOT NULL,
                    product_ref_json TEXT,
                    result_json TEXT,
                    error_code TEXT,
                    PRIMARY KEY (ground_instance_id, request_id)
                );
                CREATE TABLE IF NOT EXISTS products (
                    spacecraft_instance_id BLOB NOT NULL,
                    origin_boot_id INTEGER NOT NULL,
                    product_id INTEGER NOT NULL,
                    origin_ground_instance_id BLOB NOT NULL,
                    origin_request_id INTEGER NOT NULL,
                    product_type TEXT NOT NULL,
                    state TEXT NOT NULL,
                    path TEXT,
                    manifest_sha256 TEXT,
                    bundle_sha256 TEXT,
                    bundle_size INTEGER,
                    PRIMARY KEY (spacecraft_instance_id, origin_boot_id, product_id)
                );
                CREATE TABLE IF NOT EXISTS transfers (
                    transfer_id INTEGER PRIMARY KEY,
                    product_ref_json TEXT NOT NULL,
                    state TEXT NOT NULL,
                    attempt INTEGER NOT NULL,
                    error_code TEXT
                );
                CREATE TABLE IF NOT EXISTS events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_name TEXT NOT NULL,
                    request_key_json TEXT,
                    body_json TEXT NOT NULL,
                    created_at_ns INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tm_master_counter (
                    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                    next_count INTEGER NOT NULL CHECK(next_count BETWEEN 0 AND 256),
                    epoch INTEGER NOT NULL CHECK(epoch BETWEEN 0 AND 4294967295)
                );
                CREATE TABLE IF NOT EXISTS tm_virtual_counters (
                    virtual_channel_id INTEGER PRIMARY KEY CHECK(virtual_channel_id BETWEEN 0 AND 7),
                    next_count INTEGER NOT NULL CHECK(next_count BETWEEN 0 AND 256),
                    epoch INTEGER NOT NULL CHECK(epoch BETWEEN 0 AND 4294967295)
                );
                CREATE TABLE IF NOT EXISTS tm_packet_counters (
                    apid INTEGER PRIMARY KEY CHECK(apid BETWEEN 0 AND 2047),
                    next_sequence INTEGER NOT NULL CHECK(next_sequence BETWEEN 0 AND 16384),
                    epoch INTEGER NOT NULL CHECK(epoch BETWEEN 0 AND 4294967295)
                );
                """
            )
            self.connection.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES ('spacecraft_instance_id', ?)",
                (f"{self.spacecraft_instance_id:016x}",),
            )
            self.connection.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES ('boot_id', '0')"
            )
            self.connection.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES ('next_product_id', '1')"
            )
            self.connection.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES ('next_transfer_id', '1')"
            )
            self.connection.execute(
                "INSERT OR IGNORE INTO tm_master_counter(singleton,next_count,epoch) VALUES(1,0,0)"
            )
            self.connection.execute(
                "INSERT OR IGNORE INTO config(singleton, epoch, revision, model_threshold_bp, coverage_limit_bp) VALUES (1, 0, 0, 5000, 6000)"
            )

    def _increment_boot_id(self) -> int:
        with self._lock, self.connection:
            value = int(self.connection.execute("SELECT value FROM meta WHERE key='boot_id'").fetchone()[0])
            if value >= 0xFFFFFFFF:
                raise RuntimeError("spacecraft_boot_id exhausted; migrate spacecraft instance")
            value += 1
            self.connection.execute("UPDATE meta SET value=? WHERE key='boot_id'", (str(value),))
            return value

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                self.connection.execute("BEGIN IMMEDIATE")
                yield self.connection
                self.connection.commit()
            except Exception:
                self.connection.rollback()
                raise

    def close(self) -> None:
        with self._lock:
            self.connection.close()

    def current_config(self) -> ConfigSnapshot:
        row = self.connection.execute("SELECT epoch, revision, model_threshold_bp, coverage_limit_bp FROM config WHERE singleton=1").fetchone()
        return ConfigSnapshot(row[0], row[1], row[2], row[3])

    def lookup_request(self, request_key: RequestKey, digest: str) -> tuple[str, dict[str, Any] | None]:
        key = u64_to_bytes(request_key.ground_instance_id)
        row = self.connection.execute(
            "SELECT digest, state, result_json FROM commands WHERE ground_instance_id=? AND request_id=?",
            (key, request_key.request_id),
        ).fetchone()
        if row is not None:
            if row[0] != digest:
                return "CONFLICT", None
            return "DUPLICATE", _load_json(row[2])
        retired = self.connection.execute(
            "SELECT 1 FROM retired_request_ranges WHERE ground_instance_id=? AND start_request_id<=? AND end_request_id>=?",
            (key, request_key.request_id, request_key.request_id),
        ).fetchone()
        if retired is not None:
            return "RETIRED", None
        return "NEW", None

    def compact_request(self, request_key: RequestKey) -> None:
        """Delete a terminal full result and retain a durable retired marker."""
        key = u64_to_bytes(request_key.ground_instance_id)
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT state,opcode,result_json FROM commands WHERE ground_instance_id=? AND request_id=?",
                (key, request_key.request_id),
            ).fetchone()
            if row is None:
                raise ValueError("request is not in the full journal")
            if row[0] not in {"EXECUTED", "DISPATCHED", "COMMAND_REJECTED", "EXECUTION_FAILED"}:
                raise ValueError("only terminal command records may be compacted")
            if row[0] == "DISPATCHED":
                opcode = CommandOpcode(row[1])
                related_terminal = False
                if opcode in {CommandOpcode.SCENE_ANALYZE, CommandOpcode.ROI_REQUEST}:
                    job = connection.execute(
                        "SELECT state,product_ref_json FROM jobs WHERE ground_instance_id=? AND request_id=?",
                        (key, request_key.request_id),
                    ).fetchone()
                    if job is not None and job[0] in {"SUCCEEDED", "FAILED", "TIMEOUT", "CANCELED"}:
                        product_ref = ProductRef.from_dict(_load_json(job[1]))
                        product = connection.execute(
                            "SELECT state FROM products WHERE spacecraft_instance_id=? AND origin_boot_id=? AND product_id=?",
                            (
                                u64_to_bytes(product_ref.spacecraft_instance_id),
                                product_ref.origin_boot_id,
                                product_ref.product_id,
                            ),
                        ).fetchone()
                        related_terminal = product is not None and product[0] in {"READY", "FAILED"}
                elif opcode == CommandOpcode.PRODUCT_REQUEST_DOWNLINK:
                    result = _load_json(row[2], {})
                    transfer = connection.execute(
                        "SELECT state FROM transfers WHERE transfer_id=?",
                        (int(result.get("transfer_id", -1)),),
                    ).fetchone()
                    related_terminal = transfer is not None and transfer[0] in {
                        "SEND_COMPLETED",
                        "SEND_FAILED",
                        "CANCELED",
                    }
                if not related_terminal:
                    raise ValueError("dispatched request still has nonterminal related work")
            connection.execute(
                "DELETE FROM commands WHERE ground_instance_id=? AND request_id=?",
                (key, request_key.request_id),
            )
            ranges = connection.execute(
                "SELECT start_request_id,end_request_id FROM retired_request_ranges WHERE ground_instance_id=? AND end_request_id>=? AND start_request_id<=?",
                (key, max(0, request_key.request_id - 1), request_key.request_id + 1),
            ).fetchall()
            start = request_key.request_id
            end = request_key.request_id
            for existing_start, existing_end in ranges:
                start = min(start, existing_start)
                end = max(end, existing_end)
            connection.execute(
                "DELETE FROM retired_request_ranges WHERE ground_instance_id=? AND end_request_id>=? AND start_request_id<=?",
                (key, max(0, request_key.request_id - 1), request_key.request_id + 1),
            )
            connection.execute(
                "INSERT INTO retired_request_ranges(ground_instance_id,start_request_id,end_request_id) VALUES(?,?,?)",
                (key, start, end),
            )

    def record_command(
        self,
        request_key: RequestKey,
        opcode: int,
        digest: str,
        payload: dict[str, Any],
        state: str,
        result: dict[str, Any],
        created_at_ns: int = 0,
    ) -> None:
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO commands(ground_instance_id,request_id,opcode,digest,state,payload_json,result_json,created_at_ns) VALUES(?,?,?,?,?,?,?,?)",
                (u64_to_bytes(request_key.ground_instance_id), request_key.request_id, opcode, digest, state, _json(payload), _json(result), created_at_ns),
            )

    def update_command_result(self, request_key: RequestKey, state: str, result: dict[str, Any]) -> None:
        with self.transaction() as connection:
            connection.execute(
                "UPDATE commands SET state=?, result_json=? WHERE ground_instance_id=? AND request_id=?",
                (state, _json(result), u64_to_bytes(request_key.ground_instance_id), request_key.request_id),
            )

    def apply_config(self, expected_epoch: int, expected_revision: int, model_bp: int, coverage_bp: int) -> ConfigSnapshot:
        with self.transaction() as connection:
            row = connection.execute("SELECT epoch,revision FROM config WHERE singleton=1").fetchone()
            if (row[0], row[1]) != (expected_epoch, expected_revision):
                raise ValueError("CONFIG_REVISION_MISMATCH")
            if row[1] >= 0xFFFFFFFF:
                if row[0] >= 0xFFFFFFFF:
                    raise RuntimeError("config identity exhausted; migrate spacecraft instance")
                next_snapshot = (row[0] + 1, 0)
            else:
                next_snapshot = (row[0], row[1] + 1)
            connection.execute(
                "UPDATE config SET epoch=?,revision=?,model_threshold_bp=?,coverage_limit_bp=? WHERE singleton=1",
                (*next_snapshot, model_bp, coverage_bp),
            )
            return ConfigSnapshot(next_snapshot[0], next_snapshot[1], model_bp, coverage_bp)

    def apply_config_command(
        self,
        request_key: RequestKey,
        opcode: int,
        digest: str,
        payload: dict[str, Any],
        expected_epoch: int,
        expected_revision: int,
        model_bp: int,
        coverage_bp: int,
    ) -> tuple[ConfigSnapshot, dict[str, Any]]:
        """Commit config CAS, command journal and ACK snapshot together."""
        with self.transaction() as connection:
            row = connection.execute("SELECT epoch,revision FROM config WHERE singleton=1").fetchone()
            if (row[0], row[1]) != (expected_epoch, expected_revision):
                raise ValueError("CONFIG_REVISION_MISMATCH")
            if row[1] >= 0xFFFFFFFF:
                if row[0] >= 0xFFFFFFFF:
                    raise RuntimeError("config identity exhausted; migrate spacecraft instance")
                next_snapshot = (row[0] + 1, 0)
            else:
                next_snapshot = (row[0], row[1] + 1)
            connection.execute(
                "UPDATE config SET epoch=?,revision=?,model_threshold_bp=?,coverage_limit_bp=? WHERE singleton=1",
                (*next_snapshot, model_bp, coverage_bp),
            )
            snapshot = ConfigSnapshot(next_snapshot[0], next_snapshot[1], model_bp, coverage_bp)
            result = {"stage": "EXECUTED", "config_snapshot": snapshot.as_dict()}
            connection.execute(
                "INSERT INTO commands(ground_instance_id,request_id,opcode,digest,state,payload_json,result_json,created_at_ns) VALUES(?,?,?,?,?,?,?,?)",
                (u64_to_bytes(request_key.ground_instance_id), request_key.request_id, opcode, digest, "EXECUTED", _json(payload), _json(result), 0),
            )
            return snapshot, result

    def allocate_product_id(self) -> int:
        with self.transaction() as connection:
            value = int(connection.execute("SELECT value FROM meta WHERE key='next_product_id'").fetchone()[0])
            checked_u32(value, "product_id")
            if value == 0xFFFFFFFF:
                raise RuntimeError("product_id exhausted; migrate spacecraft instance")
            connection.execute("UPDATE meta SET value=? WHERE key='next_product_id'", (str(value + 1),))
            return value

    def allocate_transfer_id(self) -> int:
        with self.transaction() as connection:
            value = int(connection.execute("SELECT value FROM meta WHERE key='next_transfer_id'").fetchone()[0])
            checked_u32(value, "transfer_id")
            if value == 0xFFFFFFFF:
                raise RuntimeError("transfer_id exhausted; migrate spacecraft instance")
            connection.execute("UPDATE meta SET value=? WHERE key='next_transfer_id'", (str(value + 1),))
            return value

    @staticmethod
    def _validate_tm_apid(apid: int) -> int:
        if isinstance(apid, bool) or not isinstance(apid, int) or not 0 <= apid <= 0x7FF:
            raise ValueError("TM APID must be an integer in [0, 2047]")
        return apid

    @staticmethod
    def _validate_tm_vcid(virtual_channel_id: int) -> int:
        if (
            isinstance(virtual_channel_id, bool)
            or not isinstance(virtual_channel_id, int)
            or not 0 <= virtual_channel_id <= 7
        ):
            raise ValueError("TM virtual_channel_id must be an integer in [0, 7]")
        return virtual_channel_id

    @staticmethod
    def _consume_modular_counter(
        connection: sqlite3.Connection,
        *,
        table: str,
        where_sql: str,
        where_params: tuple[Any, ...],
        modulus: int,
        label: str,
    ) -> tuple[int, int]:
        row = connection.execute(
            f"SELECT next_count,epoch FROM {table} WHERE {where_sql}",
            where_params,
        ).fetchone()
        if row is None:
            raise RuntimeError(f"{label} counter row is missing")
        next_count = int(row[0])
        epoch = int(row[1])
        if next_count == modulus:
            if epoch >= 0xFFFFFFFF:
                raise RuntimeError(f"{label} epoch exhausted; migrate spacecraft instance")
            value = 0
            next_value = 1
            epoch += 1
        else:
            value = next_count
            next_value = value + 1
        connection.execute(
            f"UPDATE {table} SET next_count=?,epoch=? WHERE {where_sql}",
            (next_value, epoch, *where_params),
        )
        return value, epoch

    def allocate_tm_packet_sequence(self, apid: int) -> tuple[int, int]:
        """Allocate an APID-scoped Space Packet sequence without a TM frame."""

        normalized_apid = self._validate_tm_apid(apid)
        with self.transaction() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO tm_packet_counters(apid,next_sequence,epoch) VALUES(?,?,?)",
                (normalized_apid, 0, 0),
            )
            row = connection.execute(
                "SELECT next_sequence,epoch FROM tm_packet_counters WHERE apid=?",
                (normalized_apid,),
            ).fetchone()
            assert row is not None
            next_sequence = int(row[0])
            epoch = int(row[1])
            if next_sequence == 16_384:
                if epoch >= 0xFFFFFFFF:
                    raise RuntimeError("TM packet sequence epoch exhausted; migrate spacecraft instance")
                sequence = 0
                next_sequence = 1
                epoch += 1
            else:
                sequence = next_sequence
                next_sequence = sequence + 1
            connection.execute(
                "UPDATE tm_packet_counters SET next_sequence=?,epoch=? WHERE apid=?",
                (next_sequence, epoch, normalized_apid),
            )
            return sequence, epoch

    def allocate_tm_frame_counters(
        self,
        apid: int,
        *,
        virtual_channel_id: int = 0,
    ) -> TmFrameCounters:
        """Allocate packet, MCFC, and VCFC together before emitting a TM frame."""

        normalized_apid = self._validate_tm_apid(apid)
        vcid = self._validate_tm_vcid(virtual_channel_id)
        with self.transaction() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO tm_packet_counters(apid,next_sequence,epoch) VALUES(?,?,?)",
                (normalized_apid, 0, 0),
            )
            connection.execute(
                "INSERT OR IGNORE INTO tm_virtual_counters(virtual_channel_id,next_count,epoch) VALUES(?,?,?)",
                (vcid, 0, 0),
            )
            master, master_epoch = self._consume_modular_counter(
                connection,
                table="tm_master_counter",
                where_sql="singleton=1",
                where_params=(),
                modulus=256,
                label="TM master counter",
            )
            virtual, virtual_epoch = self._consume_modular_counter(
                connection,
                table="tm_virtual_counters",
                where_sql="virtual_channel_id=?",
                where_params=(vcid,),
                modulus=256,
                label="TM virtual counter",
            )
            row = connection.execute(
                "SELECT next_sequence,epoch FROM tm_packet_counters WHERE apid=?",
                (normalized_apid,),
            ).fetchone()
            assert row is not None
            next_sequence = int(row[0])
            packet_epoch = int(row[1])
            if next_sequence == 16_384:
                if packet_epoch >= 0xFFFFFFFF:
                    raise RuntimeError("TM packet sequence epoch exhausted; migrate spacecraft instance")
                packet_sequence = 0
                next_sequence = 1
                packet_epoch += 1
            else:
                packet_sequence = next_sequence
                next_sequence = packet_sequence + 1
            connection.execute(
                "UPDATE tm_packet_counters SET next_sequence=?,epoch=? WHERE apid=?",
                (next_sequence, packet_epoch, normalized_apid),
            )
            return TmFrameCounters(
                normalized_apid,
                vcid,
                packet_sequence,
                packet_epoch,
                master,
                master_epoch,
                virtual,
                virtual_epoch,
            )

    def tm_counter_state(self, apid: int, *, virtual_channel_id: int = 0) -> dict[str, int] | None:
        """Read durable TM allocation state for diagnostics and regression tests."""

        normalized_apid = self._validate_tm_apid(apid)
        vcid = self._validate_tm_vcid(virtual_channel_id)
        with self._lock:
            packet = self.connection.execute(
                "SELECT next_sequence,epoch FROM tm_packet_counters WHERE apid=?",
                (normalized_apid,),
            ).fetchone()
            virtual = self.connection.execute(
                "SELECT next_count,epoch FROM tm_virtual_counters WHERE virtual_channel_id=?",
                (vcid,),
            ).fetchone()
            master = self.connection.execute(
                "SELECT next_count,epoch FROM tm_master_counter WHERE singleton=1"
            ).fetchone()
        if packet is None or virtual is None or master is None:
            return None
        return {
            "next_packet_sequence": int(packet[0]),
            "packet_sequence_epoch": int(packet[1]),
            "next_virtual_channel_count": int(virtual[0]),
            "virtual_channel_epoch": int(virtual[1]),
            "next_master_channel_count": int(master[0]),
            "master_channel_epoch": int(master[1]),
        }

    def create_job(self, request_key: RequestKey, scene: dict[str, Any], roi: dict[str, Any] | None, snapshot: ConfigSnapshot, immutable_snapshot: dict[str, Any], product_ref: ProductRef) -> None:
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO jobs(ground_instance_id,request_id,state,scene_json,roi_json,config_json,snapshot_json,product_ref_json) VALUES(?,?,?,?,?,?,?,?)",
                (u64_to_bytes(request_key.ground_instance_id), request_key.request_id, "QUEUED", _json(scene), _json(roi) if roi is not None else None, _json(snapshot.as_dict()), _json(immutable_snapshot), _json(product_ref.as_dict())),
            )

    def admit_analysis(
        self,
        request_key: RequestKey,
        opcode: int,
        digest: str,
        payload: dict[str, Any],
        scene: dict[str, Any],
        roi: dict[str, Any] | None,
        snapshot: ConfigSnapshot,
        immutable_snapshot: dict[str, Any],
        product_ref: ProductRef,
    ) -> dict[str, Any]:
        """Atomically persist COMMAND_ACCEPTED, work row and product row."""
        result = {"stage": "DISPATCHED", "job_key": request_key.as_dict(), "product_ref": product_ref.as_dict()}
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO commands(ground_instance_id,request_id,opcode,digest,state,payload_json,result_json,created_at_ns) VALUES(?,?,?,?,?,?,?,?)",
                (u64_to_bytes(request_key.ground_instance_id), request_key.request_id, opcode, digest, "DISPATCHED", _json(payload), _json(result), 0),
            )
            connection.execute(
                "INSERT INTO products(spacecraft_instance_id,origin_boot_id,product_id,origin_ground_instance_id,origin_request_id,product_type,state) VALUES(?,?,?,?,?,?,?)",
                (u64_to_bytes(product_ref.spacecraft_instance_id), product_ref.origin_boot_id, product_ref.product_id, u64_to_bytes(request_key.ground_instance_id), request_key.request_id, "ANALYSIS", "STAGING"),
            )
            connection.execute(
                "INSERT INTO jobs(ground_instance_id,request_id,state,scene_json,roi_json,config_json,snapshot_json,product_ref_json) VALUES(?,?,?,?,?,?,?,?)",
                (u64_to_bytes(request_key.ground_instance_id), request_key.request_id, "QUEUED", _json(scene), _json(roi) if roi is not None else None, _json(snapshot.as_dict()), _json(immutable_snapshot), _json(product_ref.as_dict())),
            )
        return result

    def update_job(self, request_key: RequestKey, state: str, *, result: dict[str, Any] | None = None, error_code: str | None = None) -> None:
        with self.transaction() as connection:
            current = connection.execute(
                "SELECT state FROM jobs WHERE ground_instance_id=? AND request_id=?",
                (u64_to_bytes(request_key.ground_instance_id), request_key.request_id),
            ).fetchone()
            if current is None:
                raise ValueError("job not found")
            if str(current[0]) in TERMINAL_JOB_STATES and str(current[0]) != state:
                raise RuntimeError("JOB_TERMINAL_IMMUTABLE")
            connection.execute(
                "UPDATE jobs SET state=?, result_json=?, error_code=? WHERE ground_instance_id=? AND request_id=?",
                (state, _json(result) if result is not None else None, error_code, u64_to_bytes(request_key.ground_instance_id), request_key.request_id),
            )

    def transition_job(
        self,
        request_key: RequestKey,
        expected_states: set[str],
        state: str,
        *,
        result: dict[str, Any] | None = None,
        error_code: str | None = None,
    ) -> bool:
        if not expected_states:
            raise ValueError("expected job states must not be empty")
        if not isinstance(state, str) or not state:
            raise ValueError("job state must not be empty")
        placeholders = ",".join("?" for _ in expected_states)
        parameters = [
            state,
            _json(result) if result is not None else None,
            error_code,
            u64_to_bytes(request_key.ground_instance_id),
            request_key.request_id,
            *sorted(expected_states),
        ]
        with self.transaction() as connection:
            cursor = connection.execute(
                f"UPDATE jobs SET state=?, result_json=?, error_code=? WHERE ground_instance_id=? AND request_id=? AND state IN ({placeholders})",
                parameters,
            )
            return cursor.rowcount == 1

    def get_job(self, request_key: RequestKey) -> sqlite3.Row | None:
        return self.connection.execute(
            "SELECT * FROM jobs WHERE ground_instance_id=? AND request_id=?",
            (u64_to_bytes(request_key.ground_instance_id), request_key.request_id),
        ).fetchone()

    def create_product(self, product_ref: ProductRef, origin: RequestKey, product_type: str = "ANALYSIS") -> None:
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO products(spacecraft_instance_id,origin_boot_id,product_id,origin_ground_instance_id,origin_request_id,product_type,state) VALUES(?,?,?,?,?,?,?)",
                (u64_to_bytes(product_ref.spacecraft_instance_id), product_ref.origin_boot_id, product_ref.product_id, u64_to_bytes(origin.ground_instance_id), origin.request_id, product_type, "STAGING"),
            )

    def publish_product(self, product_ref: ProductRef, summary: dict[str, Any]) -> bool:
        with self.transaction() as connection:
            cursor = connection.execute(
                "UPDATE products SET state='READY',path=?,manifest_sha256=?,bundle_sha256=?,bundle_size=? WHERE spacecraft_instance_id=? AND origin_boot_id=? AND product_id=? AND state='STAGING'",
                (summary["product_directory"], summary["manifest_sha256"], summary["bundle_sha256"], summary["bundle_size"], u64_to_bytes(product_ref.spacecraft_instance_id), product_ref.origin_boot_id, product_ref.product_id),
            )
            return cursor.rowcount == 1

    def complete_job_with_product(
        self,
        request_key: RequestKey,
        expected_states: set[str],
        result: dict[str, Any],
        product_ref: ProductRef,
        summary: dict[str, Any],
    ) -> bool:
        if not expected_states:
            raise ValueError("expected job states must not be empty")
        placeholders = ",".join("?" for _ in expected_states)
        with self.transaction() as connection:
            job = connection.execute(
                f"UPDATE jobs SET state='SUCCEEDED',result_json=?,error_code=NULL WHERE ground_instance_id=? AND request_id=? AND state IN ({placeholders})",
                (
                    _json(result),
                    u64_to_bytes(request_key.ground_instance_id),
                    request_key.request_id,
                    *sorted(expected_states),
                ),
            )
            if job.rowcount != 1:
                return False
            product = connection.execute(
                "UPDATE products SET state='READY',path=?,manifest_sha256=?,bundle_sha256=?,bundle_size=? WHERE spacecraft_instance_id=? AND origin_boot_id=? AND product_id=? AND state='STAGING'",
                (
                    summary["product_directory"],
                    summary["manifest_sha256"],
                    summary["bundle_sha256"],
                    summary["bundle_size"],
                    u64_to_bytes(product_ref.spacecraft_instance_id),
                    product_ref.origin_boot_id,
                    product_ref.product_id,
                ),
            )
            if product.rowcount != 1:
                raise RuntimeError("product staging row missing during atomic publish")
            return True

    def fail_product_for_job(self, request_key: RequestKey, error_code: str) -> ProductRef | None:
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT product_ref_json FROM jobs WHERE ground_instance_id=? AND request_id=?",
                (u64_to_bytes(request_key.ground_instance_id), request_key.request_id),
            ).fetchone()
            if row is None or row[0] is None:
                return None
            product_ref = ProductRef.from_dict(_load_json(row[0]))
            connection.execute(
                "UPDATE products SET state='FAILED' WHERE spacecraft_instance_id=? AND origin_boot_id=? AND product_id=? AND state='STAGING'",
                (
                    u64_to_bytes(product_ref.spacecraft_instance_id),
                    product_ref.origin_boot_id,
                    product_ref.product_id,
                ),
            )
            return product_ref

    def get_product(self, product_ref: ProductRef) -> sqlite3.Row | None:
        return self.connection.execute(
            "SELECT * FROM products WHERE spacecraft_instance_id=? AND origin_boot_id=? AND product_id=?",
            (u64_to_bytes(product_ref.spacecraft_instance_id), product_ref.origin_boot_id, product_ref.product_id),
        ).fetchone()

    def create_transfer(self, transfer_id: int, product_ref: ProductRef) -> None:
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO transfers(transfer_id,product_ref_json,state,attempt) VALUES(?,?,?,?)",
                (transfer_id, _json(product_ref.as_dict()), "QUEUED", 1),
            )

    def get_transfer(self, transfer_id: int) -> sqlite3.Row | None:
        return self.connection.execute("SELECT * FROM transfers WHERE transfer_id=?", (transfer_id,)).fetchone()

    def get_active_transfer(self) -> sqlite3.Row | None:
        return self.connection.execute(
            "SELECT * FROM transfers WHERE state IN ('QUEUED','SENDING','CANCEL_REQUESTED','CANCEL_DRAINING','ABORTING','COOLDOWN') ORDER BY transfer_id LIMIT 1"
        ).fetchone()

    def update_transfer(self, transfer_id: int, state: str, error_code: str | None = None) -> None:
        with self.transaction() as connection:
            connection.execute("UPDATE transfers SET state=?,error_code=? WHERE transfer_id=?", (state, error_code, transfer_id))

    def admit_downlink(
        self,
        request_key: RequestKey,
        opcode: int,
        digest: str,
        payload: dict[str, Any],
        transfer_id: int,
        product_ref: ProductRef,
    ) -> dict[str, Any]:
        result = {"stage": "DISPATCHED", "transfer_id": transfer_id, "product_ref": product_ref.as_dict()}
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO commands(ground_instance_id,request_id,opcode,digest,state,payload_json,result_json,created_at_ns) VALUES(?,?,?,?,?,?,?,?)",
                (u64_to_bytes(request_key.ground_instance_id), request_key.request_id, opcode, digest, "DISPATCHED", _json(payload), _json(result), 0),
            )
            connection.execute(
                "INSERT INTO transfers(transfer_id,product_ref_json,state,attempt) VALUES(?,?,?,?)",
                (transfer_id, _json(product_ref.as_dict()), "QUEUED", 1),
            )
        return result

    def append_event(self, name: str, body: dict[str, Any], request_key: RequestKey | None = None, created_at_ns: int = 0) -> int:
        with self.transaction() as connection:
            cursor = connection.execute(
                "INSERT INTO events(event_name,request_key_json,body_json,created_at_ns) VALUES(?,?,?,?)",
                (name, _json(request_key.as_dict()) if request_key else None, _json(body), created_at_ns),
            )
            event_id = int(cursor.lastrowid)
        request_text = "" if request_key is None else f" request_key={_json(request_key.as_dict())}"
        logger.info(
            "event=%s event_id=%s%s body=%s",
            name,
            event_id,
            request_text,
            _event_log_body(body),
        )
        return event_id

    def events_after(self, event_id: int, *, limit: int = 128) -> tuple[sqlite3.Row, ...]:
        """Return immutable journal events for transport publication.

        The UDP satellite boundary is deliberately the only reader that turns
        these durable lifecycle records into TM events.  Keeping this query on
        the journal avoids a second in-memory event queue that could lose a
        worker completion during a process restart.
        """
        if isinstance(event_id, bool) or not isinstance(event_id, int) or event_id < 0:
            raise ValueError("event_id must be a non-negative integer")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 4096:
            raise ValueError("limit must be in [1, 4096]")
        with self._lock:
            rows = self.connection.execute(
                "SELECT event_id,event_name,request_key_json,body_json,created_at_ns "
                "FROM events WHERE event_id>? ORDER BY event_id LIMIT ?",
                (event_id, limit),
            ).fetchall()
        return tuple(rows)

    def reconcile_after_restart(self, staging_root: str | Path | None = None) -> list[str]:
        actions = []
        with self.transaction() as connection:
            broken = connection.execute(
                "SELECT ground_instance_id,request_id FROM commands WHERE state='COMMAND_ACCEPTED' AND NOT EXISTS (SELECT 1 FROM jobs WHERE jobs.ground_instance_id=commands.ground_instance_id AND jobs.request_id=commands.request_id)"
            ).fetchall()
            for _ in broken:
                actions.append("COMMAND_ACCEPTED_WITHOUT_WORK_ROW")
            connection.execute(
                "UPDATE jobs SET state='FAILED',error_code='RESTART_RECONCILIATION' WHERE state IN ('QUEUED','RUNNING','CANCEL_REQUESTED')"
            )
            connection.execute(
                "UPDATE products SET state='FAILED' WHERE state='STAGING'"
            )
            connection.execute(
                "UPDATE transfers SET state='SEND_FAILED',error_code='RESTART_RECONCILIATION' WHERE state IN ('QUEUED','SENDING','CANCEL_REQUESTED','CANCEL_DRAINING','ABORTING','COOLDOWN')"
            )
        if staging_root is not None:
            root = Path(staging_root)
            if root.exists():
                for path in root.glob("*.part"):
                    path.unlink(missing_ok=True)
        return actions


def _event_log_body(body: dict[str, Any]) -> str:
    """Keep console events useful while avoiding full product payload dumps."""
    visible_keys = (
        "stage",
        "state",
        "error_code",
        "transfer_id",
        "frame_count",
        "frame_length",
        "cancel_outcome",
        "science_decision",
        "science_status",
        "latency_ms",
        "job_key",
        "product_ref",
    )
    summary = {key: body[key] for key in visible_keys if key in body}
    if not summary and body:
        summary = body
    return _json(summary)
