"""Mission profile loader and fail-closed validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .canonical import canonical_json, checked_u32, checked_u64


@dataclass(frozen=True)
class MissionProfile:
    schema_version: int
    profile_id: str
    fprime_version: str
    spacecraft_id: int
    spacecraft_instance_id: int
    tm_frame_size: int
    tc_virtual_channel: int
    tm_virtual_channel: int
    byte_order: str
    tc_apid: int
    tm_telem_apid: int
    tm_event_apid: int
    tm_file_apid: int
    tm_ocf_present: bool
    tm_fecf_present: bool
    tm_secondary_header_present: bool
    fw_com_buffer_max_size: int
    fw_file_buffer_max_size: int
    packet_descriptor_size: int
    max_file_data_per_frame: int
    time_base: str
    time_epoch: str
    time_resolution_ns: int
    max_pending_jobs: int
    max_in_flight_packets: int
    ack_mailbox_capacity: int
    control_queue_capacity: int
    file_queue_capacity: int
    ack_burst: int
    control_burst: int
    file_burst: int
    worker_heartbeat_interval_ms: int
    worker_heartbeat_timeout_ms: int
    max_worker_restarts: int
    worker_restart_window_ms: int
    worker_initial_backoff_ms: int
    worker_cancel_grace_ms: int

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "MissionProfile":
        if not isinstance(value, dict):
            raise ValueError("mission profile must be a YAML object")
        required = {
            "schema_version",
            "profile_id",
            "fprime_version",
            "spacecraft_id",
            "spacecraft_instance_id",
            "tm_frame_size",
            "tc_virtual_channel",
            "tm_virtual_channel",
            "byte_order",
            "apids",
            "tm",
            "buffer_budget",
            "time",
            "scheduler",
        }
        missing = sorted(required - set(value))
        if missing:
            raise ValueError(f"mission profile missing fields: {', '.join(missing)}")
        apids = value["apids"]
        tm = value["tm"]
        buffers = value["buffer_budget"]
        time = value["time"]
        scheduler = value["scheduler"]
        if not all(isinstance(item, dict) for item in (apids, tm, buffers, time, scheduler)):
            raise ValueError("mission profile nested fields must be objects")
        result = cls(
            schema_version=int(value["schema_version"]),
            profile_id=str(value["profile_id"]),
            fprime_version=str(value["fprime_version"]),
            spacecraft_id=int(value["spacecraft_id"]),
            spacecraft_instance_id=int(str(value["spacecraft_instance_id"]), 16),
            tm_frame_size=int(value["tm_frame_size"]),
            tc_virtual_channel=int(value["tc_virtual_channel"]),
            tm_virtual_channel=int(value["tm_virtual_channel"]),
            byte_order=str(value["byte_order"]),
            tc_apid=int(apids["tc_command"]),
            tm_telem_apid=int(apids["tm_telemetry"]),
            tm_event_apid=int(apids["tm_event"]),
            tm_file_apid=int(apids["tm_file"]),
            tm_ocf_present=bool(tm["ocf_present"]),
            tm_fecf_present=bool(tm["fecf_present"]),
            tm_secondary_header_present=bool(tm["secondary_header_present"]),
            fw_com_buffer_max_size=int(buffers["fw_com_buffer_max_size"]),
            fw_file_buffer_max_size=int(buffers["fw_file_buffer_max_size"]),
            packet_descriptor_size=int(buffers["packet_descriptor_size"]),
            max_file_data_per_frame=int(buffers["max_file_data_per_frame"]),
            time_base=str(time["base"]),
            time_epoch=str(time["epoch"]),
            time_resolution_ns=int(time["resolution_ns"]),
            max_pending_jobs=int(scheduler["max_pending_jobs"]),
            max_in_flight_packets=int(scheduler["max_in_flight_packets"]),
            ack_mailbox_capacity=int(scheduler["ack_mailbox_capacity"]),
            control_queue_capacity=int(scheduler["control_queue_capacity"]),
            file_queue_capacity=int(scheduler["file_queue_capacity"]),
            ack_burst=int(scheduler["ack_burst"]),
            control_burst=int(scheduler["control_burst"]),
            file_burst=int(scheduler["file_burst"]),
            worker_heartbeat_interval_ms=int(scheduler["worker_heartbeat_interval_ms"]),
            worker_heartbeat_timeout_ms=int(scheduler["worker_heartbeat_timeout_ms"]),
            max_worker_restarts=int(scheduler["max_worker_restarts"]),
            worker_restart_window_ms=int(scheduler["worker_restart_window_ms"]),
            worker_initial_backoff_ms=int(scheduler["worker_initial_backoff_ms"]),
            worker_cancel_grace_ms=int(scheduler["worker_cancel_grace_ms"]),
        )
        result.validate()
        return result

    @classmethod
    def from_file(cls, path: str | Path) -> "MissionProfile":
        path = Path(path)
        try:
            value = yaml.safe_load(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise FileNotFoundError(f"mission profile not found: {path}") from None
        except yaml.YAMLError as exc:
            raise ValueError(f"invalid mission profile YAML: {exc}") from exc
        return cls.from_mapping(value)

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError("mission profile schema_version must be 1")
        if not self.profile_id or self.fprime_version != "v4.1.0":
            raise ValueError("MVP requires a named F Prime v4.1.0 profile")
        checked_u32(self.spacecraft_id, "spacecraft_id")
        checked_u64(self.spacecraft_instance_id, "spacecraft_instance_id")
        if self.spacecraft_id > 0x3FF:
            raise ValueError("CCSDS spacecraft_id must fit 10 bits")
        if self.tm_frame_size != 1024:
            raise ValueError("MVP TM frame size is fixed at 1024 bytes")
        if self.byte_order != "big":
            raise ValueError("MVP wire byte order is big-endian")
        if (self.tc_apid, self.tm_telem_apid, self.tm_event_apid, self.tm_file_apid) != (0, 1, 2, 3):
            raise ValueError("MVP uses stock F Prime APID mapping 0/1/2/3")
        if self.tc_virtual_channel != 0 or self.tm_virtual_channel != 0:
            raise ValueError("MVP uses VC0 for TC and TM")
        if self.tm_ocf_present or self.tm_secondary_header_present or not self.tm_fecf_present:
            raise ValueError("MVP requires OCF/secondary-header absent and TM FECF present")
        if self.fw_com_buffer_max_size != 512 or self.fw_file_buffer_max_size != 1003:
            raise ValueError("MVP buffer budget must be 512/1003 bytes")
        if self.packet_descriptor_size != 2 or self.max_file_data_per_frame != 990:
            raise ValueError("MVP FilePacket boundary must be descriptor=2 and DATA=990")
        if self.time_base != "tai" or self.time_resolution_ns != 1:
            raise ValueError("MVP time contract is integer TAI nanoseconds")
        if self.max_pending_jobs != 4 or self.max_in_flight_packets != 1:
            raise ValueError("scheduler must be bounded and single-in-flight")
        if (self.ack_mailbox_capacity, self.control_queue_capacity, self.file_queue_capacity) != (32, 64, 16):
            raise ValueError("MVP scheduler capacities must be 32/64/16")
        if min(self.ack_burst, self.control_burst, self.file_burst) <= 0:
            raise ValueError("scheduler burst values must be positive")
        if (self.ack_burst, self.control_burst, self.file_burst) != (8, 4, 8):
            raise ValueError("MVP scheduler bursts must be 8/4/8")
        if (self.worker_heartbeat_interval_ms, self.worker_heartbeat_timeout_ms) != (1000, 5000):
            raise ValueError("MVP worker heartbeat interval/timeout must be 1000/5000 ms")
        if self.max_worker_restarts != 3 or self.worker_restart_window_ms != 300000:
            raise ValueError("MVP worker restart policy must be 3 per 300000 ms")
        if self.worker_initial_backoff_ms != 250 or self.worker_cancel_grace_ms != 1000:
            raise ValueError("MVP worker backoff/cancel grace must be 250/1000 ms")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "fprime_version": self.fprime_version,
            "spacecraft_id": self.spacecraft_id,
            "spacecraft_instance_id": f"{self.spacecraft_instance_id:016x}",
            "tm_frame_size": self.tm_frame_size,
            "tc_virtual_channel": self.tc_virtual_channel,
            "tm_virtual_channel": self.tm_virtual_channel,
            "byte_order": self.byte_order,
            "apids": {
                "tc_command": self.tc_apid,
                "tm_telemetry": self.tm_telem_apid,
                "tm_event": self.tm_event_apid,
                "tm_file": self.tm_file_apid,
            },
            "tm": {
                "ocf_present": self.tm_ocf_present,
                "fecf_present": self.tm_fecf_present,
                "secondary_header_present": self.tm_secondary_header_present,
            },
            "buffer_budget": {
                "fw_com_buffer_max_size": self.fw_com_buffer_max_size,
                "fw_file_buffer_max_size": self.fw_file_buffer_max_size,
                "packet_descriptor_size": self.packet_descriptor_size,
                "max_file_data_per_frame": self.max_file_data_per_frame,
            },
            "time": {
                "base": self.time_base,
                "epoch": self.time_epoch,
                "resolution_ns": self.time_resolution_ns,
            },
            "scheduler": {
                "max_pending_jobs": self.max_pending_jobs,
                "max_in_flight_packets": self.max_in_flight_packets,
                "ack_mailbox_capacity": self.ack_mailbox_capacity,
                "control_queue_capacity": self.control_queue_capacity,
                "file_queue_capacity": self.file_queue_capacity,
                "ack_burst": self.ack_burst,
                "control_burst": self.control_burst,
                "file_burst": self.file_burst,
                "worker_heartbeat_interval_ms": self.worker_heartbeat_interval_ms,
                "worker_heartbeat_timeout_ms": self.worker_heartbeat_timeout_ms,
                "max_worker_restarts": self.max_worker_restarts,
                "worker_restart_window_ms": self.worker_restart_window_ms,
                "worker_initial_backoff_ms": self.worker_initial_backoff_ms,
                "worker_cancel_grace_ms": self.worker_cancel_grace_ms,
            },
        }

    def digest(self) -> str:
        return hashlib.sha256(canonical_json(self.as_dict())).hexdigest()


def load_fprime_constants(dictionary_path: str | Path) -> dict[str, int]:
    payload = json.loads(Path(dictionary_path).read_text(encoding="utf-8"))
    metadata = payload.get("metadata", {})
    if metadata.get("frameworkVersion") != "v4.1.0":
        raise ValueError("F Prime dictionary is not pinned to v4.1.0")
    result = {}
    for item in payload.get("constants", []):
        name = item.get("qualifiedName")
        if name in {"ComCfg.SpacecraftId", "ComCfg.TmFrameFixedSize"}:
            result[name] = int(item["value"])
    if result != {"ComCfg.SpacecraftId": 68, "ComCfg.TmFrameFixedSize": 1024}:
        raise ValueError("F Prime dictionary constants do not match MVP profile")
    return result
