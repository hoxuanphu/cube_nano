"""Frozen Phase 6 queue/watchdog/deadline SLO profile."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from protocol.canonical import canonical_json


@dataclass(frozen=True)
class SloProfile:
    schema_version: int
    config_revision: str
    benchmark_artifact_id: str
    benchmark_artifact_sha256: str
    oldest_ack_age_ms: int
    health_max_latency_ms: int
    file_min_goodput_bps: int
    deadline_safety_factor: int
    worker_heartbeat_interval_ms: int
    worker_heartbeat_timeout_ms: int
    max_worker_restarts: int
    restart_window_ms: int
    max_pending_jobs: int
    ack_mailbox_capacity: int
    control_queue_capacity: int
    file_queue_capacity: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SloProfile":
        if not isinstance(value, Mapping):
            raise ValueError("SLO profile must be an object")
        queues = value.get("queues", {})
        watchdog = value.get("watchdog", {})
        if not isinstance(queues, Mapping) or not isinstance(watchdog, Mapping):
            raise ValueError("SLO queues/watchdog must be objects")
        result = cls(
            int(value.get("schema_version", 0)),
            str(value.get("config_revision", "")),
            str(value.get("benchmark_artifact_id", "")),
            str(value.get("benchmark_artifact_sha256", "")),
            int(value.get("oldest_ack_age_ms", 0)),
            int(value.get("health_max_latency_ms", 0)),
            int(value.get("file_min_goodput_bps", 0)),
            int(value.get("deadline_safety_factor", 0)),
            int(watchdog.get("heartbeat_interval_ms", 0)),
            int(watchdog.get("heartbeat_timeout_ms", 0)),
            int(watchdog.get("max_restarts", 0)),
            int(watchdog.get("restart_window_ms", 0)),
            int(queues.get("max_pending_jobs", 0)),
            int(queues.get("ack_mailbox_capacity", 0)),
            int(queues.get("control_queue_capacity", 0)),
            int(queues.get("file_queue_capacity", 0)),
        )
        result.validate()
        return result

    @classmethod
    def from_file(cls, path: str | Path) -> "SloProfile":
        return cls.from_mapping(yaml.safe_load(Path(path).read_text(encoding="utf-8")))

    def validate(self) -> None:
        if self.schema_version != 1 or not self.config_revision:
            raise ValueError("SLO schema/config revision is invalid")
        if len(self.benchmark_artifact_sha256) != 64 or any(char not in "0123456789abcdef" for char in self.benchmark_artifact_sha256):
            raise ValueError("SLO benchmark artifact SHA-256 is invalid")
        if self.oldest_ack_age_ms != 1000 or self.health_max_latency_ms != 2000:
            raise ValueError("MVP ACK/health SLO must be 1000/2000 ms")
        if self.file_min_goodput_bps <= 0 or self.deadline_safety_factor <= 0:
            raise ValueError("file goodput and deadline factor must be positive")
        if (self.worker_heartbeat_interval_ms, self.worker_heartbeat_timeout_ms) != (1000, 5000):
            raise ValueError("worker watchdog must be 1000/5000 ms")
        if self.max_worker_restarts != 3 or self.restart_window_ms != 300000:
            raise ValueError("worker restart policy must be 3/300000 ms")
        if (self.max_pending_jobs, self.ack_mailbox_capacity, self.control_queue_capacity, self.file_queue_capacity) != (4, 32, 64, 16):
            raise ValueError("queue capacities must be 4/32/64/16")

    def digest(self) -> str:
        return hashlib.sha256(canonical_json(self.as_dict())).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "config_revision": self.config_revision,
            "benchmark_artifact_id": self.benchmark_artifact_id,
            "benchmark_artifact_sha256": self.benchmark_artifact_sha256,
            "oldest_ack_age_ms": self.oldest_ack_age_ms,
            "health_max_latency_ms": self.health_max_latency_ms,
            "file_min_goodput_bps": self.file_min_goodput_bps,
            "deadline_safety_factor": self.deadline_safety_factor,
            "watchdog": {
                "heartbeat_interval_ms": self.worker_heartbeat_interval_ms,
                "heartbeat_timeout_ms": self.worker_heartbeat_timeout_ms,
                "max_restarts": self.max_worker_restarts,
                "restart_window_ms": self.restart_window_ms,
            },
            "queues": {
                "max_pending_jobs": self.max_pending_jobs,
                "ack_mailbox_capacity": self.ack_mailbox_capacity,
                "control_queue_capacity": self.control_queue_capacity,
                "file_queue_capacity": self.file_queue_capacity,
            },
        }

    def validate_benchmark(self, artifact: Mapping[str, Any], artifact_sha256: str) -> None:
        if artifact.get("artifact_id") != self.benchmark_artifact_id:
            raise ValueError("SLO benchmark artifact ID mismatch")
        if artifact_sha256 != self.benchmark_artifact_sha256:
            raise ValueError("SLO benchmark artifact SHA mismatch")
        measurements = artifact.get("measurements", {})
        if float(measurements.get("p99_latency_ms", 0)) <= 0:
            raise ValueError("benchmark has no positive p99 latency")
