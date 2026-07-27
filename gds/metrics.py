"""Low-cardinality metrics and structured JSON logging for local SIL."""

from __future__ import annotations

import json
import logging
import logging.handlers
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from protocol.canonical import u64_to_json


class MetricsRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
        self._gauges: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
        self._observations: dict[tuple[str, tuple[tuple[str, str], ...]], list[float]] = {}

    @staticmethod
    def _key(name: str, labels: Mapping[str, Any] | None) -> tuple[str, tuple[tuple[str, str], ...]]:
        if not name or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_:" for char in name):
            raise ValueError("metric name contains invalid characters")
        values = tuple(sorted((str(key), str(value)) for key, value in (labels or {}).items()))
        return name, values

    def inc(self, name: str, value: float = 1, *, labels: Mapping[str, Any] | None = None) -> None:
        if value < 0:
            raise ValueError("counter increments must be non-negative")
        with self._lock:
            key = self._key(name, labels)
            self._counters[key] = self._counters.get(key, 0) + value

    def set(self, name: str, value: float, *, labels: Mapping[str, Any] | None = None) -> None:
        with self._lock:
            self._gauges[self._key(name, labels)] = float(value)

    def observe(self, name: str, value: float, *, labels: Mapping[str, Any] | None = None) -> None:
        with self._lock:
            key = self._key(name, labels)
            values = self._observations.setdefault(key, [])
            values.append(float(value))
            if len(values) > 10_000:
                del values[: len(values) - 10_000]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            result: dict[str, Any] = {}
            for source, values in (("counter", self._counters), ("gauge", self._gauges)):
                for (name, labels), value in values.items():
                    key = _metric_key(name, labels)
                    result.setdefault(key, {})[source] = value
            for (name, labels), values in self._observations.items():
                if values:
                    result.setdefault(_metric_key(name, labels), {})["count"] = len(values)
                    result[_metric_key(name, labels)]["last"] = values[-1]
                    result[_metric_key(name, labels)]["p95"] = sorted(values)[max(0, int(len(values) * 0.95) - 1)]
            return result

    def prometheus(self) -> str:
        lines = []
        for key, values in self.snapshot().items():
            name = key.split("{", 1)[0]
            labels = key[len(name) :] if "{" in key else ""
            for field, value in values.items():
                metric = name if field in {"counter", "gauge"} else f"{name}_{field}"
                lines.append(f"{metric}{labels} {value}")
        return "\n".join(lines) + ("\n" if lines else "")


def _metric_key(name: str, labels: tuple[tuple[str, str], ...]) -> str:
    if not labels:
        return name
    encoded = ",".join(f'{key}="{value.replace(chr(34), chr(92) + chr(34))}"' for key, value in labels)
    return f"{name}{{{encoded}}}"


@dataclass(frozen=True)
class HealthSnapshot:
    status: str
    checks: dict[str, Any]
    generated_at_us: int

    def as_dict(self) -> dict[str, Any]:
        return {"status": self.status, "checks": self.checks, "generated_at_us": self.generated_at_us}


class HealthService:
    def __init__(self, *, writer: Any | None = None, link: Any | None = None, decoder: Any | None = None, worker: Any | None = None, clock: Any | None = None):
        self.writer = writer
        self.link = link
        self.decoder = decoder
        self.worker = worker
        self.clock = clock or (lambda: int(time.time_ns() // 1_000))

    def healthz(self) -> HealthSnapshot:
        return HealthSnapshot("ok", {"process": {"status": "ok"}}, int(self.clock()))

    def readyz(self) -> HealthSnapshot:
        checks: dict[str, Any] = {}
        ready = True
        if self.writer is not None:
            try:
                wal = self.writer.wal_status()
                checks["database"] = {"ready": True, "wal_level": wal.level.value, "queue_depth": self.writer.queue_depth}
            except Exception as exc:
                checks["database"] = {"ready": False, "error": type(exc).__name__}
                ready = False
        for name, dependency in (("link", self.link), ("decoder", self.decoder), ("worker", self.worker)):
            if dependency is None:
                continue
            value = _dependency_ready(dependency)
            checks[name] = {"ready": value}
            if not value:
                ready = False
        # Scheduled blackout is a transport state, not a service fault.
        if self.link is not None and getattr(self.link, "scheduled_blackout", False):
            checks["link"]["scheduled_blackout"] = True
            if checks["link"]["ready"] is False and getattr(self.link, "process_healthy", True):
                checks["link"]["ready"] = True
                ready = all(value.get("ready", True) for name, value in checks.items() if name != "link")
        return HealthSnapshot("ok" if ready else "fail", checks, int(self.clock()))


def _dependency_ready(dependency: Any) -> bool:
    if hasattr(dependency, "ready"):
        value = dependency.ready
        return bool(value() if callable(value) else value)
    if hasattr(dependency, "is_ready"):
        value = dependency.is_ready
        return bool(value() if callable(value) else value)
    if hasattr(dependency, "health"):
        health = dependency.health()
        return str(health.get("state", health.get("status", ""))).upper() in {"READY", "OK", "HEALTHY"}
    return True


class StructuredJsonLogger:
    """JSON logger with bounded rotation and token/idempotency redaction."""

    def __init__(self, path: str | Path, *, max_bytes: int = 64 * 1024 * 1024, backup_count: int = 5):
        self.logger = logging.getLogger(f"gds.structured.{Path(path).resolve()}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        if not self.logger.handlers:
            handler = logging.handlers.RotatingFileHandler(path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(message)s"))
            self.logger.addHandler(handler)

    def log(self, level: int, event: str, **fields: Any) -> None:
        payload = {"event": event, "time_ns": time.time_ns(), **_redact(fields)}
        self.logger.log(level, json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")))


def _redact(value: Any) -> Any:
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            lowered = str(key).lower()
            result[str(key)] = "[REDACTED]" if any(token in lowered for token in ("token", "secret", "authorization", "idempotency-key")) else _redact(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    return value
