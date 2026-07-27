"""Fail-closed local-SIL topology and request-limit checks."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


class TopologyError(ValueError):
    def __init__(self, code: str, message: str, status_code: int = 400):
        super().__init__(message)
        self.code = code
        self.status_code = status_code


@dataclass(frozen=True)
class RequestLimits:
    request_body_bytes: int
    header_bytes: int
    requests_per_minute: int
    download_bytes: int
    extract_bytes: int


@dataclass(frozen=True)
class TopologyProfile:
    schema_version: int
    profile_id: str
    topology: str
    bind_host: str
    allowed_peers: tuple[str, ...]
    allowed_origins: tuple[str, ...]
    network_exposure: bool
    limits: RequestLimits

    @classmethod
    def from_file(cls, path: str | Path) -> "TopologyProfile":
        value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise TopologyError("TOPOLOGY_SCHEMA_ERROR", "runtime profile must be an object")
        limits = value.get("limits", {})
        if not isinstance(limits, Mapping):
            raise TopologyError("TOPOLOGY_SCHEMA_ERROR", "runtime limits must be an object")
        result = cls(
            int(value.get("schema_version", 0)),
            str(value.get("profile_id", "")),
            str(value.get("topology", "")),
            str(value.get("bind_host", "")),
            tuple(str(item) for item in value.get("allowed_peers", ())),
            tuple(str(item) for item in value.get("allowed_origins", ())),
            bool(value.get("network_exposure", True)),
            RequestLimits(*(int(limits.get(key, 0)) for key in ("request_body_bytes", "header_bytes", "requests_per_minute", "download_bytes", "extract_bytes"))),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.schema_version != 1 or self.topology not in {"host_local_sil", "compose_sil"}:
            raise TopologyError("TOPOLOGY_SCHEMA_ERROR", "unsupported local-SIL topology")
        public_bind = self.bind_host in {"0.0.0.0", "::", ""}
        if self.network_exposure or (public_bind and self.topology == "host_local_sil"):
            raise TopologyError("PUBLIC_BIND_FORBIDDEN", "local-SIL profile must not expose a public bind")
        if not self.allowed_peers or not self.allowed_origins:
            raise TopologyError("TOPOLOGY_SCHEMA_ERROR", "peer and origin allowlists are required")
        if min(self.limits.request_body_bytes, self.limits.header_bytes, self.limits.requests_per_minute, self.limits.download_bytes, self.limits.extract_bytes) <= 0:
            raise TopologyError("TOPOLOGY_SCHEMA_ERROR", "all request limits must be positive")

    def validate_startup(self, bind_host: str | None = None) -> None:
        self.validate()
        if bind_host is not None and bind_host != self.bind_host:
            raise TopologyError("PUBLIC_BIND_FORBIDDEN", "runtime bind host differs from the pinned profile")

    def validate_request(self, *, host: str, origin: str | None, peer: str, body_bytes: int, header_bytes: int, method: str = "GET") -> None:
        if host not in self.allowed_peers and host.split(":", 1)[0] not in self.allowed_peers:
            raise TopologyError("HOST_FORBIDDEN", "Host is outside the local allowlist", 403)
        if peer not in self.allowed_peers:
            raise TopologyError("PEER_FORBIDDEN", "peer is outside the local allowlist", 403)
        if method.upper() in {"POST", "PUT", "PATCH", "DELETE"}:
            if origin is None or origin not in self.allowed_origins:
                raise TopologyError("ORIGIN_FORBIDDEN", "unsafe request requires an allowed Origin", 403)
        elif origin is not None and origin not in self.allowed_origins:
            raise TopologyError("ORIGIN_FORBIDDEN", "Origin is outside the local allowlist", 403)
        if body_bytes < 0 or body_bytes > self.limits.request_body_bytes:
            raise TopologyError("BODY_TOO_LARGE", "request body exceeds the configured limit", 413)
        if header_bytes < 0 or header_bytes > self.limits.header_bytes:
            raise TopologyError("HEADERS_TOO_LARGE", "request headers exceed the configured limit", 431)


class RateLimiter:
    def __init__(self, limit_per_minute: int, *, clock: Any | None = None):
        if limit_per_minute <= 0:
            raise ValueError("limit_per_minute must be positive")
        self.limit = limit_per_minute
        self.clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._buckets: dict[str, list[float]] = {}

    def allow(self, key: str) -> bool:
        now = float(self.clock())
        with self._lock:
            values = self._buckets.setdefault(key, [])
            cutoff = now - 60.0
            while values and values[0] <= cutoff:
                values.pop(0)
            if len(values) >= self.limit:
                return False
            values.append(now)
            return True
