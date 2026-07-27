"""Fail-closed validation for the first local SIL topology."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_runtime_profile(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    validate_runtime_profile(value)
    return value


def validate_runtime_profile(value: dict[str, Any]) -> None:
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("runtime profile schema_version must be 1")
    if value.get("profile_id") != "host_local_sil" or value.get("topology") != "host_local_sil":
        raise ValueError("first profile must be host_local_sil")
    if value.get("bind_host") != "127.0.0.1" or value.get("network_exposure") is not False:
        raise ValueError("host_local_sil must bind loopback and disable network exposure")
    if set(value.get("allowed_peers", [])) != {"127.0.0.1"}:
        raise ValueError("host_local_sil peer allowlist must be exact loopback")
    storage = value.get("storage", {})
    replay = value.get("replay", {})
    sqlite = value.get("sqlite", {})
    if sqlite.get("journal_mode") != "WAL" or sqlite.get("synchronous") != "FULL" or sqlite.get("single_writer") is not True:
        raise ValueError("SQLite must use WAL/FULL/single-writer")
    if int(storage.get("ground_cap_bytes", 0)) <= int(storage.get("emergency_reserve_bytes", 0)):
        raise ValueError("storage cap must leave emergency reserve")
    if replay.get("schema") != "link-replay-v1" or set(replay.get("state_values", [])) != {"PRESENT", "PINNED", "EVICTED"}:
        raise ValueError("replay profile is incomplete")
