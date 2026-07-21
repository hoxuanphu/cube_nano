"""Validate host/compose/Jetson readiness gates without starting services."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gds.topology import TopologyProfile


def validate(root: Path) -> dict:
    host = TopologyProfile.from_file(root / "protocol" / "runtime_profile.yaml")
    compose = TopologyProfile.from_file(root / "deploy" / "compose_runtime_profile.yaml")
    host.validate_startup(host.bind_host)
    compose.validate_startup(compose.bind_host)
    jetson = yaml.safe_load((root / "deploy" / "jetson-l4t-profile.yaml").read_text(encoding="utf-8"))
    ready_profiles = []
    for profile in (host, compose):
        ready_profiles.append({"profile_id": profile.profile_id, "topology": profile.topology, "bind_host": profile.bind_host, "ready": True})
    jetson_ready = bool(jetson.get("deployable")) and bool(jetson.get("benchmark_artifact_id")) and bool(jetson.get("benchmark_artifact_sha256"))
    if jetson_ready:
        raise ValueError("Jetson profile must not be READY without a target benchmark artifact")
    return {
        "schema_version": 1,
        "profiles": ready_profiles,
        "jetson": {"profile_id": jetson["profile_id"], "ready": False, "reason": jetson["blocked_reason"]},
        "guards": {
            "host_loopback_only": host.bind_host == "127.0.0.1" and not host.network_exposure,
            "compose_host_publish_loopback": compose.network_exposure is False,
            "jetson_fail_closed": not jetson_ready,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    print(json.dumps(validate(args.root.resolve()), sort_keys=True))


if __name__ == "__main__":
    main()
