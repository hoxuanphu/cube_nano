"""Repeatable local-SIL operator scenario: scene -> ROI -> product verify."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gds.http_app import LocalSilMission
from protocol.schemas import CommandOpcode, RequestKey


def run(root: Path, timeout_seconds: float = 90.0) -> dict:
    with tempfile.TemporaryDirectory(prefix="cube-nano-p6-demo-") as value:
        mission = LocalSilMission(root, state_directory=Path(value))
        try:
            instance = f"{mission.instance:016x}"
            config = mission.snapshot()["configs"][instance]
            status, accepted, _ = mission.submit(
                {
                    "target_spacecraft_instance_id": instance,
                    "opcode": int(CommandOpcode.ROI_REQUEST),
                    "payload": {
                        "scene_ref": {"catalog_epoch": 1, "scene_id": 1, "scene_revision": 1},
                        "roi": {"x": 0, "y": 0, "width": 256, "height": 256},
                        "expected_config_epoch": config["config_epoch"],
                        "expected_config_revision": config["config_revision"],
                        "model_threshold_bp": config["model_threshold_bp"],
                        "coverage_limit_bp": config["coverage_limit_bp"],
                    },
                    "delivery_mode": "immediate",
                },
                "p6-demo-roi-001",
            )
            if status != 202:
                raise RuntimeError(accepted)
            deadline = time.monotonic() + timeout_seconds
            product = None
            while time.monotonic() < deadline:
                state = mission.snapshot()
                for item in state["products"].values():
                    if item.get("state") == "PUBLISHED":
                        product = item
                        break
                if product is not None:
                    break
                time.sleep(0.2)
            if product is None:
                request_key = RequestKey.from_dict(accepted["request_key"])
                job = mission.satellite.payload.journal.get_job(request_key)
                job_detail = None if job is None else {
                    "state": job["state"],
                    "error_code": job["error_code"],
                    "result": None if not job["result_json"] else json.loads(job["result_json"]),
                }
                raise TimeoutError(
                    "demo did not publish a verified product: "
                    + json.dumps(
                        {
                            "commands": mission.snapshot()["commands"],
                            "job": job_detail,
                            "worker": mission.satellite.payload.worker_client.health() if mission.satellite.payload.worker_client else None,
                        }
                    )
                )
            return {
                "status": "PASS",
                "request_key": accepted["request_key"],
                "spacecraft_instance_id": instance,
                "product": product,
                "trace": mission.command(accepted["request_key"]["ground_instance_id"], accepted["request_key"]["request_id"]),
            }
        finally:
            mission.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--timeout", type=float, default=90.0)
    args = parser.parse_args()
    print(json.dumps(run(args.root.resolve(), args.timeout), sort_keys=True))


if __name__ == "__main__":
    main()
