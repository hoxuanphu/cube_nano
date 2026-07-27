"""Run a guarded preprocessing soak benchmark on a declared target board.

The command deliberately refuses to call a Nano/Orin run a host benchmark. A
JSON report with ``status=BLOCKED`` is written when the requested board cannot
be identified. Fixture mode uses the public preprocessing facade and is useful
for validating the harness on a development machine; it is not flight trust
or TensorRT evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from preprocessing import (
    CalibrationBundle,
    CaptureManifest,
    ComputeProfile,
    PreprocessFailure,
    PreprocessRequest,
    PreprocessingProfile,
    TrustPolicy,
    preprocess_capture,
)


def _read_proc_bytes(path: str | Path) -> bytes | None:
    try:
        return Path(path).read_bytes()
    except OSError:
        return None


def _read_status_bytes(path: str | Path, field: str) -> int | None:
    payload = _read_proc_bytes(path)
    if payload is None:
        return None
    for line in payload.decode("ascii", errors="ignore").splitlines():
        if line.startswith(f"{field}:"):
            fields = line.split()
            if len(fields) >= 2:
                try:
                    return int(fields[1]) * 1024
                except ValueError:
                    return None
    return None


def _jetson_model() -> str:
    model = _read_proc_bytes("/proc/device-tree/model")
    if not model:
        return "unknown"
    text = model.decode("utf-8", errors="ignore").strip().lower()
    if "orin" in text and "nano" in text:
        return "jetson-orin-nano"
    if "nano" in text:
        return "jetson-nano"
    return "unknown"


def _thermal_max_celsius() -> float | None:
    values: list[float] = []
    for path in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try:
            raw = float(path.read_text(encoding="ascii").strip())
        except (OSError, ValueError):
            continue
        values.append(raw / 1000.0 if raw > 200.0 else raw)
    return max(values) if values else None


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _configuration(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _fixture_request(root: Path, *, rows: int, cols: int, run_index: int, compute_id: str) -> PreprocessRequest:
    source = root / "fixture-source.npy"
    if not source.exists():
        image = np.arange(rows * cols * 3, dtype=np.uint16).reshape(rows, cols, 3)
        np.save(source, image)
    profile = PreprocessingProfile.identity_fixture(shape=(rows, cols), dtype="uint16")
    calibration = CalibrationBundle.identity_fixture()
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    capture = CaptureManifest(
        capture_id="p7-hil-fixture",
        completion_marker=True,
        source_fingerprint=source_digest,
        sensor_product="fixture-sensor",
        dimensions=(rows, cols, 3),
        source_layout="YXC",
        source_dtype="uint16",
        source_byte_order="native",
        channel_schema=("red", "green", "blue"),
        nodata_semantics={"kind": "none"},
        source_path=str(source),
        preprocessing_profile_digest=profile.fingerprint,
        calibration_digest=calibration.fingerprint,
    )
    compute = ComputeProfile(
        compute_profile_id=compute_id,
        profile_version="1",
        backend="cpu",
        ram_budget_bytes=128 * 1024 * 1024,
        disk_budget_bytes=512 * 1024 * 1024,
        os_obc_reserve_bytes=1 * 1024 * 1024,
        safety_margin_bytes=1 * 1024 * 1024,
        max_strip_rows=min(rows, 32),
        queue_depth=1,
        inflight_strips=1,
        temporary_directory=str(root),
        checksum_headroom_bytes=1 * 1024 * 1024,
        thermal_policy="p7-hil-fixture",
    )
    return PreprocessRequest(
        source=source,
        capture_manifest=capture,
        calibration_bundle=calibration,
        preprocessing_profile=profile,
        compute_profile=compute,
        output_artifact_target=root / f"run-{run_index}.artifact",
        run_id=f"p7-hil-{run_index}",
        trust_policy=TrustPolicy.development(),
    )


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    if args.iterations < 5:
        raise ValueError("P7 soak benchmark requires at least five measured iterations")
    if args.warmup_runs < 1:
        raise ValueError("P7 soak benchmark requires at least one warm-up run")
    detected = _jetson_model()
    environment = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "jetson_model": detected,
        "compute_profile_id": args.compute_profile_id,
    }
    if args.target != "host" and detected != args.target:
        return {
            "schema_version": 1,
            "status": "BLOCKED",
            "target": args.target,
            "detected_board": detected,
            "environment": environment,
            "configuration": _configuration(args),
            "reason_code": "HIL_TARGET_UNAVAILABLE",
            "safe_action": "RETAIN_FOR_GROUND",
            "message": "requested HIL board was not identified; no benchmark was executed",
        }

    runs: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="cube_nano_p7_hil_") as temporary:
        root = Path(temporary)
        for index in range(args.warmup_runs):
            result = preprocess_capture(
                _fixture_request(root, rows=args.rows, cols=args.cols, run_index=-(index + 1), compute_id=args.compute_profile_id)
            )
            if isinstance(result, PreprocessFailure):
                return {
                    "schema_version": 1,
                    "status": "FAIL",
                    "target": args.target,
                    "detected_board": detected,
                    "environment": environment,
                    "configuration": _configuration(args),
                    "reason_code": result.reason_code,
                    "safe_action": result.safe_action.value,
                    "message": result.message,
                }
            shutil.rmtree(result.artifact_path, ignore_errors=True)

        for index in range(args.iterations):
            started = time.perf_counter()
            result = preprocess_capture(
                _fixture_request(root, rows=args.rows, cols=args.cols, run_index=index, compute_id=args.compute_profile_id)
            )
            elapsed = time.perf_counter() - started
            run = {
                "iteration": index,
                "elapsed_seconds": elapsed,
                "peak_rss_bytes": _read_status_bytes("/proc/self/status", "VmRSS"),
                "available_ram_bytes": _read_status_bytes("/proc/meminfo", "MemAvailable"),
                "max_temperature_c": _thermal_max_celsius(),
            }
            if isinstance(result, PreprocessFailure):
                run.update({"status": "FAIL", "reason_code": result.reason_code, "message": result.message})
                runs.append(run)
                break
            run["status"] = "complete"
            runs.append(run)
            shutil.rmtree(result.artifact_path, ignore_errors=True)

    successful = [run for run in runs if run.get("status") == "complete"]
    if len(successful) != args.iterations:
        status = "FAIL"
        reason_code = next((run.get("reason_code") for run in runs if run.get("reason_code")), "RUNTIME_FAULT")
    else:
        status = "COMPLETE"
        reason_code = None
    elapsed_values = [float(run["elapsed_seconds"]) for run in successful]
    return {
        "schema_version": 1,
        "status": status,
        "target": args.target,
        "detected_board": detected,
        "environment": environment,
        "configuration": _configuration(args),
        "reason_code": reason_code,
        "safe_action": "RETAIN_FOR_GROUND" if status != "COMPLETE" else None,
        "summary": {
            "iteration_count": len(successful),
            "elapsed_seconds_median": statistics.median(elapsed_values) if elapsed_values else None,
            "elapsed_seconds_p95": _percentile(elapsed_values, 95) if elapsed_values else None,
            "peak_rss_bytes": max((run["peak_rss_bytes"] or 0) for run in successful) if successful else None,
            "minimum_available_ram_bytes": min(
                (run["available_ram_bytes"] for run in successful if run["available_ram_bytes"] is not None),
                default=None,
            ),
            "maximum_temperature_c": max(
                (run["max_temperature_c"] for run in successful if run["max_temperature_c"] is not None),
                default=None,
            ),
        },
        "runs": runs,
        "evidence_note": "fixture source and development trust policy; not a flight qualification record",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Guarded P7 preprocessing HIL/soak benchmark")
    parser.add_argument("--target", choices=("host", "jetson-nano", "jetson-orin-nano"), required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--cols", type=int, default=64)
    parser.add_argument("--compute-profile-id", default="p7-hil-fixture")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = run_benchmark(args)
    _atomic_write_json(args.output_json, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] == "BLOCKED":
        return 2
    return 0 if payload["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
