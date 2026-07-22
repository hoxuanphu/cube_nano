"""Measured local CPU reference benchmark and deadline-model generator."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import tifffile
import yaml

from protocol.schemas import ConfigSnapshot, ROI

from .inference import InferenceConfig, SingletonModelRuntime, configure_cpu_runtime, infer_region
from .manifest import load_model_manifest
from .roi import open_memmap_scene
from .threshold_lut import ThresholdLUT


def _rss_bytes() -> int:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        )
        return int(counters.PeakWorkingSetSize)
    try:
        import resource

        scale = 1024 if os.uname().sysname != "Darwin" else 1
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale)
    except (ImportError, AttributeError):
        return 0


def _percentile(samples: list[float], percentile: float) -> float:
    if not samples:
        raise ValueError("benchmark sample set is empty")
    ordered = sorted(samples)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _create_fixture(directory: Path, name: str, shape: tuple[int, int, int], input_spec_id: str) -> tuple[Path, Path]:
    source_path = directory / f"{name}.tif"
    source = np.zeros(shape, dtype=np.uint16)
    source[0:256, 0:256] = 32768
    tifffile.imwrite(source_path, source, metadata={"axes": "YXC"}, compression=None)
    source_digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    sidecar_path = directory / f"{name}.json"
    sidecar_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_fingerprint": {"algorithm": "sha256", "digest": source_digest},
                "axes": "YXC",
                "shape": list(shape),
                "band_order": ["red", "green", "blue"],
                "dtype": "uint16",
                "input_spec_id": input_spec_id,
                "validity": {"kind": "all_valid"},
            }
        ),
        encoding="utf-8",
    )
    return source_path, sidecar_path


def _measure_scene(
    source_path: Path,
    sidecar_path: Path,
    runtime: SingletonModelRuntime,
    config: InferenceConfig,
    *,
    samples: int = 20,
) -> tuple[dict, list[float]]:
    roi = ROI(0, 0, 256, 256)
    with open_memmap_scene(source_path, sidecar_path) as scene:
        infer_region(scene, roi, runtime, config)
        measurements = []
        result = None
        for _ in range(samples):
            started = time.perf_counter()
            result = infer_region(scene, roi, runtime, config)
            measurements.append((time.perf_counter() - started) * 1000.0)
    assert result is not None
    return result, measurements


def run_benchmark(root: str | Path) -> dict:
    root = Path(root)
    deployment_profile = yaml.safe_load(
        (root / "sat_ai" / "deployment_profile.yaml").read_text(encoding="utf-8")
    )
    configure_cpu_runtime(int(deployment_profile["cpu_threads"]))
    manifest = load_model_manifest(
        root / "sat_ai" / "model_manifest.yaml",
        root / "checkpoints" / "best_model.pth",
    )
    lut = ThresholdLUT.from_file(
        root / "protocol" / "golden_vectors" / "threshold_lut.bin",
        manifest.threshold_lut_sha256,
    )
    with tempfile.TemporaryDirectory(prefix="cube-nano-benchmark-") as directory_value:
        directory = Path(directory_value)
        canonical_paths = _create_fixture(
            directory, "canonical", (512, 512, 3), manifest.input_spec.input_spec_id
        )
        scaled_paths = _create_fixture(
            directory, "scene-4x-area", (1024, 1024, 3), manifest.input_spec.input_spec_id
        )
        rss_before = _rss_bytes()
        load_started = time.perf_counter()
        runtime = SingletonModelRuntime(
            manifest,
            str(root / "checkpoints" / "best_model.pth"),
            device="cpu",
        )
        model_load_ms = (time.perf_counter() - load_started) * 1000.0
        config = InferenceConfig(ConfigSnapshot(0, 0, 5000, 6000), lut)
        canonical_result, canonical_samples = _measure_scene(
            *canonical_paths, runtime, config
        )
        scaled_result, scaled_samples = _measure_scene(*scaled_paths, runtime, config)
        rss_after = _rss_bytes()
        runtime.close()
    canonical_p95 = _percentile(canonical_samples, 0.95)
    canonical_p99 = _percentile(canonical_samples, 0.99)
    scaled_p95 = _percentile(scaled_samples, 0.95)
    ratio = scaled_p95 / max(canonical_p95, 0.001)
    patch_count = int(canonical_result["patch_count"])
    deadline_ms = max(5000, math.ceil(canonical_p99 * patch_count * 4.0))
    return {
        "schema_version": 1,
        "artifact_id": "local-cpu-pytorch-v2",
        "target_id": "local-cpu-pytorch",
        "runtime": "pytorch",
        "generated_at": "1970-01-01T00:00:00Z",
        "input_spec_id": manifest.input_spec.input_spec_id,
        "batch_sizes": [1],
        "cpu_threads": int(deployment_profile["cpu_threads"]),
        "fixture": {
            "canonical_scene_shape": [512, 512, 3],
            "scaled_scene_shape": [1024, 1024, 3],
            "roi": [0, 0, 256, 256],
            "patch_count": patch_count,
            "samples_per_scene": len(canonical_samples),
        },
        "measurements": {
            "warmup": True,
            "model_load_count": runtime.load_count,
            "model_load_ms": round(model_load_ms, 3),
            "throughput_patches_per_second": round(
                patch_count * 1000.0 / max(canonical_p95, 0.001), 3
            ),
            "p95_latency_ms": round(canonical_p95, 3),
            "p99_latency_ms": round(canonical_p99, 3),
            "scaled_scene_p95_latency_ms": round(scaled_p95, 3),
            "scene_scale_p95_ratio": round(ratio, 4),
            "rss_delta_bytes": max(0, rss_after - rss_before),
            "logical_source_bytes_read": canonical_result["reader_metrics"]["logical_source_bytes_read"],
            "logical_validity_bytes_read": canonical_result["reader_metrics"]["logical_validity_bytes_read"],
            "logical_bytes_read": canonical_result["reader_metrics"]["logical_bytes_read"],
            "deadline_ms": deadline_ms,
            "deadline_safety_factor": 4,
        },
        "guards": {
            "max_window_rss_delta_bytes": 268435456,
            "max_scene_scale_p95_ratio": 1.25,
            "rss_pass": max(0, rss_after - rss_before) <= 268435456,
            "scene_scale_pass": ratio <= 1.25,
        },
        "threshold_lut_sha256": lut.sha256,
    }


def _batch_target_status(device: str) -> tuple[str, str | None]:
    if device == "cpu":
        return "AVAILABLE", None
    if device == "cuda":
        try:
            import torch

            return ("AVAILABLE", None) if torch.cuda.is_available() else ("UNAVAILABLE", "CUDA runtime is not available")
        except Exception as exc:  # pragma: no cover - defensive for stripped images
            return "UNAVAILABLE", f"CUDA probe failed: {type(exc).__name__}"
    if platform.machine().lower() not in {"aarch64", "arm64"}:
        return "UNAVAILABLE", "Jetson target requires an ARM64 runtime"
    if shutil.which("tegrastats") is None:
        return "UNAVAILABLE", "tegrastats is not installed"
    return "AVAILABLE", None


def run_batch_matrix(
    root: str | Path,
    *,
    batch_sizes: tuple[int, ...] = (1, 2, 4, 8),
    samples: int = 2,
    targets: tuple[str, ...] = ("cpu", "cuda", "jetson"),
) -> dict:
    """Measure all candidate batch sizes and explicitly record unavailable targets.

    The deployable profile remains a separate decision: only a target with an
    artifact and passing resource guards may be marked READY.
    """
    root = Path(root).resolve()
    if not batch_sizes or any(size <= 0 for size in batch_sizes):
        raise ValueError("batch_sizes must contain positive integers")
    if samples <= 0:
        raise ValueError("samples must be positive")
    deployment_profile = yaml.safe_load((root / "sat_ai" / "deployment_profile.yaml").read_text(encoding="utf-8"))
    configure_cpu_runtime(int(deployment_profile["cpu_threads"]))
    manifest = load_model_manifest(root / "sat_ai" / "model_manifest.yaml", root / "checkpoints" / "best_model.pth")
    lut = ThresholdLUT.from_file(root / "protocol" / "golden_vectors" / "threshold_lut.bin", manifest.threshold_lut_sha256)
    target_records: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="cube-nano-batch-matrix-") as directory_value:
        directory = Path(directory_value)
        source_path, sidecar_path = _create_fixture(
            directory, "batch-matrix", (1024, 1024, 3), manifest.input_spec.input_spec_id
        )
        config = InferenceConfig(ConfigSnapshot(0, 0, 5000, 6000), lut)
        for target in targets:
            device = "cuda" if target in {"cuda", "jetson"} else "cpu"
            status, reason = _batch_target_status(device)
            record: dict = {
                "target_id": target,
                "runtime": "pytorch" if target != "jetson" else "tensorrt-candidate",
                "status": status,
                "reason": reason,
                "batch_sizes": list(batch_sizes),
                "measurements": [],
            }
            if status != "AVAILABLE":
                target_records.append(record)
                continue
            try:
                runtime = SingletonModelRuntime(manifest, str(root / "checkpoints" / "best_model.pth"), device=device)
            except Exception as exc:
                record["status"] = "UNAVAILABLE"
                record["reason"] = f"runtime initialization failed: {type(exc).__name__}"
                target_records.append(record)
                continue
            try:
                with open_memmap_scene(source_path, sidecar_path) as scene:
                    for batch_size in batch_sizes:
                        infer_region(scene, ROI(0, 0, 1024, 1024), runtime, config, batch_size=batch_size)
                        measurements: list[float] = []
                        rss_before = _rss_bytes()
                        for _ in range(samples):
                            started = time.perf_counter()
                            result = infer_region(scene, ROI(0, 0, 1024, 1024), runtime, config, batch_size=batch_size)
                            measurements.append((time.perf_counter() - started) * 1000.0)
                        p50 = _percentile(measurements, 0.50)
                        p95 = _percentile(measurements, 0.95)
                        record["measurements"].append({
                            "batch_size": batch_size,
                            "patch_count": int(result["patch_count"]),
                            "p50_latency_ms": round(p50, 3),
                            "p95_latency_ms": round(p95, 3),
                            "throughput_patches_per_second": round(result["patch_count"] * 1000.0 / max(p50, 0.001), 3),
                            "rss_delta_bytes": max(0, _rss_bytes() - rss_before),
                        })
                record["model_load_count"] = runtime.load_count
            finally:
                runtime.close()
            target_records.append(record)
    return {
        "schema_version": 1,
        "artifact_id": "phase6-batch-matrix-v1",
        "generated_at": _timestamp_for_artifact(),
        "input_spec_id": manifest.input_spec.input_spec_id,
        "batch_sizes": list(batch_sizes),
        "samples_per_batch": samples,
        "targets": target_records,
        "guards": {
            "max_window_rss_delta_bytes": 268435456,
            "max_scene_scale_p95_ratio": 1.25,
            "ready_requires_available_artifact": True,
        },
    }


def _timestamp_for_artifact() -> str:
    epoch = int(os.environ.get("SOURCE_DATE_EPOCH", "0"))
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/benchmarks/local-cpu-pytorch-v2.json"),
    )
    parser.add_argument("--batch-matrix", action="store_true")
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8")
    parser.add_argument("--samples", type=int, default=2)
    args = parser.parse_args()
    payload = (
        run_batch_matrix(
            args.root,
            batch_sizes=tuple(int(value) for value in args.batch_sizes.split(",") if value),
            samples=args.samples,
        )
        if args.batch_matrix
        else run_benchmark(args.root)
    )
    encoded = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        + b"\n"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(encoded)
    summary = payload.get("measurements", {"targets": payload.get("targets", [])})
    print(
        json.dumps(
            {
                "path": str(args.output),
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "measurements": summary,
                "guards": payload["guards"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
