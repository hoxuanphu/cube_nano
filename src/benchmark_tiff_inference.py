import argparse
import json
import os
import platform
import statistics
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from inference_large_image_trt import process_large_image


def _read_status_bytes(path, field):
    try:
        lines = Path(path).read_text(encoding="ascii").splitlines()
    except OSError:
        return None
    for line in lines:
        if line.startswith(f"{field}:"):
            return int(line.split()[1]) * 1024
    return None


class _MemoryMonitor:
    def __init__(self, interval_seconds=0.02):
        self.interval_seconds = interval_seconds
        self.peak_rss_bytes = 0
        self.minimum_available_bytes = None
        self._stop = threading.Event()
        self._thread = None

    def __enter__(self):
        self._sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._stop.set()
        self._thread.join()
        self._sample()
        return False

    def _run(self):
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def _sample(self):
        rss = _read_status_bytes("/proc/self/status", "VmRSS")
        available = _read_status_bytes("/proc/meminfo", "MemAvailable")
        if rss is not None:
            self.peak_rss_bytes = max(self.peak_rss_bytes, rss)
        if available is not None:
            if self.minimum_available_bytes is None:
                self.minimum_available_bytes = available
            else:
                self.minimum_available_bytes = min(self.minimum_available_bytes, available)


def _percentile(values, percentile):
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _runtime_environment(args):
    environment = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "jetson_model": args.jetson_model,
        "jetpack_version": args.jetpack_version,
        "power_mode": args.power_mode,
    }
    try:
        import tensorrt

        environment["tensorrt_version"] = tensorrt.__version__
    except ImportError:
        environment["tensorrt_version"] = None
    return environment


def _atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        temporary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def run_benchmark(args):
    if args.iterations < 5:
        raise ValueError("Benchmark requires at least five measured iterations")
    if args.warmup_runs < 1:
        raise ValueError("Benchmark requires at least one warm-up run")

    common = {
        "large_image_path": args.image,
        "engine_path": args.engine,
        "patch_size": args.patch_size,
        "channels": args.channels,
        "batch_size": args.batch_size,
        "threshold": args.threshold,
        "tiff_read_mode": args.tiff_read_mode,
        "tiff_cache_mode": args.tiff_cache_mode,
        "max_ram_cache_gib": args.max_ram_cache_gib,
        "max_disk_cache_gib": args.max_disk_cache_gib,
        "runtime_reserve_gib": args.runtime_reserve_gib,
        "tiff_block_cache_mib": args.tiff_block_cache_mib,
        "tiff_series": args.tiff_series,
        "tiff_level": args.tiff_level,
        "channel_mapping": args.channel_mapping,
        "input_sidecar": args.input_sidecar,
        "engine_manifest": args.engine_manifest,
        "production_contract": args.production_contract,
    }

    runs = []
    with tempfile.TemporaryDirectory(prefix="cube_nano_benchmark_") as directory:
        root = Path(directory)
        common["tiff_cache_dir"] = root / "cache"
        for index in range(args.warmup_runs):
            process_large_image(
                **common,
                out_mask=root / f"warmup_{index}.tif",
            )

        for index in range(args.iterations):
            output = root / f"measured_{index}.tif"
            with _MemoryMonitor() as memory:
                started = time.perf_counter()
                result = process_large_image(**common, out_mask=output)
                measured_seconds = time.perf_counter() - started
            runs.append(
                {
                    "iteration": index,
                    "total_seconds": measured_seconds,
                    "inference_seconds": result["elapsed_seconds"],
                    "peak_rss_bytes": memory.peak_rss_bytes,
                    "minimum_mem_available_bytes": memory.minimum_available_bytes,
                    "reader_backend": result["reader_backend"],
                    "reader_metrics": result["reader_metrics"],
                    "reader_provenance": result["reader_provenance"],
                }
            )

    total_times = [run["total_seconds"] for run in runs]
    inference_times = [run["inference_seconds"] for run in runs]
    available_values = [
        run["minimum_mem_available_bytes"]
        for run in runs
        if run["minimum_mem_available_bytes"] is not None
    ]
    payload = {
        "schema_version": 1,
        "cache_state": args.cache_state,
        "cache_state_note": (
            "The harness records cache state but does not modify the OS page cache. "
            "Prepare cold-cache runs externally on Jetson."
        ),
        "input": str(Path(args.image).resolve()),
        "engine": str(Path(args.engine).resolve()),
        "configuration": {
            "iterations": args.iterations,
            "warmup_runs": args.warmup_runs,
            "patch_size": args.patch_size,
            "channels": args.channels,
            "batch_size": args.batch_size,
            "tiff_read_mode": args.tiff_read_mode,
            "tiff_cache_mode": args.tiff_cache_mode,
        },
        "environment": _runtime_environment(args),
        "summary": {
            "total_seconds_median": statistics.median(total_times),
            "total_seconds_p95": _percentile(total_times, 95),
            "inference_seconds_median": statistics.median(inference_times),
            "inference_seconds_p95": _percentile(inference_times, 95),
            "peak_rss_bytes": max(run["peak_rss_bytes"] for run in runs),
            "minimum_mem_available_bytes": min(available_values) if available_values else None,
        },
        "runs": runs,
    }
    _atomic_write_json(args.output_json, payload)
    return payload


def build_parser():
    parser = argparse.ArgumentParser(description="Benchmark guarded TIFF TensorRT inference")
    parser.add_argument("--image", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--engine_manifest")
    parser.add_argument("--production_contract", action="store_true")
    parser.add_argument("--input_sidecar")
    parser.add_argument("--channel_mapping")
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--channels", type=int, choices=[3, 4], default=4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup_runs", type=int, default=1)
    parser.add_argument("--cache_state", choices=["cold", "warm"], required=True)
    parser.add_argument("--tiff_read_mode", choices=["auto", "stream", "full"], default="auto")
    parser.add_argument("--tiff_cache_mode", choices=["auto", "ram", "disk"], default="auto")
    parser.add_argument("--max_ram_cache_gib", default="0.5")
    parser.add_argument("--max_disk_cache_gib", default="8.0")
    parser.add_argument("--runtime_reserve_gib", default="1.5")
    parser.add_argument("--tiff_block_cache_mib", default="64")
    parser.add_argument("--tiff_series", type=int)
    parser.add_argument("--tiff_level", type=int)
    parser.add_argument("--jetson_model")
    parser.add_argument("--jetpack_version")
    parser.add_argument("--power_mode")
    return parser


if __name__ == "__main__":
    run_benchmark(build_parser().parse_args())
