"""Generate deterministic release artifacts used by the local CPU profile."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .benchmark import run_benchmark
from .threshold_lut import generate_threshold_lut


def generate_lut(path: Path) -> str:
    lut = generate_threshold_lut()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(lut.raw)
    return lut.sha256


def generate_benchmark(path: Path, *, root: Path) -> str:
    payload = run_benchmark(root)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--lut", type=Path, default=Path("protocol/golden_vectors/threshold_lut.bin"))
    parser.add_argument("--benchmark", type=Path, default=Path("artifacts/benchmarks/local-cpu-pytorch-v2.json"))
    args = parser.parse_args()
    lut_sha256 = generate_lut(args.lut)
    benchmark_sha256 = generate_benchmark(args.benchmark, root=args.root)
    print(json.dumps({"lut_sha256": lut_sha256, "benchmark_sha256": benchmark_sha256}, indent=2))


if __name__ == "__main__":
    main()
