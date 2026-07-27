"""Bounded queue/WebSocket/storage/replay soak harness for Phase 6."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import tracemalloc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gds.events import EventStore
from gds.product_store import ProductStore
from gds.retention import RetentionManager
from gds.writer import SQLiteWriter
from link_sim.replay_manager import ArtifactStatus, ReplayManager, ReplaySegment


def run_soak(iterations: int = 100) -> dict:
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    tracemalloc.start()
    with tempfile.TemporaryDirectory(prefix="cube-nano-p6-soak-") as value:
        root = Path(value)
        with SQLiteWriter(root / "gds.sqlite3") as writer:
            events = EventStore(writer)
            from gds.realtime import RealtimeHub

            hub = RealtimeHub(events, lambda: {"state": "READY"}, max_client_events=32, max_client_bytes=128 * 1024)
            _, client, _ = hub.connect()
            replay = ReplayManager(root / "replay", global_cap_bytes=64 * 1024, pin_quota_bytes=32 * 1024, max_artifact_bytes=8 * 1024)
            raw_root = root / "raw"
            raw_root.mkdir()
            products = ProductStore(writer, root / "products")
            retention = RetentionManager(writer, products)
            for index in range(iterations):
                event = events.append("SOAK_EVENT", message={"iteration": index})
                hub.publish(event)
                client.drain()
                run_id = index + 1
                if replay.reserve_artifact(run_id, index * 1_000_000):
                    data = f"replay-{index}".encode("ascii")
                    segment_path = replay.storage_root / f"{run_id:016x}" / "00000000.seg"
                    segment_path.parent.mkdir(parents=True, exist_ok=True)
                    segment_path.write_bytes(data)
                    segment = ReplaySegment(0, len(data), hashlib.sha256(data).hexdigest(), segment_path)
                    replay.finalize_artifact(run_id, ArtifactStatus.FINAL, [segment], index * 1_000_000 + 1)
                if index % 5 == 0:
                    replay.evict_oldest_unpinned()
                raw = raw_root / f"frames-{index:04d}.seg"
                raw.write_bytes(b"x" * 64)
                if index % 10 == 0:
                    raw_stat_time = 1_000_000_000
                    os.utime(raw, ns=(raw_stat_time, raw_stat_time))
            cleaned = retention.cleanup_files(
                now_us=100 * 86_400_000_000,
                raw_roots=(raw_root,),
            )["raw"]
            stats = replay.get_stats()
            current, peak = tracemalloc.get_traced_memory()
            wal = root / "gds.sqlite3-wal"
            report = {
                "schema_version": 1,
                "iterations": iterations,
                "event_count": events.latest_event_id(),
                "realtime_client_queue_depth": client.queue_depth,
                "realtime_client_queue_bytes": client.queue_bytes,
                "replay": stats,
                "raw_file_count": len(tuple(raw_root.glob("*.seg"))),
                "cleanup_count": len(cleaned),
                "wal_bytes": wal.stat().st_size if wal.exists() else 0,
                "tracemalloc_current_bytes": current,
                "tracemalloc_peak_bytes": peak,
                "guards": {
                    "bounded_client_queue": client.queue_depth <= 32 and client.queue_bytes <= 128 * 1024,
                    "bounded_replay": stats["used_bytes"] <= 64 * 1024,
                    "bounded_raw_fixture": len(tuple(raw_root.glob("*.seg"))) <= iterations,
                },
            }
    tracemalloc.stop()
    if not all(report["guards"].values()):
        raise RuntimeError(f"soak guard failed: {report}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("artifacts/soak/phase6_soak_report.json"))
    args = parser.parse_args()
    report = run_soak(args.iterations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
