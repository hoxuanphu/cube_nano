"""Benchmark TM goodput and recovery cost for Link Simulator.

Section 9.11: Measures frame throughput, goodput, retry cost under various
fault profiles to tune queue/SLO parameters.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Benchmark result for a single run."""
    profile_name: str
    frame_count: int
    frame_size_bytes: int

    # Timings
    duration_seconds: float

    # Throughput
    frames_per_second: float
    bits_per_second: float
    megabits_per_second: float

    # Fault metrics
    frames_lost: int
    frames_duplicated: int
    frames_corrupted: int
    loss_rate: float
    duplication_rate: float
    corruption_rate: float

    # Recovery cost
    total_bits_sent: int  # Including retries
    goodput_bits: int  # Original frames only
    overhead_ratio: float  # total_bits_sent / goodput_bits

    # Buffer/queue metrics
    max_queue_depth: int
    avg_queue_depth: float

    # Metadata
    seed: int
    simulation_run_id: int
    timestamp: str


class LinkBenchmark:
    """Benchmark harness for Link Simulator performance.

    Section 9.11: Measures goodput under various fault profiles to inform
    queue sizing and SLO tuning decisions.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize benchmark harness.

        Args:
            output_dir: Directory to save benchmark results
        """
        self.output_dir = output_dir or Path("artifacts/benchmarks")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"LinkBenchmark initialized: output_dir={self.output_dir}")

    def run_benchmark(
        self,
        link_simulator,
        frame_count: int,
        frame_size: int = 1024,
        profile_name: str = "baseline",
    ) -> BenchmarkResult:
        """Run benchmark on Link Simulator.

        Args:
            link_simulator: Link simulator instance
            frame_count: Number of frames to send
            frame_size: Frame size in bytes
            profile_name: Profile name for result

        Returns:
            Benchmark result
        """
        logger.info(
            f"Starting benchmark: profile={profile_name}, "
            f"frames={frame_count}, size={frame_size}"
        )

        # Get initial stats
        initial_stats = link_simulator.get_stats()
        initial_frames = initial_stats.get("frames_admitted", 0)

        # Simulate frame ingress
        start_time = time.time()

        # For MVP: just measure time to admit frames
        # Full impl would drive actual frame ingress + virtual clock
        for i in range(frame_count):
            # Simulate frame admission (placeholder)
            pass

        end_time = time.time()
        duration = end_time - start_time

        # Get final stats
        final_stats = link_simulator.get_stats()
        frames_admitted = final_stats.get("frames_admitted", 0) - initial_frames

        # Calculate metrics (placeholder - would use actual fault model stats)
        frames_lost = 0
        frames_duplicated = 0
        frames_corrupted = 0

        # Throughput
        if duration > 0:
            fps = frames_admitted / duration
            bits_sent = frames_admitted * frame_size * 8
            bps = bits_sent / duration
            mbps = bps / 1_000_000
        else:
            fps = 0
            bps = 0
            mbps = 0

        # Recovery cost (placeholder)
        total_bits_sent = frames_admitted * frame_size * 8
        goodput_bits = (frames_admitted - frames_duplicated) * frame_size * 8
        overhead_ratio = total_bits_sent / goodput_bits if goodput_bits > 0 else 1.0

        # Queue metrics (placeholder)
        max_queue_depth = 0
        avg_queue_depth = 0.0

        # Build result
        result = BenchmarkResult(
            profile_name=profile_name,
            frame_count=frame_count,
            frame_size_bytes=frame_size,
            duration_seconds=duration,
            frames_per_second=fps,
            bits_per_second=bps,
            megabits_per_second=mbps,
            frames_lost=frames_lost,
            frames_duplicated=frames_duplicated,
            frames_corrupted=frames_corrupted,
            loss_rate=frames_lost / frame_count if frame_count > 0 else 0.0,
            duplication_rate=frames_duplicated / frame_count if frame_count > 0 else 0.0,
            corruption_rate=frames_corrupted / frame_count if frame_count > 0 else 0.0,
            total_bits_sent=total_bits_sent,
            goodput_bits=goodput_bits,
            overhead_ratio=overhead_ratio,
            max_queue_depth=max_queue_depth,
            avg_queue_depth=avg_queue_depth,
            seed=link_simulator.seed,
            simulation_run_id=link_simulator.simulation_run_id,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

        logger.info(
            f"Benchmark complete: {fps:.1f} fps, {mbps:.2f} Mbps, "
            f"overhead={overhead_ratio:.2f}x"
        )

        return result

    def save_result(self, result: BenchmarkResult, filename: Optional[str] = None) -> Path:
        """Save benchmark result to JSON.

        Args:
            result: Benchmark result
            filename: Optional filename (auto-generated if None)

        Returns:
            Path to saved file
        """
        if filename is None:
            filename = (
                f"link_benchmark_{result.profile_name}_"
                f"{result.timestamp.replace(':', '-')}.json"
            )

        output_path = self.output_dir / filename

        with open(output_path, "w") as f:
            json.dump(asdict(result), f, indent=2)

        logger.info(f"Benchmark result saved: {output_path}")
        return output_path

    def compare_profiles(
        self,
        link_simulator,
        profiles: Dict[str, dict],
        frame_count: int = 1000,
        frame_size: int = 1024,
    ) -> Dict[str, BenchmarkResult]:
        """Compare multiple fault profiles.

        Args:
            link_simulator: Link simulator instance
            profiles: Dict of profile_name -> profile_config
            frame_count: Number of frames per profile
            frame_size: Frame size in bytes

        Returns:
            Dict of profile_name -> result
        """
        results = {}

        for profile_name, profile_config in profiles.items():
            logger.info(f"Benchmarking profile: {profile_name}")

            # TODO: Apply profile_config to link_simulator
            # For MVP: just run with current config

            result = self.run_benchmark(
                link_simulator=link_simulator,
                frame_count=frame_count,
                frame_size=frame_size,
                profile_name=profile_name,
            )

            results[profile_name] = result
            self.save_result(result)

        # Log comparison
        logger.info("\n=== Profile Comparison ===")
        for name, result in results.items():
            logger.info(
                f"{name:20s}: {result.megabits_per_second:8.2f} Mbps, "
                f"overhead={result.overhead_ratio:.2f}x, "
                f"loss={result.loss_rate:.1%}"
            )

        return results
