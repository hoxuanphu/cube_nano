"""Tests for Link Simulator benchmark."""

import pytest
from pathlib import Path

from link_sim.benchmark import BenchmarkResult, LinkBenchmark
from link_sim.fault_model import FaultProfile
from link_sim.link_simulator import LinkSimulator


def test_benchmark_result_creation():
    """Test creating a benchmark result."""
    result = BenchmarkResult(
        profile_name="baseline",
        frame_count=1000,
        frame_size_bytes=1024,
        duration_seconds=10.0,
        frames_per_second=100.0,
        bits_per_second=819200.0,
        megabits_per_second=0.82,
        frames_lost=0,
        frames_duplicated=0,
        frames_corrupted=0,
        loss_rate=0.0,
        duplication_rate=0.0,
        corruption_rate=0.0,
        total_bits_sent=8192000,
        goodput_bits=8192000,
        overhead_ratio=1.0,
        max_queue_depth=10,
        avg_queue_depth=5.0,
        seed=12345,
        simulation_run_id=1,
        timestamp="2026-07-19T12:00:00Z",
    )

    assert result.profile_name == "baseline"
    assert result.frame_count == 1000
    assert result.megabits_per_second == 0.82


def test_run_benchmark_baseline(tmp_path):
    """Test running baseline benchmark."""
    benchmark = LinkBenchmark(output_dir=tmp_path)

    # Create link simulator
    sim = LinkSimulator(
        simulation_run_id=1,
        seed=12345,
        uplink_profile=FaultProfile(),
        downlink_profile=FaultProfile(),
    )

    # Run benchmark (placeholder - doesn't actually send frames)
    result = benchmark.run_benchmark(
        link_simulator=sim,
        frame_count=100,
        frame_size=1024,
        profile_name="baseline",
    )

    assert result.profile_name == "baseline"
    assert result.frame_count == 100
    assert result.frame_size_bytes == 1024
    assert result.duration_seconds >= 0
    assert result.seed == 12345


def test_save_benchmark_result(tmp_path):
    """Test saving benchmark result to JSON."""
    benchmark = LinkBenchmark(output_dir=tmp_path)

    result = BenchmarkResult(
        profile_name="test",
        frame_count=100,
        frame_size_bytes=1024,
        duration_seconds=1.0,
        frames_per_second=100.0,
        bits_per_second=819200.0,
        megabits_per_second=0.82,
        frames_lost=0,
        frames_duplicated=0,
        frames_corrupted=0,
        loss_rate=0.0,
        duplication_rate=0.0,
        corruption_rate=0.0,
        total_bits_sent=819200,
        goodput_bits=819200,
        overhead_ratio=1.0,
        max_queue_depth=0,
        avg_queue_depth=0.0,
        seed=12345,
        simulation_run_id=1,
        timestamp="2026-07-19T12:00:00Z",
    )

    output_path = benchmark.save_result(result, filename="test_result.json")

    assert output_path.exists()
    assert output_path.name == "test_result.json"

    # Verify JSON content
    import json
    with open(output_path) as f:
        data = json.load(f)

    assert data["profile_name"] == "test"
    assert data["frame_count"] == 100


def test_compare_profiles(tmp_path):
    """Test comparing multiple profiles."""
    benchmark = LinkBenchmark(output_dir=tmp_path)

    sim = LinkSimulator(
        simulation_run_id=1,
        seed=12345,
        uplink_profile=FaultProfile(),
        downlink_profile=FaultProfile(),
    )

    profiles = {
        "baseline": {},
        "low_loss": {},
        "high_loss": {},
    }

    results = benchmark.compare_profiles(
        link_simulator=sim,
        profiles=profiles,
        frame_count=100,
        frame_size=1024,
    )

    assert len(results) == 3
    assert "baseline" in results
    assert "low_loss" in results
    assert "high_loss" in results

    # Verify all results saved
    saved_files = list(tmp_path.glob("*.json"))
    assert len(saved_files) == 3


def test_benchmark_metrics_calculation():
    """Test benchmark metrics are calculated correctly."""
    result = BenchmarkResult(
        profile_name="test",
        frame_count=1000,
        frame_size_bytes=1024,
        duration_seconds=10.0,
        frames_per_second=100.0,
        bits_per_second=819200.0,
        megabits_per_second=0.8192,
        frames_lost=50,  # 5% loss
        frames_duplicated=20,  # 2% duplication
        frames_corrupted=10,  # 1% corruption
        loss_rate=0.05,
        duplication_rate=0.02,
        corruption_rate=0.01,
        total_bits_sent=8192000 + (20 * 1024 * 8),  # Original + duplicates
        goodput_bits=8192000,
        overhead_ratio=1.02,
        max_queue_depth=50,
        avg_queue_depth=25.0,
        seed=12345,
        simulation_run_id=1,
        timestamp="2026-07-19T12:00:00Z",
    )

    # Verify rates
    assert result.loss_rate == 0.05
    assert result.duplication_rate == 0.02
    assert result.corruption_rate == 0.01

    # Verify overhead
    assert result.overhead_ratio == 1.02


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
