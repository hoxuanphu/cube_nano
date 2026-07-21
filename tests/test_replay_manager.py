"""Tests for replay manager quota, retention, and pinning (P3-07).

Section 9.3: Replay artifacts with PRESENT/PINNED/EVICTED states.
"""

import pytest
from pathlib import Path
from link_sim.replay_manager import (
    ArtifactStatus,
    ReplayArtifact,
    ReplayManager,
    ReplaySegment,
    ReplayState,
)


def test_reserve_artifact_success(tmp_path):
    """Reserve space for new artifact within capacity."""
    manager = ReplayManager(
        storage_root=tmp_path,
        global_cap_bytes=100_000_000,  # 100 MB
        pin_quota_bytes=50_000_000,    # 50 MB
        max_artifact_bytes=10_000_000,  # 10 MB per artifact
    )

    # First reservation should succeed
    success = manager.reserve_artifact(simulation_run_id=1, current_time_ns=1000)
    assert success is True
    assert manager._used_bytes == 10_000_000  # Reserved full amount


def test_reserve_artifact_exceeds_capacity(tmp_path):
    """Reserve fails when exceeding global cap."""
    manager = ReplayManager(
        storage_root=tmp_path,
        global_cap_bytes=25_000_000,   # 25 MB
        pin_quota_bytes=10_000_000,
        max_artifact_bytes=10_000_000,  # 10 MB per artifact
    )

    # Reserve 3 artifacts (30 MB total, but we have 25 MB cap)
    assert manager.reserve_artifact(1, 1000) is True   # 10 MB used
    assert manager.reserve_artifact(2, 2000) is True   # 20 MB used
    assert manager.reserve_artifact(3, 3000) is False  # Would exceed cap

    assert manager._used_bytes == 20_000_000


def test_finalize_artifact_releases_unused(tmp_path):
    """Finalize releases unused reservation."""
    manager = ReplayManager(
        storage_root=tmp_path,
        global_cap_bytes=100_000_000,
        max_artifact_bytes=10_000_000,
    )

    manager.reserve_artifact(1, 1000)
    assert manager._used_bytes == 10_000_000  # Reserved

    # Finalize with actual size 3 MB
    segments = [
        ReplaySegment(index=0, size_bytes=3_000_000, sha256="1" * 64, path=tmp_path / "seg0"),
    ]
    manager.finalize_artifact(1, ArtifactStatus.FINAL, segments, 2000)

    # Should release 7 MB (10 MB reserved - 3 MB actual)
    assert manager._used_bytes == 3_000_000

    artifact = manager.get_artifact(1)
    assert artifact.status == ArtifactStatus.FINAL
    assert artifact.total_bytes == 3_000_000
    assert artifact.artifact_sha256 is not None  # FINAL computes hash


def test_finalize_incomplete_no_hash(tmp_path):
    """INCOMPLETE artifacts don't compute tree hash."""
    manager = ReplayManager(
        storage_root=tmp_path,
        global_cap_bytes=100_000_000,
        max_artifact_bytes=10_000_000,  # Need to specify this
    )

    assert manager.reserve_artifact(1, 1000) is True
    segments = [
        ReplaySegment(index=0, size_bytes=1_000_000, sha256="0" * 64, path=tmp_path / "seg0"),
    ]
    manager.finalize_artifact(1, ArtifactStatus.INCOMPLETE_CRASH, segments, 2000)

    artifact = manager.get_artifact(1)
    assert artifact.status == ArtifactStatus.INCOMPLETE_CRASH
    assert artifact.artifact_sha256 is None  # No hash for INCOMPLETE


def test_pin_artifact_success(tmp_path):
    """Pin artifact within quota."""
    manager = ReplayManager(
        storage_root=tmp_path,
        global_cap_bytes=100_000_000,
        pin_quota_bytes=50_000_000,
        max_artifact_bytes=10_000_000,
    )

    assert manager.reserve_artifact(1, 1000) is True
    segments = [ReplaySegment(index=0, size_bytes=5_000_000, sha256="2" * 64, path=tmp_path / "seg0")]
    manager.finalize_artifact(1, ArtifactStatus.FINAL, segments, 2000)

    # Pin artifact
    success = manager.pin_artifact(1)
    assert success is True

    artifact = manager.get_artifact(1)
    assert artifact.replay_state == ReplayState.PINNED
    assert manager._pinned_bytes == 5_000_000


def test_pin_artifact_exceeds_quota(tmp_path):
    """Pin fails when exceeding pin quota."""
    manager = ReplayManager(
        storage_root=tmp_path,
        global_cap_bytes=100_000_000,
        pin_quota_bytes=10_000_000,  # 10 MB pin quota
        max_artifact_bytes=10_000_000,
    )

    # Create and finalize two artifacts
    assert manager.reserve_artifact(1, 1000) is True
    segments1 = [ReplaySegment(index=0, size_bytes=6_000_000, sha256="3" * 64, path=tmp_path / "seg0")]
    manager.finalize_artifact(1, ArtifactStatus.FINAL, segments1, 2000)

    assert manager.reserve_artifact(2, 3000) is True
    segments2 = [ReplaySegment(index=0, size_bytes=6_000_000, sha256="4" * 64, path=tmp_path / "seg1")]
    manager.finalize_artifact(2, ArtifactStatus.FINAL, segments2, 4000)

    # Pin first artifact (6 MB)
    assert manager.pin_artifact(1) is True
    assert manager._pinned_bytes == 6_000_000

    # Try to pin second artifact (would be 12 MB total, exceeds 10 MB quota)
    assert manager.pin_artifact(2) is False
    assert manager._pinned_bytes == 6_000_000  # Still only first artifact


def test_unpin_artifact(tmp_path):
    """Unpin artifact releases pin quota."""
    manager = ReplayManager(
        storage_root=tmp_path,
        global_cap_bytes=100_000_000,
        pin_quota_bytes=50_000_000,
        max_artifact_bytes=10_000_000,
    )

    assert manager.reserve_artifact(1, 1000) is True
    segments = [ReplaySegment(index=0, size_bytes=5_000_000, sha256="5" * 64, path=tmp_path / "seg0")]
    manager.finalize_artifact(1, ArtifactStatus.FINAL, segments, 2000)

    manager.pin_artifact(1)
    assert manager._pinned_bytes == 5_000_000

    # Unpin
    manager.unpin_artifact(1)
    artifact = manager.get_artifact(1)
    assert artifact.replay_state == ReplayState.PRESENT
    assert manager._pinned_bytes == 0


def test_evict_oldest_unpinned(tmp_path):
    """Evict oldest unpinned artifact."""
    manager = ReplayManager(
        storage_root=tmp_path,
        global_cap_bytes=100_000_000,
        pin_quota_bytes=50_000_000,
        max_artifact_bytes=10_000_000,
    )

    # Create 3 artifacts at different times
    assert manager.reserve_artifact(1, 1000) is True
    segments1 = [ReplaySegment(index=0, size_bytes=2_000_000, sha256="6" * 64, path=tmp_path / "seg1")]
    manager.finalize_artifact(1, ArtifactStatus.FINAL, segments1, 2000)

    assert manager.reserve_artifact(2, 3000) is True
    segments2 = [ReplaySegment(index=0, size_bytes=2_000_000, sha256="7" * 64, path=tmp_path / "seg2")]
    manager.finalize_artifact(2, ArtifactStatus.FINAL, segments2, 4000)

    assert manager.reserve_artifact(3, 5000) is True
    segments3 = [ReplaySegment(index=0, size_bytes=2_000_000, sha256="8" * 64, path=tmp_path / "seg3")]
    manager.finalize_artifact(3, ArtifactStatus.FINAL, segments3, 6000)

    # Pin artifact 2 (middle one)
    manager.pin_artifact(2)

    # Evict oldest unpinned (should be artifact 1, not 2)
    evicted_id = manager.evict_oldest_unpinned()
    assert evicted_id == 1

    artifact1 = manager.get_artifact(1)
    assert artifact1.replay_state == ReplayState.EVICTED

    artifact2 = manager.get_artifact(2)
    assert artifact2.replay_state == ReplayState.PINNED  # Still pinned

    # used_bytes should decrease by artifact 1's size
    assert manager._used_bytes == 4_000_000  # artifact 2 + 3


def test_evict_no_unpinned(tmp_path):
    """Evict returns None when all artifacts are pinned."""
    manager = ReplayManager(
        storage_root=tmp_path,
        global_cap_bytes=100_000_000,
        pin_quota_bytes=50_000_000,
        max_artifact_bytes=10_000_000,
    )

    assert manager.reserve_artifact(1, 1000) is True
    segments = [ReplaySegment(index=0, size_bytes=2_000_000, sha256="9" * 64, path=tmp_path / "seg0")]
    manager.finalize_artifact(1, ArtifactStatus.FINAL, segments, 2000)
    manager.pin_artifact(1)

    # Try to evict (nothing unpinned)
    evicted_id = manager.evict_oldest_unpinned()
    assert evicted_id is None


def test_get_stats(tmp_path):
    """Get storage statistics."""
    manager = ReplayManager(
        storage_root=tmp_path,
        global_cap_bytes=100_000_000,
        pin_quota_bytes=50_000_000,
        max_artifact_bytes=10_000_000,
    )

    manager.reserve_artifact(1, 1000)
    segments = [ReplaySegment(index=0, size_bytes=3_000_000, sha256="a" * 64, path=tmp_path / "seg0")]
    manager.finalize_artifact(1, ArtifactStatus.FINAL, segments, 2000)
    manager.pin_artifact(1)

    manager.reserve_artifact(2, 3000)
    segments2 = [ReplaySegment(index=0, size_bytes=4_000_000, sha256="b" * 64, path=tmp_path / "seg1")]
    manager.finalize_artifact(2, ArtifactStatus.FINAL, segments2, 4000)

    stats = manager.get_stats()
    assert stats["global_cap_bytes"] == 100_000_000
    assert stats["used_bytes"] == 7_000_000
    assert stats["available_bytes"] == 93_000_000
    assert stats["pinned_bytes"] == 3_000_000
    assert stats["pin_quota_bytes"] == 50_000_000
    assert stats["pin_available_bytes"] == 47_000_000
    assert stats["artifact_count"] == 2
    assert stats["present_count"] == 1
    assert stats["pinned_count"] == 1
    assert stats["evicted_count"] == 0


def test_compute_tree_hash():
    """Tree hash is deterministic from segment hashes."""
    artifact = ReplayArtifact(
        simulation_run_id=1,
        status=ArtifactStatus.FINAL,
        replay_state=ReplayState.PRESENT,
        segments=[
            ReplaySegment(index=0, size_bytes=1000, sha256="c" * 64, path=Path("/tmp/seg0")),
            ReplaySegment(index=1, size_bytes=2000, sha256="d" * 64, path=Path("/tmp/seg1")),
        ],
        total_bytes=3000,
        artifact_sha256=None,
        created_at_ns=1000,
        finalized_at_ns=2000,
    )

    hash1 = artifact.compute_tree_hash()
    hash2 = artifact.compute_tree_hash()
    assert hash1 == hash2  # Deterministic

    # Change segment order (should produce same hash due to sort)
    artifact.segments = [
        ReplaySegment(index=1, size_bytes=2000, sha256="d" * 64, path=Path("/tmp/seg1")),
        ReplaySegment(index=0, size_bytes=1000, sha256="c" * 64, path=Path("/tmp/seg0")),
    ]
    hash3 = artifact.compute_tree_hash()
    assert hash3 == hash1  # Same hash after reordering
