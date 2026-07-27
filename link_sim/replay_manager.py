"""Replay artifact management with quota, retention, and state tracking.

Section 9.3: Segmented self-contained replay artifacts with PRESENT/PINNED/EVICTED states.
Artifacts support deterministic replay after raw frame pruning.
"""

import hashlib
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional


class ReplayState(Enum):
    """Replay artifact state."""
    PRESENT = "PRESENT"  # Available for replay
    PINNED = "PINNED"    # User-pinned, protected from eviction
    EVICTED = "EVICTED"  # Removed, tombstone only


class ArtifactStatus(Enum):
    """Artifact completion status."""
    OPEN = "OPEN"                        # Recording in progress
    FINAL = "FINAL"                      # Successfully finalized
    INCOMPLETE_CRASH = "INCOMPLETE_CRASH"      # Process crash
    INCOMPLETE_STORAGE = "INCOMPLETE_STORAGE"  # Storage cap exhaustion


@dataclass
class ReplaySegment:
    """Replay segment metadata."""
    index: int
    size_bytes: int
    sha256: str
    path: Path


@dataclass
class ReplayArtifact:
    """Replay artifact metadata.

    Section 9.3: Self-contained segmented replay with deterministic structure.
    """
    simulation_run_id: int  # U64
    status: ArtifactStatus
    replay_state: ReplayState
    segments: List[ReplaySegment]
    total_bytes: int
    artifact_sha256: Optional[str]  # SHA-256 of tree hash (only when FINAL)
    created_at_ns: int
    finalized_at_ns: Optional[int]

    def compute_tree_hash(self) -> str:
        """Compute replay artifact SHA-256 from segment hashes.

        Section 9.3: SHA-256("link-replay-v1\\0" || for each segment: U64BE(size) || sha256_bytes).
        """
        hasher = hashlib.sha256()
        hasher.update(b"link-replay-v1\0")

        for segment in sorted(self.segments, key=lambda s: s.index):
            hasher.update(segment.size_bytes.to_bytes(8, byteorder='big'))
            hasher.update(bytes.fromhex(segment.sha256))

        return hasher.hexdigest()


class ReplayManager:
    """Manage replay artifacts with quota, retention, and pinning.

    Section 9.3: Global cap includes both pinned and unpinned artifacts.
    Pin quota is a logical quota within global cap.
    """

    def __init__(
        self,
        storage_root: Path,
        global_cap_bytes: int = 20 * 1024**3,  # 20 GiB
        pin_quota_bytes: int = 10 * 1024**3,   # 10 GiB
        max_artifact_bytes: int = 10 * 1024**3,  # 10 GiB per artifact
        retention_final_days: int = 30,
        retention_incomplete_days: int = 7,
    ):
        """Initialize replay manager.

        Args:
            storage_root: Root directory for replay artifacts
            global_cap_bytes: Total storage cap (includes pinned + unpinned)
            pin_quota_bytes: Quota for pinned artifacts (within global cap)
            max_artifact_bytes: Max size per artifact
            retention_final_days: Retention for FINAL artifacts
            retention_incomplete_days: Retention for INCOMPLETE_* artifacts
        """
        self.storage_root = Path(storage_root)
        self.global_cap_bytes = global_cap_bytes
        self.pin_quota_bytes = pin_quota_bytes
        self.max_artifact_bytes = max_artifact_bytes
        self.retention_final_days = retention_final_days
        self.retention_incomplete_days = retention_incomplete_days

        self._artifacts: Dict[int, ReplayArtifact] = {}  # run_id -> artifact
        self._used_bytes = 0
        self._pinned_bytes = 0

        # Ensure storage root exists
        self.storage_root.mkdir(parents=True, exist_ok=True)

    def reserve_artifact(self, simulation_run_id: int, current_time_ns: int) -> bool:
        """Reserve space for new artifact.

        Section 9.3: Reserve max_artifact_bytes before starting run.
        Returns False if insufficient capacity.
        """
        if self._used_bytes + self.max_artifact_bytes > self.global_cap_bytes:
            return False

        # Create artifact in OPEN state
        artifact = ReplayArtifact(
            simulation_run_id=simulation_run_id,
            status=ArtifactStatus.OPEN,
            replay_state=ReplayState.PRESENT,
            segments=[],
            total_bytes=0,
            artifact_sha256=None,
            created_at_ns=current_time_ns,
            finalized_at_ns=None,
        )
        self._artifacts[simulation_run_id] = artifact
        self._used_bytes += self.max_artifact_bytes  # Reserve full amount

        return True

    def finalize_artifact(
        self,
        simulation_run_id: int,
        status: ArtifactStatus,
        segments: List[ReplaySegment],
        current_time_ns: int,
    ) -> None:
        """Finalize artifact and release unused reservation.

        Section 9.3: Atomic finalize, compute tree hash for FINAL status.
        """
        if simulation_run_id not in self._artifacts:
            raise ValueError(f"Unknown artifact: {simulation_run_id:#018x}")

        artifact = self._artifacts[simulation_run_id]
        if artifact.status != ArtifactStatus.OPEN:
            raise ValueError(f"Artifact not OPEN: {artifact.status}")

        # Update artifact
        artifact.status = status
        artifact.segments = segments
        artifact.total_bytes = sum(s.size_bytes for s in segments)
        artifact.finalized_at_ns = current_time_ns

        if status == ArtifactStatus.FINAL:
            artifact.artifact_sha256 = artifact.compute_tree_hash()

        # Release unused reservation
        reserved = self.max_artifact_bytes
        actual = artifact.total_bytes
        if actual < reserved:
            self._used_bytes -= (reserved - actual)

    def pin_artifact(self, simulation_run_id: int) -> bool:
        """Pin artifact to protect from eviction.

        Returns False if pin quota exhausted.
        """
        if simulation_run_id not in self._artifacts:
            raise ValueError(f"Unknown artifact: {simulation_run_id:#018x}")

        artifact = self._artifacts[simulation_run_id]
        if artifact.replay_state == ReplayState.PINNED:
            return True  # Already pinned

        if artifact.replay_state == ReplayState.EVICTED:
            raise ValueError("Cannot pin evicted artifact")

        # Check pin quota
        if self._pinned_bytes + artifact.total_bytes > self.pin_quota_bytes:
            return False

        artifact.replay_state = ReplayState.PINNED
        self._pinned_bytes += artifact.total_bytes
        return True

    def unpin_artifact(self, simulation_run_id: int) -> None:
        """Unpin artifact, making it eligible for eviction."""
        if simulation_run_id not in self._artifacts:
            raise ValueError(f"Unknown artifact: {simulation_run_id:#018x}")

        artifact = self._artifacts[simulation_run_id]
        if artifact.replay_state != ReplayState.PINNED:
            return  # Not pinned

        artifact.replay_state = ReplayState.PRESENT
        self._pinned_bytes -= artifact.total_bytes

    def evict_oldest_unpinned(self) -> Optional[int]:
        """Evict oldest unpinned artifact.

        Section 9.3: Eviction converts to EVICTED tombstone.
        Returns run_id of evicted artifact, or None if nothing to evict.
        """
        # Find oldest unpinned PRESENT artifact
        candidates = [
            (run_id, artifact)
            for run_id, artifact in self._artifacts.items()
            if artifact.replay_state == ReplayState.PRESENT
        ]

        if not candidates:
            return None

        # Sort by creation time (oldest first)
        candidates.sort(key=lambda x: x[1].created_at_ns)
        run_id, artifact = candidates[0]

        return run_id if self.evict_artifact(run_id) else None

    def evict_artifact(self, simulation_run_id: int) -> bool:
        """Evict one PRESENT, unpinned artifact and retain its metadata tombstone."""

        artifact = self._artifacts.get(simulation_run_id)
        if artifact is None or artifact.replay_state != ReplayState.PRESENT:
            return False
        artifact_dir = self.storage_root / f"{simulation_run_id:016x}"
        if artifact_dir.exists():
            import shutil
            shutil.rmtree(artifact_dir)
        artifact.replay_state = ReplayState.EVICTED
        self._used_bytes -= artifact.total_bytes
        return True

    def evict_expired(
        self,
        current_time_ns: int,
        *,
        final_retention_days: int | None = None,
        incomplete_retention_days: int | None = None,
    ) -> tuple[int, ...]:
        """Evict only artifacts whose own retention deadline has passed."""

        final_days = self.retention_final_days if final_retention_days is None else final_retention_days
        incomplete_days = self.retention_incomplete_days if incomplete_retention_days is None else incomplete_retention_days
        if min(final_days, incomplete_days) <= 0:
            raise ValueError("replay retention days must be positive")
        candidates = []
        for run_id, artifact in self._artifacts.items():
            if artifact.replay_state != ReplayState.PRESENT:
                continue
            age_start = artifact.finalized_at_ns if artifact.finalized_at_ns is not None else artifact.created_at_ns
            days = final_days if artifact.status == ArtifactStatus.FINAL else incomplete_days
            if age_start + days * 86_400_000_000_000 <= current_time_ns:
                candidates.append((age_start, run_id))
        evicted: list[int] = []
        for _, run_id in sorted(candidates):
            if self.evict_artifact(run_id):
                evicted.append(run_id)
        return tuple(evicted)

    def get_artifact(self, simulation_run_id: int) -> Optional[ReplayArtifact]:
        """Get artifact metadata."""
        return self._artifacts.get(simulation_run_id)

    def get_stats(self) -> Dict:
        """Get storage statistics."""
        artifact_count = len(self._artifacts)
        present_count = sum(1 for a in self._artifacts.values() if a.replay_state == ReplayState.PRESENT)
        pinned_count = sum(1 for a in self._artifacts.values() if a.replay_state == ReplayState.PINNED)
        evicted_count = sum(1 for a in self._artifacts.values() if a.replay_state == ReplayState.EVICTED)

        return {
            "global_cap_bytes": self.global_cap_bytes,
            "used_bytes": self._used_bytes,
            "available_bytes": self.global_cap_bytes - self._used_bytes,
            "pinned_bytes": self._pinned_bytes,
            "pin_quota_bytes": self.pin_quota_bytes,
            "pin_available_bytes": self.pin_quota_bytes - self._pinned_bytes,
            "artifact_count": artifact_count,
            "present_count": present_count,
            "pinned_count": pinned_count,
            "evicted_count": evicted_count,
        }
