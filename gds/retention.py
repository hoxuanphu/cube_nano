"""Ground product/raw/replay retention and tombstone operations."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from protocol.schemas import ProductRef

from .product_store import ProductStore
from .storage import StorageGuard, StorageSnapshot
from .u64 import encode_sqlite_u64
from .writer import MutationPriority, SQLiteWriter


@dataclass(frozen=True)
class RetentionPolicy:
    raw_hours: int = 24
    rollup_days: int = 30
    product_days: int = 30
    tombstone_days: int = 90
    ground_cap_bytes: int = 20 * 1024 * 1024 * 1024
    high_watermark: float = 0.80
    hard_watermark: float = 0.90
    part_hours: int = 24
    log_days: int = 7
    replay_final_days: int = 30
    replay_incomplete_days: int = 7

    def __post_init__(self) -> None:
        if min(
            self.raw_hours,
            self.rollup_days,
            self.product_days,
            self.tombstone_days,
            self.part_hours,
            self.log_days,
            self.replay_final_days,
            self.replay_incomplete_days,
        ) <= 0:
            raise ValueError("retention durations must be positive")
        if not 0 < self.high_watermark < self.hard_watermark <= 1:
            raise ValueError("watermarks must be ordered")


@dataclass(frozen=True)
class TombstoneResult:
    product_ref: ProductRef
    status_code: int
    reason: str
    bundle_sha256: str | None
    retained_until_us: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "product_ref": self.product_ref.as_dict(),
            "status": "EVICTED",
            "status_code": self.status_code,
            "reason": self.reason,
            "bundle_sha256": self.bundle_sha256,
            "retained_until_us": self.retained_until_us,
        }


class RetentionManager:
    def __init__(
        self,
        writer: SQLiteWriter,
        product_store: ProductStore,
        *,
        storage_guard: StorageGuard | None = None,
        policy: RetentionPolicy | None = None,
        replay_manager: Any | None = None,
    ):
        self.writer = writer
        self.product_store = product_store
        self.storage_guard = storage_guard
        self.policy = policy or RetentionPolicy()
        self.replay_manager = replay_manager

    def pin(self, product_ref: ProductRef) -> None:
        self.writer.mutate(
            "pin_ground_product",
            lambda connection: connection.execute("UPDATE products SET pinned=1 WHERE spacecraft_instance_id=? AND origin_boot_id=? AND product_id=? AND state='PUBLISHED'", (encode_sqlite_u64(product_ref.spacecraft_instance_id), product_ref.origin_boot_id, product_ref.product_id)),
            priority=MutationPriority.HIGH,
        )

    def unpin(self, product_ref: ProductRef) -> None:
        self.writer.mutate(
            "unpin_ground_product",
            lambda connection: connection.execute("UPDATE products SET pinned=0 WHERE spacecraft_instance_id=? AND origin_boot_id=? AND product_id=?", (encode_sqlite_u64(product_ref.spacecraft_instance_id), product_ref.origin_boot_id, product_ref.product_id)),
            priority=MutationPriority.HIGH,
        )

    def evict_expired_products(self, now_us: int) -> tuple[TombstoneResult, ...]:
        rows_to_delete: list[tuple[ProductRef, str | None]] = []
        def mutation(connection):
            rows = connection.execute("SELECT spacecraft_instance_id,origin_boot_id,product_id,bundle_sha256,retention_until_us FROM products WHERE state='PUBLISHED' AND pinned=0 AND retention_until_us IS NOT NULL AND retention_until_us<=?", (now_us,)).fetchall()
            for row in rows:
                ref = ProductRef(int.from_bytes(bytes(row[0]), "big"), int(row[1]), int(row[2]))
                sha = None if row[3] is None else bytes(row[3]).hex()
                retained_until = now_us + self.policy.tombstone_days * 86_400_000_000
                connection.execute("UPDATE products SET state='EVICTED',local_path=NULL,evicted_at_us=?,eviction_reason='RETENTION_EXPIRED' WHERE spacecraft_instance_id=? AND origin_boot_id=? AND product_id=?", (now_us, encode_sqlite_u64(ref.spacecraft_instance_id), ref.origin_boot_id, ref.product_id))
                connection.execute("INSERT OR REPLACE INTO product_tombstones(spacecraft_instance_id,origin_boot_id,product_id,eviction_reason,bundle_sha256,retained_until_us) VALUES(?,?,?,?,?,?)", (encode_sqlite_u64(ref.spacecraft_instance_id), ref.origin_boot_id, ref.product_id, "RETENTION_EXPIRED", None if sha is None else bytes.fromhex(sha), retained_until))
                connection.execute("UPDATE product_artifacts SET state='EVICTED' WHERE spacecraft_instance_id=? AND origin_boot_id=? AND product_id=?", (encode_sqlite_u64(ref.spacecraft_instance_id), ref.origin_boot_id, ref.product_id))
                rows_to_delete.append((ref, sha))
            return len(rows)
        self.writer.mutate("evict_expired_products", mutation, priority=MutationPriority.HIGH)
        results = []
        for ref, sha in rows_to_delete:
            path = self.product_store.product_directory(ref)
            shutil.rmtree(path, ignore_errors=True)
            results.append(TombstoneResult(ref, 410, "RETENTION_EXPIRED", sha, now_us + self.policy.tombstone_days * 86_400_000_000))
        return tuple(results)

    def lookup_tombstone(self, product_ref: ProductRef, now_us: int) -> TombstoneResult | None:
        with self.writer.reader() as connection:
            row = connection.execute("SELECT eviction_reason,bundle_sha256,retained_until_us FROM product_tombstones WHERE spacecraft_instance_id=? AND origin_boot_id=? AND product_id=?", (encode_sqlite_u64(product_ref.spacecraft_instance_id), product_ref.origin_boot_id, product_ref.product_id)).fetchone()
        if row is None:
            return None
        if int(row[2]) <= now_us:
            self.writer.mutate("prune_product_tombstone", lambda connection: connection.execute("DELETE FROM product_tombstones WHERE spacecraft_instance_id=? AND origin_boot_id=? AND product_id=?", (encode_sqlite_u64(product_ref.spacecraft_instance_id), product_ref.origin_boot_id, product_ref.product_id)), priority=MutationPriority.HIGH)
            return None
        return TombstoneResult(product_ref, 410, str(row[0]), None if row[1] is None else bytes(row[1]).hex(), int(row[2]))

    def storage_snapshot(self) -> StorageSnapshot | None:
        return None if self.storage_guard is None else self.storage_guard.snapshot()

    @staticmethod
    def _file_matches(path: Path, suffixes: tuple[str, ...]) -> bool:
        name = path.name.lower()
        return any(
            name.endswith(suffix.lower())
            or f"{suffix.lower()}." in name
            for suffix in suffixes
        )

    def prune_files(
        self,
        roots: tuple[str | Path, ...] | list[str | Path],
        *,
        now_us: int,
        retention_us: int,
        suffixes: tuple[str, ...],
        max_delete_bytes: int | None = None,
    ) -> tuple[Path, ...]:
        """Remove expired rolling files without following symlinks."""

        if now_us < 0 or retention_us <= 0:
            raise ValueError("now_us must be non-negative and retention_us must be positive")
        if max_delete_bytes is not None and max_delete_bytes < 0:
            raise ValueError("max_delete_bytes must be non-negative")
        cutoff_ns = (now_us - retention_us) * 1_000
        removed: list[Path] = []
        deleted_bytes = 0
        for raw_root in roots:
            root = Path(raw_root).resolve()
            if not root.is_dir():
                continue
            for path in sorted(root.rglob("*")):
                if path.is_symlink() or not path.is_file() or not self._file_matches(path, suffixes):
                    continue
                try:
                    stat = path.stat()
                    if stat.st_mtime_ns > cutoff_ns:
                        continue
                    if max_delete_bytes is not None and deleted_bytes + stat.st_size > max_delete_bytes:
                        continue
                    path.unlink()
                    removed.append(path)
                    deleted_bytes += stat.st_size
                except FileNotFoundError:
                    continue
        return tuple(removed)

    def cleanup_files(
        self,
        *,
        now_us: int,
        raw_roots: tuple[str | Path, ...] = (),
        staging_roots: tuple[str | Path, ...] = (),
        log_roots: tuple[str | Path, ...] = (),
    ) -> dict[str, tuple[Path, ...]]:
        """Apply the local SIL raw/frame, staging and rotated-log retention policy."""

        return {
            "raw": self.prune_files(
                raw_roots,
                now_us=now_us,
                retention_us=self.policy.raw_hours * 3_600_000_000,
                suffixes=(".seg", ".raw", ".frame"),
            ),
            "staging": self.prune_files(
                staging_roots,
                now_us=now_us,
                retention_us=self.policy.part_hours * 3_600_000_000,
                suffixes=(".part",),
            ),
            "logs": self.prune_files(
                log_roots,
                now_us=now_us,
                retention_us=self.policy.log_days * 86_400_000_000,
                suffixes=(".log", ".jsonl"),
            ),
        }

    def evict_expired_replay(self, now_us: int) -> tuple[int, ...]:
        """Delegate replay state transitions to the self-contained replay manager."""

        if self.replay_manager is None:
            return ()
        evict = getattr(self.replay_manager, "evict_expired", None)
        if not callable(evict):
            return ()
        return tuple(
            int(run_id)
            for run_id in evict(
                now_us * 1_000,
                final_retention_days=self.policy.replay_final_days,
                incomplete_retention_days=self.policy.replay_incomplete_days,
            )
        )

    def prune_tombstones(self, now_us: int) -> int:
        return int(self.writer.mutate("prune_expired_tombstones", lambda connection: connection.execute("DELETE FROM product_tombstones WHERE retained_until_us<=?", (now_us,)).rowcount, priority=MutationPriority.HIGH))
