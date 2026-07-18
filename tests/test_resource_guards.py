import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from resource_guards import (  # noqa: E402
    DiskAllocation,
    FilesystemStats,
    FixedMemoryInfoProvider,
    GIB,
    MIB,
    ReaderBudget,
    parse_size_bytes,
    require_disk_allocations,
    require_memory,
    require_writable_parents,
)


class _FilesystemProvider:
    def __init__(self, records):
        self.records = records

    def stats_for(self, path):
        return self.records[Path(path).parts[0]]


class ResourceGuardTests(unittest.TestCase):
    def test_cli_sizes_are_converted_to_integer_bytes_and_invalid_values_fail(self):
        self.assertEqual(parse_size_bytes("0.5", GIB, "size"), GIB // 2)
        self.assertEqual(parse_size_bytes("64", MIB, "size"), 64 * MIB)
        self.assertEqual(parse_size_bytes("0", MIB, "size", allow_zero=True), 0)

        for value in ("0", "-1", "nan", "inf", "-inf"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_size_bytes(value, GIB, "size")

    def test_runtime_reserve_is_counted_exactly_once_at_boundary(self):
        budget = ReaderBudget(
            max_ram_cache_bytes=GIB,
            max_disk_cache_bytes=GIB,
            runtime_reserve_bytes=100,
            block_cache_bytes=0,
        )
        provider = FixedMemoryInfoProvider(1100)

        self.assertEqual(require_memory(1000, budget, provider, "test"), 1100)
        with self.assertRaises(MemoryError):
            require_memory(1001, budget, provider, "test")

    def test_disk_requirements_group_paths_by_device_and_add_headroom_once(self):
        stats_a = FilesystemStats("device-a", 10_000, 10_000)
        stats_b = FilesystemStats("device-b", 20_000, 20_000)
        provider = _FilesystemProvider({"cache": stats_a, "output": stats_b})
        allocations = [
            DiskAllocation(Path("cache/source.dat"), 1000, "source"),
            DiskAllocation(Path("cache/mask.dat"), 2000, "mask"),
            DiskAllocation(Path("output/result.tif"), 3000, "output"),
        ]

        requirements = require_disk_allocations(
            allocations,
            provider=provider,
            minimum_headroom_bytes=100,
            headroom_ratio=0.10,
        )

        self.assertEqual(requirements["device-a"], 4000)
        self.assertEqual(requirements["device-b"], 5000)

    def test_each_filesystem_must_pass_independently(self):
        provider = _FilesystemProvider(
            {
                "cache": FilesystemStats("cache-device", 10_000, 10_000),
                "output": FilesystemStats("output-device", 10_000, 500),
            }
        )

        with self.assertRaisesRegex(OSError, "output-device"):
            require_disk_allocations(
                [
                    DiskAllocation(Path("cache/source.dat"), 1000, "source"),
                    DiskAllocation(Path("output/result.tif"), 1000, "output"),
                ],
                provider=provider,
                minimum_headroom_bytes=100,
                headroom_ratio=0,
            )

    def test_unwritable_parent_is_rejected_before_allocation(self):
        allocation = DiskAllocation(ROOT / "cache" / "source.dat", 1000, "source")
        with mock.patch("resource_guards.os.access", return_value=False):
            with self.assertRaisesRegex(PermissionError, "not writable"):
                require_writable_parents([allocation])


if __name__ == "__main__":
    unittest.main()
