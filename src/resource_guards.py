import ctypes
import math
import os
import shutil
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path


MIB = 1024**2
GIB = 1024**3


def parse_size_bytes(value, unit_bytes, name, allow_zero=False):
    try:
        decimal_value = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not decimal_value.is_finite():
        raise ValueError(f"{name} must be a finite number")
    if decimal_value < 0 or (decimal_value == 0 and not allow_zero):
        comparator = "non-negative" if allow_zero else "greater than zero"
        raise ValueError(f"{name} must be {comparator}")
    byte_value = int(decimal_value * unit_bytes)
    if byte_value == 0 and decimal_value > 0:
        raise ValueError(f"{name} is too small to represent in bytes")
    return byte_value


@dataclass(frozen=True)
class ReaderBudget:
    max_ram_cache_bytes: int
    max_disk_cache_bytes: int
    runtime_reserve_bytes: int
    block_cache_bytes: int
    decoder_workers: int = 1
    mapped_readahead_bytes: int = 8 * MIB

    def __post_init__(self):
        if self.max_ram_cache_bytes <= 0:
            raise ValueError("max_ram_cache_bytes must be greater than zero")
        if self.max_disk_cache_bytes <= 0:
            raise ValueError("max_disk_cache_bytes must be greater than zero")
        if self.runtime_reserve_bytes <= 0:
            raise ValueError("runtime_reserve_bytes must be greater than zero")
        if self.block_cache_bytes < 0:
            raise ValueError("block_cache_bytes must be non-negative")
        if self.decoder_workers != 1:
            raise ValueError("Phase 1 requires decoder_workers=1 for a verifiable memory bound")
        if self.mapped_readahead_bytes < 0:
            raise ValueError("mapped_readahead_bytes must be non-negative")

    @classmethod
    def from_cli(
        cls,
        max_ram_cache_gib=0.5,
        max_disk_cache_gib=8.0,
        runtime_reserve_gib=1.5,
        tiff_block_cache_mib=64,
    ):
        return cls(
            max_ram_cache_bytes=parse_size_bytes(
                max_ram_cache_gib,
                GIB,
                "max_ram_cache_gib",
            ),
            max_disk_cache_bytes=parse_size_bytes(
                max_disk_cache_gib,
                GIB,
                "max_disk_cache_gib",
            ),
            runtime_reserve_bytes=parse_size_bytes(
                runtime_reserve_gib,
                GIB,
                "runtime_reserve_gib",
            ),
            block_cache_bytes=parse_size_bytes(
                tiff_block_cache_mib,
                MIB,
                "tiff_block_cache_mib",
                allow_zero=True,
            ),
        )


class MemoryInfoProvider:
    def available_bytes(self):
        meminfo = Path("/proc/meminfo")
        if meminfo.is_file():
            for line in meminfo.read_text(encoding="ascii").splitlines():
                if line.startswith("MemAvailable:"):
                    fields = line.split()
                    return int(fields[1]) * 1024
            raise RuntimeError("/proc/meminfo does not contain MemAvailable")

        if os.name == "nt":
            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(MemoryStatus)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                raise OSError("GlobalMemoryStatusEx failed")
            return int(status.ullAvailPhys)

        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages) * int(page_size)


class FixedMemoryInfoProvider:
    def __init__(self, available_bytes):
        self._available_bytes = int(available_bytes)

    def available_bytes(self):
        return self._available_bytes


def require_memory(operation_peak_without_reserve, budget, provider, purpose):
    operation_peak_without_reserve = int(operation_peak_without_reserve)
    available = int(provider.available_bytes())
    required = operation_peak_without_reserve + budget.runtime_reserve_bytes
    if required > available:
        raise MemoryError(
            f"Insufficient available RAM for {purpose}: required {required} bytes "
            f"including runtime reserve, available {available} bytes"
        )
    return required


@dataclass(frozen=True)
class FilesystemStats:
    device_id: object
    total_bytes: int
    free_bytes: int


class FilesystemInfoProvider:
    def stats_for(self, path):
        candidate = Path(path).expanduser().resolve(strict=False)
        parent = candidate if candidate.is_dir() else candidate.parent
        while not parent.exists():
            if parent == parent.parent:
                raise FileNotFoundError(f"No existing parent for path {path}")
            parent = parent.parent
        usage = shutil.disk_usage(parent)
        device_id = parent.stat().st_dev
        if os.name == "nt":
            device_id = (parent.drive or parent.anchor).lower()
        return FilesystemStats(device_id, int(usage.total), int(usage.free))


@dataclass(frozen=True)
class DiskAllocation:
    path: Path
    required_bytes: int
    purpose: str

    def __post_init__(self):
        if int(self.required_bytes) < 0:
            raise ValueError("Disk allocation bytes must be non-negative")


def require_writable_parents(allocations):
    checked = set()
    for allocation in allocations:
        candidate = Path(allocation.path).expanduser().resolve(strict=False)
        parent = candidate if candidate.is_dir() else candidate.parent
        while not parent.exists():
            if parent == parent.parent:
                raise FileNotFoundError(f"No existing parent for path {allocation.path}")
            parent = parent.parent
        if parent in checked:
            continue
        checked.add(parent)
        if not os.access(parent, os.W_OK):
            raise PermissionError(
                f"Directory is not writable for {allocation.purpose}: {parent}"
            )


def require_disk_allocations(
    allocations,
    provider=None,
    minimum_headroom_bytes=GIB,
    headroom_ratio=0.10,
):
    provider = provider or FilesystemInfoProvider()
    grouped = {}
    for allocation in allocations:
        if allocation.required_bytes == 0:
            continue
        stats = provider.stats_for(allocation.path)
        entry = grouped.setdefault(
            stats.device_id,
            {"stats": stats, "required": 0, "purposes": []},
        )
        entry["required"] += int(allocation.required_bytes)
        entry["purposes"].append(allocation.purpose)

    requirements = {}
    for device_id, entry in grouped.items():
        stats = entry["stats"]
        headroom = max(
            int(minimum_headroom_bytes),
            int(math.ceil(stats.total_bytes * headroom_ratio)),
        )
        required = entry["required"] + headroom
        if stats.free_bytes < required:
            purposes = ", ".join(entry["purposes"])
            raise OSError(
                f"Insufficient disk space on device {device_id!r} for {purposes}: "
                f"required {required} bytes including headroom, free {stats.free_bytes} bytes"
            )
        requirements[device_id] = required
    return requirements
