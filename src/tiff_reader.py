import math
import os
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from input_contract import load_input_sidecar, parse_channel_mapping
from resource_guards import (
    DiskAllocation,
    FilesystemInfoProvider,
    MemoryInfoProvider,
    ReaderBudget,
    require_disk_allocations,
    require_memory,
    require_writable_parents,
)


CANONICAL_BANDS = {
    3: ("red", "green", "blue"),
    4: ("red", "green", "blue", "nir"),
}


@dataclass
class ReaderMetrics:
    blocks_requested: int = 0
    blocks_decoded: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    read_latency_ms: list = field(default_factory=list, repr=False)

    def as_dict(self):
        if self.read_latency_ms:
            median = float(np.median(self.read_latency_ms))
            p95 = float(np.percentile(self.read_latency_ms, 95))
        else:
            median = p95 = 0.0
        return {
            "blocks_requested": self.blocks_requested,
            "blocks_decoded": self.blocks_decoded,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "read_calls": len(self.read_latency_ms),
            "read_latency_median_ms": median,
            "read_latency_p95_ms": p95,
        }


class ImageBlockReader(ABC):
    shape: tuple
    dtype: np.dtype
    axes: dict
    band_order: tuple
    metrics: ReaderMetrics

    @abstractmethod
    def read_rows(self, row_start, row_end):
        raise NotImplementedError

    @abstractmethod
    def physical_blocks(self, row_start, row_end):
        raise NotImplementedError

    @abstractmethod
    def close(self):
        raise NotImplementedError

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


def close_memmap(array):
    """Flush and explicitly close every memmap in an ndarray base chain."""
    current = array
    seen_arrays = set()
    mappings = []
    seen_mappings = set()
    while current is not None and id(current) not in seen_arrays:
        seen_arrays.add(id(current))
        mapping = getattr(current, "_mmap", None)
        if mapping is not None and id(mapping) not in seen_mappings:
            seen_mappings.add(id(mapping))
            mappings.append((current, mapping))
        current = getattr(current, "base", None)

    for owner, mapping in mappings:
        if not getattr(mapping, "closed", False):
            owner.flush()
    for _, mapping in mappings:
        if not getattr(mapping, "closed", False):
            mapping.close()


class TiffReader(ImageBlockReader):
    """Validated TIFF reader with session-scoped memmap or decoded cache."""

    def __init__(
        self,
        path,
        input_spec,
        *,
        read_mode="auto",
        cache_mode="auto",
        budget=None,
        cache_dir=None,
        series_index=None,
        level_index=None,
        channel_mapping=None,
        input_sidecar=None,
        patch_size=None,
        batch_size=1,
        memory_provider=None,
        filesystem_provider=None,
        disk_allocations=(),
    ):
        self.path = Path(path)
        self.input_spec = input_spec
        self.read_mode = str(read_mode).lower()
        self.cache_mode = str(cache_mode).lower()
        self.budget = budget or ReaderBudget.from_cli()
        self.cache_dir = Path(cache_dir) if cache_dir is not None else self.path.parent / ".cube_nano-cache"
        self.requested_series_index = series_index
        self.requested_level_index = level_index
        self._channel_mapping_value = channel_mapping
        self._input_sidecar_path = input_sidecar
        self.patch_size = int(patch_size or input_spec.patch_size)
        self.batch_size = int(batch_size)
        self.memory_provider = memory_provider or MemoryInfoProvider()
        self.filesystem_provider = filesystem_provider or FilesystemInfoProvider()
        self._disk_allocations = disk_allocations

        self.metrics = ReaderMetrics()
        self.backend = None
        self.provenance = {}
        self.owns_source_cache = False
        self.source_cache_path = None
        self._tiff = None
        self._array = None
        self._closed = False
        self._decoded_block_count = 0

        self._validate_options()
        try:
            self._open_and_validate()
            self._select_backend()
        except Exception:
            self.close()
            raise

    def _validate_options(self):
        if self.read_mode not in {"auto", "stream", "full"}:
            raise ValueError("tiff_read_mode must be one of: auto, stream, full")
        if self.cache_mode not in {"auto", "ram", "disk"}:
            raise ValueError("tiff_cache_mode must be one of: auto, ram, disk")
        if self.read_mode == "stream" and self.cache_mode != "auto":
            raise ValueError("stream mode requires tiff_cache_mode=auto")
        if self.patch_size <= 0 or self.batch_size <= 0:
            raise ValueError("patch_size and batch_size must be greater than zero")
        for name, value in (
            ("tiff_series", self.requested_series_index),
            ("tiff_level", self.requested_level_index),
        ):
            if value is not None and (not isinstance(value, int) or value < 0):
                raise ValueError(f"{name} must be a non-negative integer")

    def _open_and_validate(self):
        try:
            import tifffile
        except ImportError as exc:
            raise ImportError("Reading TIFF inputs requires tifffile") from exc

        self._tifffile_module = tifffile
        self._tiff = tifffile.TiffFile(self.path)
        self.series_index = self._select_unique_index(
            "series",
            self.requested_series_index,
            len(self._tiff.series),
        )
        base_series = self._tiff.series[self.series_index]
        levels = tuple(base_series.levels)
        self.level_index = self._select_unique_index(
            "pyramid level",
            self.requested_level_index,
            len(levels),
        )
        self._series = levels[self.level_index]
        if len(self._series.pages) != 1:
            raise ValueError(
                "Selected TIFF series/level spans multiple pages and is ambiguous for classification"
            )
        self._page = self._series.pages[0]

        self.original_axes = str(self._series.axes)
        self.original_shape = tuple(int(value) for value in self._series.shape)
        self.dtype = np.dtype(self._series.dtype)
        self._axis_positions, self._physical_channels = self._validate_axes(
            self.original_axes,
            self.original_shape,
        )
        self._validate_page_metadata()
        self._resolve_channel_mapping()

        height = self.original_shape[self._axis_positions["Y"]]
        width = self.original_shape[self._axis_positions["X"]]
        self.shape = (height, width, len(self._channel_indices))
        self.axes = {"original": self.original_axes, "normalized": "YXC"}
        self.band_order = tuple(self.input_spec.band_order)
        if self.shape[2] != self.input_spec.channels:
            raise ValueError(
                f"TIFF reader outputs {self.shape[2]} channels but engine requires "
                f"{self.input_spec.channels}"
            )

        self._block_geometry = self._get_block_geometry()
        self._decoded_block_count = len(self._page.dataoffsets)
        self._decoder_peak_bytes = self._estimate_decoder_peak()
        self._batch_peak_bytes = self._estimate_batch_peak()
        self._decoded_bytes = math.prod(self.original_shape) * self.dtype.itemsize
        self.decoded_bytes = self._decoded_bytes
        self.provenance = {
            "backend_library": "tifffile",
            "backend_version": tifffile.__version__,
            "compression": getattr(self._page.compression, "name", str(self._page.compression)),
            "series": self.series_index,
            "level": self.level_index,
            "axes": self.original_axes,
            "shape": self.original_shape,
            "dtype": str(self.dtype),
            "is_tiled": bool(self._page.is_tiled),
            "block_height": self._block_geometry[0],
            "block_width": self._block_geometry[1],
            "decoded_bytes": self._decoded_bytes,
        }

    @staticmethod
    def _select_unique_index(label, requested, count):
        if count <= 0:
            raise ValueError(f"TIFF contains no {label}")
        if requested is None:
            if count != 1:
                raise ValueError(
                    f"TIFF contains {count} {label} entries; select one explicitly"
                )
            return 0
        if requested >= count:
            raise ValueError(f"Requested {label} index {requested} is out of range for {count} entries")
        return requested

    @staticmethod
    def _validate_axes(axes, shape):
        if len(axes) != len(shape):
            raise ValueError(f"TIFF axes {axes!r} do not match shape {shape}")
        positions = {}
        for axis in ("Y", "X"):
            indexes = [index for index, value in enumerate(axes) if value == axis]
            if len(indexes) != 1:
                raise ValueError(f"TIFF axes must contain exactly one {axis!r} axis, got {axes!r}")
            positions[axis] = indexes[0]

        channel_indexes = [index for index, value in enumerate(axes) if value in {"C", "S"}]
        if len(channel_indexes) != 1:
            raise ValueError(f"TIFF axes must contain exactly one channel/sample axis, got {axes!r}")
        positions["C"] = channel_indexes[0]

        supported = set(positions.values())
        for index, (axis, size) in enumerate(zip(axes, shape)):
            if index not in supported and size != 1:
                raise ValueError(
                    f"Unsupported non-singleton TIFF axis {axis!r} with size {size} in {axes!r}"
                )
        channels = shape[positions["C"]]
        if channels not in (3, 4):
            raise ValueError(f"TIFF must contain 3 or 4 physical channels, got {channels}")
        return positions, channels

    def _validate_page_metadata(self):
        photometric = int(self._page.photometric)
        samples = int(self._page.samplesperpixel)
        planar = int(self._page.planarconfig)
        extras = tuple(int(value) for value in self._page.extrasamples)

        if photometric not in {1, 2}:  # MINISBLACK or RGB
            raise ValueError(
                f"Unsupported TIFF PhotometricInterpretation {self._page.photometric!s}"
            )
        if samples != self._physical_channels:
            raise ValueError(
                f"SamplesPerPixel={samples} does not match channel axis size "
                f"{self._physical_channels}"
            )
        if planar not in {1, 2}:  # CONTIG or SEPARATE
            raise ValueError(f"Unsupported TIFF PlanarConfiguration {self._page.planarconfig!s}")
        if any(value in {1, 2} for value in extras):
            raise ValueError("TIFF alpha samples are not valid RGBNIR channels")
        if any(value != 0 for value in extras):
            raise ValueError(f"Unsupported TIFF ExtraSamples values {self._page.extrasamples!r}")

        self._photometric = photometric
        self._planar = planar
        self._extra_samples = extras

    def _resolve_channel_mapping(self):
        expected_roles = tuple(self.input_spec.band_order)
        if expected_roles != CANONICAL_BANDS.get(self.input_spec.channels):
            raise ValueError("Engine input band order must be canonical RGB or RGBNIR")

        cli_mapping = parse_channel_mapping(self._channel_mapping_value)
        sidecar = None
        sidecar_mapping = None
        if self._input_sidecar_path is not None:
            sidecar = load_input_sidecar(self._input_sidecar_path, self.path)
            if sidecar.axes != self.original_axes:
                raise ValueError("Input sidecar axes do not match the selected TIFF series")
            if sidecar.shape != self.original_shape:
                raise ValueError("Input sidecar shape does not match the selected TIFF series")
            if sidecar.dtype != self.dtype:
                raise ValueError("Input sidecar dtype does not match the selected TIFF series")
            if len(sidecar.band_order) != self._physical_channels:
                raise ValueError("Input sidecar band_order does not match TIFF channel count")
            if sidecar.input_spec_id != self.input_spec.input_spec_id:
                raise ValueError("Input sidecar input_spec_id does not match engine manifest")
            if sidecar.normalization_id != self.input_spec.normalization.id:
                raise ValueError("Input sidecar normalization does not match engine manifest")
            sidecar_mapping = {
                role: sidecar.role_to_index[role]
                for role in expected_roles
                if role in sidecar.role_to_index
            }

        if cli_mapping is not None and set(cli_mapping) != set(expected_roles):
            raise ValueError(
                f"channel_mapping must define exactly these roles: {', '.join(expected_roles)}"
            )
        if sidecar_mapping is not None and set(sidecar_mapping) != set(expected_roles):
            raise ValueError("Input sidecar does not define every band required by the engine")
        if cli_mapping is not None and sidecar_mapping is not None and cli_mapping != sidecar_mapping:
            raise ValueError("channel_mapping conflicts with input sidecar band_order")

        mapping = cli_mapping or sidecar_mapping
        has_explicit_mapping = mapping is not None
        if mapping is None:
            if self._photometric != 2 or self._physical_channels != 3 or self._extra_samples:
                raise ValueError(
                    "Ambiguous TIFF channel semantics; provide channel_mapping or input_sidecar"
                )
            if expected_roles != CANONICAL_BANDS[3]:
                raise ValueError("A four-channel engine requires explicit RGBNIR metadata")
            mapping = {role: index for index, role in enumerate(expected_roles)}

        if len(self._extra_samples) > 1 and (
            not has_explicit_mapping
            or set(mapping.values()) != set(range(self._physical_channels))
        ):
            raise ValueError(
                "Multiple unspecified extra samples require an explicit mapping for every sample"
            )

        indexes = tuple(mapping[role] for role in expected_roles)
        if len(set(indexes)) != len(indexes) or any(
            index < 0 or index >= self._physical_channels for index in indexes
        ):
            raise ValueError("channel_mapping contains duplicate or out-of-range indexes")
        if self._photometric == 2 and self._physical_channels == 3 and indexes != (0, 1, 2):
            raise ValueError("channel_mapping conflicts with TIFF RGB photometric semantics")
        if self._extra_samples == (0,) and self._physical_channels == 4:
            nir_index = mapping.get("nir")
            if nir_index != 3:
                raise ValueError("The fourth unspecified RGB sample must be explicitly mapped to NIR")

        self._sidecar = sidecar
        self._channel_indices = indexes

    def _get_block_geometry(self):
        block_height = int(
            self._page.tilelength if self._page.is_tiled else self._page.rowsperstrip
        )
        block_width = int(self._page.tilewidth if self._page.is_tiled else self._page.imagewidth)
        blocks_x = math.ceil(int(self._page.imagewidth) / block_width)
        blocks_y = math.ceil(int(self._page.imagelength) / block_height)
        planes = self._physical_channels if self._planar == 2 else 1
        return block_height, block_width, blocks_x, blocks_y, planes

    def _estimate_decoder_peak(self):
        compressed = max((int(value) for value in self._page.databytecounts), default=0)
        block_height, block_width, _, _, planes = self._block_geometry
        samples_per_block = 1 if planes > 1 else self._physical_channels
        decoded_block = block_height * block_width * samples_per_block * self.dtype.itemsize
        strip_bytes = (
            min(self.patch_size, self.original_shape[self._axis_positions["Y"]])
            * self.original_shape[self._axis_positions["X"]]
            * self._physical_channels
            * self.dtype.itemsize
        )
        canonical_layout = self.original_axes in {"YXC", "YXS"}
        identity_mapping = self._channel_indices == tuple(range(len(self._channel_indices)))
        layout_copy = 0 if canonical_layout and identity_mapping and self.dtype.isnative else strip_bytes
        return self.budget.decoder_workers * (compressed + decoded_block + layout_copy)

    def _estimate_batch_peak(self):
        height, width, channels = self.shape
        raw_strip = min(self.patch_size, height) * width * self._physical_channels * self.dtype.itemsize
        normalized_batch = self.batch_size * channels * self.patch_size * self.patch_size * 4
        reorder_copy = raw_strip if self._channel_indices != tuple(range(channels)) else 0
        return raw_strip + reorder_copy + normalized_batch + normalized_batch

    def _select_backend(self):
        if self.read_mode != "full":
            try:
                self._array = self._tifffile_module.memmap(
                    self.path,
                    series=self.series_index,
                    level=self.level_index,
                    mode="r",
                )
                self._validate_decoded_array(self._array)
                require_memory(
                    self._batch_peak_bytes,
                    self.budget,
                    self.memory_provider,
                    "TIFF memmap inference working set",
                )
                self.backend = "memmap"
                self._guard_disk_allocations(include_source_cache=False)
                return
            except (OSError, ValueError):
                if self._array is not None:
                    close_memmap(self._array)
                    self._array = None

        if self.read_mode == "stream":
            raise RuntimeError(
                "TIFF is not memory-mappable and no true block backend has passed the stream exit gate"
            )

        self._require_decoder()
        selected_cache = self._select_decoded_cache()
        self._guard_disk_allocations(include_source_cache=selected_cache == "disk")
        if selected_cache == "ram":
            self._array = self._tiff.asarray(
                series=self.series_index,
                level=self.level_index,
                maxworkers=self.budget.decoder_workers,
            )
        else:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            file_descriptor, cache_name = tempfile.mkstemp(
                prefix="source_",
                suffix=".dat",
                dir=self.cache_dir,
            )
            os.close(file_descriptor)
            self.source_cache_path = Path(cache_name)
            self.owns_source_cache = True
            target = np.memmap(
                self.source_cache_path,
                mode="w+",
                dtype=self.dtype,
                shape=self.original_shape,
            )
            try:
                self._array = self._tiff.asarray(
                    series=self.series_index,
                    level=self.level_index,
                    out=target,
                    maxworkers=self.budget.decoder_workers,
                )
            except Exception:
                close_memmap(target)
                raise

        self._validate_decoded_array(self._array)
        self.backend = selected_cache
        self.metrics.blocks_decoded = self._decoded_block_count
        self.metrics.cache_misses = self._decoded_block_count

    def _require_decoder(self):
        compression = int(self._page.compression)
        try:
            self._tifffile_module.TIFF.DECOMPRESSORS[compression]
        except KeyError as exc:
            name = getattr(self._page.compression, "name", str(self._page.compression))
            raise RuntimeError(f"TIFF codec {name} is unavailable: {exc}") from exc

    def _select_decoded_cache(self):
        ram_error = None
        if self.cache_mode in {"auto", "ram"}:
            try:
                if self._decoded_bytes > self.budget.max_ram_cache_bytes:
                    raise MemoryError(
                        f"Decoded TIFF size {self._decoded_bytes} exceeds RAM cache limit "
                        f"{self.budget.max_ram_cache_bytes}"
                    )
                require_memory(
                    self._decoded_bytes + self._decoder_peak_bytes + self._batch_peak_bytes,
                    self.budget,
                    self.memory_provider,
                    "TIFF RAM decoded cache",
                )
                return "ram"
            except MemoryError as exc:
                ram_error = exc
                if self.cache_mode == "ram":
                    raise

        if self._decoded_bytes > self.budget.max_disk_cache_bytes:
            disk_error = MemoryError(
                f"Decoded TIFF size {self._decoded_bytes} exceeds disk cache limit "
                f"{self.budget.max_disk_cache_bytes}"
            )
        else:
            try:
                # The current tifffile out=memmap path cannot enforce a mapped-window bound.
                require_memory(
                    self._decoder_peak_bytes + self._decoded_bytes + self._batch_peak_bytes,
                    self.budget,
                    self.memory_provider,
                    "TIFF disk decoded cache",
                )
                return "disk"
            except MemoryError as exc:
                disk_error = exc
        if self.cache_mode == "disk":
            raise disk_error
        raise MemoryError(
            f"No decoded TIFF cache satisfies the resource guards. RAM: {ram_error}; "
            f"disk: {disk_error}"
        )

    def _guard_disk_allocations(self, include_source_cache):
        if callable(self._disk_allocations):
            allocations = list(self._disk_allocations(self))
        else:
            allocations = list(self._disk_allocations)
        if include_source_cache:
            allocations.append(
                DiskAllocation(
                    self.cache_dir / "source-cache.dat",
                    self._decoded_bytes,
                    "decoded TIFF source cache",
                )
            )
        if allocations:
            require_writable_parents(allocations)
            require_disk_allocations(allocations, provider=self.filesystem_provider)

    def _validate_decoded_array(self, array):
        if tuple(array.shape) != self.original_shape:
            raise ValueError(
                f"TIFF backend returned shape {array.shape}, expected {self.original_shape}"
            )
        if np.dtype(array.dtype) != self.dtype:
            raise ValueError(f"TIFF backend returned dtype {array.dtype}, expected {self.dtype}")

    def physical_blocks(self, row_start, row_end):
        self._validate_row_range(row_start, row_end)
        block_height, _, blocks_x, blocks_y, planes = self._block_geometry
        first_row = row_start // block_height
        last_row = (row_end - 1) // block_height
        page_index = int(getattr(self._page, "index", 0))
        blocks_per_plane = blocks_x * blocks_y
        keys = []
        for plane in range(planes):
            for block_row in range(first_row, last_row + 1):
                for block_column in range(blocks_x):
                    block_index = plane * blocks_per_plane + block_row * blocks_x + block_column
                    keys.append(
                        (
                            self.series_index,
                            self.level_index,
                            page_index,
                            plane,
                            block_index,
                        )
                    )
        return tuple(keys)

    def _validate_row_range(self, row_start, row_end):
        if not isinstance(row_start, int) or not isinstance(row_end, int):
            raise TypeError("row_start and row_end must be integers")
        if row_start < 0 or row_end <= row_start or row_end > self.shape[0]:
            raise ValueError(f"Invalid TIFF row range [{row_start}, {row_end}) for height {self.shape[0]}")

    def read_rows(self, row_start, row_end):
        self._validate_row_range(row_start, row_end)
        if self._closed or self._array is None:
            raise RuntimeError("TIFF reader is closed")
        started = time.perf_counter()

        blocks = self.physical_blocks(row_start, row_end)
        self.metrics.blocks_requested += len(blocks)
        if self.backend in {"ram", "disk"}:
            self.metrics.cache_hits += len(blocks)

        selectors = []
        remaining_axes = []
        for index, axis in enumerate(self.original_axes):
            if index == self._axis_positions["Y"]:
                selectors.append(slice(row_start, row_end))
                remaining_axes.append("Y")
            elif index == self._axis_positions["X"]:
                selectors.append(slice(None))
                remaining_axes.append("X")
            elif index == self._axis_positions["C"]:
                selectors.append(slice(None))
                remaining_axes.append("C")
            else:
                selectors.append(0)

        strip = np.asarray(self._array[tuple(selectors)])
        order = [remaining_axes.index(axis) for axis in ("Y", "X", "C")]
        if order != [0, 1, 2]:
            strip = np.transpose(strip, order)
        if self._channel_indices != tuple(range(self._physical_channels)):
            strip = np.take(strip, self._channel_indices, axis=2)
        elif len(self._channel_indices) != self._physical_channels:
            strip = np.take(strip, self._channel_indices, axis=2)
        result = np.asarray(strip)
        self.metrics.read_latency_ms.append((time.perf_counter() - started) * 1000.0)
        return result

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._array is not None:
            close_memmap(self._array)
            self._array = None
        if self._tiff is not None:
            self._tiff.close()
            self._tiff = None
        if self.owns_source_cache and self.source_cache_path is not None:
            try:
                self.source_cache_path.unlink(missing_ok=True)
            finally:
                self.source_cache_path = None
