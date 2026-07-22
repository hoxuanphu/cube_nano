import numpy as np

LEGACY_DEV_ONLY = True
import time
import math
import argparse
import os
import tempfile
from pathlib import Path
from uuid import uuid4

from input_contract import legacy_input_spec, load_engine_manifest
from resource_guards import (
    DiskAllocation,
    FilesystemInfoProvider,
    ReaderBudget,
    require_disk_allocations,
    require_writable_parents,
)
from tiff_reader import ImageBlockReader, ReaderMetrics, TiffReader, close_memmap

TIFF_EXTENSIONS = {".tif", ".tiff"}
PIL_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jp2", ".j2k", ".j2c"}
NUMPY_EXTENSIONS = {".npy", ".npz"}
HDF4_EXTENSIONS = {".h4", ".hdf4"}
HDF5_EXTENSIONS = {".h5", ".hdf5", ".hdf"}
NETCDF_EXTENSIONS = {".nc", ".netcdf"}
SUPPORTED_IMAGE_EXTENSIONS = sorted(
    TIFF_EXTENSIONS
    | PIL_EXTENSIONS
    | NUMPY_EXTENSIONS
    | HDF4_EXTENSIONS
    | HDF5_EXTENSIONS
    | NETCDF_EXTENSIONS
)
HDF4_MAGIC = b"\x0e\x03\x13\x01"


def _missing_dependency(package_name, format_name):
    return ImportError(
        f"Reading {format_name} inputs requires {package_name}. "
        f"Install it first, then retry this command."
    )


def _read_tiff(path):
    try:
        import tifffile as tiff
    except ImportError as exc:
        raise _missing_dependency("tifffile", "TIFF/GeoTIFF") from exc

    return tiff.imread(path)


def _write_tiff(path, image):
    try:
        import tifffile as tiff
    except ImportError as exc:
        raise _missing_dependency("tifffile", "TIFF output") from exc

    tiff.imwrite(path, image)


def _select_array_from_mapping(mapping, array_key, path):
    if array_key:
        if array_key not in mapping:
            available = ", ".join(mapping.keys())
            raise KeyError(f"Array key '{array_key}' not found in {path}. Available keys: {available}")
        return mapping[array_key]

    keys = list(mapping.keys())
    if len(keys) == 1:
        return mapping[keys[0]]

    preferred_keys = ["image", "arr_0", "data"]
    for key in preferred_keys:
        if key in mapping:
            return mapping[key]

    available = ", ".join(keys)
    raise ValueError(
        f"{path} contains multiple arrays. Use --array_key to select one. "
        f"Available keys: {available}"
    )


def _read_numpy(path, array_key=None):
    loaded = np.load(path)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        with loaded:
            return np.asarray(_select_array_from_mapping(loaded, array_key, path))
    return np.asarray(loaded)


def _read_with_pillow(path):
    try:
        from PIL import Image
    except ImportError as exc:
        raise _missing_dependency("Pillow", "PNG/JPEG/BMP/WebP/JPEG2000") from exc

    with Image.open(path) as image:
        if image.mode in ("1", "P", "CMYK", "YCbCr"):
            image = image.convert("RGB")
        elif image.mode == "LA":
            image = image.convert("RGBA")
        return np.asarray(image)


def _collect_hdf5_datasets(group, prefix=""):
    datasets = {}
    for key, value in group.items():
        dataset_path = f"{prefix}/{key}" if prefix else key
        if hasattr(value, "shape"):
            datasets[dataset_path] = value
        else:
            datasets.update(_collect_hdf5_datasets(value, dataset_path))
    return datasets


def _read_hdf5(path, array_key=None):
    try:
        import h5py
    except ImportError as exc:
        raise _missing_dependency("h5py", "HDF5/HDF") from exc

    with h5py.File(path, "r") as hdf:
        datasets = _collect_hdf5_datasets(hdf)
        dataset = _select_array_from_mapping(datasets, array_key, path)
        return np.asarray(dataset)


def _is_hdf4_file(path):
    """Return whether a path is explicitly HDF4 or has the HDF4 signature."""
    path = Path(path)
    if path.suffix.lower() in HDF4_EXTENSIONS:
        return True
    if path.suffix.lower() != ".hdf":
        return False
    try:
        with path.open("rb") as file:
            return file.read(len(HDF4_MAGIC)) == HDF4_MAGIC
    except OSError:
        return False


def _hdf4_dataset_info(path, array_key=None):
    from data.read_hdf4 import list_hdf4_datasets

    datasets = {dataset.name: dataset for dataset in list_hdf4_datasets(path)}
    return _select_array_from_mapping(datasets, array_key, path)


def _read_hdf4(path, array_key=None):
    from data.read_hdf4 import read_hdf4_dataset

    dataset = _hdf4_dataset_info(path, array_key=array_key)
    return read_hdf4_dataset(path, dataset_name=dataset.name)


def _read_netcdf(path, array_key=None):
    try:
        from netCDF4 import Dataset
    except ImportError as exc:
        raise _missing_dependency("netCDF4", "NetCDF") from exc

    with Dataset(path, "r") as dataset:
        variables = {
            name: variable
            for name, variable in dataset.variables.items()
            if getattr(variable, "ndim", 0) >= 2
        }
        variable = _select_array_from_mapping(variables, array_key, path)
        return np.asarray(variable[:])


def _ensure_supported_image_shape(image, path):
    image = np.asarray(image)

    if image.ndim == 2:
        raise ValueError(
            f"Large-image inference expects a multi-channel image, got a single-channel image from {path}."
        )
    if image.ndim != 3:
        raise ValueError(f"Expected image shape (H, W, C) or (C, H, W), got {image.shape}")
    if image.shape[0] in (3, 4) and image.shape[-1] not in (3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] not in (3, 4):
        raise ValueError(f"Expected 3 or 4 image channels, got shape {image.shape}")
    return image


def _read_image(path, array_key=None):
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in TIFF_EXTENSIONS:
        image = _read_tiff(path)
    elif suffix in NUMPY_EXTENSIONS:
        image = _read_numpy(path, array_key=array_key)
    elif suffix in PIL_EXTENSIONS:
        image = _read_with_pillow(path)
    elif _is_hdf4_file(path):
        image = _read_hdf4(path, array_key=array_key)
    elif suffix in HDF5_EXTENSIONS:
        image = _read_hdf5(path, array_key=array_key)
    elif suffix in NETCDF_EXTENSIONS:
        image = _read_netcdf(path, array_key=array_key)
    else:
        supported = ", ".join(SUPPORTED_IMAGE_EXTENSIONS)
        raise ValueError(f"Unsupported image format '{suffix}' for {path}. Supported formats: {supported}")

    return _ensure_supported_image_shape(image, path)


def _pad_patch(patch, patch_size):
    padded = np.zeros((patch_size, patch_size, patch.shape[-1]), dtype=patch.dtype)
    padded[: patch.shape[0], : patch.shape[1], :] = patch
    return padded


def calculate_cloud_coverage(cloud_mask):
    """Return the fraction of output-mask pixels marked as cloud."""
    cloud_mask = np.asarray(cloud_mask)
    if cloud_mask.size == 0:
        raise ValueError("cloud_mask must contain at least one pixel")
    return float(np.count_nonzero(cloud_mask > 0) / cloud_mask.size)


def is_image_accepted(cloud_mask, cloud_coverage_threshold=0.60):
    """Return false when predicted cloud coverage reaches the rejection threshold."""
    if not 0.0 <= cloud_coverage_threshold <= 1.0:
        raise ValueError("cloud_coverage_threshold must be between 0 and 1")
    return calculate_cloud_coverage(cloud_mask) < cloud_coverage_threshold


def _run_batch(trt_infer, batch_input):
    try:
        return trt_infer.infer_batch(batch_input)
    except ValueError as exc:
        if "batch" not in str(exc).lower():
            raise
        print(f"Batch inference unavailable: {exc}")
        preds = []
        probs = []
        for patch in batch_input:
            is_cloud, prob = trt_infer.infer(patch[None, ...])
            preds.append(is_cloud)
            probs.append(prob)
        return np.array(preds), np.array(probs)


def _is_channel_first(shape):
    """Detect if a 3-D array is stored as (C, H, W) rather than (H, W, C).

    Heuristic: first dim is in {3, 4} AND last dim is NOT in {3, 4}.
    """
    return len(shape) == 3 and shape[0] in (3, 4) and shape[-1] not in (3, 4)


def _shape_to_hwc(shape):
    """Canonicalize a raw 3-D shape to (H, W, C)."""
    if _is_channel_first(shape):
        return (shape[1], shape[2], shape[0])
    return tuple(shape)


def _get_image_shape(path, array_key=None):
    """Get (H, W, C) shape of an image without loading all pixel data into RAM.

    Falls back to full read for formats that don't support metadata-only access.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in TIFF_EXTENSIONS:
        try:
            import tifffile as tiff
        except ImportError as exc:
            raise _missing_dependency("tifffile", "TIFF/GeoTIFF") from exc
        with tiff.TiffFile(path) as t:
            page = t.pages[0]
            shape = page.shape
            if len(shape) == 2:
                raise ValueError(
                    f"Large-image inference expects a multi-channel image, got shape {shape} from {path}."
                )
            return _shape_to_hwc(shape)

    if suffix in NUMPY_EXTENSIONS:
        if suffix == ".npy":
            mm = np.load(path, mmap_mode="r")
            shape = tuple(mm.shape)
        else:
            loaded = np.load(path)
            arr = _select_array_from_mapping(loaded, array_key, path)
            shape = tuple(arr.shape)
        if len(shape) == 2:
            raise ValueError(f"Large-image inference expects multi-channel, got shape {shape} from {path}.")
        return _shape_to_hwc(shape)

    if _is_hdf4_file(path):
        info = _hdf4_dataset_info(path, array_key=array_key)
        shape = info.shape
        if len(shape) == 2:
            raise ValueError(f"Large-image inference expects multi-channel, got shape {shape} from {path}.")
        return _shape_to_hwc(shape)

    if suffix in HDF5_EXTENSIONS:
        try:
            import h5py
        except ImportError as exc:
            raise _missing_dependency("h5py", "HDF5/HDF") from exc
        with h5py.File(path, "r") as hdf:
            datasets = _collect_hdf5_datasets(hdf)
            dataset = _select_array_from_mapping(datasets, array_key, path)
            shape = tuple(dataset.shape)
        if len(shape) == 2:
            raise ValueError(f"Large-image inference expects multi-channel, got shape {shape} from {path}.")
        return _shape_to_hwc(shape)

    if suffix in NETCDF_EXTENSIONS:
        try:
            from netCDF4 import Dataset
        except ImportError as exc:
            raise _missing_dependency("netCDF4", "NetCDF") from exc
        with Dataset(path, "r") as ds:
            variables = {
                name: var for name, var in ds.variables.items()
                if getattr(var, "ndim", 0) >= 2
            }
            var = _select_array_from_mapping(variables, array_key, path)
            shape = tuple(var.shape)
        if len(shape) == 2:
            raise ValueError(f"Large-image inference expects multi-channel, got shape {shape} from {path}.")
        return _shape_to_hwc(shape)

    # PIL and other formats — fall back to full read
    image = _read_image(path, array_key=array_key)
    return image.shape


def _slice_strip_from_array(arr, row_start, row_end):
    """Slice a horizontal strip from a 3-D array, handling both (H,W,C) and (C,H,W).

    Always returns (strip_H, W, C) in channel-last layout.
    """
    shape = arr.shape
    if _is_channel_first(shape):
        # (C, H, W) → slice along axis 1, then moveaxis to (H, W, C)
        strip = np.asarray(arr[:, row_start:row_end, :])
        return np.moveaxis(strip, 0, -1)
    else:
        # (H, W, C) → slice along axis 0
        return np.asarray(arr[row_start:row_end])


def _read_image_strip(path, row_start, row_end, array_key=None):
    """Read a horizontal strip [row_start:row_end, :, :] from an image.

    Uses memory-efficient access where possible:
    - TIFF: memmap for uncompressed, full-read fallback for compressed
    - npy: numpy mmap_mode='r'
    - HDF4/HDF5/NetCDF: native partial read (dataset slicing)
    - PIL formats: full-read fallback

    Returns array in (H, W, C) layout, NOT normalized.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in TIFF_EXTENSIONS:
        raise RuntimeError("TIFF row reads must use a session-scoped TiffReader")

    if suffix in NUMPY_EXTENSIONS:
        if suffix == ".npy":
            mm = np.load(path, mmap_mode="r")
            return _slice_strip_from_array(mm, row_start, row_end)
        else:
            loaded = np.load(path)
            arr = np.asarray(_select_array_from_mapping(loaded, array_key, path))
            return _slice_strip_from_array(arr, row_start, row_end)

    if _is_hdf4_file(path):
        from data.read_hdf4 import read_hdf4_dataset

        info = _hdf4_dataset_info(path, array_key=array_key)
        if _is_channel_first(info.shape):
            start = [0, row_start, 0]
            count = [info.shape[0], row_end - row_start, info.shape[2]]
        else:
            start = [row_start, 0, 0]
            count = [row_end - row_start, info.shape[1], info.shape[2]]
        strip = read_hdf4_dataset(
            path,
            dataset_name=info.name,
            start=start,
            count=count,
        )
        if _is_channel_first(info.shape):
            return np.moveaxis(strip, 0, -1)
        return np.asarray(strip)

    if suffix in HDF5_EXTENSIONS:
        try:
            import h5py
        except ImportError as exc:
            raise _missing_dependency("h5py", "HDF5/HDF") from exc
        with h5py.File(path, "r") as hdf:
            datasets = _collect_hdf5_datasets(hdf)
            dataset = _select_array_from_mapping(datasets, array_key, path)
            return _slice_strip_from_array(dataset, row_start, row_end)

    if suffix in NETCDF_EXTENSIONS:
        try:
            from netCDF4 import Dataset
        except ImportError as exc:
            raise _missing_dependency("netCDF4", "NetCDF") from exc
        with Dataset(path, "r") as ds:
            variables = {
                name: var for name, var in ds.variables.items()
                if getattr(var, "ndim", 0) >= 2
            }
            var = _select_array_from_mapping(variables, array_key, path)
            return _slice_strip_from_array(var, row_start, row_end)

    # PIL and other — full read fallback
    image = _read_image(path, array_key=array_key)
    return image[row_start:row_end]


class _NativeImageReader(ImageBlockReader):
    """Adapter for non-TIFF formats outside the TIFF remediation scope."""

    def __init__(self, path, array_key=None):
        self.path = Path(path)
        self.array_key = array_key
        self.shape = _get_image_shape(self.path, array_key=array_key)
        self.dtype = None
        self.axes = {"original": None, "normalized": "YXC"}
        self.band_order = None
        self.metrics = ReaderMetrics()

    def read_rows(self, row_start, row_end):
        return _read_image_strip(
            self.path,
            row_start,
            row_end,
            array_key=self.array_key,
        )

    def physical_blocks(self, row_start, row_end):
        return ()

    def close(self):
        return None


class _MaskCache:
    def __init__(self, path, shape, owns_cache):
        self.path = Path(path)
        self.shape = tuple(shape)
        self.owns_cache = bool(owns_cache)
        self.array = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise FileExistsError(f"Mask cache path already exists: {self.path}")
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
        descriptor = os.open(self.path, flags)
        try:
            os.ftruncate(descriptor, math.prod(self.shape))
        finally:
            os.close(descriptor)
        try:
            self.array = np.memmap(
                self.path,
                mode="r+",
                dtype=np.uint8,
                shape=self.shape,
            )
            self.array[:] = 0
            return self.array
        except Exception:
            if self.owns_cache:
                self.path.unlink(missing_ok=True)
            raise

    def __exit__(self, exc_type, exc_value, traceback):
        if self.array is not None:
            close_memmap(self.array)
            self.array = None
        if self.owns_cache:
            self.path.unlink(missing_ok=True)
        return False


def _atomic_write_tiff(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        _write_tiff(temporary_path, image)
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _resolve_input_spec(engine_path, engine_manifest, channels, patch_size):
    if engine_manifest is None:
        return legacy_input_spec(channels, patch_size)

    input_spec = load_engine_manifest(engine_manifest, engine_path=engine_path)
    if channels != input_spec.channels:
        raise ValueError(
            f"--channels={channels} does not match engine manifest channels={input_spec.channels}"
        )
    if patch_size != input_spec.patch_size:
        raise ValueError(
            f"--patch_size={patch_size} does not match engine manifest patch size "
            f"{input_spec.patch_size}"
        )
    return input_spec


def _resolve_cache_paths(out_mask, mask_cache, tiff_cache_dir):
    output_path = Path(out_mask)
    if mask_cache == "":
        raise ValueError("mask_cache must not be an empty path")
    cache_dir = (
        Path(tiff_cache_dir)
        if tiff_cache_dir is not None
        else output_path.parent / ".cube_nano-cache"
    )
    if mask_cache is None:
        mask_path = cache_dir / f"mask_{os.getpid()}_{uuid4().hex}.dat"
        owns_mask_cache = True
    else:
        mask_path = Path(mask_cache)
        owns_mask_cache = False
    if mask_path.exists():
        raise FileExistsError(f"Mask cache path already exists: {mask_path}")
    return output_path, cache_dir, mask_path, owns_mask_cache


def _disk_allocations_for_shape(shape, output_path, mask_path):
    mask_bytes = int(shape[0]) * int(shape[1]) * np.dtype(np.uint8).itemsize
    return (
        DiskAllocation(output_path, mask_bytes, "atomic output TIFF temporary file"),
        DiskAllocation(mask_path, mask_bytes, "inference mask cache"),
    )


def process_large_image(
    large_image_path,
    engine_path,
    out_mask="cloud_mask_output.tif",
    patch_size=256,
    channels=3,
    batch_size=1,
    array_key=None,
    threshold=0.5,
    mask_cache=None,
    cloud_coverage_threshold=0.60,
    discard_cloudy=False,
    tiff_read_mode="auto",
    tiff_cache_mode="auto",
    max_ram_cache_gib=0.5,
    max_disk_cache_gib=8.0,
    runtime_reserve_gib=1.5,
    tiff_block_cache_mib=64,
    tiff_cache_dir=None,
    tiff_series=None,
    tiff_level=None,
    channel_mapping=None,
    input_sidecar=None,
    engine_manifest=None,
    production_contract=False,
    _memory_provider=None,
    _filesystem_provider=None,
    _trt_infer_factory=None,
    _backend_name="TensorRT",
):
    """
    Xử lý ảnh vệ tinh cực lớn (ví dụ 10000x10000) bằng phương pháp Cửa sổ trượt (Sliding Window)
    và xử lý theo lô (Batch Processing) trên Jetson Nano.

    Đọc ảnh theo từng dải hàng (row strip) thay vì nạp toàn bộ vào RAM,
    giúp tránh OOM trên thiết bị có RAM hạn chế (ví dụ Jetson Nano 4GB).

    ``cloud_coverage_threshold`` is applied to the generated coarse mask after
    all tiles are processed. It measures the fraction of image area covered by
    tiles predicted as cloud, not the exact pixel-level cloud fraction.
    ``discard_cloudy`` only removes the generated output mask when rejected;
    the source image is never deleted.
    """
    if patch_size <= 0:
        raise ValueError("patch_size must be greater than zero")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if not 0.0 <= cloud_coverage_threshold <= 1.0:
        raise ValueError("cloud_coverage_threshold must be between 0 and 1")

    image_path = Path(large_image_path)
    if production_contract:
        if engine_manifest is None:
            raise ValueError("production_contract requires an engine_manifest")
        if image_path.suffix.lower() in TIFF_EXTENSIONS and input_sidecar is None:
            raise ValueError("production_contract requires an input_sidecar for TIFF")

    input_spec = _resolve_input_spec(engine_path, engine_manifest, channels, patch_size)
    budget = ReaderBudget.from_cli(
        max_ram_cache_gib=max_ram_cache_gib,
        max_disk_cache_gib=max_disk_cache_gib,
        runtime_reserve_gib=runtime_reserve_gib,
        tiff_block_cache_mib=tiff_block_cache_mib,
    )
    output_path, cache_dir, mask_path, owns_mask_cache = _resolve_cache_paths(
        out_mask,
        mask_cache,
        tiff_cache_dir,
    )

    # Resource guards are evaluated after model/runtime allocations.
    print(f"Initializing {_backend_name}...")
    if _trt_infer_factory is None:
        from inference_tensorrt import CloudTRTInfer

        trt_infer_factory = CloudTRTInfer
    else:
        trt_infer_factory = _trt_infer_factory
    trt_infer = trt_infer_factory(
        engine_path,
        channels=channels,
        patch_size=patch_size,
        threshold=threshold,
        input_spec=input_spec,
    )

    filesystem_provider = _filesystem_provider or FilesystemInfoProvider()
    print(f"Reading image metadata from {large_image_path}...")
    if image_path.suffix.lower() in TIFF_EXTENSIONS:
        reader = TiffReader(
            image_path,
            input_spec,
            read_mode=tiff_read_mode,
            cache_mode=tiff_cache_mode,
            budget=budget,
            cache_dir=cache_dir,
            series_index=tiff_series,
            level_index=tiff_level,
            channel_mapping=channel_mapping,
            input_sidecar=input_sidecar,
            patch_size=patch_size,
            batch_size=batch_size,
            memory_provider=_memory_provider,
            filesystem_provider=filesystem_provider,
            disk_allocations=lambda active_reader: _disk_allocations_for_shape(
                active_reader.shape,
                output_path,
                mask_path,
            ),
        )
    else:
        reader = _NativeImageReader(image_path, array_key=array_key)
        allocations = _disk_allocations_for_shape(reader.shape, output_path, mask_path)
        require_writable_parents(allocations)
        require_disk_allocations(
            allocations,
            provider=filesystem_provider,
        )

    reader_backend = None
    reader_metrics = None
    reader_provenance = None
    with reader:
        H_img, W_img, C_img = reader.shape
        if C_img != input_spec.channels:
            raise ValueError(
                f"Image reader outputs {C_img} channels but {_backend_name} input requires "
                f"{input_spec.channels}"
            )

        num_patches_h = math.ceil(H_img / patch_size)
        num_patches_w = math.ceil(W_img / patch_size)
        total_patches = num_patches_h * num_patches_w
        print(f"Image size: {H_img}x{W_img}x{C_img}")
        print(f"Total {patch_size}x{patch_size} patches: {total_patches}")

        start_time = time.time()
        with _MaskCache(mask_path, (H_img, W_img), owns_mask_cache) as cloud_mask:
            batch_patches = []
            batch_coords = []
            processed_count = 0

            for row_start in range(0, H_img, patch_size):
                row_end = min(row_start + patch_size, H_img)
                strip = reader.read_rows(row_start, row_end)

                for column_start in range(0, W_img, patch_size):
                    column_end = min(column_start + patch_size, W_img)
                    patch = strip[:, column_start:column_end, :]
                    patch = input_spec.normalization.apply(patch)
                    if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                        patch = _pad_patch(patch, patch_size)
                    batch_patches.append(np.transpose(patch, (2, 0, 1)))
                    batch_coords.append(
                        (row_start, row_end, column_start, column_end)
                    )

                    if len(batch_patches) == batch_size or (
                        row_end == H_img and column_end == W_img
                    ):
                        batch_input = np.stack(batch_patches, axis=0)
                        is_cloud_batch, _ = _run_batch(trt_infer, batch_input)
                        for index, (r_start, r_end, c_start, c_end) in enumerate(
                            batch_coords
                        ):
                            if is_cloud_batch[index]:
                                cloud_mask[r_start:r_end, c_start:c_end] = 255

                        processed_count += len(batch_patches)
                        if processed_count % 1000 == 0:
                            print(f"Processed: {processed_count}/{total_patches} patches...")
                        batch_patches = []
                        batch_coords = []

            cloud_coverage = calculate_cloud_coverage(cloud_mask)
            accepted = is_image_accepted(cloud_mask, cloud_coverage_threshold)
            cloud_mask.flush()
            _atomic_write_tiff(output_path, cloud_mask)

        elapsed = time.time() - start_time
        reader_backend = getattr(reader, "backend", "native") or "native"
        reader_metrics = reader.metrics.as_dict()
        reader_provenance = dict(getattr(reader, "provenance", {}))
        reader_provenance["selected_backend"] = reader_backend
        reader_provenance["source_cache_peak_bytes"] = (
            int(getattr(reader, "decoded_bytes", 0))
            if reader_backend == "disk"
            else 0
        )

    print(f"\nCompleted {H_img}x{W_img} image in {elapsed:.2f} seconds")
    print(
        f"Cloud-covered mask area: {cloud_coverage:.2%} "
        f"(rejection threshold: {cloud_coverage_threshold:.2%})"
    )

    output_mask = str(output_path)
    if not accepted and discard_cloudy:
        output_path.unlink()
        output_mask = None
        print("Image rejected at the cloud threshold; output mask was removed.")
    elif accepted:
        print(f"Image accepted; cloud mask saved to {output_path}.")
    else:
        print(
            f"Image marked as rejected; mask retained at {output_path}. "
            "Use --discard-cloudy to remove rejected masks."
        )

    return {
        "accepted": accepted,
        "cloud_coverage": cloud_coverage,
        "cloud_coverage_threshold": cloud_coverage_threshold,
        "out_mask": output_mask,
        "reader_backend": reader_backend,
        "reader_metrics": reader_metrics,
        "reader_provenance": reader_provenance,
        "elapsed_seconds": elapsed,
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run TensorRT cloud inference on a large multi-channel image")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help=(
            "Path to a 3- or 4-channel image. Supported extensions: "
            + ", ".join(SUPPORTED_IMAGE_EXTENSIONS)
        ),
    )
    parser.add_argument("--engine", type=str, default="cloud_model.engine", help="Path to TensorRT engine")
    parser.add_argument("--out_mask", type=str, default="cloud_mask_output.tif", help="Output mask path")
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--channels", type=int, default=3, choices=[3, 4])
    parser.add_argument("--legacy", action="store_true", help="Explicitly enable the legacy 4-channel development path")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--array_key",
        type=str,
        default=None,
        help="Array/dataset key for .npz, HDF4, HDF5, or NetCDF inputs when the file contains multiple arrays.",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for cloud classification")
    parser.add_argument(
        "--cloud_coverage_threshold",
        "--cloud-coverage-threshold",
        dest="cloud_coverage_threshold",
        type=float,
        default=0.60,
        help="Reject image when output cloud coverage reaches this ratio (default: 0.60)",
    )
    parser.add_argument(
        "--discard-cloudy",
        action="store_true",
        help="Delete the generated output mask when cloud coverage reaches the rejection threshold",
    )
    parser.add_argument(
        "--mask_cache",
        type=str,
        default=None,
        help="Optional path for the temporary file-backed mask (useful when RAM is limited).",
    )
    parser.add_argument(
        "--tiff_read_mode",
        choices=["auto", "stream", "full"],
        default="auto",
    )
    parser.add_argument(
        "--tiff_cache_mode",
        choices=["auto", "ram", "disk"],
        default="auto",
    )
    parser.add_argument("--max_ram_cache_gib", default=None)
    parser.add_argument("--max_disk_cache_gib", default=None)
    parser.add_argument("--runtime_reserve_gib", default="1.5")
    parser.add_argument("--tiff_block_cache_mib", default="64")
    parser.add_argument("--tiff_cache_dir", type=str, default=None)
    parser.add_argument("--tiff_series", type=int, default=None)
    parser.add_argument("--tiff_level", type=int, default=None)
    parser.add_argument("--channel_mapping", type=str, default=None)
    parser.add_argument("--input_sidecar", type=str, default=None)
    parser.add_argument("--engine_manifest", type=str, default=None)
    parser.add_argument("--production_contract", action="store_true")
    args = parser.parse_args()
    if args.channels == 4 and not args.legacy:
        raise RuntimeError("legacy 4-channel inference requires --legacy")

    if args.tiff_read_mode == "stream":
        if args.tiff_cache_mode != "auto":
            parser.error("--tiff_read_mode=stream requires --tiff_cache_mode=auto")
        if args.max_ram_cache_gib is not None or args.max_disk_cache_gib is not None:
            parser.error(
                "stream mode does not accept explicit decoded-cache size options"
            )

    process_large_image(
        args.image,
        args.engine,
        out_mask=args.out_mask,
        patch_size=args.patch_size,
        channels=args.channels,
        batch_size=args.batch_size,
        array_key=args.array_key,
        threshold=args.threshold,
        mask_cache=args.mask_cache,
        cloud_coverage_threshold=args.cloud_coverage_threshold,
        discard_cloudy=args.discard_cloudy,
        tiff_read_mode=args.tiff_read_mode,
        tiff_cache_mode=args.tiff_cache_mode,
        max_ram_cache_gib=(
            "0.5" if args.max_ram_cache_gib is None else args.max_ram_cache_gib
        ),
        max_disk_cache_gib=(
            "8.0" if args.max_disk_cache_gib is None else args.max_disk_cache_gib
        ),
        runtime_reserve_gib=args.runtime_reserve_gib,
        tiff_block_cache_mib=args.tiff_block_cache_mib,
        tiff_cache_dir=args.tiff_cache_dir,
        tiff_series=args.tiff_series,
        tiff_level=args.tiff_level,
        channel_mapping=args.channel_mapping,
        input_sidecar=args.input_sidecar,
        engine_manifest=args.engine_manifest,
        production_contract=args.production_contract,
    )
