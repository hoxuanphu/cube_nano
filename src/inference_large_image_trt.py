import numpy as np
import time
import math
import argparse
import os
import tempfile
from pathlib import Path

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


def _normalize_patch(patch):
    if np.issubdtype(patch.dtype, np.floating):
        patch = patch.astype(np.float32)
        if patch.max() > 1.0:
            scale = 65535.0 if patch.max() > 255.0 else 255.0
            patch = patch / scale
        return np.clip(patch, 0.0, 1.0)

    if np.issubdtype(patch.dtype, np.integer):
        patch = patch.astype(np.float32) / float(np.iinfo(patch.dtype).max)
        return np.clip(patch, 0.0, 1.0)

    raise ValueError(f"Unsupported image dtype {patch.dtype}")


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
        try:
            import tifffile as tiff
        except ImportError as exc:
            raise _missing_dependency("tifffile", "TIFF/GeoTIFF") from exc
        # Try memory-mapped access first (works for uncompressed TIFFs)
        try:
            mm = tiff.memmap(path)
            strip = _slice_strip_from_array(mm, row_start, row_end)
            return strip
        except ValueError:
            # Compressed TIFF — memmap not supported, fall back to full read
            full = tiff.imread(path)
            strip = _slice_strip_from_array(full, row_start, row_end)
            return strip

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


def process_large_image(
    large_image_path,
    engine_path,
    out_mask="cloud_mask_output.tif",
    patch_size=256,
    channels=4,
    batch_size=1,
    array_key=None,
    threshold=0.5,
    mask_cache=None,
    cloud_coverage_threshold=0.60,
    discard_cloudy=False,
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

    print(f"Đang đọc metadata ảnh từ {large_image_path}...")
    H_img, W_img, C_img = _get_image_shape(large_image_path, array_key=array_key)

    # Khởi tạo TensorRT Engine
    print("Khởi tạo TensorRT...")
    from inference_tensorrt import CloudTRTInfer

    trt_infer = CloudTRTInfer(engine_path, channels=channels, patch_size=patch_size, threshold=threshold)
    
    # Tính toán số lượng patch
    num_patches_h = math.ceil(H_img / patch_size)
    num_patches_w = math.ceil(W_img / patch_size)
    total_patches = num_patches_h * num_patches_w
    
    print(f"Kích thước ảnh: {H_img}x{W_img}x{C_img}")
    print(f"Tổng số ô {patch_size}x{patch_size} cần xử lý: {total_patches}")
    
    # File-backed mask keeps RAM usage independent of the source image size.
    if mask_cache:
        cache_path = Path(mask_cache)
    else:
        cache_fd, cache_name = tempfile.mkstemp(prefix="cloud_mask_", suffix=".dat")
        os.close(cache_fd)
        cache_path = Path(cache_name)
    cloud_mask = np.memmap(cache_path, mode="w+", dtype=np.uint8, shape=(H_img, W_img))
    cloud_mask[:] = 0
    
    start_time = time.time()
    
    batch_patches = []
    batch_coords = []
    processed_count = 0
    
    # Trượt cửa sổ theo từng dải hàng (Row Strip) thay vì load toàn bộ ảnh
    for row_start in range(0, H_img, patch_size):
        row_end = min(row_start + patch_size, H_img)

        # Chỉ đọc dải hàng hiện tại vào RAM
        strip = _read_image_strip(large_image_path, row_start, row_end, array_key=array_key)

        for j in range(0, W_img, patch_size):
            j_end = min(j + patch_size, W_img)
            
            patch = strip[:, j:j_end, :]
            patch = _normalize_patch(patch)
            if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                patch = _pad_patch(patch, patch_size)
            patch = np.transpose(patch, (2, 0, 1))
            
            # Gộp vào batch
            batch_patches.append(patch)
            batch_coords.append((row_start, row_end, j, j_end))
            
            # Khi đủ 1 batch hoặc là patch cuối cùng
            if len(batch_patches) == batch_size or (row_end == H_img and j_end == W_img):
                # Gộp mảng list thành tensor (B, C, H, W)
                batch_input = np.stack(batch_patches, axis=0).astype(np.float32)
                is_cloud_batch, _ = _run_batch(trt_infer, batch_input)
                
                # Ghi kết quả vào ảnh mask lớn
                for idx, (r_start, r_end, c_start, c_end) in enumerate(batch_coords):
                    if is_cloud_batch[idx]:
                        cloud_mask[r_start:r_end, c_start:c_end] = 255
                
                processed_count += len(batch_patches)
                if processed_count % 1000 == 0:
                    print(f"Đã xử lý: {processed_count}/{total_patches} patches...")
                
                # Xóa batch để nạp lứa mới
                batch_patches = []
                batch_coords = []

        # strip được giải phóng khi vòng lặp row chuyển sang dải tiếp theo

    end_time = time.time()
    cloud_coverage = calculate_cloud_coverage(cloud_mask)
    accepted = is_image_accepted(cloud_mask, cloud_coverage_threshold)

    print(f"\nHoàn tất! Tổng thời gian xử lý ảnh {H_img}x{W_img}: {end_time - start_time:.2f} giây")
    print(
        f"Tỷ lệ vùng bị đánh dấu mây: {cloud_coverage:.2%} "
        f"(ngưỡng loại: {cloud_coverage_threshold:.2%})"
    )
    cloud_mask.flush()
    _write_tiff(out_mask, cloud_mask)
    del cloud_mask
    if mask_cache is None:
        os.unlink(cache_path)

    output_mask = str(out_mask)
    if not accepted and discard_cloudy:
        os.remove(out_mask)
        output_mask = None
        print("Ảnh bị loại vì tỷ lệ mây đạt ngưỡng; mask đầu ra đã được xóa.")
    elif accepted:
        print(f"Ảnh được giữ lại; mask mây đã được lưu tại {out_mask}.")
    else:
        print(
            f"Ảnh bị đánh dấu loại; mask vẫn được giữ tại {out_mask}. "
            "Dùng --discard-cloudy để xóa mask bị loại."
        )

    return {
        "accepted": accepted,
        "cloud_coverage": cloud_coverage,
        "cloud_coverage_threshold": cloud_coverage_threshold,
        "out_mask": output_mask,
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
    parser.add_argument("--channels", type=int, default=4, choices=[3, 4])
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
    args = parser.parse_args()

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
    )
