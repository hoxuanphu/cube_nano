"""Read Scientific Datasets from HDF4 files.

HDF4 is different from HDF5, so ``h5py`` cannot read it.  This module uses
``pyhdf`` and keeps the dependency optional until a reader function is used.

Examples:
    python src/data/read_hdf4.py scene.hdf --list
    python src/data/read_hdf4.py scene.hdf --dataset "Reflectance" --output scene.npy
    python src/data/read_hdf4.py scene.hdf --dataset "Reflectance" --output scene.tif
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


TIFF_EXTENSIONS = {".tif", ".tiff"}


@dataclass(frozen=True)
class HDF4DatasetInfo:
    """Description of one HDF4 Scientific Dataset."""

    name: str
    index: int
    shape: tuple[int, ...]
    dimensions: tuple[str, ...]
    dtype: str
    attributes: dict[str, Any]


def _missing_dependency():
    return ImportError(
        "Reading HDF4 files requires the 'pyhdf' package. "
        "Install project dependencies with: pip install -r requirements.txt"
    )


def _load_pyhdf():
    try:
        from pyhdf.SD import SD, SDC
    except ImportError as exc:
        raise _missing_dependency() from exc
    return SD, SDC


def _validate_path(path: str | Path) -> Path:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"HDF4 file not found: {path}")
    return path


def _jsonable(value: Any) -> Any:
    """Convert HDF4 attributes into values that can be serialized as JSON."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return _jsonable(value.item()) if value.ndim == 0 else [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _attributes(obj: Any) -> dict[str, Any]:
    try:
        attributes = obj.attributes()
    except (AttributeError, TypeError):
        return {}
    return {str(name): _jsonable(value) for name, value in attributes.items()}


def _dataset_descriptions(hdf: Any) -> dict[str, tuple[Any, ...]]:
    descriptions = hdf.datasets()
    return {str(name): tuple(description) for name, description in descriptions.items()}


def _parse_dataset_description(name: str, description: tuple[Any, ...]):
    if len(description) < 4:
        raise ValueError(f"Unexpected HDF4 dataset description for '{name}': {description!r}")

    dimension_names, shape, type_code, index = description[:4]
    if not isinstance(dimension_names, (list, tuple)) or not isinstance(shape, (list, tuple)):
        raise ValueError(f"Unexpected HDF4 dataset description for '{name}': {description!r}")
    return (
        tuple(str(dimension) for dimension in dimension_names),
        tuple(int(size) for size in shape),
        str(type_code),
        int(index),
    )


def _dataset_info(hdf: Any, name: str, description: tuple[Any, ...]) -> HDF4DatasetInfo:
    dimensions, shape, dtype, index = _parse_dataset_description(name, description)

    dataset = hdf.select(name)
    try:
        return HDF4DatasetInfo(
            name=name,
            index=index,
            shape=shape,
            dimensions=dimensions,
            dtype=dtype,
            attributes=_attributes(dataset),
        )
    finally:
        dataset.endaccess()


def list_hdf4_datasets(path: str | Path) -> list[HDF4DatasetInfo]:
    """Return names, shapes, dtypes, and attributes for all HDF4 datasets."""
    path = _validate_path(path)
    SD, SDC = _load_pyhdf()
    hdf = SD(str(path), SDC.READ)
    try:
        descriptions = _dataset_descriptions(hdf)
        return [
            _dataset_info(hdf, name, descriptions[name])
            for name in descriptions
        ]
    finally:
        hdf.end()


def read_hdf4_metadata(path: str | Path) -> dict[str, Any]:
    """Read file-level and dataset-level metadata without loading pixel data."""
    path = _validate_path(path)
    SD, SDC = _load_pyhdf()
    hdf = SD(str(path), SDC.READ)
    try:
        descriptions = _dataset_descriptions(hdf)
        datasets = [
            asdict(_dataset_info(hdf, name, descriptions[name]))
            for name in descriptions
        ]
        return {
            "path": str(path),
            "attributes": _attributes(hdf),
            "datasets": datasets,
        }
    finally:
        hdf.end()


def _select_dataset_name(
    descriptions: dict[str, tuple[Any, ...]],
    dataset_name: str | None,
    path: Path,
) -> str:
    if dataset_name:
        if dataset_name not in descriptions:
            available = ", ".join(descriptions) or "<none>"
            raise KeyError(
                f"Dataset '{dataset_name}' not found in {path}. "
                f"Available datasets: {available}"
            )
        return dataset_name

    if len(descriptions) == 1:
        return next(iter(descriptions))
    if not descriptions:
        raise ValueError(f"HDF4 file contains no Scientific Dataset: {path}")

    available = ", ".join(descriptions)
    raise ValueError(
        f"HDF4 file contains multiple datasets. Set dataset_name explicitly. "
        f"Available datasets: {available}"
    )


def _vector(
    value: Sequence[int] | None,
    ndim: int,
    name: str,
) -> list[int] | None:
    if value is None:
        return None
    values = [int(item) for item in value]
    if len(values) != ndim:
        raise ValueError(f"{name} must contain {ndim} values, got {len(values)}")
    if any(item < 0 for item in values):
        raise ValueError(f"{name} values must be non-negative, got {values}")
    if name == "stride" and any(item == 0 for item in values):
        raise ValueError("stride values must be greater than zero")
    return values


def _read_sds(dataset: Any, shape: tuple[int, ...], start=None, count=None, stride=None):
    """Read an SDS, optionally using HDF4's native hyperslab access."""
    if start is None and count is None and stride is None:
        return np.asarray(dataset[:])

    start = _vector(start, len(shape), "start") or [0] * len(shape)
    stride = _vector(stride, len(shape), "stride") or [1] * len(shape)
    if count is None:
        count = [
            max(0, (size - offset + step - 1) // step)
            for size, offset, step in zip(shape, start, stride)
        ]
    else:
        count = _vector(count, len(shape), "count")

    for size, offset, amount, step in zip(shape, start, count, stride):
        if amount == 0:
            raise ValueError("count values must be greater than zero")
        if offset >= size:
            raise ValueError(f"start {start} is outside dataset shape {shape}")
        if offset + (amount - 1) * step >= size:
            raise ValueError(
                f"Requested HDF4 slice exceeds dataset shape: "
                f"start={start}, count={count}, stride={stride}, shape={shape}"
            )
    return np.asarray(dataset.get(start, count, stride))


def read_hdf4_dataset(
    path: str | Path,
    dataset_name: str | None = None,
    start: Sequence[int] | None = None,
    count: Sequence[int] | None = None,
    stride: Sequence[int] | None = None,
) -> np.ndarray:
    """Read one HDF4 Scientific Dataset as a NumPy array.

    ``start``, ``count``, and ``stride`` are optional per-axis values for a
    hyperslab read. The returned array keeps the source dataset dtype and axis
    order; no scale factor or offset is applied automatically.
    """
    path = _validate_path(path)
    SD, SDC = _load_pyhdf()
    hdf = SD(str(path), SDC.READ)
    try:
        descriptions = _dataset_descriptions(hdf)
        name = _select_dataset_name(descriptions, dataset_name, path)
        description = descriptions[name]
        _, shape, _, _ = _parse_dataset_description(name, description)
        dataset = hdf.select(name)
        try:
            return _read_sds(dataset, shape, start=start, count=count, stride=stride)
        finally:
            dataset.endaccess()
    finally:
        hdf.end()


def _tiff_axes(array: np.ndarray) -> str:
    if array.ndim == 2:
        return "YX"
    if array.ndim != 3:
        raise ValueError(
            f"TIFF output requires a 2-D or 3-D array, got shape {array.shape}"
        )
    if array.shape[0] <= 16:
        return "CYX"
    if array.shape[-1] <= 16:
        return "YXC"
    return "QYX"


def write_tiff(
    destination: Any,
    array: np.ndarray,
    compression: str | None = None,
) -> None:
    """Write a 2-D or multi-band 3-D array as TIFF.

    Channel-first arrays are stored with ``CYX`` axes. This preserves HDF4
    datasets such as MODIS ``EV_250_RefSB`` without changing their band order.
    The output is a regular TIFF, not a georeferenced GeoTIFF.
    """
    try:
        import tifffile
    except ImportError as exc:
        raise ImportError(
            "Writing TIFF files requires 'tifffile'. "
            "Install project dependencies with: pip install -r requirements.txt"
        ) from exc

    array = np.asarray(array)
    axes = _tiff_axes(array)
    options = {
        "bigtiff": array.nbytes >= 4 * 1024**3,
        "compression": compression,
        "metadata": {"axes": axes},
    }
    if axes == "YXC" and array.shape[-1] in (3, 4):
        options["photometric"] = "rgb"
    elif axes == "CYX" and array.shape[0] in (3, 4):
        options["photometric"] = "rgb"
        options["planarconfig"] = "separate"
    else:
        options["photometric"] = "minisblack"
    tifffile.imwrite(destination, array, **options)


def save_dataset(
    output: str | Path,
    array: np.ndarray,
    compression: str | None = None,
) -> Path:
    """Save a dataset as NumPy or TIFF based on the output extension."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()
    if suffix == ".npy":
        np.save(output, array)
    elif suffix in TIFF_EXTENSIONS:
        write_tiff(output, array, compression=compression)
    else:
        raise ValueError(
            f"Unsupported output format '{suffix}'. Use .npy, .tif, or .tiff."
        )
    return output


def _parse_vector(value: str | None) -> list[int] | None:
    if value is None:
        return None
    try:
        return [int(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected comma-separated integers, got '{value}'"
        ) from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read HDF4 Scientific Datasets")
    parser.add_argument("path", type=Path, help="Path to the HDF4 file")
    parser.add_argument(
        "--list",
        dest="list_datasets",
        action="store_true",
        help="Print file and dataset metadata as JSON",
    )
    parser.add_argument("--dataset", help="Scientific Dataset name to read")
    parser.add_argument(
        "--output",
        type=Path,
        help="Save the selected dataset as .npy, .tif, or .tiff",
    )
    parser.add_argument(
        "--compression",
        choices=("none", "deflate"),
        default="none",
        help="TIFF compression (default: none)",
    )
    parser.add_argument("--start", type=_parse_vector, help="Slice start, e.g. 0,0,0")
    parser.add_argument("--count", type=_parse_vector, help="Slice size, e.g. 256,256,4")
    parser.add_argument("--stride", type=_parse_vector, help="Slice stride, e.g. 1,1,1")
    args = parser.parse_args(argv)

    if args.list_datasets:
        print(json.dumps(read_hdf4_metadata(args.path), indent=2, ensure_ascii=False))
        return 0

    array = read_hdf4_dataset(
        args.path,
        dataset_name=args.dataset,
        start=args.start,
        count=args.count,
        stride=args.stride,
    )
    if args.output:
        compression = None if args.compression == "none" else args.compression
        output = save_dataset(args.output, array, compression=compression)
        print(f"Saved {array.shape} {array.dtype} to {output}")
    else:
        print(f"Read dataset: shape={array.shape}, dtype={array.dtype}")
        print(array)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
