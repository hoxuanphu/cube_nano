"""Convert JP2 rasters to lossless TIFF files.

The converter copies decoded pixel samples without scaling, clipping, or
normalizing them.  It can also stack three single-band JP2 files into one
three-band TIFF in the order supplied on the command line.  Pixel data is
preserved exactly; geospatial metadata such as CRS/geotransform from GMLJP2
is not copied by Pillow/tifffile.

Examples:
    python src/data/convert_jp2_to_tiff.py \
        src/data/T48PYS_20260101T031141_B02_10m.jp2 \
        --output src/data/tiff

    python src/data/convert_jp2_to_tiff.py --stack \
        src/data/T48PYS_20260101T031141_B04_10m.jp2 \
        src/data/T48PYS_20260101T031141_B03_10m.jp2 \
        src/data/T48PYS_20260101T031141_B02_10m.jp2 \
        --output src/data/T48PYS_20260101T031141_RGB.tif
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import tifffile
from PIL import Image


JP2_EXTENSIONS = {".jp2", ".j2k", ".j2c"}
TIFF_EXTENSIONS = {".tif", ".tiff"}
VERIFY_ROW_CHUNK = 256

# Sentinel-2 10 m rasters are larger than Pillow's default safety limit.
Image.MAX_IMAGE_PIXELS = None


def _resolve_inputs(inputs: Iterable[Path]) -> list[Path]:
    """Expand input files/directories and preserve explicit file order."""
    resolved: list[Path] = []
    for input_path in inputs:
        if input_path.is_dir():
            resolved.extend(
                sorted(
                    path
                    for path in input_path.rglob("*")
                    if path.is_file() and path.suffix.lower() in JP2_EXTENSIONS
                )
            )
            continue

        if not input_path.is_file():
            raise FileNotFoundError(f"Input path does not exist: {input_path}")
        if input_path.suffix.lower() not in JP2_EXTENSIONS:
            raise ValueError(f"Expected a JP2 input, got: {input_path}")
        resolved.append(input_path)

    if not resolved:
        raise FileNotFoundError("No JP2 files found in the supplied input paths.")

    duplicate_paths = {path for path in resolved if resolved.count(path) > 1}
    if duplicate_paths:
        names = ", ".join(str(path) for path in sorted(duplicate_paths))
        raise ValueError(f"The same input was supplied more than once: {names}")
    return resolved


def read_jp2(path: Path) -> np.ndarray:
    """Decode one JP2 file while preserving its pixel dtype and values."""
    with Image.open(path) as image:
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError(f"Expected one image frame in {path}")
        # Make an independent array before closing the Pillow image.
        array = np.array(image, copy=True)

    if array.ndim not in (2, 3):
        raise ValueError(f"Unsupported decoded shape {array.shape} from {path}")
    return array


def _verify_tiff(source: np.ndarray, output_path: Path) -> None:
    """Verify TIFF shape, dtype, and every pixel against the source array."""
    with tifffile.TiffFile(output_path) as tif:
        if len(tif.pages) != 1:
            raise ValueError(f"Expected one TIFF page in {output_path}")
        page = tif.pages[0]
        if tuple(page.shape) != tuple(source.shape):
            raise ValueError(
                f"TIFF shape mismatch for {output_path}: "
                f"expected {source.shape}, got {page.shape}"
            )
        if np.dtype(page.dtype) != source.dtype:
            raise ValueError(
                f"TIFF dtype mismatch for {output_path}: "
                f"expected {source.dtype}, got {page.dtype}"
            )

    # The default output is uncompressed, so memmap verifies rows without
    # allocating a second full-size decoded TIFF array.
    try:
        written = tifffile.memmap(output_path, mode="r")
    except ValueError:
        written = tifffile.imread(output_path)

    try:
        for row_start in range(0, source.shape[0], VERIFY_ROW_CHUNK):
            row_end = min(row_start + VERIFY_ROW_CHUNK, source.shape[0])
            if not np.array_equal(source[row_start:row_end], written[row_start:row_end]):
                raise ValueError(
                    f"Pixel data mismatch found in {output_path} "
                    f"at rows {row_start}:{row_end}"
                )
    finally:
        del written


def write_tiff(
    array: np.ndarray,
    output_path: Path,
    *,
    overwrite: bool = False,
    verify: bool = True,
) -> None:
    """Write an array to an uncompressed TIFF and optionally verify it."""
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --overwrite to replace it."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # No compression is used by default.  This preserves exact samples and
    # avoids requiring an optional codec package on the target device.
    photometric = "rgb" if array.ndim == 3 and array.shape[-1] == 3 else "minisblack"
    tifffile.imwrite(
        output_path,
        array,
        photometric=photometric,
        compression=None,
        metadata=None,
    )

    if verify:
        _verify_tiff(array, output_path)


def _stack_three_bands(paths: list[Path]) -> np.ndarray:
    """Stack three single-band inputs in the caller-supplied band order."""
    if len(paths) != 3:
        raise ValueError("--stack requires exactly three JP2 files.")

    arrays = [read_jp2(path) for path in paths]
    if any(array.ndim != 2 for array in arrays):
        raise ValueError("--stack expects three single-band JP2 files.")

    shape = arrays[0].shape
    dtype = arrays[0].dtype
    for path, array in zip(paths, arrays):
        if array.shape != shape:
            raise ValueError(
                f"All stacked bands must have the same shape: {path} has {array.shape}, "
                f"expected {shape}"
            )
        if array.dtype != dtype:
            raise ValueError(
                f"All stacked bands must have the same dtype: {path} has {array.dtype}, "
                f"expected {dtype}"
            )

    stacked = np.stack(arrays, axis=-1)
    del arrays
    return stacked


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert JP2 rasters to uncompressed, lossless TIFF files."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="JP2 file(s), or a directory containing JP2 files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output TIFF path for one/stacked input, or output directory for batch conversion.",
    )
    parser.add_argument(
        "--stack",
        action="store_true",
        help="Stack exactly three input bands in the supplied order into one 3-band TIFF.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing TIFF output files.",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip the pixel-by-pixel verification after writing.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    paths = _resolve_inputs(args.inputs)
    verify = not args.no_verify

    if args.stack:
        if any(path.is_dir() for path in args.inputs):
            raise ValueError(
                "Use explicit JP2 file paths with --stack so band order is unambiguous."
            )
        if args.output.suffix.lower() not in TIFF_EXTENSIONS:
            raise ValueError("--output must be a .tif or .tiff file when using --stack.")
        array = _stack_three_bands(paths)
        write_tiff(array, args.output, overwrite=args.overwrite, verify=verify)
        print(f"Converted {len(paths)} JP2 files -> {args.output} shape={array.shape} dtype={array.dtype}")
        return

    if len(paths) == 1 and args.output.suffix.lower() in TIFF_EXTENSIONS:
        destinations = [(paths[0], args.output)]
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        destinations = [
            (path, args.output / f"{path.stem}.tif")
            for path in paths
        ]

    output_names = [destination.name for _, destination in destinations]
    if len(set(output_names)) != len(output_names):
        raise ValueError("Multiple inputs would produce the same TIFF filename.")

    for source_path, destination in destinations:
        array = read_jp2(source_path)
        write_tiff(array, destination, overwrite=args.overwrite, verify=verify)
        print(f"Converted {source_path} -> {destination} shape={array.shape} dtype={array.dtype}")


if __name__ == "__main__":
    main()
