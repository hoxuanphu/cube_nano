"""Count cloud pixels in a TIFF mask."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile


def count_cloud_pixels(mask_path: str | Path) -> tuple[int, int, float]:
    """Return cloud-pixel count, total-pixel count, and cloud percentage."""
    path = Path(mask_path)
    mask = tifffile.imread(path)
    total_pixels = int(mask.size)
    cloud_pixels = int(np.count_nonzero(mask > 155))
    cloud_percentage = (cloud_pixels / total_pixels * 100.0) if total_pixels else 0.0
    return cloud_pixels, total_pixels, cloud_percentage


def main() -> None:
    parser = argparse.ArgumentParser(description="Count pixels equal to 255 in a TIFF mask")
    parser.add_argument("mask", type=Path, help="Path to the TIFF mask")
    args = parser.parse_args()

    cloud_pixels, total_pixels, cloud_percentage = count_cloud_pixels(args.mask)
    print(f"Mask: {args.mask}")
    print(f"Cloud pixels (value 255): {cloud_pixels}")
    print(f"Total pixels: {total_pixels}")
    print(f"Cloud percentage: {cloud_percentage:.6f}%")


if __name__ == "__main__":
    main()
