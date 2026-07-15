"""Convert the 95-Cloud TIFF dataset into paired image and mask patches.

Expected input layout (the layout distributed by Kaggle):
    <data_dir>/train_{red,green,blue,nir,gt}/<channel>_<scene>.TIF

Output images are grouped by their source-patch label while masks are stored in
a separate ``masks`` directory with matching filenames.
"""

import argparse
from pathlib import Path

import numpy as np
import tifffile as tiff
from tqdm import tqdm


CHANNELS = ("red", "green", "blue", "nir")


def _find_file(folder: Path, prefix: str, scene: str) -> Path | None:
    """Find a channel file while tolerating extension and case differences."""
    if not folder.is_dir():
        return None
    expected = f"{prefix}_{scene}".lower()
    for path in folder.iterdir():
        if path.is_file() and path.stem.lower() == expected:
            return path
    return None


def _validate_cloud_ratio_threshold(value):
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"cloud_ratio_threshold must be between 0 and 1, got {value}")


def validate_output_pairs(output_dir):
    """Ensure every processed image has exactly one mask with the same name."""
    output_dir = Path(output_dir)
    image_paths = [
        path
        for label in ("cloud", "clear")
        for path in sorted((output_dir / label).glob("*.npy"))
    ]
    image_names = []
    seen_names = set()
    duplicate_names = set()
    for path in image_paths:
        image_names.append(path.name)
        if path.name in seen_names:
            duplicate_names.add(path.name)
        seen_names.add(path.name)
    duplicate_names = sorted(duplicate_names)
    if duplicate_names:
        raise ValueError(f"Duplicate image filenames across cloud/clear: {duplicate_names[:5]}")

    mask_dir = output_dir / "masks"
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Expected mask directory not found: {mask_dir}")
    mask_names = {path.name for path in mask_dir.glob("*.npy")}
    image_name_set = set(image_names)
    missing_masks = sorted(image_name_set - mask_names)
    orphan_masks = sorted(mask_names - image_name_set)
    if missing_masks or orphan_masks:
        raise ValueError(
            "Processed image/mask pairing failed: "
            f"missing_masks={missing_masks[:5]}, orphan_masks={orphan_masks[:5]}"
        )
    if not image_paths:
        raise ValueError(f"No full-size patches were created in {output_dir}")
    return len(image_paths)


def process_scene(
    scene,
    data_dir,
    output_dir,
    patch_size=384,
    cloud_ratio_threshold=0.10,
    channels=4,
):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    if patch_size <= 0:
        raise ValueError(f"patch_size must be greater than zero, got {patch_size}")
    if channels not in (3, 4):
        raise ValueError(f"channels must be 3 or 4, got {channels}")
    _validate_cloud_ratio_threshold(cloud_ratio_threshold)

    channel_dirs = {name: data_dir / f"train_{name}" for name in CHANNELS}
    names = CHANNELS[:channels]
    files = {name: _find_file(channel_dirs[name], name, scene) for name in names}
    gt_file = _find_file(data_dir / "train_gt", "gt", scene)

    missing = [str(channel_dirs[name]) for name, path in files.items() if path is None]
    if gt_file is None:
        missing.append(str(data_dir / "train_gt"))
    if missing:
        raise FileNotFoundError(f"Missing 95-Cloud files for scene {scene}: {', '.join(missing)}")

    arrays = [tiff.imread(files[name]) for name in names]
    gt = tiff.imread(gt_file)
    if any(array.ndim != 2 for array in arrays) or gt.ndim != 2:
        raise ValueError(f"Expected 2D channel arrays and mask for scene {scene}")
    if len({array.shape for array in arrays + [gt]}) != 1:
        raise ValueError(f"Channel/mask shape mismatch for scene {scene}")

    image = np.stack(arrays, axis=-1)
    height, width = gt.shape
    patch_id = 0
    for row in range(0, height, patch_size):
        for col in range(0, width, patch_size):
            if row + patch_size > height or col + patch_size > width:
                continue
            patch_gt = gt[row:row + patch_size, col:col + patch_size]
            patch_mask = (patch_gt > 0).astype(np.uint8)
            label = "cloud" if np.mean(patch_mask) >= cloud_ratio_threshold else "clear"
            patch = image[row:row + patch_size, col:col + patch_size]
            filename = f"{scene}_p{patch_id}.npy"
            np.save(output_dir / label / filename, patch)
            np.save(output_dir / "masks" / filename, patch_mask)
            patch_id += 1
    return patch_id


def main():
    parser = argparse.ArgumentParser(description="Preprocess the 95-Cloud dataset")
    parser.add_argument("--data_dir", default="data/95-Cloud/95-Cloud_training")
    parser.add_argument("--out_dir", default="data/processed/all")
    parser.add_argument(
        "--patch_size", type=int, default=384,
        help="Source patch size saved for training crops (default: 384)",
    )
    parser.add_argument(
        "--cloud_ratio_threshold",
        "--threshold",
        dest="cloud_ratio_threshold",
        type=float,
        default=0.10,
        help="Minimum cloud-pixel ratio for a cloud label (default: 0.10)",
    )
    parser.add_argument("--channels", type=int, choices=(3, 4), default=4)
    parser.add_argument("--force", action="store_true", help="Delete existing processed .npy files before preprocessing")
    args = parser.parse_args()

    if args.patch_size <= 0:
        raise ValueError("patch_size must be greater than zero")
    _validate_cloud_ratio_threshold(args.cloud_ratio_threshold)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.out_dir)
    for directory in ("cloud", "clear", "masks"):
        (output_dir / directory).mkdir(parents=True, exist_ok=True)
    existing = [
        path
        for directory in ("cloud", "clear", "masks")
        for path in (output_dir / directory).glob("*.npy")
    ]
    if existing and not args.force:
        raise FileExistsError(
            f"Found {len(existing)} existing image/mask files in {output_dir}. "
            "Use --force to rebuild them."
        )
    if args.force:
        for path in existing:
            path.unlink()

    red_dir = data_dir / "train_red"
    if not red_dir.is_dir():
        raise FileNotFoundError(
            f"Could not find {red_dir}. Extract 95-Cloud so it contains train_red, "
            "train_green, train_blue, train_nir and train_gt."
        )

    scenes = sorted({path.stem[4:] for path in red_dir.iterdir()
                     if path.is_file() and path.stem.lower().startswith("red_")})
    if not scenes:
        raise ValueError(f"No red_*.TIF files found in {red_dir}")

    for scene in tqdm(scenes, desc="95-Cloud"):
        process_scene(
            scene,
            data_dir,
            output_dir,
            args.patch_size,
            args.cloud_ratio_threshold,
            args.channels,
        )
    patch_count = validate_output_pairs(output_dir)
    print(
        f"Preprocessing completed: {len(scenes)} scenes, {patch_count} paired patches "
        f"(cloud_ratio_threshold={args.cloud_ratio_threshold:.2f}) -> {output_dir}"
    )


if __name__ == "__main__":
    main()
