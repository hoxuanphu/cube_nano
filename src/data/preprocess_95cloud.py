"""Convert the 95-Cloud TIFF dataset into paired image and mask patches.

Canonical input layout:
    <data_dir>/train_{red,green,blue,nir,gt}/<channel>_<scene>.TIF

Nested Kaggle layouts are also supported when the band can be inferred from
the TIFF filename or one of its parent directory names.

Output images are grouped by their source-patch label while masks are stored in
a separate ``masks`` directory with matching filenames.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import tifffile as tiff
from tqdm import tqdm


CHANNELS = ("red", "green", "blue", "nir")
SUPPORTED_SUFFIXES = {".tif", ".tiff"}
BAND_ALIASES = {
    "red": ("red",),
    "green": ("green",),
    "blue": ("blue",),
    "nir": ("nir", "near_infrared", "nearinfrared"),
    "gt": ("gt", "mask", "masks", "label", "labels", "ground_truth", "groundtruth"),
}
KNOWN_GROUND_TRUTH_VALUES = frozenset({0, 1, 255})


def _normalized_name(value):
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def _detect_band(path):
    """Infer a band from a TIFF filename or one of its parent directories."""
    path = Path(path)
    stem = _normalized_name(path.stem)
    for band, aliases in BAND_ALIASES.items():
        if any(stem == alias or stem.startswith(f"{alias}_") for alias in aliases):
            return band

    for parent in reversed(path.parents[:4]):
        name = _normalized_name(parent.name)
        tokens = set(name.split("_"))
        for band, aliases in BAND_ALIASES.items():
            if any(alias in tokens or name == alias or name.endswith(f"_{alias}") for alias in aliases):
                return band
    return None


def _scene_id(path, band):
    """Normalize filenames from band-specific folders to one scene identifier."""
    stem = _normalized_name(Path(path).stem)
    aliases = BAND_ALIASES[band]
    for alias in aliases:
        for prefix in (f"{alias}_", f"train_{alias}_"):
            if stem.startswith(prefix):
                return stem[len(prefix):]
        for suffix in (f"_{alias}", f"_train_{alias}"):
            if stem.endswith(suffix):
                return stem[:-len(suffix)]
    return stem


def discover_scene_files(data_dir, channels=4):
    """Discover complete 95-Cloud scenes across common Kaggle layouts."""
    data_dir = Path(data_dir)
    if channels not in (3, 4):
        raise ValueError(f"channels must be 3 or 4, got {channels}")
    if not data_dir.is_dir():
        raise FileNotFoundError(f"95-Cloud directory not found: {data_dir}")

    records = {}
    band_counts = {band: 0 for band in (*CHANNELS, "gt")}
    sample_paths = []
    for path in data_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        if len(sample_paths) < 10:
            sample_paths.append(str(path.relative_to(data_dir)))
        band = _detect_band(path)
        if band is None:
            continue
        scene = _scene_id(path, band)
        if not scene:
            continue
        scene_record = records.setdefault(scene, {})
        if band in scene_record and scene_record[band] != path:
            raise ValueError(
                f"Duplicate {band} files for scene {scene}: "
                f"{scene_record[band]} and {path}"
            )
        scene_record[band] = path
        band_counts[band] += 1

    required_bands = set(CHANNELS[:channels]) | {"gt"}
    complete = {
        scene: record
        for scene, record in records.items()
        if required_bands.issubset(record)
    }
    if not complete:
        top_level = sorted(path.name for path in data_dir.iterdir())[:30]
        raise FileNotFoundError(
            "Could not find complete 95-Cloud scenes. "
            f"root={data_dir}, required_bands={sorted(required_bands)}, "
            f"detected_band_counts={band_counts}, top_level={top_level}, "
            f"sample_tiff_paths={sample_paths}"
        )
    return dict(sorted(complete.items()))


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


def decode_ground_truth(ground_truth, invalid_values=()):
    """Decode audited 95-Cloud binary labels without losing invalid pixels."""

    ground_truth = np.asarray(ground_truth)
    if ground_truth.ndim != 2:
        raise ValueError(f"ground truth must be a 2D array, got {ground_truth.shape}")
    if not np.issubdtype(ground_truth.dtype, np.integer) and not np.issubdtype(ground_truth.dtype, np.bool_):
        raise ValueError(f"ground truth must be integer or boolean, got {ground_truth.dtype}")
    invalid = {int(value) for value in invalid_values}
    if any(value < 0 or value > 255 for value in invalid):
        raise ValueError("invalid ground-truth values must be in [0, 255]")
    observed = {int(value) for value in np.unique(ground_truth)}
    unknown = observed - KNOWN_GROUND_TRUTH_VALUES - invalid
    if unknown:
        raise ValueError(
            f"unsupported ground-truth values {sorted(unknown)}; "
            "audit the encoding before preprocessing"
        )
    valid = ~np.isin(ground_truth, tuple(invalid))
    cloud = valid & np.isin(ground_truth, (1, 255))
    return cloud.astype(np.uint8), valid.astype(np.uint8)


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
    validity_dir = output_dir / "validity"
    if validity_dir.is_dir():
        validity_names = {path.name for path in validity_dir.glob("*.npy")}
        missing_validity = sorted(image_name_set - validity_names)
        orphan_validity = sorted(validity_names - image_name_set)
        if missing_validity or orphan_validity:
            raise ValueError(
                "Processed image/validity pairing failed: "
                f"missing_validity={missing_validity[:5]}, orphan_validity={orphan_validity[:5]}"
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
    scene_files=None,
    invalid_ground_truth_values=(),
):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    for directory in ("cloud", "clear", "masks", "validity", "raw_masks"):
        (output_dir / directory).mkdir(parents=True, exist_ok=True)
    if patch_size is not None and patch_size <= 0:
        raise ValueError(f"patch_size must be greater than zero, got {patch_size}")
    if channels not in (3, 4):
        raise ValueError(f"channels must be 3 or 4, got {channels}")
    _validate_cloud_ratio_threshold(cloud_ratio_threshold)

    names = CHANNELS[:channels]
    if scene_files is None:
        discovered = discover_scene_files(data_dir, channels=channels)
        if scene not in discovered:
            raise FileNotFoundError(f"Scene {scene} not found under {data_dir}")
        scene_files = discovered[scene]
    files = {name: Path(scene_files[name]) for name in names}
    gt_file = Path(scene_files["gt"])

    arrays = [tiff.imread(files[name]) for name in names]
    gt = tiff.imread(gt_file)
    if any(array.ndim != 2 for array in arrays) or gt.ndim != 2:
        raise ValueError(f"Expected 2D channel arrays and mask for scene {scene}")
    if len({array.shape for array in arrays + [gt]}) != 1:
        raise ValueError(f"Channel/mask shape mismatch for scene {scene}")

    image = np.stack(arrays, axis=-1)
    height, width = gt.shape

    if patch_size is None:
        patch_mask, patch_validity = decode_ground_truth(
            gt,
            invalid_values=invalid_ground_truth_values,
        )
        valid_count = int(np.count_nonzero(patch_validity))
        cloud_ratio = float(np.count_nonzero(patch_mask)) / valid_count if valid_count else 0.0
        label = "cloud" if cloud_ratio >= cloud_ratio_threshold else "clear"
        filename = f"{scene}_p0.npy"
        np.save(output_dir / label / filename, image)
        np.save(output_dir / "masks" / filename, patch_mask)
        np.save(output_dir / "validity" / filename, patch_validity)
        np.save(output_dir / "raw_masks" / filename, gt)
        return 1

    patch_id = 0
    for row in range(0, height, patch_size):
        for col in range(0, width, patch_size):
            if row + patch_size > height or col + patch_size > width:
                continue
            patch_gt = gt[row:row + patch_size, col:col + patch_size]
            patch_mask, patch_validity = decode_ground_truth(
                patch_gt,
                invalid_values=invalid_ground_truth_values,
            )
            valid_count = int(np.count_nonzero(patch_validity))
            cloud_ratio = (
                float(np.count_nonzero(patch_mask)) / valid_count
                if valid_count
                else 0.0
            )
            label = "cloud" if cloud_ratio >= cloud_ratio_threshold else "clear"
            patch = image[row:row + patch_size, col:col + patch_size]
            filename = f"{scene}_p{patch_id}.npy"
            np.save(output_dir / label / filename, patch)
            np.save(output_dir / "masks" / filename, patch_mask)
            np.save(output_dir / "validity" / filename, patch_validity)
            np.save(output_dir / "raw_masks" / filename, patch_gt)
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
        "--keep-native-size",
        action="store_true",
        help="Save one full-resolution image/mask pair per scene for flexible-size segmentation training",
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
    parser.add_argument(
        "--invalid-ground-truth-values",
        nargs="*",
        type=int,
        default=(),
        help="Explicit raw GT values to exclude; unknown values fail closed",
    )
    parser.add_argument("--force", action="store_true", help="Delete existing processed .npy files before preprocessing")
    args = parser.parse_args()

    if args.patch_size <= 0:
        raise ValueError("patch_size must be greater than zero")
    if args.keep_native_size and args.channels != 3:
        raise ValueError("native-size SegFormer preprocessing requires --channels 3")
    _validate_cloud_ratio_threshold(args.cloud_ratio_threshold)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.out_dir)
    for directory in ("cloud", "clear", "masks", "validity", "raw_masks"):
        (output_dir / directory).mkdir(parents=True, exist_ok=True)
    existing = [
        path
        for directory in ("cloud", "clear", "masks", "validity", "raw_masks")
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

    scene_files = discover_scene_files(data_dir, channels=args.channels)
    scenes = sorted(scene_files)

    effective_patch_size = None if args.keep_native_size else args.patch_size
    for scene in tqdm(scenes, desc="95-Cloud"):
        process_scene(
            scene,
            data_dir,
            output_dir,
            effective_patch_size,
            args.cloud_ratio_threshold,
            args.channels,
            scene_files[scene],
            args.invalid_ground_truth_values,
        )
    patch_count = validate_output_pairs(output_dir)
    print(
        f"Preprocessing completed: {len(scenes)} scenes, {patch_count} paired patches "
        f"(cloud_ratio_threshold={args.cloud_ratio_threshold:.2f}, "
        f"source_size={'native' if args.keep_native_size else args.patch_size}) -> {output_dir}"
    )


if __name__ == "__main__":
    main()
