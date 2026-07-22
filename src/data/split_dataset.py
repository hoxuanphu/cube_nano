import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path


def scene_id_from_patch(path):
    """Extract scene ID from patch filename.

    Patch filenames follow the pattern ``{scene_id}_p{patch_index}.npy``
    (e.g. ``patch_10_11_p5.npy`` → scene ID ``patch_10_11``).
    """
    stem = Path(path).stem
    if "_p" not in stem:
        raise ValueError(
            f"Cannot infer scene id from patch name: {path}. "
            "Expected format: {{scene_id}}_p{{index}}.npy"
        )
    return stem.rsplit("_p", 1)[0]


def collect_scene_files(src_dir):
    """Group all patches by scene ID across cloud/clear classes."""
    scene_files = {}
    src_dir = Path(src_dir)
    seen_names = set()
    for label in ("cloud", "clear"):
        label_dir = src_dir / label
        if not label_dir.exists():
            raise FileNotFoundError(f"Expected class directory not found: {label_dir}")
        for path in sorted(label_dir.glob("*.npy")):
            if path.name in seen_names:
                raise ValueError(f"Duplicate patch filename across cloud/clear: {path.name}")
            seen_names.add(path.name)
            scene_id = scene_id_from_patch(path)
            scene_files.setdefault(scene_id, {"cloud": [], "clear": []})[label].append(path)
    return scene_files


def validate_image_mask_pairs(src_dir, scene_files):
    """Validate all source image/mask pairs before mutating split outputs."""
    src_dir = Path(src_dir)
    mask_dir = src_dir / "masks"
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Expected mask directory not found: {mask_dir}")

    image_paths = [
        path
        for classes in scene_files.values()
        for label in ("cloud", "clear")
        for path in classes[label]
    ]
    image_names = {path.name for path in image_paths}
    mask_paths = {path.name: path for path in mask_dir.glob("*.npy")}
    mask_names = set(mask_paths)
    missing_masks = sorted(image_names - mask_names)
    orphan_masks = sorted(mask_names - image_names)
    if missing_masks or orphan_masks:
        raise ValueError(
            "Source image/mask pairing failed: "
            f"missing_masks={missing_masks[:5]}, orphan_masks={orphan_masks[:5]}"
        )
    validity_dir = src_dir / "validity"
    validity_paths = None
    if validity_dir.is_dir():
        validity_paths = {path.name: path for path in validity_dir.glob("*.npy")}
        validity_names = set(validity_paths)
        missing_validity = sorted(image_names - validity_names)
        orphan_validity = sorted(validity_names - image_names)
        if missing_validity or orphan_validity:
            raise ValueError(
                "Source image/validity pairing failed: "
                f"missing_validity={missing_validity[:5]}, orphan_validity={orphan_validity[:5]}"
            )
    raw_mask_dir = src_dir / "raw_masks"
    raw_mask_paths = None if not raw_mask_dir.is_dir() else {
        path.name: path for path in raw_mask_dir.glob("*.npy")
    }
    if raw_mask_paths is not None:
        raw_names = set(raw_mask_paths)
        missing_raw = sorted(image_names - raw_names)
        orphan_raw = sorted(raw_names - image_names)
        if missing_raw or orphan_raw:
            raise ValueError(
                "Source image/raw-ground-truth pairing failed: "
                f"missing_raw={missing_raw[:5]}, orphan_raw={orphan_raw[:5]}"
            )
    return mask_paths, validity_paths, raw_mask_paths


def scene_split_lineage_id(
    src_dir: str | Path,
    scene_splits: dict[str, list[str]],
    *,
    preprocessing_config: dict | None = None,
) -> str:
    """Hash source files, scene assignment and preprocessing config deterministically."""

    src_dir = Path(src_dir)
    files = []
    for path in sorted(src_dir.rglob("*.npy")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files.append({"path": str(path.relative_to(src_dir)).replace("\\", "/"), "sha256": digest})
    payload = {
        "schema_version": 1,
        "files": files,
        "scene_splits": {key: sorted(value) for key, value in sorted(scene_splits.items())},
        "preprocessing_config": preprocessing_config or {},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def split_scenes(scene_ids, val_ratio, test_ratio, seed):
    """Shuffle scene IDs and split into train/val/test lists."""
    scene_ids = list(scene_ids)
    rng = random.Random(seed)
    rng.shuffle(scene_ids)

    total = len(scene_ids)
    train_end = int(total * (1.0 - val_ratio - test_ratio))
    val_end = train_end + int(total * val_ratio)

    return {
        "train": scene_ids[:train_end],
        "val": scene_ids[train_end:val_end],
        "test": scene_ids[val_end:],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Split processed cloud patches into train/val/test folders (scene-level)"
    )
    parser.add_argument(
        "--src_dir", type=str, default="data/processed/all",
        help="Source folder with cloud/ and clear/ patches",
    )
    parser.add_argument(
        "--out_dir", type=str, default="data/processed",
        help="Output root containing train/val/test",
    )
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--move", action="store_true", help="Move files instead of copying them")
    parser.add_argument(
        "--force", action="store_true",
        help="Clear existing split directories before writing. Required if output dirs are non-empty.",
    )
    args = parser.parse_args()

    if args.val_ratio < 0 or args.test_ratio < 0 or args.val_ratio + args.test_ratio >= 1:
        raise ValueError("val_ratio and test_ratio must be >= 0 and sum to less than 1")

    src_root = Path(args.src_dir)
    if not src_root.exists():
        raise FileNotFoundError(f"Source directory not found: {src_root}")

    operation = shutil.move if args.move else shutil.copy2

    # Check for existing files in output directories
    existing_files = []
    source_subdirs = ["cloud", "clear", "masks"]
    if (src_root / "validity").is_dir():
        source_subdirs.append("validity")
    if (src_root / "raw_masks").is_dir():
        source_subdirs.append("raw_masks")
    for split in ("train", "val", "test"):
        for class_name in source_subdirs:
            split_dir = Path(args.out_dir, split, class_name)
            if split_dir.exists():
                existing_files.extend(split_dir.glob("*.npy"))
    if existing_files and not args.force:
        raise RuntimeError(
            f"Output directories already contain {len(existing_files)} .npy files. "
            "Use --force to clear them and re-split, or choose a different --out_dir."
        )

    # Collect and group patches by scene
    scene_files = collect_scene_files(src_root)
    if not scene_files:
        raise ValueError(f"No processed patches found under {src_root}")
    mask_paths, validity_paths, raw_mask_paths = validate_image_mask_pairs(src_root, scene_files)

    print(f"Found {len(scene_files)} scenes.")

    # Split scenes (not files)
    scene_splits = split_scenes(
        scene_files.keys(), args.val_ratio, args.test_ratio, args.seed
    )

    # Verify no overlap
    split_sets = {split: set(ids) for split, ids in scene_splits.items()}
    assert split_sets["train"].isdisjoint(split_sets["val"]), "train/val scene overlap!"
    assert split_sets["train"].isdisjoint(split_sets["test"]), "train/test scene overlap!"
    assert split_sets["val"].isdisjoint(split_sets["test"]), "val/test scene overlap!"

    # Create output directories (clear old files if --force)
    for split in scene_splits:
        for class_name in source_subdirs:
            out_dir = Path(args.out_dir, split, class_name)
            out_dir.mkdir(parents=True, exist_ok=True)
            if args.force:
                for old_file in out_dir.glob("*.npy"):
                    old_file.unlink()

    # Copy/move patches grouped by scene
    manifest = {}
    for split, scene_ids in scene_splits.items():
        counts = {"cloud": 0, "clear": 0}
        mask_count = 0
        for scene_id in scene_ids:
            for label in ("cloud", "clear"):
                for src_path in scene_files[scene_id][label]:
                    dest = Path(args.out_dir, split, label, src_path.name)
                    operation(str(src_path), str(dest))
                    mask_src = mask_paths[src_path.name]
                    mask_dest = Path(args.out_dir, split, "masks", src_path.name)
                    operation(str(mask_src), str(mask_dest))
                    if validity_paths is not None:
                        operation(
                            str(validity_paths[src_path.name]),
                            str(Path(args.out_dir, split, "validity", src_path.name)),
                        )
                    if raw_mask_paths is not None:
                        operation(
                            str(raw_mask_paths[src_path.name]),
                            str(Path(args.out_dir, split, "raw_masks", src_path.name)),
                        )
                    counts[label] += 1
                    mask_count += 1
        image_count = counts["cloud"] + counts["clear"]
        manifest[split] = {
            "scene_count": len(scene_ids),
            "scenes": scene_ids,
            "patch_counts": counts,
            "image_count": image_count,
            "mask_count": mask_count,
            "pairing_valid": image_count == mask_count,
        }
        print(
            f"{split}: scenes={len(scene_ids)}, "
            f"cloud={counts['cloud']}, clear={counts['clear']}, masks={mask_count}"
        )

    lineage_id = scene_split_lineage_id(
        src_root,
        scene_splits,
        preprocessing_config={
            "patch_size": "source-manifest-defined",
            "validity": "separate-mask-or-all-valid",
            "seed": args.seed,
        },
    )
    for details in manifest.values():
        details["lineage_id"] = lineage_id
        details["validity_artifact"] = validity_paths is not None
        details["raw_ground_truth_artifact"] = raw_mask_paths is not None
    lineage_manifest = {
        "schema_version": 1,
        "lineage_id": lineage_id,
        "source_directory": str(src_root.resolve()),
        "scene_splits": scene_splits,
        "validity_artifact": validity_paths is not None,
        "raw_ground_truth_artifact": raw_mask_paths is not None,
    }
    # Save manifest
    manifest_path = Path(args.out_dir, "scene_split_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    with open(Path(args.out_dir, "scene_split_lineage.json"), "w") as f:
        json.dump(lineage_manifest, f, indent=2, sort_keys=True)
    print(f"\nScene split manifest saved: {manifest_path}")
    print("Dataset split completed (scene-level).")


if __name__ == "__main__":
    main()
