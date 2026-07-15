import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import tifffile as tiff


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DATA_SRC = SRC / "data"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(DATA_SRC))

from data.cloud_dataset import CloudDataset  # noqa: E402
from data.preprocess_95cloud import process_scene, validate_output_pairs  # noqa: E402
from data import split_dataset  # noqa: E402
from inference_large_image_trt import (  # noqa: E402
    calculate_cloud_coverage,
    is_image_accepted,
)
from train import initialize_wandb  # noqa: E402


def make_processed_dirs(root):
    for name in ("cloud", "clear", "masks"):
        (root / name).mkdir(parents=True, exist_ok=True)


def save_pair(root, label, filename, image, mask):
    np.save(root / label / filename, image)
    np.save(root / "masks" / filename, mask)


class Preprocess95CloudTests(unittest.TestCase):
    def test_process_scene_saves_binary_mask_pairs_at_ten_percent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            raw = temp / "raw"
            output = temp / "processed"
            make_processed_dirs(output)

            image = np.arange(400, dtype=np.uint16).reshape(20, 20)
            gt = np.zeros((20, 20), dtype=np.uint8)
            gt[0, :10] = 255  # 10/100 cloud pixels in the first 10x10 patch.
            for channel in ("red", "green", "blue", "nir"):
                channel_dir = raw / f"train_{channel}"
                channel_dir.mkdir(parents=True)
                tiff.imwrite(channel_dir / f"{channel}_scene_a.TIF", image)
            gt_dir = raw / "train_gt"
            gt_dir.mkdir(parents=True)
            tiff.imwrite(gt_dir / "gt_scene_a.TIF", gt)

            count = process_scene(
                "scene_a",
                raw,
                output,
                patch_size=10,
                cloud_ratio_threshold=0.10,
                channels=4,
            )

            self.assertEqual(count, 4)
            self.assertEqual(validate_output_pairs(output), 4)
            self.assertTrue((output / "cloud" / "scene_a_p0.npy").is_file())
            mask = np.load(output / "masks" / "scene_a_p0.npy")
            self.assertEqual(mask.dtype, np.uint8)
            self.assertEqual(set(np.unique(mask)), {0, 1})
            self.assertEqual(int(mask.sum()), 10)


class CloudDatasetTests(unittest.TestCase):
    def test_threshold_boundary_and_estimated_label_counts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            make_processed_dirs(root)
            image = np.zeros((10, 10, 3), dtype=np.uint8)

            cloud_mask = np.zeros((10, 10), dtype=np.uint8)
            cloud_mask.flat[:10] = 1
            clear_mask = np.zeros((10, 10), dtype=np.uint8)
            clear_mask.flat[:9] = 1
            save_pair(root, "cloud", "scene_a_p0.npy", image, cloud_mask)
            save_pair(root, "clear", "scene_b_p0.npy", image, clear_mask)

            dataset = CloudDataset(
                root,
                is_train=False,
                target_channels=3,
                crop_size=10,
                cloud_ratio_threshold=0.10,
            )

            self.assertEqual(dataset[0][1].item(), 1.0)
            self.assertEqual(dataset[1][1].item(), 0.0)
            self.assertEqual(dataset.estimate_label_counts(4, seed=7), (4, 4))

    def test_random_crop_uses_matching_image_and_mask_coordinates(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            make_processed_dirs(root)
            mask = np.zeros((12, 12), dtype=np.uint8)
            mask[2:8, 4:10] = 1
            image = np.zeros((12, 12, 3), dtype=np.uint8)
            image[:, :, 0] = mask * 255
            save_pair(root, "cloud", "scene_a_p0.npy", image, mask)

            dataset = CloudDataset(
                root,
                transform=lambda tensor: tensor,
                is_train=True,
                target_channels=3,
                crop_size=10,
                cloud_ratio_threshold=0.10,
            )
            np.random.seed(11)
            image_tensor, label = dataset[0]
            image_cloud_ratio = float((image_tensor[0] > 0).float().mean())

            self.assertEqual(label.item(), float(image_cloud_ratio >= 0.10))

    def test_validation_center_crop_is_deterministic(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            make_processed_dirs(root)
            image = np.arange(12 * 12 * 3, dtype=np.uint16).reshape(12, 12, 3)
            mask = np.zeros((12, 12), dtype=np.uint8)
            mask[1:11, 1:11] = 1
            save_pair(root, "cloud", "scene_a_p0.npy", image, mask)

            dataset = CloudDataset(
                root,
                is_train=False,
                target_channels=3,
                crop_size=10,
                cloud_ratio_threshold=0.10,
            )
            first_image, first_label = dataset[0]
            second_image, second_label = dataset[0]

            self.assertTrue(first_image.equal(second_image))
            self.assertEqual(first_label.item(), second_label.item())

    def test_missing_mask_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            make_processed_dirs(root)
            np.save(root / "cloud" / "scene_a_p0.npy", np.zeros((10, 10, 3)))

            with self.assertRaisesRegex(ValueError, "missing_masks"):
                CloudDataset(root, target_channels=3, crop_size=10)

    def test_shape_mismatch_and_oversized_crop_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            make_processed_dirs(root)
            image = np.zeros((10, 10, 3), dtype=np.uint8)
            save_pair(
                root,
                "cloud",
                "scene_a_p0.npy",
                image,
                np.zeros((9, 10), dtype=np.uint8),
            )
            dataset = CloudDataset(root, target_channels=3, crop_size=10)
            with self.assertRaisesRegex(ValueError, "shape mismatch"):
                dataset[0]

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            make_processed_dirs(root)
            save_pair(
                root,
                "clear",
                "scene_b_p0.npy",
                np.zeros((10, 10, 3), dtype=np.uint8),
                np.zeros((10, 10), dtype=np.uint8),
            )
            dataset = CloudDataset(root, target_channels=3, crop_size=11)
            with self.assertRaisesRegex(ValueError, "exceeds patch shape"):
                dataset[0]


class SplitDatasetTests(unittest.TestCase):
    def test_scene_split_copies_every_image_with_its_mask(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            source = temp / "all"
            output = temp / "split"
            make_processed_dirs(source)
            for index, scene in enumerate(("a", "b", "c", "d")):
                label = "cloud" if index % 2 == 0 else "clear"
                filename = f"scene_{scene}_p0.npy"
                (source / label / filename).write_bytes(b"image")
                (source / "masks" / filename).write_bytes(b"mask")

            argv = [
                "split_dataset.py",
                "--src_dir", str(source),
                "--out_dir", str(output),
                "--val_ratio", "0.25",
                "--test_ratio", "0.25",
                "--seed", "7",
            ]
            with mock.patch.object(sys, "argv", argv):
                split_dataset.main()

            manifest = json.loads((output / "scene_split_manifest.json").read_text())
            all_scenes = set()
            for split_name, details in manifest.items():
                all_scenes.update(details["scenes"])
                self.assertTrue(details["pairing_valid"])
                self.assertEqual(details["image_count"], details["mask_count"])
                image_names = {
                    path.name
                    for label in ("cloud", "clear")
                    for path in (output / split_name / label).glob("*.npy")
                }
                mask_names = {
                    path.name for path in (output / split_name / "masks").glob("*.npy")
                }
                self.assertEqual(image_names, mask_names)
            self.assertEqual(all_scenes, {"scene_a", "scene_b", "scene_c", "scene_d"})

    def test_source_pairing_rejects_orphan_mask(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            make_processed_dirs(root)
            (root / "cloud" / "scene_a_p0.npy").write_bytes(b"image")
            (root / "masks" / "scene_a_p0.npy").write_bytes(b"mask")
            (root / "masks" / "orphan_p0.npy").write_bytes(b"mask")
            scene_files = split_dataset.collect_scene_files(root)

            with self.assertRaisesRegex(ValueError, "orphan_masks"):
                split_dataset.validate_image_mask_pairs(root, scene_files)


class LargeImageInferenceTests(unittest.TestCase):
    def test_cloud_coverage_rejects_at_sixty_percent(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask.flat[:60] = 255

        self.assertEqual(calculate_cloud_coverage(mask), 0.60)
        self.assertFalse(is_image_accepted(mask, 0.60))

        mask.flat[59] = 0
        self.assertEqual(calculate_cloud_coverage(mask), 0.59)
        self.assertTrue(is_image_accepted(mask, 0.60))

    def test_cloud_coverage_threshold_is_validated(self):
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            is_image_accepted(np.zeros((2, 2), dtype=np.uint8), 1.1)


class WandbLoggingTests(unittest.TestCase):
    def test_initialize_wandb_passes_tracking_configuration(self):
        fake_wandb = mock.Mock()
        fake_run = mock.Mock()
        fake_wandb.init.return_value = fake_run
        args = SimpleNamespace(
            wandb=True,
            wandb_project="cube-nano",
            wandb_entity="example-team",
            wandb_run_name="baseline-rgbnir",
            wandb_group="95-cloud",
            wandb_tags=["kaggle", "baseline"],
            wandb_mode="offline",
            out_dir="checkpoints",
        )
        training_config = {"epochs": 2, "channels": 4}

        with mock.patch.dict(sys.modules, {"wandb": fake_wandb}):
            result = initialize_wandb(args, training_config)

        self.assertIs(result, fake_run)
        fake_wandb.init.assert_called_once_with(
            project="cube-nano",
            entity="example-team",
            name="baseline-rgbnir",
            group="95-cloud",
            tags=["kaggle", "baseline"],
            config=training_config,
            dir="checkpoints",
            mode="offline",
        )


if __name__ == "__main__":
    unittest.main()
