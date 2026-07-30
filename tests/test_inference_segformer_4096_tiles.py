import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import tifffile


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import inference_segformer_4096_tiles as fixed_grid_inference  # noqa: E402
from resource_guards import FilesystemStats, GIB  # noqa: E402


def _red_channel_probability(_model, batch, _device, _tile_size):
    return np.asarray(batch[:, 0], dtype=np.float32)


class _FilesystemProvider:
    def stats_for(self, _path):
        return FilesystemStats("test-device", 100 * GIB, 100 * GIB)


class FixedGridSegFormerInferenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _run(self, source, **kwargs):
        with (
            mock.patch.object(fixed_grid_inference, "load_segformer", return_value=object()),
            mock.patch.object(
                fixed_grid_inference,
                "_predict_probability",
                side_effect=_red_channel_probability,
            ),
        ):
            return fixed_grid_inference.infer_resized_4096_tiles(
                source,
                checkpoint_path=self.root / "unused.pth",
                batch_size=2,
                threshold=0.5,
                device="cpu",
                resize_size=8,
                tile_size=4,
                runtime_reserve_gib="0.01",
                _filesystem_provider=_FilesystemProvider(),
                **kwargs,
            )

    def test_merged_mode_writes_a_mask_matching_the_resized_image(self):
        source = self.root / "source.npy"
        mask = self.root / "merged-mask.tif"
        image = np.zeros((3, 5, 3), dtype=np.uint8)
        image[:, :, 0] = 255
        np.save(source, image)

        result = self._run(
            source,
            output_mode="merged",
            output_mask=mask,
        )

        written_mask = tifffile.imread(mask)
        self.assertEqual(written_mask.shape, (8, 8))
        np.testing.assert_array_equal(written_mask, np.full((8, 8), 255, dtype=np.uint8))
        self.assertEqual(result["output_mode"], "merged")
        self.assertEqual(result["resized_image_shape"], [8, 8, 3])
        self.assertEqual(result["tile_count"], 4)
        self.assertEqual(result["cloud_pixel_count"], 64)
        self.assertEqual(result["mask_tiles"], [])

    def test_tiles_mode_writes_one_mask_per_tile(self):
        source = self.root / "source.npy"
        output_dir = self.root / "mask-tiles"
        image = np.zeros((3, 5, 3), dtype=np.uint8)
        image[:, :, 0] = 255
        np.save(source, image)

        result = self._run(
            source,
            output_mode="tiles",
            output_dir=output_dir,
        )

        tile_paths = sorted(output_dir.glob("*.tif"))
        self.assertEqual([path.name for path in tile_paths], [
            "mask_r00_c00.tif",
            "mask_r00_c01.tif",
            "mask_r01_c00.tif",
            "mask_r01_c01.tif",
        ])
        for tile_path in tile_paths:
            np.testing.assert_array_equal(
                tifffile.imread(tile_path),
                np.full((4, 4), 255, dtype=np.uint8),
            )
        self.assertIsNone(result["mask"])
        self.assertEqual(result["mask_tiles"], [str(path) for path in tile_paths])


if __name__ == "__main__":
    unittest.main()
