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

import inference_segformer_large_image as large_inference  # noqa: E402
from resource_guards import FilesystemStats, GIB  # noqa: E402


def _red_channel_probability(_model, batch, _device, _tile_size):
    return np.asarray(batch[:, 0], dtype=np.float32)


class _FilesystemProvider:
    def stats_for(self, path):
        return FilesystemStats("test-device", 100 * GIB, 100 * GIB)


class LargeSegFormerInferenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _run(self, source, mask, probability=None, **kwargs):
        with (
            mock.patch.object(large_inference, "load_segformer", return_value=object()),
            mock.patch.object(
                large_inference,
                "_predict_probability",
                side_effect=_red_channel_probability,
            ),
        ):
            return large_inference.infer_large_image_streaming(
                source,
                checkpoint_path=self.root / "unused.pth",
                output_mask=mask,
                output_probability=probability,
                tile_size=4,
                batch_size=2,
                threshold=0.5,
                device="cpu",
                runtime_reserve_gib="0.01",
                progress_every=1000,
                _filesystem_provider=_FilesystemProvider(),
                **kwargs,
            )

    def test_npy_memmap_writes_mask_and_probability_without_full_output_array(self):
        source = self.root / "source.npy"
        mask = self.root / "mask.tif"
        probability = self.root / "probability.tif"
        image = np.zeros((9, 10, 3), dtype=np.uint8)
        image[:4, :4, 0] = 255
        image[-1, -1, 0] = 255
        np.save(source, image)

        result = self._run(source, mask, probability)

        expected_mask = np.zeros((9, 10), dtype=np.uint8)
        expected_mask[:4, :4] = 255
        expected_mask[-1, -1] = 255
        np.testing.assert_array_equal(tifffile.imread(mask), expected_mask)
        np.testing.assert_allclose(
            tifffile.imread(probability),
            image[:, :, 0].astype(np.float32) / 255.0,
        )
        self.assertEqual(result["reader_backend"], "npy-memmap")
        self.assertEqual(result["cloud_pixel_count"], 17)
        self.assertEqual(result["stride"], 2)
        self.assertEqual(result["overlap"], 2)
        self.assertFalse(result["mask_bigtiff"])

    def test_overlapping_windows_average_probability_before_threshold(self):
        source = self.root / "source.npy"
        mask = self.root / "mask.tif"
        probability = self.root / "probability.tif"
        np.save(source, np.zeros((5, 5, 3), dtype=np.uint8))
        tile_values = iter((0.1, 0.3, 0.5, 0.7))

        def _constant_probability(_model, batch, _device, tile_size):
            value = next(tile_values)
            return np.full(
                (batch.shape[0], tile_size, tile_size),
                value,
                dtype=np.float32,
            )

        with (
            mock.patch.object(large_inference, "load_segformer", return_value=object()),
            mock.patch.object(
                large_inference,
                "_predict_probability",
                side_effect=_constant_probability,
            ),
        ):
            result = large_inference.infer_large_image_streaming(
                source,
                checkpoint_path=self.root / "unused.pth",
                output_mask=mask,
                output_probability=probability,
                tile_size=4,
                stride=2,
                batch_size=1,
                threshold=0.45,
                device="cpu",
                runtime_reserve_gib="0.01",
                progress_every=1000,
                _filesystem_provider=_FilesystemProvider(),
            )

        expected_probability = np.asarray(
            [
                [0.1, 0.2, 0.2, 0.2, 0.3],
                [0.3, 0.4, 0.4, 0.4, 0.5],
                [0.3, 0.4, 0.4, 0.4, 0.5],
                [0.3, 0.4, 0.4, 0.4, 0.5],
                [0.5, 0.6, 0.6, 0.6, 0.7],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(tifffile.imread(probability), expected_probability)
        np.testing.assert_array_equal(
            tifffile.imread(mask),
            np.where(expected_probability >= 0.45, 255, 0).astype(np.uint8),
        )
        self.assertEqual(result["tile_count"], 4)
        self.assertEqual(result["stride"], 2)
        self.assertFalse(list(self.root.glob(".segformer-probability-*.tmp")))

    def test_uncompressed_tiff_stream_mode_preserves_edge_pixels(self):
        source = self.root / "source.tif"
        mask = self.root / "mask.tif"
        image = np.zeros((5, 7, 3), dtype=np.uint8)
        image[4, 6, 0] = 255
        tifffile.imwrite(source, image, photometric="rgb", metadata={"axes": "YXC"})

        result = self._run(source, mask, tiff_read_mode="stream")

        expected_mask = np.zeros((5, 7), dtype=np.uint8)
        expected_mask[4, 6] = 255
        np.testing.assert_array_equal(tifffile.imread(mask), expected_mask)
        self.assertEqual(result["reader_backend"], "memmap")

    def test_bigtiff_threshold_accounts_for_tiff_header_margin(self):
        self.assertFalse(
            large_inference._needs_bigtiff(
                large_inference._CLASSIC_TIFF_LIMIT
                - large_inference._BIGTIFF_SAFETY_MARGIN
                - 1
            )
        )
        self.assertTrue(
            large_inference._needs_bigtiff(
                large_inference._CLASSIC_TIFF_LIMIT
                - large_inference._BIGTIFF_SAFETY_MARGIN
            )
        )


if __name__ == "__main__":
    unittest.main()
