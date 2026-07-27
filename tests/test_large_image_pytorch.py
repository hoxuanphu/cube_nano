import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import tifffile
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from inference_large_image import CloudTorchInfer, process_large_image  # noqa: E402
from resource_guards import (  # noqa: E402
    FilesystemStats,
    FixedMemoryInfoProvider,
    GIB,
)


class _MeanLogitModel(torch.nn.Module):
    def forward(self, batch):
        return batch.mean(dim=(1, 2, 3), keepdim=True)


class _FilesystemProvider:
    def stats_for(self, path):
        return FilesystemStats("test-device", 100 * GIB, 100 * GIB)


class CloudTorchInferTests(unittest.TestCase):
    def test_infer_batch_returns_thresholded_probabilities(self):
        with mock.patch(
            "inference_large_image.load_model", return_value=_MeanLogitModel()
        ):
            infer = CloudTorchInfer(
                "unused.pth",
                channels=3,
                patch_size=4,
                threshold=0.6,
                device="cpu",
            )

        batch = np.stack(
            [
                np.zeros((3, 4, 4), dtype=np.float32),
                np.ones((3, 4, 4), dtype=np.float32),
            ]
        )
        predictions, probabilities = infer.infer_batch(batch)

        np.testing.assert_array_equal(predictions, [False, True])
        np.testing.assert_allclose(
            probabilities,
            [0.5, 1.0 / (1.0 + np.exp(-1.0))],
            rtol=1e-6,
        )

    def test_infer_batch_rejects_wrong_patch_shape(self):
        with mock.patch(
            "inference_large_image.load_model", return_value=_MeanLogitModel()
        ):
            infer = CloudTorchInfer(
                "unused.pth",
                channels=3,
                patch_size=4,
                device="cpu",
            )

        with self.assertRaisesRegex(ValueError, "Expected batch shape"):
            infer.infer_batch(np.zeros((1, 3, 5, 4), dtype=np.float32))


class LargeImagePyTorchIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_process_large_tiff_writes_edge_safe_mask(self):
        source = self.root / "source.tif"
        output = self.root / "mask.tif"
        image = np.zeros((9, 10, 3), dtype=np.uint8)
        image[:4, :4, :] = 255
        tifffile.imwrite(
            source,
            image,
            photometric="rgb",
            compression="deflate",
            rowsperstrip=3,
            metadata={"axes": "YXC"},
        )

        with mock.patch(
            "inference_large_image.load_model", return_value=_MeanLogitModel()
        ):
            result = process_large_image(
                source,
                model_path="unused.pth",
                out_mask=output,
                patch_size=4,
                channels=3,
                batch_size=2,
                threshold=0.6,
                device="cpu",
                runtime_reserve_gib="0.01",
                tiff_cache_dir=self.root / "cache",
                _memory_provider=FixedMemoryInfoProvider(8 * GIB),
                _filesystem_provider=_FilesystemProvider(),
            )

        mask = tifffile.imread(output)
        expected = np.zeros((9, 10), dtype=np.uint8)
        expected[:4, :4] = 255
        np.testing.assert_array_equal(mask, expected)
        self.assertEqual(result["device"], "cpu")
        self.assertEqual(result["reader_backend"], "ram")


if __name__ == "__main__":
    unittest.main()
