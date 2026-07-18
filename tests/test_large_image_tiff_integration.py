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

from inference_large_image_trt import (  # noqa: E402
    _atomic_write_tiff,
    process_large_image,
)
from resource_guards import (  # noqa: E402
    FilesystemStats,
    FixedMemoryInfoProvider,
    GIB,
)


class _FilesystemProvider:
    def stats_for(self, path):
        return FilesystemStats("test-device", 100 * GIB, 100 * GIB)


class _RecordingTRT:
    instances = []

    def __init__(self, *args, **kwargs):
        self.batches = []
        self.input_spec = kwargs["input_spec"]
        self.instances.append(self)

    def infer_batch(self, batch):
        self.batches.append(np.array(batch, copy=True))
        probabilities = np.asarray([item.mean() for item in batch], dtype=np.float32)
        return probabilities > 0.3, probabilities

    def infer(self, batch):
        predictions, probabilities = self.infer_batch(batch)
        return bool(predictions[0]), float(probabilities[0])


class _FailingTRT(_RecordingTRT):
    def infer_batch(self, batch):
        raise RuntimeError("injected inference failure")


class LargeImageTiffIntegrationTests(unittest.TestCase):
    def setUp(self):
        _RecordingTRT.instances.clear()
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.memory = FixedMemoryInfoProvider(8 * GIB)
        self.filesystem = _FilesystemProvider()

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _write_source(self, name="source.tif"):
        source = self.root / name
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
        return source, image

    def _process(self, source, factory=_RecordingTRT, **kwargs):
        return process_large_image(
            source,
            "unused.engine",
            out_mask=kwargs.pop("out_mask", self.root / "mask.tif"),
            patch_size=4,
            channels=3,
            batch_size=2,
            runtime_reserve_gib="0.01",
            tiff_cache_dir=kwargs.pop("tiff_cache_dir", self.root / "cache"),
            _memory_provider=self.memory,
            _filesystem_provider=self.filesystem,
            _trt_infer_factory=factory,
            **kwargs,
        )

    def test_process_uses_fixed_normalization_and_writes_edge_safe_mask(self):
        source, _ = self._write_source()
        output = self.root / "result.tif"

        result = self._process(source, out_mask=output)
        mask = tifffile.imread(output)

        expected = np.zeros((9, 10), dtype=np.uint8)
        expected[:4, :4] = 255
        np.testing.assert_array_equal(mask, expected)
        self.assertEqual(result["reader_backend"], "ram")
        self.assertGreater(result["reader_metrics"]["cache_hits"], 0)
        self.assertEqual(result["reader_metrics"]["blocks_decoded"], 3)
        first_patch = _RecordingTRT.instances[0].batches[0][0]
        self.assertEqual(first_patch.dtype, np.float32)
        self.assertEqual(float(first_patch.max()), 1.0)
        self.assertEqual(list((self.root / "cache").glob("*")), [])

    def test_internal_source_and_mask_caches_are_removed_on_inference_failure(self):
        source, _ = self._write_source()
        cache_dir = self.root / "failure-cache"

        with self.assertRaisesRegex(RuntimeError, "injected inference failure"):
            self._process(
                source,
                factory=_FailingTRT,
                tiff_cache_dir=cache_dir,
                max_ram_cache_gib="0.00000001",
                max_disk_cache_gib=1,
            )

        self.assertTrue(cache_dir.is_dir())
        self.assertEqual(list(cache_dir.glob("*")), [])

    def test_user_mask_cache_is_retained_on_failure_and_existing_path_is_rejected(self):
        source, _ = self._write_source()
        user_cache = self.root / "user-mask.dat"

        with self.assertRaisesRegex(RuntimeError, "injected inference failure"):
            self._process(source, factory=_FailingTRT, mask_cache=user_cache)
        self.assertTrue(user_cache.is_file())
        self.assertEqual(user_cache.stat().st_size, 9 * 10)

        with self.assertRaisesRegex(FileExistsError, "already exists"):
            self._process(source, mask_cache=user_cache)

    def test_production_contract_requires_engine_manifest_and_input_sidecar(self):
        source, _ = self._write_source()
        with self.assertRaisesRegex(ValueError, "engine_manifest"):
            self._process(source, production_contract=True)

    def test_atomic_output_preserves_existing_file_when_replace_fails(self):
        output = self.root / "existing.tif"
        output.write_bytes(b"existing-output")

        with mock.patch(
            "inference_large_image_trt.os.replace",
            side_effect=OSError("injected replace failure"),
        ):
            with self.assertRaisesRegex(OSError, "replace failure"):
                _atomic_write_tiff(output, np.zeros((4, 4), dtype=np.uint8))

        self.assertEqual(output.read_bytes(), b"existing-output")
        self.assertEqual(list(self.root.glob(f".{output.name}.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
