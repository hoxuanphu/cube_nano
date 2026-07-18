import hashlib
import json
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

from input_contract import legacy_input_spec  # noqa: E402
from resource_guards import (  # noqa: E402
    FilesystemStats,
    FixedMemoryInfoProvider,
    GIB,
    ReaderBudget,
)
from tiff_reader import TiffReader  # noqa: E402


class _FilesystemProvider:
    def __init__(self, free_bytes=100 * GIB):
        self.free_bytes = free_bytes

    def stats_for(self, path):
        return FilesystemStats("test-device", 100 * GIB, self.free_bytes)


class TiffReaderTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.memory = FixedMemoryInfoProvider(8 * GIB)
        self.filesystem = _FilesystemProvider()
        self.budget = ReaderBudget.from_cli(
            max_ram_cache_gib=1,
            max_disk_cache_gib=2,
            runtime_reserve_gib="0.01",
            tiff_block_cache_mib=0,
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _reader(self, path, channels=3, patch_size=4, **kwargs):
        return TiffReader(
            path,
            legacy_input_spec(channels, patch_size),
            budget=kwargs.pop("budget", self.budget),
            cache_dir=kwargs.pop("cache_dir", self.root / "cache"),
            patch_size=patch_size,
            memory_provider=kwargs.pop("memory_provider", self.memory),
            filesystem_provider=kwargs.pop("filesystem_provider", self.filesystem),
            **kwargs,
        )

    def test_memmap_is_opened_once_and_rows_match_oracle(self):
        path = self.root / "rgb_uncompressed.tif"
        expected = np.arange(11 * 13 * 3, dtype=np.uint16).reshape(11, 13, 3)
        tifffile.imwrite(
            path,
            expected,
            photometric="rgb",
            rowsperstrip=5,
            metadata={"axes": "YXC"},
        )

        with mock.patch.object(tifffile, "memmap", wraps=tifffile.memmap) as memmap:
            with self._reader(path) as reader:
                actual = np.concatenate(
                    [reader.read_rows(0, 4), reader.read_rows(4, 8), reader.read_rows(8, 11)]
                )
                self.assertEqual(reader.backend, "memmap")

        self.assertEqual(memmap.call_count, 1)
        np.testing.assert_array_equal(actual, expected)

    def test_compressed_tiff_decodes_once_then_slices_ram_cache(self):
        path = self.root / "rgb_deflate.tif"
        expected = np.arange(12 * 13 * 3, dtype=np.uint16).reshape(12, 13, 3)
        tifffile.imwrite(
            path,
            expected,
            photometric="rgb",
            compression="deflate",
            rowsperstrip=5,
            metadata={"axes": "YXC"},
        )
        original_asarray = tifffile.TiffFile.asarray

        def call_original(instance, *args, **kwargs):
            return original_asarray(instance, *args, **kwargs)

        with mock.patch.object(
            tifffile.TiffFile,
            "asarray",
            autospec=True,
            side_effect=call_original,
        ) as asarray:
            with self._reader(path) as reader:
                actual = np.concatenate(
                    [reader.read_rows(0, 4), reader.read_rows(4, 8), reader.read_rows(8, 12)]
                )
                metrics = reader.metrics.as_dict()
                decoded_blocks = len(reader._page.dataoffsets)
                self.assertEqual(reader.backend, "ram")

        self.assertEqual(asarray.call_count, 1)
        self.assertEqual(metrics["blocks_decoded"], decoded_blocks)
        self.assertGreater(metrics["cache_hits"], 0)
        np.testing.assert_array_equal(actual, expected)

    def test_stream_rejects_compressed_tiff_before_decode(self):
        path = self.root / "stream_deflate.tif"
        tifffile.imwrite(
            path,
            np.zeros((8, 9, 3), dtype=np.uint8),
            photometric="rgb",
            compression="deflate",
            metadata={"axes": "YXC"},
        )

        with mock.patch.object(tifffile.TiffFile, "asarray", autospec=True) as asarray:
            with self.assertRaisesRegex(RuntimeError, "no true block backend"):
                self._reader(path, read_mode="stream")
        asarray.assert_not_called()

    def test_full_mode_skips_memmap_even_for_uncompressed_tiff(self):
        path = self.root / "full_uncompressed.tif"
        expected = np.arange(8 * 9 * 3, dtype=np.uint16).reshape(8, 9, 3)
        tifffile.imwrite(path, expected, photometric="rgb", metadata={"axes": "YXC"})

        with mock.patch.object(tifffile, "memmap", wraps=tifffile.memmap) as memmap:
            with self._reader(path, read_mode="full", cache_mode="ram") as reader:
                self.assertEqual(reader.backend, "ram")
                np.testing.assert_array_equal(reader.read_rows(0, 8), expected)
        memmap.assert_not_called()

    def test_memory_guard_fails_before_decoder_is_called(self):
        path = self.root / "memory_guard.tif"
        tifffile.imwrite(
            path,
            np.zeros((20, 20, 3), dtype=np.uint16),
            photometric="rgb",
            compression="deflate",
            metadata={"axes": "YXC"},
        )

        with mock.patch.object(tifffile.TiffFile, "asarray", autospec=True) as asarray:
            with self.assertRaises(MemoryError):
                self._reader(
                    path,
                    memory_provider=FixedMemoryInfoProvider(1),
                )
        asarray.assert_not_called()

    def test_missing_codec_fails_before_decoder_is_called(self):
        path = self.root / "codec_guard.tif"
        tifffile.imwrite(
            path,
            np.zeros((8, 9, 3), dtype=np.uint8),
            photometric="rgb",
            compression="deflate",
            metadata={"axes": "YXC"},
        )
        codec_type = type(tifffile.TIFF.DECOMPRESSORS)

        with mock.patch.object(
            codec_type,
            "__getitem__",
            side_effect=KeyError("codec unavailable"),
        ), mock.patch.object(tifffile.TiffFile, "asarray", autospec=True) as asarray:
            with self.assertRaisesRegex(RuntimeError, "codec DEFLATE is unavailable"):
                self._reader(path)
        asarray.assert_not_called()

    def test_disk_cache_is_removed_after_close(self):
        path = self.root / "disk_deflate.tif"
        expected = np.arange(20 * 20 * 3, dtype=np.uint16).reshape(20, 20, 3)
        tifffile.imwrite(
            path,
            expected,
            photometric="rgb",
            compression="deflate",
            metadata={"axes": "YXC"},
        )
        disk_budget = ReaderBudget.from_cli(
            max_ram_cache_gib="0.00000001",
            max_disk_cache_gib=1,
            runtime_reserve_gib="0.01",
            tiff_block_cache_mib=0,
        )

        reader = self._reader(path, budget=disk_budget)
        source_cache = reader.source_cache_path
        self.assertEqual(reader.backend, "disk")
        self.assertTrue(source_cache.is_file())
        np.testing.assert_array_equal(reader.read_rows(4, 8), expected[4:8])
        reader.close()

        self.assertFalse(source_cache.exists())

    def test_disk_guard_fails_before_cache_file_creation(self):
        path = self.root / "disk_guard.tif"
        tifffile.imwrite(
            path,
            np.zeros((20, 20, 3), dtype=np.uint16),
            photometric="rgb",
            compression="deflate",
            metadata={"axes": "YXC"},
        )

        with self.assertRaisesRegex(OSError, "Insufficient disk space"):
            self._reader(
                path,
                cache_mode="disk",
                filesystem_provider=_FilesystemProvider(free_bytes=1),
                disk_allocations=(),
            )
        self.assertFalse((self.root / "cache").exists())

    def test_channel_first_planar_separate_is_returned_as_hwc(self):
        path = self.root / "rgb_planar.tif"
        physical = np.arange(3 * 9 * 10, dtype=np.uint16).reshape(3, 9, 10)
        tifffile.imwrite(
            path,
            physical,
            photometric="rgb",
            planarconfig="separate",
            compression="deflate",
            metadata={"axes": "CYX"},
        )

        with self._reader(path) as reader:
            actual = reader.read_rows(2, 7)

        np.testing.assert_array_equal(actual, np.moveaxis(physical[:, 2:7, :], 0, -1))
        self.assertEqual(actual.shape, (5, 10, 3))

    def test_explicit_mapping_reorders_bgr_to_rgb(self):
        path = self.root / "bgr.tif"
        physical = np.zeros((6, 7, 3), dtype=np.uint8)
        physical[:, :, 0] = 10
        physical[:, :, 1] = 20
        physical[:, :, 2] = 30
        tifffile.imwrite(
            path,
            physical,
            photometric="minisblack",
            planarconfig="contig",
            compression="deflate",
            metadata={"axes": "YXC"},
        )

        with self._reader(
            path,
            channel_mapping="red=2,green=1,blue=0",
        ) as reader:
            actual = reader.read_rows(0, 6)

        np.testing.assert_array_equal(actual, physical[:, :, [2, 1, 0]])
        self.assertEqual(reader.band_order, ("red", "green", "blue"))

    def test_rgbnir_requires_mapping_and_rgba_is_always_rejected(self):
        rgbnir_path = self.root / "rgbnir.tif"
        rgbnir = np.arange(6 * 7 * 4, dtype=np.uint16).reshape(6, 7, 4)
        tifffile.imwrite(
            rgbnir_path,
            rgbnir,
            photometric="rgb",
            extrasamples=["unspecified"],
            compression="deflate",
            metadata={"axes": "YXC"},
        )

        with self.assertRaisesRegex(ValueError, "Ambiguous TIFF channel semantics"):
            self._reader(rgbnir_path, channels=4)
        with self._reader(
            rgbnir_path,
            channels=4,
            channel_mapping="red=0,green=1,blue=2,nir=3",
        ) as reader:
            np.testing.assert_array_equal(reader.read_rows(0, 6), rgbnir)

        rgba_path = self.root / "rgba.tif"
        tifffile.imwrite(
            rgba_path,
            np.zeros((6, 7, 4), dtype=np.uint8),
            photometric="rgb",
            extrasamples=["unassalpha"],
            metadata={"axes": "YXC"},
        )
        with self.assertRaisesRegex(ValueError, "alpha"):
            self._reader(
                rgba_path,
                channels=4,
                channel_mapping="red=0,green=1,blue=2,nir=3",
            )

    def test_sidecar_metadata_is_checked_and_must_agree_with_cli_mapping(self):
        path = self.root / "sidecar_rgbnir.tif"
        image = np.arange(6 * 7 * 4, dtype=np.uint16).reshape(6, 7, 4)
        tifffile.imwrite(
            path,
            image,
            photometric="rgb",
            extrasamples=["unspecified"],
            compression="deflate",
            metadata={"axes": "YXC"},
        )
        sidecar = self.root / "sidecar_rgbnir.json"
        sidecar.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "source_fingerprint": {
                        "algorithm": "sha256",
                        "digest": hashlib.sha256(path.read_bytes()).hexdigest(),
                    },
                    "axes": "YXC",
                    "shape": [6, 7, 4],
                    "band_order": ["red", "green", "blue", "nir"],
                    "dtype": "uint16",
                    "input_spec_id": "legacy-dtype-range-v1",
                    "normalization": "legacy-dtype-range-v1",
                }
            ),
            encoding="utf-8",
        )

        with self._reader(path, channels=4, input_sidecar=sidecar) as reader:
            np.testing.assert_array_equal(reader.read_rows(0, 6), image)
        with self.assertRaisesRegex(ValueError, "conflicts with input sidecar"):
            self._reader(
                path,
                channels=4,
                input_sidecar=sidecar,
                channel_mapping="red=1,green=0,blue=2,nir=3",
            )

    def test_multiple_series_require_explicit_selection(self):
        path = self.root / "multi_series.tif"
        tifffile.imwrite(path, np.zeros((6, 7, 3), dtype=np.uint8), photometric="rgb")
        tifffile.imwrite(
            path,
            np.zeros((8, 9, 3), dtype=np.uint8),
            photometric="rgb",
            append=True,
        )

        with self.assertRaisesRegex(ValueError, "select one explicitly"):
            self._reader(path)
        with self._reader(path, series_index=1) as reader:
            self.assertEqual(reader.shape, (8, 9, 3))


if __name__ == "__main__":
    unittest.main()
