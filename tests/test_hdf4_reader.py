import io
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from data import read_hdf4  # noqa: E402
from inference_large_image_trt import _is_hdf4_file  # noqa: E402


class _FakeSDC:
    READ = 1


class _FakeDataset:
    def __init__(self, data, attributes):
        self.data = np.asarray(data)
        self._attributes = attributes
        self.closed = False

    def attributes(self):
        return self._attributes

    def __getitem__(self, key):
        return self.data[key]

    def get(self, start, count, stride):
        slices = tuple(
            slice(offset, offset + amount * step, step)
            for offset, amount, step in zip(start, count, stride)
        )
        return self.data[slices]

    def endaccess(self):
        self.closed = True


class _FakeSD:
    files = {}

    def __init__(self, path, mode):
        self.file = self.files[path]
        self.closed = False

    def datasets(self):
        return {
            name: (
                tuple(f"dim_{axis}" for axis in range(dataset.data.ndim)),
                dataset.data.shape,
                str(dataset.data.dtype),
                index,
            )
            for index, (name, dataset) in enumerate(self.file["datasets"].items())
        }

    def select(self, name):
        return self.file["datasets"][name]

    def attributes(self):
        return self.file["attributes"]

    def end(self):
        self.closed = True


class HDF4ReaderTests(unittest.TestCase):
    def setUp(self):
        self.path = ROOT / "fake_scene.hdf"
        _FakeSD.files[str(self.path)] = {
            "attributes": {"title": b"test scene", "orbit": np.int32(42)},
            "datasets": {
                "Reflectance": _FakeDataset(
                    np.arange(3 * 4 * 2, dtype=np.int16).reshape(3, 4, 2),
                    {"scale_factor": np.float32(0.01)},
                ),
                "Quality": _FakeDataset(np.ones((3, 4), dtype=np.uint8), {}),
            },
        }
        self.path_validation = mock.patch.object(
            read_hdf4,
            "_validate_path",
            side_effect=lambda path: Path(path),
        )
        self.pyhdf = mock.patch.object(
            read_hdf4,
            "_load_pyhdf",
            return_value=(_FakeSD, _FakeSDC),
        )
        self.path_validation.start()
        self.pyhdf.start()

    def tearDown(self):
        self.pyhdf.stop()
        self.path_validation.stop()
        _FakeSD.files.clear()

    def test_lists_metadata_without_loading_the_datasets(self):
        metadata = read_hdf4.read_hdf4_metadata(self.path)

        self.assertEqual(metadata["attributes"], {"title": "test scene", "orbit": 42})
        self.assertEqual(metadata["datasets"][0]["name"], "Reflectance")
        self.assertEqual(metadata["datasets"][0]["shape"], (3, 4, 2))
        self.assertEqual(metadata["datasets"][0]["dimensions"], ("dim_0", "dim_1", "dim_2"))
        self.assertAlmostEqual(
            metadata["datasets"][0]["attributes"]["scale_factor"],
            0.01,
        )

    def test_requires_a_dataset_name_when_file_has_multiple_datasets(self):
        with self.assertRaisesRegex(ValueError, "multiple datasets"):
            read_hdf4.read_hdf4_dataset(self.path)

    def test_reads_a_hyperslab(self):
        actual = read_hdf4.read_hdf4_dataset(
            self.path,
            dataset_name="Reflectance",
            start=[1, 1, 0],
            count=[2, 2, 1],
            stride=[1, 1, 1],
        )
        expected = np.arange(3 * 4 * 2, dtype=np.int16).reshape(3, 4, 2)[1:3, 1:3, :1]

        np.testing.assert_array_equal(actual, expected)

    def test_detects_hdf4_signature_for_ambiguous_hdf_extension(self):
        file_handle = mock.MagicMock()
        file_handle.__enter__.return_value.read.return_value = b"\x0e\x03\x13\x01"
        with mock.patch.object(Path, "open", return_value=file_handle):
            self.assertTrue(_is_hdf4_file(self.path))

        file_handle.__enter__.return_value.read.return_value = b"\x89HDF"
        with mock.patch.object(Path, "open", return_value=file_handle):
            self.assertFalse(_is_hdf4_file(self.path))

    def test_writes_channel_first_tiff_without_changing_values(self):
        import tifffile

        expected = np.arange(2 * 8 * 8, dtype=np.uint16).reshape(2, 8, 8)
        output = io.BytesIO()

        read_hdf4.write_tiff(output, expected, compression="deflate")
        output.seek(0)
        actual = tifffile.imread(output)

        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
