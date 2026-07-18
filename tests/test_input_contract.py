import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from input_contract import (  # noqa: E402
    NormalizationSpec,
    load_engine_manifest,
    load_input_sidecar,
    parse_channel_mapping,
)


class InputContractTests(unittest.TestCase):
    def test_channel_mapping_rejects_duplicate_missing_and_invalid_indexes(self):
        self.assertEqual(
            parse_channel_mapping("red=2,green=1,blue=0"),
            {"red": 2, "green": 1, "blue": 0},
        )
        for value in (
            "red=0,green=0,blue=2",
            "red=-1,green=1,blue=2",
            "red=zero,green=1,blue=2",
            "thermal=0,green=1,blue=2",
        ):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_channel_mapping(value)

    def test_fixed_dtype_normalization_does_not_depend_on_patch_maximum(self):
        normalization = NormalizationSpec.from_value("dtype-range-v1", channels=3)
        low_patch = np.array([[[0, 1, 2]]], dtype=np.uint16)
        high_patch = np.array([[[65535, 32768, 0]]], dtype=np.uint16)

        low = normalization.apply(low_patch)
        high = normalization.apply(high_patch)

        np.testing.assert_allclose(low, low_patch.astype(np.float32) / 65535.0)
        np.testing.assert_allclose(high, high_patch.astype(np.float32) / 65535.0)

    def test_float_dtype_range_requires_values_already_in_zero_one_range(self):
        normalization = NormalizationSpec.from_value("dtype-range-v1", channels=3)
        with self.assertRaisesRegex(ValueError, r"already in \[0, 1\]"):
            normalization.apply(np.full((2, 2, 3), 255.0, dtype=np.float32))

    def test_sidecar_validates_source_fingerprint_and_required_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.tif"
            source.write_bytes(b"source-bytes")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            sidecar = root / "source.json"
            sidecar.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "source_fingerprint": {"algorithm": "sha256", "digest": digest},
                        "axes": "YXC",
                        "shape": [10, 11, 4],
                        "band_order": ["red", "green", "blue", "nir"],
                        "dtype": "uint16",
                        "input_spec_id": "rgbnir-v1",
                        "normalization": "dtype-range-v1",
                    }
                ),
                encoding="utf-8",
            )

            loaded = load_input_sidecar(sidecar, source)
            self.assertEqual(loaded.role_to_index["nir"], 3)

            source.write_bytes(b"different-source")
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                load_input_sidecar(sidecar, source)

    def test_engine_manifest_validates_engine_fingerprint_shape_and_dtype(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = root / "model.engine"
            engine.write_bytes(b"engine-bytes")
            digest = hashlib.sha256(engine.read_bytes()).hexdigest()
            manifest = root / "model.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "engine_fingerprint": {"algorithm": "sha256", "digest": digest},
                        "input_shape": [1, 3, 4, 4],
                        "input_dtype": "float32",
                        "optimization_profile": {
                            "min": [1, 3, 4, 4],
                            "opt": [1, 3, 4, 4],
                            "max": [1, 3, 4, 4],
                        },
                        "input_spec": {
                            "input_spec_id": "rgb-v1",
                            "band_order": ["red", "green", "blue"],
                            "normalization": "dtype-range-v1",
                        },
                    }
                ),
                encoding="utf-8",
            )

            loaded = load_engine_manifest(manifest, engine_path=engine)
            self.assertEqual(loaded.channels, 3)
            self.assertEqual(loaded.patch_size, 4)
            self.assertEqual(loaded.input_dtype, np.dtype(np.float32))

            engine.write_bytes(b"different-engine")
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                load_engine_manifest(manifest, engine_path=engine)


if __name__ == "__main__":
    unittest.main()
