import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from create_tiff_sidecar import build_sidecar_payload, write_sidecar  # noqa: E402
from input_contract import load_input_sidecar  # noqa: E402


class SidecarCreationTests(unittest.TestCase):
    def test_creates_loadable_rgbnir_sidecar_with_source_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "rgbnir.tif"
            output = root / "rgbnir.json"
            tifffile.imwrite(
                source,
                np.zeros((8, 9, 4), dtype=np.uint16),
                photometric="rgb",
                extrasamples=["unspecified"],
                metadata={"axes": "YXC"},
            )

            payload = build_sidecar_payload(
                source,
                "red,green,blue,nir",
                "rgbnir-v1",
                "dtype-range-v1",
            )
            write_sidecar(output, payload)
            loaded = load_input_sidecar(output, source)

            self.assertEqual(loaded.axes, "YXC")
            self.assertEqual(loaded.shape, (8, 9, 4))
            self.assertEqual(loaded.band_order, ("red", "green", "blue", "nir"))

    def test_rejects_alpha_and_does_not_overwrite_existing_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "rgba.tif"
            tifffile.imwrite(
                source,
                np.zeros((8, 9, 4), dtype=np.uint8),
                photometric="rgb",
                extrasamples=["unassalpha"],
                metadata={"axes": "YXC"},
            )

            with self.assertRaisesRegex(ValueError, "alpha"):
                build_sidecar_payload(
                    source,
                    "red,green,blue,nir",
                    "rgbnir-v1",
                    "dtype-range-v1",
                )

            output = root / "existing.json"
            output.write_text("existing", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                write_sidecar(output, {"schema_version": 1})
            self.assertEqual(output.read_text(encoding="utf-8"), "existing")


if __name__ == "__main__":
    unittest.main()
