import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import tifffile
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from protocol.schemas import ConfigSnapshot, ProductRef, ROI, RequestKey  # noqa: E402
from sat_ai.inference import InferenceConfig, InsufficientValidData, infer_region  # noqa: E402
from sat_ai.manifest import load_model_manifest  # noqa: E402
from sat_ai.products import build_products  # noqa: E402
from sat_ai.roi import iter_patch_windows, open_memmap_scene  # noqa: E402
from sat_ai.threshold_lut import ThresholdLUT, coverage_accepted, coverage_ratio_bp  # noqa: E402


class SatAiMissionTests(unittest.TestCase):
    def test_manifest_binds_released_checkpoint_and_input_spec(self):
        manifest = load_model_manifest(ROOT / "sat_ai" / "model_manifest.yaml", ROOT / "checkpoints" / "best_model.pth")
        self.assertEqual(manifest.input_spec.channels, 3)
        self.assertEqual(manifest.input_spec.patch_size, 256)
        self.assertEqual(manifest.input_spec.integer_scale, 65535)
        self.assertEqual(manifest.assurance_level, "demo_non_validated")

    def test_scene_grid_remains_anchored_when_roi_shifts_one_pixel(self):
        first = [(window.x, window.y) for window in iter_patch_windows((512, 512, 3), ROI(0, 0, 256, 256), 256)]
        shifted = [(window.x, window.y) for window in iter_patch_windows((512, 512, 3), ROI(1, 1, 256, 256), 256)]
        self.assertEqual(first, [(0, 0)])
        self.assertEqual(shifted, [(0, 0), (256, 0), (0, 256), (256, 256)])

    def test_threshold_lut_and_strict_integer_coverage(self):
        lut = ThresholdLUT.from_file(ROOT / "protocol" / "golden_vectors" / "threshold_lut.bin")
        self.assertTrue(lut.classify(-100.0, 0))
        self.assertFalse(lut.classify(100.0, 10000))
        self.assertEqual(coverage_ratio_bp(6, 10), 6000)
        self.assertFalse(coverage_accepted(6, 10, 6000))
        self.assertTrue(coverage_accepted(5, 10, 6000))

    def test_strict_validity_rejects_nodata_in_context(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.tif"
            mask = root / "validity.tif"
            tifffile.imwrite(source, np.zeros((256, 256, 3), dtype=np.uint16), metadata={"axes": "YXC"}, compression=None)
            mask_array = np.ones((256, 256), dtype=np.uint8)
            mask_array[0, 0] = 0
            tifffile.imwrite(mask, mask_array, metadata={"axes": "YX"}, compression=None)
            sidecar = root / "source.json"
            sidecar.write_text(json.dumps({"schema_version": 1, "source_fingerprint": {"algorithm": "sha256", "digest": hashlib.sha256(source.read_bytes()).hexdigest()}, "axes": "YXC", "shape": [256, 256, 3], "band_order": ["red", "green", "blue"], "dtype": "uint16", "input_spec_id": "rgb-legacy-dtype-range-v1", "validity": {"kind": "mask", "relative_path": "validity.tif", "sha256": hashlib.sha256(mask.read_bytes()).hexdigest()}}), encoding="utf-8")
            scene = open_memmap_scene(source, sidecar)
            manifest = load_model_manifest(ROOT / "sat_ai" / "model_manifest.yaml")
            runtime = SimpleNamespace(manifest=manifest, infer_logits=lambda batch: np.zeros((len(batch),), dtype=np.float32))
            config = InferenceConfig(ConfigSnapshot(0, 0, 5000, 6000), ThresholdLUT.from_file(ROOT / "protocol" / "golden_vectors" / "threshold_lut.bin"))
            with self.assertRaises(InsufficientValidData):
                infer_region(scene, ROI(0, 0, 256, 256), runtime, config)
            scene.close()

    def test_all_valid_metrics_are_per_inference_and_do_not_fake_mask_io(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.tif"
            tifffile.imwrite(source, np.zeros((256, 256, 3), dtype=np.uint16), metadata={"axes": "YXC"}, compression=None)
            sidecar = root / "source.json"
            sidecar.write_text(json.dumps({"schema_version": 1, "source_fingerprint": {"algorithm": "sha256", "digest": hashlib.sha256(source.read_bytes()).hexdigest()}, "axes": "YXC", "shape": [256, 256, 3], "band_order": ["red", "green", "blue"], "dtype": "uint16", "input_spec_id": "rgb-legacy-dtype-range-v1", "validity": {"kind": "all_valid"}}), encoding="utf-8")
            manifest = load_model_manifest(ROOT / "sat_ai" / "model_manifest.yaml")
            runtime = SimpleNamespace(manifest=manifest, infer_logits=lambda batch: np.zeros((len(batch),), dtype=np.float32))
            config = InferenceConfig(ConfigSnapshot(0, 0, 5000, 6000), ThresholdLUT.from_file(ROOT / "protocol" / "golden_vectors" / "threshold_lut.bin"))
            with open_memmap_scene(source, sidecar) as scene:
                first = infer_region(scene, ROI(0, 0, 256, 256), runtime, config)
                second = infer_region(scene, ROI(0, 0, 256, 256), runtime, config)
            for result in (first, second):
                self.assertEqual(result["reader_metrics"]["logical_source_bytes_read"], 393216)
                self.assertEqual(result["reader_metrics"]["logical_validity_bytes_read"], 0)
                self.assertEqual(result["reader_metrics"]["logical_bytes_read"], 393216)

    def test_deployable_cpu_profile_is_bound_to_passing_v2_benchmark(self):
        profile = yaml.safe_load((ROOT / "sat_ai" / "deployment_profile.yaml").read_text(encoding="utf-8"))
        artifact_path = ROOT / "artifacts" / "benchmarks" / f"{profile['benchmark_artifact_id']}.json"
        artifact_bytes = artifact_path.read_bytes()
        artifact = json.loads(artifact_bytes)
        self.assertTrue(profile["deployable"])
        self.assertEqual(hashlib.sha256(artifact_bytes).hexdigest(), profile["benchmark_artifact_sha256"])
        self.assertEqual(artifact["target_id"], profile["target_id"])
        self.assertEqual(artifact["runtime"], profile["runtime"])
        self.assertTrue(artifact["guards"]["rss_pass"])
        self.assertTrue(artifact["guards"]["scene_scale_pass"])
        self.assertLessEqual(artifact["measurements"]["scene_scale_p95_ratio"], profile["scene_scale_p95_ratio"])

    def test_product_publish_failure_leaves_no_partial_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.tif"
            tifffile.imwrite(source, np.zeros((256, 256, 3), dtype=np.uint16), metadata={"axes": "YXC"}, compression=None)
            sidecar = root / "source.json"
            source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
            sidecar.write_text(json.dumps({"schema_version": 1, "source_fingerprint": {"algorithm": "sha256", "digest": source_sha}, "axes": "YXC", "shape": [256, 256, 3], "band_order": ["red", "green", "blue"], "dtype": "uint16", "input_spec_id": "rgb-legacy-dtype-range-v1", "validity": {"kind": "all_valid"}}), encoding="utf-8")
            manifest = load_model_manifest(ROOT / "sat_ai" / "model_manifest.yaml")
            runtime = SimpleNamespace(manifest=manifest, infer_logits=lambda batch: np.zeros((len(batch),), dtype=np.float32))
            config = InferenceConfig(ConfigSnapshot(0, 0, 5000, 6000), ThresholdLUT.from_file(ROOT / "protocol" / "golden_vectors" / "threshold_lut.bin"))
            product_root = root / "products"
            with open_memmap_scene(source, sidecar) as scene:
                result = infer_region(scene, ROI(0, 0, 256, 256), runtime, config)
                result["scene_ref"] = {"catalog_epoch": 1, "scene_id": 1, "scene_revision": 1}
                with mock.patch("sat_ai.products.os.replace", side_effect=OSError("injected publish failure")):
                    with self.assertRaisesRegex(OSError, "injected"):
                        build_products(result, scene, product_root, ProductRef(1, 1, 1), RequestKey(1, 1), source_sha256=source_sha)
            self.assertFalse((product_root / "00000001" / "00000001").exists())
            self.assertEqual(list(product_root.glob("**/.staging-*")), [])

    def test_infer_region_streams_bounded_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.tif"
            tifffile.imwrite(source, np.zeros((512, 512, 3), dtype=np.uint16), metadata={"axes": "YXC"}, compression=None)
            sidecar = root / "source.json"
            sidecar.write_text(json.dumps({"schema_version": 1, "source_fingerprint": {"algorithm": "sha256", "digest": hashlib.sha256(source.read_bytes()).hexdigest()}, "axes": "YXC", "shape": [512, 512, 3], "band_order": ["red", "green", "blue"], "dtype": "uint16", "input_spec_id": "rgb-legacy-dtype-range-v1", "validity": {"kind": "all_valid"}}), encoding="utf-8")
            manifest = load_model_manifest(ROOT / "sat_ai" / "model_manifest.yaml")
            batch_sizes = []

            def infer_logits(batch):
                batch_sizes.append(len(batch))
                return np.zeros((len(batch),), dtype=np.float32)

            runtime = SimpleNamespace(manifest=manifest, infer_logits=infer_logits)
            config = InferenceConfig(ConfigSnapshot(0, 0, 5000, 6000), ThresholdLUT.from_file(ROOT / "protocol" / "golden_vectors" / "threshold_lut.bin"))
            with open_memmap_scene(source, sidecar) as scene:
                result = infer_region(scene, ROI(0, 0, 512, 512), runtime, config, batch_size=2)
            self.assertEqual(result["patch_count"], 4)
            self.assertEqual(batch_sizes, [2, 2])
