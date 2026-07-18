import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from input_contract import legacy_input_spec  # noqa: E402


class _Logger:
    WARNING = 1

    def __init__(self, level):
        self.level = level


class _Engine:
    def __iter__(self):
        return iter(("input", "output"))

    @staticmethod
    def binding_is_input(binding):
        return binding == "input"

    @staticmethod
    def get_binding_shape(binding):
        return (1, 3, 4, 4) if binding == "input" else (1, 1)

    @staticmethod
    def get_binding_dtype(binding):
        return "float32"


class TensorRTInputContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fake_trt = types.ModuleType("tensorrt")
        fake_trt.Logger = _Logger
        fake_trt.nptype = lambda dtype: np.float32
        fake_cuda = types.ModuleType("pycuda.driver")
        fake_pycuda = types.ModuleType("pycuda")
        fake_autoinit = types.ModuleType("pycuda.autoinit")
        fake_pycuda.driver = fake_cuda
        fake_pycuda.autoinit = fake_autoinit
        cls.module_patch = mock.patch.dict(
            sys.modules,
            {
                "tensorrt": fake_trt,
                "pycuda": fake_pycuda,
                "pycuda.driver": fake_cuda,
                "pycuda.autoinit": fake_autoinit,
            },
        )
        cls.module_patch.start()
        sys.modules.pop("inference_tensorrt", None)
        cls.inference_tensorrt = importlib.import_module("inference_tensorrt")

    @classmethod
    def tearDownClass(cls):
        sys.modules.pop("inference_tensorrt", None)
        cls.module_patch.stop()

    def test_binding_is_the_source_of_actual_shape_channel_and_dtype(self):
        infer = self.inference_tensorrt.CloudTRTInfer.__new__(
            self.inference_tensorrt.CloudTRTInfer
        )
        infer.engine = _Engine()
        infer.input_spec = legacy_input_spec(3, 4)

        infer._resolve_input_contract(channels=3, patch_size=4)

        self.assertEqual(infer.input_shape, (1, 3, 4, 4))
        self.assertEqual(infer.channels, 3)
        self.assertEqual(infer.patch_size, 4)
        self.assertEqual(infer.input_dtype, np.dtype(np.float32))

    def test_channel_mismatch_fails_instead_of_padding_or_truncating(self):
        infer = self.inference_tensorrt.CloudTRTInfer.__new__(
            self.inference_tensorrt.CloudTRTInfer
        )
        infer.channels = 4
        infer.patch_size = 4
        infer.input_dtype = np.dtype(np.float32)

        with self.assertRaisesRegex(ValueError, "exactly 4"):
            infer._prepare_input(np.zeros((1, 3, 4, 4), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "exactly 4"):
            infer._prepare_input(np.zeros((1, 5, 4, 4), dtype=np.float32))

    def test_cli_channel_assertion_must_match_binding(self):
        infer = self.inference_tensorrt.CloudTRTInfer.__new__(
            self.inference_tensorrt.CloudTRTInfer
        )
        infer.engine = _Engine()
        infer.input_spec = legacy_input_spec(3, 4)

        with self.assertRaisesRegex(ValueError, "binding channels=3"):
            infer._resolve_input_contract(channels=4, patch_size=4)


if __name__ == "__main__":
    unittest.main()
