"""Export the fixed-batch SegFormer MVP graph and its artifact contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

try:
    from models.segformer_b0 import SEGFORMER_IMPLEMENTATION_ID, get_segformer_b0
except ModuleNotFoundError:  # Package invocation: python -m src.export_segformer_onnx
    from src.models.segformer_b0 import SEGFORMER_IMPLEMENTATION_ID, get_segformer_b0


SEGFORMER_ONNX_OPSET = 17
INPUT_SHAPE = (1, 3, 256, 256)
OUTPUT_SHAPE = (1, 2, 64, 64)


def _load_state_dict(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        checkpoint = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must contain a state dictionary")
    if checkpoint and all(str(key).startswith("module.") for key in checkpoint):
        checkpoint = {str(key).removeprefix("module."): value for key, value in checkpoint.items()}
    return checkpoint


def export_segformer_onnx(
    output_path: str | Path,
    *,
    checkpoint_path: str | Path | None = None,
    allow_untrained: bool = False,
    opset_version: int = SEGFORMER_ONNX_OPSET,
) -> dict:
    if opset_version != SEGFORMER_ONNX_OPSET:
        raise ValueError(f"SegFormer MVP requires pinned ONNX opset {SEGFORMER_ONNX_OPSET}")
    model = get_segformer_b0().eval()
    checkpoint_digest = None
    if checkpoint_path is not None and Path(checkpoint_path).is_file():
        checkpoint_file = Path(checkpoint_path)
        model.load_state_dict(_load_state_dict(checkpoint_file))
        checkpoint_digest = hashlib.sha256(checkpoint_file.read_bytes()).hexdigest()
    elif not allow_untrained:
        raise FileNotFoundError("trained SegFormer checkpoint is required for export")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(INPUT_SHAPE, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        output,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=None,
    )
    contract = {
        "schema_version": 1,
        "artifact_type": "segformer-onnx",
        "implementation_id": SEGFORMER_IMPLEMENTATION_ID,
        "opset_version": opset_version,
        "input_name": "input",
        "input_shape": list(INPUT_SHAPE),
        "input_dtype": "float32",
        "output_name": "logits",
        "output_shape": list(OUTPUT_SHAPE),
        "output_dtype": "float32",
        "checkpoint_sha256": checkpoint_digest,
        "onnx_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(contract, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return contract


def main() -> None:
    parser = argparse.ArgumentParser(description="Export SegFormer-B0 to fixed-shape ONNX")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default="checkpoints/segformer_b0_rgb_r1.onnx")
    parser.add_argument("--allow-untrained", action="store_true")
    args = parser.parse_args()
    print(json.dumps(export_segformer_onnx(
        args.output,
        checkpoint_path=args.checkpoint,
        allow_untrained=args.allow_untrained,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
