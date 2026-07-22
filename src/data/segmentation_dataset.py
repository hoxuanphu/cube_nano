"""Paired 95-Cloud segmentation dataset with shared runtime preprocessing."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import Dataset

from sat_ai.manifest import InputSpec


def default_segmentation_input_spec() -> InputSpec:
    return InputSpec.from_mapping(
        {
            "input_spec_id": "segformer-rgb-dtype-range-v1",
            "channels": 3,
            "band_order": ["red", "green", "blue"],
            "patch_size": 256,
            "source_dtype": "uint16",
            "tensor_dtype": "float32",
            "tensor_layout": "NCHW",
            "input_shape": [None, 3, 256, 256],
            "normalization": {
                "id": "segformer-rgb-dtype-range-v1",
                "kind": "dtype-range",
                "integer_scale": 65535,
            },
            "padding": {
                "id": "scene-edge-constant-raw-v1",
                "kind": "constant",
                "value_space": "source",
                "values": [0, 0, 0],
            },
        }
    )


@dataclass(frozen=True)
class SegmentationRecord:
    image_path: Path
    mask_path: Path
    validity_path: Path | None
    scene_id: str


class SegmentationDataset(Dataset):
    """Return synchronized native-size samples or deterministic padded tiles."""

    def __init__(
        self,
        data_dir: str | Path,
        *,
        input_spec: InputSpec | None = None,
        is_train: bool = True,
        tile_size: int = 256,
        stride: int = 256,
        preserve_native_size: bool = False,
        ignore_index: int = 255,
        transform: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]] | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.input_spec = input_spec or default_segmentation_input_spec()
        self.is_train = bool(is_train)
        self.tile_size = int(tile_size)
        self.stride = int(stride)
        self.preserve_native_size = bool(preserve_native_size)
        self.ignore_index = int(ignore_index)
        self.transform = transform
        if self.tile_size != 256 or self.stride <= 0:
            raise ValueError("SegFormer MVP tile_size must be 256 and stride must be positive")
        if self.ignore_index != 255:
            raise ValueError("SegFormer training ignore_index is pinned to 255")

        image_paths = [
            path
            for label in ("cloud", "clear")
            for path in sorted((self.data_dir / label).glob("*.npy"))
        ]
        if not image_paths:
            raise ValueError(f"no processed RGB patches found under {self.data_dir}")
        mask_dir = self.data_dir / "masks"
        if not mask_dir.is_dir():
            raise FileNotFoundError(f"segmentation mask directory not found: {mask_dir}")
        validity_dir = self.data_dir / "validity"
        self.records: list[SegmentationRecord] = []
        seen_names: set[str] = set()
        for image_path in image_paths:
            if image_path.name in seen_names:
                raise ValueError(f"duplicate processed image filename: {image_path.name}")
            seen_names.add(image_path.name)
            mask_path = mask_dir / image_path.name
            if not mask_path.is_file():
                raise ValueError(f"missing segmentation mask for {image_path.name}")
            validity_path = validity_dir / image_path.name if validity_dir.is_dir() else None
            if validity_path is not None and not validity_path.is_file():
                raise ValueError(f"missing validity mask for {image_path.name}")
            self.records.append(
                SegmentationRecord(
                    image_path,
                    mask_path,
                    validity_path,
                    self._scene_id(image_path.stem),
                )
            )
        self._indices = self._build_indices()

    @staticmethod
    def _scene_id(stem: str) -> str:
        match = re.match(r"^(.*)_p\d+$", stem)
        return match.group(1) if match else stem

    def _build_indices(self) -> list[tuple[int, int | None, int | None]]:
        if self.preserve_native_size:
            return [(index, None, None) for index in range(len(self.records))]
        if self.is_train:
            return [(index, None, None) for index in range(len(self.records))]
        indices: list[tuple[int, int | None, int | None]] = []
        for index, record in enumerate(self.records):
            image = np.load(record.image_path, allow_pickle=False, mmap_mode="r")
            if image.ndim != 3 or image.shape[2] != self.input_spec.channels:
                raise ValueError(f"expected HWC RGB source, got {image.shape} from {record.image_path}")
            for row in range(0, image.shape[0], self.stride):
                for col in range(0, image.shape[1], self.stride):
                    indices.append((index, row, col))
        if not indices:
            raise ValueError("deterministic segmentation tile index is empty")
        return indices

    @staticmethod
    def _validate_mask(mask: np.ndarray, path: Path, *, allow_ignore: bool = False) -> None:
        if mask.ndim != 2:
            raise ValueError(f"expected HxW mask, got {mask.shape} from {path}")
        allowed = (0, 1, 255) if allow_ignore else (0, 1)
        if not np.isin(np.unique(mask), allowed).all():
            raise ValueError(f"mask contains values outside {allowed}: {path}")

    def _read_record(self, record: SegmentationRecord) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        image = np.load(record.image_path, allow_pickle=False)
        mask = np.load(record.mask_path, allow_pickle=False)
        validity = (
            np.load(record.validity_path, allow_pickle=False)
            if record.validity_path is not None
            else np.ones(mask.shape, dtype=np.uint8)
        )
        if image.ndim != 3 or image.shape[2] != self.input_spec.channels:
            raise ValueError(f"expected HWC RGB source, got {image.shape} from {record.image_path}")
        self._validate_mask(mask, record.mask_path, allow_ignore=True)
        self._validate_mask(validity, record.validity_path or record.mask_path)
        if image.shape[:2] != mask.shape or mask.shape != validity.shape:
            raise ValueError(
                f"image/mask/validity shape mismatch for {record.image_path.name}: "
                f"{image.shape[:2]}, {mask.shape}, {validity.shape}"
            )
        return image, mask, validity

    def _window(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        validity: np.ndarray,
        row: int,
        col: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int, int, int]]:
        height, width = image.shape[:2]
        row_end = min(row + self.tile_size, height)
        col_end = min(col + self.tile_size, width)
        image_tile = np.zeros((self.tile_size, self.tile_size, self.input_spec.channels), dtype=image.dtype)
        mask_tile = np.full((self.tile_size, self.tile_size), self.ignore_index, dtype=np.uint8)
        validity_tile = np.zeros((self.tile_size, self.tile_size), dtype=np.uint8)
        image_tile[: row_end - row, : col_end - col] = image[row:row_end, col:col_end]
        mask_tile[: row_end - row, : col_end - col] = mask[row:row_end, col:col_end]
        validity_tile[: row_end - row, : col_end - col] = validity[row:row_end, col:col_end]
        return image_tile, mask_tile, validity_tile, (col, row, col_end, row_end)

    def _training_window(self, image: np.ndarray, mask: np.ndarray, validity: np.ndarray):
        height, width = image.shape[:2]
        row = int(np.random.randint(max(height - self.tile_size, 0) + 1))
        col = int(np.random.randint(max(width - self.tile_size, 0) + 1))
        return self._window(image, mask, validity, row, col)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record_index, row, col = self._indices[index]
        record = self.records[record_index]
        image, mask, validity = self._read_record(record)
        if self.preserve_native_size:
            coordinates = (0, 0, image.shape[1], image.shape[0])
        elif self.is_train:
            image, mask, validity, coordinates = self._training_window(image, mask, validity)
        else:
            assert row is not None and col is not None
            image, mask, validity, coordinates = self._window(image, mask, validity, row, col)

        image_tensor = torch.from_numpy(
            np.ascontiguousarray(self.input_spec.normalize(image))
        ).permute(2, 0, 1)
        valid_tensor = torch.from_numpy(np.ascontiguousarray(validity)).to(torch.bool)
        target = torch.from_numpy(np.asarray(mask, dtype=np.int64))
        target = torch.where(valid_tensor, target, torch.full_like(target, self.ignore_index))
        if torch.any(valid_tensor):
            valid_values = target[valid_tensor]
            if not torch.all((valid_values == 0) | (valid_values == 1)):
                raise ValueError(f"valid segmentation target contains values other than 0/1: {record.mask_path}")

        tensors = {"image": image_tensor, "mask": target, "validity_mask": valid_tensor}
        if self.is_train:
            if torch.rand(()) > 0.5:
                tensors["image"] = torch.flip(tensors["image"], dims=[1])
                tensors["mask"] = torch.flip(tensors["mask"], dims=[0])
                tensors["validity_mask"] = torch.flip(tensors["validity_mask"], dims=[0])
            if torch.rand(()) > 0.5:
                tensors["image"] = torch.flip(tensors["image"], dims=[2])
                tensors["mask"] = torch.flip(tensors["mask"], dims=[1])
                tensors["validity_mask"] = torch.flip(tensors["validity_mask"], dims=[1])
            turns = int(torch.randint(0, 4, (1,)).item())
            if turns:
                tensors["image"] = torch.rot90(tensors["image"], turns, dims=[1, 2])
                tensors["mask"] = torch.rot90(tensors["mask"], turns, dims=[0, 1])
                tensors["validity_mask"] = torch.rot90(tensors["validity_mask"], turns, dims=[0, 1])
        if self.transform is not None:
            tensors = self.transform(tensors)
        return {
            **tensors,
            "scene_id": record.scene_id,
            "tile_coordinates": coordinates,
        }


CloudSegmentationDataset = SegmentationDataset
