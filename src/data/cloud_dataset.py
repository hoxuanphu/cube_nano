import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class CloudDataset(Dataset):
    def __init__(
        self,
        data_dir,
        transform=None,
        is_train=True,
        target_channels=4,
        channel_dropout_p=0.0,
        crop_size=256,
        cloud_ratio_threshold=0.10,
    ):
        """Load paired 95-Cloud image/mask patches.

        Training uses a random crop; validation and test use a center crop. The
        binary label is derived from the paired crop mask instead of the source
        patch's cloud/clear directory.
        """
        self.data_dir = os.fspath(data_dir)
        self.transform = transform
        self.is_train = is_train
        self.target_channels = target_channels
        self.channel_dropout_p = (
            channel_dropout_p if is_train and target_channels == 4 else 0.0
        )
        self.crop_size = crop_size
        self.cloud_ratio_threshold = cloud_ratio_threshold

        if self.target_channels not in (3, 4):
            raise ValueError(f"target_channels must be 3 or 4, got {self.target_channels}")
        if not 0.0 <= self.channel_dropout_p <= 1.0:
            raise ValueError(
                f"channel_dropout_p must be between 0 and 1, got {self.channel_dropout_p}"
            )
        if self.crop_size <= 0:
            raise ValueError(f"crop_size must be greater than zero, got {self.crop_size}")
        if not 0.0 <= self.cloud_ratio_threshold <= 1.0:
            raise ValueError(
                "cloud_ratio_threshold must be between 0 and 1, "
                f"got {self.cloud_ratio_threshold}"
            )

        self.cloud_files = sorted(
            glob.glob(os.path.join(self.data_dir, "cloud", "*.npy"))
        )
        self.clear_files = sorted(
            glob.glob(os.path.join(self.data_dir, "clear", "*.npy"))
        )
        self.files = self.cloud_files + self.clear_files
        if not self.files:
            raise ValueError(
                f"No .npy patches found in {self.data_dir}. "
                "Expected cloud/*.npy and clear/*.npy subfolders."
            )

        image_names = [os.path.basename(path) for path in self.files]
        duplicate_names = self._find_duplicates(image_names)
        if duplicate_names:
            raise ValueError(
                f"Duplicate patch filenames across cloud/clear: {duplicate_names[:5]}"
            )

        mask_dir = os.path.join(self.data_dir, "masks")
        if not os.path.isdir(mask_dir):
            raise FileNotFoundError(f"Expected mask directory not found: {mask_dir}")
        mask_paths = {
            os.path.basename(path): path
            for path in sorted(glob.glob(os.path.join(mask_dir, "*.npy")))
        }
        image_name_set = set(image_names)
        mask_name_set = set(mask_paths)
        missing_masks = sorted(image_name_set - mask_name_set)
        orphan_masks = sorted(mask_name_set - image_name_set)
        if missing_masks or orphan_masks:
            raise ValueError(
                "Dataset image/mask pairing failed: "
                f"missing_masks={missing_masks[:5]}, orphan_masks={orphan_masks[:5]}"
            )

        self.records = [
            (image_path, mask_paths[os.path.basename(image_path)])
            for image_path in self.files
        ]

        print(
            f"Loaded {len(self.records)} paired patches from {self.data_dir} "
            f"(source_cloud={len(self.cloud_files)}, source_clear={len(self.clear_files)}, "
            f"channels={self.target_channels}, cloud_ratio_threshold={self.cloud_ratio_threshold:.2f})"
        )

    @staticmethod
    def _find_duplicates(values):
        seen = set()
        duplicates = set()
        for value in values:
            if value in seen:
                duplicates.add(value)
            seen.add(value)
        return sorted(duplicates)

    @staticmethod
    def _validate_mask(mask, mask_path):
        if mask.ndim != 2:
            raise ValueError(f"Expected mask shape (H, W), got {mask.shape} from {mask_path}")
        if not (
            np.issubdtype(mask.dtype, np.bool_)
            or np.issubdtype(mask.dtype, np.integer)
            or np.issubdtype(mask.dtype, np.floating)
        ):
            raise ValueError(f"Unsupported mask dtype {mask.dtype} from {mask_path}")

    def _validate_spatial_size(self, height, width, file_path):
        if height < self.crop_size or width < self.crop_size:
            raise ValueError(
                f"crop_size={self.crop_size} exceeds patch shape ({height}, {width}) "
                f"from {file_path}"
            )

    def _normalize(self, img, file_path):
        if np.issubdtype(img.dtype, np.floating):
            img = img.astype(np.float32)
            if img.max() > 1.0:
                scale = 65535.0 if img.max() > 255.0 else 255.0
                img = img / scale
            return np.clip(img, 0.0, 1.0)

        if np.issubdtype(img.dtype, np.integer):
            scale = float(np.iinfo(img.dtype).max)
            img = img.astype(np.float32) / scale
            return np.clip(img, 0.0, 1.0)

        raise ValueError(f"Unsupported patch dtype {img.dtype} from {file_path}")

    def _crop_coordinates(self, height, width):
        max_row = height - self.crop_size
        max_col = width - self.crop_size
        if self.is_train:
            row = int(np.random.randint(max_row + 1))
            col = int(np.random.randint(max_col + 1))
        else:
            row = max_row // 2
            col = max_col // 2
        return row, col

    @staticmethod
    def _cloud_ratio(mask_crop):
        return float(np.count_nonzero(mask_crop > 0) / mask_crop.size)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        file_path, mask_path = self.records[idx]
        img = np.load(file_path, allow_pickle=False)
        mask = np.load(mask_path, allow_pickle=False)

        if img.ndim != 3:
            raise ValueError(
                f"Expected patch shape (H, W, C), got {img.shape} from {file_path}"
            )
        if img.shape[2] not in (3, 4):
            raise ValueError(
                f"Expected 3 or 4 channels, got {img.shape[2]} from {file_path}"
            )
        self._validate_mask(mask, mask_path)
        if img.shape[:2] != mask.shape:
            raise ValueError(
                f"Image/mask shape mismatch: image={img.shape[:2]} from {file_path}, "
                f"mask={mask.shape} from {mask_path}"
            )
        self._validate_spatial_size(img.shape[0], img.shape[1], file_path)

        img = self._normalize(img, file_path)
        row, col = self._crop_coordinates(img.shape[0], img.shape[1])
        row_end = row + self.crop_size
        col_end = col + self.crop_size
        img = img[row:row_end, col:col_end]
        mask_crop = mask[row:row_end, col:col_end]

        cloud_ratio = self._cloud_ratio(mask_crop)
        label = float(cloud_ratio >= self.cloud_ratio_threshold)

        if img.shape[2] < self.target_channels:
            pad_channels = self.target_channels - img.shape[2]
            padding = np.zeros((*img.shape[:2], pad_channels), dtype=img.dtype)
            img = np.concatenate([img, padding], axis=2)
        elif img.shape[2] > self.target_channels:
            img = img[:, :, :self.target_channels]

        img_tensor = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)

        if self.channel_dropout_p > 0 and torch.rand(1).item() < self.channel_dropout_p:
            img_tensor[3].zero_()

        if self.transform:
            img_tensor = self.transform(img_tensor)
        elif self.is_train:
            if torch.rand(1) > 0.5:
                img_tensor = torch.flip(img_tensor, dims=[1])
            if torch.rand(1) > 0.5:
                img_tensor = torch.flip(img_tensor, dims=[2])
            k = torch.randint(0, 4, (1,)).item()
            img_tensor = torch.rot90(img_tensor, k, dims=[1, 2])

        label_tensor = torch.tensor(label, dtype=torch.float32)
        return img_tensor, label_tensor

    def estimate_label_counts(self, samples_per_patch=8, seed=42):
        """Estimate the random-crop label distribution deterministically."""
        if samples_per_patch <= 0:
            raise ValueError(
                f"samples_per_patch must be greater than zero, got {samples_per_patch}"
            )

        rng = np.random.default_rng(seed)
        cloud_count = 0
        clear_count = 0
        for _, mask_path in self.records:
            mask = np.load(mask_path, allow_pickle=False)
            self._validate_mask(mask, mask_path)
            height, width = mask.shape
            self._validate_spatial_size(height, width, mask_path)
            max_row = height - self.crop_size
            max_col = width - self.crop_size

            for _ in range(samples_per_patch):
                row = int(rng.integers(max_row + 1))
                col = int(rng.integers(max_col + 1))
                mask_crop = mask[
                    row:row + self.crop_size,
                    col:col + self.crop_size,
                ]
                if self._cloud_ratio(mask_crop) >= self.cloud_ratio_threshold:
                    cloud_count += 1
                else:
                    clear_count += 1

        return cloud_count, clear_count
