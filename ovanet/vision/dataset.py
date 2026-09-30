"""
ovanet/vision/dataset.py

PyTorch Dataset and augmentation pipeline for OvaNet U-Net training.
Supports loading raw ultrasound images with 2-channel binary masks:
  Channel 0: Ovary sector ROI
  Channel 1: Follicles
"""

from pathlib import Path
import random
import numpy as np
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset

from ovanet.contracts import IMAGE_SIZE


class OvaNetSegmentationDataset(Dataset):
    """
    Dataset for ultrasound segmentation.

    Parameters
    ----------
    manifest : pd.DataFrame | list[Path | str]
        Either a DataFrame containing columns ('path', 'mask_file') or a list of image paths.
    raw_root : Path | str
        Root directory for raw images.
    mask_dir : Path | str
        Directory containing generated pseudo-masks (.npz).
    is_train : bool
        If True, applies data augmentations (flips, slight rotations, contrast).
    """

    def __init__(
        self,
        manifest: pd.DataFrame | list[str | Path],
        raw_root: str | Path,
        mask_dir: str | Path,
        is_train: bool = True,
    ):
        self.raw_root = Path(raw_root)
        self.mask_dir = Path(mask_dir)
        self.is_train = is_train

        if isinstance(manifest, pd.DataFrame):
            self.df = manifest.reset_index(drop=True)
            self.has_manifest = True
        else:
            self.image_paths = [Path(p) for p in manifest]
            self.has_manifest = False

    def __len__(self) -> int:
        return len(self.df) if self.has_manifest else len(self.image_paths)

    def _augment(
        self, image: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply geometric augmentations consistently to image and mask."""
        if random.random() > 0.5:
            image = np.fliplr(image)
            mask = np.flip(mask, axis=2)

        if random.random() > 0.8:
            image = np.flipud(image)
            mask = np.flip(mask, axis=1)

        if random.random() > 0.5:
            angle = random.uniform(-10.0, 10.0)
            center = (IMAGE_SIZE / 2, IMAGE_SIZE / 2)
            rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(
                image, rot_mat, (IMAGE_SIZE, IMAGE_SIZE), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
            )
            ch0 = cv2.warpAffine(
                mask[0], rot_mat, (IMAGE_SIZE, IMAGE_SIZE), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
            )
            ch1 = cv2.warpAffine(
                mask[1], rot_mat, (IMAGE_SIZE, IMAGE_SIZE), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
            )
            mask = np.stack([ch0, ch1], axis=0)

        if random.random() > 0.5:
            alpha = random.uniform(0.85, 1.15)
            beta = random.uniform(-0.08, 0.08)
            image = np.clip(image * alpha + beta, 0.0, 1.0)

        return image, mask

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        if self.has_manifest:
            row = self.df.iloc[idx]
            rel_path = Path(row["path"])
            img_path = self.raw_root / rel_path
            mask_filename = row.get("mask_file", f"{rel_path.stem}.npz")
            mask_path = self.mask_dir / mask_filename
            stem = row.get("stem", rel_path.stem)
        else:
            img_path = self.image_paths[idx]
            if not img_path.is_absolute():
                img_path = self.raw_root / img_path
            mask_path = self.mask_dir / f"{img_path.stem}.npz"
            stem = img_path.stem

        raw_bgr = cv2.imread(str(img_path))
        if raw_bgr is None:
            gray = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        else:
            gray = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY)
            if gray.shape != (IMAGE_SIZE, IMAGE_SIZE):
                gray = cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE))
            gray = gray.astype(np.float32) / 255.0

        if mask_path.exists():
            data = np.load(mask_path)
            ovary_mask = data["ovary"].astype(np.float32)
            follicle_mask = data["follicles"].astype(np.float32)
            mask = np.stack([ovary_mask, follicle_mask], axis=0)
        else:
            mask = np.zeros((2, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

        if self.is_train:
            gray, mask = self._augment(gray, mask)

        image_tensor = torch.from_numpy(np.ascontiguousarray(gray)).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask)).float()

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "name": stem,
        }
