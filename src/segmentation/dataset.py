from __future__ import annotations

from pathlib import Path
from typing import Tuple, List

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class FijiSegmentationDataset(Dataset):
    """
    Loads paired data from:
      data/segmentation/images/<stem>.jpg
      data/segmentation/masks/<stem>.png
    Returns:
      image: float32 tensor [3,H,W] in 0..1
      mask:  float32 tensor [1,H,W] in {0,1}
    """

    def __init__(self, images_dir: str | Path, masks_dir: str | Path, size: int = 256):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.size = int(size)

        self.images: List[Path] = sorted(
            [p for p in self.images_dir.iterdir() if p.suffix.lower() in {".jpg", ".png", ".jpeg"}]
        )

        if not self.images:
            raise FileNotFoundError(f"No images found in {self.images_dir}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.images[idx]
        stem = img_path.stem
        mask_path = self.masks_dir / f"{stem}.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask for {stem}: {mask_path}")

        img_bgr = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if img_bgr is None or mask is None:
            raise ValueError(f"Failed to read image/mask for {stem}")

        # Resize (keep it simple for now)
        img_bgr = cv2.resize(img_bgr, (self.size, self.size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)

        # Convert image to RGB, normalize
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_t = torch.from_numpy(img_rgb).permute(2, 0, 1)  # [3,H,W]

        # --- Robust mask binarization with polarity handling ---
        white_pixels = int((mask > 0).sum())
        black_pixels = int((mask == 0).sum())
        defect = (mask > 0) if white_pixels <= black_pixels else (mask == 0)

        mask_t = torch.from_numpy(defect.astype(np.float32)).unsqueeze(0)  # [1,H,W]

        return img_t, mask_t
