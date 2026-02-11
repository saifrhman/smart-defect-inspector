from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class PreprocessConfig:
    to_gray: bool = True
    denoise: bool = True
    clahe: bool = True
    blur_ksize: int = 3  # must be odd (e.g., 3,5,7)
    normalize: bool = True


def apply_preprocess(img_bgr: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """
    Input: BGR uint8 image (OpenCV default)
    Output: preprocessed image (uint8), grayscale if cfg.to_gray else BGR.
    """
    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("Empty image passed to apply_preprocess().")

    out = img_bgr

    # 1) Convert to grayscale
    if cfg.to_gray:
        out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)

    # 2) Denoise
    if cfg.denoise:
        if cfg.to_gray:
            out = cv2.medianBlur(out, ksize=3)
        else:
            out = cv2.GaussianBlur(out, (cfg.blur_ksize, cfg.blur_ksize), 0)

    # 3) Contrast enhancement (CLAHE)
    if cfg.clahe:
        if not cfg.to_gray:
            # Apply CLAHE on L channel in LAB
            lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l2 = clahe.apply(l)
            lab2 = cv2.merge((l2, a, b))
            out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            out = clahe.apply(out)

    # 4) Normalize to full 0..255 (keeps uint8)
    if cfg.normalize:
        out = cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX)

    return out.astype(np.uint8)
