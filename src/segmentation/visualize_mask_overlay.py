from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np


def main() -> None:
    img_path = Path("data/processed/images/example.png")
    mask_path = Path("data/processed/masks/example_mask.png")

    if not img_path.exists():
        raise FileNotFoundError("Missing image. Run: python -m src.data.make_dataset")
    if not mask_path.exists():
        raise FileNotFoundError("Missing mask. Run: python -m src.segmentation.make_synthetic_mask")

    img = cv2.imread(str(img_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    # Make a red overlay where mask is 255
    overlay = img.copy()
    red = np.zeros_like(img)
    red[:, :, 2] = 255  # BGR -> red channel

    alpha = 0.45
    m = mask > 0
    overlay[m] = (alpha * red[m] + (1 - alpha) * overlay[m]).astype(np.uint8)

    out_path = Path("outputs/segmentation_mask_overlay.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)

    print(f"Saved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
