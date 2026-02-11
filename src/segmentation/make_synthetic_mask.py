from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np


def main() -> None:
    img_path = Path("outputs/smoke_test.png")
    if not img_path.exists():
        raise FileNotFoundError("Run: python -m src.utils.smoke_test first")

    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]

    # Create binary mask (0 background, 255 defect)
    mask = np.zeros((h, w), dtype=np.uint8)

    # Rectangle coords must match smoke_test.py
    cv2.rectangle(mask, (60, 80), (180, 140), 255, thickness=-1)

    out_dir = Path("data/processed/masks")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "example_mask.png"
    cv2.imwrite(str(out_path), mask)

    print(f"Saved mask: {out_path.resolve()}")


if __name__ == "__main__":
    main()
