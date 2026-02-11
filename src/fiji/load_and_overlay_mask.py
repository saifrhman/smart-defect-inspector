from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np

from src.db.log_run import create_run, log_metrics


def main() -> None:
    # Update these if you saved a different file in Fiji
    img_path = Path("data/processed/images/scratches/scratches_23.jpg")
    mask_path = Path("data/processed/fiji_masks/scratches_23.png")

    if not img_path.exists():
        raise FileNotFoundError(f"Missing image: {img_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Missing Fiji mask: {mask_path}")

    img = cv2.imread(str(img_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"Could not read image: {img_path}")
    if mask is None:
        raise ValueError(f"Could not read mask: {mask_path}")

    if img.shape[:2] != mask.shape[:2]:
        raise ValueError(f"Image/mask size mismatch: img={img.shape[:2]} mask={mask.shape[:2]}")

    # --- Determine which pixels correspond to the defect (auto-invert if needed) ---
    white_pixels = int((mask > 0).sum())
    black_pixels = int((mask == 0).sum())
    total_pixels = int(mask.size)

    # Assume defect is the minority class (usually smaller area than background)
    if white_pixels <= black_pixels:
        defect = mask > 0
        polarity = "white_is_defect"
    else:
        defect = mask == 0
        polarity = "black_is_defect"

    defect_pixels = int(defect.sum())
    defect_ratio = defect_pixels / total_pixels

    # --- Overlay red where defect is True ---
    overlay = img.copy()
    red = np.zeros_like(img)
    red[:, :, 2] = 255  # BGR -> red channel

    alpha = 0.45
    overlay[defect] = (alpha * red[defect] + (1 - alpha) * overlay[defect]).astype(np.uint8)

    out_path = Path("outputs/fiji_mask_overlay.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)

    # --- Log to SQLite ---
    run_id = create_run(
        task="fiji-segmentation",
        model_name="fiji-threshold",
        dataset_name="NEU-DET",
        notes=f"polarity={polarity} mask={mask_path.name} image={img_path.name}",
    )
    log_metrics(
        run_id,
        {
            "defect_pixels": float(defect_pixels),
            "defect_ratio": float(defect_ratio),
            "white_pixels": float(white_pixels),
            "black_pixels": float(black_pixels),
        },
    )

    print(f"Logged run_id: {run_id}")
    print(f"Saved: {out_path.resolve()}")
    print(f"Polarity: {polarity}")
    print(f"Defect pixels: {defect_pixels}")
    print(f"Defect ratio: {defect_ratio:.4f}")
    print(f"White pixels: {white_pixels} | Black pixels: {black_pixels} | Total: {total_pixels}")


if __name__ == "__main__":
    main()
