from __future__ import annotations

from pathlib import Path
import cv2

from src.db.log_run import create_run, log_metrics


def infer_image_path_from_mask(mask_path: Path) -> Path:
    """
    mask: data/processed/fiji_masks/<stem>_mask.png
    image: data/processed/images/**/<stem>.jpg
    """
    stem = mask_path.stem.replace("_mask", "")
    matches = list(Path("data/processed/images").rglob(f"{stem}.jpg"))
    if not matches:
        matches = list(Path("data/processed/images").rglob(f"{stem}.png"))
    if not matches:
        raise FileNotFoundError(f"No matching image found for mask: {mask_path.name}")
    return matches[0]


def main() -> None:
    masks_dir = Path("data/processed/fiji_masks")
    if not masks_dir.exists():
        raise FileNotFoundError("Missing folder: data/processed/fiji_masks")

    mask_paths = sorted(masks_dir.glob("*_mask.png"))
    if not mask_paths:
        raise FileNotFoundError("No *_mask.png files found. Generate masks in Fiji first.")

    ok = 0
    skipped = 0

    for mask_path in mask_paths:
        try:
            img_path = infer_image_path_from_mask(mask_path)

            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError("Could not read mask")

            # auto polarity: defect = minority class
            white_pixels = int((mask > 0).sum())
            black_pixels = int((mask == 0).sum())
            total = int(mask.size)

            if white_pixels <= black_pixels:
                defect_pixels = white_pixels
                polarity = "white_is_defect"
            else:
                defect_pixels = black_pixels
                polarity = "black_is_defect"

            defect_ratio = defect_pixels / total

            run_id = create_run(
                task="fiji-segmentation",
                model_name="fiji-threshold-batch",
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
            ok += 1

        except Exception as e:
            print(f"Skipped {mask_path.name}: {e}")
            skipped += 1

    print(f"Logged {ok} Fiji masks. Skipped {skipped}.")


if __name__ == "__main__":
    main()
