from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
import torch

import argparse

from src.segmentation.unet import UNetSmall
from src.db.log_run import create_run, log_metrics


def load_pair(stem: str | None) -> tuple[Path, Path]:
    img_dir = Path("data/segmentation/images")
    msk_dir = Path("data/segmentation/masks")

    if stem:
        # allow passing either "scratches_23" or "scratches_23.jpg"
        s = Path(stem).stem
        matches = list(img_dir.glob(f"{s}.*"))
        if not matches:
            raise FileNotFoundError(f"No image found for stem '{s}' in {img_dir}")
        img_path = matches[0]
        mask_path = msk_dir / f"{img_path.stem}.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask: {mask_path}")
        return img_path, mask_path

    imgs = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".png", ".jpeg"}])
    if not imgs:
        raise FileNotFoundError("No images in data/segmentation/images")

    img_path = imgs[0]
    mask_path = msk_dir / f"{img_path.stem}.png"
    if not mask_path.exists():
        raise FileNotFoundError(f"Missing mask: {mask_path}")
    return img_path, mask_path


def dice(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    pred = pred.astype(np.bool_)
    gt = gt.astype(np.bool_)
    inter = (pred & gt).sum()
    union = pred.sum() + gt.sum()
    return float((2 * inter + eps) / (union + eps))


def main() -> None:
    weights = Path("outputs/unet/best.pt")
    if not weights.exists():
        raise FileNotFoundError("Missing weights. Train first: python -m src.segmentation.train_unet")

    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=str, default=None, help="Image stem or filename, e.g. scratches_23")
    args = parser.parse_args()

    img_path, gt_mask_path = load_pair(args.sample)


    # Load + preprocess
    img_bgr = cv2.imread(str(img_path))
    gt_raw = cv2.imread(str(gt_mask_path), cv2.IMREAD_GRAYSCALE)
    if img_bgr is None or gt_raw is None:
        raise ValueError("Failed to read image or mask")

    size = 256
    img_bgr_r = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_AREA)
    gt_r = cv2.resize(gt_raw, (size, size), interpolation=cv2.INTER_NEAREST)

    img_rgb = cv2.cvtColor(img_bgr_r, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]

    # Robust GT binarization (same polarity logic as dataset)
    white_pixels = int((gt_r > 0).sum())
    black_pixels = int((gt_r == 0).sum())
    gt = (gt_r > 0) if white_pixels <= black_pixels else (gt_r == 0)

    # Model
    model = UNetSmall(base=32)
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits)[0, 0].numpy()  # [H,W]
        pred = prob > 0.5

    d = dice(pred, gt)

    # Save predicted mask (white=defect)
    pred_mask = (pred.astype(np.uint8) * 255)
    sample_name = img_path.stem
    out_dir = Path("outputs/unet") / sample_name
    out_dir.mkdir(parents=True, exist_ok=True)

    out_pred = out_dir / "pred_mask.png"
    out_overlay = out_dir / "pred_overlay.png"

    cv2.imwrite(str(out_pred), pred_mask)

    # Save overlay on original resized image
    overlay = img_bgr_r.copy()
    red = np.zeros_like(overlay)
    red[:, :, 2] = 255
    alpha = 0.45
    overlay[pred] = (alpha * red[pred] + (1 - alpha) * overlay[pred]).astype(np.uint8)

    cv2.imwrite(str(out_overlay), overlay)

    # Log
    run_id = create_run(
        task="segmentation-infer",
        model_name="unet-small",
        dataset_name="NEU-DET (Fiji masks)",
        notes=f"sample={img_path.name} weights=best.pt thr=0.5",
    )
    log_metrics(run_id, {"dice_vs_fiji": float(d)})

    print(f"Image: {img_path}")
    print(f"GT mask: {gt_mask_path}")
    print(f"Saved pred mask: {out_pred.resolve()}")
    print(f"Saved overlay:  {out_overlay.resolve()}")
    print(f"Dice vs Fiji: {d:.4f}")
    print(f"Logged run_id: {run_id}")


if __name__ == "__main__":
    main()
