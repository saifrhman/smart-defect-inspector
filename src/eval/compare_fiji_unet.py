from __future__ import annotations

from html import parser
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

import argparse

from src.segmentation.unet import UNetSmall


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=str, default="scratches_23", help="Stem or filename, e.g. scratches_23")
    args = parser.parse_args()
    sample = Path(args.sample).stem


    img_path = Path(f"data/segmentation/images/{sample}.jpg")
    fiji_mask_path = Path(f"data/segmentation/masks/{sample}.png")
    weights = Path("outputs/unet/best.pt")

    if not img_path.exists():
        raise FileNotFoundError(img_path)
    if not fiji_mask_path.exists():
        raise FileNotFoundError(fiji_mask_path)
    if not weights.exists():
        raise FileNotFoundError(weights)

    # Load image + mask
    img_bgr = cv2.imread(str(img_path))
    fiji_mask = cv2.imread(str(fiji_mask_path), 0)

    size = 256
    img_bgr_r = cv2.resize(img_bgr, (size, size))
    fiji_mask_r = cv2.resize(fiji_mask, (size, size), interpolation=cv2.INTER_NEAREST)

    img_rgb = cv2.cvtColor(img_bgr_r, cv2.COLOR_BGR2RGB)

    # Polarity handling (defect = minority)
    white = (fiji_mask_r > 0).sum()
    black = (fiji_mask_r == 0).sum()
    fiji_bin = (fiji_mask_r > 0) if white <= black else (fiji_mask_r == 0)

    # Prepare tensor
    x = torch.from_numpy(img_rgb.astype(np.float32) / 255.0)
    x = x.permute(2, 0, 1).unsqueeze(0)

    # Load model
    model = UNetSmall(base=32)
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits)[0, 0].numpy()
        pred = prob > 0.5

    # Overlay prediction
    overlay = img_rgb.copy()
    overlay[pred] = [255, 0, 0]  # red defect

    # Plot
    plt.figure(figsize=(10, 6))

    plt.subplot(2, 2, 1)
    plt.title("Original Image")
    plt.imshow(img_rgb)
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.title("Fiji Mask")
    plt.imshow(fiji_bin, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.title("U-Net Prediction")
    plt.imshow(pred, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.title("Prediction Overlay")
    plt.imshow(overlay)
    plt.axis("off")

    out_path = Path("outputs/fiji_vs_unet.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
