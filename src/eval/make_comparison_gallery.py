from __future__ import annotations

import random
import subprocess
import sys
from pathlib import Path


def main() -> None:
    img_dir = Path("data/segmentation/images")
    samples = sorted([p.stem for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".png", ".jpeg"}])
    if len(samples) < 5:
        raise SystemExit("Need at least 5 samples in data/segmentation/images")

    random.seed(42)
    picks = random.sample(samples, 5)

    out_dir = Path("outputs/gallery")
    out_dir.mkdir(parents=True, exist_ok=True)

    for s in picks:
        print(f"Generating: {s}")
        subprocess.check_call([sys.executable, "-m", "src.eval.compare_fiji_unet", "--sample", s])
        # move the output to a unique filename
        src = Path("outputs/fiji_vs_unet.png")
        dst = out_dir / f"{s}_fiji_vs_unet.png"
        if src.exists():
            src.replace(dst)

    print("✅ Gallery saved in:", out_dir.resolve())
    print("Files:")
    for p in sorted(out_dir.glob("*.png")):
        print("-", p.name)


if __name__ == "__main__":
    main()
