from pathlib import Path
import cv2

import numpy as np
from src.db.log_run import create_run, log_metrics

from src.preprocess.transforms import PreprocessConfig, apply_preprocess

from src.utils.config import load_config


def main() -> None:
    in_path = Path("outputs/smoke_test.png")
    if not in_path.exists():
        raise FileNotFoundError("Run: python src/utils/smoke_test.py first")

    img = cv2.imread(str(in_path))
    app_cfg = load_config()
    p = app_cfg["preprocess"]

    cfg = PreprocessConfig(
        to_gray=bool(p["to_gray"]),
        denoise=bool(p["denoise"]),
        clahe=bool(p["clahe"]),
        blur_ksize=int(p["blur_ksize"]),
        normalize=bool(p["normalize"]),
    )
    out = apply_preprocess(img, cfg)

    out_path = Path("outputs/smoke_test_preprocessed.png")
    cv2.imwrite(str(out_path), out)

    # --- Simple metrics (just to demonstrate logging) ---
    # mean intensity + edge strength proxy (Laplacian variance)
    if out.ndim == 3:
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    else:
        gray = out

    mean_intensity = float(np.mean(gray))
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    run_id = create_run(
        task="preprocess",
        model_name="opencv",
        dataset_name="synthetic",
        notes="smoke_test preprocessing demo",
    )
    log_metrics(run_id, {"mean_intensity": mean_intensity, "laplacian_var": lap_var})

    print(f"Saved: {out_path.resolve()}")
    print(f"Logged run_id: {run_id}")

if __name__ == "__main__":
    main()
