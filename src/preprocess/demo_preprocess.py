from pathlib import Path
import cv2

from src.preprocess.transforms import PreprocessConfig, apply_preprocess


def main() -> None:
    in_path = Path("outputs/smoke_test.png")
    if not in_path.exists():
        raise FileNotFoundError("Run: python src/utils/smoke_test.py first")

    img = cv2.imread(str(in_path))
    cfg = PreprocessConfig(to_gray=True, denoise=True, clahe=True, normalize=True)
    out = apply_preprocess(img, cfg)

    out_path = Path("outputs/smoke_test_preprocessed.png")
    cv2.imwrite(str(out_path), out)
    print(f"Saved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
