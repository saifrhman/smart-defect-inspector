from pathlib import Path
import cv2
import matplotlib.pyplot as plt


def main() -> None:
    before_path = Path("outputs/smoke_test.png")
    after_path = Path("outputs/smoke_test_preprocessed.png")

    if not before_path.exists() or not after_path.exists():
        raise FileNotFoundError(
            "Missing outputs. Run:\n"
            "1) python -m src.utils.smoke_test\n"
            "2) python -m src.preprocess.demo_preprocess"
        )

    before = cv2.imread(str(before_path))
    after = cv2.imread(str(after_path), cv2.IMREAD_GRAYSCALE)

    # Convert BGR -> RGB for matplotlib
    before_rgb = cv2.cvtColor(before, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title("Before")
    plt.imshow(before_rgb)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("After (Preprocessed)")
    plt.imshow(after, cmap="gray")
    plt.axis("off")

    out_path = Path("outputs/preprocess_comparison.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
