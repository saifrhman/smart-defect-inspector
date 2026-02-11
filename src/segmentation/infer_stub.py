from __future__ import annotations

from pathlib import Path


def main() -> None:
    img_path = Path("data/processed/images/example.png")
    if not img_path.exists():
        raise FileNotFoundError("Run: python -m src.data.make_dataset first")

    print("Segmentation scaffold is ready.")
    print("Next: add torch + a small U-Net, then predict a mask.")
    print(f"Would run segmentation on: {img_path.resolve()}")


if __name__ == "__main__":
    main()
