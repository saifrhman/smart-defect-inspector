from __future__ import annotations

from pathlib import Path


def main() -> None:
    in_path = Path("data/processed/images/example.png")
    if not in_path.exists():
        raise FileNotFoundError("Run: python -m src.data.make_dataset first")

    print("Detection scaffold is ready.")
    print("Next: install ultralytics and implement real YOLO inference.")
    print(f"Would run inference on: {in_path.resolve()}")


if __name__ == "__main__":
    main()
