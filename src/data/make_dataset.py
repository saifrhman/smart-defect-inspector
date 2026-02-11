from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil


@dataclass
class DatasetConfig:
    raw_dir: Path = Path("data/raw/NEU-DET/train/images")
    processed_dir: Path = Path("data/processed/images")


def ensure_dirs(cfg: DatasetConfig) -> None:
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)


def ingest_neu_det(cfg: DatasetConfig) -> None:
    if not cfg.raw_dir.exists():
        raise FileNotFoundError(f"NEU-DET not found at {cfg.raw_dir}")

    image_extensions = {".jpg", ".png", ".jpeg", ".bmp"}

    count = 0
    for class_dir in cfg.raw_dir.iterdir():
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        target_class_dir = cfg.processed_dir / class_name
        target_class_dir.mkdir(parents=True, exist_ok=True)

        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() not in image_extensions:
                continue

            shutil.copy2(img_path, target_class_dir / img_path.name)
            count += 1

    print(f"Ingested {count} images into {cfg.processed_dir}")


def main() -> None:
    cfg = DatasetConfig()
    ensure_dirs(cfg)
    ingest_neu_det(cfg)


if __name__ == "__main__":
    main()
