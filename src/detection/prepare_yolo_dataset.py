from __future__ import annotations

from pathlib import Path
import shutil


CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


def yolo_label_full_image(class_id: int) -> str:
    # YOLO format: class cx cy w h (normalized). Full image box => 0.5 0.5 1.0 1.0
    return f"{class_id} 0.5 0.5 1.0 1.0\n"


def copy_split(split_name: str, src_root: Path, dst_root: Path) -> int:
    """
    Copies images and writes full-image bbox labels.
    Expects:
      src_root/<split_name>/images/<class>/*.jpg
    Creates:
      dst_root/images/<split_name>/*.jpg
      dst_root/labels/<split_name>/*.txt
    """
    img_dst = dst_root / "images" / split_name
    lab_dst = dst_root / "labels" / split_name
    img_dst.mkdir(parents=True, exist_ok=True)
    lab_dst.mkdir(parents=True, exist_ok=True)

    count = 0
    for cls in CLASSES:
        cls_dir = src_root / split_name / "images" / cls
        if not cls_dir.exists():
            continue

        class_id = CLASSES.index(cls)

        for img_path in cls_dir.glob("*.jpg"):
            # copy image
            dst_img = img_dst / img_path.name
            shutil.copy2(img_path, dst_img)

            # write label
            (lab_dst / f"{img_path.stem}.txt").write_text(yolo_label_full_image(class_id), encoding="utf-8")
            count += 1

    return count


def main() -> None:
    src_root = Path("data/raw/NEU-DET")
    if not src_root.exists():
        raise FileNotFoundError("Expected data/raw/NEU-DET")

    out_root = Path("data/yolo_neu_det")
    out_root.mkdir(parents=True, exist_ok=True)

    n_train = copy_split("train", src_root, out_root)
    n_val = copy_split("validation", src_root, out_root)

    # ultralytics dataset YAML
    yaml_text = f"""path: {out_root.as_posix()}
train: images/train
val: images/validation

names:
"""
    for i, name in enumerate(CLASSES):
        yaml_text += f"  {i}: {name}\n"

    (out_root / "neu_det.yaml").write_text(yaml_text, encoding="utf-8")

    print("YOLO dataset prepared (using NEU-DET split):")
    print(f"- train: {n_train} images")
    print(f"- val:   {n_val} images")
    print("Config:", (out_root / "neu_det.yaml").resolve())


if __name__ == "__main__":
    main()
