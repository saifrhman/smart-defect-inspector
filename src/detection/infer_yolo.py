from __future__ import annotations

from pathlib import Path
import cv2

from ultralytics import YOLO

from src.db.log_run import create_run, log_metrics


def main() -> None:
    # Pick the first available image from processed NEU-DET folders
    candidates = list(Path("data/processed/images").rglob("*.jpg"))
    if not candidates:
        candidates = list(Path("data/processed/images").rglob("*.png"))
    if not candidates:
        raise FileNotFoundError("No images found in data/processed/images. Run: python -m src.data.make_dataset")

    img_path = candidates[0]

    if not img_path.exists():
        raise FileNotFoundError("Run: python -m src.data.make_dataset first")

    # Load YOLOv8 nano pretrained (we'll fine-tune later)
    model = YOLO("yolov8x.pt")

    # Run inference
    results = model(str(img_path))

    # Plot overlay (ultralytics returns RGB image)
    overlay_rgb = results[0].plot()
    overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)

    out_path = Path("outputs/yolo_example_pred.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay_bgr)

    # Print basic info
    n_boxes = 0 if results[0].boxes is None else len(results[0].boxes)
    run_id = create_run(
        task="detection",
        model_name="yolov8n",
        dataset_name="synthetic",
        notes="pretrained yolov8n inference on example.png",
    )
    log_metrics(run_id, {"detections": float(n_boxes)})

    print(f"Logged run_id: {run_id}")


    print(f"Saved: {out_path.resolve()}")
    print(f"Detections: {n_boxes}")


if __name__ == "__main__":
    main()
