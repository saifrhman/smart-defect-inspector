from pathlib import Path
import numpy as np
import cv2

def main() -> None:
    # Create a simple synthetic image (black with a white rectangle = "defect")
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.rectangle(img, (60, 80), (180, 140), (255, 255, 255), thickness=-1)

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "smoke_test.png"
    cv2.imwrite(str(out_path), img)

    print("Smoke test passed.")
    print(f"Saved: {out_path.resolve()}")

if __name__ == "__main__":
    main()
