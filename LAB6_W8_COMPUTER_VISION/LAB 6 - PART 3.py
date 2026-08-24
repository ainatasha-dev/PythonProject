from pathlib import Path

from ultralytics import YOLO

# LAB 6 - PART 3
# Object detection using a YOLO model on a local image.

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "yolo26m.pt"
IMAGE_PATH = BASE_DIR / "car.jpg"
OUTPUT_DIR = BASE_DIR / "runs" / "detect"


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Image file not found: {IMAGE_PATH}")

    print("Loading YOLO model...")
    model = YOLO(str(MODEL_PATH))

    print("Running detection...")
    model.predict(
        source=str(IMAGE_PATH),
        imgsz=640,
        conf=0.25,
        iou=0.45,
        save=True,
        save_txt=True,
        save_conf=True,
        project=str(OUTPUT_DIR),
        name="exp",
        exist_ok=True,
    )

    print("YOLO detection finished.")
    print(f"Results saved in: {OUTPUT_DIR / 'exp'}")


if __name__ == "__main__":
    main()