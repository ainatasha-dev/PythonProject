from pathlib import Path

from ultralytics import YOLO

# LAB 6 - PART 4
# Run YOLO detection on a video if available, otherwise on a sample image.

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "yolo26m.pt"
IMAGE_FALLBACK = BASE_DIR / "car.jpg"
VIDEO_CANDIDATES = [
    BASE_DIR / "sample_video.mov",
    BASE_DIR / "sample_video.mp4",
    BASE_DIR / "sample.mp4",
]
OUTPUT_DIR = BASE_DIR / "runs" / "detect"


def choose_source() -> Path:
    for video in VIDEO_CANDIDATES:
        if video.exists():
            print(f"Using video input: {video.name}")
            return video
    if IMAGE_FALLBACK.exists():
        print(f"No video found, using image input: {IMAGE_FALLBACK.name}")
        return IMAGE_FALLBACK
    raise FileNotFoundError(
        "No input file found. Add a video (e.g., sample_vid_2.mov) "
        "or ensure car.jpg exists."
    )


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    source_path = choose_source()

    print("Loading YOLO model...")
    model = YOLO(str(MODEL_PATH))

    print("Running detection...")
    model.predict(
        source=str(source_path),
        imgsz=640,
        conf=0.25,
        iou=0.45,
        save=True,
        save_txt=False,
        save_conf=False,
        project=str(OUTPUT_DIR),
        name="part4_results",
        exist_ok=True,
    )

    print("Processing completed.")
    print(f"Results saved in: {OUTPUT_DIR / 'part4_results'}")


if __name__ == "__main__":
    main()
