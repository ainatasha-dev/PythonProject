import os
import subprocess
import sys

YOLO_DIR = r"C:\Users\aena zam\Downloads\yolov7"
SOURCE_IMAGE = r"C:\Users\aena zam\Downloads\vader.jpg"
WEIGHTS = r"C:\Users\aena zam\Downloads\yolo26n.pt"

yolo_detect = os.path.join(YOLO_DIR, "detect.py")
if not os.path.isfile(yolo_detect):
    raise FileNotFoundError(f"YOLO detect.py not found: {yolo_detect}")
if not os.path.isfile(SOURCE_IMAGE):
    raise FileNotFoundError(f"Image not found: {SOURCE_IMAGE}")
if not os.path.isfile(WEIGHTS):
    raise FileNotFoundError(f"Weights not found: {WEIGHTS}")

cmd = [
    sys.executable, yolo_detect,
    "--weights", WEIGHTS,
    "--source", SOURCE_IMAGE,
    "--img-size", "640",
    "--conf-thres", "0.25",
    "--iou-thres", "0.45",
    "--save-txt",
    "--save-conf",
    "--project", os.path.join(YOLO_DIR, "runs", "detect"),
    "--name", "exp",
    "--exist-ok",
]

print("Running YOLO detection...")
result = subprocess.run(cmd, cwd=YOLO_DIR, text=True, capture_output=True)

print("STDOUT:\n", result.stdout)
print("STDERR:\n", result.stderr)

if result.returncode != 0:
    raise RuntimeError(f"Detection failed with exit code {result.returncode}")

print("YOLO detection finished.")
print("Results are in:", os.path.join(YOLO_DIR, "runs", "detect", "exp"))