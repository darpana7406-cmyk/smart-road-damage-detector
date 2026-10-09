"""
Run YOLO inference on a single image and save annotated output.
Usage: python src\detect.py path\to\image.jpg
"""
from ultralytics import YOLO
import cv2
import os
import sys

MODEL_PATH = r"D:\PBEL\runs\yolov8n_rdd2022\weights\best.pt"
OUTPUT_DIR = r"D:\PBEL\outputs\images"

CLASS_NAMES = {
    0: "Longitudinal crack",
    1: "Transverse crack",
    2: "Alligator crack",
    3: "Pothole",
}


def detect(image_path, conf=0.35):
    if not os.path.exists(MODEL_PATH):
        raise SystemExit(f"Model not found: {MODEL_PATH}\nWait for training to finish.")

    model = YOLO(MODEL_PATH)
    results = model(image_path, conf=conf)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    counts = {name: 0 for name in CLASS_NAMES.values()}
    detections = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls)
            conf_val = float(box.conf)
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            label = CLASS_NAMES.get(cls, f"class_{cls}")
            counts[label] += 1
            detections.append({
                "class": label,
                "confidence": conf_val,
                "bbox": (x1, y1, x2, y2),
                "width_px": x2 - x1,
                "height_px": y2 - y1,
            })

        annotated = r.plot()
        out_path = os.path.join(OUTPUT_DIR, os.path.basename(image_path))
        cv2.imwrite(out_path, annotated)
        print(f"Saved annotated image: {out_path}")

    print("\n========== DETECTION SUMMARY ==========")
    for name, count in counts.items():
        if count:
            print(f"{name:22s}: {count:02d}")
    print(f"Total defects       : {sum(counts.values()):02d}")
    print("========================================\n")
    return detections, counts


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src\\detect.py <image_path>")
        sys.exit(1)
    detect(sys.argv[1])