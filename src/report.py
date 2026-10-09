"""
Full pipeline: detect -> measure -> severity -> GPS -> formatted report.

Usage:
    python src\report.py path\to\image.jpg
    python src\report.py path\to\image.jpg --ref-px 400 --ref-cm 350
    python src\report.py path\to\image.jpg --lat 18.5204 --lon 73.8567
    python src\report.py path\to\image.jpg --real-gps
"""
from ultralytics import YOLO
import cv2
import os
import sys
import argparse
import platform
from datetime import datetime

# Local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from measure import compute_cm_per_pixel, pothole_area_cm2, crack_length_cm
from severity import pothole_severity, crack_severity, overall_severity
from gps import read_gps, mock_gps

MODEL_PATH = r"D:\PBEL\runs\yolov8n_rdd2022\weights\best.pt"
OUTPUT_DIR = r"D:\PBEL\outputs"

CLASS_NAMES = {
    0: "Longitudinal crack",
    1: "Transverse crack",
    2: "Alligator crack",
    3: "Pothole",
}


def format_gps(lat, lon):
    """Format lat/lon as decimal degrees with N/S/E/W suffix."""
    lat_dir = "N" if lat >= 0 else "S"
    lon_dir = "E" if lon >= 0 else "W"
    return f"{abs(lat):.4f}\u00b0 {lat_dir}\n  {abs(lon):.4f}\u00b0 {lon_dir}"


def run(image_path, ref_px=400, ref_cm=350, use_mock_gps=True,
        lat=None, lon=None, auto_open=True):
    if not os.path.exists(MODEL_PATH):
        raise SystemExit(f"Model not found: {MODEL_PATH}")

    model = YOLO(MODEL_PATH)
    results = model(image_path, conf=0.35, verbose=False, workers=0)

    cm_per_px = compute_cm_per_pixel(ref_px, ref_cm)

    # Image dimensions
    r0 = results[0]
    img_h, img_w = r0.orig_shape
    road_area_cm2 = (img_w * cm_per_px) * (img_h * cm_per_px)
    road_area_m2 = road_area_cm2 / 10000.0  # cm² -> m²

    potholes = []
    cracks = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls)
            conf = float(box.conf)
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            name = CLASS_NAMES.get(cls, f"class_{cls}")

            if cls == 3:  # pothole
                width_cm = (x2 - x1) * cm_per_px
                area_cm2 = pothole_area_cm2((x1, y1, x2, y2), cm_per_px)
                sev = pothole_severity(width_cm)
                potholes.append({
                    "width_cm": width_cm,
                    "area_cm2": area_cm2,
                    "severity": sev,
                    "conf": conf,
                })
            else:  # crack
                if cls == 0:
                    ctype = "longitudinal"
                elif cls == 1:
                    ctype = "transverse"
                else:
                    ctype = "alligator"
                length_cm = crack_length_cm((x1, y1, x2, y2), cm_per_px, ctype)
                cracks.append({
                    "type": ctype,
                    "name": name,
                    "length_cm": length_cm,
                    "conf": conf,
                })

    # Save annotated image
    os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
    annotated = results[0].plot()
    out_img = os.path.join(OUTPUT_DIR, "images", os.path.basename(image_path))
    cv2.imwrite(out_img, annotated)

    # -------- CRACK DENSITY (linear density in m/m²) --------
    # Empirical reference: 1 m of crack per 100 m² road ≈ 1% density.
    # This gives a meaningful 0-100% scale for severity classification.
    total_len_cm = sum(c["length_cm"] for c in cracks)
    total_len_m = total_len_cm / 100.0  # cm -> m

    if road_area_m2 > 0:
        # meters of crack per 100 square meters of road
        density_per_100m2 = (total_len_m / road_area_m2) * 100
        density_pct = min(density_per_100m2, 100.0)
    else:
        density_pct = 0.0

    # Largest pothole
    largest = max(potholes, key=lambda p: p["width_cm"]) if potholes else None

    # GPS handling
    if lat is not None and lon is not None:
        gps = {"lat": lat, "lon": lon, "alt": 0.0}
    elif use_mock_gps:
        gps = mock_gps()
    else:
        gps = read_gps()

    # Overall severity
    all_sev = [p["severity"] for p in potholes]
    for c in cracks:
        all_sev.append(crack_severity(c["length_cm"], density_pct))
    overall = overall_severity(all_sev) if all_sev else "LOW"

    # Build report text
    lines = []
    lines.append("=" * 40)
    lines.append("         ROAD CONDITION REPORT")
    lines.append("=" * 40)
    lines.append(f"Potholes : {len(potholes):02d}")
    lines.append(f"Cracks   : {len(cracks):02d}")
    lines.append("")

    if largest:
        lines.append("Largest pothole:")
        lines.append(f"  Width    : {largest['width_cm']:.1f} cm")
        lines.append(f"  Area     : {largest['area_cm2']:.0f} cm2")
        lines.append(f"  Severity : {largest['severity']}")
        lines.append("")

    if cracks:
        lines.append("Crack metrics:")
        lines.append(f"  Total length   : {total_len_cm:.1f} cm ({total_len_m:.2f} m)")
        lines.append(f"  Crack density  : {density_pct:.2f}%")
        lines.append("")

    if gps:
        lines.append("GPS:")
        lines.append(f"  {format_gps(gps['lat'], gps['lon'])}")
        lines.append("")

    lines.append(f"Total defects     : {len(potholes) + len(cracks):02d}")
    lines.append(f"Overall severity  : {overall}")
    lines.append(f"Timestamp         : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 40)

    report_text = "\n".join(lines)
    print(report_text)

    # Save report
    os.makedirs(os.path.join(OUTPUT_DIR, "reports"), exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(OUTPUT_DIR, "reports", f"report_{ts}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\nReport saved: {report_path}")
    print(f"Image saved : {out_img}")

    # Auto-open the annotated image
    if auto_open and platform.system() == "Windows":
        os.system(f'start "" "{out_img}"')

    return report_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to road image")
    parser.add_argument("--ref-px", type=float, default=400,
                        help="Pixel width of reference object (default 400)")
    parser.add_argument("--ref-cm", type=float, default=350,
                        help="Real-world width of reference in cm (default 350)")
    parser.add_argument("--lat", type=float, default=None,
                        help="Latitude override (e.g. 18.5204)")
    parser.add_argument("--lon", type=float, default=None,
                        help="Longitude override (e.g. 73.8567)")
    parser.add_argument("--real-gps", action="store_true",
                        help="Read from real GPS module instead of mock")
    parser.add_argument("--no-open", action="store_true",
                        help="Don't auto-open annotated image")

    args = parser.parse_args()

    run(
        args.image,
        ref_px=args.ref_px,
        ref_cm=args.ref_cm,
        use_mock_gps=not args.real_gps,
        lat=args.lat,
        lon=args.lon,
        auto_open=not args.no_open,
    )