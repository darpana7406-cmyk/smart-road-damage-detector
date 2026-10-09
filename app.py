"""
Flask web server for Smart Road Damage Detector.
Features: single + batch detection, report, CSV export, PDF export, annotations editor.
Run: python app.py
Open: http://localhost:5000
"""
import os
import sys
import io
import csv
import uuid
import json
from datetime import datetime
from flask import (Flask, render_template, request, redirect, url_for,
                   send_from_directory, jsonify, Response, send_file)

# Add src to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from ultralytics import YOLO
from measure import compute_cm_per_pixel, pothole_area_cm2, crack_length_cm
from severity import pothole_severity, crack_severity, overall_severity
from gps import mock_gps, read_gps

# PDF
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Image as RLImage, Table, TableStyle)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

import cv2

app = Flask(__name__)

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_IMAGES_DIR = os.path.join(BASE_DIR, "outputs", "images")
OUTPUT_REPORTS_DIR = os.path.join(BASE_DIR, "outputs", "reports")
OUTPUT_CSV_DIR = os.path.join(BASE_DIR, "outputs", "csv")
OUTPUT_PDF_DIR = os.path.join(BASE_DIR, "outputs", "pdf")
MODEL_PATH = os.path.join(BASE_DIR, "runs", "yolov8n_rdd2022", "weights", "best.pt")

ALLOWED_EXT = {"jpg", "jpeg", "png"}

for d in [UPLOAD_DIR, OUTPUT_IMAGES_DIR, OUTPUT_REPORTS_DIR, OUTPUT_CSV_DIR, OUTPUT_PDF_DIR]:
    os.makedirs(d, exist_ok=True)

CLASS_NAMES = {
    0: "Longitudinal crack",
    1: "Transverse crack",
    2: "Alligator crack",
    3: "Pothole",
}

# Load model once at startup
print(f"Loading model from: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
print("Model loaded.")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


# ---------------- CORE PIPELINE ----------------

def analyze_image(image_path, lat=None, lon=None, ref_px=400, ref_cm=350,
                  custom_detections=None):
    """
    Run full pipeline: detect -> measure -> severity -> report.
    If custom_detections is provided, use those instead of running the model.
    """
    if custom_detections is None:
        results = model(image_path, conf=0.35, verbose=False, workers=0)
        r0 = results[0]
        img_h, img_w = r0.orig_shape

        detections = []
        for box in r0.boxes:
            cls = int(box.cls)
            conf = float(box.conf)
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            detections.append({
                "class_id": cls,
                "class_name": CLASS_NAMES.get(cls, f"class_{cls}"),
                "confidence": round(conf, 3),
                "bbox": [x1, y1, x2, y2],
            })

        annotated = r0.plot()
    else:
        img = cv2.imread(image_path)
        img_h, img_w = img.shape[:2]

        detections = []
        for d in custom_detections:
            cls = int(d["class_id"])
            detections.append({
                "class_id": cls,
                "class_name": CLASS_NAMES.get(cls, f"class_{cls}"),
                "confidence": 1.0,
                "bbox": [float(v) for v in d["bbox"]],
            })

        palette = {0: (255, 100, 0), 1: (150, 100, 255),
                   2: (200, 50, 200), 3: (0, 165, 255)}
        for d in detections:
            x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
            cls = d["class_id"]
            color = palette.get(cls, (255, 255, 255))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = CLASS_NAMES[cls]
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
            cv2.putText(img, label, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        annotated = img

    cm_per_px = compute_cm_per_pixel(ref_px, ref_cm)
    road_area_m2 = ((img_w * cm_per_px) * (img_h * cm_per_px)) / 10000.0

    potholes = []
    cracks = []

    for d in detections:
        cls = d["class_id"]
        x1, y1, x2, y2 = d["bbox"]
        conf = d["confidence"]

        if cls == 3:
            width_cm = (x2 - x1) * cm_per_px
            area_cm2 = pothole_area_cm2((x1, y1, x2, y2), cm_per_px)
            potholes.append({
                "width_cm": round(width_cm, 1),
                "area_cm2": round(area_cm2, 0),
                "severity": pothole_severity(width_cm),
                "confidence": conf,
            })
        else:
            ctype = "longitudinal" if cls == 0 else "transverse" if cls == 1 else "alligator"
            length_cm = crack_length_cm((x1, y1, x2, y2), cm_per_px, ctype)
            cracks.append({
                "type": ctype,
                "name": CLASS_NAMES[cls],
                "length_cm": round(length_cm, 1),
                "confidence": conf,
            })

    # Save annotated image
    stem = os.path.splitext(os.path.basename(image_path))[0]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"{stem}_annotated_{ts}.jpg"
    out_path = os.path.join(OUTPUT_IMAGES_DIR, out_name)
    cv2.imwrite(out_path, annotated)

    # Crack density
    total_len_cm = sum(c["length_cm"] for c in cracks)
    total_len_m = total_len_cm / 100.0
    density_pct = min((total_len_m / road_area_m2) * 100, 100.0) if road_area_m2 > 0 else 0.0

    largest = max(potholes, key=lambda p: p["width_cm"]) if potholes else None

    # GPS
    if lat is not None and lon is not None:
        gps = {"lat": lat, "lon": lon, "alt": 0.0}
    else:
        gps = mock_gps()

    # Overall severity
    all_sev = [p["severity"] for p in potholes]
    for c in cracks:
        all_sev.append(crack_severity(c["length_cm"], density_pct))
    overall = overall_severity(all_sev) if all_sev else "LOW"

    # Report text
    report_lines = [
        "=" * 40,
        "         ROAD CONDITION REPORT",
        "=" * 40,
        f"Potholes : {len(potholes):02d}",
        f"Cracks   : {len(cracks):02d}",
        "",
    ]
    if largest:
        report_lines += [
            "Largest pothole:",
            f"  Width    : {largest['width_cm']} cm",
            f"  Area     : {largest['area_cm2']} cm2",
            f"  Severity : {largest['severity']}",
            "",
        ]
    if cracks:
        report_lines += [
            "Crack metrics:",
            f"  Total length   : {total_len_cm:.1f} cm ({total_len_m:.2f} m)",
            f"  Crack density  : {density_pct:.2f}%",
            "",
        ]
    report_lines += [
        "GPS:",
        f"  {abs(gps['lat']):.4f}\u00b0 {'N' if gps['lat'] >= 0 else 'S'}",
        f"  {abs(gps['lon']):.4f}\u00b0 {'E' if gps['lon'] >= 0 else 'W'}",
        "",
        f"Total defects     : {len(potholes) + len(cracks):02d}",
        f"Overall severity  : {overall}",
        f"Timestamp         : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 40,
    ]
    report_text = "\n".join(report_lines)

    report_name = f"report_{ts}.txt"
    report_path = os.path.join(OUTPUT_REPORTS_DIR, report_name)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    return {
        "annotated_image": out_name,
        "report_text": report_text,
        "report_file": report_name,
        "original_file": os.path.basename(image_path),
        "potholes": potholes,
        "cracks": cracks,
        "largest": largest,
        "density_pct": round(density_pct, 2),
        "total_length_cm": round(total_len_cm, 1),
        "gps": gps,
        "overall_severity": overall,
        "total_defects": len(potholes) + len(cracks),
        "detections": detections,
        "image_size": [img_w, img_h],
        "timestamp": ts,
    }


# ---------------- ROUTES: SINGLE UPLOAD ----------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return "No file part", 400
    file = request.files["image"]
    if file.filename == "":
        return "No selected file", 400
    if not allowed_file(file.filename):
        return "File type not allowed. Use jpg, jpeg, or png.", 400

    ext = file.filename.rsplit(".", 1)[1].lower()
    uid = uuid.uuid4().hex[:8]
    saved_name = f"{uid}.{ext}"
    saved_path = os.path.join(UPLOAD_DIR, saved_name)
    file.save(saved_path)

    lat = request.form.get("lat", type=float)
    lon = request.form.get("lon", type=float)

    result = analyze_image(saved_path, lat=lat, lon=lon)

    return render_template("result.html", result=result, original=saved_name)


@app.route("/reanalyze", methods=["POST"])
def reanalyze():
    """Receives user-edited annotations and re-runs measurement pipeline."""
    data = request.get_json()
    original_file = data.get("original")
    detections = data.get("detections", [])
    lat = data.get("lat")
    lon = data.get("lon")

    if not original_file:
        return jsonify({"error": "Missing original file"}), 400

    original_path = os.path.join(UPLOAD_DIR, original_file)
    if not os.path.exists(original_path):
        return jsonify({"error": "Original image not found"}), 404

    result = analyze_image(original_path, lat=lat, lon=lon,
                           custom_detections=detections)
    return jsonify(result)


# ---------------- ROUTES: EXPORTS ----------------

@app.route("/exports/csv/<image_stem>")
def export_csv(image_stem):
    detections_json = request.args.get("d", "")
    if not detections_json:
        return "No detections provided", 400
    try:
        detections = json.loads(detections_json)
    except Exception:
        return "Invalid detections JSON", 400

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["#", "Class", "Confidence", "X1", "Y1", "X2", "Y2",
                     "Width_px", "Height_px"])
    for i, d in enumerate(detections, 1):
        x1, y1, x2, y2 = d["bbox"]
        writer.writerow([
            i, d["class_name"], d["confidence"],
            round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1),
            round(x2 - x1, 1), round(y2 - y1, 1),
        ])

    csv_bytes = output.getvalue().encode("utf-8")
    filename = f"detections_{image_stem}.csv"
    return Response(
        csv_bytes,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@app.route("/exports/pdf/<image_stem>")
def export_pdf(image_stem):
    detections_json = request.args.get("d", "")
    metrics_json = request.args.get("m", "")

    try:
        detections = json.loads(detections_json) if detections_json else []
        metrics = json.loads(metrics_json) if metrics_json else {}
    except Exception:
        return "Invalid JSON", 400

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=1.5 * cm, rightMargin=1.5 * cm,
                            topMargin=1.5 * cm, bottomMargin=1.5 * cm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleStyle", parent=styles["Title"],
                                  textColor=colors.HexColor("#0f172a"))
    h2_style = ParagraphStyle("H2Style", parent=styles["Heading2"],
                               textColor=colors.HexColor("#1e293b"),
                               spaceBefore=12, spaceAfter=6)

    story = []
    story.append(Paragraph("Road Condition Report", title_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                            styles["Italic"]))
    story.append(Spacer(1, 0.5 * cm))

    summary_data = [
        ["Metric", "Value"],
        ["Potholes", str(len(metrics.get("potholes", [])))],
        ["Cracks", str(len(metrics.get("cracks", [])))],
        ["Total defects", str(metrics.get("total_defects", 0))],
        ["Overall severity", metrics.get("overall_severity", "LOW")],
    ]
    if metrics.get("largest"):
        lg = metrics["largest"]
        summary_data += [
            ["Largest pothole width", f"{lg.get('width_cm')} cm"],
            ["Largest pothole area", f"{lg.get('area_cm2')} cm2"],
        ]
    if metrics.get("cracks"):
        summary_data += [
            ["Total crack length", f"{metrics.get('total_length_cm')} cm"],
            ["Crack density", f"{metrics.get('density_pct')}%"],
        ]
    if metrics.get("gps"):
        g = metrics["gps"]
        summary_data.append(["GPS",
                             f"{abs(g['lat']):.4f} {'N' if g['lat'] >= 0 else 'S'}, "
                             f"{abs(g['lon']):.4f} {'E' if g['lon'] >= 0 else 'W'}"])

    t = Table(summary_data, colWidths=[6 * cm, 9 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3b82f6")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
        ("TOPPADDING", (0, 0), (-1, 0), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f9")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.5 * cm))

    if metrics.get("annotated_image"):
        img_path = os.path.join(OUTPUT_IMAGES_DIR, metrics["annotated_image"])
        if os.path.exists(img_path):
            story.append(Paragraph("Detected Defects", h2_style))
            story.append(RLImage(img_path, width=16 * cm, height=10 * cm,
                                  kind="proportional"))
            story.append(Spacer(1, 0.4 * cm))

    if detections:
        story.append(Paragraph(f"Detection Details ({len(detections)})", h2_style))
        rows = [["#", "Class", "Confidence", "X1", "Y1", "X2", "Y2"]]
        for i, d in enumerate(detections, 1):
            x1, y1, x2, y2 = d["bbox"]
            rows.append([
                str(i), d["class_name"], f"{d['confidence']:.2f}",
                f"{x1:.0f}", f"{y1:.0f}", f"{x2:.0f}", f"{y2:.0f}"
            ])
        t2 = Table(rows, colWidths=[1 * cm, 5 * cm, 2.5 * cm,
                                     1.8 * cm, 1.8 * cm, 1.8 * cm, 1.8 * cm])
        t2.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(t2)

    doc.build(story)
    buf.seek(0)

    filename = f"report_{image_stem}.pdf"
    return send_file(buf, mimetype="application/pdf",
                     as_attachment=True, download_name=filename)


# ---------------- ROUTES: BATCH ----------------

@app.route("/batch")
def batch_page():
    return render_template("batch.html")


@app.route("/batch-upload", methods=["POST"])
def batch_upload():
    if "images" not in request.files:
        return "No files provided", 400

    files = request.files.getlist("images")
    files = [f for f in files if f.filename and allowed_file(f.filename)]

    if not files:
        return "No valid image files provided", 400
    if len(files) > 20:
        return "Maximum 20 images per batch", 400

    lat = request.form.get("lat", type=float)
    lon = request.form.get("lon", type=float)

    results = []
    for file in files:
        ext = file.filename.rsplit(".", 1)[1].lower()
        uid = uuid.uuid4().hex[:8]
        saved_name = f"{uid}.{ext}"
        saved_path = os.path.join(UPLOAD_DIR, saved_name)
        file.save(saved_path)

        try:
            result = analyze_image(saved_path, lat=lat, lon=lon)
            result["original_upload"] = saved_name
            result["original_name"] = file.filename
            results.append(result)
        except Exception as e:
            results.append({
                "original_name": file.filename,
                "error": str(e),
            })

    valid_results = [r for r in results if "error" not in r]
    total_potholes = sum(len(r.get("potholes", [])) for r in valid_results)
    total_cracks = sum(len(r.get("cracks", [])) for r in valid_results)
    severities = [r.get("overall_severity", "LOW") for r in valid_results]
    overall = overall_severity(severities) if severities else "LOW"

    summary = {
        "num_images": len(valid_results),
        "num_errors": len(results) - len(valid_results),
        "total_potholes": total_potholes,
        "total_cracks": total_cracks,
        "total_defects": total_potholes + total_cracks,
        "overall_severity": overall,
    }

    return render_template("batch_result.html",
                           results=results,
                           summary=summary,
                           gps={"lat": lat or 18.5204, "lon": lon or 73.8567})


@app.route("/exports/batch-csv")
def export_batch_csv():
    detections_json = request.args.get("d", "")
    if not detections_json:
        return "No data", 400
    try:
        data = json.loads(detections_json)
    except Exception:
        return "Invalid JSON", 400

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Image", "Detection#", "Class", "Confidence",
                     "X1", "Y1", "X2", "Y2", "Width_px", "Height_px"])

    for item in data:
        img_name = item.get("image_name", "unknown")
        for i, d in enumerate(item.get("detections", []), 1):
            x1, y1, x2, y2 = d["bbox"]
            writer.writerow([
                img_name, i, d["class_name"], d["confidence"],
                round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1),
                round(x2 - x1, 1), round(y2 - y1, 1),
            ])

    csv_bytes = output.getvalue().encode("utf-8")
    filename = f"batch_detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return Response(
        csv_bytes,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


# ---------------- STATIC SERVING ----------------

@app.route("/uploads/<filename>")
def serve_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/outputs/images/<filename>")
def serve_output_image(filename):
    return send_from_directory(OUTPUT_IMAGES_DIR, filename)


@app.route("/outputs/reports/<filename>")
def serve_report(filename):
    return send_from_directory(OUTPUT_REPORTS_DIR, filename, as_attachment=True)


# ---------------- MAIN ----------------

if __name__ == "__main__":
    print("\n===========================================")
    print(" Smart Road Damage Detector - Web Dashboard")
    print(" Open: http://localhost:5000")
    print("===========================================\n")
    app.run(host="0.0.0.0", port=5000, debug=False)