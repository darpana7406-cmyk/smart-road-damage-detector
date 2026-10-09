"""
Convert pixel measurements to real-world cm using a simple
reference-based calibration.

Assumptions:
    - We know the road width in the image (or the width of a lane marking).
    - User provides the pixel width of a known reference + its real width in cm.
    - Same scale applies to all defects in the image (valid for flat roads).
"""
import numpy as np


def compute_cm_per_pixel(reference_width_px, reference_width_cm):
    """
    Given a reference object of known width in the image,
    return the pixel-to-cm ratio.
    """
    if reference_width_px <= 0:
        raise ValueError("reference_width_px must be > 0")
    return reference_width_cm / reference_width_px


def pothole_area_cm2(bbox_px, cm_per_px):
    """
    Approximate pothole area from bounding box (ellipse assumption).
    bbox_px = (x1, y1, x2, y2)
    """
    w_px = bbox_px[2] - bbox_px[0]
    h_px = bbox_px[3] - bbox_px[1]
    w_cm = w_px * cm_per_px
    h_cm = h_px * cm_per_px
    # Ellipse area
    return np.pi * (w_cm / 2) * (h_cm / 2)


def crack_length_cm(bbox_px, cm_per_px, crack_type):
    """
    Approximate crack length.
    Longitudinal cracks run along the taller dimension.
    Transverse cracks run along the wider dimension.
    Alligator cracks: use bounding box diagonal as proxy.
    """
    w_px = bbox_px[2] - bbox_px[0]
    h_px = bbox_px[3] - bbox_px[1]

    if crack_type == "longitudinal":
        length_px = max(w_px, h_px)
    elif crack_type == "transverse":
        length_px = max(w_px, h_px)
    elif crack_type == "alligator":
        length_px = np.sqrt(w_px**2 + h_px**2) * 1.5  # network factor
    else:
        length_px = max(w_px, h_px)

    return length_px * cm_per_px


def crack_density_pct(total_crack_area_px, road_area_px):
    """
    Crack density = total crack pixel area / total road area * 100.
    """
    if road_area_px <= 0:
        return 0.0
    return (total_crack_area_px / road_area_px) * 100