"""
Severity classification based on physical measurements.

Thresholds calibrated to typical road inspection standards
(Japan Road Association / AASHTO guidelines):

    Pothole:
        LOW    : width < 15 cm
        MEDIUM : 15-30 cm
        HIGH   : > 30 cm

    Crack:
        LOW    : length < 1 m  AND  density < 5%
        MEDIUM : length 1-5 m  OR   density 5-20%
        HIGH   : length > 5 m  OR   density > 20%
"""


def pothole_severity(width_cm, depth_cm=None):
    """Return 'LOW' / 'MEDIUM' / 'HIGH'."""
    if width_cm >= 30 or (depth_cm is not None and depth_cm >= 8):
        return "HIGH"
    if width_cm >= 15 or (depth_cm is not None and depth_cm >= 3):
        return "MEDIUM"
    return "LOW"


def crack_severity(length_cm, density_pct):
    """Return 'LOW' / 'MEDIUM' / 'HIGH'."""
    if length_cm >= 500 or density_pct >= 20.0:
        return "HIGH"
    if length_cm >= 100 or density_pct >= 5.0:
        return "MEDIUM"
    return "LOW"


def overall_severity(severities):
    """Aggregate: HIGH if any HIGH, else MEDIUM if any MEDIUM, else LOW."""
    if "HIGH" in severities:
        return "HIGH"
    if "MEDIUM" in severities:
        return "MEDIUM"
    return "LOW"