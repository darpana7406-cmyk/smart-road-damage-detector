"""
Read GPS coordinates from a NEO-6M / NEO-M8N module connected via USB.

Fallback: if no GPS hardware present, return a mock location (for testing).
"""
import serial
import pynmea2


def read_gps(port="COM3", baudrate=9600, timeout=5):
    """
    Try to read NMEA data from GPS module.
    Returns dict with lat, lon. On failure, returns None.
    """
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
        for _ in range(20):
            line = ser.readline().decode("ascii", errors="ignore").strip()
            if line.startswith("$GPGGA") or line.startswith("$GNGGA"):
                msg = pynmea2.parse(line)
                if msg.latitude and msg.longitude:
                    ser.close()
                    return {
                        "lat": msg.latitude,
                        "lon": msg.longitude,
                        "alt": msg.altitude,
                    }
        ser.close()
    except Exception as e:
        print(f"[GPS] Not available: {e}")
    return None


def mock_gps():
    """Return a mock GPS location (Pune, India) for testing."""
    return {"lat": 18.5204, "lon": 73.8567, "alt": 560.0}