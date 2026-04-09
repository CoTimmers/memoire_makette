import cv2
import numpy as np
import json

# Plage HSV calibrée sur la vraie caméra
LOWER_GREEN = np.array([44, 150, 100])
UPPER_GREEN = np.array([65, 255, 255])

with open("config/environment.json") as f:
    env = json.load(f)

METERS_PER_PIXEL = env["meters_per_pixel"]
WALL1_X_PIXEL = env["wall_configurations"]["wall_closed"]["wall1"]["top"][0]


def detect_marker(image):
    """
    Accepte soit un chemin (str) soit une frame numpy (depuis cv2.VideoCapture).
    Retourne le COM en pixels et la distance au wall 1 en mètres.
    """
    if isinstance(image, str):
        img = cv2.imread(image)
    else:
        img = image

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    c = max(contours, key=cv2.contourArea)
    M = cv2.moments(c)
    if M["m00"] == 0:
        return None

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    delta_x_m = (cx - WALL1_X_PIXEL) * METERS_PER_PIXEL

    return {
        "pixel": (cx, cy),
        "delta_x_m": round(delta_x_m, 4),
        "delta_x_cm": round(delta_x_m * 100, 2)
    }


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "green_marker.jpeg"
    result = detect_marker(path)
    if result:
        print(f"COM pixel     : {result['pixel']}")
        print(f"Delta x wall1 : {result['delta_x_cm']} cm")
    else:
        print("Marqueur non détecté")
