import cv2
import numpy as np

# Plage HSV du carré vert fluo (RGB: 52, 238, 66)
LOWER_GREEN = np.array([47, 150, 150])
UPPER_GREEN = np.array([77, 255, 255])

METERS_PER_PIXEL = 0.00121
WALL1_X_PIXEL = 482  # position x du wall 1 en pixels

def find_green_marker(image_path, draw=True):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image non trouvée: {image_path}")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Masque couleur
    mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("Aucun marqueur vert détecté")
        return None

    # Plus grand contour
    c = max(contours, key=cv2.contourArea)

    # Bounding box
    x, y, w, h = cv2.boundingRect(c)

    # Centre de masse
    M = cv2.moments(c)
    if M["m00"] == 0:
        return None

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    # Distance au wall 1 en mètres
    delta_x_px = cx - WALL1_X_PIXEL
    delta_x_m  = delta_x_px * METERS_PER_PIXEL

    if draw:
        result = img.copy()
        cv2.rectangle(result, (x, y), (x+w, y+h), (255, 255, 255), 3)
        cv2.circle(result, (cx, cy), 8, (0, 0, 255), -1)
        cv2.circle(result, (cx, cy), 14, (255, 255, 255), 2)
        cv2.putText(result, f"COM ({cx},{cy})", (cx+16, cy-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(result, f"dx={delta_x_m*100:.1f}cm to wall1", (cx+16, cy+16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        cv2.imwrite('detection_result.jpg', result)

    return {
        "pixel": (cx, cy),
        "delta_x_px": delta_x_px,
        "delta_x_m": round(delta_x_m, 4),
        "delta_x_cm": round(delta_x_m * 100, 2)
    }

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "scene.jpg"
    result = find_green_marker(path)
    if result:
        print(f"COM pixel     : {result['pixel']}")
        print(f"Delta x       : {result['delta_x_cm']} cm vers wall 1")
        print(f"Commande grue : bouge de {result['delta_x_m']} m")
