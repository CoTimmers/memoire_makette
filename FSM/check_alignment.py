import cv2
import numpy as np
import json
import math

with open("config/environment.json") as f:
    env = json.load(f)

METERS_PER_PIXEL  = env["meters_per_pixel"]
PIXELS_PER_METER  = env["pixels_per_meter"]
WALL1_X           = env["wall_configurations"]["wall_closed"]["wall1"]["top"][0]

LOWER_RED1 = np.array([0,   100, 80])
UPPER_RED1 = np.array([10,  255, 255])
LOWER_RED2 = np.array([170, 100, 80])
UPPER_RED2 = np.array([180, 255, 255])

ALIGNMENT_THRESHOLD_M  = 0.01
ALIGNMENT_THRESHOLD_PX = int(ALIGNMENT_THRESHOLD_M * PIXELS_PER_METER)
LONG_SIDE_THRESHOLD_CM = 35.0


def detect_crate_corners(image):
    if isinstance(image, str):
        img = cv2.imread(image)
    else:
        img = image.copy()

    hsv   = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, LOWER_RED1, UPPER_RED1)
    mask2 = cv2.inRange(hsv, LOWER_RED2, UPPER_RED2)
    mask  = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((9, 9), np.uint8)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("Bac rouge non détecté")

    c    = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    box  = np.int32(cv2.boxPoints(rect))
    return box


def get_wall1_side_corners(corners):
    sorted_by_x = sorted(corners, key=lambda p: p[0])
    return sorted_by_x[:2]


def is_aligned(image, draw=False):
    """
    Retourne :
      aligned  (bool)   — les 2 coins sont à < 1cm de wall 1
      d1, d2   (float)  — distances en cm des 2 coins à wall 1
      side     (str)    — "LONG" si côté long aligné, "SHORT" si côté court
      side_cm  (float)  — longueur du côté aligné en cm
    """
    corners       = detect_crate_corners(image)
    wall1_corners = get_wall1_side_corners(corners)

    c1 = wall1_corners[0]
    c2 = wall1_corners[1]

    # Distance de chaque coin à wall 1
    distances_px = [abs(pt[0] - WALL1_X) for pt in wall1_corners]
    distances_cm = [d * METERS_PER_PIXEL * 100 for d in distances_px]

    aligned = all(d_px <= ALIGNMENT_THRESHOLD_PX for d_px in distances_px)

    # Distance entre les 2 coins (longueur du côté aligné)
    side_px = math.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)
    side_cm = side_px * METERS_PER_PIXEL * 100
    side    = "LONG" if side_cm >= LONG_SIDE_THRESHOLD_CM else "SHORT"

    if draw:
        if isinstance(image, str):
            img = cv2.imread(image)
        else:
            img = image.copy()

        # Tous les coins
        cv2.drawContours(img, [corners], 0, (0, 255, 255), 2)
        for pt in corners:
            cv2.circle(img, tuple(pt), 6, (0, 255, 255), -1)

        # Wall 1
        wall1_top    = env["wall_configurations"]["wall_closed"]["wall1"]["top"]
        wall1_bottom = env["wall_configurations"]["wall_closed"]["wall1"]["bottom"]
        cv2.line(img, tuple(wall1_top), tuple(wall1_bottom), (255, 0, 0), 2)

        # 2 coins côté wall 1
        for i, pt in enumerate([c1, c2]):
            color = (0, 255, 0) if distances_px[i] <= ALIGNMENT_THRESHOLD_PX else (0, 0, 255)
            cv2.circle(img, tuple(pt), 10, color, -1)
            cv2.putText(img, f"{distances_cm[i]:.1f}cm",
                        (pt[0]+12, pt[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Ligne entre les 2 coins avec longueur
        cv2.line(img, tuple(c1), tuple(c2), (255, 255, 0), 2)
        mid = ((c1[0]+c2[0])//2, (c1[1]+c2[1])//2)
        cv2.putText(img, f"{side_cm:.1f}cm ({side})",
                    (mid[0]+12, mid[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Statut global
        status_text = f"{'ALIGNED' if aligned else 'NOT ALIGNED'} - {side} SIDE"
        color = (0, 255, 0) if aligned else (0, 0, 255)
        cv2.putText(img, status_text, (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

        cv2.imwrite("alignment_result.jpg", img)

    return aligned, distances_cm[0], distances_cm[1], side, side_cm


if __name__ == "__main__":
    import sys
    image_path = sys.argv[1] if len(sys.argv) > 1 else "green_marker.jpeg"

    aligned, d1, d2, side, side_cm = is_aligned(image_path, draw=True)

    print(f"Coin 1 distance wall 1 : {d1:.1f} cm")
    print(f"Coin 2 distance wall 1 : {d2:.1f} cm")
    print(f"Longueur côté aligné   : {side_cm:.1f} cm → {side}")
    print(f"Aligné                 : {'OUI' if aligned else 'NON'}")
    if aligned:
        if side == "LONG":
            print("Action suivante        : FINAL POSITIONING")
        else:
            print("Action suivante        : CORNERING")


# ── Alignement wall 2 ─────────────────────────────────────────────

WALL2_LEFT  = env["wall_configurations"]["wall_closed"]["wall2"]["left"]
WALL2_RIGHT = env["wall_configurations"]["wall_closed"]["wall2"]["right"]
WALL2_Y     = int((WALL2_LEFT[1] + WALL2_RIGHT[1]) / 2)


def get_wall2_side_corners(corners):
    """
    Retourne les 2 coins les plus proches de wall 2
    (ceux avec le y le plus petit = côté haut du bac).
    """
    sorted_by_y = sorted(corners, key=lambda p: p[1])
    return sorted_by_y[:2]


def is_aligned_wall2(image, draw=False):
    """
    Vérifie si le côté haut du bac est aligné avec wall 2.
    Retourne (aligned, d1_cm, d2_cm)
    """
    corners       = detect_crate_corners(image)
    wall2_corners = get_wall2_side_corners(corners)

    c1 = wall2_corners[0]
    c2 = wall2_corners[1]

    distances_px = [abs(pt[1] - WALL2_Y) for pt in wall2_corners]
    distances_cm = [d * METERS_PER_PIXEL * 100 for d in distances_px]

    aligned = all(d_px <= ALIGNMENT_THRESHOLD_PX for d_px in distances_px)

    if draw:
        if isinstance(image, str):
            img = cv2.imread(image)
        else:
            img = image.copy()

        cv2.drawContours(img, [corners], 0, (0, 255, 255), 2)

        # Wall 2
        cv2.line(img, tuple(WALL2_LEFT), tuple(WALL2_RIGHT), (255, 128, 0), 2)

        # 2 coins côté wall 2
        for i, pt in enumerate([c1, c2]):
            color = (0, 255, 0) if distances_px[i] <= ALIGNMENT_THRESHOLD_PX else (0, 0, 255)
            cv2.circle(img, tuple(pt), 10, color, -1)
            cv2.putText(img, f"{distances_cm[i]:.1f}cm",
                        (pt[0]+12, pt[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        status = "ALIGNED wall2" if aligned else "NOT ALIGNED wall2"
        color  = (0, 255, 0) if aligned else (0, 0, 255)
        cv2.putText(img, status, (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

        cv2.imwrite("alignment_wall2_result.jpg", img)

    return aligned, distances_cm[0], distances_cm[1]
