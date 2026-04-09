import cv2
import numpy as np
import json

# Charger la config environnement
with open("config/environment.json") as f:
    env = json.load(f)

METERS_PER_PIXEL = env["meters_per_pixel"]
PIXELS_PER_METER = env["pixels_per_meter"]

# Paramètres de trajectoire
V_MAX = 0.4    # m/s
A_MAX = 0.089  # m/s²

# Couleur marqueur vert fluo
LOWER_GREEN = np.array([44, 150, 100])
UPPER_GREEN = np.array([65, 255, 255])




APPROACH_PIXEL = tuple(env["wall_configurations"]["wall_closed"]["approach_point"]["pixel"])

print(f"Point d'approche calculé : {APPROACH_PIXEL} px")


def detect_marker(image):
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
        raise RuntimeError("Marqueur vert non détecté")

    c = max(contours, key=cv2.contourArea)
    M = cv2.moments(c)
    if M["m00"] == 0:
        raise RuntimeError("Impossible de calculer le centre")

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)


def compute_command(current_pixel, target_pixel):
    dx_m = (target_pixel[0] - current_pixel[0]) * METERS_PER_PIXEL
    dy_m = (target_pixel[1] - current_pixel[1]) * METERS_PER_PIXEL
    return dx_m, dy_m


def send_command_to_crane(dx_m, dy_m):
    command = [round(dx_m, 4), round(dy_m, 4)]
    print(f"  --> Commande grue : {command} m")
    print(f"      v_max={V_MAX} m/s  a_max={A_MAX} m/s²")
    # crane.send(command)  # décommenter quand le contrôleur est connecté


def draw_result(image_path, current_pixel, target_pixel, dx_m, dy_m):
    img = cv2.imread(image_path)
    cv2.circle(img, current_pixel, 8, (0, 0, 255), -1)
    cv2.circle(img, current_pixel, 14, (255, 255, 255), 2)
    cv2.putText(img, f"COM {current_pixel}",
                (current_pixel[0]+16, current_pixel[1]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.circle(img, target_pixel, 8, (0, 255, 255), -1)
    cv2.circle(img, target_pixel, 14, (255, 255, 0), 2)
    cv2.putText(img, f"Approach {target_pixel}",
                (target_pixel[0]+16, target_pixel[1]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.arrowedLine(img, current_pixel, target_pixel, (0, 200, 255), 2, tipLength=0.03)
    cv2.putText(img, f"dx={dx_m*100:.1f}cm  dy={dy_m*100:.1f}cm",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(img, f"v_max={V_MAX}m/s  a_max={A_MAX}m/s2",
                (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2)
    cv2.imwrite("move_result.jpg", img)


if __name__ == "__main__":
    import sys
    image_path = sys.argv[1] if len(sys.argv) > 1 else "green_marker.jpeg"

    print(f"1. Détection du marqueur vert sur '{image_path}'...")
    current = detect_marker(image_path)
    print(f"   COM détecté : {current} px")

    print(f"2. Calcul du déplacement vers {APPROACH_PIXEL} px...")
    dx_m, dy_m = compute_command(current, APPROACH_PIXEL)
    print(f"   dx = {dx_m*100:.2f} cm  |  dy = {dy_m*100:.2f} cm")

    print(f"3. Sending command...")
    send_command_to_crane(dx_m, dy_m)

    draw_result(image_path, current, APPROACH_PIXEL, dx_m, dy_m)
    print(f"4. Résultat sauvegardé dans 'move_result.jpg'")
