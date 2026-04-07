import cv2
import numpy as np
import json

# ── Charger la config environnement ──────────────────────────────
with open("environment.json") as f:
    env = json.load(f)

METERS_PER_PIXEL = env["meters_per_pixel"]

# ── Paramètres de trajectoire (depuis votre slide) ────────────────
V_MAX = 0.4          # m/s  (pendulum impact experiment)
A_MAX = 0.089        # m/s² (swing constraint, x_max=5cm, L=2.2m)

# ── Couleur du marqueur vert fluo ─────────────────────────────────
LOWER_GREEN = np.array([47, 150, 150])
UPPER_GREEN = np.array([77, 255, 255])

# ── Point d'arrivée souhaité (en pixels) ─────────────────────────
TARGET_PIXEL = (480, 367)


def detect_marker(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image non trouvée: {image_path}")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("Marqueur vert non détecté dans l'image")

    c = max(contours, key=cv2.contourArea)
    M = cv2.moments(c)
    if M["m00"] == 0:
        raise RuntimeError("Impossible de calculer le centre du marqueur")

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)


def compute_command(current_pixel, target_pixel):
    dx_px = target_pixel[0] - current_pixel[0]
    dy_px = target_pixel[1] - current_pixel[1]

    dx_m = dx_px * METERS_PER_PIXEL
    dy_m = dy_px * METERS_PER_PIXEL

    return dx_m, dy_m


def draw_result(image_path, current_pixel, target_pixel, dx_m, dy_m):
    img = cv2.imread(image_path)

    # Point actuel (COM vert)
    cv2.circle(img, current_pixel, 8, (0, 0, 255), -1)
    cv2.circle(img, current_pixel, 14, (255, 255, 255), 2)
    cv2.putText(img, f"COM actuel {current_pixel}",
                (current_pixel[0]+16, current_pixel[1]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    # Point cible
    cv2.circle(img, target_pixel, 8, (0, 255, 255), -1)
    cv2.circle(img, target_pixel, 14, (255, 255, 0), 2)
    cv2.putText(img, f"Cible {target_pixel}",
                (target_pixel[0]+16, target_pixel[1]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    # Flèche de déplacement
    cv2.arrowedLine(img, current_pixel, target_pixel, (0, 200, 255), 2, tipLength=0.03)

    # Commande
    cv2.putText(img, f"dx={dx_m*100:.1f}cm  dy={dy_m*100:.1f}cm",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(img, f"v_max={V_MAX}m/s  a_max={A_MAX}m/s2",
                (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2)

    cv2.imwrite("move_result.jpg", img)


def send_command_to_crane(dx_m, dy_m):
    """
    Remplacez cette fonction par l'appel réel à votre contrôleur.
    La grue reçoit un vecteur [dx, dy] en mètres.
    """
    command = [round(dx_m, 4), round(dy_m, 4)]
    print(f"  --> Commande envoyée à la grue : {command} m")
    print(f"      v_max = {V_MAX} m/s  |  a_max = {A_MAX} m/s²")
    # crane.send(command)  # décommentez quand le contrôleur est connecté


# ── MAIN ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    image_path = sys.argv[1] if len(sys.argv) > 1 else "scene.jpg"

    print(f"1. Détection du marqueur vert sur '{image_path}'...")
    current = detect_marker(image_path)
    print(f"   COM détecté : {current} px")

    print(f"2. Calcul du déplacement vers {TARGET_PIXEL} px...")
    dx_m, dy_m = compute_command(current, TARGET_PIXEL)
    print(f"   dx = {dx_m*100:.2f} cm  |  dy = {dy_m*100:.2f} cm")

    print(f"3. Envoi de la commande...")
    send_command_to_crane(dx_m, dy_m)

    draw_result(image_path, current, TARGET_PIXEL, dx_m, dy_m)
    print(f"4. Résultat sauvegardé dans 'move_result.jpg'")
