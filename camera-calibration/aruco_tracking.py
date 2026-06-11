"""
ArUco marker tracking: position, orientation, velocity, angular velocity.
Loads camera calibration from output/calibration_data.pkl
"""

import cv2
import numpy as np
import pickle
import time
import csv
import os
from collections import deque
from crate_state import CrateStateMachine, MODE_COLORS

# --- Configuration ---
CAMERA_ID = 1
MARKER_SIZE = 0.157         # physical size of the ArUco marker in meters
ARUCO_DICT = cv2.aruco.DICT_4X4_50
CALIBRATION_FILE = "output/calibration_data.pkl"
SMOOTHING_WINDOW = 5        # number of frames for velocity smoothing

# --- Offset ArUco -> COM du bac (dans le repere du marqueur, tourne avec le bac) ---
# Workflow :
#   1. Poser l'ArUco exactement au coin (origine {W}), appuyer sur 'r'
#      → le centre de l'ArUco devient (0,0,0), les axes REF restent affichés
#   2. Déplacer l'ArUco et le poser sur le bac
#      → le COM est calculé automatiquement : pos_COM = pos_ArUco + R_ArUco @ offset
# Rouge X, Vert Y, Bleu Z
COM_OFFSET_X = 0.0   # (m)
COM_OFFSET_Y = 0.13  # 13 cm dans la direction Y (verte) du marqueur
COM_OFFSET_Z = 0.0   # (m)

# --- Affichage ---
DISPLAY_SCALE = 0.5  # reduit la fenetre d'affichage (1.0 = taille originale)

# --- Logging ---
LOG_INTERVAL_MS = 100   # intervalle d'enregistrement en millisecondes (100ms = 10 Hz)

# --- Load calibration ---
def load_calibration(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        camera_matrix = np.array(data.get("camera_matrix") if data.get("camera_matrix") is not None else data.get("mtx"))
        dist_coeffs   = np.array(data.get("distortion_coefficients") if data.get("distortion_coefficients") is not None else data.get("dist"))
    else:
        camera_matrix = np.array(data.camera_matrix)
        dist_coeffs   = np.array(data.dist_coeffs)
    return camera_matrix, dist_coeffs


def rotation_vector_to_euler(rvec):
    """Convert rotation vector to Euler angles (roll, pitch, yaw) in degrees."""
    R, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        roll  = np.degrees(np.arctan2( R[2, 1], R[2, 2]))
        pitch = np.degrees(np.arctan2(-R[2, 0], sy))
        yaw   = np.degrees(np.arctan2( R[1, 0], R[0, 0]))
    else:
        roll  = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
        pitch = np.degrees(np.arctan2(-R[2, 0], sy))
        yaw   = 0.0
    return np.array([roll, pitch, yaw])


def angular_velocity_from_rvecs(rvec1, rvec2, dt):
    """
    Estimate angular velocity (rad/s) in body frame from two rotation vectors.
    omega = (R1^T * R2 - I) / dt expressed as the skew-symmetric part.
    """
    R1, _ = cv2.Rodrigues(rvec1)
    R2, _ = cv2.Rodrigues(rvec2)
    dR = R1.T @ R2
    # Extract angular velocity from dR using Rodrigues log
    dr, _ = cv2.Rodrigues(dR)
    return dr.flatten() / dt  # rad/s


class MarkerTracker:
    def __init__(self, smoothing=SMOOTHING_WINDOW):
        self.positions  = deque(maxlen=smoothing + 1)   # (t, tvec)
        self.rotations  = deque(maxlen=smoothing + 1)   # (t, rvec)

    def update(self, t, tvec, rvec):
        self.positions.append((t, tvec.flatten()))
        self.rotations.append((t, rvec.flatten()))

    def velocity(self):
        """Linear velocity in m/s (finite difference over available window)."""
        if len(self.positions) < 2:
            return None
        t0, p0 = self.positions[0]
        t1, p1 = self.positions[-1]
        dt = t1 - t0
        if dt < 1e-6:
            return None
        return (p1 - p0) / dt

    def angular_velocity(self):
        """Angular velocity in rad/s."""
        if len(self.rotations) < 2:
            return None
        t0, r0 = self.rotations[0]
        t1, r1 = self.rotations[-1]
        dt = t1 - t0
        if dt < 1e-6:
            return None
        return angular_velocity_from_rvecs(r0, r1, dt)


def relative_pose(tvec, rvec, ref_tvec, ref_rvec):
    """Calcule position et orientation relatives à la référence.
    La position est exprimée dans le repère de référence (axes rouge/vert/bleu).
    """
    R_ref,  _ = cv2.Rodrigues(ref_rvec)
    R_curr, _ = cv2.Rodrigues(rvec)

    # Vecteur difference dans le repere camera, puis projete dans le repere ref
    pos_cam = tvec.flatten() - ref_tvec.flatten()
    pos_rel = R_ref.T @ pos_cam   # ← maintenant aligné avec X(rouge) Y(vert) Z(bleu)

    R_rel = R_ref.T @ R_curr
    rvec_rel, _ = cv2.Rodrigues(R_rel)
    euler_rel = rotation_vector_to_euler(rvec_rel)
    return pos_rel, euler_rel


def draw_overlay(frame, marker_id, tvec, rvec, vel, ang_vel, euler,
                 ref_tvec=None, ref_rvec=None):
    t = tvec.flatten()
    e = euler

    has_ref = ref_tvec is not None and ref_rvec is not None
    if has_ref:
        pos_rel, euler_rel = relative_pose(tvec, rvec, ref_tvec, ref_rvec)
        pr, er = pos_rel, euler_rel
        text_lines = [
            f"ID: {marker_id}   [ref actif]",
            f"Pos abs  x={t[0]:.3f}  y={t[1]:.3f}  z={t[2]:.3f} m",
            f"Pos rel  x={pr[0]:.3f}  y={pr[1]:.3f}  z={pr[2]:.3f} m",
            f"Euler abs  R={e[0]:.1f}  P={e[1]:.1f}  Y={e[2]:.1f} deg",
            f"Euler rel  R={er[0]:.1f}  P={er[1]:.1f}  Y={er[2]:.1f} deg",
        ]
    else:
        text_lines = [
            f"ID: {marker_id}   [appuie 'r' pour fixer ref]",
            f"Pos  x={t[0]:.3f}  y={t[1]:.3f}  z={t[2]:.3f} m",
            f"Euler  R={e[0]:.1f}  P={e[1]:.1f}  Y={e[2]:.1f} deg",
        ]

    if vel is not None:
        text_lines.append(f"Vel  x={vel[0]:.3f}  y={vel[1]:.3f}  z={vel[2]:.3f} m/s")
        text_lines.append(f"|v|={np.linalg.norm(vel):.3f} m/s")
    if ang_vel is not None:
        text_lines.append(f"AngVel  x={ang_vel[0]:.2f}  y={ang_vel[1]:.2f}  z={ang_vel[2]:.2f} rad/s")
        text_lines.append(f"|w|={np.linalg.norm(ang_vel):.2f} rad/s")

    x0, y0 = 10, 25
    for i, line in enumerate(text_lines):
        cv2.putText(frame, line, (x0, y0 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (x0, y0 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)


def draw_ref_markers(frame, references, ref_pixels, camera_matrix, dist_coeffs):
    """Dessine les axes X/Y/Z du repere de reference sauvegarde."""
    for marker_id, (ref_tvec, ref_rvec) in references.items():
        # Axes 3D (X=rouge, Y=vert, Z=bleu) — meme taille que le marqueur
        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs,
                          ref_rvec, ref_tvec, MARKER_SIZE * 0.7)
        # Label a cote
        if marker_id in ref_pixels:
            px = ref_pixels[marker_id]
            cx, cy = int(px[0]), int(px[1])
            label = f"REF {marker_id}"
            cv2.putText(frame, label, (cx + 20, cy - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, label, (cx + 20, cy - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)


def draw_state_overlay(frame, status_lines, color):
    """Dessine le mode juste sous les infos de pose (position fixe)."""
    n = len(status_lines)
    y_start = 220        # juste sous le bloc pose
    step    = 30
    box_h   = n * step + 16
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y_start - 8), (700, y_start + box_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    for i, line in enumerate(status_lines):
        y = y_start + i * step + 22
        cv2.putText(frame, line, (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)


def main():
    camera_matrix, dist_coeffs = load_calibration(CALIBRATION_FILE)
    print("Camera matrix loaded:")
    print(camera_matrix)
    print("Distortion coefficients:", dist_coeffs.flatten())

    aruco_dict   = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    aruco_params = cv2.aruco.DetectorParameters()
    detector     = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    trackers       = {}  # marker_id -> MarkerTracker
    references     = {}  # marker_id -> (ref_tvec, ref_rvec)
    ref_pixels     = {}  # marker_id -> (cx, cy) pixel position de la ref
    state_machines = {}  # marker_id -> CrateStateMachine
    last_status    = (["Mode: UNKNOWN", "Appuie 'r' pour fixer ref"], (128, 128, 128))

    # --- Logger CSV ---
    logging_active   = False
    log_file         = None
    log_writer       = None
    last_log_time    = 0.0   # timestamp du dernier enregistrement
    LOG_DIR          = "logs"
    os.makedirs(LOG_DIR, exist_ok=True)

    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  10000)  # demande le max, la camera prendra ce qu'elle peut
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 10000)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Resolution camera : {actual_w} x {actual_h}")

    print(f"Press 'r' = fixer ref | 'c' = effacer | 'n' = reset FSM | 'l' = log ({LOG_INTERVAL_MS}ms) | 'q' = quitter")

    while True:
        ret, frame = cap.read()
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if not ret:
            print("Camera read failed.")
            break

        t_now = time.perf_counter()
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            for i, marker_id in enumerate(ids.flatten()):
                c = corners[i]

                # Pose estimation
                obj_points = np.array([
                    [-MARKER_SIZE/2,  MARKER_SIZE/2, 0],
                    [ MARKER_SIZE/2,  MARKER_SIZE/2, 0],
                    [ MARKER_SIZE/2, -MARKER_SIZE/2, 0],
                    [-MARKER_SIZE/2, -MARKER_SIZE/2, 0]
                ], dtype=np.float32)

                success, rvec, tvec_marker = cv2.solvePnP(
                    obj_points, c[0], camera_matrix, dist_coeffs
                )
                rvec        = rvec.reshape(1, 3)
                tvec_marker = tvec_marker.reshape(1, 3)

                # tvec_com = position du COM (avec offset dans le repere du marqueur)
                if COM_OFFSET_X != 0.0 or COM_OFFSET_Y != 0.0 or COM_OFFSET_Z != 0.0:
                    R_marker, _ = cv2.Rodrigues(rvec)
                    com_offset_body = np.array([[COM_OFFSET_X],
                                                [COM_OFFSET_Y],
                                                [COM_OFFSET_Z]])
                    tvec_com = tvec_marker + (R_marker @ com_offset_body).T
                else:
                    tvec_com = tvec_marker

                # Axes dessines sur le marqueur (pas sur le COM)
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs,
                                  rvec, tvec_marker, MARKER_SIZE * 0.5)

                # tvec utilise pour tout le reste = COM
                tvec = tvec_com

                # Update tracker
                if marker_id not in trackers:
                    trackers[marker_id] = MarkerTracker()
                trackers[marker_id].update(t_now, tvec, rvec)

                vel     = trackers[marker_id].velocity()
                ang_vel = trackers[marker_id].angular_velocity()
                euler   = rotation_vector_to_euler(rvec)

                ref_tvec, ref_rvec = references.get(marker_id, (None, None))
                draw_overlay(frame, marker_id, tvec, rvec, vel, ang_vel, euler,
                             ref_tvec, ref_rvec)

                # State machine update (only when reference is set)
                if marker_id not in state_machines:
                    state_machines[marker_id] = CrateStateMachine()

                if ref_tvec is not None:
                    pos_rel, euler_rel = relative_pose(tvec, rvec, ref_tvec, ref_rvec)
                    # Rouge(X camera)=Mur2=y_these, Vert(Y camera)=Mur1=x_these
                    x_c = -pos_rel[0]   # rouge  (Y camera) → x du mémoire (Wall 1)
                    y_c = pos_rel[1]   # vert (X camera) → y du mémoire (Wall 2)
                    theta_c  = euler_rel[2]  # yaw = rotation in plane
                    vx_c = vel[0]    if vel     is not None else 0.0
                    vy_c = vel[1]    if vel     is not None else 0.0
                    om_c = ang_vel[2] if ang_vel is not None else 0.0

                    sm = state_machines[marker_id]
                    sm.update(x_c, y_c, theta_c, vx_c, vy_c, om_c)

                    status_lines, mode_color = sm.status_lines()
                    last_status = (status_lines, mode_color)

                    # --- Log CSV (respecte l'intervalle) ---
                    elapsed_since_log = (t_now - last_log_time) * 1000  # en ms
                    if logging_active and log_writer is not None and elapsed_since_log >= LOG_INTERVAL_MS:
                        last_log_time = t_now
                        log_writer.writerow({
                            "t":              round(t_now, 4),
                            "x":              round(pos_rel[0], 4),
                            "y":              round(pos_rel[1], 4),
                            "z":              round(pos_rel[2], 4),
                            "theta_deg":      round(theta_c, 2),
                            "vx":             round(vx_c, 4),
                            "vy":             round(vy_c, 4),
                            "omega":          round(om_c, 4),
                            "mode":           sm.current_mode,
                            "can_transition": int(sm.can_transition),
                        })

                    print(f"\r[ID {marker_id}] "
                          f"rel=({pos_rel[0]:.3f}, {pos_rel[1]:.3f}, {pos_rel[2]:.3f}) m  "
                          f"yaw={theta_c:.1f}°  "
                          f"mode={sm.current_mode}"
                          + ("  ✓ TRANSITION" if sm.can_transition else "")
                          + ("  [LOG]" if logging_active else ""),
                          end="", flush=True)
                else:
                    t = tvec.flatten()
                    print(f"\r[ID {marker_id}] "
                          f"pos=({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}) m  "
                          f"euler=({euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f}) deg  "
                          + (f"|v|={np.linalg.norm(vel):.3f} m/s" if vel is not None else "")
                          + (f"  |w|={np.linalg.norm(ang_vel):.2f} rad/s" if ang_vel is not None else ""),
                          end="", flush=True)
        else:
            cv2.putText(frame, "No marker detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Dessine les axes de reference meme si le marqueur n'est plus visible
        draw_ref_markers(frame, references, ref_pixels, camera_matrix, dist_coeffs)

        # Dessine le mode toujours visible (dernier etat connu)
        draw_state_overlay(frame, last_status[0], last_status[1])

        # Indicateur LOG en haut a droite
        if logging_active:
            h_f, w_f = frame.shape[:2]
            cv2.circle(frame, (w_f - 30, 30), 14, (0, 0, 200), -1)
            cv2.putText(frame, "REC", (w_f - 80, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2, cv2.LINE_AA)

        # Redimensionne pour l'affichage uniquement
        if DISPLAY_SCALE != 1.0:
            h, w = frame.shape[:2]
            display = cv2.resize(frame, (int(w * DISPLAY_SCALE), int(h * DISPLAY_SCALE)))
        else:
            display = frame
        cv2.imshow("ArUco Tracking", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            if ids is not None:
                for i, marker_id in enumerate(ids.flatten()):
                    obj_points = np.array([
                        [-MARKER_SIZE/2,  MARKER_SIZE/2, 0],
                        [ MARKER_SIZE/2,  MARKER_SIZE/2, 0],
                        [ MARKER_SIZE/2, -MARKER_SIZE/2, 0],
                        [-MARKER_SIZE/2, -MARKER_SIZE/2, 0]
                    ], dtype=np.float32)
                    success, rvec_ref, tvec_ref = cv2.solvePnP(
                        obj_points, corners[i][0], camera_matrix, dist_coeffs
                    )
                    tvec_ref = tvec_ref.reshape(1, 3)
                    rvec_ref = rvec_ref.reshape(1, 3)
                    # On stocke le centre ArUco brut : c'est l'origine {W}
                    # Le COM offset sera applique dynamiquement a chaque frame
                    references[marker_id] = (tvec_ref.copy(), rvec_ref.copy())
                    center = corners[i][0].mean(axis=0)
                    ref_pixels[marker_id] = center
                    # Reset FSM pour repartir de zero
                    state_machines[marker_id] = CrateStateMachine()
                    print(f"\n[REF] Origine {{W}} fixée au centre ArUco — ID {marker_id}")
        elif key == ord('c'):
            references.clear()
            ref_pixels.clear()
            state_machines.clear()
            print("\n[REF] Référence effacée.")
        elif key == ord('n'):
            state_machines.clear()
            print("\n[FSM] State machine réinitialisée.")
        elif key == ord('l'):
            if not logging_active:
                fname = os.path.join(LOG_DIR, f"log_{time.strftime('%Y%m%d_%H%M%S')}.csv")
                log_file   = open(fname, "w", newline="")
                fields     = ["t", "x", "y", "z", "theta_deg", "vx", "vy", "omega", "mode", "can_transition"]
                log_writer = csv.DictWriter(log_file, fieldnames=fields)
                log_writer.writeheader()
                logging_active = True
                print(f"\n[LOG] Enregistrement démarré → {fname}")
            else:
                logging_active = False
                if log_file:
                    log_file.close()
                    log_file = None
                print("\n[LOG] Enregistrement arrêté.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()



