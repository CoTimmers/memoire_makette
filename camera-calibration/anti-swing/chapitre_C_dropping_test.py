"""
ArUco drop test: tracks the crate COM pose (height + roll/pitch/yaw) during a drop.

Workflow:
  - Press SPACE  -> the console asks for a trial name
                 -> countdown (COUNTDOWN_S) to get ready
                 -> automatic recording for RECORD_DURATION_S seconds
                 -> CSV written to logs/<name>.csv
  - Press 't'    -> tare: zero the current height and angles (optional baseline)
  - Press 'c'    -> clear the tare
  - Press 'q'    -> quit

No velocity is computed. Only the marker/COM axes are drawn on the camera image.
Loads camera calibration from output/calibration_data.pkl
"""

import cv2
import numpy as np
import pickle
import time
import csv
import os

# --- Configuration (identical to the tracking script) ---
CAMERA_ID = 1
MARKER_SIZE = 0.157
ARUCO_DICT = cv2.aruco.DICT_4X4_50
CALIBRATION_FILE = "output/calibration_data.pkl"

# --- Offset ArUco -> COM (marker body frame, metres) ---
# Red=X, Green=Y, Blue=Z
COM_OFFSET_X = 0.0
COM_OFFSET_Y = 0.13
COM_OFFSET_Z = 0.0

# --- Display ---
DISPLAY_SCALE = 0.5
DISPLAY_COLOR = (255, 0, 255)

# --- Recording ---
RECORD_DURATION_S = 3.0     # fixed recording duration after the countdown
COUNTDOWN_S       = 2.0     # time to get ready after typing the trial name
LOG_DIR           = "logs"
# A 3-4 cm drop lasts ~90 ms, so every frame is logged (no interval throttling).

# --- Height axis in the camera frame ---
# 0 = X, 1 = Y, 2 = Z (depth). Use 2 if the camera looks straight down at the
# crate, 1 (or 0) for a side view. The frame is rotated 90 deg CCW below, but
# that only affects the image, not the 3D camera frame.
HEIGHT_AXIS = 2
HEIGHT_SIGN = 1.0           # set to -1.0 if the height should increase upwards


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


def unwrap_deg(prev, curr):
    """Keep angles continuous across the +/-180 deg wrap."""
    if prev is None:
        return curr
    return prev + (curr - prev + 180.0) % 360.0 - 180.0


def marker_pose(corners, camera_matrix, dist_coeffs):
    """solvePnP on a single marker -> (rvec, tvec) as (1,3) arrays."""
    obj_points = np.array([
        [-MARKER_SIZE/2,  MARKER_SIZE/2, 0],
        [ MARKER_SIZE/2,  MARKER_SIZE/2, 0],
        [ MARKER_SIZE/2, -MARKER_SIZE/2, 0],
        [-MARKER_SIZE/2, -MARKER_SIZE/2, 0]
    ], dtype=np.float32)
    _, rvec, tvec = cv2.solvePnP(obj_points, corners, camera_matrix, dist_coeffs)
    return rvec.reshape(1, 3), tvec.reshape(1, 3)


def com_position(rvec, tvec_marker):
    """Apply the marker -> COM offset expressed in the marker body frame."""
    if COM_OFFSET_X == 0.0 and COM_OFFSET_Y == 0.0 and COM_OFFSET_Z == 0.0:
        return tvec_marker
    R_marker, _ = cv2.Rodrigues(rvec)
    offset_body = np.array([[COM_OFFSET_X], [COM_OFFSET_Y], [COM_OFFSET_Z]])
    return tvec_marker + (R_marker @ offset_body).T


def ask_trial_name():
    """Blocking console prompt. Empty input -> timestamped default name."""
    default = f"drop_{time.strftime('%Y%m%d_%H%M%S')}"
    try:
        name = input(f"\n[CSV] Trial name (Enter for '{default}'): ").strip()
    except EOFError:
        name = ""
    if not name:
        name = default
    # Sanitise
    name = "".join(ch for ch in name if ch.isalnum() or ch in "-_. ").strip()
    if not name:
        name = default
    if not name.lower().endswith(".csv"):
        name += ".csv"
    return os.path.join(LOG_DIR, name)


def write_csv(path, samples):
    fields = ["t", "height", "roll_deg", "pitch_deg", "yaw_deg",
              "x_cam", "y_cam", "z_cam", "marker_id"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(samples)


def main():
    camera_matrix, dist_coeffs = load_calibration(CALIBRATION_FILE)
    print("Camera matrix loaded:")
    print(camera_matrix)
    print("Distortion coefficients:", dist_coeffs.flatten())

    aruco_dict   = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    aruco_params = cv2.aruco.DetectorParameters()
    detector     = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    os.makedirs(LOG_DIR, exist_ok=True)

    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  10000)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 10000)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {actual_w} x {actual_h}")
    print(f"Camera FPS (reported): {cap.get(cv2.CAP_PROP_FPS):.1f}")
    print("Press SPACE = new trial | 't' = tare | 'c' = clear tare | 'q' = quit")

    # --- Recording state ---
    state        = "idle"        # idle | countdown | recording
    samples      = []
    csv_path     = None
    phase_start  = 0.0

    # --- Tare (optional baseline subtracted from height and angles) ---
    tare = None                  # np.array([height, roll, pitch, yaw]) or None

    prev_euler = None            # for unwrapping

    while True:
        ret, frame = cap.read()
        if not ret:
            print("\nCamera read failed.")
            break
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        t_now = time.perf_counter()
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        current = None           # (marker_id, height, roll, pitch, yaw, pos_cam)

        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                rvec, tvec_marker = marker_pose(corners[i][0], camera_matrix, dist_coeffs)
                tvec_com = com_position(rvec, tvec_marker)

                # Axes only: marker frame + COM frame
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs,
                                  rvec, tvec_marker, MARKER_SIZE * 0.5)
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs,
                                  rvec, tvec_com, MARKER_SIZE * 0.35)

                if current is None:      # track the first detected marker
                    euler = rotation_vector_to_euler(rvec)
                    if prev_euler is not None:
                        euler = np.array([unwrap_deg(prev_euler[k], euler[k]) for k in range(3)])
                    prev_euler = euler

                    pos_cam = tvec_com.flatten()
                    height  = HEIGHT_SIGN * pos_cam[HEIGHT_AXIS]
                    current = (int(marker_id), height, euler[0], euler[1], euler[2], pos_cam)

        # --- Apply tare ---
        if current is not None:
            mid, height, roll, pitch, yaw, pos_cam = current
            if tare is not None:
                height -= tare[0]
                roll   -= tare[1]
                pitch  -= tare[2]
                yaw    -= tare[3]

        # --- State machine ---
        if state == "countdown":
            remaining = COUNTDOWN_S - (t_now - phase_start)
            if remaining <= 0:
                state       = "recording"
                phase_start = t_now
                samples.clear()
                print(f"[REC] Recording {RECORD_DURATION_S:.1f} s -> {csv_path}")
            else:
                cv2.putText(frame, f"{remaining:.1f}", (40, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 0), 10, cv2.LINE_AA)
                cv2.putText(frame, f"{remaining:.1f}", (40, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 3.0, DISPLAY_COLOR, 5, cv2.LINE_AA)

        elif state == "recording":
            elapsed = t_now - phase_start
            if current is not None:
                samples.append({
                    "t":         round(elapsed, 5),
                    "height":    round(height, 5),
                    "roll_deg":  round(roll, 3),
                    "pitch_deg": round(pitch, 3),
                    "yaw_deg":   round(yaw, 3),
                    "x_cam":     round(pos_cam[0], 5),
                    "y_cam":     round(pos_cam[1], 5),
                    "z_cam":     round(pos_cam[2], 5),
                    "marker_id": mid,
                })
            if elapsed >= RECORD_DURATION_S:
                state = "idle"
                if samples:
                    write_csv(csv_path, samples)
                    rate = len(samples) / RECORD_DURATION_S
                    print(f"[CSV] {len(samples)} samples ({rate:.1f} Hz) -> {csv_path}")
                else:
                    print("[CSV] No marker detected during the trial, nothing written.")
            else:
                h_f, w_f = frame.shape[:2]
                cv2.circle(frame, (w_f - 30, 30), 14, (0, 0, 200), -1)
                cv2.putText(frame, f"REC {elapsed:.2f}s", (w_f - 210, 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2, cv2.LINE_AA)

        # --- Console readout ---
        if current is not None:
            print(f"\rh={height:+.4f} m  roll={roll:+7.2f}  pitch={pitch:+7.2f}  yaw={yaw:+7.2f} deg"
                  + ("  [TARE]" if tare is not None else "")
                  + (f"  [{state.upper()}]" if state != "idle" else "   "),
                  end="", flush=True)

        # --- Display ---
        if DISPLAY_SCALE != 1.0:
            h, w = frame.shape[:2]
            display = cv2.resize(frame, (int(w * DISPLAY_SCALE), int(h * DISPLAY_SCALE)))
        else:
            display = frame
        cv2.imshow("ArUco Drop Test", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            if state == "idle":
                csv_path    = ask_trial_name()
                state       = "countdown"
                phase_start = time.perf_counter()
                # Flush stale frames buffered while the console prompt was blocking
                for _ in range(5):
                    cap.read()
            else:
                print("\n[REC] Trial already in progress.")
        elif key == ord('t'):
            if current is not None:
                base = np.array([height, roll, pitch, yaw])
                tare = base if tare is None else tare + base
                print(f"\n[TARE] Baseline set: h={base[0]:.4f} m, "
                      f"roll={base[1]:.2f}, pitch={base[2]:.2f}, yaw={base[3]:.2f} deg")
            else:
                print("\n[TARE] No marker detected.")
        elif key == ord('c'):
            tare = None
            print("\n[TARE] Cleared.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()