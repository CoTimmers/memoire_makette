"""Full anti-sway state feedback: a = -K (z - z_ref), z = [x, x_dot, theta, theta_dot].

x, x_dot come from the robot encoders (RTDE, no differentiation needed).
theta comes from the camera; theta_dot is differentiated over N_DERIV frames.
The camera runs in its own thread so the control loop keeps a steady 125 Hz.

Before the first run: set ROBOT_IP, AXE, CAMERA_ID, MARKER_SIZE, COM_OFFSET.
Sign check first: set X_REF = 0, push the crate by hand, the robot must move
towards the side the load swings to. If it moves the other way, set SIGN = -1.
"""

import rtde_control
import rtde_receive
import numpy as np
import cv2
import pickle
import threading
import csv
import time

# ---------------- configuration ----------------
ROBOT_IP = "192.168.56.102"
AXE      = 0                    # 0 = X, 1 = Y, 2 = Z
DT       = 1.0 / 125

TD      = 1.45                  # measured pendulum period [s]
G       = 9.81
OMEGA   = 2 * np.pi / TD
L_CABLE = G / OMEGA ** 2        # 0.522 m for Td = 1.45 s
SIGN    = +1                    # flip to -1 if the sign check fails

K = np.array([1.000, 2.323, -6.339, -3.001])    # from pole placement

X_REF = 0.10                    # target displacement [m], 0 for the sign check
DUREE = 15.0
A_MAX, V_MAX = 0.5, 0.15
X_MIN, X_MAX = -0.05, 0.25
THETA_MAX = np.radians(20)
PERTE_MAX = 0.20                # max time without a marker [s]

CAMERA_ID   = 1
MARKER_SIZE = 0.157             # printed marker side [m]
COM_OFFSET  = np.array([0.0, 0.13, 0.0])
CALIB_FILE  = "output/calibration_data.pkl"
N_CALIB     = 60
N_DERIV     = 3

print(f"l = {L_CABLE:.3f} m, omega = {OMEGA:.3f} rad/s, K = {K}")

# ---------------- camera thread ----------------
etat_bac = {"theta": 0.0, "theta_dot": 0.0, "t": None, "pret": False}
stop_cam = threading.Event()


def camera():
    with open(CALIB_FILE, "rb") as f:
        data = pickle.load(f)
    mtx = np.array(data.get("camera_matrix", data.get("mtx")))
    dist = np.array(data.get("distortion_coefficients", data.get("dist")))
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
        cv2.aruco.DetectorParameters())
    h = MARKER_SIZE / 2
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float32)

    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)

    pivot, calib, histo = None, [], []
    while not stop_cam.is_set():
        ok, frame = cap.read()
        if not ok:
            continue
        t = time.perf_counter()
        corners, ids, _ = detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if ids is None:
            continue
        ok, rvec, tvec = cv2.solvePnP(obj, corners[0][0], mtx, dist)
        if not ok:
            continue
        R, _ = cv2.Rodrigues(rvec)
        com = tvec.flatten() + R @ COM_OFFSET

        if pivot is None:                       # self-calibration, crate at rest
            calib.append(com)
            if len(calib) >= N_CALIB:
                pivot = np.mean(calib, axis=0) - np.array([0, 0, L_CABLE])
                print(f"\npivot calibrated: {np.round(pivot, 4)}")
            continue

        c = com - pivot
        theta = SIGN * np.arctan2(c[0], c[2])
        histo.append((t, theta))
        if len(histo) > N_DERIV + 1:
            histo.pop(0)
        theta_dot = ((theta - histo[0][1]) / (t - histo[0][0])
                     if len(histo) == N_DERIV + 1 else 0.0)
        etat_bac.update(theta=theta, theta_dot=theta_dot, t=t, pret=True)
    cap.release()


threading.Thread(target=camera, daemon=True).start()
print("waiting for the camera (keep the crate still)...")
while not etat_bac["pret"]:
    time.sleep(0.1)
print("camera ready.")

# ---------------- control loop ----------------
rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
init_q = rtde_r.getActualQ()
x0 = rtde_r.getActualTCPPose()[AXE]

log = open(f"antiswing_{time.strftime('%H%M%S')}.csv", "w", newline="")
w = csv.writer(log)
w.writerow(["t", "x", "x_dot", "theta", "theta_dot", "u", "v_cmd"])

v_cmd = 0.0
t_start = time.perf_counter()
raison = "duration"

try:
    while True:
        t_cycle = rtde_c.initPeriod()
        t = time.perf_counter() - t_start
        if t > DUREE:
            break

        x = rtde_r.getActualTCPPose()[AXE] - x0
        x_dot = rtde_r.getActualTCPSpeed()[AXE]
        theta, theta_dot = etat_bac["theta"], etat_bac["theta_dot"]

        if abs(theta) > THETA_MAX:
            raison = f"angle {np.degrees(theta):.1f} deg"; break
        if not X_MIN <= x <= X_MAX:
            raison = f"out of range x = {x:.3f} m"; break
        if time.perf_counter() - etat_bac["t"] > PERTE_MAX:
            raison = "marker lost"; break

        e = np.array([x - X_REF, x_dot, theta, theta_dot])
        u = float(np.clip(-K @ e, -A_MAX, A_MAX))
        v_cmd = float(np.clip(v_cmd + u * DT, -V_MAX, V_MAX))

        speed = [0.0] * 6
        speed[AXE] = v_cmd
        rtde_c.speedL(speed, A_MAX, DT)

        w.writerow([f"{t:.4f}", f"{x:.5f}", f"{x_dot:.5f}", f"{theta:.5f}",
                    f"{theta_dot:.5f}", f"{u:.4f}", f"{v_cmd:.5f}"])
        print(f"\rt = {t:5.2f} s   x = {x:+.4f} m   theta = {np.degrees(theta):+6.2f} deg"
              f"   u = {u:+.3f}", end="", flush=True)

        rtde_c.waitPeriod(t_cycle)

except KeyboardInterrupt:
    raison = "keyboard"

finally:
    rtde_c.speedStop()
    stop_cam.set()
    log.close()
    rtde_c.moveJ(init_q)
    rtde_c.stopScript()
    print(f"\nstopped: {raison}")