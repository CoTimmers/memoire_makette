"""Measure the end-to-end latency of the vision chain.

Principle. The robot is moved sinusoidally while the camera, carried by the tool,
watches a static marker. The encoders give the position with negligible delay;
the camera gives the same motion later. The lag between the two signals is the
latency to subtract from the vision timestamps.

Two independent estimates are produced:
  - cross-correlation, which needs no assumption on the waveform;
  - phase of the fitted sinusoid, which is more precise but assumes a clean tone.

The marker must stay in the field of view during the whole motion, and must not
move. Motion amplitude is small: +-1.6 cm at 0.05 m/s and 0.5 Hz.
"""

import rtde_control
import rtde_receive
import numpy as np
import cv2
import pickle
import threading
import time

ROBOT_IP    = "192.168.56.102"
AXE         = 0                 # base axis used for the motion
CAMERA_ID   = 0
ID_MARQUEUR = 8
TAILLE      = 0.157
CALIB_FILE  = "output/calibration_data.pkl"

DT     = 1.0 / 125
V_AMP  = 0.05                   # velocity amplitude [m/s]
F_TEST = 0.5                    # frequency [Hz]
DUREE  = 12.0                   # duration [s]
ACC    = 0.5

# ---------------- vision thread ----------------
vision = []                     # (timestamp, marker position along the axis)
stop = threading.Event()


def camera():
    with open(CALIB_FILE, "rb") as f:
        data = pickle.load(f)
    mtx = np.array(data.get("camera_matrix", data.get("mtx")))
    dist = np.array(data.get("distortion_coefficients", data.get("dist")))
    det = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
        cv2.aruco.DetectorParameters())
    h = TAILLE / 2
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float32)

    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)

    while not stop.is_set():
        ok, frame = cap.read()
        t = time.perf_counter()          # timestamp as the control loop sees it
        if not ok:
            continue
        corners, ids, _ = det.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if ids is None or ID_MARQUEUR not in ids.flatten():
            continue
        i = int(np.where(ids.flatten() == ID_MARQUEUR)[0][0])
        ok, rvec, tvec = cv2.solvePnP(obj, corners[i][0], mtx, dist)
        if ok:
            vision.append((t, float(tvec.flatten()[0])))
    cap.release()


threading.Thread(target=camera, daemon=True).start()
time.sleep(2.0)
if not vision:
    raise SystemExit("marker not detected, aborting.")
print(f"camera running, {len(vision)} frames in 2 s "
      f"({len(vision)/2:.0f} fps)")

# ---------------- motion ----------------
rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
robot = []                      # (timestamp, robot position along the axis)

print(f"moving: +-{V_AMP} m/s at {F_TEST} Hz for {DUREE} s")
t0 = time.perf_counter()
try:
    while True:
        t_cycle = rtde_c.initPeriod()
        t = time.perf_counter()
        if t - t0 > DUREE:
            break
        v = V_AMP * np.sin(2 * np.pi * F_TEST * (t - t0))
        vec = [0.0] * 6
        vec[AXE] = v
        rtde_c.speedL(vec, ACC, DT)
        robot.append((t, rtde_r.getActualTCPPose()[AXE]))
        rtde_c.waitPeriod(t_cycle)
finally:
    rtde_c.speedStop()
    stop.set()
    time.sleep(0.3)
    rtde_c.stopScript()

# ---------------- analysis ----------------
t_r = np.array([r[0] for r in robot])
x_r = np.array([r[1] for r in robot])
vis = [v for v in vision if t_r[0] <= v[0] <= t_r[-1]]
t_v = np.array([v[0] for v in vis])
x_v = np.array([v[1] for v in vis])
print(f"\n{len(t_r)} robot samples, {len(t_v)} vision samples")

# the camera moves with the robot, so a static marker moves the opposite way
x_r = x_r - x_r.mean()
x_v = -(x_v - x_v.mean())

# 1) cross-correlation: shift the vision timestamps back and correlate
retards = np.arange(0.0, 0.201, 0.001)
corr = []
for tau in retards:
    xi = np.interp(t_v - tau, t_r, x_r)
    corr.append(np.corrcoef(xi, x_v)[0, 1])
corr = np.array(corr)
tau_corr = retards[int(np.argmax(corr))]

# 2) phase of the fitted sinusoid on each signal
def phase(t, x, f):
    w = 2 * np.pi * f
    M = np.column_stack([np.cos(w * t), np.sin(w * t), np.ones_like(t)])
    c, *_ = np.linalg.lstsq(M, x, rcond=None)
    return np.arctan2(-c[1], c[0]), np.hypot(c[0], c[1])

ph_r, amp_r = phase(t_r, x_r, F_TEST)
ph_v, amp_v = phase(t_v, x_v, F_TEST)
dphi = (ph_v - ph_r + np.pi) % (2 * np.pi) - np.pi
tau_phase = dphi / (2 * np.pi * F_TEST)

print(f"\ncross-correlation : {1000*tau_corr:.1f} ms   (peak r = {corr.max():.4f})")
print(f"sinusoid phase    : {1000*tau_phase:.1f} ms")
print(f"amplitude ratio   : {amp_v/amp_r:.3f}  (1.0 = the camera sees the full motion)")
print(f"\nRETARD_CAM = {1000*np.mean([tau_corr, abs(tau_phase)]):.3f}e-3   "
      f"# seconds, to paste into main.py")

try:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    ax[0].plot(t_r - t_r[0], 1000 * x_r, label="robot encoders")
    ax[0].plot(t_v - t_r[0], 1000 * x_v, ".", ms=3, label="vision, raw")
    ax[0].plot(t_v - t_r[0] - tau_corr, 1000 * x_v, ".", ms=3,
               label=f"vision shifted by {1000*tau_corr:.0f} ms")
    ax[0].set_xlim(2, 6); ax[0].set_ylabel("position [mm]")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax[1].plot(1000 * retards, corr)
    ax[1].axvline(1000 * tau_corr, color="r", ls="--")
    ax[1].set_xlabel("assumed delay [ms]"); ax[1].set_ylabel("correlation")
    ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig("retard_camera.png", dpi=130)
    print("figure: retard_camera.png")
except ImportError:
    pass