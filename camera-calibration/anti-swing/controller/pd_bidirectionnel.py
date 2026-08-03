"""Move from the current position to a known target, with anti-sway control.

Correction du sway que dans une direction (direction du déplacement). Le sway dans l'autre direction n'est pas corrigé.
"""

import rtde_control
import rtde_receive
import numpy as np
import csv
import time

# ---------------- configuration ----------------
ROBOT_IP = "192.168.56.102"
DT       = 1.0 / 125

TARGET = np.array([0.35, -0.20, 0.40])   # target position [m], robot base frame

USE_CAMERA = False                       # True once the crate and marker are mounted

TD      = 1.45                           # measured pendulum period [s]
G       = 9.81
OMEGA   = 2 * np.pi / TD
L_CABLE = G / OMEGA ** 2
K = np.array([1.000, 2.323, -6.339, -3.001])

A_MAX, V_MAX = 0.5, 0.15
MARGE     = 0.05                         # allowed overshoot along the path [m]
DZ_MAX    = 0.01                         # max vertical component of the move [m]
THETA_MAX = np.radians(20)
DUREE     = 20.0

rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)

pose0 = np.array(rtde_r.getActualTCPPose())
delta = TARGET - pose0[:3]
DIST = float(np.linalg.norm(delta))
if DIST < 1e-3:
    raise SystemExit("Already at the target.")
DIR = delta / DIST

print(f"start  {np.round(pose0[:3], 3)}")
print(f"target {np.round(TARGET, 3)}")
print(f"distance {DIST:.3f} m   direction {np.round(DIR, 3)}")
if abs(delta[2]) > DZ_MAX:
    print(f"WARNING: vertical component {delta[2]:+.3f} m. The model assumes a "
          f"horizontal transfer at constant cable length.")

# ---------------- camera (optional) ----------------
etat = {"theta": 0.0, "theta_dot": 0.0, "t": time.perf_counter()}
if USE_CAMERA:
    import controller.camera_thread as camera_thread                  # start_camera() fills 'etat'
    camera_thread.start(etat, L_CABLE)
    print("waiting for the camera (keep the crate still)...")
    while etat.get("pret") is not True:
        time.sleep(0.1)

# ---------------- control loop ----------------
log = open(f"goto_{time.strftime('%H%M%S')}.csv", "w", newline="")
w = csv.writer(log)
w.writerow(["t", "s", "s_dot", "theta", "theta_dot", "u", "v_cmd"])

v_cmd = 0.0
t_start = time.perf_counter()
raison = "duration"

try:
    while True:
        t_cycle = rtde_c.initPeriod()
        t = time.perf_counter() - t_start
        if t > DUREE:
            break

        pose = np.array(rtde_r.getActualTCPPose()[:3])
        vit = np.array(rtde_r.getActualTCPSpeed()[:3])
        s = float(np.dot(pose - pose0[:3], DIR))     # travelled distance
        s_dot = float(np.dot(vit, DIR))              # speed along the path

        theta, theta_dot = (etat["theta"], etat["theta_dot"]) if USE_CAMERA else (0.0, 0.0)

        if not -MARGE <= s <= DIST + MARGE:
            raison = f"out of path (s = {s:.3f} m)"; break
        if USE_CAMERA:
            if abs(theta) > THETA_MAX:
                raison = f"angle {np.degrees(theta):.1f} deg"; break
            if time.perf_counter() - etat["t"] > 0.20:
                raison = "marker lost"; break

        e = np.array([s - DIST, s_dot, theta, theta_dot])
        u = float(np.clip(-K @ e, -A_MAX, A_MAX))
        v_cmd = float(np.clip(v_cmd + u * DT, -V_MAX, V_MAX))

        speed = list(v_cmd * DIR) + [0.0, 0.0, 0.0]   # no rotation: camera stays put
        rtde_c.speedL(speed, A_MAX, DT)

        w.writerow([f"{t:.4f}", f"{s:.5f}", f"{s_dot:.5f}", f"{theta:.5f}",
                    f"{theta_dot:.5f}", f"{u:.4f}", f"{v_cmd:.5f}"])
        print(f"\rt = {t:5.2f} s   s = {s:.4f} / {DIST:.3f} m   "
              f"theta = {np.degrees(theta):+6.2f} deg", end="", flush=True)

        rtde_c.waitPeriod(t_cycle)

except KeyboardInterrupt:
    raison = "keyboard"

finally:
    rtde_c.speedStop()
    log.close()
    final = np.array(rtde_r.getActualTCPPose()[:3])
    print(f"\nstopped: {raison}")
    print(f"final position {np.round(final, 4)}   error "
          f"{1000*np.linalg.norm(final - TARGET):.1f} mm")
    rtde_c.stopScript()