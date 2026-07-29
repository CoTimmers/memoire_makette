"""Two-axis anti-sway: the same control law applied independently on X and Y.

For small angles the two sway directions are decoupled, each obeying
theta_ddot = -(g/l) theta - a/l in its own vertical plane. The controller is
therefore the same law used twice, with the same gains, since K depends only on
the cable length.

    u_x = -[k1 (x - x_ref) + k2 x_dot + k3 theta_x + k4 theta_dot_x]
    u_y = -[k1 (y - y_ref) + k2 y_dot + k3 theta_y + k4 theta_dot_y]

The robot can then move sideways to damp a sway that is transverse to the path,
which a single-axis controller cannot do.

Sign check before the first run: set TARGET to the current position, push the
crate along +X of the base, and check that the robot follows it. Repeat along +Y.
Fix CAM2BASE if a direction is inverted or swapped.
"""

import rtde_control
import rtde_receive
import numpy as np
import csv
import time
import camera_thread2d as cam

# ---------------- configuration ----------------
ROBOT_IP = "192.168.56.102"
DT       = 1.0 / 125

TARGET = np.array([0.35, -0.20])         # target [x, y] in the robot base frame [m]
USE_CAMERA = True

TD      = 1.45
G       = 9.81
OMEGA   = 2 * np.pi / TD
L_CABLE = G / OMEGA ** 2
K = np.array([1.000, 2.323, -6.339, -3.001])

# Maps the camera sway components to the base axes.
# Identity assumes camera x // base x and camera y // base y.
# Use [[-1, 0], [0, 1]] to invert x, [[0, 1], [1, 0]] to swap the axes, etc.
CAM2BASE = np.array([[1.0, 0.0],
                     [0.0, 1.0]])

A_MAX, V_MAX = 0.5, 0.15
MARGE     = 0.06                         # allowed excursion around the path [m]
THETA_MAX = np.radians(20)
DUREE     = 20.0

rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)

pose0 = np.array(rtde_r.getActualTCPPose())
p0 = pose0[:2]
print(f"start  {np.round(p0, 4)}")
print(f"target {np.round(TARGET, 4)}   distance {np.linalg.norm(TARGET - p0):.3f} m")
print(f"l = {L_CABLE:.3f} m   K = {K}")

etat = {}
if USE_CAMERA:
    cam.start(etat, L_CABLE)
    print("waiting for the camera (keep the crate still)...")
    while not etat.get("pret"):
        time.sleep(0.1)
    print("camera ready.")

log = open(f"goto2d_{time.strftime('%H%M%S')}.csv", "w", newline="")
w = csv.writer(log)
w.writerow(["t", "x", "y", "vx", "vy", "th_x", "th_y",
            "thd_x", "thd_y", "ux", "uy", "vcx", "vcy"])

v_cmd = np.zeros(2)
t_start = time.perf_counter()
raison = "duration"

try:
    while True:
        t_cycle = rtde_c.initPeriod()
        t = time.perf_counter() - t_start
        if t > DUREE:
            break

        # ---- state of the trolley: two independent axes ----
        p = np.array(rtde_r.getActualTCPPose()[:2])
        v = np.array(rtde_r.getActualTCPSpeed()[:2])

        # ---- state of the load: two sway components, mapped to the base axes ----
        if USE_CAMERA:
            th = CAM2BASE @ np.array([etat["theta_x"], etat["theta_y"]])
            thd = CAM2BASE @ np.array([etat["theta_dot_x"], etat["theta_dot_y"]])
            if np.max(np.abs(th)) > THETA_MAX:
                raison = f"angle {np.degrees(np.max(np.abs(th))):.1f} deg"; break
            if time.perf_counter() - etat["t"] > 0.20:
                raison = "marker lost"; break
        else:
            th = thd = np.zeros(2)

        # ---- safety: stay inside a box around the straight path ----
        if np.any(np.abs(p - p0) > np.abs(TARGET - p0) + MARGE):
            raison = f"out of box (p = {np.round(p, 3)})"; break

        # ---- the same law, applied on each axis ----
        u = np.array([
            -(K[0] * (p[0] - TARGET[0]) + K[1] * v[0] + K[2] * th[0] + K[3] * thd[0]),
            -(K[0] * (p[1] - TARGET[1]) + K[1] * v[1] + K[2] * th[1] + K[3] * thd[1]),
        ])
        u = np.clip(u, -A_MAX, A_MAX)
        v_cmd = np.clip(v_cmd + u * DT, -V_MAX, V_MAX)

        rtde_c.speedL([v_cmd[0], v_cmd[1], 0.0, 0.0, 0.0, 0.0], A_MAX, DT)

        w.writerow([f"{t:.4f}", f"{p[0]:.5f}", f"{p[1]:.5f}",
                    f"{v[0]:.5f}", f"{v[1]:.5f}",
                    f"{th[0]:.5f}", f"{th[1]:.5f}",
                    f"{thd[0]:.5f}", f"{thd[1]:.5f}",
                    f"{u[0]:.4f}", f"{u[1]:.4f}",
                    f"{v_cmd[0]:.5f}", f"{v_cmd[1]:.5f}"])
        print(f"\rt={t:5.2f}s  p=({p[0]:+.3f},{p[1]:+.3f})  "
              f"th=({np.degrees(th[0]):+5.1f},{np.degrees(th[1]):+5.1f}) deg",
              end="", flush=True)

        rtde_c.waitPeriod(t_cycle)

except KeyboardInterrupt:
    raison = "keyboard"

finally:
    rtde_c.speedStop()
    if USE_CAMERA:
        cam.stop()
    log.close()
    final = np.array(rtde_r.getActualTCPPose()[:2])
    print(f"\nstopped: {raison}")
    print(f"final {np.round(final, 4)}   error "
          f"{1000*np.linalg.norm(final - TARGET):.1f} mm")
    rtde_c.stopScript()