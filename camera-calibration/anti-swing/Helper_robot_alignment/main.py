"""Experiment: bring the helper robot to a sequence of points of the crate frame.

    frames.py       vision: position error and raw sway angle, with a timestamp
    estimator.py    Kalman filter: theta and theta_dot, camera delay compensated
    controller.py   gains, feedback law, command limiter, end-of-move test
    main.py         sequence of steps, robot I/O, safety, logging

One control cycle, at 125 Hz:

    read the robot            x, v
    read the vision           position error, raw theta, timestamp
    Kalman predict            with the acceleration actually applied
    Kalman update             only when a new image is available
    feedback                  v_fb = K_x e + K_theta theta + K_thetadot theta_dot
    command                   v_cmd = v_ref + v_fb, then limited
    send                      speedL, no rotation of the tool
    check                     safety, then end-of-move

Adding a step to the experiment means adding a line to SEQUENCE.
"""

import rtde_control
import rtde_receive
import numpy as np
import csv
import time

import frames
import estimator
import controller
import trajectory

# ---------------- configuration ----------------
ROBOT_IP = "192.168.56.102"
DT       = 1.0 / 125            # CB3
L_CABLE  = 0.5225               # from the measured period of 1.45 s
ZETA_N   = 0.00228              # measured natural damping

SEQUENCE = [
    ("approach the corner", (0.30, 0.10)),
]

# vision
RETARD_CAM = 0.000              # pipeline delay [s], to be measured, subtracted
                                # from the timestamp before feeding the filter

# feedforward
ACCORDE = True                  # ramps last one full period: no residual sway
V_TRAJ  = 0.10                  # cruise speed of the reference [m/s]
T1      = 1.45                  # ramp duration = measured damped period [s]
                                # a half period would double the sway, not cancel it

# limits
V_MAX, A_MAX, JERK_MAX = 0.15, 0.50, 5.0
V_FB_MAX   = 0.12               # saturation of the feedback part alone [m/s]
COURSE_MAX = 0.40               # excursion allowed from the start pose [m]
THETA_MAX  = np.radians(20)
AGE_MAX    = 0.30               # max age of a vision measurement [s]
TIMEOUT    = 40.0               # per step [s]

# ---------------- gains ----------------
gains = controller.Gains(L=L_CABLE, zeta=ZETA_N, zeta_cl=0.7, omega_t=1.0)

# ---------------- vision ----------------
etat = {}
frames.start(etat, L_CABLE, d1=SEQUENCE[0][1])
print("\nwaiting for the camera. Keep the helper still, both markers visible.")
while not etat["pret"]:
    time.sleep(0.1)
print("vision ready.")

# ---------------- estimator, limiter, end-of-move ----------------
est = estimator.SwayEstimator2D(L=L_CABLE, zeta=ZETA_N, Ts=DT,
                                sigma_a=0.05, sigma_theta=np.radians(0.2))
limiteur = controller.CommandLimiter(DT, V_MAX, A_MAX, JERK_MAX)
fin = controller.FinDeMouvement()

# ---------------- robot ----------------
rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
init_q = rtde_r.getActualQ()
p_start = np.array(rtde_r.getActualTCPPose()[:2])
print(f"robot connected, start {np.round(p_start, 4)}\n")

log = open(f"exp_{time.strftime('%Y%m%d_%H%M%S')}.csv", "w", newline="")
w = csv.writer(log)
w.writerow(["t", "etape", "ex", "ey", "vx", "vy",
            "th_mes_x", "th_mes_y", "th_hat_x", "th_hat_y",
            "thd_hat_x", "thd_hat_y", "vfb_x", "vfb_y",
            "vcmd_x", "vcmd_y", "ax", "ay", "sigma_th", "innov_x"])

# ---------------- run ----------------
etape = 0
t_image = None                  # timestamp of the last measurement consumed
a_applied = np.zeros(2)


def nouvelle_etape(indice: int, t: float):
    """Load the target of a step and plan the shaped move towards it."""
    nom, cible = SEQUENCE[indice]
    frames.set_target(*cible)
    time.sleep(0.05)                        # let the vision publish the new error
    e0 = np.asarray(etat["erreur"], float)  # helper - target, so move by -e0
    traj = trajectory.Trajectoire(-e0, V_TRAJ, T1, t0=t,
                                  accorde=ACCORDE, a_max=A_MAX)
    print(f"\nstep {indice+1}/{len(SEQUENCE)}: {nom}   target {cible}")
    print(f"  {traj}   start error {1000*np.linalg.norm(e0):.0f} mm")
    return e0, traj


t_start = time.perf_counter()
e0, traj = nouvelle_etape(0, 0.0)
t_etape = time.perf_counter()
raison = "sequence finished"

try:
    while True:
        t_cycle = rtde_c.initPeriod()
        t = time.perf_counter() - t_start

        # ---- 1. robot ----
        p = np.array(rtde_r.getActualTCPPose()[:2])
        v = np.array(rtde_r.getActualTCPSpeed()[:2])

        # ---- 2. vision ----
        e = np.asarray(etat["erreur"], float)          # x_ref - x is -e here
        th_mes = np.asarray(etat["theta"], float)
        t_mes = etat["t"]

        # ---- 3. Kalman ----
        est.predict(a_applied, time.perf_counter())
        if t_mes != t_image:                            # a new image is available
            est.update(th_mes, t_mes - RETARD_CAM)
            t_image = t_mes
        th, thd = est.theta, est.theta_dot

        # ---- 4. safety ----
        if time.perf_counter() - t_mes > AGE_MAX:
            raison = "vision lost"; break
        if not all(etat["vus"]):
            raison = "a marker is not visible"; break
        if np.max(np.abs(th)) > THETA_MAX:
            raison = f"sway {np.degrees(np.max(np.abs(th))):.1f} deg"; break
        if np.any(np.abs(p - p_start) > COURSE_MAX):
            raison = f"out of range {np.round(p - p_start, 3)}"; break
        if not np.all(np.isfinite(np.r_[e, th, thd])):
            raison = "non finite value"; break
        if time.perf_counter() - t_etape > TIMEOUT:
            raison = f"timeout on step {etape + 1}"; break

        # ---- 5. feedforward: where the load should be now, and how fast ----
        p_ref, v_ref, _ = traj(t)
        # reference error w.r.t. the final target, from e0 down to zero
        r = e0 + p_ref
        # the feedback only sees the deviation from that moving reference
        v_fb = controller.feedback(r - e, th, thd, gains, V_FB_MAX)
        v_cmd = limiteur(v_ref + v_fb)
        a_applied = limiteur.a_applied

        # ---- 6. send, tool orientation unchanged ----
        rtde_c.speedL([v_cmd[0], v_cmd[1], 0.0, 0.0, 0.0, 0.0], A_MAX, DT)

        w.writerow([f"{t:.4f}", etape] +
                   [f"{x:.5f}" for x in (*e, *v, *th_mes, *th, *thd,
                                         *v_fb, *v_cmd, *a_applied)] +
                   [f"{est.x.sigma[0]:.6f}", f"{est.x.innovation:.6f}"])

        # ---- 7. end of step: only tested once the planned move is over ----
        if traj.terminee(t) and fin(e, th, thd, v, t):
            print(f"\n  reached in {time.perf_counter()-t_etape:.2f} s   "
                  f"error {1000*np.linalg.norm(e):.1f} mm   "
                  f"sway {np.degrees(np.max(np.abs(th))):.2f} deg")
            etape += 1
            if etape >= len(SEQUENCE):
                break
            fin.reset()
            e0, traj = nouvelle_etape(etape, t)
            t_etape = time.perf_counter()

        if int(t / DT) % 25 == 0:
            print(f"\r  |e| {1000*np.linalg.norm(e):5.0f} mm   "
                  f"th {np.degrees(th[0]):+5.1f},{np.degrees(th[1]):+5.1f} deg   "
                  f"v {v_cmd[0]:+.3f},{v_cmd[1]:+.3f} m/s", end="", flush=True)

        rtde_c.waitPeriod(t_cycle)

except KeyboardInterrupt:
    raison = "keyboard"

finally:
    # Stop and stay. The robot is deliberately NOT sent back to its starting
    # configuration: the helper must be left where it was brought, and after a
    # safety abort the load is swinging, so a moveJ would drive it fast with no
    # anti-sway at all. Returning, if wanted, is a separate deliberate action.
    rtde_c.speedStop()
    frames.stop()
    log.close()
    p_fin = np.array(rtde_r.getActualTCPPose()[:2])
    print(f"\n\nstopped: {raison}   steps completed: {etape}/{len(SEQUENCE)}")
    print(f"final position {np.round(p_fin, 4)}   "
          f"moved {1000*np.linalg.norm(p_fin - p_start):.0f} mm from the start")
    print(f"saturation: {100*limiteur.taux_saturation:.1f} % of cycles   "
          f"rejected measurements: {est.x.n_rejets} / {est.y.n_rejets}")
    rtde_c.stopScript()