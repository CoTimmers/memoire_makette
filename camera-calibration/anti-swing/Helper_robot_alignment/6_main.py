"""Anti-sway: bring the load into the target zone, damping the sway on the way.

Three configurations can be run, to compare the two layers:

    FEEDFORWARD  FEEDBACK   what is tested
    -----------  --------   -------------------------------------------------
        True       False    open loop: the shaped ramps alone. Nothing corrects
                            a model error or a disturbance.
        False      True     closed loop only: the motion is produced by the
                            position term, the sway is damped reactively.
        True       True     both layers, the intended configuration.

The target is not a TCP position. The camera measures, in robot base axes, the
vector from the cable attachment point to the centre of the target zone; that
vector is the error. The TCP places itself, since at convergence the sway is
zero and the load hangs right under the tool.

Every run writes a CSV named after its configuration, so the three can be
compared directly.
"""

import numpy as np
import time
import csv
import rtde_control
import rtde_receive
import vision
import trajectory
from estimator import SwayEstimator2D
from controller import Gains, feedback, CommandLimiter, FinDeMouvement

# ---------------- configuration ----------------
ROBOT_IP = "192.168.56.102"
L_CABLE  = 1.17
Ts       = 1 / 125

FEEDFORWARD = True              # shaped trapezoid built from the measured error
FEEDBACK    = True              # state feedback on the deviation and the sway

T1     = 2 * np.pi * np.sqrt(L_CABLE / 9.81)    # damped period, ramp duration [s]
V_TRAJ = 0.15                                    # cruise speed of the plan [m/s]

ZETA_CL, OMEGA_T = 0.7, 1.0     # closed-loop sway damping, position bandwidth

V_MAX, A_MAX, JERK_MAX = 0.3, 0.8, 8.0
V_FB_MAX   = 0.25               # saturation of the feedback part alone [m/s]
COURSE_MAX = 0.50               # excursion allowed from the start pose [m]
THETA_MAX  = np.radians(20)
AGE_MAX    = 0.30               # max age of a vision measurement [s]
T_OBSERV   = 5.0                # recording time after the plan ends [s]
TIMEOUT    = 90.0
DRY_RUN    = False              # True = no motion, display only

if not (FEEDFORWARD or FEEDBACK):
    raise SystemExit("At least one of FEEDFORWARD and FEEDBACK must be True.")
CONFIG = ("ff+fb" if FEEDFORWARD and FEEDBACK
          else "ff_seul" if FEEDFORWARD else "fb_seul")
print(f"configuration: {CONFIG}")

# name of the log, asked before anything else so the prompt is not buried
_def = f"essai_{CONFIG}_{time.strftime('%H%M%S')}"
NOM_CSV = input(f"nom du fichier CSV [{_def}]: ").strip() or _def
if not NOM_CSV.endswith(".csv"):
    NOM_CSV += ".csv"
print(f"journal: {NOM_CSV}\n")

# ---------------- robot ----------------
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
POSE_DEPART = rtde_r.getActualTCPPose()
p_start = np.array(POSE_DEPART[:2])
print("pose de depart:", [round(x, 3) for x in POSE_DEPART])

# ---------------- vision ----------------
etat = {}
vision.AFFICHAGE = True
vision.start(etat, L_CABLE)
print("bac immobile pour la calibration du pivot...")
while not etat["pret"]:
    time.sleep(0.1)
print("pret.\n")

# ---------------- control blocks ----------------
gains = Gains(L=L_CABLE, zeta_cl=ZETA_CL, omega_t=OMEGA_T)
est   = SwayEstimator2D(L=L_CABLE, Ts=Ts)
lim   = CommandLimiter(Ts, V_MAX, A_MAX, JERK_MAX)
fin   = FinDeMouvement(eps_x=0.010, eps_theta=np.radians(1.0),
                       eps_theta_dot=0.02, eps_v=0.010, T_dwell=0.5)

# the plan comes from the error measured by the camera, not from a hardcoded pose
e0 = np.asarray(etat["erreur"], float)
traj = trajectory.Trajectoire(e0, V_TRAJ, T1, t0=0.0, accorde=True, a_max=A_MAX)
print(f"T1 = {T1:.3f} s   {traj}")
print(f"erreur initiale {1000*np.linalg.norm(e0):.0f} mm\n")

log = open(NOM_CSV, "w", newline="")
w = csv.writer(log)
w.writerow(["t", "ex", "ey", "th_x", "th_y", "thd_x", "thd_y", "yaw",
            "vref_x", "vref_y", "vfb_x", "vfb_y", "vcmd_x", "vcmd_y",
            "tcp_x", "tcp_y", "dans_cible"])

t_last = 0.0
t_start = time.perf_counter()
raison = "cible atteinte"
pic_ballant = 0.0

try:
    while True:
        t_cycle = rtde_c.initPeriod()
        t = time.perf_counter() - t_start

        # ---- estimation ----
        est.predict(lim.a_applied, time.perf_counter())
        if etat["t"] > t_last:
            est.update(etat["theta"], etat["t"])
            t_last = etat["t"]
        th, thd = est.theta, est.theta_dot
        pic_ballant = max(pic_ballant, float(np.max(np.abs(th))))

        # ---- measurements ----
        erreur = np.asarray(etat["erreur"], float)      # cible - accroche
        tcp = np.array(rtde_r.getActualTCPPose()[:2])
        v_tcp = np.array(rtde_r.getActualTCPSpeed()[:2])

        # ---- safety ----
        if time.perf_counter() - etat["t"] > AGE_MAX:
            raison = "vision perdue"; break
        if not all(etat["vus"]):
            raison = "un marqueur n'est plus visible"; break
        if np.max(np.abs(th)) > THETA_MAX:
            raison = f"ballant {np.degrees(np.max(np.abs(th))):.1f} deg"; break
        if np.any(np.abs(tcp - p_start) > COURSE_MAX):
            raison = f"hors course {np.round(tcp - p_start, 3)}"; break
        if t > TIMEOUT:
            raison = "timeout"; break

        # ---- feedforward ----
        if FEEDFORWARD:
            p_ref, v_ref, _ = traj(t)
            err_fb = erreur - (e0 - p_ref)        # deviation from the plan
        else:
            v_ref = np.zeros(2)
            err_fb = erreur                       # no plan: aim straight at the target

        # ---- feedback ----
        v_fb = (feedback(err_fb, th, thd, gains, V_FB_MAX) if FEEDBACK
                else np.zeros(2))

        v = lim(v_ref + v_fb)

        if DRY_RUN:
            print(f"\re {1000*erreur[0]:+6.0f},{1000*erreur[1]:+6.0f} mm | "
                  f"th {np.degrees(th[0]):+5.1f},{np.degrees(th[1]):+5.1f} deg | "
                  f"v {v[0]:+.3f},{v[1]:+.3f} | "
                  f"{'IN ' if etat.get('dans_cible') else 'out'}",
                  end="", flush=True)
        else:
            rtde_c.speedL([v[0], v[1], 0.0, 0.0, 0.0, 0.0], A_MAX, Ts)

        w.writerow([f"{t:.4f}"] + [f"{x:.5f}" for x in
                   (*erreur, *th, *thd, etat.get("yaw", 0.0),
                    *v_ref, *v_fb, *v, *tcp)] +
                   [int(bool(etat.get("dans_cible")))])

        # ---- end of run ----
        if FEEDBACK:
            # converged: plan over, inside the zone, everything at rest
            if ((not FEEDFORWARD or traj.terminee(t))
                    and etat.get("dans_cible")
                    and fin(erreur, th, thd, v_tcp, t)):
                break
        else:
            # open loop: nothing will converge, so record the residual sway
            if traj.terminee(t) and t > traj.duree + T_OBSERV:
                raison = "fin du plan, ballant residuel enregistre"
                break

        if int(t / Ts) % 25 == 0 and not DRY_RUN:
            print(f"\r|e| {1000*np.linalg.norm(erreur):5.0f} mm | "
                  f"th {np.degrees(th[0]):+5.1f},{np.degrees(th[1]):+5.1f} deg | "
                  f"{'IN ' if etat.get('dans_cible') else 'out'}",
                  end="", flush=True)

        rtde_c.waitPeriod(t_cycle)

except KeyboardInterrupt:
    raison = "interruption clavier"

finally:
    # Stop and stay: the load is left where it was brought, and after a safety
    # abort it may still be swinging, so no automatic return here.
    if not DRY_RUN:
        rtde_c.speedStop()
    time.sleep(0.2)
    log.close()
    e_fin = np.asarray(etat.get("erreur", np.zeros(2)), float)
    print(f"\n\nconfiguration {CONFIG}   arret: {raison}")
    print(f"erreur finale     {1000*np.linalg.norm(e_fin):6.1f} mm   "
          f"{'dans la cible' if etat.get('dans_cible') else 'hors cible'}")
    print(f"ballant maximal   {np.degrees(pic_ballant):6.2f} deg")
    print(f"ballant final     {np.degrees(np.max(np.abs(est.theta))):6.2f} deg")
    print(f"duree             {time.perf_counter()-t_start:6.2f} s")
    print(f"saturation {100*lim.taux_saturation:.1f} %   "
          f"mesures rejetees {est.x.n_rejets}/{est.y.n_rejets}")
    print(f"journal ecrit: {NOM_CSV}")
    vision.stop()
    rtde_c.stopScript()