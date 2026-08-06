"""Experiment 8: carry the crate into a helper standing on the floor, and
measure how much the helper moves.

The control is the one from 6_main.py, unchanged: a shaped trapezoid built from
the error the camera measures, plus state feedback on the deviation and the
sway. What is new is the third marker and what is done with it.

    ID 5    static marker on the floor, defines the world frame
    ID 8    crate, hanging from the cable, transported as before
    ID 12   helper, standing on the floor, free to be pushed

The helper used to be the world reference. It cannot be any more: an object that
moves cannot define the frame its own motion is expressed in. Hence marker 5.

What the run measures. The crate is driven along its plan and meets the helper.
The helper's displacement from its reference pose is logged throughout, and the
instant it first exceeds SEUIL_BOUGE is taken as the impact. At that instant the
crate's own speed is recorded, so a series of runs at different plan speeds
gives displacement against impact speed. The crate's speed is not the tool's
speed: the load also moves relative to the tool when it swings, so

    v_charge = v_tcp + L * theta_dot

Two things worth watching in the log. The sway created by the impact tells how
much energy went back into the pendulum rather than into the helper. And the
contact detection guards the case where the crate ends up resting against the
helper while the plan still asks the tool to move forward: the model is a free
pendulum, it does not know about the obstacle, and theta then reports a
deflection the trolley cannot undo.
"""

import numpy as np
import time
import csv
from importlib import import_module

import trajectory
import estimator
import controller

import rtde_control
import rtde_receive

# 8_vision is not a valid identifier, so it is loaded by name. Everything else
# is the same module set 6_main uses, imported normally.
vision = import_module("8_vision")

SwayEstimator2D = estimator.SwayEstimator2D
Gains           = controller.Gains
feedback        = controller.feedback
CommandLimiter  = controller.CommandLimiter
FinDeMouvement  = controller.FinDeMouvement
Integrateur     = controller.Integrateur

# ---------------- configuration ----------------
ROBOT_IP = "192.168.56.102"
L_VRAI   = 1.11
L_MODELE = 1.11
Ts       = 1 / 500

FEEDFORWARD = True
FEEDBACK    = True

T1     = 2 * np.pi * np.sqrt(L_MODELE / 9.81)
V_TRAJ = 0.15                   # plan speed: the variable of the experiment

ZETA_CL, OMEGA_T = 0.15, 0.3
K_I, I_MAX = 0.15, 0.05
SEUIL_INTEG = 0.05

V_MAX, A_MAX, JERK_MAX = 0.3, 0.8, 32.0
V_FB_MAX   = 0.25
COURSE_MAX = 0.60
THETA_MAX  = np.radians(20)
AGE_MAX    = 0.30
REF_PERDUE_MAX = 2.0
T_OBSERV   = 5.0                # recording after the plan, to see the helper settle
TIMEOUT    = 120.0
NIS_CHOC   = 400.0              # relaxed outlier test once contact has happened
DRY_RUN    = False

APPUI_D_ERR = 0.005
APPUI_V_MIN = 0.005
APPUI_T     = 1.0
APPUI_MAX   = 6.0

if not (FEEDFORWARD or FEEDBACK):
    raise SystemExit("At least one of FEEDFORWARD and FEEDBACK must be True.")
CONFIG = ("ff+fb" if FEEDFORWARD and FEEDBACK
          else "ff_seul" if FEEDFORWARD else "fb_seul")
print(f"configuration: {CONFIG}   V_TRAJ = {V_TRAJ} m/s   T_d = {T1:.3f} s")

_def = f"essai8_{CONFIG}_v{int(1000*V_TRAJ)}_{time.strftime('%H%M%S')}"
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
vision.start(etat, L_VRAI)
print("tout immobile: calibration de la suspension et pose de reference du helper...")
while not etat["pret"]:
    time.sleep(0.1)
print("pret.\n")

# ---------------- control blocks ----------------
gains = Gains(L=L_MODELE, zeta_cl=ZETA_CL, omega_t=OMEGA_T)
est   = SwayEstimator2D(L=L_MODELE, Ts=Ts)
lim   = CommandLimiter(Ts, V_MAX, A_MAX, JERK_MAX)
integ = Integrateur(K_i=K_I, v_max=I_MAX, Ts=Ts)
fin   = FinDeMouvement(eps_x=0.040, eps_theta=np.radians(1.5),
                       eps_theta_dot=0.02, eps_v=0.010, T_dwell=0.5)

e0 = np.asarray(etat["erreur"], float)
traj = trajectory.Trajectoire(e0, V_TRAJ, T1, t0=0.0, accorde=True, a_max=A_MAX)
print(f"{traj}")
print(f"erreur initiale {1000*np.linalg.norm(e0):.0f} mm\n")

log = open(NOM_CSV, "w", newline="")
w = csv.writer(log)
w.writerow(["t", "ex", "ey", "th_x", "th_y", "thd_x", "thd_y", "yaw",
            "vref_x", "vref_y", "vfb_x", "vfb_y", "vcmd_x", "vcmd_y",
            "tcp_x", "tcp_y", "dans_cible", "th_brut_x", "th_brut_y",
            "hx", "hy", "hdx", "hdy", "hdyaw", "hbouge"])

t_last = 0.0
t_ref_vue = time.perf_counter()
err_ref, t_err_ref, appui = None, 0.0, False
t_start = time.perf_counter()
raison = "cible atteinte"
pic_ballant = 0.0
n_cycles = 0

# impact bookkeeping
t_choc, v_choc, th_avant_choc = None, None, None
depl_max, dyaw_max = 0.0, 0.0
pic_apres_choc = 0.0

try:
    while True:
        t_cycle = rtde_c.initPeriod()
        t = time.perf_counter() - t_start
        n_cycles += 1

        # ---- estimation ----
        est.predict(lim.a_applied, time.perf_counter())
        if etat["t"] > t_last:
            est.update(etat["theta"], etat["t"],
                       NIS_CHOC if t_choc is not None else 25.0)
            t_last = etat["t"]
        th, thd = est.theta, est.theta_dot
        pic_ballant = max(pic_ballant, float(np.max(np.abs(th))))

        # ---- measurements ----
        erreur = np.asarray(etat["erreur"], float)
        tcp = np.array(rtde_r.getActualTCPPose()[:2])
        v_tcp = np.array(rtde_r.getActualTCPSpeed()[:2])
        depl = np.asarray(etat["helper_depl"], float)
        dyaw = float(etat["helper_dyaw"])

        # speed of the load itself: the tool plus what the swing adds
        v_charge = v_tcp + L_VRAI * thd

        # ---- impact ----
        if t_choc is None and etat.get("helper_bouge"):
            t_choc = t
            v_choc = float(np.linalg.norm(v_charge))
            th_avant_choc = float(np.max(np.abs(th)))
            est.gonfle()            # the collision is not in the pendulum model
            print(f"\n[choc] t = {t:.2f} s   vitesse charge "
                  f"{1000*v_choc:.0f} mm/s   ballant avant "
                  f"{np.degrees(th_avant_choc):.2f} deg")
        if t_choc is not None:
            pic_apres_choc = max(pic_apres_choc, float(np.max(np.abs(th))))
        depl_max = max(depl_max, float(np.linalg.norm(depl)))
        dyaw_max = max(dyaw_max, abs(dyaw))

        # ---- contact held: theta is no longer sway ----
        n_err = float(np.linalg.norm(erreur))
        if err_ref is None or abs(n_err - err_ref) > APPUI_D_ERR:
            err_ref, t_err_ref = n_err, t
        appui = ((t - t_err_ref) > APPUI_T
                 and float(np.linalg.norm(lim.v_prev)) > APPUI_V_MIN)

        # ---- safety ----
        if time.perf_counter() - etat["t"] > AGE_MAX:
            raison = "vision perdue"; break
        if not etat["vus"][1]:
            raison = "marqueur de charge (8) perdu"; break
        if etat["vus"][0]:
            t_ref_vue = time.perf_counter()
        elif time.perf_counter() - t_ref_vue > REF_PERDUE_MAX:
            raison = "marqueur de reference (5) perdu"; break
        if np.max(np.abs(th)) > THETA_MAX:
            raison = f"ballant {np.degrees(np.max(np.abs(th))):.1f} deg"; break
        if np.any(np.abs(tcp - p_start) > COURSE_MAX):
            raison = f"hors course {np.round(tcp - p_start, 3)}"; break
        if appui and (t - t_err_ref) > APPUI_MAX:
            raison = (f"appui persistant {t - t_err_ref:.1f} s, la charge bute "
                      f"sur quelque chose"); break
        if t > TIMEOUT:
            raison = "timeout"; break

        # ---- feedforward ----
        if FEEDFORWARD:
            p_ref, v_ref, _ = traj(t)
            err_fb = erreur - (e0 - p_ref)
        else:
            v_ref = np.zeros(2)
            err_fb = erreur

        # ---- feedback ----
        if FEEDBACK:
            actif = ((not FEEDFORWARD) or traj.terminee(t)) and not appui \
                    and np.linalg.norm(err_fb) < SEUIL_INTEG
            v_fb = feedback(err_fb, th, thd, gains, V_FB_MAX) + integ(err_fb, actif)
        else:
            v_fb = np.zeros(2)

        v = lim(v_ref + v_fb)

        if DRY_RUN:
            print(f"\re {1000*erreur[0]:+6.0f},{1000*erreur[1]:+6.0f} mm | "
                  f"th {np.degrees(th[0]):+5.1f},{np.degrees(th[1]):+5.1f} deg | "
                  f"helper {1000*np.linalg.norm(depl):5.1f} mm",
                  end="", flush=True)
        else:
            rtde_c.speedL([v[0], v[1], 0.0, 0.0, 0.0, 0.0], A_MAX, Ts)

        w.writerow([f"{t:.4f}"] + [f"{x:.5f}" for x in
                   (*erreur, *th, *thd, etat.get("yaw", 0.0),
                    *v_ref, *v_fb, *v, *tcp)] +
                   [int(bool(etat.get("dans_cible")))] +
                   [f"{etat['theta'][0]:.5f}", f"{etat['theta'][1]:.5f}"] +
                   [f"{x:.5f}" for x in (*etat["helper_pos"], *depl, dyaw)] +
                   [int(bool(etat.get("helper_bouge")))])

        # ---- end of run ----
        if FEEDBACK:
            if ((not FEEDFORWARD or traj.terminee(t))
                    and etat.get("dans_cible")
                    and fin(erreur, th, thd, v_tcp, t)):
                break
        else:
            if traj.terminee(t) and t > traj.duree + T_OBSERV:
                raison = "fin du plan, etat final enregistre"
                break

        if n_cycles % 100 == 0 and not DRY_RUN:
            print(f"\r|e| {1000*np.linalg.norm(erreur):5.0f} mm | "
                  f"th {np.degrees(th[0]):+5.1f},{np.degrees(th[1]):+5.1f} deg | "
                  f"helper {1000*np.linalg.norm(depl):5.1f} mm"
                  f"{'  BOUGE' if etat.get('helper_bouge') else '       '}",
                  end="", flush=True)

        rtde_c.waitPeriod(t_cycle)

except KeyboardInterrupt:
    raison = "interruption clavier"

finally:
    if not DRY_RUN:
        rtde_c.speedStop()
    time.sleep(0.2)

    # let the helper settle before reading its final pose
    if not DRY_RUN and raison in ("cible atteinte", "fin du plan, etat final enregistre"):
        print("\n\nattente de l'immobilisation...")
        time.sleep(3.0)
    depl_fin = np.asarray(etat.get("helper_depl", np.zeros(2)), float)
    dyaw_fin = float(etat.get("helper_dyaw", 0.0))
    log.close()

    duree = time.perf_counter() - t_start
    print(f"\n\nconfiguration {CONFIG}   arret: {raison}")
    print(f"\n--- grue ---")
    e_fin = np.asarray(etat.get("erreur", np.zeros(2)), float)
    print(f"erreur finale      {1000*np.linalg.norm(e_fin):6.1f} mm   "
          f"{'dans ' + str(etat.get('zone')) if etat.get('dans_cible') else 'hors cible'}")
    print(f"ballant maximal    {np.degrees(pic_ballant):6.2f} deg")
    print(f"ballant final      {np.degrees(np.max(np.abs(est.theta))):6.2f} deg")

    print(f"\n--- helper (marqueur 12) ---")
    if t_choc is None:
        print(f"pas de contact detecte (seuil {1000*vision.SEUIL_BOUGE:.0f} mm)")
        print(f"deplacement observe {1000*depl_max:5.1f} mm, "
              f"donc sous le seuil ou pas touche")
    else:
        print(f"choc a t = {t_choc:.2f} s")
        print(f"vitesse de la charge au choc   {1000*v_choc:6.0f} mm/s")
        print(f"ballant juste avant le choc    {np.degrees(th_avant_choc):6.2f} deg")
        print(f"ballant maximal apres le choc  {np.degrees(pic_apres_choc):6.2f} deg")
        print(f"deplacement maximal            {1000*depl_max:6.1f} mm")
        print(f"deplacement final              {1000*np.linalg.norm(depl_fin):6.1f} mm"
              f"   ({1000*depl_fin[0]:+.1f}, {1000*depl_fin[1]:+.1f})")
        print(f"rotation finale                {np.degrees(dyaw_fin):+6.1f} deg   "
              f"(max {np.degrees(dyaw_max):.1f})")
        print("\n  le deplacement final donne l'energie absorbee par le sol:")
        print("  E = mu * m * g * d, avec mu le coefficient de frottement du")
        print("  helper sur le sol et m sa masse. Repeter a plusieurs V_TRAJ")
        print("  donne le deplacement en fonction de la vitesse d'impact.")

    print(f"\nduree {duree:6.2f} s   frequence {n_cycles/max(duree,1e-9):5.0f} Hz")
    print(f"saturation {100*lim.taux_saturation:.1f} %   "
          f"mesures rejetees {est.x.n_rejets}/{est.y.n_rejets}")
    print(f"journal ecrit: {NOM_CSV}")

    if not DRY_RUN and raison in ("cible atteinte", "fin du plan, etat final enregistre"):
        if input("retour a la pose de depart ? [o/N] ").strip().lower() == "o":
            rtde_c.moveL(POSE_DEPART, 0.05, 0.2)
            print("retour termine.")

    vision.stop()
    rtde_c.stopScript()