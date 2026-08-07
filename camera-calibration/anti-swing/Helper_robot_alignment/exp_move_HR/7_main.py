"""Experiment 7: launch the load into the crate, then bring it back and damp it.

Four phases, run one after the other:

    0  ALLER     feedback only, drive the load to D1 and let it settle. This
                 only fixes the starting point, so that every repetition of the
                 experiment covers the same distance and reaches the same speed.
    1  APPROCHE  shaped ramp then cruise, aimed at D3, with feedback correcting
                 the deviation from the plan. The ramp lasts one full pendulum
                 period, so the load hangs straight when the cruise begins.
    2  LIBRE     the crane is stopped dead and the feedback is switched off. The
                 load keeps the cruise speed, swings forward, and meets the
                 crate. Nothing acts on it during this phase; that is the point.
    3  RETOUR    feedback only again, back to D1, damping the sway the impact
                 left behind.

Why the stop launches the load. Before it, trolley and load share the velocity v
and the cable is vertical. After it the trolley is fixed and the load still
carries v, so the pendulum starts from theta = 0 with theta_dot = v / L. Its
reach is v sqrt(L/g) and it gets there in a quarter period. Put the crate closer
than that reach and contact happens; the closer it is, the faster the contact.
7_trajectory.py prints the numbers, and this script prints them again at launch
with the values actually configured.

The geometry is measured, not assumed: D1 and D3 are defined in 7_vision.py
relative to the crate marker, and the error driving the controller is the vector
from the load pivot to the active point, in robot base axes.
"""

import numpy as np
import time
import csv
from importlib import import_module

import rtde_control
import rtde_receive

# The modules are named 7_*.py, which is not a valid Python identifier, so they
# are loaded by name rather than with a plain import statement.
vision     = import_module("7_vision")
trajectory = import_module("7_trajectory")
_est_mod   = import_module("7_estimator")
_ctl_mod   = import_module("7_controller")

SwayEstimator2D = _est_mod.SwayEstimator2D
Gains           = _ctl_mod.Gains
feedback        = _ctl_mod.feedback
CommandLimiter  = _ctl_mod.CommandLimiter
FinDeMouvement  = _ctl_mod.FinDeMouvement
Integrateur     = _ctl_mod.Integrateur

# ---------------- configuration ----------------
ROBOT_IP = "192.168.56.102"
L_VRAI   = 1.0                 # measured cable length [m]
L_MODELE = 1.0                 # length the controller believes in [m]
Ts       = 1 / 500

ALLER_A_D1 = True               # phase 0: reposition before launching
FEEDBACK   = True               # feedback during phases 0, 1 and 3

T1     = 2 * np.pi * np.sqrt(L_MODELE / 9.81)   # ramp duration, one period [s]
# Cruise speed. This one number decides the whole experiment: it sets how far
# the load swings after the stop, hence whether it reaches the crate at all.
# 7_simulation.py, which runs this same control chain against a nonlinear
# pendulum, puts the contact threshold at 0.16 m/s for D3 at 50 mm. At 0.15 the
# load stops about 3 mm short. Raise this, or move D3 in, before expecting a
# contact.
V_TRAJ = 0.25
T_LIBRE = 0.8     # duration of the free phase after the stop [s]
T_OBSERV = 8.0                  # enregistrement apres le retour, pour le yaw [s]

ZETA_CL, OMEGA_T = 0.3, 0.5    # closed-loop sway damping, position bandwidth
K_I, I_MAX = 0.4, 0.06
# The integral term exists to beat static friction at the end of a move, not to
# produce the move. Left running over a 350 mm error it saturates and carries
# the load well past the target, so it only wakes up inside this radius.
SEUIL_INTEG = 0.10              # [m]

V_MAX, A_MAX, JERK_MAX = 0.5, 1.1, 32.0
A_STOP     = 4.0                # deceleration of the deliberate stop [m/s2]
V_FB_MAX   = 0.25               # saturation of the feedback part alone [m/s]
COURSE_MAX = 0.60               # excursion allowed from the start pose [m]
THETA_MAX  = np.radians(20)
AGE_MAX    = 0.30               # max age of a vision measurement [s]
REF_PERDUE_MAX = 3.0
CHARGE_PERDUE_MAX = 3.0         # tolerated blackout on the crate marker [s]
TIMEOUT    = 120.0
NIS_LIBRE  = 400.0              # relaxed outlier test around the impact
DRY_RUN    = False              # True = no motion, display only

ALLER, APPROCHE, LIBRE, RETOUR = 0, 1, 2, 3
NOM_PHASE = {ALLER: "aller D1", APPROCHE: "approche D3",
             LIBRE: "ballant libre", RETOUR: "retour D1"}

# ---------------- what the configuration implies ----------------
G = 9.81
portee = trajectory.portee_ballant(V_TRAJ, L_MODELE)
d3 = float(np.linalg.norm(vision.D3[:2]))
d1 = float(np.linalg.norm(vision.D1[:2]))
v_contact = trajectory.vitesse_contact(V_TRAJ, L_MODELE, d3)

print(f"L = {L_VRAI} m   T_d = {T1:.3f} s   T_d/4 = {T1/4:.3f} s")
print(f"D1 a {1000*d1:.0f} mm de l'origine, D3 a {1000*d3:.0f} mm, "
      f"course {1000*(d1-d3):.0f} mm")
print(f"arret a {V_TRAJ} m/s -> portee {1000*portee:.1f} mm, "
      f"theta max {np.degrees(trajectory.angle_ballant(V_TRAJ, L_MODELE)):.2f} deg")
marge = portee - d3
if marge <= 0:
    print(f"PAS DE CONTACT: la portee est {-1000*marge:.1f} mm trop courte. "
          f"Rapprocher D3 ou monter V_TRAJ.")
elif marge < 0.010:
    print(f"CONTACT INCERTAIN: marge de seulement {1000*marge:.1f} mm. La "
          f"formule annonce {1000*v_contact:.0f} mm/s, mais le ballant residuel")
    print(f"  de la rampe et les {1000*V_TRAJ/A_STOP:.0f} ms de freinage reel "
          f"mangent cette marge; en simulation la charge s'arrete court.")
else:
    print(f"contact prevu avec le bac a {1000*v_contact:.0f} mm/s "
          f"apres {T1/4:.2f} s environ, marge {1000*marge:.1f} mm")
if T_LIBRE < T1 / 4:
    print(f"ATTENTION: T_LIBRE = {T_LIBRE} s < T_d/4 = {T1/4:.2f} s, "
          f"le retour partira avant que la charge ait touche.")
print()

_def = f"essai7_v{int(1000*V_TRAJ)}_{time.strftime('%H%M%S')}"
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
vision.CIBLE_ACTIVE = "d1" if ALLER_A_D1 else "d3"
vision.start(etat, L_VRAI)
print("charge immobile pour la calibration de la suspension...")
while not etat["pret"]:
    time.sleep(0.1)
print("pret.\n")

# ---------------- control blocks ----------------
gains = Gains(L=L_MODELE, zeta_cl=ZETA_CL, omega_t=OMEGA_T)
est   = SwayEstimator2D(L=L_MODELE, Ts=Ts)
lim   = CommandLimiter(Ts, V_MAX, A_MAX, JERK_MAX)
integ = Integrateur(K_i=K_I, v_max=I_MAX, Ts=Ts)
fin_aller = FinDeMouvement(eps_x=0.020, eps_theta=np.radians(1.5),
                           eps_theta_dot=0.05, eps_v=0.015, T_dwell=1.0)
fin_retour = FinDeMouvement(eps_x=0.015, eps_theta=np.radians(1.5),
                            eps_theta_dot=0.02, eps_v=0.010, T_dwell=0.5)

log = open(NOM_CSV, "w", newline="")
w = csv.writer(log)
w.writerow(["t", "phase", "ex", "ey", "th_x", "th_y", "thd_x", "thd_y", "yaw",
            "pos_x", "pos_y", "dist_origine",
            "vref_x", "vref_y", "vfb_x", "vfb_y", "vcmd_x", "vcmd_y",
            "tcp_x", "tcp_y", "th_brut_x", "th_brut_y"])

phase = ALLER if ALLER_A_D1 else APPROCHE
t_phase = 0.0
traj = None
traj_ret = None
e0 = np.zeros(2)
e0_ret = np.zeros(2)
t_last = 0.0
t_ref_vue = time.perf_counter()
t_charge_vue = time.perf_counter()
raison = "sequence terminee"
n_cycles = 0
pic_ballant = 0.0
journal_phases = []

# recorded around the impact, for the report
pic_libre, t_pic_libre, dist_min = 0.0, 0.0, 1e9
tcp_arret = None
tcp_d1 = None


def demarre_approche(t):
    """Build the launch profile from the error the camera measures right now."""
    global traj, e0
    vision.CIBLE_ACTIVE = "d3"
    time.sleep(0.05)                       # let one frame refresh etat["erreur"]
    e0 = np.asarray(etat["err_d3"], float)
    traj = trajectory.Trajectoire(e0, V_TRAJ, T1, t0=0.0,
                                  accorde=True, a_max=A_MAX, sans_decel=True)
    print(f"\n[approche] {traj}")
    print(f"[approche] distance mesuree {1000*np.linalg.norm(e0):.0f} mm")


t_start = time.perf_counter()
if phase == APPROCHE:
    demarre_approche(0.0)

try:
    while True:
        t_cycle = rtde_c.initPeriod()
        t = time.perf_counter() - t_start
        tau = t - t_phase
        n_cycles += 1

        # ---- estimation ----
        est.predict(lim.a_applied, time.perf_counter())
        if etat["t"] > t_last:
            seuil = NIS_LIBRE if phase in (LIBRE, RETOUR) else 25.0
            est.update(etat["theta"], etat["t"], seuil)
            t_last = etat["t"]
        th, thd = est.theta, est.theta_dot
        pic_ballant = max(pic_ballant, float(np.max(np.abs(th))))

        # ---- measurements ----
        err_d1 = np.asarray(etat["err_d1"], float)
        err_d3 = np.asarray(etat["err_d3"], float)
        pos = np.asarray(etat.get("pos_monde", np.zeros(2)), float)
        tcp = np.array(rtde_r.getActualTCPPose()[:2])
        v_tcp = np.array(rtde_r.getActualTCPSpeed()[:2])

        # ---- safety ----
        if phase != RETOUR:
            if time.perf_counter() - etat["t"] > AGE_MAX:
                raison = "vision perdue"; break
            if etat["vus"][1]:
                t_charge_vue = time.perf_counter()
            elif time.perf_counter() - t_charge_vue > CHARGE_PERDUE_MAX:
                raison = "marqueur de charge (12) perdu"; break
            if etat["vus"][0]:
                t_ref_vue = time.perf_counter()
            elif time.perf_counter() - t_ref_vue > REF_PERDUE_MAX:
                raison = "marqueur de reference (8) perdu"; break
        if np.max(np.abs(th)) > THETA_MAX:
            raison = f"ballant {np.degrees(np.max(np.abs(th))):.1f} deg"; break
        if np.any(np.abs(tcp - p_start) > COURSE_MAX):
            raison = f"hors course {np.round(tcp - p_start, 3)}"; break
        if t > TIMEOUT:
            raison = "timeout"; break

        # ---- phase logic ----
        v_ref = np.zeros(2)
        v_fb = np.zeros(2)

        if phase == ALLER:
            erreur = err_d1
            if FEEDBACK:
                proche = np.linalg.norm(erreur) < SEUIL_INTEG
                v_fb = (feedback(erreur, th, thd, gains, V_FB_MAX)
                        + integ(erreur, proche))
            if fin_aller(erreur, th, thd, v_tcp, t):
                tcp_d1 = rtde_r.getActualTCPPose()
                journal_phases.append((NOM_PHASE[phase], tau))
                phase, t_phase = APPROCHE, t
                integ.reset()
                demarre_approche(t)

        elif phase == APPROCHE:
            erreur = err_d3
            p_ref, v_ref, _ = traj(tau)
            err_fb = erreur - (e0 - p_ref)          # deviation from the plan
            if FEEDBACK:
                v_fb = feedback(err_fb, th, thd, gains, V_FB_MAX)
            if traj.terminee(tau):
                journal_phases.append((NOM_PHASE[phase], tau))
                phase, t_phase = LIBRE, t
                lim.regle(a_max=A_STOP, jerk_max=None)   # brake hard, and say so
                tcp_arret = tcp.copy()
                print(f"\n[arret] v = {np.linalg.norm(v_tcp):.3f} m/s   "
                      f"theta = {np.degrees(np.abs(th)).round(2)} deg   "
                      f"distance au bac {1000*etat['dist_origine']:.0f} mm")

        elif phase == LIBRE:
            erreur = err_d3
            # nothing commanded: the load is on its own, this is the measurement
            n_th = float(np.linalg.norm(th))
            if n_th > pic_libre:
                pic_libre, t_pic_libre = n_th, tau
            dist_min = min(dist_min, etat["dist_origine"])
            if tau > T_LIBRE:
                journal_phases.append((NOM_PHASE[phase], tau))
                print(f"\n[libre] pic {np.degrees(pic_libre):.2f} deg a "
                      f"{t_pic_libre:.3f} s (T/4 = {T1/4:.3f} s)   "
                      f"approche mini {1000*dist_min:.0f} mm")
                raison = "phase libre terminee"
                break

        else:                                        # RETOUR
            erreur = err_d1
            if traj_ret is None:
                raison = "pas de plan de retour"; break
            _, v_ref, _ = traj_ret(tau)
            if traj_ret.terminee(tau):
                journal_phases.append((NOM_PHASE[phase], tau))
                raison = "plan de retour termine"
                break

        v = lim(v_ref + v_fb)
        a_cmd = A_STOP if phase == LIBRE else A_MAX

        if DRY_RUN:
            print(f"\r[{NOM_PHASE[phase]:14s}] e {1000*erreur[0]:+6.0f},"
                  f"{1000*erreur[1]:+6.0f} mm | th "
                  f"{np.degrees(th[0]):+5.1f},{np.degrees(th[1]):+5.1f} deg | "
                  f"v {v[0]:+.3f},{v[1]:+.3f}", end="", flush=True)
        else:
            rtde_c.speedL([v[0], v[1], 0.0, 0.0, 0.0, 0.0], a_cmd, Ts)

        w.writerow([f"{t:.4f}", phase] + [f"{x:.5f}" for x in
                   (*erreur, *th, *thd, etat.get("yaw", 0.0),
                    *pos, etat.get("dist_origine", 0.0),
                    *v_ref, *v_fb, *v, *tcp)] +
                   [f"{etat['theta'][0]:.5f}", f"{etat['theta'][1]:.5f}"])

        if n_cycles % 100 == 0 and not DRY_RUN:
            print(f"\r[{NOM_PHASE[phase]:14s}] |e| "
                  f"{1000*np.linalg.norm(erreur):5.0f} mm | th "
                  f"{np.degrees(th[0]):+5.1f},{np.degrees(th[1]):+5.1f} deg | "
                  f"bac {1000*etat['dist_origine']:5.0f} mm", end="", flush=True)

        rtde_c.waitPeriod(t_cycle)

except KeyboardInterrupt:
    raison = "interruption clavier"

finally:
    if not DRY_RUN:
        rtde_c.speedStop()
    time.sleep(0.2)

    # retour a D1 par les encodeurs, puis observation du yaw qui s'amortit
    if not DRY_RUN and tcp_d1 is not None:
        print("\nretour a D1 par les encodeurs...")
        rtde_c.moveL(tcp_d1, 0.15, 0.5)
        print(f"retour termine, observation du yaw pendant {T_OBSERV:.0f} s...")
        t_obs = time.perf_counter()
        while time.perf_counter() - t_obs < T_OBSERV:
            t = time.perf_counter() - t_start
            est.predict(np.zeros(2), time.perf_counter())
            if etat["t"] > t_last:
                est.update(etat["theta"], etat["t"], NIS_LIBRE)
                t_last = etat["t"]
            tcp = np.array(rtde_r.getActualTCPPose()[:2])
            w.writerow([f"{t:.4f}", RETOUR] + [f"{x:.5f}" for x in
                       (*np.asarray(etat["err_d1"], float),
                        *est.theta, *est.theta_dot, etat.get("yaw", 0.0),
                        *np.asarray(etat.get("pos_monde", np.zeros(2)), float),
                        etat.get("dist_origine", 0.0),
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, *tcp)] +
                       [f"{etat['theta'][0]:.5f}", f"{etat['theta'][1]:.5f}"])
            time.sleep(0.01)

    log.close()

    duree = time.perf_counter() - t_start
    print(f"\n\narret: {raison}")
    for nom, d in journal_phases:
        print(f"  {nom:16s} {d:6.2f} s")
    print(f"\nballant maximal      {np.degrees(pic_ballant):6.2f} deg")
    print(f"pic en phase libre   {np.degrees(pic_libre):6.2f} deg a "
          f"{t_pic_libre:.3f} s   (T_d/4 = {T1/4:.3f} s)")
    if dist_min < 1e8:
        print(f"approche mini du bac {1000*dist_min:6.1f} mm")
        print("  un pic nettement avant T_d/4 et une approche proche de zero "
              "signent le contact;")
        print("  un pic a T_d/4 pile veut dire que la charge n'a rien touche.")
    print(f"ballant final        {np.degrees(np.max(np.abs(est.theta))):6.2f} deg")
    e_fin = np.asarray(etat.get("err_d1", np.zeros(2)), float)
    print(f"erreur finale a D1   {1000*np.linalg.norm(e_fin):6.1f} mm")
    print(f"duree                {duree:6.2f} s   "
          f"frequence {n_cycles/max(duree, 1e-9):5.0f} Hz")
    print(f"saturation {100*lim.taux_saturation:.1f} %   "
          f"mesures rejetees {est.x.n_rejets}/{est.y.n_rejets}")
    print(f"journal ecrit: {NOM_CSV}")

    if not DRY_RUN:
        if input("\nretour a la pose de depart ? [o/N] ").strip().lower() == "o":
            rtde_c.moveL(POSE_DEPART, 0.05, 0.2)
            print("retour termine.")

    vision.stop()
    rtde_c.stopScript()