"""Damping test: excite the sway, then let the controller kill it.

Why not measure this during a transport
---------------------------------------
During a transport the residual sway depends on the travel length, on how well
the shaped ramps matched the pendulum period, and on whatever the load bumped
into. None of that is the damping. Here the transport is removed: the same
oscillation is induced every time, the tool ends up back where it started, and
the only thing left for the controller to do is to remove the swing.

Sequence, identical for every run:

    1. wait for the load to be still          (same initial conditions)
    2. excite: velocity sinusoid at the pendulum frequency, N_PERIODES cycles
    3. record T_MESURE seconds while the controller damps

The excitation is a velocity sinusoid over a whole number of periods, so its
integral is zero and the tool returns exactly to its starting point. The
controller therefore has no travel to perform: it only has to hold position and
take the energy out of the pendulum. Driving at the natural frequency means a
few centimetres of tool motion produce a large, repeatable sway.

Edit ZETA_CL, OMEGA_T and FEEDBACK below, run, repeat. The CSV is named after
the settings, so runs can be compared afterwards with compare_essais.py.

Run FEEDBACK = False once: that is the natural damping of the pendulum, the
reference every other run is measured against. With zeta = 0.00228 it should
barely decay at all over 20 s, which is what makes the comparison striking.
"""

import numpy as np
import time
import csv
import os
import rtde_control
import rtde_receive
import vision_amortissement as vision
from estimator import SwayEstimator2D
from controller import Gains, feedback, CommandLimiter

# ---------------- what to change between runs ----------------
ZETA_CL  = 0.15                  # closed-loop damping asked for
OMEGA_T  = 0.3                  # position loop bandwidth [rad/s]
FEEDBACK = False                 # False = natural damping, the reference run

# ---------------- fixed for the whole campaign ----------------
ROBOT_IP = "192.168.56.102"
L_CABLE  = 1.08
Ts       = 1 / 500

AXE_EXCITATION = 0              # base axis the sway is induced along
V_EXCITATION   = 0.05           # velocity amplitude of the sinusoid [m/s]
N_PERIODES     = 3              # number of cycles
T_MESURE       = 20.0           # recording after the excitation [s]

V_MAX, A_MAX, JERK_MAX = 0.3, 0.8, 8.0
V_FB_MAX  = 0.25
THETA_MAX = np.radians(25)
AGE_MAX   = 0.30
COURSE_MAX = 0.40

THETA_REPOS = np.radians(0.5)   # load considered still below this
T_REPOS_MAX = 90.0

DOSSIER = "amortissement"

# ---------------------------------------------------------------- setup
omega_n = np.sqrt(9.81 / L_CABLE)
F_EXC   = omega_n / (2 * np.pi)         # drive at the natural frequency
T_EXC   = N_PERIODES / F_EXC

nom = (f"{'fb' if FEEDBACK else 'libre'}"
       f"_z{ZETA_CL:.2f}_w{OMEGA_T:.2f}".replace(".", ""))
os.makedirs(DOSSIER, exist_ok=True)
chemin = os.path.join(DOSSIER, nom + ".csv")
if os.path.exists(chemin):
    i = 2
    while os.path.exists(os.path.join(DOSSIER, f"{nom}_{i}.csv")):
        i += 1
    chemin = os.path.join(DOSSIER, f"{nom}_{i}.csv")

print(f"pendule: omega_n = {omega_n:.3f} rad/s, T_n = {1/F_EXC:.3f} s")
print(f"excitation: +-{V_EXCITATION} m/s a {F_EXC:.3f} Hz pendant "
      f"{T_EXC:.2f} s ({N_PERIODES} periodes)")
print(f"mesure: {T_MESURE:.0f} s   feedback: {FEEDBACK}")
if FEEDBACK:
    print(f"gains: zeta_cl = {ZETA_CL}, omega_t = {OMEGA_T}")
print(f"journal: {chemin}\n")

rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
POSE_DEPART = rtde_r.getActualTCPPose()
p_start = np.array(POSE_DEPART[:2])
print("pose de depart:", [round(x, 3) for x in POSE_DEPART])

etat = {}
vision.AFFICHAGE = False
vision.start(etat, L_CABLE)
print("charge immobile pour la calibration du pivot...")
while not etat["pret"]:
    time.sleep(0.1)
print("pret.\n")

gains = Gains(L=L_CABLE, zeta_cl=ZETA_CL, omega_t=OMEGA_T)
est   = SwayEstimator2D(L=L_CABLE, Ts=Ts)
lim   = CommandLimiter(Ts, V_MAX, A_MAX, JERK_MAX)

# ---------------------------------------------------------------- rest check
print("\nattente de l'immobilisation de la charge...")
t0 = time.perf_counter()
while time.perf_counter() - t0 < T_REPOS_MAX:
    th_repos = np.max(np.abs(etat["theta"]))
    print(f"\r  {np.degrees(th_repos):5.2f} deg   "
          f"({time.perf_counter()-t0:4.1f} s)", end="", flush=True)
    if th_repos < THETA_REPOS:
        break
    time.sleep(0.2)
print(f"\r  conditions initiales: {np.degrees(th_repos):5.2f} deg"
      f"{' ':25s}")
if th_repos > THETA_REPOS:
    vision.stop(); rtde_c.stopScript()
    raise SystemExit("la charge n'est pas immobile, essai annule.")

input("\nEntree pour lancer l'essai: ")

# ---------------------------------------------------------------- run
log = open(chemin, "w", newline="")
w = csv.writer(log)
w.writerow(["t", "phase", "th_x", "th_y", "thd_x", "thd_y",
            "vcmd_x", "vcmd_y", "tcp_x", "tcp_y", "th_brut_x", "th_brut_y"])

t_last = 0.0
t_start = time.perf_counter()
raison = "ok"
T, TH, PHASE = [], [], []

try:
    while True:
        t_cycle = rtde_c.initPeriod()
        t = time.perf_counter() - t_start

        est.predict(lim.a_applied, time.perf_counter())
        if etat["t"] > t_last:
            est.update(etat["theta"], etat["t"])
            t_last = etat["t"]
        th, thd = est.theta, est.theta_dot
        tcp = np.array(rtde_r.getActualTCPPose()[:2])

        if time.perf_counter() - etat["t"] > AGE_MAX:
            raison = "vision perdue"; break
        if not etat["vus"]:
            raison = "marqueur de charge perdu"; break
        if np.max(np.abs(th)) > THETA_MAX:
            raison = f"ballant {np.degrees(np.max(np.abs(th))):.1f} deg"; break
        if np.any(np.abs(tcp - p_start) > COURSE_MAX):
            raison = f"hors course {np.round(tcp - p_start, 3)}"; break
        if t > T_EXC + T_MESURE:
            break

        if t < T_EXC:
            # ---- excitation, open loop, identical every run ----
            phase = 0
            v_raw = np.zeros(2)
            v_raw[AXE_EXCITATION] = V_EXCITATION * np.sin(2 * np.pi * F_EXC * t)
        else:
            # ---- damping: hold the start position, take out the energy ----
            phase = 1
            err_fb = p_start - tcp
            v_raw = (feedback(err_fb, th, thd, gains, V_FB_MAX) if FEEDBACK
                     else np.zeros(2))

        v = lim(v_raw)
        rtde_c.speedL([v[0], v[1], 0.0, 0.0, 0.0, 0.0], A_MAX, Ts)

        w.writerow([f"{t:.4f}", phase] + [f"{x:.5f}" for x in
                   (*th, *thd, *v, *tcp,
                    etat["theta"][0], etat["theta"][1])])
        T.append(t); TH.append(th.copy()); PHASE.append(phase)

        if int(t / Ts) % 100 == 0:
            print(f"\r  {'excitation' if phase == 0 else 'amortissement'}  "
                  f"t = {t:5.1f} s   ballant "
                  f"{np.degrees(np.max(np.abs(th))):5.2f} deg", end="",
                  flush=True)

        rtde_c.waitPeriod(t_cycle)

except KeyboardInterrupt:
    raison = "interruption clavier"

finally:
    rtde_c.speedStop()
    time.sleep(0.2)
    log.close()

# ---------------------------------------------------------------- metrics
T, TH, PHASE = np.array(T), np.array(TH), np.array(PHASE)
print(f"\n\narret: {raison}")

if len(T) > 100 and (PHASE == 1).sum() > 100:
    amp = np.max(np.abs(TH), axis=1)
    mesure = PHASE == 1
    T_m, amp_m = T[mesure], amp[mesure]
    T_m = T_m - T_m[0]

    pic = float(amp_m[:int(0.5 / Ts)].max())      # sway at the end of the drive

    def temps_sous(frac):
        """Last instant the sway is still above frac of its initial value."""
        idx = np.where(amp_m > frac * pic)[0]
        return float(T_m[idx[-1]]) if len(idx) else 0.0

    t50, t10 = temps_sous(0.5), temps_sous(0.10)
    residuel = float(amp_m[-int(2.0 / Ts):].mean())
    predite = np.log(2) / (ZETA_CL * gains.omega_cl)

    print(f"\nballant induit      {np.degrees(pic):6.2f} deg")
    print(f"demi-vie mesuree    {t50:6.2f} s", end="")
    print(f"   predite {predite:.2f} s" if FEEDBACK else "   (amortissement naturel)")
    print(f"decroissance a 10 % {t10:6.2f} s")
    print(f"ballant residuel    {np.degrees(residuel):6.3f} deg  "
          f"({100*residuel/pic:.1f} % du pic)")
    print(f"\njournal: {chemin}")
else:
    print("essai trop court, pas de metriques.")

vision.stop()
rtde_c.stopScript()