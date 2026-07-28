# -*- coding: utf-8 -*-
"""
Test hardware-in-the-loop : robot REEL (URSim ou UR10), pendule SIMULE.

Le pendule n'existe pas dans le simulateur ni sur le banc tant que la camera
n'est pas prete. Ici il est integre dans le code, entraine par l'acceleration
REELLE du robot mesuree par RTDE. La boucle de commande est celle qui tournera
au banc, sans aucune modification.

Ce que ce test valide :
  - la boucle complete a 125 Hz (lecture, calcul, envoi, cadencement)
  - le generateur de trajectoire et l'input shaper
  - l'integration acceleration -> vitesse et les saturations
  - les securites (course, angle, duree)

Ce qu'il ne valide pas : la chaine de mesure par vision, et le comportement du
vrai pendule.

Usage :
    python controle_hil.py                 -> URSim (127.0.0.1)
    python controle_hil.py 192.168.0.100   -> vrai robot
"""

import sys
import time
import csv
import os
import numpy as np
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

# ============================ CONFIG ============================

ROBOT_IP = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
AXE      = 0                      # 0 = X, 1 = Y, 2 = Z
DT       = 0.008                  # 125 Hz (CB3)

# --- Physique du pendule simule ---
G       = 9.81
L_CABLE = 0.60                    # longueur du cable [m] (banc)
ZETA_N  = 0.00228                 # amortissement naturel mesure
OMEGA   = np.sqrt(G / L_CABLE)

# --- Controleur (gains calcules pour l = 0.60 m) ---
K = np.array([1.000, 2.346, -6.793, -3.189])

# --- Trajectoire ---
X_REF   = 0.15                    # deplacement demande [m] - petit pour un 1er essai
V_PROF  = 0.10                    # vitesse de croisiere du profil [m/s]
A_PROF  = 0.30                    # acceleration du profil [m/s^2]
T0      = 1.0                     # instant de depart [s]
SHAPING = False                   # False pour comparer sans shaper

# --- Securites ---
A_MAX   = 0.5                     # acceleration commandee max [m/s^2]
V_MAX   = 0.20                    # vitesse commandee max [m/s]
X_MIN, X_MAX = -0.05, 0.35        # limites de course relatives [m]
THETA_MAX = np.radians(25)
DUREE   = 15.0

# ========================= PREPARATION ==========================

# Shaper ZV : deux impulsions espacees d'une demi-periode
Kz = np.exp(-ZETA_N * np.pi / np.sqrt(1 - ZETA_N**2))
A1, A2 = 1 / (1 + Kz), Kz / (1 + Kz)
TD = np.pi / (OMEGA * np.sqrt(1 - ZETA_N**2))
print(f"Pendule simule : l = {L_CABLE} m, omega = {OMEGA:.3f} rad/s, T = {2*np.pi/OMEGA:.2f} s")
print(f"Shaper ZV      : A1 = {A1:.4f}, A2 = {A2:.4f}, Td = {TD:.3f} s"
      if SHAPING else "Shaper        : desactive")

t_acc = V_PROF / A_PROF
d_acc = 0.5 * A_PROF * t_acc**2
t_cst = (X_REF - 2 * d_acc) / V_PROF
if t_cst < 0:
    raise SystemExit("Profil impossible : augmenter A_PROF ou reduire V_PROF.")


def trapeze(tt):
    """Position, vitesse, acceleration du profil de base."""
    tau = tt - T0
    if tau <= 0:
        return 0.0, 0.0, 0.0
    if tau <= t_acc:
        return 0.5 * A_PROF * tau**2, A_PROF * tau, A_PROF
    if tau <= t_acc + t_cst:
        return d_acc + V_PROF * (tau - t_acc), V_PROF, 0.0
    if tau <= 2 * t_acc + t_cst:
        tr = tau - t_acc - t_cst
        return (d_acc + V_PROF * t_cst + V_PROF * tr - 0.5 * A_PROF * tr**2,
                V_PROF - A_PROF * tr, -A_PROF)
    return X_REF, 0.0, 0.0


def reference(tt):
    if not SHAPING:
        return trapeze(tt)
    p1, v1, a1 = trapeze(tt)
    p2, v2, a2 = trapeze(tt - TD)
    return A1 * p1 + A2 * p2, A1 * v1 + A2 * v2, A1 * a1 + A2 * a2


rtde_c = RTDEControlInterface(ROBOT_IP)
rtde_r = RTDEReceiveInterface(ROBOT_IP)
x0 = rtde_r.getActualTCPPose()[AXE]
print(f"Connecte. Position initiale axe {AXE} : {x0:.4f} m")

os.makedirs("logs", exist_ok=True)
fname = os.path.join("logs", f"hil_{time.strftime('%Y%m%d_%H%M%S')}.csv")
log = open(fname, "w", newline="")
w_csv = csv.writer(log)
w_csv.writerow(["t", "x", "x_dot", "theta", "theta_dot",
                "p_ref", "u", "v_cmd", "a_mesuree"])

# Etat du pendule simule : [theta, theta_dot]
pend = np.array([0.0, 0.0])
v_cmd = 0.0
v_prev = 0.0
a_filt = 0.0
ALPHA = 0.7                        # filtrage de l'acceleration mesuree
arret = None

print(f"Essai : deplacement de {X_REF} m, shaping {'ON' if SHAPING else 'OFF'}. Ctrl+C pour arreter.")
time.sleep(1.0)

# ============================ BOUCLE ============================

t_start = time.perf_counter()
try:
    while True:
        t_cycle = rtde_c.initPeriod()
        t = time.perf_counter() - t_start
        if t > DUREE:
            arret = "duree atteinte"
            break

        # ---------- 1. Etat du chariot : mesure REELLE ----------
        x = rtde_r.getActualTCPPose()[AXE] - x0
        x_dot = rtde_r.getActualTCPSpeed()[AXE]

        # acceleration reelle du robot, par difference finie filtree
        a_mes = (x_dot - v_prev) / DT
        a_filt = ALPHA * a_filt + (1 - ALPHA) * a_mes
        v_prev = x_dot

        # ---------- 2. Pendule SIMULE, entraine par le robot reel ----------
        th, thd = pend
        thdd = (-a_filt * np.cos(th) - G * np.sin(th)) / L_CABLE - 2 * ZETA_N * OMEGA * thd
        pend = pend + DT * np.array([thd, thdd])
        theta, theta_dot = pend

        # ---------- 3. Securites ----------
        if abs(theta) > THETA_MAX:
            arret = f"angle excessif ({np.degrees(theta):.1f} deg)"; break
        if not (X_MIN <= x <= X_MAX):
            arret = f"hors course (x = {x:.3f} m)"; break

        # ---------- 4. Consigne + loi de commande ----------
        p_ref, v_ref, a_ref = reference(t)
        u = a_ref - float(K @ np.array([x - p_ref, x_dot - v_ref, theta, theta_dot]))
        u = float(np.clip(u, -A_MAX, A_MAX))

        # ---------- 5. Integration acceleration -> vitesse ----------
        v_cmd = float(np.clip(v_cmd + u * DT, -V_MAX, V_MAX))

        # ---------- 6. Envoi ----------
        vec = [0.0] * 6
        vec[AXE] = v_cmd
        rtde_c.speedL(vec, A_MAX, DT)

        w_csv.writerow([f"{t:.4f}", f"{x:.5f}", f"{x_dot:.5f}",
                        f"{theta:.5f}", f"{theta_dot:.5f}", f"{p_ref:.5f}",
                        f"{u:.4f}", f"{v_cmd:.5f}", f"{a_filt:.4f}"])
        print(f"\rt={t:5.2f}s  x={x:+.3f} m  theta={np.degrees(theta):+6.2f} deg  "
              f"u={u:+.3f}  v={v_cmd:+.3f} m/s", end="", flush=True)

        rtde_c.waitPeriod(t_cycle)

except KeyboardInterrupt:
    arret = "interruption clavier"
finally:
    rtde_c.speedStop()
    rtde_c.stopScript()
    log.close()
    print(f"\nArret : {arret}. Donnees dans {fname}")

# ======================== ANALYSE ========================

d = np.genfromtxt(fname, delimiter=",", names=True)
th_deg = np.degrees(d["theta"])
print(f"angle max        : {np.max(np.abs(th_deg)):.2f} deg")
print(f"position finale  : {d['x'][-1]:.4f} m (cible {X_REF})")
print(f"erreur finale    : {1000*abs(d['x'][-1]-X_REF):.1f} mm")
print(f"cadence moyenne  : {1/np.mean(np.diff(d['t'])):.1f} Hz "
      f"(pire ecart {1000*np.max(np.diff(d['t'])):.1f} ms)")

try:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    ax[0].plot(d["t"], d["x"], label="x mesure")
    ax[0].plot(d["t"], d["p_ref"], "--", label="consigne shapee")
    ax[0].set_ylabel("x [m]"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax[1].plot(d["t"], th_deg, color="tab:orange")
    ax[1].set_ylabel("theta simule [deg]"); ax[1].grid(alpha=.3)
    ax[2].plot(d["t"], d["u"], lw=.8, label="u")
    ax[2].plot(d["t"], d["v_cmd"], label="v commandee")
    ax[2].set_ylabel("commande"); ax[2].set_xlabel("t [s]")
    ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)
    fig.suptitle(f"Hardware-in-the-loop : robot reel, pendule simule "
                 f"(shaping {'ON' if SHAPING else 'OFF'})")
    plt.tight_layout(); plt.savefig("hil_resultat.png", dpi=130); plt.show()
except ImportError:
    pass