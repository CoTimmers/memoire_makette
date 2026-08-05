"""Anti-sway: déplace le TCP vers une cible en amortissant le balancement."""

import numpy as np
import time
import rtde_control
import rtde_receive
import vision
from estimator import SwayEstimator2D
from controller import Gains, feedback, CommandLimiter, FinDeMouvement

ROBOT_IP = "192.168.56.102"
L_CABLE  = 1.17
Ts       = 1 / 125

CIBLE = np.array([-0.581, 0.572])      # cible du TCP en base [m] - A REGLER

V_MAX, A_MAX, JERK_MAX = 0.3, 0.8, 8.0
DRY_RUN = False                        # True = pas de mouvement, affichage seul

rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)

POSE_DEPART = rtde_r.getActualTCPPose()
print("pose de depart:", [round(x, 3) for x in POSE_DEPART])

etat = {}
vision.start(etat, L_CABLE)
print("bac immobile pour la calibration du pivot...")
while not etat["pret"]:
    time.sleep(0.1)
print("pret.\n")

gains = Gains(L=L_CABLE, zeta_cl=0.15, omega_t=1.5)
est   = SwayEstimator2D(L=L_CABLE, Ts=Ts)
lim   = CommandLimiter(Ts, V_MAX, A_MAX, JERK_MAX)
fin   = FinDeMouvement()

t_last = 0.0
try:
    while True:
        t0 = time.perf_counter()

        est.predict(lim.a_applied, t0)

        if etat["t"] > t_last:
            est.update(etat["theta"], etat["t"])
            t_last = etat["t"]

        tcp = np.array(rtde_r.getActualTCPPose()[:2])
        erreur = CIBLE - tcp

        v = lim(feedback(erreur, est.theta, est.theta_dot, gains, V_MAX))

        if DRY_RUN:
            print(f"\re {1000*erreur[0]:+6.0f},{1000*erreur[1]:+6.0f} mm | "
                  f"th {np.degrees(est.theta[0]):+5.1f},"
                  f"{np.degrees(est.theta[1]):+5.1f} deg | "
                  f"v {v[0]:+.3f},{v[1]:+.3f} m/s", end="", flush=True)
        else:
            rtde_c.speedL([v[0], v[1], 0, 0, 0, 0], A_MAX, Ts * 2)

        if fin(erreur, est.theta, est.theta_dot, v, t0):
            print("\ncible atteinte.")
            break

        time.sleep(max(0, Ts - (time.perf_counter() - t0)))

except KeyboardInterrupt:
    pass
finally:
    if not DRY_RUN:
        rtde_c.speedStop()
        time.sleep(0.3)
        print("\nretour a la position de depart...")
        rtde_c.moveL(POSE_DEPART, 0.05, 0.2)
    vision.stop()
    rtde_c.stopScript()
    print("\nstop.")