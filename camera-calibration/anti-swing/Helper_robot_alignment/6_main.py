"""Anti-sway: bring the load into the target zone, damping the sway on the way.

The target is not a TCP position. The camera measures, in robot base axes, the
vector from the cable attachment point of the load to the centre of the target
zone; that vector is the error the controller cancels. The TCP places itself,
because at convergence the sway is zero and the load hangs right under the tool.

Nothing here needs the transform between the robot base and the world: only the
orientation mapping CAM2BASE, which is constant since the tool does not rotate.
"""

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

V_MAX, A_MAX, JERK_MAX = 0.3, 0.8, 8.0
V_FB_MAX   = 0.25
COURSE_MAX = 0.50                  # excursion allowed from the start pose [m]
THETA_MAX  = np.radians(20)
AGE_MAX    = 0.30                  # max age of a vision measurement [s]
TIMEOUT    = 60.0
DRY_RUN    = False                 # True = no motion, display only

rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
POSE_DEPART = rtde_r.getActualTCPPose()
p_start = np.array(POSE_DEPART[:2])
print("pose de depart:", [round(x, 3) for x in POSE_DEPART])

etat = {}
vision.AFFICHAGE = True
vision.start(etat, L_CABLE)
print("bac immobile pour la calibration du pivot...")
while not etat["pret"]:
    time.sleep(0.1)
print("pret.\n")

gains = Gains(L=L_CABLE, zeta_cl=0.7, omega_t=1.0)
est   = SwayEstimator2D(L=L_CABLE, Ts=Ts)
lim   = CommandLimiter(Ts, V_MAX, A_MAX, JERK_MAX)
fin   = FinDeMouvement(eps_x=0.010, eps_theta=np.radians(1.0),
                       eps_theta_dot=0.02, eps_v=0.010, T_dwell=0.5)

t_last = 0.0
t_start = time.perf_counter()
raison = "cible atteinte"

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

        # ---- error, measured by vision, already in base axes ----
        erreur = np.asarray(etat["erreur"], float)      # target - load
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

        # ---- control ----
        v = lim(feedback(erreur, th, thd, gains, V_FB_MAX))

        if DRY_RUN:
            print(f"\re {1000*erreur[0]:+6.0f},{1000*erreur[1]:+6.0f} mm | "
                  f"th {np.degrees(th[0]):+5.1f},{np.degrees(th[1]):+5.1f} deg | "
                  f"v {v[0]:+.3f},{v[1]:+.3f} m/s | "
                  f"{'IN ' if etat.get('dans_cible') else 'out'}",
                  end="", flush=True)
        else:
            rtde_c.speedL([v[0], v[1], 0.0, 0.0, 0.0, 0.0], A_MAX, Ts)

        # ---- finished: inside the zone and everything at rest ----
        if etat.get("dans_cible") and fin(erreur, th, thd, v_tcp, t):
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
    print(f"\n\narret: {raison}")
    print(f"erreur finale {1000*np.linalg.norm(np.asarray(etat['erreur'])):.1f} mm"
          f"   ballant {np.degrees(np.max(np.abs(est.theta))):.2f} deg")
    print(f"saturation {100*lim.taux_saturation:.1f} %   "
          f"mesures rejetees {est.x.n_rejets}/{est.y.n_rejets}")
    vision.stop()
    rtde_c.stopScript()