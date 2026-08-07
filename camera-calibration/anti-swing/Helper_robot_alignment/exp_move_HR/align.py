"""Alignement du helper: poste calcule depuis phi, lancement par arret net.

Le helper pend au cable et porte deux murs. Le bac est fixe et definit le
repere monde. L'angle entre les deux reperes est phi, nul quand ils coincident.
La vision publie phi directement (= -yaw, la convention de correction voulue).

Les points de travail, recalcules a chaque image tant que rien n'a touche
--------------------------------------------------------------------------
    D1   x = X_D1                          mise en place, en recul
         y = -sin(phi) * D_MUR + MARGE_Y

    D3   x = X_D3                          arret net, tout pres du bac
         y = le meme que D1

    D4   fixe                              repli apres le choc

D_MUR est la distance du pivot au bout du mur qui va frapper. Le terme en
sin(phi) place le pivot sur la ligne telle que ce bout de mur se presente en
face du bac quand on avance selon -x. MARGE_Y ecarte de quelques centimetres
pour que rien ne touche avant l'arret.

MARGE_Y est ajoutee telle quelle, sans suivre le signe de phi: a phi = +90 elle
rapproche de l'axe (y = -0.30 + 0.04 = -0.26), a phi = -90 elle en eloigne
(y = +0.34). C'est asymetrique et c'est assume tant que la formule n'est
validee que d'un cote.

D1 et D3 ayant le meme y, l'approche est une ligne droite selon -x, de X_D1 a
X_D3. C'est cette course qui doit loger la rampe accordee, qui couvre
V_TRAJ * T1 / 2. Le script imprime les deux longueurs avant de bouger et
Trajectoire refuse si la rampe est trop longue.

Les phases
----------
    0  ALLER      feedback vers D1. D1 et D3 suivent phi: rien n'a encore
                  touche le helper, son orientation ne change pas toute seule.
    1  APPROCHE   rampe accordee vers D3, sans deceleration. Les points sont
                  figes: la trajectoire est construite une fois pour toutes.
    2  LIBRE      arret net. Le chariot s'arrete, le helper garde la vitesse,
                  part en avant et rencontre le bac. Rien n'est commande.

Le repli
--------
Apres le choc le helper tourne vite et son marqueur devient illisible, donc la
vision ne peut plus servir. La pose TCP correspondant a D4 est donc calculee a
l'instant de l'arret net, quand la mesure est encore bonne, et le retour se
fait par moveL sur cette pose, aux encodeurs.

Ce calcul suppose le cable presque vertical a cet instant: on ajoute err_d4 a
la pose TCP courante, ce qui n'est exact que si le pivot est a l'aplomb de
l'outil. La ligne [arret] imprime theta; a 2 degres l'erreur induite vaut
3,5 cm, au dela il faut en tenir compte.
"""

import numpy as np
import time
import csv
from importlib import import_module

import rtde_control
import rtde_receive

vision     = import_module("7_vision_2")
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
L_VRAI   = 1.0
L_MODELE = 1.0
Ts       = 1 / 500

# --- geometrie: D1 et D3 depuis phi ---
D_MUR      = 0.30       # pivot -> bout du mur qui frappe [m]
MARGE_Y    = 0.04       # ecart lateral avant l'arret [m]
X_D1       = 0.30       # recul de la mise en place [m]
X_D3       = -0.1       # arret net, tout pres du bac [m]
X_SUIT_COS = False      # True: X_D1 et X_D3 multiplies par cos(phi)
D4         = np.array([0.328, -0.328, 0.0])   # repli fixe

# --- mouvement ---
T1      =  2* np.pi * np.sqrt(L_MODELE / 9.81)
V_TRAJ  = 0.20          # vitesse de croisiere du plan: la variable de l'essai
T_LIBRE = 0.8           # duree de la phase libre apres l'arret [s]

ZETA_CL, OMEGA_T = 0.3, 0.5
K_I, I_MAX  = 0.4, 0.06
SEUIL_INTEG = 0.10

V_MAX, A_MAX, JERK_MAX = 0.5, 1.5, 32.0
A_STOP     = 4.0
V_FB_MAX   = 0.25
COURSE_MAX = 0.60
THETA_MAX  = np.radians(20)
AGE_MAX    = 0.30
REF_PERDUE_MAX    = 3.0
CHARGE_PERDUE_MAX = 3.0
TIMEOUT    = 120.0
T_OBSERV   = 30.0        # enregistrement apres le repli [s]
NIS_LIBRE  = 400.0
DRY_RUN    = False       # commencer par True: verifier D1 et D3 a l'ecran

V_ALLER = 0.15 

ALLER, APPROCHE, LIBRE, REPLI = 0, 1, 2, 3
NOM_PHASE = {ALLER: "aller D1", APPROCHE: "approche D3",
             LIBRE: "libre", REPLI: "repli D4"}

def points(phi):
    """D1 et D3 en axes monde, origine au bac, pour l'angle phi.

    x est constant: D1 en recul, D3 tout pres du bac, l'approche est une ligne
    droite selon -x. y suit la projection du mur, cos(phi - 90) = sin(phi),
    plus la marge qui evite tout contact avant l'arret net.
    """
    y = - D_MUR * np.cos(phi - np.pi / 2) + MARGE_Y
    return (np.array([X_D1, y, 0.0]),
            np.array([X_D3, y, 0.0]))


# ---------------- journal ----------------
_def = f"align_v{int(1000*V_TRAJ)}_{time.strftime('%H%M%S')}"
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
vision.D4 = D4
vision.CIBLE_ACTIVE = "d1"
vision.start(etat, L_VRAI)
print("helper immobile pour la calibration de la suspension...")
while not etat["pret"]:
    time.sleep(0.1)

phi0 = float(etat["phi"])
d1_0, d3_0 = points(phi0)
course = float(np.linalg.norm(d1_0[:2] - d3_0[:2]))
rampe = V_TRAJ * T1 / 2
portee = V_TRAJ * np.sqrt(L_MODELE / 9.81)

print(f"\nphi initial {np.degrees(phi0):+.1f} deg")
print(f"D1 {np.round(1000*d1_0[:2]).astype(int)} mm   "
      f"D3 {np.round(1000*d3_0[:2]).astype(int)} mm   "
      f"D4 {np.round(1000*D4[:2]).astype(int)} mm")
print(f"course D1->D3 {1000*course:.0f} mm   "
      f"la rampe accordee en couvre {1000*rampe:.0f} mm")
if rampe > course:
    print(f"  la rampe est plus longue que la course: Trajectoire refusera.")
    print(f"  baisser V_TRAJ sous {2*course/T1:.3f} m/s, ou augmenter X_D1.")
print(f"portee du ballant apres l'arret {1000*portee:.0f} mm, "
      f"depuis X_D3 = {1000*X_D3:.0f} mm "
      f"-> {'contact prevu' if portee > X_D3 else 'PAS DE CONTACT'}")
print("\nVERIFIER a l'ecran que D1 et D3 tombent du bon cote du bac.\n")
vision.SUIVI_PHI = False
time.sleep(0.10)
e0_aller = np.asarray(etat["err_d1"], float)
try:
    traj_aller = trajectory.Trajectoire(e0_aller, V_ALLER, T1, t0=0.0,
                                        accorde=True, a_max=A_MAX)
    print(f"[aller] {traj_aller}")
except ValueError as exc:
    raise SystemExit(f"plan d'aller impossible: {exc}")

# ---------------- blocs de commande ----------------
gains = Gains(L=L_MODELE, zeta_cl=ZETA_CL, omega_t=OMEGA_T)
est   = SwayEstimator2D(L=L_MODELE, Ts=Ts)
lim   = CommandLimiter(Ts, V_MAX, A_MAX, JERK_MAX)
# integ = Integrateur(K_i=K_I, v_max=I_MAX, Ts=Ts)
fin_aller = FinDeMouvement(eps_x=0.030, eps_theta=np.radians(1.5),
                           eps_theta_dot=0.05, eps_v=0.025, T_dwell=0.7)

log = open(NOM_CSV, "w", newline="")
w = csv.writer(log)
w.writerow(["t", "phase", "ex", "ey", "th_x", "th_y", "thd_x", "thd_y",
            "phi", "pos_x", "pos_y", "dist_origine",
            "vref_x", "vref_y", "vfb_x", "vfb_y", "vcmd_x", "vcmd_y",
            "tcp_x", "tcp_y", "d1_x", "d1_y", "d3_x", "d3_y"])

phase, t_phase = ALLER, 0.0
traj = None
e0 = np.zeros(2)
t_last = 0.0
t_ref_vue = t_charge_vue = time.perf_counter()
raison = "sequence terminee"
n_cycles = 0
journal_phases = []
pic_ballant = 0.0
pic_libre, t_pic_libre, dist_min = 0.0, 0.0, 1e9
tcp_d4 = None
d1_c, d3_c = d1_0.copy(), d3_0.copy()
phi_fige = phi0
erreur = np.zeros(2)

t_start = time.perf_counter()

try:
    while True:
        t_cycle = rtde_c.initPeriod()
        t = time.perf_counter() - t_start
        tau = t - t_phase
        n_cycles += 1

        # ---- estimation ----
        est.predict(lim.a_applied, time.perf_counter())
        if etat["t"] > t_last:
            est.update(etat["theta"], etat["t"],
                       NIS_LIBRE if phase == LIBRE else 25.0)
            t_last = etat["t"]
        th, thd = est.theta, est.theta_dot
        pic_ballant = max(pic_ballant, float(np.max(np.abs(th))))

        phi = float(etat["phi"])
        tcp = np.array(rtde_r.getActualTCPPose()[:2])
        v_tcp = np.array(rtde_r.getActualTCPSpeed()[:2])

        # ---- securite ----
        if time.perf_counter() - etat["t"] > AGE_MAX:
            raison = "vision perdue"; break
        if etat["vus"][1]:
            t_charge_vue = time.perf_counter()
        elif time.perf_counter() - t_charge_vue > CHARGE_PERDUE_MAX:
            raison = "marqueur helper (12) perdu"; break
        if etat["vus"][0]:
            t_ref_vue = time.perf_counter()
        elif time.perf_counter() - t_ref_vue > REF_PERDUE_MAX:
            raison = "marqueur bac (8) perdu"; break
        if np.max(np.abs(th)) > THETA_MAX:
            raison = f"ballant {np.degrees(np.max(np.abs(th))):.1f} deg"; break
        if np.any(np.abs(tcp - p_start) > COURSE_MAX):
            raison = f"hors course {np.round(tcp - p_start, 3)}"; break
        if t > TIMEOUT:
            raison = "timeout"; break

        v_ref = np.zeros(2)
        v_fb = np.zeros(2)

        if phase == ALLER:
            erreur = np.asarray(etat["err_d1"], float)
            _, v_ref, _ = traj_aller(tau)
            if traj_aller.terminee(tau):
                phi_fige = phi
                vision.SUIVI_PHI = False
                journal_phases.append((NOM_PHASE[phase], tau))
                print(f"\n[D1] atteint, phi = {np.degrees(phi):+.1f} deg")
                vision.CIBLE_ACTIVE = "d3"
                time.sleep(0.10)
                e0 = np.asarray(etat["err_d3"], float)
                try:
                    traj = trajectory.Trajectoire(e0, V_TRAJ, T1, t0=0.0,
                                                  accorde=True, a_max=A_MAX,
                                                  sans_decel=True)
                except ValueError as exc:
                    raison = f"plan impossible: {exc}"; break
                print(f"[approche] {traj}")
                print(f"[approche] distance mesuree "
                      f"{1000*np.linalg.norm(e0):.0f} mm")
                phase, t_phase = APPROCHE, t

        elif phase == APPROCHE:
            erreur = np.asarray(etat["err_d3"], float)
            p_ref, v_ref, _ = traj(tau)
            err_fb = erreur - (e0 - p_ref)
            v_fb = feedback(err_fb, th, thd, gains, V_FB_MAX)
            if traj.terminee(tau):
                journal_phases.append((NOM_PHASE[phase], tau))
                phase, t_phase = LIBRE, t
                lim.regle(a_max=A_STOP, jerk_max=None)
                print(f"\n[arret] v = {np.linalg.norm(v_tcp):.3f} m/s   "
                      f"theta = {np.degrees(np.abs(th)).round(2)} deg   "
                      f"bac {1000*etat['dist_origine']:.0f} mm")
                # pose TCP du repli, calculee tant que la vision est bonne
                e_d4 = np.asarray(etat["err_d4"], float)
                pose = list(rtde_r.getActualTCPPose())
                pose[0] += float(e_d4[0])
                pose[1] += float(e_d4[1])
                tcp_d4 = pose
                print(f"[repli] D4 a {1000*np.linalg.norm(e_d4):.0f} mm, "
                      f"pose TCP calculee")

        else:                                   # LIBRE
            erreur = np.asarray(etat["err_d3"], float)
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

        v = lim(v_ref + v_fb)
        a_cmd = A_STOP if phase == LIBRE else A_MAX

        if DRY_RUN:
            print(f"\r[{NOM_PHASE[phase]:12s}] phi {np.degrees(phi):+6.1f} | "
                  f"e {1000*erreur[0]:+6.0f},{1000*erreur[1]:+6.0f} mm | "
                  f"D1 {1000*d1_c[0]:+5.0f},{1000*d1_c[1]:+5.0f} "
                  f"D3 {1000*d3_c[0]:+5.0f},{1000*d3_c[1]:+5.0f} | "
                  f"v {v[0]:+.3f},{v[1]:+.3f}", end="", flush=True)
        else:
            rtde_c.speedL([v[0], v[1], 0.0, 0.0, 0.0, 0.0], a_cmd, Ts)

        w.writerow([f"{t:.4f}", phase] + [f"{x:.5f}" for x in
                   (*erreur, *th, *thd, phi,
                    *np.asarray(etat.get("pos_monde", np.zeros(2)), float),
                    etat.get("dist_origine", 0.0),
                    *v_ref, *v_fb, *v, *tcp, *d1_c[:2], *d3_c[:2])])

        if n_cycles % 100 == 0 and not DRY_RUN:
            print(f"\r[{NOM_PHASE[phase]:12s}] "
                  f"|e| {1000*np.linalg.norm(erreur):5.1f}/20 mm | "
                  f"th {np.degrees(np.max(np.abs(th))):4.2f}/1.5 deg | "
                  f"thd {np.max(np.abs(thd)):5.3f}/0.05 | "
                  f"v {1000*np.linalg.norm(v_tcp):5.1f}/15 mm/s",
                  end="", flush=True)

        rtde_c.waitPeriod(t_cycle)

except KeyboardInterrupt:
    raison = "interruption clavier"

finally:
    if not DRY_RUN:
        rtde_c.speedStop()
    time.sleep(0.2)

    # repli sur D4 par les encodeurs, puis observation de phi qui se stabilise
    if not DRY_RUN and tcp_d4 is not None:
        vision.CIBLE_ACTIVE = "d4"
        print("\nrepli sur D4 par les encodeurs...")
        rtde_c.moveL(tcp_d4, 0.15, 0.5)
        print(f"repli termine, observation pendant {T_OBSERV:.0f} s...")
        t_obs = time.perf_counter()
        while time.perf_counter() - t_obs < T_OBSERV:
            t = time.perf_counter() - t_start
            est.predict(np.zeros(2), time.perf_counter())
            if etat["t"] > t_last:
                est.update(etat["theta"], etat["t"], NIS_LIBRE)
                t_last = etat["t"]
            tcp = np.array(rtde_r.getActualTCPPose()[:2])
            w.writerow([f"{t:.4f}", REPLI] + [f"{x:.5f}" for x in
                       (*np.asarray(etat["err_d4"], float),
                        *est.theta, *est.theta_dot, float(etat["phi"]),
                        *np.asarray(etat.get("pos_monde", np.zeros(2)), float),
                        etat.get("dist_origine", 0.0),
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, *tcp,
                        *d1_c[:2], *d3_c[:2])])
            time.sleep(0.01)

    log.close()

    phi_fin = float(etat.get("phi", 0.0))
    print(f"\n\narret: {raison}")
    for nom, d in journal_phases:
        print(f"  {nom:14s} {d:6.2f} s")
    print(f"\nphi initial      {np.degrees(phi0):+7.2f} deg")
    print(f"phi a l'approche {np.degrees(phi_fige):+7.2f} deg")
    print(f"phi final        {np.degrees(phi_fin):+7.2f} deg")
    print(f"correction       {np.degrees(abs(phi0) - abs(phi_fin)):+7.2f} deg")
    print(f"\nballant maximal {np.degrees(pic_ballant):6.2f} deg")
    print(f"pic en libre    {np.degrees(pic_libre):6.2f} deg a "
          f"{t_pic_libre:.3f} s   (T/4 = {T1/4:.3f} s)")
    if dist_min < 1e8:
        print(f"approche mini   {1000*dist_min:6.1f} mm")
        print("  un pic nettement avant T/4 et une approche proche de zero "
              "signent le contact;")
        print("  un pic a T/4 pile veut dire que le helper n'a rien touche.")
    print(f"duree {time.perf_counter()-t_start:6.2f} s   "
          f"saturation {100*lim.taux_saturation:.1f} %   "
          f"rejets {est.x.n_rejets}/{est.y.n_rejets}")
    print(f"journal ecrit: {NOM_CSV}")

    if not DRY_RUN:
        if input("\nretour a la pose de depart ? [o/N] ").strip().lower() == "o":
            rtde_c.moveL(POSE_DEPART, 0.05, 0.2)
            print("retour termine.")

    vision.stop()
    rtde_c.stopScript()