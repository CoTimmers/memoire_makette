"""Why does the trolley keep moving at the target? Read it off a run.

    python 7_diagnostic.py essai.csv
    python 7_diagnostic.py essai.csv --L 1.11 --fenetre 6

Works on logs from 6_main.py and 7_main.py alike; only the columns common to
both are used.

With K_thetadot = 0 the command at the target reduces to v = K_theta * theta, so
the trolley moving while the load swings is the anti-sway term working, not a
fault. What matters is whether it converges. Three things stop it converging,
and they leave different traces.

1. Bias on theta. If the estimate reads theta_0 instead of zero at rest, the
   command never returns to zero either. Without the integral term the loop
   settles at a standing position error

       e = - K_theta * theta_0 / K_x

   The ratio K_theta / K_x is about 3.3 here, so half a degree of bias becomes
   28 mm of position error. Causes: the suspension was calibrated while the load
   was still moving, OFFSET_PIVOT is off, or the marker plane is not parallel to
   the floor.

2. A limit cycle at the pendulum period. The sway stops decaying and settles at
   a floor, oscillating at T_d. Either the feedback is pumping instead of
   damping, which on one axis only means a sign error in CAM2BASE, or dry
   friction is doing it: the command asks for a few mm/s, the joints hold, break
   free, jump, and that jump re-excites the pendulum.

3. An oscillation much slower than T_d. That one is the position loop, not the
   pendulum: omega_t too high, so the two loops fight.

The script tests for each and says which pattern it sees.
"""

import sys
import numpy as np

G = 9.81


def charge(nom):
    d = np.genfromtxt(nom, delimiter=",", names=True)
    if d.size == 0:
        raise SystemExit(f"{nom}: vide.")
    return d


def periode_dominante(t, x):
    """Period from the mean crossings, in seconds. None if too few."""
    y = x - x.mean()
    s = np.signbit(y)
    croix = np.where(s[:-1] != s[1:])[0]
    if len(croix) < 3:
        return None
    # linear interpolation of each crossing instant
    inst = []
    for i in croix:
        y0, y1 = y[i], y[i + 1]
        f = y0 / (y0 - y1) if y1 != y0 else 0.0
        inst.append(t[i] + f * (t[i + 1] - t[i]))
    return float(2 * np.mean(np.diff(inst)))


def analyse(nom, L=1.11, fenetre=6.0, K_theta=0.975, K_x=0.30):
    d = charge(nom)
    cols = d.dtype.names
    t = d["t"]
    wn = np.sqrt(G / L)
    T_d = 2 * np.pi / wn

    th = np.hypot(d["th_x"], d["th_y"])
    thx, thy = d["th_x"], d["th_y"]
    brut_x = d["th_brut_x"] if "th_brut_x" in cols else thx
    brut_y = d["th_brut_y"] if "th_brut_y" in cols else thy
    v = np.hypot(d["vcmd_x"], d["vcmd_y"])
    e = np.hypot(d["ex"], d["ey"])
    tcp = np.column_stack([d["tcp_x"], d["tcp_y"]])

    fin = t > t[-1] - fenetre
    if fin.sum() < 20:
        raise SystemExit("run trop court pour la fenetre demandee.")

    print(f"\n=== {nom} ===")
    print(f"duree {t[-1]:.1f} s, analyse sur les {fenetre:.0f} dernieres secondes")
    print(f"L = {L} m -> T_d = {T_d:.3f} s, omega_n = {wn:.3f} rad/s\n")

    # ---------- 1. bias ----------
    bx, by = brut_x[fin].mean(), brut_y[fin].mean()
    biais = float(np.hypot(bx, by))
    err_implicite = K_theta * biais / K_x if K_x > 1e-9 else np.inf
    print("1. BIAIS SUR THETA")
    print(f"   theta brut moyen   {np.degrees(bx):+6.2f}, {np.degrees(by):+6.2f} deg"
          f"   (norme {np.degrees(biais):.2f} deg)")
    print(f"   erreur de position que ce biais impose : "
          f"{1000*err_implicite:.0f} mm")
    print(f"   erreur mesuree en fin de run           : {1000*e[-1]:.0f} mm")
    if np.degrees(biais) > 0.4:
        print("   -> biais significatif. Refaire la calibration de la suspension")
        print("      charge parfaitement immobile, et verifier OFFSET_PIVOT.")
    else:
        print("   -> pas de biais notable.")

    # ---------- 2. residual sway and its period ----------
    amp_deb = float(np.abs(th[t < t[0] + fenetre]).max())
    amp_fin = float(np.abs(th[fin] - th[fin].mean()).max())
    per_x = periode_dominante(t[fin], thx[fin])
    per_y = periode_dominante(t[fin], thy[fin])
    print("\n2. BALLANT RESIDUEL")
    print(f"   amplitude au debut {np.degrees(amp_deb):5.2f} deg   "
          f"a la fin {np.degrees(amp_fin):5.2f} deg   "
          f"rapport {amp_fin/max(amp_deb,1e-9):.2f}")
    negligeable = np.degrees(amp_fin) < 0.15
    if negligeable:
        print("   le ballant residuel est negligeable, la periode n'a pas de sens")
    for nom_axe, per in (() if negligeable else (("x", per_x), ("y", per_y))):
        if per is None:
            print(f"   axe {nom_axe}: pas d'oscillation nette")
            continue
        r = per / T_d
        if 0.75 < r < 1.35:
            verdict = "au periode du pendule -> boucle qui pompe, ou frottement"
        elif r >= 1.35:
            verdict = f"{r:.1f} fois plus lent que le pendule -> boucle de position"
        else:
            verdict = "plus rapide que le pendule -> bruit, pas une oscillation"
        print(f"   axe {nom_axe}: periode {per:.2f} s  ({r:.2f} x T_d)  {verdict}")
    if np.degrees(amp_fin) > 0.5 and amp_fin > 0.4 * amp_deb:
        print("   -> le ballant ne decroit pas. Verifier le signe de CAM2BASE")
        print("      axe par axe: pousser la charge vers +X, theta[0] doit suivre.")

    # ---------- 3. does the robot follow? ----------
    dt = np.gradient(t)
    course_cmd = float(np.sum(v[fin] * dt[fin]))
    course_reelle = float(np.sum(np.linalg.norm(np.diff(tcp[fin], axis=0), axis=1)))
    suivi = course_reelle / course_cmd if course_cmd > 1e-9 else np.nan
    v_moy = float(v[fin].mean())
    print("\n3. SUIVI DU ROBOT (frottement sec)")
    print(f"   vitesse commandee moyenne {1000*v_moy:6.1f} mm/s")
    print(f"   chemin commande {1000*course_cmd:6.1f} mm   "
          f"parcouru {1000*course_reelle:6.1f} mm   ratio {suivi:.2f}")
    if course_cmd > 1e-4 and suivi < 0.6:
        print("   -> le robot ne suit pas la commande. Frottement sec probable:")
        print("      le TCP colle, decroche, saute, et le saut relance le pendule.")
        print("      Remedes: monter K_I, ou monter omega_t pour sortir de la")
        print("      zone de tres basse vitesse plus vite.")
    elif not np.isnan(suivi):
        print("   -> suivi correct.")

    # ---------- summary ----------
    print("\nRESUME")
    causes = []
    if np.degrees(biais) > 0.4:
        causes.append("biais sur theta")
    if np.degrees(amp_fin) > 0.5 and amp_fin > 0.4 * amp_deb:
        causes.append("ballant entretenu")
    if course_cmd > 1e-4 and suivi < 0.6:
        causes.append("frottement sec")
    if causes:
        print("   " + ", ".join(causes))
    else:
        print("   rien d'anormal: le chariot bouge parce que la charge oscille,")
        print("   et les deux convergent. C'est le comportement attendu.")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        raise SystemExit(__doc__.split("\n\n")[1])
    opt = {}
    for i, a in enumerate(sys.argv):
        if a == "--L":
            opt["L"] = float(sys.argv[i + 1])
        if a == "--fenetre":
            opt["fenetre"] = float(sys.argv[i + 1])
    for nom in args:
        analyse(nom, **opt)