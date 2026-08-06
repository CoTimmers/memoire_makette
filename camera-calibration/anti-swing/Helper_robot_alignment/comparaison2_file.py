"""Superimpose several runs and compare them, for a deposit against a wall.

    python compare_essais.py a.csv b.csv c.csv
    python compare_essais.py                    -> asks for the names
    python compare_essais.py *.csv              -> all of them

What is measured
----------------
The crate ends its travel in contact with a wall, so the residual sway and the
final position error are imposed by that contact, not by the control law. They
say little about the two layers. What the contact does depend on is the speed of
the LOAD when it arrives, and the rotation the impact then induces.

    t_impact    instant of contact, detected on the yaw break
    v_charge    speed of the load at that instant                        [m/s]
    d_yaw       rotation induced by the impact                           [deg]
    pic         peak sway before contact                                 [deg]

Speed of the load
-----------------
Not the tool speed. The load hangs from a cable, so its horizontal position is
the tool position plus L sin(theta), and differentiating gives

    v_charge = v_tcp + L cos(theta) theta_dot

theta and theta_dot come from the Kalman filter, which reconstructs them at the
control rate with the camera delay already compensated. That is far better than
differentiating the 30 Hz vision measurement, which would need smoothing over
more than one video period and would flatten the very peak being measured.

v_tcp is read from the log if the run recorded it (columns vtcp_x, vtcp_y), and
differentiated from the logged TCP position otherwise. Prefer logging it: a
differentiated position is noisier, and the command is not a substitute since it
is what was asked, not what happened.

The formula describes a load hanging free. Once the crate is against the wall it
no longer moves, and the theta_dot the filter still produces is estimator noise
rather than motion, so the curve is cut at the impact.

Runs do not start from the same distance, so the impact speed is also given
normalised by the cruise speed of the plan, which is the only fair comparison
when the travel lengths differ. That correction does not cover everything: a
longer travel also leaves the sway more time to build up.

Set L_CABLE and V_TRAJ below to the values used in the runs.
"""

import sys
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

L_CABLE = 1.11          # cable length used in the runs [m]
V_TRAJ  = 0.15          # cruise speed of the plan, for normalisation [m/s]
LISSAGE = 25            # moving-average width for the derivatives [samples]


def lisse(x, n=LISSAGE):
    """Moving average, correctly normalised at the edges."""
    if n < 2 or len(x) < n:
        return x
    noyau = np.ones(n)
    poids = np.convolve(np.ones_like(x), noyau, mode="same")
    return np.convolve(x, noyau, mode="same") / poids


def derive(x, t, n=LISSAGE):
    """Smoothed time derivative."""
    return lisse(np.gradient(lisse(x, n), t), n)


def vitesse_tcp(d, t):
    """Tool speed: from the log if present, differentiated otherwise."""
    noms = d.dtype.names
    if "vtcp_x" in noms and "vtcp_y" in noms:
        return d["vtcp_x"], d["vtcp_y"], True
    return derive(d["tcp_x"], t), derive(d["tcp_y"], t), False


def instant_impact(t, yaw, t_min=0.5):
    """Instant of contact, from the sharpest break in the yaw.

    The crate turns freely on its cable during transport, slowly. The contact
    makes it pivot against the wall, which shows up as a sudden change of the
    yaw rate. The largest absolute yaw rate is taken as the impact, with the
    first and last second excluded: an impact never happens there, and the
    edges of a smoothed derivative are unreliable.

    Returns (t_impact, index) or (None, None) if nothing stands out.
    """
    masque = (t > t_min) & (t < t[-1] - 1.0)
    if masque.sum() < 3 * LISSAGE:
        return None, None
    dyaw = np.abs(derive(yaw, t))
    i = int(np.argmax(np.where(masque, dyaw, 0.0)))
    # a real impact stands well above the ordinary drift
    fond = np.median(dyaw[masque])
    if dyaw[i] < 4 * max(fond, 1e-6):
        return None, None
    return float(t[i]), i


fichiers = sys.argv[1:]
if not fichiers:
    saisie = input("fichiers CSV a comparer (separes par un espace, "
                   "Entree = tous): ").strip()
    fichiers = saisie.split() if saisie else sorted(glob.glob("*.csv"))

fichiers = [f for f in fichiers if os.path.exists(f)]
if not fichiers:
    raise SystemExit("aucun fichier trouve.")

donnees = []
for f in fichiers:
    d = np.genfromtxt(f, delimiter=",", names=True)
    if d.size == 0:
        print(f"{f}: vide, ignore")
        continue
    donnees.append((os.path.splitext(os.path.basename(f))[0], d))
if not donnees:
    raise SystemExit("aucun fichier exploitable.")

fig, ax = plt.subplots(4, 1, figsize=(11, 11.2), sharex=True)
i_bal, i_vch, i_yaw, i_err = 0, 1, 2, 3

couleurs = plt.rcParams["axes.prop_cycle"].by_key()["color"]

entete = (f"\n{'essai':18s} {'d0':>7s} {'t imp':>7s} {'v charge':>9s} "
          f"{'v/vplan':>8s} {'d yaw':>8s} {'pic av':>8s} {'err fin':>9s}")
print(entete)
print("-" * (len(entete) - 1))

vtcp_logue = True

for k, (nom, d) in enumerate(donnees):
    c = couleurs[k % len(couleurs)]
    t = d["t"]
    thx, thy = np.degrees(d["th_x"]), np.degrees(d["th_y"])
    e = 1000 * np.hypot(d["ex"], d["ey"])
    yaw = np.degrees(np.unwrap(d["yaw"]))
    yaw = yaw - yaw[0]

    # ---- speed of the load ----
    vtcp_x, vtcp_y, logue = vitesse_tcp(d, t)
    vtcp_logue = vtcp_logue and logue
    vch_x = vtcp_x + L_CABLE * np.cos(d["th_x"]) * d["thd_x"]
    vch_y = vtcp_y + L_CABLE * np.cos(d["th_y"]) * d["thd_y"]
    v_charge = np.hypot(vch_x, vch_y)

    d0 = e[0]                                   # travel length of this run
    t_imp, i_imp = instant_impact(t, yaw)

    if i_imp is not None:
        v_imp = v_charge[i_imp]
        apres = t > t_imp
        d_yaw = (np.median(yaw[apres][-int(0.2 * apres.sum()):])
                 if apres.sum() > 10 else yaw[-1]) - yaw[i_imp]
        avant = t <= t_imp
        pic = max(np.abs(thx[avant]).max(), np.abs(thy[avant]).max())
        ligne = (f"{nom:18s} {d0:6.0f}mm {t_imp:6.2f}s {v_imp:8.3f} "
                 f"{v_imp/V_TRAJ:8.2f} {d_yaw:+7.1f}d {pic:7.2f}d "
                 f"{e[-1]:8.1f}mm")
        for a in (i_bal, i_vch, i_yaw, i_err):
            ax[a].axvline(t_imp, color=c, ls=":", lw=1, alpha=.7)
        ax[i_vch].plot([t_imp], [v_imp], "o", color=c, ms=6)
        # the free-pendulum formula stops being valid at the contact
        # fin_v = i_imp + 1
    else:
        ligne = (f"{nom:18s} {d0:6.0f}mm {'--':>6s}  {'--':>8s} {'--':>8s} "
                 f"{'--':>7s}  {'--':>7s}  {e[-1]:8.1f}mm")
        # fin_v = len(t)

    ax[i_bal].plot(t, thx, color=c, lw=1.3, label=f"{nom}  x")
    ax[i_bal].plot(t, thy, color=c, lw=1.1, ls="--", label=f"{nom}  y")
    # ax[i_vch].plot(t[:fin_v], v_charge[:fin_v], color=c, lw=1.2, label=nom)
    ax[i_vch].plot(t, v_charge, color=c, lw=1.2, label=nom)
    ax[i_yaw].plot(t, yaw, color=c, lw=1.2, label=nom)
    ax[i_err].plot(t, e, color=c, lw=1.3, label=nom)

    print(ligne)

print(f"\nv/vplan = vitesse d'impact rapportee a la vitesse de croisiere du "
      f"plan (V_TRAJ = {V_TRAJ} m/s)")
print("pointilles verticaux = instant d'impact detecte sur la rupture du yaw")
print("la vitesse de la charge s'arrete a l'impact: au-dela la formule du "
      "pendule libre ne decrit plus rien")
if not vtcp_logue:
    print("ATTENTION: vtcp absent d'au moins un essai, vitesse outil derivee "
          "de la position TCP (plus bruitee). Ajouter vtcp_x, vtcp_y au log.")

ax[i_bal].set_ylabel("ballant [deg]")
ax[i_bal].axhline(0, color="k", lw=.5)
ax[i_bal].legend(fontsize=8, ncol=len(donnees))
ax[i_vch].set_ylabel("vitesse de la charge [m/s]")
ax[i_yaw].set_ylabel("yaw, ecart au depart [deg]")
ax[i_yaw].axhline(0, color="k", lw=.5)
ax[i_err].set_ylabel("erreur de position [mm]")
ax[i_err].set_xlabel("t [s]")
for a in ax:
    a.grid(alpha=.3)

fig.suptitle("Comparaison des essais - depose au contact")
plt.tight_layout()

sortie = "comparaison_" + "_".join(nom for nom, _ in donnees[:3]) + ".png"
plt.savefig(sortie, dpi=130)
print(f"\nfigure: {sortie}")
plt.show()