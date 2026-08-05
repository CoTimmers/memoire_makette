"""Superimpose several runs and compare them.

    python compare_essais.py a.csv b.csv c.csv
    python compare_essais.py                    -> asks for the names
    python compare_essais.py *.csv              -> all of them

Any number of files can be given. The legend uses the file name without its
extension, so name the runs meaningfully when main_vision.py asks.

The two sway axes are drawn separately, solid for x and dashed for y, one
colour per run. The magnitude alone would hide the shape of the motion: a sway
confined to one axis and a conical sway, where the load traces a circle, give
the same magnitude but call for different explanations.

The yaw panel appears only if the runs carry that column. Yaw is not a
controlled variable, since the tool never rotates and a cable transmits no
torque about its own axis. It is plotted to show how much the crate turns on
its own during transport, which decides whether it can still be set down
aligned with the target zone.
"""

import sys
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

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

avec_yaw = all("yaw" in d.dtype.names for _, d in donnees)
if not avec_yaw:
    print("colonne yaw absente d'au moins un essai: panneau yaw omis.")

panneaux = 4 if avec_yaw else 3
fig, ax = plt.subplots(panneaux, 1, figsize=(11, 2.8 * panneaux), sharex=True)
i_err, i_yaw, i_vit = 1, 2, panneaux - 1     # i_yaw unused when avec_yaw is False

couleurs = plt.rcParams["axes.prop_cycle"].by_key()["color"]

entete = (f"\n{'essai':20s} {'pic x':>8s} {'pic y':>8s} {'residuel':>9s} "
          f"{'err fin':>9s} {'duree':>7s} {'v max':>7s}")
if avec_yaw:
    entete += f" {'yaw fin':>8s} {'d yaw':>7s}"
print(entete)
print("-" * (len(entete) - 1))

for k, (nom, d) in enumerate(donnees):
    c = couleurs[k % len(couleurs)]
    t = d["t"]
    thx, thy = np.degrees(d["th_x"]), np.degrees(d["th_y"])
    th = np.hypot(thx, thy)                             # sway magnitude
    e = 1000 * np.hypot(d["ex"], d["ey"])               # position error, mm
    v = np.hypot(d["vcmd_x"], d["vcmd_y"])              # commanded speed

    fenetre = t > t[-1] - 2.0                           # last two seconds
    ligne = (f"{nom:20s} {np.abs(thx).max():7.2f}d {np.abs(thy).max():7.2f}d "
             f"{th[fenetre].max():8.2f}d {e[-1]:8.1f}mm {t[-1]:6.2f}s "
             f"{v.max():7.3f}")

    ax[0].plot(t, thx, color=c, lw=1.3, label=f"{nom}  x")
    ax[0].plot(t, thy, color=c, lw=1.1, ls="--", label=f"{nom}  y")
    ax[i_err].plot(t, e, color=c, lw=1.3, label=nom)
    ax[i_vit].plot(t, v, color=c, lw=1.0, label=nom)

    if avec_yaw:
        # unwrap first: a crate sitting near +-180 deg would otherwise jump
        yaw = np.degrees(np.unwrap(d["yaw"]))
        yaw = yaw - yaw[0]                              # drift from the start
        ax[i_yaw].plot(t, yaw, color=c, lw=1.2, label=nom)
        ligne += f" {yaw[-1]:+7.1f}d {np.abs(yaw).max():6.1f}d"

    print(ligne)

ax[0].set_ylabel("ballant [deg]")
ax[0].axhline(0, color="k", lw=.5)
ax[0].legend(fontsize=8, ncol=len(donnees))
ax[i_err].set_ylabel("erreur de position [mm]")
ax[i_vit].set_ylabel("vitesse commandee [m/s]")
ax[i_vit].set_xlabel("t [s]")
if avec_yaw:
    ax[i_yaw].set_ylabel("yaw, ecart au depart [deg]")
    ax[i_yaw].axhline(0, color="k", lw=.5)
for a in ax:
    a.grid(alpha=.3)

fig.suptitle("Comparaison des essais")
plt.tight_layout()

sortie = "comparaison_" + "_".join(nom for nom, _ in donnees[:3]) + ".png"
plt.savefig(sortie, dpi=130)
print(f"\nfigure: {sortie}")
plt.show()