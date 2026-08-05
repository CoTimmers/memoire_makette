"""Superimpose several runs and compare them.

    python compare_essais.py a.csv b.csv c.csv
    python compare_essais.py                    -> asks for the names
    python compare_essais.py *.csv              -> all of them

Any number of files can be given. The legend uses the file name without its
extension, so name the runs meaningfully when main_vision.py asks.
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

fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
print(f"\n{'essai':22s} {'pic':>8s} {'residuel':>9s} {'err fin':>9s} "
      f"{'duree':>8s} {'v max':>8s}")
print("-" * 70)

for f in fichiers:
    d = np.genfromtxt(f, delimiter=",", names=True)
    if d.size == 0:
        print(f"{f}: vide, ignore")
        continue
    nom = os.path.splitext(os.path.basename(f))[0]
    t = d["t"]
    th = np.degrees(np.hypot(d["th_x"], d["th_y"]))     # sway magnitude
    e = 1000 * np.hypot(d["ex"], d["ey"])               # position error, mm
    v = np.hypot(d["vcmd_x"], d["vcmd_y"])              # commanded speed

    fenetre = t > t[-1] - 2.0                           # last two seconds
    print(f"{nom:22s} {th.max():7.2f}d {th[fenetre].max():8.2f}d "
          f"{e[-1]:8.1f}mm {t[-1]:7.2f}s {v.max():7.3f}")

    ax[0].plot(t, th, label=nom, lw=1.3)
    ax[1].plot(t, e, label=nom, lw=1.3)
    ax[2].plot(t, v, label=nom, lw=1.0)

ax[0].set_ylabel("ballant [deg]")
ax[0].legend(fontsize=9)
ax[0].grid(alpha=.3)
ax[1].set_ylabel("erreur de position [mm]")
ax[1].grid(alpha=.3)
ax[2].set_ylabel("vitesse commandee [m/s]")
ax[2].set_xlabel("t [s]")
ax[2].grid(alpha=.3)
fig.suptitle("Comparaison des essais")
plt.tight_layout()

sortie = "comparaison_" + "_".join(
    os.path.splitext(os.path.basename(f))[0] for f in fichiers[:3]) + ".png"
plt.savefig(sortie, dpi=130)
print(f"\nfigure: {sortie}")
plt.show()