"""Analyse le mouvement du helper pendant un essai de 8_main.py.

    python analyse_helper.py essai8_ff+fb_v150_143022.csv
    python analyse_helper.py a.csv b.csv          -> superpose plusieurs essais
    python analyse_helper.py                      -> demande les noms

Ce que le fichier contient et ce qu'il faut en faire
----------------------------------------------------
8_main.py journalise, pour chaque cycle de la boucle de commande:

    hx, hy      pose du helper dans le repere monde        [m]
    hdx, hdy    deplacement depuis la pose de reference    [m]
    hdyaw       rotation depuis la pose de reference       [rad]
    hbouge      1 des que |hd| depasse SEUIL_BOUGE

La vitesse n'y est pas, il faut la deriver. Attention: la boucle tourne a
500 Hz mais la camera ne fournit une pose neuve qu'a la cadence video, ~30 Hz.
Les colonnes helper sont donc des escaliers, chaque marche repetee une
quinzaine de fois. Deriver ligne a ligne donne un pic a chaque marche et zero
entre, ce qui n'a aucun sens physique. On ne derive donc que sur les instants
ou la mesure change vraiment, puis on lisse.

Le lissage a un cout: il etale le front d'impact. La demi-largeur de la fenetre
est affichee, c'est la resolution temporelle de la vitesse. La prendre trop
large ecrete le pic de vitesse qu'on cherche justement a mesurer.

L'instant du choc retenu est le premier passage de hbouge a 1, c'est-a-dire le
premier depassement de SEUIL_BOUGE. C'est un seuil sur le deplacement, donc il
arrive necessairement un peu apres le contact physique: le helper doit d'abord
bouger de SEUIL_BOUGE. Le decalage vaut a peu pres SEUIL_BOUGE divise par la
vitesse initiale du helper.
"""

import sys
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

# demi-largeur de la fenetre de lissage de la vitesse, en echantillons vision
LISSAGE = 2

COLS_HELPER = ("hx", "hy", "hdx", "hdy", "hdyaw", "hbouge")


def charger(chemin):
    d = np.genfromtxt(chemin, delimiter=",", names=True)
    if d.size == 0:
        return None
    manquantes = [c for c in COLS_HELPER if c not in d.dtype.names]
    if manquantes:
        print(f"{chemin}: colonnes {manquantes} absentes, essai ignore "
              f"(journal produit par une version anterieure de 8_main.py)")
        return None
    return d


def vitesse_helper(t, x, y, demi=LISSAGE):
    """Derive la pose du helper en ne gardant que les mesures vision reelles.

    Retourne (t_mes, vx, vy) aux instants de mesure. Les differences sont
    centrees, donc valides meme si la cadence video n'est pas parfaitement
    reguliere, ce qui est le cas des que la detection rate une image.
    """
    # une mesure neuve = la pose a change depuis la ligne precedente
    neuf = np.ones(len(t), dtype=bool)
    neuf[1:] = (np.diff(x) != 0) | (np.diff(y) != 0)
    idx = np.flatnonzero(neuf)
    if len(idx) < 3:
        return np.array([]), np.array([]), np.array([])

    tm, xm, ym = t[idx], x[idx], y[idx]

    if demi > 0 and len(tm) > 2 * demi + 1:
        noyau = np.ones(2 * demi + 1) / (2 * demi + 1)
        xm = np.convolve(xm, noyau, mode="same")
        ym = np.convolve(ym, noyau, mode="same")
        # les bords sont fausses par le noyau tronque, on les coupe
        tm, xm, ym = tm[demi:-demi], xm[demi:-demi], ym[demi:-demi]

    vx = np.gradient(xm, tm)
    vy = np.gradient(ym, tm)
    return tm, vx, vy


def instant_choc(t, hbouge):
    """Premier passage de hbouge a 1, ou None si le seuil n'est jamais franchi."""
    i = np.flatnonzero(hbouge > 0.5)
    return float(t[i[0]]) if len(i) else None


# ---------------- entree ----------------
fichiers = sys.argv[1:]
if not fichiers:
    saisie = input("fichiers CSV a analyser (separes par un espace, "
                   "Entree = tous): ").strip()
    fichiers = saisie.split() if saisie else sorted(glob.glob("*.csv"))

fichiers = [f for f in fichiers if os.path.exists(f)]
if not fichiers:
    raise SystemExit("aucun fichier trouve.")

donnees = []
for f in fichiers:
    d = charger(f)
    if d is not None:
        donnees.append((os.path.splitext(os.path.basename(f))[0], d))
if not donnees:
    raise SystemExit("aucun essai exploitable.")

# ---------------- figure ----------------
fig, ax = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
couleurs = plt.rcParams["axes.prop_cycle"].by_key()["color"]

entete = (f"\n{'essai':22s} {'t choc':>7s} {'|d| max':>8s} {'|d| fin':>8s} "
          f"{'dx fin':>7s} {'dy fin':>7s} {'yaw fin':>8s} {'|v| max':>8s} "
          f"{'n vis':>6s} {'f vis':>6s}")
print(entete)
print("-" * (len(entete) - 1))

for k, (nom, d) in enumerate(donnees):
    c = couleurs[k % len(couleurs)]
    t = d["t"]
    dx, dy = 1000 * d["hdx"], 1000 * d["hdy"]       # mm
    dn = np.hypot(dx, dy)
    yaw = np.degrees(d["hdyaw"])

    tm, vx, vy = vitesse_helper(t, d["hx"], d["hy"])
    vn = np.hypot(vx, vy) if len(tm) else np.array([])

    tc = instant_choc(t, d["hbouge"])

    # cadence video reelle, utile pour juger la resolution temporelle
    neuf = np.ones(len(t), dtype=bool)
    neuf[1:] = (np.diff(d["hx"]) != 0) | (np.diff(d["hy"]) != 0)
    n_vis = int(neuf.sum())
    f_vis = n_vis / t[-1] if t[-1] > 0 else 0.0

    # ---- panneau 1: deplacement ----
    ax[0].plot(t, dn, color=c, lw=1.4, label=f"{nom}  |d|")
    ax[0].plot(t, dx, color=c, lw=0.9, ls="--", alpha=.7, label=f"{nom}  x")
    ax[0].plot(t, dy, color=c, lw=0.9, ls=":", alpha=.7, label=f"{nom}  y")

    # ---- panneau 2: yaw ----
    ax[1].plot(t, yaw, color=c, lw=1.3, label=nom)

    # ---- panneau 3: vitesse du helper ----
    if len(tm):
        ax[2].plot(tm, 1000 * vn, color=c, lw=1.4, label=f"{nom}  |v|")
        ax[2].plot(tm, 1000 * vx, color=c, lw=0.9, ls="--", alpha=.7)
        ax[2].plot(tm, 1000 * vy, color=c, lw=0.9, ls=":", alpha=.7)

    # ---- panneau 4: trajectoire dans le plan ----
    ax[3].plot(t, 1000 * np.hypot(d["vcmd_x"], d["vcmd_y"]),
               color=c, lw=1.2, label=f"{nom}  v commandee")

    # marque du choc sur tous les panneaux
    if tc is not None:
        for a in ax:
            a.axvline(tc, color=c, lw=1.0, ls="-.", alpha=.6)

    v_max = 1000 * vn.max() if len(vn) else 0.0
    print(f"{nom:22s} "
          f"{(f'{tc:6.2f}s' if tc is not None else '     --'):>7s} "
          f"{dn.max():7.1f}m {dn[-1]:7.1f}m {dx[-1]:+6.1f} {dy[-1]:+6.1f} "
          f"{yaw[-1]:+7.2f}d {v_max:7.1f} {n_vis:6d} {f_vis:5.1f}H")

ax[0].set_ylabel("deplacement helper [mm]")
ax[0].axhline(0, color="k", lw=.5)
ax[0].legend(fontsize=7, ncol=max(1, len(donnees)))
ax[1].set_ylabel("rotation helper [deg]")
ax[1].axhline(0, color="k", lw=.5)
ax[1].legend(fontsize=8)
ax[2].set_ylabel("vitesse helper [mm/s]")
ax[2].axhline(0, color="k", lw=.5)
ax[2].legend(fontsize=7)
ax[3].set_ylabel("vitesse commandee grue [mm/s]")
ax[3].set_xlabel("t [s]")
ax[3].legend(fontsize=8)
for a in ax:
    a.grid(alpha=.3)

fig.suptitle(f"Helper: deplacement, rotation, vitesse   "
             f"(lissage vitesse: +/-{LISSAGE} mesures video)")
plt.tight_layout()

sortie = "helper_" + "_".join(nom for nom, _ in donnees[:3]) + ".png"
plt.savefig(sortie, dpi=130)
print(f"\ntrait mixte vertical = premier depassement de SEUIL_BOUGE")
print(f"figure: {sortie}")
plt.show()