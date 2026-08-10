"""Vitesse de la charge due au seul ballant, comparee a un seuil.

    python comparaison_ballant.py a.csv b.csv
    python comparaison_ballant.py                  -> demande les noms
    python comparaison_ballant.py *.csv            -> tous

Ce qui est calcule
------------------
La charge pend au bout du cable. Sa vitesse est celle du chariot plus ce que
l'oscillation ajoute:

    v_charge = v_tcp + L * theta_dot

Le premier terme est le transport voulu, le second est le ballant. C'est le
second qu'on veut borner: il subsiste quand le chariot est a l'arret et c'est
lui qui donne la vitesse d'impact residuelle.

    v_ballant = L * theta_dot          [m/s], vecteur a deux composantes

Aucune soustraction n'est necessaire pour l'obtenir: L*theta_dot est deja la
part du ballant seule, theta_dot etant mesure relativement au chariot. La
soustraction v_charge - v_tcp redonnerait exactement la meme chose. Le script
calcule quand meme v_tcp, pour deux raisons: verifier que v_charge reste dans
des valeurs sensees, et voir si le ballant s'ajoute ou se retranche au
mouvement du chariot au moment qui compte.

Sur la vitesse du chariot
-------------------------
8_main.py ne journalise pas getActualTCPSpeed(), seulement les positions
tcp_x et tcp_y. La vitesse encodeur est donc obtenue en derivant ces
positions, lissees sur LISSAGE_TCP echantillons. A 500 Hz la derivee brute
d'une position encodeur est tres bruitee, le lissage n'est pas optionnel.

Pour les essais suivants, il vaut mieux ajouter les colonnes au log. Dans
8_main.py, l'en-tete:

    w.writerow([..., "tcp_x", "tcp_y", "vtcp_x", "vtcp_y", ...])

et dans la boucle, ajouter *v_tcp a cote de *tcp. La variable existe deja.
Le script detecte ces colonnes et les utilise si elles sont presentes.

Sur theta_dot
-------------
theta_dot vient de l'estimateur, pas de la camera: c'est un etat du filtre,
deja lisse. Il herite donc du modele de pendule libre. Apres un choc, le
modele est faux pendant quelques dixiemes de seconde et theta_dot est a
prendre avec precaution sur cette fenetre.
"""

import sys
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

# ---------------- reglages ----------------
L_CABLE     = 1.10      # longueur du cable [m], doit valoir L_VRAI de l'essai
SEUIL       = 0.095     # vitesse de ballant a ne pas depasser [m/s]
LISSAGE_TCP = 25        # demi-fenetre de lissage de la position TCP [echantillons]
T_IGNORE    = 0.3       # debut ignore: transitoire de demarrage [s]


def charger(chemin):
    d = np.genfromtxt(chemin, delimiter=",", names=True)
    if d.size == 0:
        print(f"{chemin}: vide, ignore")
        return None
    for c in ("t", "thd_x", "thd_y", "tcp_x", "tcp_y"):
        if c not in d.dtype.names:
            print(f"{chemin}: colonne {c} absente, ignore")
            return None
    return d


def vitesse_tcp(d):
    """Vitesse du chariot, depuis le log si elle y est, sinon par derivation."""
    if "vtcp_x" in d.dtype.names and "vtcp_y" in d.dtype.names:
        return d["vtcp_x"], d["vtcp_y"], "encodeur"

    t = d["t"]
    x, y = d["tcp_x"].copy(), d["tcp_y"].copy()
    n = 2 * LISSAGE_TCP + 1
    if len(t) > n:
        noyau = np.ones(n) / n
        x = np.convolve(x, noyau, mode="same")
        y = np.convolve(y, noyau, mode="same")
        # bords fausses par le noyau tronque: on les recopie
        x[:LISSAGE_TCP] = x[LISSAGE_TCP]
        x[-LISSAGE_TCP:] = x[-LISSAGE_TCP - 1]
        y[:LISSAGE_TCP] = y[LISSAGE_TCP]
        y[-LISSAGE_TCP:] = y[-LISSAGE_TCP - 1]
    return np.gradient(x, t), np.gradient(y, t), "derivee"


# ---------------- entree ----------------
fichiers = sys.argv[1:]
if not fichiers:
    saisie = input("fichiers CSV a comparer (Entree = tous): ").strip()
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
fig, ax = plt.subplots(4, 1, figsize=(11, 11.5), sharex=True)
couleurs = plt.rcParams["axes.prop_cycle"].by_key()["color"]

entete = (f"\n{'essai':20s} {'vb max':>8s} {'t(vb max)':>10s} {'vb fin':>8s} "
          f"{'% > seuil':>10s} {'1er dep.':>9s} {'vb choc':>8s} {'v_tcp':>8s}")
print(entete)
print("-" * (len(entete) - 1))

for k, (nom, d) in enumerate(donnees):
    c = couleurs[k % len(couleurs)]
    t = d["t"]

    # --- vitesse due au ballant seul ---
    vb_x = L_CABLE * d["thd_x"]
    vb_y = L_CABLE * d["thd_y"]
    vb = np.hypot(vb_x, vb_y)

    # --- vitesse du chariot ---
    vt_x, vt_y, source = vitesse_tcp(d)
    vt = np.hypot(vt_x, vt_y)

    # --- vitesse de la charge ---
    vc = np.hypot(vt_x + vb_x, vt_y + vb_y)

    # le transitoire du debut n'est pas representatif
    utile = t > T_IGNORE

    vb_u, t_u = vb[utile], t[utile]
    i_max = int(np.argmax(vb_u))
    part_au_dessus = 100.0 * np.mean(vb_u > SEUIL)
    depasse = np.flatnonzero(vb_u > SEUIL)
    t_dep = t_u[depasse[0]] if len(depasse) else None

    # valeur au moment du choc, si la colonne existe
    vb_choc = None
    if "hbouge" in d.dtype.names:
        i = np.flatnonzero((d["hbouge"] > 0.5) & utile)
        if len(i):
            vb_choc = float(vb[i[0]])

    # --- trace ---
    ax[0].plot(t, 1000 * vb, color=c, lw=1.5, label=f"{nom}  |v ballant|")
    ax[0].plot(t, 1000 * vb_x, color=c, lw=0.8, ls="--", alpha=.6)
    ax[0].plot(t, 1000 * vb_y, color=c, lw=0.8, ls=":", alpha=.6)

    ax[1].plot(t, 1000 * vt, color=c, lw=1.3, label=f"{nom}  |v chariot| ({source})")
    ax[2].plot(t, 1000 * vc, color=c, lw=1.3, label=f"{nom}  |v charge|")
    ax[3].plot(t, np.degrees(np.hypot(d["th_x"], d["th_y"])), color=c, lw=1.3,
               label=f"{nom}  |theta|")

    if vb_choc is not None:
        for a in ax:
            a.axvline(t[np.flatnonzero(d["hbouge"] > 0.5)[0]],
                      color=c, lw=1.0, ls="-.", alpha=.5)

    print(f"{nom:20s} {1000*vb_u.max():7.1f}m {t_u[i_max]:9.2f}s "
          f"{1000*vb[-1]:7.1f}m {part_au_dessus:9.1f}% "
          f"{(f'{t_dep:8.2f}s' if t_dep is not None else '   jamais'):>9s} "
          f"{(f'{1000*vb_choc:7.1f}m' if vb_choc is not None else '      --'):>8s} "
          f"{1000*vt.max():7.1f}m")

# seuil sur le panneau du ballant
ax[0].axhline(1000 * SEUIL, color="r", lw=1.4, ls="--",
              label=f"seuil {1000*SEUIL:.0f} mm/s")
ax[0].set_ylabel("vitesse due au ballant [mm/s]")
ax[0].legend(fontsize=7, ncol=2)
ax[1].set_ylabel("vitesse chariot [mm/s]")
ax[1].legend(fontsize=7)
ax[2].axhline(1000 * SEUIL, color="r", lw=1.0, ls=":", alpha=.6)
ax[2].set_ylabel("vitesse charge [mm/s]")
ax[2].legend(fontsize=7)
ax[3].set_ylabel("ballant |theta| [deg]")
ax[3].set_xlabel("t [s]")
ax[3].legend(fontsize=7)
for a in ax:
    a.grid(alpha=.3)
    a.axhline(0, color="k", lw=.5)

fig.suptitle(f"Vitesse due au ballant   L = {L_CABLE:.2f} m   "
             f"seuil {1000*SEUIL:.0f} mm/s   "
             f"(premiere {T_IGNORE:.1f} s ignoree)")
plt.tight_layout()

sortie = "ballant_" + "_".join(nom for nom, _ in donnees[:3]) + ".png"
plt.savefig(sortie, dpi=130)
print(f"\nvb = L * theta_dot, la part de la vitesse de la charge qui vient")
print(f"de l'oscillation seule. Trait mixte vertical = detection de contact.")
print(f"figure: {sortie}")
plt.show()