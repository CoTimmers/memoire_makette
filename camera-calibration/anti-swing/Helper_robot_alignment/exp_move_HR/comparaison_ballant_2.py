"""Compare des essais de 7_align_swing.py: phi, sa vitesse, et la vitesse de
la charge a l'impact.

    python comparaison_align.py test_1.csv test_2.csv
    python comparaison_align.py                 -> demande les noms
    python comparaison_align.py *.csv           -> tous

Trois panneaux
--------------
    1  phi, l'angle entre le helper et le bac. C'est la grandeur que l'essai
       cherche a annuler: son ecart entre le debut et la fin est la correction
       obtenue.

    2  phi_dot, sa vitesse. Le pic date le contact bien plus nettement que phi
       lui-meme, et sa valeur dit combien de rotation a ete communiquee.

    3  la vitesse de la charge, somme vectorielle de deux termes:

           v_charge = v_tcp + L * theta_dot

       le transport voulu plus ce que l'oscillation ajoute. C'est cette vitesse
       la, a l'instant du contact, qui produit la rotation. Les deux termes
       sont traces en transparence a cote de la somme, parce qu'ils peuvent
       s'ajouter ou se retrancher selon la phase de l'oscillation: un ballant
       important peut etre inoffensif s'il tombe a contre-phase.

Deux precautions de mesure
--------------------------
phi vient de la camera, rafraichi a la cadence video, alors que le journal
ecrit a 500 Hz: les colonnes camera sont des escaliers. La derivee ne porte
donc que sur les instants ou la valeur change vraiment, puis elle est lissee.

v_tcp n'est pas journalise, seules les positions tcp_x et tcp_y le sont. La
vitesse du chariot est donc obtenue en derivant ces positions, lissees sur
LISSAGE_TCP echantillons; a 500 Hz la derivee brute d'une position encodeur
est trop bruitee pour etre utilisable telle quelle.

Pendant la phase libre le chariot est arrete, donc v_charge s'y reduit a
L * theta_dot. C'est normal et c'est la valeur qui compte pour l'impact.
"""

import sys
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

# ---------------- reglages ----------------
L_CABLE     = 1.00      # longueur du cable [m], doit valoir L_VRAI de l'essai
LISSAGE     = 2         # demi-fenetre de lissage de phi_dot [mesures video]
LISSAGE_TCP = 25        # demi-fenetre de lissage de la position TCP [echant.]
T_IGNORE    = 0.3       # debut ignore pour les statistiques [s]
LIBRE       = 2         # numero de la phase libre dans 7_align_swing.py


def charger(chemin):
    d = np.genfromtxt(chemin, delimiter=",", names=True)
    if d.size == 0:
        print(f"{chemin}: vide, ignore")
        return None
    if "phi" not in d.dtype.names and "yaw" not in d.dtype.names:
        print(f"{chemin}: ni colonne phi ni colonne yaw, ignore")
        return None
    for c in ("t", "thd_x", "thd_y", "tcp_x", "tcp_y"):
        if c not in d.dtype.names:
            print(f"{chemin}: colonne {c} absente, ignore")
            return None
    return d


def angle(d):
    """phi en radians. Les anciens journaux n'ont que yaw, qui vaut -phi."""
    return d["phi"] if "phi" in d.dtype.names else -d["yaw"]


def derivee_video(t, x, demi=LISSAGE):
    """Derive une grandeur camera journalisee a 500 Hz.

    Les colonnes camera repetent la meme valeur une quinzaine de fois entre
    deux images. Deriver ligne a ligne donnerait un pic par marche, on ne garde
    donc que les instants ou la valeur change.
    """
    y = np.unwrap(x)
    neuf = np.ones(len(t), dtype=bool)
    neuf[1:] = np.diff(y) != 0
    idx = np.flatnonzero(neuf)
    if len(idx) < 5:
        return np.array([]), np.array([])
    tm, ym = t[idx], y[idx]
    if demi > 0 and len(tm) > 2 * demi + 1:
        noyau = np.ones(2 * demi + 1) / (2 * demi + 1)
        ym = np.convolve(ym, noyau, mode="same")
        tm, ym = tm[demi:-demi], ym[demi:-demi]
    return tm, np.gradient(ym, tm)


def vitesse_tcp(d):
    """Vitesse du chariot, derivee des positions encodeur lissees."""
    t = d["t"]
    x, y = d["tcp_x"].copy(), d["tcp_y"].copy()
    n = 2 * LISSAGE_TCP + 1
    if len(t) > n:
        noyau = np.ones(n) / n
        x = np.convolve(x, noyau, mode="same")
        y = np.convolve(y, noyau, mode="same")
        x[:LISSAGE_TCP] = x[LISSAGE_TCP]
        x[-LISSAGE_TCP:] = x[-LISSAGE_TCP - 1]
        y[:LISSAGE_TCP] = y[LISSAGE_TCP]
        y[-LISSAGE_TCP:] = y[-LISSAGE_TCP - 1]
    return np.gradient(x, t), np.gradient(y, t)


def instant_contact(d, tw, wphi):
    """Date le contact: minimum de dist_origine en phase libre, sinon pic de
    phi_dot. Le premier ne depend d'aucun filtre, il est preferable."""
    if "dist_origine" in d.dtype.names:
        m = np.ones(len(d["t"]), dtype=bool)
        if "phase" in d.dtype.names:
            m = d["phase"] == LIBRE
        if m.any():
            i = np.flatnonzero(m)[np.argmin(d["dist_origine"][m])]
            return float(d["t"][i])
    if len(tw):
        return float(tw[int(np.argmax(np.abs(wphi)))])
    return None


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
fig, ax = plt.subplots(3, 1, figsize=(11, 9.5), sharex=True)
couleurs = plt.rcParams["axes.prop_cycle"].by_key()["color"]

entete = (f"\n{'essai':22s} {'phi deb':>8s} {'phi fin':>8s} {'correction':>11s} "
          f"{'wphi pic':>9s} {'t contact':>10s} {'v charge':>9s} {'v ballant':>10s}")
print(entete)
print("-" * (len(entete) - 1))

for k, (nom, d) in enumerate(donnees):
    c = couleurs[k % len(couleurs)]
    t = d["t"]
    tw, wphi = derivee_video(t, angle(d))          # derive l'angle BRUT
    wphi = np.degrees(wphi) if len(tw) else np.array([])
    phi = np.degrees(np.unwrap(angle(d))) % 360.0  # replie APRES
    # --- vitesse de la charge ---
    vb_x = L_CABLE * d["thd_x"]
    vb_y = L_CABLE * d["thd_y"]
    vt_x, vt_y = vitesse_tcp(d)
    vb = np.hypot(vb_x, vb_y)               # ce que l'oscillation ajoute
    vt = np.hypot(vt_x, vt_y)               # ce que le chariot transporte
    vc = np.hypot(vt_x + vb_x, vt_y + vb_y)  # somme vectorielle

    t_c = instant_contact(d, tw, wphi)

    # fond: la phase libre, une seule fois
    if k == 0 and "phase" in d.dtype.names:
        lib = d["phase"] == LIBRE
        if lib.any():
            i = np.flatnonzero(lib)
            for a in ax:
                a.axvspan(t[i[0]], t[i[-1]], color="0.85", alpha=.5, zorder=0)

    ax[0].plot(t, phi, color=c, lw=1.5, label=nom)
    if len(tw):
        ax[1].plot(tw, wphi, color=c, lw=1.4, label=nom)
    ax[2].plot(t, 1000 * vc, color=c, lw=1.5, label=f"{nom}  charge")
    ax[2].plot(t, 1000 * vt, color=c, lw=0.8, ls="--", alpha=.5)
    ax[2].plot(t, 1000 * vb, color=c, lw=0.8, ls=":", alpha=.5)

    if t_c is not None:
        for a in ax:
            a.axvline(t_c, color=c, lw=1.0, ls="-.", alpha=.7)
        i_c = int(np.argmin(np.abs(t - t_c)))
        vc_c, vb_c = 1000 * vc[i_c], 1000 * vb[i_c]
    else:
        vc_c = vb_c = np.nan

    utile = t > T_IGNORE
    phi_deb = float(phi[utile][0])
    phi_fin = float(phi[-1])
    w_pic = float(np.max(np.abs(wphi))) if len(wphi) else np.nan

    def f(v, u=""):
        return "      --" if not np.isfinite(v) else f"{v:7.1f}{u}"

    print(f"{nom:22s} {phi_deb:+7.1f}d {phi_fin:+7.1f}d "
          f"{abs(phi_deb)-abs(phi_fin):+10.1f}d "
          f"{f(w_pic):>9s} "
          f"{(f'{t_c:9.2f}s' if t_c is not None else '       --'):>10s} "
          f"{f(vc_c):>9s} {f(vb_c):>10s}")

ax[0].set_ylabel("phi [deg]")
ax[0].axhline(0, color="k", lw=.8)
ax[0].legend(fontsize=8)
ax[1].set_ylabel("phi_dot [deg/s]")
ax[1].axhline(0, color="k", lw=.5)
ax[1].legend(fontsize=8)
ax[2].set_ylabel("vitesse [mm/s]\n(plein: charge, tirets: chariot,\n"
                 "pointilles: ballant)")
ax[2].set_xlabel("t [s]")
ax[2].axhline(0, color="k", lw=.5)
ax[2].legend(fontsize=8)
for a in ax:
    a.grid(alpha=.3)

fig.suptitle(f"Alignement du helper   L = {L_CABLE:.2f} m   "
             f"bande grise = phase libre, trait mixte = contact")
plt.tight_layout()

sortie = "align_" + "_".join(nom for nom, _ in donnees[:3]) + ".png"
plt.savefig(sortie, dpi=130)
print(f"\nv charge et v ballant sont pris a l'instant du contact.")
print(f"Le contact est date par le minimum de dist_origine en phase libre;")
print(f"a defaut, par le pic de phi_dot.")
print(f"figure: {sortie}")
plt.show()