"""Rotation du bac: yaw, vitesse de yaw, et amortissement identifie.

    python comparaison_rotation.py essai7_v250_143022.csv autre.csv
    python comparaison_rotation.py               -> demande les noms
    python comparaison_rotation.py *.csv         -> tous

A quoi ca sert
--------------
Le critere de validation est un angle total parcouru: le bac doit tourner assez
pour que le helper puisse ensuite s'inserer contre lui. La question est donc
quelle vitesse de rotation il faut lui donner au choc pour qu'il parcoure
ROTATION_CIBLE degres avant de s'arreter.

La reponse depend entierement de la nature de l'amortissement, et les deux cas
plausibles donnent des reponses tres differentes:

  VISQUEUX      le couple resistant est proportionnel a la vitesse.
                omega(t) = omega0 exp(-t/tau), et l'angle total parcouru vaut
                    theta_tot = omega0 * tau
                Doubler l'angle demande de doubler la vitesse initiale.

  COULOMB       le couple resistant est constant (frottement sec dans la
                torsion du cable, contact au point d'accroche).
                omega(t) = omega0 - k t, et l'angle total vaut
                    theta_tot = omega0^2 / (2k)
                Doubler l'angle ne demande que sqrt(2) fois la vitesse.

Le script ajuste les deux modeles sur la decroissance mesuree, affiche le R^2
de chacun, et calcule la vitesse initiale requise selon celui qui colle le
mieux. Si les deux R^2 sont proches, la plage des deux predictions est donnee:
c'est l'incertitude honnete a ce stade, et elle se leve en faisant deux essais
a des vitesses differentes et en verifiant lequel des deux modeles predit le
second a partir du premier.

Un troisieme cas est possible et se voit tout de suite sur le panneau du yaw:
si le cable se tord, il exerce un couple de rappel et le yaw ne decroit pas
mais OSCILLE autour d'une valeur. Aucun des deux modeles ne s'applique alors,
et l'angle atteint est borne par la raideur du cable, pas par l'amortissement.
Le script le signale s'il detecte un changement de signe de omega apres le choc.

Note sur la mesure. omega est derive du yaw camera, qui n'est rafraichi qu'a la
cadence video alors que le journal ecrit a 500 Hz. Seuls les instants ou la
valeur change sont derives, puis lisses sur LISSAGE mesures.
"""

import sys
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

# ---------------- reglages ----------------
ROTATION_CIBLE = 540.0   # angle total a parcourir, 3 demi-tours [deg]
LISSAGE   = 2            # demi-fenetre de lissage de omega [mesures video]
SEUIL_OMEGA = 3.0        # en dessous, on considere la rotation finie [deg/s]
LIBRE     = 2            # numero de la phase libre


def charger(chemin):
    d = np.genfromtxt(chemin, delimiter=",", names=True)
    if d.size == 0 or "yaw" not in d.dtype.names:
        print(f"{chemin}: vide ou sans colonne yaw, ignore")
        return None
    return d


def omega_yaw(t, yaw_rad, demi=LISSAGE):
    """Vitesse de yaw en deg/s, derivee des seules mesures camera reelles."""
    y = np.unwrap(yaw_rad)
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
    return tm, np.degrees(np.gradient(ym, tm))


def r2(y, y_fit):
    ss_res = float(np.sum((y - y_fit) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0


def identifie(tm, om):
    """Ajuste les deux modeles sur la decroissance de |omega| apres le pic.

    Retourne un dict avec tau (visqueux), k (Coulomb), les deux R^2, et la
    vitesse initiale requise pour ROTATION_CIBLE selon chaque modele.
    """
    if len(tm) < 8:
        return None

    a = np.abs(om)
    i0 = int(np.argmax(a))                    # le pic: debut de la decroissance
    m = (np.arange(len(a)) >= i0) & (a > SEUIL_OMEGA)
    if m.sum() < 6:
        return None
    tt, aa = tm[m] - tm[m][0], a[m]

    # --- visqueux: ln|omega| lineaire en t ---
    p_v = np.polyfit(tt, np.log(aa), 1)
    tau = -1.0 / p_v[0] if p_v[0] < 0 else np.inf
    r2_v = r2(np.log(aa), np.polyval(p_v, tt))

    # --- Coulomb: |omega| lineaire en t ---
    p_c = np.polyfit(tt, aa, 1)
    k = -p_c[0] if p_c[0] < 0 else np.inf
    r2_c = r2(aa, np.polyval(p_c, tt))

    om0 = float(aa[0])
    # angle reellement parcouru pendant la decroissance
    angle_mesure = float(np.trapezoid(aa, tt)) if hasattr(np, "trapezoid") \
        else float(np.trapz(aa, tt))

    # vitesse initiale requise pour ROTATION_CIBLE
    om_v = ROTATION_CIBLE / tau if np.isfinite(tau) and tau > 0 else np.nan
    om_c = np.sqrt(2 * k * ROTATION_CIBLE) if np.isfinite(k) and k > 0 else np.nan

    # rappel elastique? omega change de signe apres le pic
    oscille = bool(np.any(np.sign(om[i0:]) != np.sign(om[i0])))

    return {"om0": om0, "tau": tau, "k": k, "r2_v": r2_v, "r2_c": r2_c,
            "angle": angle_mesure, "om_v": om_v, "om_c": om_c,
            "oscille": oscille, "t0": tm[m][0], "tt": tt, "aa": aa,
            "fit_v": np.exp(np.polyval(p_v, tt)), "fit_c": np.polyval(p_c, tt)}


# ---------------- entree ----------------
fichiers = sys.argv[1:]
if not fichiers:
    saisie = input("fichiers CSV a analyser (Entree = tous): ").strip()
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
fig, ax = plt.subplots(3, 1, figsize=(11, 10), sharex=False)
couleurs = plt.rcParams["axes.prop_cycle"].by_key()["color"]

print(f"\ncible: {ROTATION_CIBLE:.0f} deg parcourus au total\n")
entete = (f"{'essai':22s} {'om pic':>8s} {'angle':>8s} {'tau':>7s} {'R2 visq':>8s} "
          f"{'k':>8s} {'R2 coul':>8s} {'om0 req':>18s}")
print(entete)
print("-" * len(entete))

for k_, (nom, d) in enumerate(donnees):
    c = couleurs[k_ % len(couleurs)]
    t = d["t"]
    yaw = np.degrees(np.unwrap(d["yaw"]))
    yaw = yaw - yaw[0]
    tm, om = omega_yaw(t, d["yaw"])

    # fond: phase libre
    if k_ == 0 and "phase" in d.dtype.names:
        lib = d["phase"] == LIBRE
        if lib.any():
            i = np.flatnonzero(lib)
            for a in ax[:2]:
                a.axvspan(t[i[0]], t[i[-1]], color="0.85", alpha=.5, zorder=0)

    ax[0].plot(t, yaw, color=c, lw=1.4, label=nom)
    if len(tm):
        ax[1].plot(tm, om, color=c, lw=1.3, label=nom)

    res = identifie(tm, om)
    if res is None:
        print(f"{nom:22s}   pas assez de rotation pour identifier")
        continue

    # panneau 3: la decroissance et les deux ajustements
    ax[2].plot(res["tt"], res["aa"], color=c, lw=1.6, label=f"{nom} mesure")
    ax[2].plot(res["tt"], res["fit_v"], color=c, lw=1.0, ls="--",
               label=f"{nom} visqueux R2={res['r2_v']:.3f}")
    ax[2].plot(res["tt"], res["fit_c"], color=c, lw=1.0, ls=":",
               label=f"{nom} Coulomb R2={res['r2_c']:.3f}")

    meilleur = "visqueux" if res["r2_v"] > res["r2_c"] else "Coulomb"
    om_req = res["om_v"] if meilleur == "visqueux" else res["om_c"]
    print(f"{nom:22s} {res['om0']:7.1f}d/s {res['angle']:7.1f}d "
          f"{res['tau']:6.2f}s {res['r2_v']:8.3f} {res['k']:7.1f} "
          f"{res['r2_c']:8.3f} "
          f"{om_req:8.0f} d/s ({meilleur})")
    if res["oscille"]:
        print(f"{'':22s}   ATTENTION: omega change de signe, le cable exerce "
              f"un couple de rappel.")
        print(f"{'':22s}   Les deux modeles sont inapplicables tels quels.")
    if abs(res["r2_v"] - res["r2_c"]) < 0.05:
        print(f"{'':22s}   les deux modeles collent aussi bien: la vitesse "
              f"requise est entre {min(res['om_v'], res['om_c']):.0f} et "
              f"{max(res['om_v'], res['om_c']):.0f} deg/s.")
        print(f"{'':22s}   Deux essais a des vitesses differentes trancheront.")

ax[0].set_ylabel("yaw du bac [deg]")
ax[0].axhline(0, color="k", lw=.5)
ax[0].axhline(ROTATION_CIBLE, color="r", lw=1.2, ls="--",
              label=f"cible {ROTATION_CIBLE:.0f} deg")
ax[0].axhline(-ROTATION_CIBLE, color="r", lw=1.2, ls="--")
ax[0].legend(fontsize=7)
ax[0].set_xlabel("t [s]")

ax[1].set_ylabel("vitesse de yaw [deg/s]")
ax[1].axhline(0, color="k", lw=.5)
ax[1].legend(fontsize=7)
ax[1].set_xlabel("t [s]")

ax[2].set_ylabel("|omega| decroissance [deg/s]")
ax[2].set_xlabel("temps depuis le pic [s]")
ax[2].legend(fontsize=7)
for a in ax:
    a.grid(alpha=.3)

fig.suptitle(f"Rotation du bac et amortissement   cible {ROTATION_CIBLE:.0f} deg")
plt.tight_layout()

sortie = "rotation_" + "_".join(nom for nom, _ in donnees[:3]) + ".png"
plt.savefig(sortie, dpi=130)
print(f"\nangle = angle reellement parcouru depuis le pic, par integration de "
      f"|omega|.")
print(f"om0 req = vitesse de rotation a atteindre au choc pour parcourir "
      f"{ROTATION_CIBLE:.0f} deg.")
print(f"figure: {sortie}")
plt.show()