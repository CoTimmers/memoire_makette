"""Compare damping runs produced by essai_amortissement.py.

    python compare_amortissement.py libre_z070_w030.csv fb_z070_w030.csv
    python compare_amortissement.py                     -> asks for the names
    python compare_amortissement.py amortissement/*.csv -> all of them

What is measured
----------------
Every run induces the same sway, then hands over to the controller. Only the
damping phase is analysed, so the runs are aligned on the handover instant
rather than on the start of the file: t = 0 on the plots is the moment the
controller takes over, which is the only instant they have in common.

    induit      sway at the handover, must match across runs        [deg]
    demi-vie    time for the sway to halve after the handover        [s]
    predite     ln(2)/(zeta_cl omega_n), what the model promises     [s]
    t 10 %      time to fall to a tenth of the induced sway          [s]
    residuel    mean sway over the last 2 s                         [deg]
    tremble     std of the command over the last 2 s, the noise the
                controller injects into the tool                   [mm/s]

The induced sway is printed for every run because it is the control of the
experiment: if one run starts at 4 deg and another at 9, their half-lives are
not comparable and the difference says nothing about the gains.

zeta_cl and omega_t are read back from the file name, which
essai_amortissement.py builds from the settings, so the predicted half-life can
be shown next to the measured one. A file named otherwise simply gets no
prediction.
"""

import sys
import glob
import os
import re
import numpy as np
import matplotlib.pyplot as plt

L_CABLE = 1.17          # cable length used in the runs [m]
OMEGA_N = np.sqrt(9.81 / L_CABLE)


def parametres(nom):
    """Recover (zeta_cl, omega_t, feedback) from the file name."""
    m = re.search(r"(fb|libre)_z(\d+)_w(\d+)", nom)
    if not m:
        return None, None, None
    fb = m.group(1) == "fb"
    # z070 -> 0.70, w030 -> 0.30
    zeta = int(m.group(2)) / 100.0
    omega = int(m.group(3)) / 100.0
    return zeta, omega, fb


fichiers = sys.argv[1:]
if not fichiers:
    saisie = input("fichiers CSV (separes par un espace, Entree = tous "
                   "dans amortissement/): ").strip()
    fichiers = saisie.split() if saisie else sorted(
        glob.glob(os.path.join("amortissement", "*.csv")))

fichiers = [f for f in fichiers if os.path.exists(f)]
if not fichiers:
    raise SystemExit("aucun fichier trouve.")

donnees = []
for f in fichiers:
    d = np.genfromtxt(f, delimiter=",", names=True)
    if d.size == 0 or "phase" not in d.dtype.names:
        print(f"{f}: vide ou pas un essai d'amortissement, ignore")
        continue
    donnees.append((os.path.splitext(os.path.basename(f))[0], d))
if not donnees:
    raise SystemExit("aucun fichier exploitable.")

fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
i_bal, i_amp, i_cmd = 0, 1, 2
couleurs = plt.rcParams["axes.prop_cycle"].by_key()["color"]

entete = (f"\n{'essai':22s} {'induit':>8s} {'demi-vie':>9s} {'predite':>8s} "
          f"{'t 10%':>7s} {'residuel':>9s} {'tremble':>9s}")
print(entete)
print("-" * (len(entete) - 1))

for k, (nom, d) in enumerate(donnees):
    c = couleurs[k % len(couleurs)]
    zeta_cl, omega_t, fb = parametres(nom)

    mesure = d["phase"] == 1
    if mesure.sum() < 100:
        print(f"{nom:22s} phase de mesure trop courte, ignore")
        continue

    t = d["t"][mesure]
    t = t - t[0]                                # 0 = handover to the controller
    thx = np.degrees(d["th_x"][mesure])
    thy = np.degrees(d["th_y"][mesure])
    amp = np.hypot(thx, thy)                    # sway magnitude
    vcmd = np.hypot(d["vcmd_x"][mesure], d["vcmd_y"][mesure])

    # induced sway: peak over the first half second after the handover
    n0 = max(int(0.5 / np.median(np.diff(t))), 10)
    induit = float(amp[:n0].max())

    def temps_sous(frac):
        """Last instant the sway is still above frac of the induced value."""
        idx = np.where(amp > frac * induit)[0]
        return float(t[idx[-1]]) if len(idx) else 0.0

    t50, t10 = temps_sous(0.5), temps_sous(0.10)

    n_fin = max(int(2.0 / np.median(np.diff(t))), 10)
    residuel = float(amp[-n_fin:].mean())
    tremble = float(np.std(vcmd[-n_fin:]))

    if fb and zeta_cl:
        predite = np.log(2) / (zeta_cl * OMEGA_N)
        s_pred = f"{predite:7.2f}s"
    else:
        predite = None
        s_pred = f"{'--':>8s}"

    print(f"{nom:22s} {induit:6.2f}d {t50:8.2f}s {s_pred} {t10:6.2f}s "
          f"{residuel:8.3f}d {1000*tremble:7.1f}mm/s")

    ax[i_bal].plot(t, thx, color=c, lw=1.2, label=f"{nom}  x")
    ax[i_bal].plot(t, thy, color=c, lw=1.0, ls="--", label=f"{nom}  y")
    ax[i_amp].plot(t, amp, color=c, lw=1.3, label=nom)
    ax[i_amp].axhline(0.5 * induit, color=c, ls=":", lw=.8, alpha=.5)
    if t50 > 0:
        ax[i_amp].plot([t50], [0.5 * induit], "o", color=c, ms=5)
    ax[i_cmd].plot(t, vcmd, color=c, lw=1.0, label=nom)

print("\nt = 0 : instant ou le controleur prend le relais")
print("points sur le 2e panneau : demi-vie mesuree")
print("'induit' doit etre comparable d'un essai a l'autre, sinon les "
      "demi-vies ne le sont pas.")

ax[i_bal].set_ylabel("ballant [deg]")
ax[i_bal].axhline(0, color="k", lw=.5)
ax[i_bal].legend(fontsize=7, ncol=max(1, len(donnees) // 2))
ax[i_amp].set_ylabel("amplitude du ballant [deg]")
ax[i_amp].set_yscale("log")          # an exponential decay is a straight line
ax[i_cmd].set_ylabel("vitesse commandee [m/s]")
ax[i_cmd].set_xlabel("t depuis la reprise par le controleur [s]")
for a in ax:
    a.grid(alpha=.3)

fig.suptitle("Amortissement du ballant")
plt.tight_layout()

sortie = "amortissement_" + "_".join(nom for nom, _ in donnees[:3]) + ".png"
plt.savefig(sortie, dpi=130)
print(f"\nfigure: {sortie}")
plt.show()