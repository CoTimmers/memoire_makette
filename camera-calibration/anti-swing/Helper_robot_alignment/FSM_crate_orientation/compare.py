"""
Superposition de plusieurs essais sur la phase de pivoting.

Accepte les CSV produits par main.py (colonne "etape") comme
ceux produits par test_mode.py (colonne "phase"), et les
metadonnees l_cable / masse / essai si elles sont presentes.

Figures :

    1. trajectoires XY superposees, phase pivoting
    2. yaw vs temps superpose, recale au debut du pivoting
    3. yaw final par essai

Usage :

    python compare.py fichier1.csv fichier2.csv ...
    python compare.py data/*.csv
"""

import os
import sys
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

SAUVER = True
DPI = 150

# Etapes retenues comme "pivoting"
PIVOTING = ["engage_pivoting", "finish_pivoting"]

# Tolerance sur l'orientation finale visee [deg]
YAW_CIBLE = 0.0
YAW_TOL = 5.0


# ------------------------------------------------------------
# Geometrie   (identique a plot.py)
# ------------------------------------------------------------

ANGLE_BASE_MARQUEUR = 93.5      # deg

T_MARKER_IN_BASE = np.array([0.3220, -0.0695])   # m

_a = np.radians(ANGLE_BASE_MARQUEUR)

R_BASE_TO_MARKER = np.array([
    [np.cos(_a), -np.sin(_a)],
    [np.sin(_a),  np.cos(_a)],
])

E_CAISSE = np.array([0.0158, 0.0062])     # m
C_CONST = np.array([-0.0099, -0.0112])    # m

CORRIGER_E = True
SIGNE_E = +1


# ------------------------------------------------------------
# Lissage d'affichage
# ------------------------------------------------------------
#
# Moyenne glissante centree sur les trajectoires, pour attenuer
# le bruit d'estimation de pose ArUco (quelques mm par image).
#
# A 100 Hz, 15 echantillons = 0.15 s, soit un ordre de grandeur
# de moins que la dynamique du pendule (~2 s) : le mouvement
# reel n'est pas deforme.
#
# Le lissage ne sert QU'A L'AFFICHAGE. Les valeurs chiffrees du
# bilan sont calculees sur les donnees brutes.
#
# Mettre 1 pour desactiver.

LISSAGE = 15


# ------------------------------------------------------------
# Fenetrage temporel
# ------------------------------------------------------------
#
# Pendant que l'operateur ouvre le mur, la caisse reste en place
# plusieurs secondes, et le geste peut la faire re-osciller.
# Cette perturbation est un artefact de la manipulation, pas une
# propriete du mecanisme etudie.
#
# On coupe donc, dans CHAQUE etape et de facon identique pour
# tous les fichiers :
#
#   - on garde les DUREE_MAX_ETAPE premieres secondes
#   - on garde les PRE_TRANSITION dernieres secondes, juste
#     avant l'envoi de la commande suivante
#   - on jette le milieu
#
# Regle fixe et declarative, pas un seuil adaptatif : elle
# s'applique pareil a tous les essais, quels que soient leurs
# aleas.
#
# Mettre FENETRER = False pour tracer tout.

FENETRER = True

DUREE_MAX_ETAPE = 8.0     # s, depuis le debut de l'etape
PRE_TRANSITION = 1.0      # s, avant la fin de l'etape


# ------------------------------------------------------------
# Etiquettes et couleurs par fichier
# ------------------------------------------------------------
#
# Cle = nom du fichier sans extension.
# Valeur = (libelle affiche, couleur).
#
# Les fichiers absents de cette table prennent une couleur
# automatique et leur nom de fichier comme libelle.

ETIQUETTES = {
    "nominal_fsm":     ("Nominal",             "#1f77b4"),
    "mass_fsm":        ("Mass +1 kg",          "#2ca02c"),
    "length_fsm":      ("Cable +5 %",          "#9467bd"),
    "mass_length_fsm": ("Mass +1 kg, cable +5 %", "#d62728"),
}

# Ordre d'affichage souhaite, par cle. Les autres suivent.
ORDRE = [
    "nominal_fsm",
    "mass_fsm",
    "length_fsm",
    "mass_length_fsm",
]

# Le TCP est identique pour tous les essais : une seule courbe.
TCP_UNIQUE = True
C_TCP_TRACE = "#ff7f0e" 


# ------------------------------------------------------------
# Convention d'affichage   (identique a plot.py)
# ------------------------------------------------------------

REPERE = "base"

ORIGINE_AFFICHAGE = np.array([0.0, 0.0])

SWAP_XY = False

MIROIR_X = -1
MIROIR_Y = -1


# ------------------------------------------------------------
# Consignes commandees [mm], repere base robot
# ------------------------------------------------------------

CONSIGNES = {
    "engage_pivoting":  (530.0, -141.0),
    "finish_pivoting":  (530.0, -293.0),
}


M = np.array([MIROIR_X, MIROIR_Y])


def vue(p_mm):
    """Origine d'affichage, echange eventuel des axes, miroirs."""

    p = np.asarray(p_mm, dtype=float) - ORIGINE_AFFICHAGE

    if SWAP_XY:
        p = p[..., ::-1]

    return p * M


CONSIGNES_VUE = {
    nom: tuple(vue(np.array(xy)))
    for nom, xy in CONSIGNES.items()
}

MARQUEUR_MM = vue(
    1000.0 * T_MARKER_IN_BASE
    if REPERE == "base"
    else np.zeros(2)
)


# ============================================================
# CHARGEMENT
# ============================================================

fichiers = []

for motif in sys.argv[1:]:
    fichiers += sorted(glob.glob(motif))

if not fichiers:
    print("Aucun fichier. Usage : python compare.py *.csv")
    sys.exit(1)


def fenetrer(df):
    """
    Ne garde, dans chaque etape, que le debut et la seconde qui
    precede la transition. Le milieu est remplace par des NaN,
    ce qui interrompt le trace au lieu d'inventer un segment.
    """

    if not FENETRER:
        df["garde"] = True
        return df

    garde = np.zeros(len(df), bool)

    for nom, s in df.groupby("etape", sort=False):

        t = s["t"].to_numpy()

        debut = t - t[0] <= DUREE_MAX_ETAPE

        fin = t[-1] - t <= PRE_TRANSITION

        garde[s.index] = debut | fin

    df["garde"] = garde

    return df


def charger(chemin):
    """Lit un CSV, transforme dans le repere d'affichage."""

    df = pd.read_csv(chemin)

    # colonne d'etape, selon le script d'origine
    col = "etape" if "etape" in df.columns else "phase"
    df["etape"] = df[col]

    # --- accroche ---
    A = df[["attach_x", "attach_y"]].to_numpy(float)

    CORR = np.tile(C_CONST, (len(df), 1))

    if CORRIGER_E:

        yb = df["yaw_ref"].to_numpy(float) + _a
        e = SIGNE_E * E_CAISSE

        CORR = CORR + np.stack([
            np.cos(yb) * e[0] - np.sin(yb) * e[1],
            np.sin(yb) * e[0] + np.cos(yb) * e[1],
        ], axis=1)

    P = df[["tcp_base_x", "tcp_base_y"]].to_numpy(float)

    if REPERE == "base":
        A = A @ R_BASE_TO_MARKER + T_MARKER_IN_BASE - CORR
    else:
        A = A - CORR @ R_BASE_TO_MARKER.T
        P = (P - T_MARKER_IN_BASE) @ R_BASE_TO_MARKER.T

    A = vue(A * 1000.0)
    P = vue(P * 1000.0)

    df["attach_x"], df["attach_y"] = A[:, 0], A[:, 1]
    df["tcp_x"], df["tcp_y"] = P[:, 0], P[:, 1]

    df["yaw_deg"] = np.degrees(df["yaw_ref"])

    # --- versions lissees, pour l'affichage seulement ---

    for col in ["attach_x", "attach_y", "tcp_x", "tcp_y", "yaw_deg"]:

        if LISSAGE > 1:
            df[col + "_l"] = (
                df[col]
                .rolling(LISSAGE, center=True, min_periods=1)
                .mean()
            )
        else:
            df[col + "_l"] = df[col]

    # --- etiquette et couleur ---
    cle = os.path.splitext(os.path.basename(chemin))[0]

    if cle in ETIQUETTES:
        label, couleur = ETIQUETTES[cle]

    elif {"l_cable", "masse"}.issubset(df.columns):

        label = (
            f"L={1000*df['l_cable'].iloc[0]:.0f} mm, "
            f"m={df['masse'].iloc[0]:.1f} kg"
        )

        if "essai" in df.columns:
            label += f" #{df['essai'].iloc[0]}"

        couleur = None

    else:
        label = cle
        couleur = None

    df.attrs["cle"] = cle

    df = fenetrer(df)

    return df, label, couleur


essais = [charger(f) for f in fichiers]

# ordre d'affichage
def rang(e):
    cle = e[0].attrs.get("cle", "")
    return ORDRE.index(cle) if cle in ORDRE else len(ORDRE)

essais.sort(key=rang)

print(f"{len(essais)} fichiers charges\n")


# ------------------------------------------------------------
# Couleurs : celles de la table, sinon palette automatique
# ------------------------------------------------------------

repli = plt.cm.viridis(np.linspace(0.05, 0.85, len(essais)))

couleurs = [
    c if c is not None else repli[i]
    for i, (_, _, c) in enumerate(essais)
]

essais = [(df, lab) for df, lab, _ in essais]


# ============================================================
# 1. TRAJECTOIRES SUPERPOSEES
# ============================================================

fig, ax = plt.subplots(figsize=(8, 7.5))

i_tcp = 0

for (df, label), c in zip(essais, couleurs):

    sel = df[df["etape"].isin(PIVOTING)]

    if sel.empty:
        print(f"  {label} : pas de phase pivoting, ignore")
        continue

    # TCP : commande identique pour tous les essais,
    # on ne trace donc qu'une seule courbe
    if (not TCP_UNIQUE) or i_tcp == 0:
        ax.plot(
            sel["tcp_x_l"], sel["tcp_y_l"],
            color=C_TCP_TRACE if TCP_UNIQUE else c,
            lw=1.0, ls="--", alpha=0.9,
            label="TCP" if TCP_UNIQUE else None,
            zorder=2,
        )
        i_tcp += 1

    # trace fenetre : NaN sur les portions jetees, ce qui
    # interrompt la ligne au lieu d'inventer un raccourci
    xs = sel["attach_x_l"].where(sel["garde"])
    ys = sel["attach_y_l"].where(sel["garde"])

    ax.plot(xs, ys, color=c, lw=1.6, label=label)

    # rond    : dernier point de la portion coupee (DUREE_MAX_ETAPE)
    # losange : etat juste avant la transition
    for nom, g in sel[sel["garde"]].groupby("etape", sort=False):

        coupe = g[g["t"] - g["t"].min() <= DUREE_MAX_ETAPE]

        if len(coupe):
            ax.plot(
                coupe["attach_x_l"].iloc[-1],
                coupe["attach_y_l"].iloc[-1],
                "o", color=c, ms=9,
                mfc="white", mew=2.0, zorder=6,
            )

        fin = g[g["t"] >= g["t"].max() - PRE_TRANSITION]

        ax.plot(
            fin["attach_x_l"].mean(),
            fin["attach_y_l"].mean(),
            "D", color=c, ms=9,
            mec="white", mew=1.2, zorder=6,
        )

    xv = xs.dropna()
    yv = ys.dropna()

    ax.plot(xv.iloc[0], yv.iloc[0], "o", color=c,
            ms=8, mfc="white", mew=1.8)

    ax.plot(xv.iloc[-1], yv.iloc[-1], "s", color=c, ms=7)

    print(
        f"  {label:32s} {len(sel):5d} pts -> "
        f"{int(sel['garde'].sum()):5d} conserves"
    )


# consignes commandees : croix noire, sans libelle
for nom, (cx, cy) in CONSIGNES_VUE.items():
    ax.plot(cx, cy, "kx", ms=11, mew=2.0, zorder=5)

# marqueur helper : petit point noir
ax.plot(
    MARQUEUR_MM[0], MARQUEUR_MM[1],
    "k.", ms=6, zorder=5,
)


# legende des styles
if not TCP_UNIQUE:
    ax.plot([], [], color="0.4", lw=1.6, label="— accroche")
    ax.plot([], [], color="0.4", lw=1.6, ls="--", label="-- TCP")

ax.set_xlabel("X [mm]")
ax.set_ylabel("Y [mm]")
ax.set_title(
    "Crate and TCP XY Trajectories"
)

ax.set_xlim(-900, -350)
ax.set_ylim(-250, 500)

ax.set_aspect("equal", adjustable="box")
ax.grid(alpha=0.3)
ax.legend(fontsize=8, loc="lower left")

fig.tight_layout()

if SAUVER:
    fig.savefig("cmp_traj_pivoting.png", dpi=DPI)


# ============================================================
# 2. YAW SUPERPOSE, RECALE AU DEBUT DU PIVOTING
# ============================================================

fig, ax = plt.subplots(figsize=(10, 4.5))

for (df, label), c in zip(essais, couleurs):

    sel = df[df["etape"].isin(PIVOTING)]

    if sel.empty:
        continue

    t = sel["t"].to_numpy() - sel["t"].iloc[0]

    ax.plot(t, sel["yaw_deg_l"].where(sel["garde"]), color=c, lw=1.4, label=label)


ax.axhspan(
    YAW_CIBLE - YAW_TOL,
    YAW_CIBLE + YAW_TOL,
    color="#c8e6c9",
    alpha=0.6,
    zorder=0,
    label=f"tolerance ±{YAW_TOL:.0f}°",
)

ax.set_xlabel("t depuis le debut du pivoting [s]")
ax.set_ylabel("yaw [deg]")
ax.set_title("Orientation de la caisse pendant le pivoting")
ax.grid(alpha=0.3)
ax.legend(fontsize=8, loc="best")

fig.tight_layout()

if SAUVER:
    fig.savefig("cmp_yaw_pivoting.png", dpi=DPI)


# ============================================================
# 3. BILAN PAR ESSAI
# ============================================================

print(
    f"{'essai':32s} {'yaw debut':>10s} "
    f"{'yaw fin':>9s} {'rotation':>10s} {'ok':>4s}"
)

finaux = []

for df, label in essais:

    sel = df[df["etape"].isin(PIVOTING)]

    if sel.empty:
        continue

    # moyennes sur les premiers et derniers 10 %
    n = len(sel)
    y0 = sel["yaw_deg"].iloc[:max(1, n // 10)].mean()
    y1 = sel["yaw_deg"].iloc[-max(1, n // 10):].mean()

    ok = abs(y1 - YAW_CIBLE) <= YAW_TOL

    finaux.append((label, y1, ok))

    print(
        f"{label:32s} {y0:+10.1f} {y1:+9.1f} "
        f"{y1-y0:+10.1f} {'oui' if ok else 'NON':>4s}"
    )


if len(finaux) >= 2:

    vals = np.array([f[1] for f in finaux])

    print(
        f"\nyaw final : moyenne {vals.mean():+.2f}°, "
        f"ecart-type {vals.std(ddof=1):.2f}°, "
        f"etendue {vals.max()-vals.min():.2f}°"
    )


# ------------------------------------------------------------
# Figure de synthese
# ------------------------------------------------------------

if finaux:

    fig, ax = plt.subplots(figsize=(8, 4.5))

    idx = np.arange(len(finaux))

    ax.axhspan(
        YAW_CIBLE - YAW_TOL,
        YAW_CIBLE + YAW_TOL,
        color="#c8e6c9",
        alpha=0.6,
        zorder=0,
        label=f"tolerance ±{YAW_TOL:.0f}°",
    )

    ax.axhline(YAW_CIBLE, color="0.4", lw=0.9, ls="--")

    for i, (label, y1, ok) in enumerate(finaux):

        ax.plot(
            i, y1,
            "o" if ok else "X",
            ms=11,
            color="#00798c" if ok else "#d1495b",
        )

    ax.set_xticks(idx)
    ax.set_xticklabels(
        [f[0] for f in finaux],
        rotation=30,
        ha="right",
        fontsize=8,
    )

    ax.set_ylabel("yaw final [deg]")
    ax.set_title("Orientation finale par essai")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)

    fig.tight_layout()

    if SAUVER:
        fig.savefig("cmp_yaw_final.png", dpi=DPI)


if SAUVER:
    print("\nfigures enregistrees.")

plt.show()