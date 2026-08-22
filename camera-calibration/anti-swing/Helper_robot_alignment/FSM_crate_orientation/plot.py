"""
Traces des resultats FSM.

Tout est ramene dans le repere BASE ROBOT :

    - le TCP y est deja  (colonnes tcp_base_*)
    - l'accroche vient du repere ArUco helper et y est transformee

Figures produites :

    1. yaw vs temps
    2. diagnostic  residu accroche - TCP
    3. trajectoire  phase "approche home"
    4. trajectoire  phase "pivoting"
    5. trajectoire  phase "shifts"

Usage :

    python plot.py
    python plot.py mon_fichier.csv
"""

import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

CSV = sys.argv[1] if len(sys.argv) > 1 else "fsm_test.csv"

SAUVER = True
DPI = 150

C_ATTACH = "#d1495b"
C_TCP = "#00798c"
C_FOND = "0.85"

MIROIR_X = -1
MIROIR_Y = -1


# ------------------------------------------------------------
# Geometrie
# ------------------------------------------------------------
#
# Angle mesure sur les deux transitions propres du CSV :
#   init -> home        ( 76 mm, yaw constant)  ->  93.2 deg
#   shift_1 -> shift_2  (316 mm, yaw constant)  ->  93.7 deg
# Ratio des amplitudes 0.95 et 1.00 : echelle correcte.
#
# Les ajustements globaux donnaient 100-106 deg, biaises par
# les transitions de pivoting ou le bras de levier tourne.

ANGLE_BASE_MARQUEUR = 93.5      # deg

T_MARKER_IN_BASE = np.array([0.3220, -0.0695])   # m

_a = np.radians(ANGLE_BASE_MARQUEUR)

R_BASE_TO_MARKER = np.array([
    [np.cos(_a), -np.sin(_a)],
    [np.sin(_a),  np.cos(_a)],
])


# ------------------------------------------------------------
# Corrections de calibration
# ------------------------------------------------------------
#
# Ajustees sur les etats SANS contact uniquement
# (init, home, shift_1, shift_2).
#
#   E_CAISSE  erreur sur OFFSET_ACCROCHE, repere de la caisse,
#             tourne donc avec le yaw
#   C_CONST   decalage constant, repere base
#             ~15 mm, soit 0.8 deg d'inclinaison du plan
#             du marqueur helper sur 1.08 m de cable

E_CAISSE = np.array([0.0158, 0.0062])     # m
C_CONST = np.array([-0.0099, -0.0112])    # m

# Mettre False si le CSV a ete enregistre avec
# OFFSET_ACCROCHE deja corrige dans vision.py.
CORRIGER_E = True

# Le signe de E_CAISSE etait indetermine sur les donnees
# d'ajustement (deux orientations seulement). Si le residu
# hors contact augmente au lieu de diminuer, passer a -1.
SIGNE_E = +1


# ------------------------------------------------------------
# Consignes commandees [mm], repere base robot
# ------------------------------------------------------------

CONSIGNES = {
    "home":             (530.0,   39.0),
    "engage_pivoting":  (530.0, -141.0),
    "finish_pivoting":  (530.0, -293.0),
    "shift_1":          (595.0, -293.0),
    "shift_2":          (595.0,   23.0),
    "shift_3":          (565.0,  -33.0),
}


# Etapes ou la caisse est contrainte par les murs du helper
CONTACT = ["engage_pivoting", "finish_pivoting"]


PHASES = [
    (
        "approche_home",
        ["init", "home"],
        "Phase 1   position initiale -> home",
    ),
    (
        "pivoting",
        ["engage_pivoting", "finish_pivoting"],
        "Phase 2   engage et finish pivoting",
    ),
    (
        "shifts",
        ["shift_1", "shift_2", "shift_3"],
        "Phase 3   shifts 1 a 3",
    ),
]


# ============================================================
# LECTURE
# ============================================================

df = pd.read_csv(CSV)


# ------------------------------------------------------------
# Accroche : repere marqueur -> repere base robot
# ------------------------------------------------------------

Q = df[["attach_x", "attach_y"]].to_numpy(float)
Q = Q @ R_BASE_TO_MARKER + T_MARKER_IN_BASE


# erreur tournante sur OFFSET_ACCROCHE
if CORRIGER_E:

    yb = df["yaw_ref"].to_numpy(float) + _a

    e = SIGNE_E * E_CAISSE

    Q = Q - np.stack([
        np.cos(yb) * e[0] - np.sin(yb) * e[1],
        np.sin(yb) * e[0] + np.cos(yb) * e[1],
    ], axis=1)


# decalage constant
Q = Q - C_CONST

df["attach_x"] = Q[:, 0]
df["attach_y"] = Q[:, 1]


# ------------------------------------------------------------
# TCP : deja dans le repere base
# ------------------------------------------------------------

df["tcp_x"] = df["tcp_base_x"]
df["tcp_y"] = df["tcp_base_y"]


# ------------------------------------------------------------
# metres -> millimetres
# ------------------------------------------------------------

for col in ["attach_x", "attach_y", "tcp_x", "tcp_y"]:
    df[col] = 1000.0 * df[col]

# miroirs d'affichage autour de x = 0 et y = 0
df["attach_x"] *= MIROIR_X
df["attach_y"] *= MIROIR_Y
df["tcp_x"] *= MIROIR_X
df["tcp_y"] *= MIROIR_Y

MARQUEUR_MM = 1000.0 * T_MARKER_IN_BASE
MARQUEUR_MM[0] *= MIROIR_X
MARQUEUR_MM[1] *= MIROIR_Y

CONSIGNES = {
    nom: (MIROIR_X * x, MIROIR_Y * y)
    for nom, (x, y) in CONSIGNES.items()
}

df["yaw_deg"] = np.degrees(df["yaw_ref"])


# ------------------------------------------------------------
# Instants de changement d'etape
# ------------------------------------------------------------

changements = df.index[
    df["etape"] != df["etape"].shift()
].tolist()

etapes = [
    (df.loc[i, "t"], df.loc[i, "etape"])
    for i in changements
]

print(f"{len(df)} echantillons, {df['t'].iloc[-1]:.1f} s")
print("etapes :", ", ".join(nom for _, nom in etapes))


def marquer_etapes(ax):

    for t_e, nom in etapes:

        ax.axvline(t_e, color="0.75", lw=0.8, ls="--", zorder=0)

        ax.annotate(
            nom,
            xy=(t_e, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(3, -4),
            textcoords="offset points",
            fontsize=7,
            color="0.4",
            rotation=90,
            va="top",
        )


# ============================================================
# 1. YAW
# ============================================================

fig, ax = plt.subplots(figsize=(10, 3.8))

ax.plot(df["t"], df["yaw_deg"], color="#3d405b", lw=1.3)

marquer_etapes(ax)

ax.set_xlabel("t [s]")
ax.set_ylabel("yaw [deg]")
ax.set_title("Orientation de la caisse dans le repere helper")
ax.grid(alpha=0.3)

fig.tight_layout()

if SAUVER:
    fig.savefig("fig_yaw.png", dpi=DPI)


# ============================================================
# 2. DIAGNOSTIC : residu accroche - TCP
# ============================================================

df["dx"] = df["attach_x"] - df["tcp_x"]
df["dy"] = df["attach_y"] - df["tcp_y"]

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axes[0].plot(df["t"], df["dx"], color=C_TCP, lw=1.2)
axes[0].set_ylabel("dx [mm]")

axes[1].plot(df["t"], df["dy"], color=C_ATTACH, lw=1.2)
axes[1].set_ylabel("dy [mm]")

axes[2].plot(df["t"], df["yaw_deg"], color="#3d405b", lw=1.2)
axes[2].set_ylabel("yaw [deg]")
axes[2].set_xlabel("t [s]")

for ax in axes:
    ax.grid(alpha=0.3)
    ax.axhline(0, color="0.6", lw=0.8)
    marquer_etapes(ax)

axes[0].set_title("Residu accroche - TCP, repere base robot")

fig.tight_layout()

if SAUVER:
    fig.savefig("fig_residu.png", dpi=DPI)


# ------------------------------------------------------------
# Bilan chiffre, sur le dernier quart de chaque etape
# ------------------------------------------------------------

print(
    f"\n{'etape':18s} {'duree':>7s} "
    f"{'dx':>7s} {'dy':>7s} {'|d|':>7s}"
)

for nom in df["etape"].unique():

    s = df[df["etape"] == nom]
    duree = s["t"].max() - s["t"].min()

    s = s.iloc[int(0.75 * len(s)):]

    ddx = s["dx"].mean()
    ddy = s["dy"].mean()

    tag = ""
    if nom in CONTACT:
        tag = "  <- contact avec les murs"
    elif duree < 10:
        tag = "  <- trop court pour stabiliser"

    print(
        f"{nom:18s} {duree:6.1f}s "
        f"{ddx:+7.1f} {ddy:+7.1f} "
        f"{np.hypot(ddx, ddy):7.1f}{tag}"
    )

print(
    "\nAttendu : moins de 5 mm hors contact, "
    "40 a 60 mm pendant le pivoting."
)


# ============================================================
# TRAJECTOIRES PAR PHASE
# ============================================================

def fleches(ax, x, y, couleur, n=4):
    """Quelques fleches indiquant le sens de parcours."""

    x = np.asarray(x)
    y = np.asarray(y)

    if len(x) < 4:
        return

    for i in np.linspace(len(x) // 8, len(x) - 2, n, dtype=int):

        if np.hypot(x[i + 1] - x[i], y[i + 1] - y[i]) < 1e-6:
            continue

        ax.annotate(
            "",
            xy=(x[i + 1], y[i + 1]),
            xytext=(x[i], y[i]),
            arrowprops=dict(
                arrowstyle="-|>",
                color=couleur,
                lw=0,
                mutation_scale=16,
            ),
        )


def trace_phase(cle, noms, titre):

    sel = df[df["etape"].isin(noms)]

    if sel.empty:
        print(f"  phase '{cle}' vide, ignoree")
        return

    fig, ax = plt.subplots(figsize=(7.5, 7))

    # ----- trajectoire complete en fond -----

    ax.plot(df["tcp_x"], df["tcp_y"],
            color=C_FOND, lw=1.0, zorder=1)

    ax.plot(df["attach_x"], df["attach_y"],
            color=C_FOND, lw=1.0, zorder=1)

    # ----- phase courante -----

    ax.plot(
        sel["tcp_x"], sel["tcp_y"],
        color=C_TCP, lw=1.8, label="TCP", zorder=3,
    )

    ax.plot(
        sel["attach_x"], sel["attach_y"],
        color=C_ATTACH, lw=1.5,
        label="point d'accroche", zorder=3,
    )

    fleches(ax, sel["tcp_x"].to_numpy(),
            sel["tcp_y"].to_numpy(), C_TCP)

    fleches(ax, sel["attach_x"].to_numpy(),
            sel["attach_y"].to_numpy(), C_ATTACH)

    # ----- debut et fin -----

    for cx, cy, couleur in [
        ("tcp_x", "tcp_y", C_TCP),
        ("attach_x", "attach_y", C_ATTACH),
    ]:
        ax.plot(
            sel[cx].iloc[0], sel[cy].iloc[0],
            "o", color=couleur, ms=9,
            mfc="white", mew=2.0, zorder=4,
        )

        ax.plot(
            sel[cx].iloc[-1], sel[cy].iloc[-1],
            "s", color=couleur, ms=8, zorder=4,
        )

    # ----- consignes commandees -----

    for nom in noms:

        if nom not in CONSIGNES:
            continue

        cx, cy = CONSIGNES[nom]

        ax.plot(cx, cy, "k+", ms=13, mew=1.6, zorder=5)

        ax.annotate(
            nom,
            xy=(cx, cy),
            xytext=(8, 6),
            textcoords="offset points",
            fontsize=9,
            zorder=5,
        )

    # ----- marqueur helper -----

    ax.plot(
        MARQUEUR_MM[0], MARQUEUR_MM[1],
        "kx", ms=10, mew=1.5, zorder=5,
    )

    ax.annotate(
        "ArUco helper",
        xy=(MARQUEUR_MM[0], MARQUEUR_MM[1]),
        xytext=(8, -14),
        textcoords="offset points",
        fontsize=8,
        color="0.3",
    )

    ax.set_xlabel("x [mm]   repere base robot")
    ax.set_ylabel("y [mm]   repere base robot")
    ax.set_title(titre)

    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()

    if SAUVER:
        fig.savefig(f"fig_traj_{cle}.png", dpi=DPI)


for cle, noms, titre in PHASES:
    trace_phase(cle, noms, titre)


# ============================================================

if SAUVER:
    print("\nfigures enregistrees.")

plt.show()