"""
Trace les 4 grandeurs d'un essai de chute : hauteur, roll, pitch, yaw.

Usage:
    python plot_drop.py                  -> menu de selection parmi les CSV de logs/
    python plot_drop.py drop_0-0.csv     -> un fichier precis (cherche dans logs/)
    python plot_drop.py drop_0-0         -> le .csv est optionnel
    python plot_drop.py drop_0-0 drop_1-0   -> superpose plusieurs essais
    python plot_drop.py --last           -> le CSV le plus recent, sans menu
"""

import sys
import glob
import os
import csv

import matplotlib.pyplot as plt

LOG_DIR = "logs"
OUT_DIR = "figures"

# (colonne CSV, titre, unite)
CHANNELS = [
    ("height",    "Hauteur COM",  "m"),
    ("roll_deg",  "Roll",         "deg"),
    ("pitch_deg", "Pitch",        "deg"),
    ("yaw_deg",   "Yaw",          "deg"),
]


def read_csv(path):
    """Lit le CSV -> dict {colonne: [valeurs float]}."""
    data = {"t": []}
    for col, _, _ in CHANNELS:
        data[col] = []

    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            data["t"].append(float(row["t"]))
            for col, _, _ in CHANNELS:
                data[col].append(float(row[col]))

    if not data["t"]:
        raise ValueError(f"{path} ne contient aucune donnee.")
    return data


def list_logs():
    """Tous les CSV de logs/, du plus recent au plus ancien."""
    files = glob.glob(os.path.join(LOG_DIR, "*.csv"))
    if not files:
        raise FileNotFoundError(f"Aucun CSV dans {LOG_DIR}/")
    return sorted(files, key=os.path.getmtime, reverse=True)


def resolve(name):
    """Accepte 'drop_0-0', 'drop_0-0.csv' ou un chemin complet."""
    candidates = [
        name,
        name + ".csv",
        os.path.join(LOG_DIR, name),
        os.path.join(LOG_DIR, name + ".csv"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(
        f"'{name}' introuvable. Essais disponibles :\n  "
        + "\n  ".join(os.path.basename(f) for f in list_logs())
    )


def choose():
    """Menu numerote quand aucun argument n'est donne."""
    files = list_logs()
    print("\nEssais disponibles :")
    for i, f in enumerate(files, 1):
        print(f"  {i:2d}. {os.path.basename(f)}")

    rep = input("\nNumero (Enter = le plus recent, "
                "plusieurs numeros separes par un espace) : ").strip()
    if not rep:
        return [files[0]]

    choix = []
    for tok in rep.split():
        idx = int(tok) - 1
        if not 0 <= idx < len(files):
            raise ValueError(f"Numero hors liste : {tok}")
        choix.append(files[idx])
    return choix


def main():
    args = sys.argv[1:]
    if not args:
        paths = choose()
    elif args == ["--last"]:
        paths = [list_logs()[0]]
    else:
        paths = [resolve(a) for a in args]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    axes = axes.ravel()

    for path in paths:
        data = read_csv(path)
        label = os.path.splitext(os.path.basename(path))[0]
        n, duree = len(data["t"]), data["t"][-1]
        print(f"{label}: {n} points sur {duree:.2f} s "
              f"({n / duree:.1f} Hz moyen)")

        for ax, (col, titre, unite) in zip(axes, CHANNELS):
            ax.plot(data["t"], data[col], marker="o", markersize=3,
                    linewidth=1.2, label=label)
            ax.set_title(titre)
            ax.set_xlabel("temps [s]")
            ax.set_ylabel(f"{titre} [{unite}]")
            ax.grid(True, alpha=0.3)

    if len(paths) > 1:
        axes[0].legend(fontsize=8)

    titre_fig = paths[0] if len(paths) == 1 else f"{len(paths)} essais"
    fig.suptitle(f"Drop test - {titre_fig}")
    fig.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(paths[0]))[0]
    out = os.path.join(OUT_DIR, f"{base}.png")
    fig.savefig(out, dpi=150)
    print(f"-> {out}")

    plt.show()


if __name__ == "__main__":
    main()