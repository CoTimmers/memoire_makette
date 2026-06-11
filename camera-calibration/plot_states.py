"""
Plot de la trajectoire du bac colorée par mode.
Usage : python plot_states.py logs/log_YYYYMMDD_HHMMSS.csv
"""

import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection

# Couleurs par mode (RGB normalisé)
MODE_COLORS_PLT = {
    "UNKNOWN":           (0.5,  0.5,  0.5),
    "CORNER_APPROACH":   (1.0,  0.65, 0.0),
    "ENGAGING_PIVOTING": (1.0,  0.0,  1.0),
    "SHIFTING_IN":       (1.0,  0.78, 0.0),
    "DONE":              (0.0,  0.9,  0.0),
}

def load_csv(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "t":              float(row["t"]),
                "x":              float(row["x"]),
                "y":              float(row["y"]),
                "theta_deg":      float(row["theta_deg"]),
                "vx":             float(row["vx"]),
                "vy":             float(row["vy"]),
                "omega":          float(row["omega"]),
                "mode":           row["mode"],
                "can_transition": int(row["can_transition"]),
            })
    return rows


def plot(rows):
    t        = np.array([r["t"]         for r in rows]) - rows[0]["t"]
    x        = np.array([r["x"]         for r in rows])
    y        = np.array([r["y"]         for r in rows])
    theta    = np.array([r["theta_deg"] for r in rows])
    vx       = np.array([r["vx"]        for r in rows])
    vy       = np.array([r["vy"]        for r in rows])
    omega    = np.array([r["omega"]     for r in rows])
    modes    = [r["mode"] for r in rows]
    trans    = np.array([r["can_transition"] for r in rows])
    speed    = np.sqrt(vx**2 + vy**2)

    colors = [MODE_COLORS_PLT.get(m, (0.5, 0.5, 0.5)) for m in modes]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Trajectoire du bac — transitions de mode", fontsize=14)

    # ── 1. Trajectoire XY colorée par mode ──
    ax = axes[0, 0]
    points  = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    seg_colors = colors[:-1]
    lc = LineCollection(segments, colors=seg_colors, linewidth=2)
    ax.add_collection(lc)
    ax.autoscale()
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Trajectoire XY")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    # Marque les transitions
    trans_idx = np.where(np.diff(trans) != 0)[0]
    for idx in trans_idx:
        ax.plot(x[idx], y[idx], "k*", markersize=12)
    ax.plot(x[0], y[0], "go", markersize=8, label="Départ")
    ax.plot(x[-1], y[-1], "rs", markersize=8, label="Fin")
    ax.legend()

    # ── 2. x(t) et y(t) ──
    ax = axes[0, 1]
    for i in range(len(t)-1):
        ax.plot(t[i:i+2], x[i:i+2], color=colors[i], linewidth=1.5)
        ax.plot(t[i:i+2], y[i:i+2], color=colors[i], linewidth=1.5, linestyle="--")
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Position (m)")
    ax.set_title("x(t) — y(t)")
    ax.legend(handles=[
        mpatches.Patch(color="black", label="x (trait plein)"),
        mpatches.Patch(color="gray",  label="y (tirets)"),
    ])
    ax.grid(True, alpha=0.3)

    # ── 3. theta(t) ──
    ax = axes[1, 0]
    for i in range(len(t)-1):
        ax.plot(t[i:i+2], theta[i:i+2], color=colors[i], linewidth=2)
    ax.axhline(45,  color="gray", linestyle=":", label="45°")
    ax.axhline(90,  color="gray", linestyle="--", label="90°")
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Orientation (°)")
    ax.set_title("Orientation θ(t)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── 4. Vitesse + vitesse angulaire ──
    ax = axes[1, 1]
    for i in range(len(t)-1):
        ax.plot(t[i:i+2], speed[i:i+2],  color=colors[i], linewidth=2)
        ax.plot(t[i:i+2], np.abs(omega[i:i+2]), color=colors[i], linewidth=1.5, linestyle="--")
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Vitesse")
    ax.set_title("|v| m/s  —  |ω| rad/s")
    ax.legend(handles=[
        mpatches.Patch(color="black", label="|v| (trait plein)"),
        mpatches.Patch(color="gray",  label="|ω| (tirets)"),
    ])
    ax.grid(True, alpha=0.3)

    # ── Légende des modes ──
    legend_patches = [
        mpatches.Patch(color=c, label=m)
        for m, c in MODE_COLORS_PLT.items()
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=5,
               fontsize=9, title="Modes", bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig("trajectory_plot.png", dpi=150, bbox_inches="tight")
    print("Sauvegardé : trajectory_plot.png")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Prend le dernier log automatiquement
        import glob
        files = sorted(glob.glob("logs/log_*.csv"))
        if not files:
            print("Aucun fichier log trouvé. Usage: python plot_states.py logs/log_xxx.csv")
            sys.exit(1)
        path = files[-1]
        print(f"Chargement du dernier log : {path}")
    else:
        path = sys.argv[1]

    rows = load_csv(path)
    print(f"{len(rows)} frames chargées.")
    plot(rows)
