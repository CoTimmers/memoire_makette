"""
plot_energy_en.py — Potential energy curve U(θ) of the crate suspended by crane cable.

Principle : crane cable attached to the CoM → pendulum.
U(θ) = T · y_CoM(θ), with the lowest corner resting on wall 1 (y=0).
Le minimum de U = le minimum de y_CoM = position la plus stable.

Usage :
    python plot_energy.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

# ── Crate parameters ─────────────────────────────────────────────────────────
a = 0.3   # short side (m)
b = 0.4   # long side  (m)

corners_body = np.array([
    [-a/2, -b/2],
    [ a/2, -b/2],
    [ a/2,  b/2],
    [-a/2,  b/2],
])


def y_com(theta):
    """
    y of CoM when the lowest corner rests on wall 1 (y_coin = 0).

    In physical coordinates (y upward) :
        y_coin_monde = y_com + sin(theta)*bx + cos(theta)*by
    Le coin actif est celui avec le y_coin_monde minimal.
    On pose y_coin = 0  →  y_com = -(sin*bx + cos*by)
    """
    c, s = np.cos(theta), np.sin(theta)
    y_coins = s * corners_body[:, 0] + c * corners_body[:, 1]
    idx = np.argmin(y_coins)
    bx, by = corners_body[idx]
    return -(s * bx + c * by)


# ── Calcul de la courbe ───────────────────────────────────────────────────────
thetas_deg = np.linspace(0, 360, 3601)
thetas_rad = np.radians(thetas_deg)
U = np.array([y_com(t) for t in thetas_rad])

# Valeurs analytiques exactes
U_min_global = a / 2          # short side flat  → 0.150 m
U_min_local  = b / 2          # long side flat   → 0.200 m
U_max        = np.sqrt((a/2)**2 + (b/2)**2)  # coin vers le bas → 0.250 m


# ── Extrema detection ─────────────────────────────────────────────────────
def find_extrema(U, thetas_deg, tol=1e-3):
    minima, maxima = [], []
    n = len(U)
    for i in range(1, n - 1):
        if U[i] < U[i-1] - tol and U[i] < U[i+1] - tol:
            minima.append((thetas_deg[i], U[i]))
        if U[i] > U[i-1] + tol and U[i] > U[i+1] + tol:
            maxima.append((thetas_deg[i], U[i]))
    return minima, maxima

minima, maxima = find_extrema(U, thetas_deg)


# ── Figure ────────────────────────────────────────────────────────────────────
fig, (ax_main, ax_crates) = plt.subplots(
    2, 1, figsize=(11, 8),
    gridspec_kw={'height_ratios': [3, 1]},
)
fig.suptitle(
    "Potential energy U(θ) — crate suspended by crane cable\n"
    "Pivot on wall 1 : minimum y_CoM = stable position",
    fontsize=12, y=0.98,
)

# ── Main plot ──────────────────────────────────────────────────────────
ax = ax_main

# Colour zones for each equilibrium type
zone_alpha = 0.08
ax.axhspan(U_min_global - 0.002, U_min_global + 0.002, color='green',  alpha=0.25, zorder=0)
ax.axhspan(U_min_local  - 0.002, U_min_local  + 0.002, color='royalblue', alpha=0.20, zorder=0)
ax.axhspan(U_max        - 0.002, U_max        + 0.002, color='tomato', alpha=0.20, zorder=0)

# Reference lines
ax.axhline(U_min_global, color='green',     lw=1.2, ls='--', alpha=0.7,
           label=f'Min global  y = a/2 = {U_min_global:.3f} m  (long side flat)')
ax.axhline(U_min_local,  color='royalblue', lw=1.2, ls='--', alpha=0.7,
           label=f'Min local   y = b/2 = {U_min_local:.3f} m  (short side flat)')
ax.axhline(U_max,        color='tomato',    lw=1.2, ls='--', alpha=0.7,
           label=f'Maximum     y = d/2 = {U_max:.3f} m  (corner pointing down)')

# Curve U(θ)
ax.plot(thetas_deg, U, color='#2C2C2A', lw=2.2, label='U(θ) = y_CoM(θ)')

# Stable equilibrium points (minima)
for th, u in minima:
    if abs(u - U_min_global) < 0.002:
        col, mk, ms, lbl = 'green', 'o', 10, 'global min'
    else:
        col, mk, ms, lbl = 'royalblue', 's', 9, 'local min'
    ax.plot(th, u, mk, color=col, ms=ms, zorder=5)
    ax.annotate(
        f'{lbl}\nθ={th:.0f}°\ny={u:.3f}',
        xy=(th, u), xytext=(th + 8, u - 0.012),
        fontsize=8, color=col,
        arrowprops=dict(arrowstyle='->', color=col, lw=0.8),
    )

# Unstable equilibrium points (maxima)
for th, u in maxima:
    ax.plot(th, u, '^', color='tomato', ms=9, zorder=5)
    ax.annotate(
        f'max\nθ={th:.0f}°',
        xy=(th, u), xytext=(th + 6, u + 0.006),
        fontsize=8, color='tomato',
        arrowprops=dict(arrowstyle='->', color='tomato', lw=0.8),
    )

# Double arrow to show energy barrier
barrier_theta = 160
ax.annotate(
    '', xy=(barrier_theta, U_max), xytext=(barrier_theta, U_min_local),
    arrowprops=dict(arrowstyle='<->', color='gray', lw=1.2),
)
ax.text(barrier_theta + 4, (U_max + U_min_local) / 2,
        f'barrier\n{U_max - U_min_local:.3f} m',
        fontsize=8, color='gray', va='center')

ax.set_xlim(0, 360)
ax.set_ylim(0.12, 0.28)
ax.set_xlabel('θ — crate orientation (°)', fontsize=11)
ax.set_ylabel('y_CoM  (m)', fontsize=11)
ax.set_xticks(range(0, 361, 45))
ax.legend(fontsize=9, loc='upper center', ncol=2, framealpha=0.9)
ax.grid(True, alpha=0.25)
ax.set_title(
    f'a = {a} m (short side)   b = {b} m (long side)   '
    f'diagonal/2 = {U_max:.3f} m',
    fontsize=9, color='gray'
)


# ── Sketches of the 3 key positions ──────────────────────────────────────────────
ax2 = ax_crates
ax2.set_aspect('equal')
ax2.axis('off')

def draw_crate_sketch(ax, cx, cy, theta, a, b, color, label, scale=0.8):
    """Draw the crate top view with wall at bottom."""
    c, s = np.cos(theta), np.sin(theta)
    raw = np.array([[-a/2,-b/2],[a/2,-b/2],[a/2,b/2],[-a/2,b/2],[-a/2,-b/2]])
    world = np.array([[cx + (c*x - s*y)*scale, cy + (s*x + c*y)*scale]
                      for x, y in raw])
    ax.fill(world[:-1, 0], world[:-1, 1], alpha=0.25, color=color)
    ax.plot(world[:, 0], world[:, 1], color=color, lw=1.8)

    # Wall below the crate
    min_y = world[:-1, 1].min()
    ax.plot([cx - scale*0.9, cx + scale*0.9], [min_y - 0.03, min_y - 0.03],
            color='black', lw=3)

    # CoM
    ax.plot(cx, cy, 'k.', ms=6)

    # Cable arrow (upward)
    ax.annotate('', xy=(cx, cy + scale*0.5), xytext=(cx, cy),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.2))

    # Label and y_CoM to the left of the crate, not overlapping it
    left_x = world[:-1, 0].min() - 0.10
    mid_y  = (world[:-1, 1].max() + world[:-1, 1].min()) / 2
    # small leader line from crate edge to text
    ax.plot([world[:-1, 0].min(), left_x + 0.03], [mid_y, mid_y],
            color=color, lw=0.7, alpha=0.5)
    ax.text(left_x, mid_y + 0.07, label, ha='right', va='center',
            fontsize=8.5, color=color, fontweight='bold')
    # y_CoM value below the label
    _idx = np.argmin([np.sin(theta)*bx + np.cos(theta)*by for bx, by in corners_body])
    _y_val = abs(-(np.sin(theta)*corners_body[_idx, 0]
                   + np.cos(theta)*corners_body[_idx, 1]))
    right_x = world[:-1, 0].max() + 0.10
    ax.text(right_x, mid_y, f'y_CoM = {_y_val:.3f} m',
            ha='left', va='center', fontsize=7.5, color='gray')


configs = [
    (1.5,  0,    np.pi/2,  'green',      'long side flat\n(global min)'),
    (4.5,  0,    np.pi,    'royalblue',  'short side flat\n(local min)'),
    (7.5,  0,    np.arctan2(b/2, a/2) + np.pi/2,
                           'tomato',     'corner down\n(unstable maximum)'),
]

for cx, cy, theta, col, lbl in configs:
    draw_crate_sketch(ax2, cx, cy, theta, a, b, col, lbl, scale=0.9)

ax2.set_xlim(-1, 10)
ax2.set_ylim(-1.0, 1.2)
ax2.set_title('Corresponding configurations', fontsize=9, color='gray', pad=2)

plt.tight_layout(rect=[0, 0, 1, 0.96])
from pathlib import Path
out = Path(__file__).parent / 'energy_minima_en.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved : {out}')
plt.show()
