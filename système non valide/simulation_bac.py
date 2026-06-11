import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Paramètres ─────────────────────────────────────────────────────────────────
dt = 0.01           # pas de temps (s)
t_end = 6.0         # durée totale (s)
v_max = 150         # mm/s
a_max = 1200        # mm/s^2
y_wall = 150        # mm (mur à y=150mm)
x_final = 180       # mm

np.random.seed(42)

# ── Simulation ─────────────────────────────────────────────────────────────────
t_arr = np.arange(0, t_end, dt)
n = len(t_arr)

x     = np.zeros(n)
y     = np.zeros(n)
theta = np.zeros(n)
xd    = np.zeros(n)
yd    = np.zeros(n)
thd   = np.zeros(n)

# Conditions initiales
x[0]     = 50.0       # mm
y[0]     = 600.0      # mm (loin du mur)
theta[0] = np.radians(25)  # orientation initiale
xd[0]    = 20.0       # mm/s
yd[0]    = -140.0     # mm/s (se dirige vers le mur)
thd[0]   = np.radians(-15) # rad/s

# Coefficients de restitution pour les rebonds
restitution = 0.35
friction     = 0.6
contact      = False
t_contact    = None

for i in range(1, n):
    t = t_arr[i]

    # Bruit capteur
    noise_pos = 0.3
    noise_vel = 1.5
    noise_ang = np.radians(0.5)

    # Gravité/amortissement naturel
    ax = -friction * xd[i-1] * 0.1
    ay = -friction * yd[i-1] * 0.05
    at = -0.8 * thd[i-1]

    # Mise à jour vitesse
    xd[i] = np.clip(xd[i-1] + ax * dt, -v_max, v_max)
    yd[i] = yd[i-1] + ay * dt
    thd[i] = thd[i-1] + at * dt

    # Mise à jour position
    x[i]     = x[i-1] + xd[i] * dt
    y[i]     = y[i-1] + yd[i] * dt
    theta[i] = theta[i-1] + thd[i] * dt

    # Contact avec le mur (y <= y_wall)
    if y[i] <= y_wall and not contact:
        contact = True
        t_contact = t
        # Rebond
        yd[i] = -restitution * yd[i-1] + np.random.normal(0, 3)
        xd[i] = xd[i-1] * (1 - friction) + np.random.normal(0, 2)
        thd[i] = -restitution * thd[i-1] * 0.7 + np.random.normal(0, np.radians(2))
        y[i] = y_wall

    elif y[i] <= y_wall and contact:
        # Rebonds successifs avec amortissement
        if yd[i] < 0:
            yd[i] = -restitution * yd[i] * 0.6 + np.random.normal(0, 1)
            thd[i] = thd[i] * 0.5 + np.random.normal(0, np.radians(1))
        y[i] = max(y[i], y_wall)

        # Convergence vers position finale
        dt_contact = t - t_contact
        x[i] += (x_final - x[i]) * 0.02
        theta[i] += (0 - theta[i]) * 0.03

    # Bruit capteur
    x[i]     += np.random.normal(0, noise_pos)
    y[i]     += np.random.normal(0, noise_pos)
    theta[i] += np.random.normal(0, noise_ang)
    xd[i]    += np.random.normal(0, noise_vel)
    yd[i]    += np.random.normal(0, noise_vel)
    thd[i]   += np.random.normal(0, np.radians(1))

# Vitesse linéaire totale
v_lin = np.sqrt(xd**2 + yd**2)

# ── Phases ─────────────────────────────────────────────────────────────────────
PHASES = {
    "Transport": "#4A90D9",
    "Contact":   "#E8A838",
    "Stabilisé": "#4CAF50",
}

phase = np.array(["Transport"] * n, dtype=object)
if t_contact is not None:
    idx_contact = np.searchsorted(t_arr, t_contact)
    phase[idx_contact:] = "Contact"
    # Stabilisé quand vitesse < 5 mm/s et ang vel < 2 deg/s
    for i in range(idx_contact, n):
        if v_lin[i] < 5 and abs(np.degrees(thd[i])) < 2:
            phase[i:] = "Stabilisé"
            break

colors = np.array([PHASES[p] for p in phase])

# ── Figure ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(14, 10))
fig.suptitle("Simulation — Bac transporté par grue, contact et stabilisation contre le mur",
             fontsize=13, fontweight='bold')

def colored_plot(ax, data, ylabel, title, hline=None, hline_label=None):
    for i in range(len(t_arr) - 1):
        ax.plot(t_arr[i:i+2], data[i:i+2], color=colors[i], linewidth=1.5)
    if hline is not None:
        ax.axhline(hline, color='red', linestyle='--', linewidth=1, label=hline_label)
        ax.legend(fontsize=8)
    ax.set_xlabel("t (s)", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)
    if t_contact:
        ax.axvline(t_contact, color='gray', linestyle=':', linewidth=1, alpha=0.6)

colored_plot(axes[0, 0], x,               "x (mm)",       "Position x du bac",         hline=x_final,  hline_label=f"x_final = {x_final}mm")
colored_plot(axes[0, 1], y,               "y (mm)",       "Position y du bac",          hline=y_wall,   hline_label=f"y_mur = {y_wall}mm")
colored_plot(axes[1, 0], np.degrees(theta),"θ (deg)",     "Orientation θ du bac",       hline=0,        hline_label="θ = 0°")
colored_plot(axes[1, 1], v_lin,           "‖v‖ (mm/s)",   "Vitesse linéaire totale",    hline=5,        hline_label="seuil stabilité = 5 mm/s")
colored_plot(axes[2, 0], xd,              "ẋ (mm/s)",     "Vitesse x",)
colored_plot(axes[2, 1], np.degrees(thd), "θ̇ (deg/s)",   "Vitesse angulaire",          hline=2,        hline_label="seuil stabilité = 2°/s")

# Légende phases
patches = [mpatches.Patch(color=v, label=k) for k, v in PHASES.items()]
fig.legend(handles=patches, loc='upper right', fontsize=10, title="Phase")

plt.tight_layout(rect=[0, 0, 0.95, 1])
plt.savefig("simulation_bac.png", dpi=150, bbox_inches='tight')
plt.show()
print("Graphe sauvegardé dans simulation_bac.png")
