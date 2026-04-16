"""
Simulation d'une grue à portique (gantry crane) avec charge pendulaire
Modèle linéarisé : θ̈ + (g/L)θ = a(t)/L  (sans amortissement)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, FancyArrowPatch
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────
# 1. PARAMÈTRES
# ─────────────────────────────────────────────
g       = 9.81       # m/s²
L       = 2.2        # m  — longueur du câble
a_phys  = 0.085      # m/s²  — accélération physique maximale admissible
alpha   = 0.70       # marge de sécurité
a_max   = alpha * a_phys          # = 0.0595 m/s²
theta_max_deg = 5.0               # degrés — débattement maximal souhaité
theta_max     = np.radians(theta_max_deg)

wn = np.sqrt(g / L)               # pulsation naturelle [rad/s]
Tn = 2 * np.pi / wn               # période naturelle [s]

# Durée de transition jerk (S-curve)
# Pour limiter le dépassement dynamique, on choisit t_j ~ Tn/2
# (transition lente devant la dynamique du pendule)
# On part de t_j = Tn/2 et on vérifie que θ_max est respecté
t_j  = Tn / 2                     # ≈ 1.49 s
t_acc = 8.0                       # durée totale de la phase d'accélération [s]

# Jerk maximal correspondant
j_max = a_max / t_j

print("=" * 50)
print("PARAMÈTRES DU SYSTÈME")
print("=" * 50)
print(f"  L            = {L} m")
print(f"  g            = {g} m/s²")
print(f"  ωₙ           = {wn:.4f} rad/s")
print(f"  Tₙ           = {Tn:.4f} s")
print(f"  a_phys_max   = {a_phys} m/s²")
print(f"  α            = {alpha}")
print(f"  a_max        = {a_max:.4f} m/s²")
print(f"  t_j (jerk)   = {t_j:.4f} s  (= Tₙ/2)")
print(f"  j_max        = {j_max:.5f} m/s³")
print(f"  θ_max cible  = {theta_max_deg}°")
print("=" * 50)

# ─────────────────────────────────────────────
# 2. PROFIL D'ACCÉLÉRATION S-CURVE
# ─────────────────────────────────────────────
def accel_profile(t, a_max, t_j, t_acc):
    """
    Profil S-curve :
      [0, t_j]           : montée linéaire 0 → a_max
      [t_j, t_acc-t_j]   : palier a_max
      [t_acc-t_j, t_acc] : descente linéaire a_max → 0
      [t_acc, ∞]         : a = 0
    """
    if t < 0:
        return 0.0
    elif t < t_j:
        return a_max * (t / t_j)
    elif t < t_acc - t_j:
        return a_max
    elif t < t_acc:
        return a_max * (1.0 - (t - (t_acc - t_j)) / t_j)
    else:
        return 0.0

# ─────────────────────────────────────────────
# 3. INTÉGRATION NUMÉRIQUE (Runge-Kutta 4)
# ─────────────────────────────────────────────
dt     = 0.005
T_sim  = t_acc + 4.0   # on laisse un peu de temps après la fin de l'accélération
N      = int(T_sim / dt) + 1
t_arr  = np.linspace(0, T_sim, N)

# État : [θ, dθ/dt, x_trolley, v_trolley]
state = np.zeros((N, 4))
state[0] = [0.0, 0.0, 0.0, 0.0]

def deriv(state, t):
    theta, dtheta, x, v = state
    a = accel_profile(t, a_max, t_j, t_acc)
    ddtheta = a / L - (g / L) * theta   # modèle linéarisé, sans amortissement
    dv      = a
    dx      = v
    return np.array([dtheta, ddtheta, dx, dv])

# RK4
for i in range(N - 1):
    t  = t_arr[i]
    s  = state[i]
    k1 = deriv(s,            t)
    k2 = deriv(s + dt/2*k1,  t + dt/2)
    k3 = deriv(s + dt/2*k2,  t + dt/2)
    k4 = deriv(s + dt*k3,    t + dt)
    state[i+1] = s + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

theta_arr  = state[:, 0]
dtheta_arr = state[:, 1]
x_arr      = state[:, 2]
v_arr      = state[:, 3]
a_arr      = np.array([accel_profile(t, a_max, t_j, t_acc) for t in t_arr])

# Position de la charge relative au trolley
x_load_rel = L * np.sin(theta_arr)   # déplacement horizontal relatif
y_load_rel = -L * np.cos(theta_arr)  # toujours négatif (suspendu)

theta_max_sim = np.degrees(np.max(np.abs(theta_arr)))
print(f"\n  θ_max simulé = {theta_max_sim:.3f}°  (cible ≤ {theta_max_deg}°)")
print(f"  Contrainte {'✓ respectée' if theta_max_sim <= theta_max_deg else '✗ dépassée'}")

# ─────────────────────────────────────────────
# 4. GRAPHES STATIQUES
# ─────────────────────────────────────────────
fig_graphs, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
fig_graphs.suptitle(
    f"Dynamique de la grue à portique  —  L={L} m, a_max={a_max:.4f} m/s², "
    f"t_j={t_j:.2f} s\n"
    f"θ_max simulé = {theta_max_sim:.2f}°  (limite = {theta_max_deg}°)",
    fontsize=12, y=0.98
)

colors = {"theta": "#378ADD", "pos": "#3B6D11", "vel": "#993556", "acc": "#D85A30"}

# — Angle
ax = axes[0]
ax.plot(t_arr, np.degrees(theta_arr), color=colors["theta"], lw=1.5, label="θ(t)")
ax.axhline( theta_max_deg, color="gray", ls="--", lw=0.8, label=f"±{theta_max_deg}°")
ax.axhline(-theta_max_deg, color="gray", ls="--", lw=0.8)
ax.axhline(0, color="black", lw=0.4, ls=":")
ax.fill_between(t_arr, -theta_max_deg, theta_max_deg, alpha=0.06, color="gray")
ax.set_ylabel("θ (°)")
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, alpha=0.3)

# — Position de la charge relative au trolley (horizontal)
ax = axes[1]
ax.plot(t_arr, x_load_rel * 100, color=colors["pos"], lw=1.5, label="Δx charge / trolley")
ax.axhline(0, color="black", lw=0.4, ls=":")
ax.set_ylabel("Δx charge [cm]")
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, alpha=0.3)

# — Vitesse du trolley
ax = axes[2]
ax.plot(t_arr, v_arr, color=colors["vel"], lw=1.5, label="v_trolley(t)")
ax.axhline(0, color="black", lw=0.4, ls=":")
ax.set_ylabel("v trolley (m/s)")
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, alpha=0.3)

# — Accélération
ax = axes[3]
ax.plot(t_arr, a_arr, color=colors["acc"], lw=1.5, label="a(t) — profil S-curve")
ax.axhline(a_max,   color="gray", ls="--", lw=0.8, label=f"a_max = {a_max:.4f} m/s²")
ax.axhline(a_phys,  color="red",  ls=":",  lw=0.8, label=f"a_phys_max = {a_phys} m/s²")
ax.axhline(0, color="black", lw=0.4, ls=":")
ax.set_ylabel("a (m/s²)")
ax.set_xlabel("Temps (s)")
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, alpha=0.3)

# Zone d'accélération active
for ax in axes:
    ax.axvspan(0, t_acc, alpha=0.04, color="#378ADD")

plt.tight_layout()
plt.savefig("gantry_graphs.png", dpi=150, bbox_inches="tight")
print("\nGraphes sauvegardés → gantry_graphs.png")

# ─────────────────────────────────────────────
# 5. ANIMATION
# ─────────────────────────────────────────────
# Sous-échantillonnage pour l'animation (30 fps effectif)
frame_step  = max(1, int(1 / (30 * dt)))
idx_frames  = np.arange(0, N, frame_step)
n_frames    = len(idx_frames)

fig_anim = plt.figure(figsize=(13, 7))
fig_anim.patch.set_facecolor("#F8F8F6")
gs = gridspec.GridSpec(2, 2, figure=fig_anim, width_ratios=[2, 1], hspace=0.38, wspace=0.32)

# Panneau principal : vue physique de la grue
ax_crane = fig_anim.add_subplot(gs[:, 0])
ax_theta = fig_anim.add_subplot(gs[0, 1])
ax_vel   = fig_anim.add_subplot(gs[1, 1])

# ── Crane view ──────────────────────────────
VIEW_W     = 3.0   # demi-largeur de la vue [m]
RAIL_Y     = 1.0   # hauteur du rail
TROLLEY_H  = 0.18
TROLLEY_W  = 0.35
LOAD_R     = 0.12

ax_crane.set_xlim(-VIEW_W, VIEW_W)
ax_crane.set_ylim(-L - 0.6, RAIL_Y + 0.5)
ax_crane.set_aspect("equal")
ax_crane.set_facecolor("#FAFAF8")
ax_crane.set_xlabel("Position x (m)", fontsize=10)
ax_crane.set_title("Vue physique de la grue", fontsize=10, fontweight="500")
ax_crane.axhline(RAIL_Y, color="#888", lw=1.5, ls="-", zorder=1)

# Rail décoratif
for xi in np.linspace(-VIEW_W, VIEW_W, 15):
    ax_crane.plot([xi, xi + 0.1], [RAIL_Y, RAIL_Y + 0.1], color="#aaa", lw=0.8)

trolley_patch = plt.Rectangle(
    (-TROLLEY_W/2, RAIL_Y - TROLLEY_H), TROLLEY_W, TROLLEY_H,
    fc="#378ADD", ec="#185FA5", lw=1.2, zorder=4
)
ax_crane.add_patch(trolley_patch)

cable_line,  = ax_crane.plot([], [], color="#555", lw=1.5, zorder=3)
load_circle  = Circle((0, 0), LOAD_R, fc="#D85A30", ec="#993C1D", lw=1.2, zorder=5)
ax_crane.add_patch(load_circle)

# Flèche vitesse sur le trolley
arrow_vel = FancyArrowPatch((0, RAIL_Y - TROLLEY_H/2), (0, RAIL_Y - TROLLEY_H/2),
                             arrowstyle="-|>", color="#993556",
                             mutation_scale=14, lw=1.5, zorder=6)
ax_crane.add_patch(arrow_vel)

time_text  = ax_crane.text(0.02, 0.97, "", transform=ax_crane.transAxes,
                           fontsize=9.5, va="top", fontfamily="monospace")
theta_text = ax_crane.text(0.02, 0.90, "", transform=ax_crane.transAxes,
                           fontsize=9.5, va="top", color="#378ADD")
vel_text   = ax_crane.text(0.02, 0.83, "", transform=ax_crane.transAxes,
                           fontsize=9.5, va="top", color="#993556")

# ── Angle history ───────────────────────────
ax_theta.set_xlim(0, T_sim)
ax_theta.set_ylim(-theta_max_deg * 1.3, theta_max_deg * 1.3)
ax_theta.axhline( theta_max_deg, color="gray", ls="--", lw=0.8)
ax_theta.axhline(-theta_max_deg, color="gray", ls="--", lw=0.8)
ax_theta.axhline(0, color="black", lw=0.4, ls=":")
ax_theta.set_ylabel("θ (°)", fontsize=9)
ax_theta.set_title("Angle de la charge", fontsize=9)
ax_theta.grid(True, alpha=0.25)
ax_theta.fill_between([0, T_sim], -theta_max_deg, theta_max_deg, alpha=0.07, color="gray")
theta_line, = ax_theta.plot([], [], color="#378ADD", lw=1.4)
theta_dot,  = ax_theta.plot([], [], "o", color="#378ADD", ms=4)

# ── Velocity history ────────────────────────
ax_vel.set_xlim(0, T_sim)
v_max = np.max(v_arr) * 1.2 + 0.001
ax_vel.set_ylim(-0.005, v_max)
ax_vel.axhline(0, color="black", lw=0.4, ls=":")
ax_vel.set_ylabel("v trolley (m/s)", fontsize=9)
ax_vel.set_xlabel("t (s)", fontsize=9)
ax_vel.set_title("Vitesse du trolley", fontsize=9)
ax_vel.grid(True, alpha=0.25)
vel_line, = ax_vel.plot([], [], color="#993556", lw=1.4)
vel_dot,  = ax_vel.plot([], [], "o", color="#993556", ms=4)

# ── Mise à jour ─────────────────────────────
def init_anim():
    cable_line.set_data([], [])
    load_circle.center = (0, -L)
    theta_line.set_data([], [])
    vel_line.set_data([], [])
    return cable_line, load_circle, trolley_patch, theta_line, theta_dot, vel_line, vel_dot

def update_anim(frame_num):
    i  = idx_frames[frame_num]
    t  = t_arr[i]
    th = theta_arr[i]
    xc = x_arr[i]       # position trolley
    vc = v_arr[i]

    # Trolley (centré dans la vue relative)
    trolley_x = xc % (2 * VIEW_W) - VIEW_W   # scroll cyclique pour la viz
    trolley_patch.set_xy((trolley_x - TROLLEY_W/2, RAIL_Y - TROLLEY_H))

    # Câble + charge
    lx = trolley_x + L * np.sin(th)
    ly = RAIL_Y - TROLLEY_H - L * np.cos(th)
    cable_line.set_data([trolley_x, lx], [RAIL_Y - TROLLEY_H, ly])
    load_circle.center = (lx, ly)

    # Flèche vitesse (échelle : 1 m/s → 0.5 m visuel)
    v_scale = 4.0
    arrow_vel.set_positions(
        (trolley_x, RAIL_Y - TROLLEY_H/2),
        (trolley_x + vc * v_scale, RAIL_Y - TROLLEY_H/2)
    )

    # Textes
    time_text.set_text(f"t = {t:.2f} s")
    theta_text.set_text(f"θ = {np.degrees(th):+.2f}°")
    vel_text.set_text(f"v = {vc:.4f} m/s")

    # Graphes rolling
    i0 = max(0, i - 1)
    theta_line.set_data(t_arr[:i0], np.degrees(theta_arr[:i0]))
    theta_dot.set_data([t_arr[i]], [np.degrees(th)])
    vel_line.set_data(t_arr[:i0], v_arr[:i0])
    vel_dot.set_data([t_arr[i]], [vc])

    return cable_line, load_circle, trolley_patch, arrow_vel, theta_line, theta_dot, vel_line, vel_dot

anim = animation.FuncAnimation(
    fig_anim, update_anim, frames=n_frames,
    init_func=init_anim, interval=33, blit=True
)

anim.save(
    "gantry_animation.gif",
    writer="pillow", fps=30, dpi=110
)
print("Animation sauvegardée → gantry_animation.gif")

plt.close("all")
print("\nTerminé.")
