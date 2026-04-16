"""
Simulation d'une grue à portique (gantry crane) avec charge pendulaire
Modèle linéarisé : θ̈ + (g/L)θ = a(t)/L  (sans amortissement)
Profil de mouvement : S-curve accélération → vitesse croisière → S-curve décélération
Distance cible : 2 m
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, FancyArrowPatch
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────
# 1. PARAMÈTRES
# ─────────────────────────────────────────────
g             = 9.81    # m/s²
L             = 2.2     # m — longueur du câble
a_phys        = 0.085   # m/s² — accélération physique maximale admissible
alpha         = 0.70    # marge de sécurité
a_max         = alpha * a_phys          # = 0.0595 m/s²
theta_max_deg = 5.0                     # degrés — débattement maximal souhaité
X_target      = 2.0                     # m — distance à parcourir

wn = np.sqrt(g / L)          # pulsation naturelle [rad/s]
Tn = 2 * np.pi / wn          # période naturelle [s]

# Durée de la rampe de jerk = Tn/2  →  minimise l'excitation résonnante
t_j   = Tn / 2               # ≈ 1.49 s
j_max = a_max / t_j          # jerk maximal [m/s³]

# ─────────────────────────────────────────────
# 2. CALCUL DU PROFIL DE VITESSE (trapèze symétrique)
# ─────────────────────────────────────────────
#
#  Phase acc  : rampe montée t_j | palier t_plateau | rampe descente t_j
#  Phase cruise : vitesse constante v_cruise pendant t_cruise
#  Phase dec  : miroir symétrique de la phase acc
#
#  On fixe t_plateau = t_j (palier = durée d'une rampe).
#  v_cruise = a_max * (t_j + t_plateau) = a_max * 2*t_j

t_plateau   = t_j
v_cruise    = a_max * (t_j + t_plateau)     # vitesse de croisière [m/s]

# Distance couverte durant la phase d'accélération (intégration exacte)
#   Rampe montée  [0, t_j]           : x += a_max*t_j²/6
#   Palier        [t_j, t_j+t_p]     : v(t_j)=a_max*t_j/2 ; x += v(t_j)*t_p + a_max*t_p²/2
#   Rampe descente [t_j+t_p, 2t_j+t_p]: x += v(t_j+t_p)*t_j - a_max*t_j²/6
v_end_ramp_up = 0.5 * a_max * t_j
x_ramp_up     = a_max * t_j**2 / 6
x_plateau_acc = v_end_ramp_up * t_plateau + 0.5 * a_max * t_plateau**2
v_end_plateau = v_end_ramp_up + a_max * t_plateau
x_ramp_down   = v_end_plateau * t_j - a_max * t_j**2 / 6
x_acc_phase   = x_ramp_up + x_plateau_acc + x_ramp_down   # = x_dec_phase par symétrie

x_cruise = X_target - 2 * x_acc_phase

if x_cruise < -1e-6:
    raise ValueError(
        f"Distance cible {X_target} m trop courte pour le profil choisi "
        f"(minimum = {2*x_acc_phase:.3f} m avec t_j={t_j:.3f} s). "
        "Réduire t_j ou a_max."
    )
x_cruise  = max(x_cruise, 0.0)
t_cruise  = x_cruise / v_cruise

# Instants clés
t1 = t_j
t2 = t_j + t_plateau
t3 = t2 + t_j           # fin accélération
t4 = t3 + t_cruise      # fin croisière
t5 = t4 + t_j
t6 = t5 + t_plateau
t7 = t6 + t_j           # arrêt complet

T_move = t7
T_sim  = T_move + 3.0   # fenêtre post-mouvement

print("=" * 55)
print("PARAMÈTRES DU SYSTÈME")
print("=" * 55)
print(f"  L              = {L} m")
print(f"  g              = {g} m/s²")
print(f"  ωₙ             = {wn:.4f} rad/s")
print(f"  Tₙ             = {Tn:.4f} s")
print(f"  a_phys_max     = {a_phys} m/s²")
print(f"  α              = {alpha}")
print(f"  a_max          = {a_max:.4f} m/s²")
print(f"  t_j (jerk)     = {t_j:.4f} s  (= Tₙ/2)")
print(f"  j_max          = {j_max:.5f} m/s³")
print(f"  v_cruise       = {v_cruise:.5f} m/s")
print(f"  x_acc_phase    = {x_acc_phase:.5f} m")
print(f"  x_cruise       = {x_cruise:.5f} m")
print(f"  t_cruise       = {t_cruise:.4f} s")
print(f"  T_mouvement    = {T_move:.4f} s")
print(f"  X_target       = {X_target} m")
print(f"  θ_max cible    = {theta_max_deg}°")
print("=" * 55)

# ─────────────────────────────────────────────
# 3. PROFIL D'ACCÉLÉRATION S-CURVE COMPLET
# ─────────────────────────────────────────────
def accel_profile(t):
    if t < 0:
        return 0.0
    elif t < t1:                    # rampe montée acc
        return a_max * (t / t_j)
    elif t < t2:                    # palier acc
        return a_max
    elif t < t3:                    # rampe descente acc
        return a_max * (1.0 - (t - t2) / t_j)
    elif t < t4:                    # croisière
        return 0.0
    elif t < t5:                    # rampe montée déc
        return -a_max * ((t - t4) / t_j)
    elif t < t6:                    # palier déc
        return -a_max
    elif t < t7:                    # rampe descente déc
        return -a_max * (1.0 - (t - t6) / t_j)
    else:
        return 0.0

# ─────────────────────────────────────────────
# 4. INTÉGRATION NUMÉRIQUE (Runge-Kutta 4)
# ─────────────────────────────────────────────
dt    = 0.005
N     = int(T_sim / dt) + 1
t_arr = np.linspace(0, T_sim, N)

state    = np.zeros((N, 4))   # [θ, dθ/dt, x_trolley, v_trolley]
state[0] = [0.0, 0.0, 0.0, 0.0]

def deriv(s, t):
    theta, dtheta, x, v = s
    a = accel_profile(t)
    ddtheta = a / L - (g / L) * theta   # modèle linéarisé, sans amortissement
    # Après la fin du mouvement, on gèle la position x du trolley
    if t >= t7:
        v = 0.0
        a = 0.0
    return np.array([dtheta, ddtheta, v, a])

for i in range(N - 1):
    t  = t_arr[i]
    s  = state[i]
    k1 = deriv(s,            t)
    k2 = deriv(s + dt/2*k1,  t + dt/2)
    k3 = deriv(s + dt/2*k2,  t + dt/2)
    k4 = deriv(s + dt*k3,    t + dt)
    state[i+1] = s + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

# Clamp trolley position/velocity after stop (corrects RK4 symmetry drift)
idx_stop = np.searchsorted(t_arr, t7)
state[idx_stop:, 2] = X_target
state[idx_stop:, 3] = 0.0

theta_arr = state[:, 0]
x_arr     = state[:, 2]
v_arr     = state[:, 3]
a_arr     = np.array([accel_profile(t) for t in t_arr])

x_load_rel = L * np.sin(theta_arr)   # déplacement horizontal charge / trolley [m]

theta_max_sim = np.degrees(np.max(np.abs(theta_arr)))
x_final       = x_arr[-1]

print(f"\n  θ_max simulé    = {theta_max_sim:.3f}°  (cible ≤ {theta_max_deg}°)")
print(f"  Contrainte θ    {'✓ respectée' if theta_max_sim <= theta_max_deg else '✗ dépassée'}")
print(f"  Position finale = {x_final:.5f} m  (cible = {X_target} m)")
print(f"  Erreur distance = {abs(x_final - X_target)*1000:.3f} mm")

# ─────────────────────────────────────────────
# 5. GRAPHES STATIQUES
# ─────────────────────────────────────────────
fig_graphs, axes = plt.subplots(4, 1, figsize=(11, 13), sharex=True)
fig_graphs.suptitle(
    f"Grue à portique  —  L={L} m | distance={X_target} m | "
    f"a_max={a_max:.4f} m/s² | t_j={t_j:.2f} s\n"
    f"v_cruise={v_cruise:.4f} m/s  |  T_mouvement={T_move:.2f} s  |  "
    f"θ_max simulé={theta_max_sim:.2f}° (limite {theta_max_deg}°)",
    fontsize=11, y=0.99
)

colors = {"theta": "#378ADD", "pos": "#3B6D11", "vel": "#993556", "acc": "#D85A30"}

phase_zones = [
    (0,  t3, "#378ADD", "Accélération"),
    (t3, t4, "#3B6D11", "Croisière"),
    (t4, t7, "#D85A30", "Décélération"),
]

# — Angle θ(t)
ax = axes[0]
ax.plot(t_arr, np.degrees(theta_arr), color=colors["theta"], lw=1.5, label="θ(t)")
ax.axhline( theta_max_deg, color="gray", ls="--", lw=0.8, label=f"±{theta_max_deg}°")
ax.axhline(-theta_max_deg, color="gray", ls="--", lw=0.8)
ax.axhline(0, color="black", lw=0.4, ls=":")
ax.fill_between(t_arr, -theta_max_deg, theta_max_deg, alpha=0.06, color="gray")
ax.set_ylabel("θ (°)")
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, alpha=0.3)

# — Δx charge / trolley
ax = axes[1]
ax.plot(t_arr, x_load_rel * 100, color=colors["pos"], lw=1.5, label="Δx charge / trolley")
ax.axhline(0, color="black", lw=0.4, ls=":")
ax.set_ylabel("Δx charge [cm]")
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, alpha=0.3)

# — Vitesse trolley
ax = axes[2]
ax.plot(t_arr, v_arr, color=colors["vel"], lw=1.5, label="v trolley(t)")
ax.axhline(v_cruise, color="gray", ls="--", lw=0.8, label=f"v_cruise={v_cruise:.4f} m/s")
ax.axhline(0, color="black", lw=0.4, ls=":")
ax.set_ylabel("v trolley (m/s)")
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, alpha=0.3)

# — Accélération
ax = axes[3]
ax.plot(t_arr, a_arr, color=colors["acc"], lw=1.5, label="a(t) — S-curve")
ax.axhline( a_max,  color="gray", ls="--", lw=0.8, label=f"±a_max={a_max:.4f} m/s²")
ax.axhline(-a_max,  color="gray", ls="--", lw=0.8)
ax.axhline( a_phys, color="red",  ls=":",  lw=0.8, label=f"a_phys_max={a_phys} m/s²")
ax.axhline(0, color="black", lw=0.4, ls=":")
ax.set_ylabel("a (m/s²)")
ax.set_xlabel("Temps (s)")
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, alpha=0.3)

for ax in axes:
    for (ta, tb, col, _) in phase_zones:
        ax.axvspan(ta, min(tb, T_sim), alpha=0.05, color=col)
    ax.axvline(T_move, color="#888", ls=":", lw=0.9)

plt.tight_layout()
plt.savefig("gantry_graphs.png", dpi=150, bbox_inches="tight")
print("\nGraphes sauvegardés → gantry_graphs.png")

# ─────────────────────────────────────────────
# 6. ANIMATION
# ─────────────────────────────────────────────
frame_step = max(1, int(1 / (30 * dt)))
idx_frames = np.arange(0, N, frame_step)
n_frames   = len(idx_frames)

fig_anim = plt.figure(figsize=(14, 7))
fig_anim.patch.set_facecolor("#F8F8F6")
gs = gridspec.GridSpec(2, 2, figure=fig_anim, width_ratios=[2, 1], hspace=0.40, wspace=0.32)
ax_crane = fig_anim.add_subplot(gs[:, 0])
ax_theta = fig_anim.add_subplot(gs[0, 1])
ax_vel   = fig_anim.add_subplot(gs[1, 1])

# ── Crane view ──────────────────────────────
MARGIN    = 0.4
RAIL_Y    = 1.0
TROLLEY_H = 0.18
TROLLEY_W = 0.35
LOAD_R    = 0.12

ax_crane.set_xlim(-MARGIN, X_target + MARGIN)
ax_crane.set_ylim(-L - 0.6, RAIL_Y + 0.5)
ax_crane.set_aspect("equal")
ax_crane.set_facecolor("#FAFAF8")
ax_crane.set_xlabel("Position x (m)", fontsize=10)
ax_crane.set_title(f"Vue physique  —  distance cible = {X_target} m", fontsize=10, fontweight="500")
ax_crane.axhline(RAIL_Y, color="#888", lw=2.0, zorder=1)

for xi in np.linspace(-MARGIN, X_target + MARGIN, 22):
    ax_crane.plot([xi, xi + 0.07], [RAIL_Y, RAIL_Y + 0.07], color="#bbb", lw=0.8)

# Marqueur position cible
ax_crane.axvline(X_target, color="#D85A30", ls="--", lw=1.0, alpha=0.7, zorder=1)
ax_crane.text(X_target + 0.03, -L - 0.45, f"cible\n{X_target} m",
              fontsize=8, color="#D85A30", va="bottom")

trolley_patch = plt.Rectangle(
    (-TROLLEY_W/2, RAIL_Y - TROLLEY_H), TROLLEY_W, TROLLEY_H,
    fc="#378ADD", ec="#185FA5", lw=1.2, zorder=4
)
ax_crane.add_patch(trolley_patch)

cable_line,  = ax_crane.plot([], [], color="#555", lw=1.8, zorder=3)
load_circle  = Circle((0, 0), LOAD_R, fc="#D85A30", ec="#993C1D", lw=1.2, zorder=5)
ax_crane.add_patch(load_circle)
trace_line,  = ax_crane.plot([], [], color="#D85A30", lw=0.8, alpha=0.35, zorder=2)

arrow_vel = FancyArrowPatch(
    (0, RAIL_Y - TROLLEY_H/2), (0, RAIL_Y - TROLLEY_H/2),
    arrowstyle="-|>", color="#993556", mutation_scale=14, lw=1.5, zorder=6
)
ax_crane.add_patch(arrow_vel)

time_text  = ax_crane.text(0.02, 0.97, "", transform=ax_crane.transAxes,
                           fontsize=9.5, va="top", fontfamily="monospace")
theta_text = ax_crane.text(0.02, 0.90, "", transform=ax_crane.transAxes,
                           fontsize=9.5, va="top", color="#378ADD")
vel_text   = ax_crane.text(0.02, 0.83, "", transform=ax_crane.transAxes,
                           fontsize=9.5, va="top", color="#993556")
pos_text   = ax_crane.text(0.02, 0.76, "", transform=ax_crane.transAxes,
                           fontsize=9.5, va="top", color="#3B6D11")

# ── Angle history ───────────────────────────
ax_theta.set_xlim(0, T_sim)
ax_theta.set_ylim(-theta_max_deg * 1.4, theta_max_deg * 1.4)
ax_theta.axhline( theta_max_deg, color="gray", ls="--", lw=0.8)
ax_theta.axhline(-theta_max_deg, color="gray", ls="--", lw=0.8)
ax_theta.axhline(0, color="black", lw=0.4, ls=":")
ax_theta.axvline(T_move, color="#888", ls=":", lw=0.9)
ax_theta.set_ylabel("θ (°)", fontsize=9)
ax_theta.set_title("Angle de la charge", fontsize=9)
ax_theta.grid(True, alpha=0.25)
ax_theta.fill_between([0, T_sim], -theta_max_deg, theta_max_deg, alpha=0.07, color="gray")
theta_line, = ax_theta.plot([], [], color="#378ADD", lw=1.4)
theta_dot,  = ax_theta.plot([], [], "o", color="#378ADD", ms=4)

# ── Velocity history ────────────────────────
v_ylim = max(v_cruise * 1.3, 0.001)
ax_vel.set_xlim(0, T_sim)
ax_vel.set_ylim(-v_ylim * 0.15, v_ylim)
ax_vel.axhline(v_cruise, color="gray", ls="--", lw=0.8)
ax_vel.axhline(0, color="black", lw=0.4, ls=":")
ax_vel.axvline(T_move, color="#888", ls=":", lw=0.9)
ax_vel.set_ylabel("v trolley (m/s)", fontsize=9)
ax_vel.set_xlabel("t (s)", fontsize=9)
ax_vel.set_title("Vitesse du trolley", fontsize=9)
ax_vel.grid(True, alpha=0.25)
vel_line, = ax_vel.plot([], [], color="#993556", lw=1.4)
vel_dot,  = ax_vel.plot([], [], "o", color="#993556", ms=4)

TRACE_LEN = 200

def init_anim():
    cable_line.set_data([], [])
    trace_line.set_data([], [])
    load_circle.center = (0, RAIL_Y - TROLLEY_H - L)
    theta_line.set_data([], [])
    vel_line.set_data([], [])
    return cable_line, load_circle, trolley_patch, trace_line, theta_line, theta_dot, vel_line, vel_dot

def update_anim(frame_num):
    i  = idx_frames[frame_num]
    t  = t_arr[i]
    th = theta_arr[i]
    xc = x_arr[i]
    vc = v_arr[i]

    trolley_patch.set_xy((xc - TROLLEY_W/2, RAIL_Y - TROLLEY_H))

    lx = xc + L * np.sin(th)
    ly = RAIL_Y - TROLLEY_H - L * np.cos(th)
    cable_line.set_data([xc, lx], [RAIL_Y - TROLLEY_H, ly])
    load_circle.center = (lx, ly)

    i_start  = max(0, i - TRACE_LEN * frame_step)
    idx_tr   = np.arange(i_start, i + 1, frame_step)
    lx_tr    = x_arr[idx_tr] + L * np.sin(theta_arr[idx_tr])
    ly_tr    = RAIL_Y - TROLLEY_H - L * np.cos(theta_arr[idx_tr])
    trace_line.set_data(lx_tr, ly_tr)

    v_scale = 10.0
    arrow_vel.set_positions(
        (xc, RAIL_Y - TROLLEY_H/2),
        (xc + vc * v_scale, RAIL_Y - TROLLEY_H/2)
    )

    time_text.set_text(f"t = {t:.2f} s")
    theta_text.set_text(f"θ = {np.degrees(th):+.3f}°")
    vel_text.set_text(f"v = {vc:.5f} m/s")
    pos_text.set_text(f"x = {xc:.4f} m / {X_target} m")

    i0 = max(0, i - 1)
    theta_line.set_data(t_arr[:i0], np.degrees(theta_arr[:i0]))
    theta_dot.set_data([t_arr[i]], [np.degrees(th)])
    vel_line.set_data(t_arr[:i0], v_arr[:i0])
    vel_dot.set_data([t_arr[i]], [vc])

    return (cable_line, load_circle, trolley_patch, trace_line,
            arrow_vel, theta_line, theta_dot, vel_line, vel_dot)

anim = animation.FuncAnimation(
    fig_anim, update_anim, frames=n_frames,
    init_func=init_anim, interval=33, blit=True
)

anim.save("gantry_animation.gif", writer="pillow", fps=30, dpi=110)
print("Animation sauvegardée → gantry_animation.gif")

plt.close("all")
print("\nTerminé.")
