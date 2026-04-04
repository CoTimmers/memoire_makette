import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ==========================================
# PARAMETRES PHYSIQUES
# ==========================================
m = 7.0
mu = 0.2
theta_deg = 34.0
theta = np.deg2rad(theta_deg)
c = np.cos(theta) - mu * np.sin(theta)

# ==========================================
# PARAMETRES DE SIMULATION
# ==========================================
dt = 0.001
t_final = 8.0
time = np.arange(0, t_final + dt, dt)
l0 = 0.20
u_max_allowed = 50.0      # [N] limite physique de la force

# ==========================================
# GRILLE Kp / Ki A TESTER
# ==========================================
Kp_values = [1.0, 2.0, 3.0, 5.0, 8.0, 12.0]
Ki_values = [0.2, 0.5, 1.0, 2.0, 4.0]

# ==========================================
# FONCTION DE SIMULATION (ordre 3 : l, v, I)
# ==========================================
def simulate(Kp, Ki):
    l = np.zeros_like(time)
    v = np.zeros_like(time)
    I = np.zeros_like(time)
    u = np.zeros_like(time)

    l[0] = l0

    for k in range(len(time) - 1):
        if l[k] <= 0:
            l[k] = 0.0
            v[k] = 0.0
            u[k] = 0.0
            l[k+1] = 0.0
            v[k+1] = 0.0
            I[k+1] = I[k]
            continue

        # Commande PI avec saturation
        u[k] = Kp * l[k] + Ki * I[k]
        u[k] = np.clip(u[k], 0, u_max_allowed)

        # Dynamique ordre 3
        a_k = -(u[k] * c) / m
        v[k+1] = v[k] + dt * a_k
        l[k+1] = l[k] + dt * v[k]
        I[k+1] = I[k] + dt * l[k]

        if l[k+1] < 0:
            l[k+1] = 0.0
            v[k+1] = 0.0

    u[-1] = np.clip(Kp * l[-1] + Ki * I[-1], 0, u_max_allowed) if l[-1] > 0 else 0.0
    return l, v, u, I

# ==========================================
# CALCUL DES CRITERES
# ==========================================
def compute_score(l, v, u):
    # 1. Temps d'arrivée
    idx = np.where(l <= 1e-4)[0]
    t_arr = time[idx[0]] if len(idx) > 0 else t_final

    # 2. Dépassement : l ne doit jamais devenir négatif (overshoot = rebond)
    overshoot = np.sum(l < -1e-4)  # nombre de pas en négatif (avant butée)

    # 3. Commande max
    u_peak = np.max(u)

    # 4. Oscillations : nombre de changements de signe de v
    v_sign = np.sign(v)
    v_sign = v_sign[v_sign != 0]
    oscillations = np.sum(np.diff(v_sign) != 0)

    return t_arr, u_peak, oscillations

# ==========================================
# SWEEP ET AFFICHAGE
# ==========================================
n_total = len(Kp_values) * len(Ki_values)
colors = cm.viridis(np.linspace(0, 1, n_total))

fig, axes = plt.subplots(len(Ki_values), 1, figsize=(12, 3 * len(Ki_values)), sharex=True)
fig.suptitle("Sweep Kp/Ki — Distance l(t) — Système ordre 3", fontsize=13, fontweight='bold')

summary = []

for j, Ki in enumerate(Ki_values):
    ax = axes[j]
    ax.set_title(f"Ki = {Ki}", fontsize=10)
    ax.set_ylabel("l(t) [m]")
    ax.axhline(0, color='black', linewidth=1.5, linestyle='--', label='cible')
    ax.set_ylim(-0.02, 0.22)
    ax.grid(True)

    color_list = cm.plasma(np.linspace(0.1, 0.9, len(Kp_values)))

    for i, Kp in enumerate(Kp_values):
        l, v, u, I = simulate(Kp, Ki)
        t_arr, u_peak, osc = compute_score(l, v, u)

        label = f"Kp={Kp} | t={t_arr:.2f}s | u_max={u_peak:.1f} | osc={osc}"
        ax.plot(time, l, color=color_list[i], label=label, linewidth=1.5)

        summary.append({
            "Kp": Kp, "Ki": Ki,
            "t_arr": t_arr, "u_peak": u_peak, "osc": osc
        })

    ax.legend(fontsize=7, loc='upper right')

axes[-1].set_xlabel("Temps [s]")
plt.tight_layout()

plt.show()

# ==========================================
# RESUME CONSOLE : top 5 combinaisons
# ==========================================
print("\n===== RESUME DES CRITERES =====")
print(f"{'Kp':>6} {'Ki':>6} {'t_arr':>8} {'u_max':>8} {'osc':>6}")
print("-" * 40)

# Score normalisé : on cherche t_arr petit, u_peak petit, osc=0
t_vals   = np.array([s["t_arr"]   for s in summary])
u_vals   = np.array([s["u_peak"]  for s in summary])
osc_vals = np.array([s["osc"]     for s in summary])

def norm(x):
    r = np.max(x) - np.min(x)
    return (x - np.min(x)) / r if r > 0 else np.zeros_like(x)

score = norm(t_vals) + norm(u_vals) + norm(osc_vals)  # plus bas = mieux

idx_sorted = np.argsort(score)

print("\nTop 5 meilleures combinaisons (score composite bas = bien) :")
print(f"{'Kp':>6} {'Ki':>6} {'t_arr':>8} {'u_max':>8} {'osc':>6} {'score':>8}")
print("-" * 50)
for idx in idx_sorted[:5]:
    s = summary[idx]
    print(f"{s['Kp']:>6.1f} {s['Ki']:>6.2f} {s['t_arr']:>8.3f} {s['u_peak']:>8.2f} {s['osc']:>6}  {score[idx]:>8.3f}")
