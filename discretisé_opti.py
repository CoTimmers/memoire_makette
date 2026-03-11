"""
OCP — Multiple Shooting avec RK4 manuel (sans plugin .dll)
RK4 implémenté comme contraintes de continuité dans CasADi/IPOPT directement,
sans passer par Rockit pour l'intégration.
"""
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ══════════════════════════════════════════════════════════════════════
# PARAMÈTRES PHYSIQUES
# ══════════════════════════════════════════════════════════════════════
m      = 7.0
a      = 0.3
b      = 0.4
I_A    = (m / 3.0) * (a**2 + b**2)
mu     = 0.3
T_wall = 6.0
T_sim  = 10.0

rAB_body = ca.vertcat(0.0, b)
rAO_body = ca.vertcat(a/2, b/2)


# ══════════════════════════════════════════════════════════════════════
# PROFIL DU MUR 2
# ══════════════════════════════════════════════════════════════════════
def wall_profile_ca(t):
    psi_start = ca.pi / 2
    psi_end   = ca.pi
    dpsi      = psi_end - psi_start
    t1  = T_wall * 0.4
    t2  = T_wall * 0.6
    acc = dpsi / (0.5*t1**2 + (t2-t1)*t1 + 0.5*(T_wall-t2)**2)
    v1  = acc * t1
    p1  = psi_start + 0.5*acc*t1**2
    p2  = p1 + v1*(t2 - t1)

    psi = ca.if_else(t <= t1,
            psi_start + 0.5*acc*t**2,
          ca.if_else(t <= t2,
            p1 + v1*(t - t1),
          ca.if_else(t <= T_wall,
            p2 + v1*(t - t2) - 0.5*acc*(t - t2)**2,
            psi_end)))

    psidot = ca.if_else(t <= t1, acc*t,
             ca.if_else(t <= t2, v1,
             ca.if_else(t <= T_wall, v1 - acc*(t - t2), 0.0)))

    psiddot = ca.if_else(t <= t1,     acc,
              ca.if_else(t <= t2,     0.0,
              ca.if_else(t <= T_wall, -acc, 0.0)))

    return psi, psidot, psiddot


# ══════════════════════════════════════════════════════════════════════
# GÉOMÉTRIE
# ══════════════════════════════════════════════════════════════════════
def rot_ca(theta):
    return ca.vertcat(
        ca.horzcat(ca.cos(theta), -ca.sin(theta)),
        ca.horzcat(ca.sin(theta),  ca.cos(theta))
    )

def rot_prime_ca(theta):
    return ca.vertcat(
        ca.horzcat(-ca.sin(theta), -ca.cos(theta)),
        ca.horzcat( ca.cos(theta), -ca.sin(theta))
    )

def cross2(r, f):
    return r[0]*f[1] - r[1]*f[0]


# ══════════════════════════════════════════════════════════════════════
# CINÉMATIQUE DE δ
# ══════════════════════════════════════════════════════════════════════
def delta_kinematics_ca(l, ldot, psi, psidot, psiddot):
    eps = 1e-9
    u_raw = (l / b) * ca.sin(psi)
    u = ca.fmin(1.0 - eps, ca.fmax(-1.0 + eps, u_raw))

    one_minus = ca.fmax(eps, 1 - u*u)
    k  = 1.0 / ca.sqrt(one_minus)
    k3 = k**3

    delta = psi - 0.5*ca.pi + ca.asin(u)

    u_l      = (1/b) * ca.sin(psi)
    u_psi    = (l/b) * ca.cos(psi)
    u_lpsi   = (1/b) * ca.cos(psi)
    u_psipsi = -(l/b) * ca.sin(psi)

    delta_l      = k * u_l
    delta_psi    = 1 + k * u_psi
    delta_ll     = u * k3 * u_l * u_l
    delta_lpsi   = k * u_lpsi   + u * k3 * u_l   * u_psi
    delta_psipsi = k * u_psipsi + u * k3 * u_psi * u_psi

    delta_dot = delta_l * ldot + delta_psi * psidot

    return delta, delta_dot, delta_l, delta_psi, delta_ll, delta_lpsi, delta_psipsi


# ══════════════════════════════════════════════════════════════════════
# DYNAMIQUE  f(x, u, t) = [ldot, lddot]
# Résout M·[lddot, fyA, fBn] = rhs via ca.solve (symbolique)
# Retourne aussi fyA, fBn pour les contraintes de contact
# ══════════════════════════════════════════════════════════════════════
def dynamics_ca(l, ldot, fx, fy, t):
    psi, psidot, psiddot = wall_profile_ca(t)

    (delta, delta_dot, delta_l, delta_psi,
     delta_ll, delta_lpsi, delta_psipsi) = delta_kinematics_ca(
        l, ldot, psi, psidot, psiddot)

    R  = rot_ca(delta)
    Rp = rot_prime_ca(delta)
    rAO_w = R  @ rAO_body
    rAB_w = R  @ rAB_body
    Rp_r  = Rp @ rAO_body
    R_r   = R  @ rAO_body

    ax_coeff = 1.0 + Rp_r[0] * delta_l
    ay_coeff =       Rp_r[1] * delta_l

    nl_ddot = (delta_ll * ldot**2
             + 2*delta_lpsi  * ldot * psidot
             + delta_psipsi  * psidot**2
             + delta_psi     * psiddot)

    ax_nl = Rp_r[0] * nl_ddot - R_r[0] * delta_dot**2
    ay_nl = Rp_r[1] * nl_ddot - R_r[1] * delta_dot**2

    tB    = ca.vertcat( ca.cos(psi),  ca.sin(psi))
    nB    = ca.vertcat( ca.sin(psi), -ca.cos(psi))
    fBdir = nB - mu * tB

    M = ca.vertcat(
        ca.horzcat(m * ax_coeff,   mu,   fBdir[0]             ),
        ca.horzcat(m * ay_coeff,  -1.0,  fBdir[1]             ),
        ca.horzcat(I_A * delta_l,  0.0,  cross2(rAB_w, fBdir) )
    )
    rhs = ca.vertcat(
        fx - m * ax_nl,
        fy - m * ay_nl,
        cross2(rAO_w, ca.vertcat(fx, fy)) - I_A * nl_ddot
    )

    sol   = ca.solve(M, rhs)
    lddot = sol[0]
    fyA   = sol[1]
    fBn   = sol[2]

    # ẋ = [ldot, lddot]
    return ca.vertcat(ldot, lddot), fyA, fBn


# ══════════════════════════════════════════════════════════════════════
# RK4 MANUEL : intègre x sur un pas h à partir de (t, x, u)
# ══════════════════════════════════════════════════════════════════════
def rk4_step_ca(x, u, t, h):
    """
    x = [l, ldot],  u = [fx, fy]
    Retourne x_next et les forces de contact au point k1.
    """
    l_s, ld_s = x[0], x[1]
    fx_s, fy_s = u[0], u[1]

    f1, fyA1, fBn1 = dynamics_ca(l_s,              ld_s,              fx_s, fy_s, t      )
    f2, _,    _    = dynamics_ca(l_s + h/2*f1[0],  ld_s + h/2*f1[1], fx_s, fy_s, t + h/2)
    f3, _,    _    = dynamics_ca(l_s + h/2*f2[0],  ld_s + h/2*f2[1], fx_s, fy_s, t + h/2)
    f4, _,    _    = dynamics_ca(l_s + h*f3[0],    ld_s + h*f3[1],   fx_s, fy_s, t + h  )

    x_next = x + (h/6) * (f1 + 2*f2 + 2*f3 + f4)

    return x_next, fyA1, fBn1


# ══════════════════════════════════════════════════════════════════════
# POIDS
# ══════════════════════════════════════════════════════════════════════
r_f       = 0.01       # régularisation commande
q_tf_l    = 10000.0    # coût terminal sur l=b
q_tf_ldot = 1000.0     # coût terminal sur ldot=0


# ══════════════════════════════════════════════════════════════════════
# MULTIPLE SHOOTING AVEC RK4 MANUEL — CasADi Opti
# ══════════════════════════════════════════════════════════════════════
N  = 50
h  = T_sim / N   # pas de temps de chaque intervalle

opti = ca.Opti()

# ── Variables de décision ──
X  = opti.variable(2, N+1)   # états  [l; ldot]  aux nœuds
U  = opti.variable(2, N)     # commandes [fx; fy] sur chaque intervalle
# lddot aux nœuds k1 de chaque intervalle (pour contraindre phase 1)
LDDOT = opti.variable(1, N)

# ── Grille temporelle ──
t_grid = np.linspace(0, T_sim, N+1)

# ── Contraintes de tir (RK4) + contact ──
# t_switch : instant où psi atteint 135°
T_SWITCH = 3.0000   # [s]

for k in range(N):
    x_k  = X[:, k]
    u_k  = U[:, k]
    t_k  = t_grid[k]

    x_next, fyA_k, fBn_k = rk4_step_ca(x_k, u_k, t_k, h)

    # lddot au point k1 (état courant, commande courante)
    f_k, _, _ = dynamics_ca(x_k[0], x_k[1], u_k[0], u_k[1], t_k)
    lddot_k   = f_k[1]   # deuxième composante de [ldot, lddot]
    opti.subject_to(LDDOT[0, k] == lddot_k)

    # Continuité RK4
    opti.subject_to(X[:, k+1] == x_next)

    # Contact
    opti.subject_to(fyA_k >= 0)
    opti.subject_to(fBn_k >= 0)

    # Bornes sur l
    opti.subject_to(X[0, k] >= 0)
    opti.subject_to(X[0, k] <= b)

    # Phase 1 (t <= T_SWITCH) : lddot=0 → pas d'accélération → l reste à 0
    if t_k <= T_SWITCH:
        opti.subject_to(LDDOT[0, k] == 0.0)

# Borne sur le dernier nœud
opti.subject_to(X[0, N] >= 0)
opti.subject_to(X[0, N] <= b)

# ── Condition initiale ──
opti.subject_to(X[0, 0] == 0.0)
opti.subject_to(X[1, 0] == 0.0)

# ── Coût ──
# L'OCP décide librement de la trajectoire.
# Seul objectif : atteindre l=b, ldot=0 en T_sim
# avec le minimum d'effort de commande.
J = 0
for k in range(N):
    u_k = U[:, k]
    J += r_f * (u_k[0]**2 + u_k[1]**2)   # régularisation commande

# Coût terminal fort
J += q_tf_l    * (X[0,N] - b)**2    # l(T) = b
J += q_tf_ldot * (X[1,N])**2        # ldot(T) = 0

opti.minimize(J)

# ── Initialisation ──
# Warm-start cohérent avec la contrainte l=0 avant T_SWITCH
for k in range(N+1):
    t_k = t_grid[k]
    if t_k < T_SWITCH:
        opti.set_initial(X[0, k], 0.0)
        opti.set_initial(X[1, k], 0.0)
    else:
        s = (t_k - T_SWITCH) / (T_sim - T_SWITCH)
        s = min(1.0, max(0.0, s))
        opti.set_initial(X[0, k], b * (3*s**2 - 2*s**3))
        opti.set_initial(X[1, k], 0.0)
opti.set_initial(U, 0.0)

# ── Solveur IPOPT ──
opti.solver('ipopt', {}, {
    'print_level': 5,
    'max_iter':    200,
    'tol':         1e-6,
})

# ══════════════════════════════════════════════════════════════════════
# RÉSOLUTION
# ══════════════════════════════════════════════════════════════════════
print("Résolution de l'OCP (Multiple Shooting + RK4 manuel)...")
sol = opti.solve()

# ══════════════════════════════════════════════════════════════════════
# EXTRACTION
# ══════════════════════════════════════════════════════════════════════
l_sol    = sol.value(X[0, :])
ldot_sol = sol.value(X[1, :])
fx_sol   = sol.value(U[0, :])
fy_sol   = sol.value(U[1, :])
t_sol    = t_grid

# Pas de référence temporelle — l'OCP décide librement
l_ref_sol    = np.full_like(t_sol, np.nan)
ldot_ref_sol = np.full_like(t_sol, np.nan)
psi_sol      = np.array([float(wall_profile_ca(ca.DM(ti))[0]) for ti in t_sol])
delta_sol    = (psi_sol - np.pi/2) + np.arcsin(
    np.clip((l_sol / b) * np.sin(psi_sol), -1+1e-9, 1-1e-9))

print(f"\n── Résultats ────────────────────────────────")
print(f"l final    = {l_sol[-1]:.4f} m   (cible = {b} m)")
print(f"ldot final = {ldot_sol[-1]:.4f} m/s (cible = 0)")
print(f"fx ∈ [{min(fx_sol):.2f}, {max(fx_sol):.2f}] N")
print(f"fy ∈ [{min(fy_sol):.2f}, {max(fy_sol):.2f}] N")


# ══════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.patch.set_facecolor('#0d0f14')

def style(ax):
    ax.set_facecolor('#13161e')
    ax.tick_params(colors='#8899bb')
    ax.grid(True, alpha=0.2, color='#232840')
    for sp in ax.spines.values(): sp.set_edgecolor('#232840')
    ax.xaxis.label.set_color('#8899bb')
    ax.yaxis.label.set_color('#8899bb')
    ax.title.set_color('#c8d4f0')

for ax in axes.flat: style(ax)

axes[0,0].plot(t_sol, l_sol,     color='#5ee7ff', lw=2,   label='l(t) OCP')
axes[0,0].plot(t_sol, l_ref_sol, color='#56f0a0', lw=1.5, ls='--', label='l_ref')
axes[0,0].axhline(b, color='#ffb347', lw=1, ls=':', alpha=0.6)
axes[0,0].set_title('l(t)'); axes[0,0].set_ylabel('l (m)'); axes[0,0].set_xlabel('t (s)')
axes[0,0].legend(fontsize=8, facecolor='#13161e', labelcolor='#c8d4f0')

axes[0,1].plot(t_sol, ldot_sol,     color='#b57aff', lw=2,   label='ḷ OCP')
axes[0,1].plot(t_sol, ldot_ref_sol, color='#56f0a0', lw=1.5, ls='--', label='ḷ_ref')
axes[0,1].axhline(0, color='#3a4560', lw=1)
axes[0,1].set_title('ḷ(t)'); axes[0,1].set_ylabel('ḷ (m/s)'); axes[0,1].set_xlabel('t (s)')
axes[0,1].legend(fontsize=8, facecolor='#13161e', labelcolor='#c8d4f0')

axes[0,2].plot(t_sol, l_sol - l_ref_sol,       color='#ff5f6d', lw=1.8, label='e_l')
axes[0,2].plot(t_sol, ldot_sol - ldot_ref_sol, color='#ffb347', lw=1.8, label='e_ḷ')
axes[0,2].axhline(0, color='#3a4560', lw=1)
axes[0,2].set_title('Erreur de suivi')
axes[0,2].set_ylabel('erreur'); axes[0,2].set_xlabel('t (s)')
axes[0,2].legend(fontsize=8, facecolor='#13161e', labelcolor='#c8d4f0')

# Commande (N points, pas N+1)
axes[1,0].step(t_sol[:-1], fx_sol, color='#5ee7ff', lw=1.8, label='fx', where='post')
axes[1,0].step(t_sol[:-1], fy_sol, color='#ff5f6d', lw=1.8, label='fy', where='post')
axes[1,0].axhline(0, color='#3a4560', lw=1)
axes[1,0].set_title('Commande [fx, fy]')
axes[1,0].set_ylabel('N'); axes[1,0].set_xlabel('t (s)')
axes[1,0].legend(fontsize=8, facecolor='#13161e', labelcolor='#c8d4f0')

axes[1,1].plot(t_sol, np.degrees(psi_sol),   color='#ffb347', lw=1.8, label='ψ')
axes[1,1].plot(t_sol, np.degrees(delta_sol), color='#56f0a0', lw=1.8, label='δ')
axes[1,1].set_title('Angles'); axes[1,1].set_ylabel('°'); axes[1,1].set_xlabel('t (s)')
axes[1,1].legend(fontsize=8, facecolor='#13161e', labelcolor='#c8d4f0')

axes[1,2].plot(l_sol,     ldot_sol,     color='#5ee7ff', lw=1.8, label='OCP')
axes[1,2].plot(l_ref_sol, ldot_ref_sol, color='#56f0a0', lw=1.5, ls='--', label='ref')
axes[1,2].scatter([l_sol[0]],  [ldot_sol[0]],  color='#56f0a0', s=60, zorder=5)
axes[1,2].scatter([l_sol[-1]], [ldot_sol[-1]], color='#ff5f6d', s=60, zorder=5)
axes[1,2].set_title('Portrait de phase')
axes[1,2].set_xlabel('l (m)'); axes[1,2].set_ylabel('ḷ (m/s)')
axes[1,2].legend(fontsize=8, facecolor='#13161e', labelcolor='#c8d4f0')

plt.suptitle(
    f'OCP — Multiple Shooting + RK4 manuel, N={N}\n'
    f'l_f={l_sol[-1]:.4f}m  ḷ_f={ldot_sol[-1]:.4f}m/s',
    color='#c8d4f0', fontsize=11, fontweight='bold'
)
plt.tight_layout()
plt.savefig('ocp_rk4_results.png', dpi=150, bbox_inches='tight')
plt.show()


# ══════════════════════════════════════════════════════════════════════
# ANIMATION
# ══════════════════════════════════════════════════════════════════════
fig_anim, ax = plt.subplots(figsize=(7, 7))
fig_anim.patch.set_facecolor('#0d0f14')
ax.set_facecolor('#13161e')
ax.set_xlim(-0.15, 0.75); ax.set_ylim(-0.15, 0.65)
ax.set_aspect('equal')
ax.tick_params(colors='#8899bb')
ax.grid(True, alpha=0.2, color='#232840')
for sp in ax.spines.values(): sp.set_edgecolor('#232840')
ax.set_title('Animation du mécanisme', color='#c8d4f0', fontsize=13)
ax.axhline(0, color='#8899bb', lw=1.5, alpha=0.5)
ax.plot(0, 0, 'o', color='#8899bb', ms=6)

wall_line,  = ax.plot([], [], color='#ffb347', lw=2.5,  label='Mur (ψ)')
body_patch, = ax.plot([], [], color='#5ee7ff', lw=2.0,  label='Corps')
traj_line,  = ax.plot([], [], color='#5ee7ff', lw=0.8,  alpha=0.3, ls='--')
point_A,    = ax.plot([], [], 'o', color='#56f0a0', ms=9, label='A')
point_B,    = ax.plot([], [], 'o', color='#ff5f6d', ms=9, label='B')
point_G,    = ax.plot([], [], 'x', color='#b57aff', ms=9, mew=2.5, label='G')
time_text   = ax.text(0.03, 0.96, '', transform=ax.transAxes,
                      color='#c8d4f0', fontsize=10, va='top')
ax.legend(fontsize=9, facecolor='#13161e', labelcolor='#c8d4f0', loc='upper right')

corners_body = np.array([[0,0],[a,0],[a,b],[0,b],[0,0]])

def _body_world(l_v, d_v):
    R = np.array([[np.cos(d_v), -np.sin(d_v)],
                  [np.sin(d_v),  np.cos(d_v)]])
    A = np.array([l_v, 0.0])
    cw = A + (R @ corners_body.T).T
    B  = A + R @ np.array([0., b])
    G  = A + R @ np.array([a/2, b/2])
    return A, B, G, cw

G_traj = np.array([_body_world(l_sol[i], delta_sol[i])[2] for i in range(len(t_sol))])

def animate(i):
    psi_v = psi_sol[i]; d_v = delta_sol[i]; l_v = l_sol[i]
    wlen = 0.8
    wall_line.set_data([-wlen*np.cos(psi_v), wlen*np.cos(psi_v)],
                       [-wlen*np.sin(psi_v), wlen*np.sin(psi_v)])
    A, B, G, cw = _body_world(l_v, d_v)
    body_patch.set_data(cw[:,0], cw[:,1])
    traj_line.set_data(G_traj[:i+1,0], G_traj[:i+1,1])
    point_A.set_data([A[0]], [A[1]])
    point_B.set_data([B[0]], [B[1]])
    point_G.set_data([G[0]], [G[1]])
    time_text.set_text(f't={t_sol[i]:.2f}s  ψ={np.degrees(psi_v):.1f}°  l={l_v:.3f}m')
    return wall_line, body_patch, traj_line, point_A, point_B, point_G, time_text

anim = FuncAnimation(fig_anim, animate, frames=len(t_sol), interval=150, blit=True)
plt.tight_layout()
plt.show()