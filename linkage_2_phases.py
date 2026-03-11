import numpy as np
import casadi as ca
from rockit import Ocp, MultipleShooting
import matplotlib.pyplot as plt

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
# PROFIL DU MUR 2  ψ : π/2 → π  (CasADi symbolique)
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

    psidot = ca.if_else(t <= t1,
               acc*t,
             ca.if_else(t <= t2,
               v1,
             ca.if_else(t <= T_wall,
               v1 - acc*(t - t2),
               0.0)))

    psiddot = ca.if_else(t <= t1,  acc,
              ca.if_else(t <= t2,  0.0,
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
# CINÉMATIQUE DE δ  (CasADi symbolique)
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
# DYNAMIQUE CONTINUE  ẋ = f(x, u, t)
# Retourne aussi fyA et fBn pour les contraintes de contact
# ══════════════════════════════════════════════════════════════════════
def dynamics_ca(l, ldot, fx, fy, t):
    """
    Résout M·[l̈, fyA, fBn]ᵀ = rhs symboliquement (CasADi).
    Retourne : (ldot, lddot, fyA, fBn)
    """
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

    # Coefficients linéaires de l̈
    ax_coeff = 1.0 + Rp_r[0] * delta_l
    ay_coeff =       Rp_r[1] * delta_l

    # Termes non-linéaires
    nl_ddot = (delta_ll      * ldot**2
             + 2*delta_lpsi  * ldot * psidot
             + delta_psipsi  * psidot**2
             + delta_psi     * psiddot)

    ax_nl = Rp_r[0] * nl_ddot - R_r[0] * delta_dot**2
    ay_nl = Rp_r[1] * nl_ddot - R_r[1] * delta_dot**2

    # Vecteurs de contact en B
    tB    = ca.vertcat( ca.cos(psi),  ca.sin(psi))
    nB    = ca.vertcat( ca.sin(psi), -ca.cos(psi))
    fBdir = nB - mu * tB

    # Matrice M 3×3  —  inconnues z = [l̈, fyA, fBn]
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

    sol   = ca.solve(M, rhs)   # CasADi résout symboliquement
    lddot = sol[0]
    fyA   = sol[1]
    fBn   = sol[2]

    return ldot, lddot, fyA, fBn


# ══════════════════════════════════════════════════════════════════════
# TRAJECTOIRE DE RÉFÉRENCE  x_ref(t) = [l_ref(t), ldot_ref(t)]
# Phase 1 (t < t_switch) : l_ref = 0, ldot_ref = 0
# Phase 2 (t >= t_switch) : rampe cubique 0 → b
# ══════════════════════════════════════════════════════════════════════
PSI_SWITCH = 3 * ca.pi / 4   # 135°

def get_t_switch():
    """Calcule numériquement l'instant où psi atteint 135°."""
    t1  = T_wall * 0.4
    t2  = T_wall * 0.6
    psi_start = np.pi / 2
    dpsi = np.pi / 2
    acc = dpsi / (0.5*t1**2 + (t2-t1)*t1 + 0.5*(T_wall-t2)**2)
    v1  = acc * t1
    p1  = psi_start + 0.5*acc*t1**2
    # psi = p1 + v1*(t-t1) = 3pi/4  → t = t1 + (3pi/4 - p1)/v1
    t_switch = t1 + (3*np.pi/4 - p1) / v1
    return float(t_switch)

T_SWITCH = get_t_switch()
print(f"t_switch = {T_SWITCH:.4f} s  (ψ atteint 135°)")

def l_ref_ca(t):
    """Trajectoire de référence pour l — CasADi symbolique."""
    dt_phase2 = T_sim - T_SWITCH
    s = ca.fmin(1.0, ca.fmax(0.0, (t - T_SWITCH) / dt_phase2))
    l_ref    = b * (3*s**2 - 2*s**3)
    ldot_ref = b * 6*s*(1-s) / dt_phase2
    # Phase 1 : l_ref = 0, ldot_ref = 0
    l_ref    = ca.if_else(t < T_SWITCH, 0.0, l_ref)
    ldot_ref = ca.if_else(t < T_SWITCH, 0.0, ldot_ref)
    return l_ref, ldot_ref


# ══════════════════════════════════════════════════════════════════════
# POIDS  Q, R, Qf
# ══════════════════════════════════════════════════════════════════════
q_l    = 100.0
q_ldot = 10.0
r_f    = 0.01

Q  = np.diag([q_l, q_ldot])
R  = np.diag([r_f, r_f])
Qf = 10.0 * Q   # coût terminal 10× plus fort


# ══════════════════════════════════════════════════════════════════════
# FORMULATION OCP AVEC ROCKIT
# ══════════════════════════════════════════════════════════════════════
N  = 50   # nombre de pas de shooting

ocp = Ocp(T=T_sim)

# ── États ──
l    = ocp.state()
ldot = ocp.state()

# ── Commandes ──
fx = ocp.control()
fy = ocp.control()

# ── Temps courant (rockit expose t comme variable symbolique) ──
t = ocp.t

# ── Dynamique ──
_, lddot, fyA, fBn = dynamics_ca(l, ldot, fx, fy, t)

ocp.set_der(l,    ldot )
ocp.set_der(ldot, lddot)

# ── Trajectoire de référence ──
l_r, ldot_r = l_ref_ca(t)
e  = ca.vertcat(l - l_r, ldot - ldot_r)   # erreur d'état 2×1
u  = ca.vertcat(fx, fy)                    # commande 2×1

# ── Coût de stage : eᵀQe + uᵀRu ──
stage_cost = ca.mtimes(ca.mtimes(e.T,  ca.DM(Q)),  e) \
           + ca.mtimes(ca.mtimes(u.T,  ca.DM(R)),  u)
ocp.add_objective(ocp.integral(stage_cost))

# ── Coût terminal : eᵀQf e ──
e_f = ca.vertcat(l - b, ldot - 0.0)       # cible finale : l=b, ldot=0
ocp.add_objective(ocp.at_tf(ca.mtimes(ca.mtimes(e_f.T, ca.DM(Qf)), e_f)))

# ── Contraintes d'état ──
ocp.subject_to(l    >= 0.0)
ocp.subject_to(l    <= b  )

# ── Contraintes de contact ──
ocp.subject_to(fyA  >= 0.0)   # normale en A ≥ 0  (pas d'arrachement)
ocp.subject_to(fBn  >= 0.0)   # normale en B ≥ 0  (pas d'arrachement)

# ── Condition initiale ──
ocp.subject_to(ocp.at_t0(l)    == 0.0)
ocp.subject_to(ocp.at_t0(ldot) == 0.0)

# ── Méthode de discrétisation ──
ocp.method(MultipleShooting(N=N, intg='cvodes'))

# ── Solveur ──
ocp.solver('ipopt', {
    'ipopt.print_level': 5,
    'ipopt.max_iter':    100,
    'ipopt.tol':         1e-6,
    'print_time':        1,
})

# ══════════════════════════════════════════════════════════════════════
# RÉSOLUTION
# ══════════════════════════════════════════════════════════════════════
print("\nRésolution de l'OCP...")
sol = ocp.solve()

# ══════════════════════════════════════════════════════════════════════
# EXTRACTION DES RÉSULTATS
# ══════════════════════════════════════════════════════════════════════
t_sol,    l_sol    = sol.sample(l,    grid='control')
_,        ldot_sol = sol.sample(ldot, grid='control')
_,        fx_sol   = sol.sample(fx,   grid='control')
_,        fy_sol   = sol.sample(fy,   grid='control')

# Trajectoire de référence sur la même grille
l_ref_sol    = np.array([float(l_ref_ca(ca.DM(ti))[0])    for ti in t_sol])
ldot_ref_sol = np.array([float(l_ref_ca(ca.DM(ti))[1]) for ti in t_sol])

# Angles psi et delta
psi_sol = np.array([float(wall_profile_ca(ca.DM(ti))[0]) for ti in t_sol])

print(f"\n── Résultats ──────────────────────────")
print(f"l final    = {l_sol[-1]:.4f} m  (cible = {b} m)")
print(f"ldot final = {ldot_sol[-1]:.4f} m/s  (cible = 0)")
print(f"fx range   = [{min(fx_sol):.2f}, {max(fx_sol):.2f}] N")
print(f"fy range   = [{min(fy_sol):.2f}, {max(fy_sol):.2f}] N")


# ══════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.patch.set_facecolor('#0d0f14')

def style(ax):
    ax.set_facecolor('#13161e')
    ax.tick_params(colors='#8899bb')
    ax.grid(True, alpha=0.2, color='#232840')
    for sp in ax.spines.values():
        sp.set_edgecolor('#232840')
    ax.xaxis.label.set_color('#8899bb')
    ax.yaxis.label.set_color('#8899bb')
    ax.title.set_color('#c8d4f0')

for ax in axes.flat:
    style(ax)

def vline(ax):
    ax.axvline(T_SWITCH, color='#ffb347', lw=1.2, ls='--', alpha=0.8, label='ψ=135°')

# ── l(t) ──
axes[0,0].plot(t_sol, l_sol,      color='#5ee7ff', lw=2,   label='l(t) OCP')
axes[0,0].plot(t_sol, l_ref_sol,  color='#56f0a0', lw=1.5, ls='--', label='l_ref(t)')
axes[0,0].axhline(b, color='#ffb347', lw=1, ls=':', alpha=0.6, label=f'b={b}m')
vline(axes[0,0])
axes[0,0].set_title('l(t)'); axes[0,0].set_ylabel('l (m)'); axes[0,0].set_xlabel('t (s)')
axes[0,0].legend(fontsize=8, facecolor='#13161e', labelcolor='#c8d4f0')

# ── ldot(t) ──
axes[0,1].plot(t_sol, ldot_sol,      color='#b57aff', lw=2,   label='ḷ(t) OCP')
axes[0,1].plot(t_sol, ldot_ref_sol,  color='#56f0a0', lw=1.5, ls='--', label='ḷ_ref(t)')
axes[0,1].axhline(0, color='#3a4560', lw=1)
vline(axes[0,1])
axes[0,1].set_title('ḷ(t)'); axes[0,1].set_ylabel('ḷ (m/s)'); axes[0,1].set_xlabel('t (s)')
axes[0,1].legend(fontsize=8, facecolor='#13161e', labelcolor='#c8d4f0')

# ── Erreur de suivi ──
axes[0,2].plot(t_sol, l_sol - l_ref_sol,      color='#ff5f6d', lw=1.8, label='e_l = l - l_ref')
axes[0,2].plot(t_sol, ldot_sol - ldot_ref_sol, color='#ffb347', lw=1.8, label='e_ḷ = ḷ - ḷ_ref')
axes[0,2].axhline(0, color='#3a4560', lw=1)
vline(axes[0,2])
axes[0,2].set_title('Erreur de suivi')
axes[0,2].set_ylabel('erreur'); axes[0,2].set_xlabel('t (s)')
axes[0,2].legend(fontsize=8, facecolor='#13161e', labelcolor='#c8d4f0')

# ── Commande fx, fy ──
axes[1,0].plot(t_sol, fx_sol, color='#5ee7ff', lw=1.8, label='fx')
axes[1,0].plot(t_sol, fy_sol, color='#ff5f6d', lw=1.8, label='fy')
axes[1,0].axhline(0, color='#3a4560', lw=1)
vline(axes[1,0])
axes[1,0].set_title('Commande [fx, fy]')
axes[1,0].set_ylabel('N'); axes[1,0].set_xlabel('t (s)')
axes[1,0].legend(fontsize=8, facecolor='#13161e', labelcolor='#c8d4f0')

# ── Angles ──
axes[1,1].plot(t_sol, np.degrees(psi_sol), color='#ffb347', lw=1.8, label='ψ (mur 2)')
axes[1,1].axhline(135, color='#ffb347', lw=1, ls=':', alpha=0.5)
vline(axes[1,1])
axes[1,1].set_title('Angle ψ(t)')
axes[1,1].set_ylabel('degrés'); axes[1,1].set_xlabel('t (s)')
axes[1,1].legend(fontsize=8, facecolor='#13161e', labelcolor='#c8d4f0')

# ── Portrait de phase ──
axes[1,2].plot(l_sol, ldot_sol, color='#5ee7ff', lw=1.8, label='OCP')
axes[1,2].plot(l_ref_sol, ldot_ref_sol, color='#56f0a0', lw=1.5, ls='--', label='référence')
axes[1,2].scatter([l_sol[0]],  [ldot_sol[0]],  color='#56f0a0', s=60, zorder=5, label='début')
axes[1,2].scatter([l_sol[-1]], [ldot_sol[-1]], color='#ff5f6d', s=60, zorder=5, label='fin')
axes[1,2].set_title('Portrait de phase (l, ḷ)')
axes[1,2].set_xlabel('l (m)'); axes[1,2].set_ylabel('ḷ (m/s)')
axes[1,2].legend(fontsize=8, facecolor='#13161e', labelcolor='#c8d4f0')

plt.suptitle(
    f'OCP Rockit  —  Q=diag({q_l},{q_ldot}), R={r_f}·I,  N={N} pas\n'
    f'l_f={l_sol[-1]:.4f}m  ḷ_f={ldot_sol[-1]:.4f}m/s',
    color='#c8d4f0', fontsize=11, fontweight='bold'
)
plt.tight_layout()
plt.savefig('ocp_results.png', dpi=150, bbox_inches='tight')
plt.show()


# ══════════════════════════════════════════════════════════════════════
# ANIMATION DU MÉCANISME
# ══════════════════════════════════════════════════════════════════════
from matplotlib.animation import FuncAnimation

# Calcul numérique de δ sur la grille
delta_sol = (psi_sol - np.pi / 2) + np.arcsin(
    np.clip((l_sol / b) * np.sin(psi_sol), -1 + 1e-9, 1 - 1e-9)
)

fig_anim, ax = plt.subplots(figsize=(7, 7))
fig_anim.patch.set_facecolor('#0d0f14')
ax.set_facecolor('#13161e')
ax.set_xlim(-0.15, 0.75)
ax.set_ylim(-0.15, 0.65)
ax.set_aspect('equal')
ax.tick_params(colors='#8899bb')
ax.grid(True, alpha=0.2, color='#232840')
for sp in ax.spines.values():
    sp.set_edgecolor('#232840')
ax.xaxis.label.set_color('#8899bb')
ax.yaxis.label.set_color('#8899bb')
ax.set_title('Animation du mécanisme', color='#c8d4f0', fontsize=13)

# Sol fixe
ax.axhline(0, color='#8899bb', lw=1.5, alpha=0.5)
ax.axvline(0, color='#8899bb', lw=1,   alpha=0.3)
ax.plot(0, 0, 'o', color='#8899bb', ms=6)  # pivot du mur

wall_line,  = ax.plot([], [], color='#ffb347', lw=2.5,  label='Mur (ψ)')
body_patch, = ax.plot([], [], color='#5ee7ff', lw=2.0,  label='Corps', solid_capstyle='round')
traj_line,  = ax.plot([], [], color='#5ee7ff', lw=0.8,  alpha=0.3, ls='--')
point_A,    = ax.plot([], [], 'o', color='#56f0a0', ms=9,  label='A (sol)',   zorder=6)
point_B,    = ax.plot([], [], 'o', color='#ff5f6d', ms=9,  label='B (mur)',   zorder=6)
point_G,    = ax.plot([], [], 'x', color='#b57aff', ms=9,  mew=2.5, label='G (cdm)', zorder=6)
time_text   = ax.text(0.03, 0.96, '', transform=ax.transAxes,
                      color='#c8d4f0', fontsize=10, va='top')
ax.legend(fontsize=9, facecolor='#13161e', labelcolor='#c8d4f0', loc='upper right')

# Trajectoire de A au sol (trace statique)
ax.plot(l_ref_sol, np.zeros_like(l_ref_sol), color='#56f0a0',
        lw=1, ls=':', alpha=0.4)

# Coins du corps en repère local : A=[0,0], puis [a,0],[a,b],[0,b]
corners_body = np.array([[0, 0], [a, 0], [a, b], [0, b], [0, 0]])

def _body_world(l_v, d_v):
    R = np.array([[np.cos(d_v), -np.sin(d_v)],
                  [np.sin(d_v),  np.cos(d_v)]])
    A_pos = np.array([l_v, 0.0])
    cw = A_pos + (R @ corners_body.T).T
    B_pos = A_pos + R @ np.array([0.0, b])
    G_pos = A_pos + R @ np.array([a / 2, b / 2])
    return A_pos, B_pos, G_pos, cw

# Pré-calcul de la trace du cdm
G_traj = np.array([_body_world(l_sol[i], delta_sol[i])[2] for i in range(len(t_sol))])

def init():
    wall_line.set_data([], [])
    body_patch.set_data([], [])
    traj_line.set_data([], [])
    point_A.set_data([], [])
    point_B.set_data([], [])
    point_G.set_data([], [])
    time_text.set_text('')
    return wall_line, body_patch, traj_line, point_A, point_B, point_G, time_text

def animate(i):
    l_v   = l_sol[i]
    psi_v = psi_sol[i]
    d_v   = delta_sol[i]

    # Mur : ligne passant par l'origine dans la direction ψ
    wlen = 0.8
    wall_line.set_data(
        [-wlen * np.cos(psi_v), wlen * np.cos(psi_v)],
        [-wlen * np.sin(psi_v), wlen * np.sin(psi_v)],
    )

    A_pos, B_pos, G_pos, cw = _body_world(l_v, d_v)
    body_patch.set_data(cw[:, 0], cw[:, 1])
    traj_line.set_data(G_traj[:i+1, 0], G_traj[:i+1, 1])
    point_A.set_data([A_pos[0]], [A_pos[1]])
    point_B.set_data([B_pos[0]], [B_pos[1]])
    point_G.set_data([G_pos[0]], [G_pos[1]])
    time_text.set_text(
        f't = {t_sol[i]:.2f} s   ψ = {np.degrees(psi_v):.1f}°'
        f'   l = {l_v:.3f} m'
    )
    return wall_line, body_patch, traj_line, point_A, point_B, point_G, time_text

anim = FuncAnimation(
    fig_anim, animate, init_func=init,
    frames=len(t_sol), interval=150, blit=True
)
plt.tight_layout()
plt.show()
