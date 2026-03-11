"""Optimal Control Problem — Rockit / IPOPT
   Dynamique identique à dynamique_discrétisee.py.
   Objectif : suivre l_ref(t)  (sans minimiser l'effort → grandes forces possibles)
   Ajouter   + W_u * ocp.integral(fx**2 + fy**2)  pour régulariser si besoin.
"""

import numpy as np
import casadi as ca
from rockit import Ocp, MultipleShooting
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
F_max  = 30.0          # borne sur les forces de commande [N]

rAB_body = np.array([0.0, b])
rAO_body = np.array([a/2, b/2])

# ══════════════════════════════════════════════════════════════════════
# PROFIL DU MUR  ψ : π/2 → π  (version CasADi symbolique)
# ══════════════════════════════════════════════════════════════════════
_t1   = T_wall * 0.4
_t2   = T_wall * 0.6
_dpsi = np.pi / 2
_acc  = _dpsi / (0.5*_t1**2 + (_t2 - _t1)*_t1 + 0.5*(T_wall - _t2)**2)
_v1   = _acc * _t1
_p1   = np.pi/2 + 0.5*_acc*_t1**2
_p2   = _p1 + _v1*(_t2 - _t1)

def wall_profile_ca(t):
    """Retourne (psi, psidot, psiddot) sous forme CasADi symbolique."""
    psi = ca.if_else(t <= _t1,
            np.pi/2 + 0.5*_acc*t**2,
          ca.if_else(t <= _t2,
            _p1 + _v1*(t - _t1),
          ca.if_else(t <= T_wall,
            _p2 + _v1*(t - _t2) - 0.5*_acc*(t - _t2)**2,
            np.pi)))

    psidot = ca.if_else(t <= _t1,
               _acc * t,
             ca.if_else(t <= _t2,
               _v1,
             ca.if_else(t <= T_wall,
               _v1 - _acc*(t - _t2),
               0.0)))

    psiddot = ca.if_else(t <= _t1,
                _acc,
              ca.if_else(t <= _t2,
                0.0,
              ca.if_else(t <= T_wall,
                -_acc,
                0.0)))
    return psi, psidot, psiddot


# ══════════════════════════════════════════════════════════════════════
# DYNAMIQUE SYMBOLIQUE  →  (l̈, fyA, fBn)
# ══════════════════════════════════════════════════════════════════════
def dynamics_ca(l, ldot, fx, fy, t):
    """
    Résout le système 3×3  M·[l̈, fyA, fBn] = rhs  en CasADi symbolique.
    Retourne  lddot, fyA, fBn.
    """
    psi, psidot, psiddot = wall_profile_ca(t)

    # ── Cinématique de δ ─────────────────────────────────────────────
    eps   = 1e-9
    u     = (l / b) * ca.sin(psi)
    u_c   = ca.fmax(-1 + eps, ca.fmin(1 - eps, u))
    k     = 1.0 / ca.sqrt(1.0 - u_c**2 + eps)
    k3    = k**3

    delta = psi - np.pi/2 + ca.arcsin(u_c)

    u_l      = ca.sin(psi) / b
    u_psi    = (l / b) * ca.cos(psi)
    u_lpsi   = ca.cos(psi) / b
    u_psipsi = -(l / b) * ca.sin(psi)

    delta_l      = k * u_l
    delta_psi    = 1.0 + k * u_psi
    delta_ll     = u_c * k3 * u_l * u_l
    delta_lpsi   = k * u_lpsi   + u_c * k3 * u_l   * u_psi
    delta_psipsi = k * u_psipsi + u_c * k3 * u_psi * u_psi

    delta_dot = delta_l * ldot + delta_psi * psidot

    # ── Rotation ─────────────────────────────────────────────────────
    cd = ca.cos(delta);  sd = ca.sin(delta)
    R  = ca.vertcat(ca.horzcat( cd, -sd), ca.horzcat( sd,  cd))
    Rp = ca.vertcat(ca.horzcat(-sd, -cd), ca.horzcat( cd, -sd))

    rAO_w = ca.mtimes(R,  ca.DM(rAO_body))
    rAB_w = ca.mtimes(R,  ca.DM(rAB_body))
    Rp_r  = ca.mtimes(Rp, ca.DM(rAO_body))
    R_r   = ca.mtimes(R,  ca.DM(rAO_body))

    # ── Coefficients cinématiques ─────────────────────────────────────
    ax_coeff = 1.0 + Rp_r[0] * delta_l
    ay_coeff =       Rp_r[1] * delta_l

    nl_ddot = (delta_ll      * ldot**2
             + 2*delta_lpsi  * ldot * psidot
             + delta_psipsi  * psidot**2
             + delta_psi     * psiddot)

    ax_nl = Rp_r[0] * nl_ddot - R_r[0] * delta_dot**2
    ay_nl = Rp_r[1] * nl_ddot - R_r[1] * delta_dot**2

    # ── Contact en B ─────────────────────────────────────────────────
    tB    = ca.vertcat( ca.cos(psi),  ca.sin(psi))
    nB    = ca.vertcat( ca.sin(psi), -ca.cos(psi))
    fBdir = nB - mu * tB

    def cross2(r, f):
        return r[0]*f[1] - r[1]*f[0]

    # ── Système 3×3 ──────────────────────────────────────────────────
    M_mat = ca.vertcat(
        ca.horzcat( m * ax_coeff,    mu,  -fBdir[0]             ),
        ca.horzcat( m * ay_coeff,  -1.0,  -fBdir[1]             ),
        ca.horzcat( I_A * delta_l,   0.0, -cross2(rAB_w, fBdir) )
    )

    f_vec = ca.vertcat(fx, fy)
    rhs   = ca.vertcat(
        fx - m * ax_nl,
        fy - m * ay_nl,
        cross2(rAO_w, f_vec) - I_A * nl_ddot
    )

    z     = ca.solve(M_mat, rhs)
    lddot = z[0]
    fyA   = z[1]
    fBn   = z[2]

    return lddot, fyA, fBn


# ══════════════════════════════════════════════════════════════════════
# RÉFÉRENCE  l_ref(t)
#   Phase 1  (t < t_sw) : l_ref = 0       (l ne peut pas augmenter)
#   Phase 2  (t_sw ≤ t ≤ T_wall) : rampe linéaire 0 → b
#   Après    (t > T_wall) : b constant
#
# t_sw = instant où ψ atteint 135° = 3π/4
# ψ atteint 135° dans la phase à vitesse constante (t1 < t_sw < t2) :
#   ψ(t) = p1 + v1*(t - t1)  →  t_sw = t1 + (3π/4 - p1) / v1
# ══════════════════════════════════════════════════════════════════════
_t_sw = _t1 + (3*np.pi/4 - _p1) / _v1   # ≈ 3.0 s

def l_ref_ca(t):
    ramp = b * (t - _t_sw) / (T_wall - _t_sw)   # 0 à b sur [t_sw, T_wall]
    return ca.if_else(t < _t_sw,
               0.0,
           ca.if_else(t <= T_wall,
               ramp,
               b))


# ══════════════════════════════════════════════════════════════════════
# OCP  ROCKIT
# ══════════════════════════════════════════════════════════════════════
ocp  = Ocp(T=T_sim)

l    = ocp.state()
ldot = ocp.state()
fx   = ocp.control()
fy   = ocp.control()

t = ocp.t

lddot, fyA, fBn = dynamics_ca(l, ldot, fx, fy, t)

ocp.set_der(l,    ldot)
ocp.set_der(ldot, lddot)

# ── Fonction objectif ────────────────────────────────────────────────
l_ref = l_ref_ca(t)
ocp.add_objective(ocp.integral((l - l_ref)**2))

# Coût terminal : atteindre l=b avec ldot=0
ocp.add_objective(1e2 * (ocp.at_tf(l) - b)**2)
ocp.add_objective(1e1 *  ocp.at_tf(ldot)**2)

# Petite régularisation pour le conditionnement numérique
W_u = 1e-4
ocp.add_objective(W_u * ocp.integral(fx**2 + fy**2))

# ── Contraintes physiques (souples) ──────────────────────────────────
# Les contraintes dures causent une infaisabilité car IPOPT ne peut pas
# trouver un point initial cohérent. On utilise des pénalités quadratiques
# sur les violations : W_c * max(0, -f)^2
W_c = 1e2
ocp.add_objective(W_c * ocp.integral(ca.fmax(0.0, -fyA)**2))   # fyA >= 0
ocp.add_objective(W_c * ocp.integral(ca.fmax(0.0, -fBn)**2))   # fBn >= 0

# ── Contraintes dures sur la commande et l'état ───────────────────────
ocp.subject_to(fx >= -F_max)
ocp.subject_to(fx <=  F_max)
ocp.subject_to(fy >= -F_max)
ocp.subject_to(fy <=  F_max)

ocp.subject_to(l >= 0)
ocp.subject_to(l <= b)

# ── Conditions initiales ─────────────────────────────────────────────
ocp.subject_to(ocp.at_t0(l)    == 0.0)
ocp.subject_to(ocp.at_t0(ldot) == 0.0)

# ── Solveur ──────────────────────────────────────────────────────────
ocp.solver('ipopt', {
    'ipopt.print_level'  : 5,
    'ipopt.max_iter'     : 1000,
    'ipopt.tol'          : 1e-6,
})
ocp.method(MultipleShooting(N=100, intg='rk'))

print("Résolution OCP...")
sol = ocp.solve()
print("Résolu !")

# ══════════════════════════════════════════════════════════════════════
# EXTRACTION DE LA SOLUTION
# ══════════════════════════════════════════════════════════════════════
ts,   l_sol    = sol.sample(l,    grid='control')
_,    ldot_sol = sol.sample(ldot, grid='control')
_,    fx_sol   = sol.sample(fx,   grid='control')
_,    fy_sol   = sol.sample(fy,   grid='control')

def l_ref_np(t):
    if t < _t_sw:
        return 0.0
    elif t <= T_wall:
        return b * (t - _t_sw) / (T_wall - _t_sw)
    else:
        return b

l_ref_num = np.array([l_ref_np(float(ti)) for ti in ts])

# Recalcul numérique des forces de contact le long de la solution
def contact_forces_np(l, ldot, fx, fy, t):
    """Version numpy de la dynamique pour post-traitement."""
    psi_start = np.pi / 2
    t1, t2 = _t1, _t2
    acc, v1, p1, p2 = _acc, _v1, _p1, _p2
    if t <= t1:
        psi, psidot, psiddot = psi_start + 0.5*acc*t**2, acc*t, acc
    elif t <= t2:
        psi, psidot, psiddot = p1 + v1*(t-t1), v1, 0.0
    elif t <= T_wall:
        psi = p2 + v1*(t-t2) - 0.5*acc*(t-t2)**2
        psidot, psiddot = v1 - acc*(t-t2), -acc
    else:
        psi, psidot, psiddot = np.pi, 0.0, 0.0

    eps = 1e-9
    u   = np.clip((l/b)*np.sin(psi), -1+eps, 1-eps)
    k   = 1.0/np.sqrt(1 - u**2 + eps);  k3 = k**3
    delta = psi - np.pi/2 + np.arcsin(u)
    u_l, u_psi = np.sin(psi)/b, (l/b)*np.cos(psi)
    delta_l   = k * u_l
    delta_psi = 1 + k * u_psi
    nl_ddot   = ((u*k3*u_l**2)*ldot**2
                 + 2*(k*(np.cos(psi)/b) + u*k3*u_l*u_psi)*ldot*psidot
                 + (k*(-(l/b)*np.sin(psi)) + u*k3*u_psi**2)*psidot**2
                 + delta_psi*psiddot)
    cd, sd    = np.cos(delta), np.sin(delta)
    R         = np.array([[cd,-sd],[sd,cd]])
    Rp        = np.array([[-sd,-cd],[cd,-sd]])
    rAO_w, rAB_w = R@rAO_body, R@rAB_body
    Rp_r, R_r    = Rp@rAO_body, R@rAO_body
    ax_coeff = 1.0 + Rp_r[0]*delta_l;  ay_coeff = Rp_r[1]*delta_l
    delta_dot = delta_l*ldot + delta_psi*psidot
    ax_nl = Rp_r[0]*nl_ddot - R_r[0]*delta_dot**2
    ay_nl = Rp_r[1]*nl_ddot - R_r[1]*delta_dot**2
    tB = np.array([np.cos(psi), np.sin(psi)])
    nB = np.array([np.sin(psi),-np.cos(psi)])
    fBdir = nB - mu*tB
    def c2(r, f): return r[0]*f[1] - r[1]*f[0]
    M = np.array([[m*ax_coeff, mu, -fBdir[0]],
                  [m*ay_coeff,-1.0,-fBdir[1]],
                  [I_A*delta_l,0.0,-c2(rAB_w,fBdir)]])
    rhs = np.array([fx - m*ax_nl, fy - m*ay_nl,
                    c2(rAO_w, np.array([fx,fy])) - I_A*nl_ddot])
    try:
        z = np.linalg.solve(M, rhs)
        return z[1], z[2]   # fyA, fBn
    except np.linalg.LinAlgError:
        return 0.0, 0.0

fyA_sol = np.array([contact_forces_np(float(li), float(ldi), float(fxi), float(fyi), float(ti))[0]
                    for li, ldi, fxi, fyi, ti in zip(l_sol, ldot_sol, fx_sol, fy_sol, ts)])
fBn_sol = np.array([contact_forces_np(float(li), float(ldi), float(fxi), float(fyi), float(ti))[1]
                    for li, ldi, fxi, fyi, ti in zip(l_sol, ldot_sol, fx_sol, fy_sol, ts)])

# ══════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
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

# ── l(t) vs l_ref ──
axes[0,0].plot(ts, l_sol,    color='#5ee7ff', lw=1.8, label='l(t) OCP')
axes[0,0].plot(ts, l_ref_num, color='#ffb347', lw=1.4, ls='--', label='l_ref')
axes[0,0].axhline(b, color='#56f0a0', lw=1, ls=':', alpha=0.7, label=f'cible b={b} m')
axes[0,0].set_title('l(t) — suivi de référence')
axes[0,0].set_ylabel('l (m)'); axes[0,0].set_xlabel('t (s)')
axes[0,0].legend(fontsize=8, facecolor='#13161e', labelcolor='#c8d4f0')

# ── ḷ(t) ──
axes[0,1].plot(ts, ldot_sol, color='#b57aff', lw=1.8)
axes[0,1].axhline(0, color='#3a4560', lw=1)
axes[0,1].set_title('ḷ(t)')
axes[0,1].set_ylabel('ḷ (m/s)'); axes[0,1].set_xlabel('t (s)')

# ── Commande fx, fy ──
axes[1,0].plot(ts, fx_sol, color='#5ee7ff', lw=1.8, label='fx')
axes[1,0].plot(ts, fy_sol, color='#ff5f6d', lw=1.8, label='fy')
axes[1,0].axhline(0, color='#3a4560', lw=1)
axes[1,0].axhline( F_max, color='#ffffff', lw=0.8, ls=':', alpha=0.4, label=f'±{F_max} N')
axes[1,0].axhline(-F_max, color='#ffffff', lw=0.8, ls=':', alpha=0.4)
axes[1,0].set_title('Commande optimale [fx, fy]')
axes[1,0].set_ylabel('N'); axes[1,0].set_xlabel('t (s)')
axes[1,0].legend(fontsize=8, facecolor='#13161e', labelcolor='#c8d4f0')

# ── Erreur de suivi ──
axes[1,1].plot(ts, l_sol - l_ref_num, color='#56f0a0', lw=1.8)
axes[1,1].axhline(0, color='#3a4560', lw=1)
axes[1,1].set_title('Erreur l(t) − l_ref(t)')
axes[1,1].set_ylabel('m'); axes[1,1].set_xlabel('t (s)')

# ── Forces de contact ──
axes[0,2].plot(ts, fyA_sol, color='#5ee7ff', lw=1.8, label='fyA')
axes[0,2].axhline(0, color='#3a4560', lw=1)
axes[0,2].set_title('fyA — contact mur 1')
axes[0,2].set_ylabel('N'); axes[0,2].set_xlabel('t (s)')
axes[0,2].legend(fontsize=8, facecolor='#13161e', labelcolor='#c8d4f0')

axes[1,2].plot(ts, fBn_sol, color='#ffb347', lw=1.8, label='fBn')
axes[1,2].axhline(0, color='#ff5f6d', lw=1, ls='--', alpha=0.6, label='limite (0)')
axes[1,2].set_title('fBn — contact mur 2')
axes[1,2].set_ylabel('N'); axes[1,2].set_xlabel('t (s)')
axes[1,2].legend(fontsize=8, facecolor='#13161e', labelcolor='#c8d4f0')

plt.suptitle('OCP Rockit — commande optimale pour suivi de l_ref',
             color='#c8d4f0', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('ocp_results.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nl final       = {l_sol[-1]:.4f} m  (cible = {b} m)")
print(f"ḷ final       = {ldot_sol[-1]:.4f} m/s")
print(f"Erreur max    = {np.max(np.abs(l_sol - l_ref_num)):.4f} m")
print(f"fx max        = {np.max(np.abs(fx_sol)):.2f} N")
print(f"fy max        = {np.max(np.abs(fy_sol)):.2f} N")
print(f"fyA min       = {np.min(fyA_sol):.3f} N  (doit être ≥ 0)")
print(f"fBn min       = {np.min(fBn_sol):.3f} N  (doit être ≥ 0)")


# ══════════════════════════════════════════════════════════════════════
# ANIMATION
# ══════════════════════════════════════════════════════════════════════

# ── Reconstruction géométrique sur la grille de contrôle ─────────────
def wall_profile_np(t):
    if t <= _t1:
        return np.pi/2 + 0.5*_acc*t**2, _acc*t, _acc
    elif t <= _t2:
        return _p1 + _v1*(t - _t1), _v1, 0.0
    elif t <= T_wall:
        return _p2 + _v1*(t - _t2) - 0.5*_acc*(t - _t2)**2, _v1 - _acc*(t - _t2), -_acc
    else:
        return np.pi, 0.0, 0.0

def delta_kinematics_np(l, ldot, psi, psidot):
    eps = 1e-9
    u = np.clip((l / b) * np.sin(psi), -1 + eps, 1 - eps)
    k = 1.0 / np.sqrt(max(eps, 1 - u*u))
    delta    = psi - np.pi/2 + np.arcsin(u)
    delta_l  = k * np.sin(psi) / b
    delta_psi = 1 + k * (l / b) * np.cos(psi)
    delta_dot = delta_l * ldot + delta_psi * psidot
    return delta, delta_dot

psi_anim   = np.array([wall_profile_np(float(t))[0]   for t in ts])
psidot_anim = np.array([wall_profile_np(float(t))[1]  for t in ts])
delta_anim = np.array([delta_kinematics_np(float(li), float(ldi), float(pi), float(pdi))[0]
                       for li, ldi, pi, pdi in zip(l_sol, ldot_sol, psi_anim, psidot_anim)])

xA_anim = l_sol
yA_anim = np.zeros(len(ts))

def rot_np(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])

xO_anim = np.zeros(len(ts))
yO_anim = np.zeros(len(ts))
xB_anim = np.zeros(len(ts))
yB_anim = np.zeros(len(ts))
for i in range(len(ts)):
    R = rot_np(delta_anim[i])
    rAO = R @ rAO_body
    rAB = R @ rAB_body
    xO_anim[i] = xA_anim[i] + rAO[0];  yO_anim[i] = yA_anim[i] + rAO[1]
    xB_anim[i] = xA_anim[i] + rAB[0];  yB_anim[i] = yA_anim[i] + rAB[1]

# ── Figure d'animation ────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(8, 7))
fig2.patch.set_facecolor('#0d0f14')
ax2.set_facecolor('#13161e')
ax2.set_xlim(-0.15, 0.65);  ax2.set_ylim(-0.1, 0.65)
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.2, color='#232840')
ax2.tick_params(colors='#8899bb')
for sp in ax2.spines.values():
    sp.set_edgecolor('#232840')
ax2.set_xlabel('x (m)', color='#8899bb')
ax2.set_ylabel('y (m)', color='#8899bb')
ax2.set_title('Animation OCP — commande optimale', color='#c8d4f0')

wall1_line, = ax2.plot([0, 0.6], [0, 0],  color='#5ee7ff', lw=3,  label='mur 1')
wall2_line, = ax2.plot([], [],             color='#ffad3b', lw=3,  label='mur 2')
body_line,  = ax2.plot([], [],             color='#c8d4f0', lw=2,  label='bac')
A_pt,       = ax2.plot([], [], 'o',        color='#ff5f6d', ms=7,  label='A')
B_pt,       = ax2.plot([], [], 'o',        color='#56f0a0', ms=7,  label='B')
O_pt,       = ax2.plot([], [], 'o',        color='#ffb347', ms=7,  label='O (COM)')
force_line, = ax2.plot([], [],             color='#b57aff', lw=2,  label='force grue')
lref_line,  = ax2.plot([], [], 's',        color='#ffb347', ms=5,  alpha=0.5, label='l_ref')
traj_line,  = ax2.plot([], [],             color='#ffb347', lw=1,  alpha=0.4)

info_txt = ax2.text(0.02, 0.97, '', transform=ax2.transAxes,
                    color='#c8d4f0', fontsize=9, va='top', fontfamily='monospace',
                    bbox=dict(facecolor='#13161e', alpha=0.7, edgecolor='#232840'))

ax2.legend(loc='upper right', fontsize=8,
           facecolor='#13161e', labelcolor='#c8d4f0', edgecolor='#232840')

traj_x, traj_y = [], []

def update(i):
    psi_i   = psi_anim[i]
    delta_i = delta_anim[i]
    l_i     = float(l_sol[i])
    t_i     = float(ts[i])
    fx_i    = float(fx_sol[i])
    fy_i    = float(fy_sol[i])
    lref_i  = float(l_ref_num[i])

    # Mur 2
    L2 = 0.55
    wall2_line.set_data([0, L2*np.cos(psi_i)], [0, L2*np.sin(psi_i)])

    # Corps rigide
    R = rot_np(delta_i)
    corners = np.array([[0,0],[a,0],[a,b],[0,b],[0,0]])
    world   = np.array([[xA_anim[i], yA_anim[i]] + R @ c for c in corners])
    body_line.set_data(world[:, 0], world[:, 1])

    A_pt.set_data([xA_anim[i]], [yA_anim[i]])
    B_pt.set_data([xB_anim[i]], [yB_anim[i]])
    O_pt.set_data([xO_anim[i]], [yO_anim[i]])

    # Position de référence (point sur mur 1)
    lref_line.set_data([lref_i], [0.0])

    # Force appliquée
    scale = 0.02
    force_line.set_data([xO_anim[i], xO_anim[i] + fx_i*scale],
                        [yO_anim[i], yO_anim[i] + fy_i*scale])

    # Trace du COM
    traj_x.append(xO_anim[i]);  traj_y.append(yO_anim[i])
    traj_line.set_data(traj_x, traj_y)

    info_txt.set_text(
        f"t    = {t_i:.2f} s\n"
        f"ψ    = {np.degrees(psi_i):.1f}°\n"
        f"δ    = {np.degrees(delta_i):.1f}°\n"
        f"l    = {l_i:.3f} m\n"
        f"lref = {lref_i:.3f} m\n"
        f"fx   = {fx_i:.1f} N\n"
        f"fy   = {fy_i:.1f} N"
    )
    return wall2_line, body_line, A_pt, B_pt, O_pt, force_line, lref_line, traj_line, info_txt

dt_anim = float(ts[1] - ts[0])
ani = FuncAnimation(fig2, update, frames=len(ts),
                    interval=int(1000 * dt_anim), blit=False)

plt.tight_layout()
plt.show()
