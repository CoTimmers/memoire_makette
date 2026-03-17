"""Optimal Control Problem — Rockit / IPOPT  —  TEMPS MINIMAL
   Même dynamique que dynamique_discrétisee.py.
   Objectif : minimiser le temps T pour amener l de 0 à b avec ḷ(T)=0.
   T est une variable libre (FreeTime). Contraintes terminales DURES.
"""

import numpy as np
import casadi as ca
from rockit import Ocp, MultipleShooting, FreeTime
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
T_sim  = 10.0          # borne supérieure sur T
F_max  = 30.0          # borne sur les forces [N]
dF_max = 20.0          # slew rate maximal [N/s]

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
_t_sw = _t1 + (3*np.pi/4 - _p1) / _v1   # instant ψ = 135° ≈ 3.0 s

def wall_profile_ca(t):
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
    psi, psidot, psiddot = wall_profile_ca(t)

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
    delta_dot    = delta_l * ldot + delta_psi * psidot

    cd = ca.cos(delta);  sd = ca.sin(delta)
    R  = ca.vertcat(ca.horzcat( cd, -sd), ca.horzcat( sd,  cd))
    Rp = ca.vertcat(ca.horzcat(-sd, -cd), ca.horzcat( cd, -sd))

    rAO_w = ca.mtimes(R,  ca.DM(rAO_body))
    rAB_w = ca.mtimes(R,  ca.DM(rAB_body))
    Rp_r  = ca.mtimes(Rp, ca.DM(rAO_body))
    R_r   = ca.mtimes(R,  ca.DM(rAO_body))

    ax_coeff = 1.0 + Rp_r[0] * delta_l
    ay_coeff =       Rp_r[1] * delta_l

    nl_ddot = (delta_ll      * ldot**2
             + 2*delta_lpsi  * ldot * psidot
             + delta_psipsi  * psidot**2
             + delta_psi     * psiddot)

    ax_nl = Rp_r[0] * nl_ddot - R_r[0] * delta_dot**2
    ay_nl = Rp_r[1] * nl_ddot - R_r[1] * delta_dot**2

    tB    = ca.vertcat( ca.cos(psi),  ca.sin(psi))
    nB    = ca.vertcat( ca.sin(psi), -ca.cos(psi))
    fBdir = nB - mu * tB

    def cross2(r, f):
        return r[0]*f[1] - r[1]*f[0]

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
    return z[0], z[1], z[2]   # lddot, fyA, fBn


# ══════════════════════════════════════════════════════════════════════
# OCP  ROCKIT — TEMPS MINIMAL
# ══════════════════════════════════════════════════════════════════════
ocp = Ocp(T=FreeTime(T_sim))   # T libre, initialisé à T_sim

# ── États ─────────────────────────────────────────────────────────────
l    = ocp.state()
ldot = ocp.state()
fx   = ocp.state()    # force en x (état pour limiter le slew rate)
fy   = ocp.state()    # force en y

# ── Contrôles = taux de changement des forces ─────────────────────────
dfx  = ocp.control()
dfy  = ocp.control()

t = ocp.t

lddot, fyA, fBn = dynamics_ca(l, ldot, fx, fy, t)

ocp.set_der(l,    ldot)
ocp.set_der(ldot, lddot)
ocp.set_der(fx,   dfx)
ocp.set_der(fy,   dfy)

# ── Objectif : minimiser le temps total ───────────────────────────────
ocp.add_objective(ocp.T)

# Petite régularisation pour le conditionnement numérique
# ocp.add_objective(1e-4 * ocp.integral(fx**2 + fy**2))

# ── Contraintes physiques (souples) ───────────────────────────────────
W_c = 1e2
ocp.add_objective(W_c * ocp.integral(ca.fmax(0.0, -fyA)**2))
ocp.add_objective(W_c * ocp.integral(ca.fmax(0.0, -fBn)**2))

# ── Contraintes terminales DURES ──────────────────────────────────────
ocp.subject_to(ocp.at_tf(l)    == b)      # l(T) = b
ocp.subject_to(ocp.at_tf(ldot) == 0.0)   # ḷ(T) = 0

# ── Bornes sur T ──────────────────────────────────────────────────────
ocp.subject_to(ocp.T >= _t_sw)    # T ≥ t_sw (le bac ne peut pas bouger avant)
ocp.subject_to(ocp.T <= T_sim)    # T ≤ T_sim

# ── Bornes sur les forces ─────────────────────────────────────────────
ocp.subject_to(fx >= -F_max)
ocp.subject_to(fx <=  F_max)
ocp.subject_to(fy >= -F_max)
ocp.subject_to(fy <=  F_max)

# ── Slew rate ─────────────────────────────────────────────────────────
ocp.subject_to(dfx**2 + dfy**2 <= dF_max**2)

# ── Bornes sur l ──────────────────────────────────────────────────────
ocp.subject_to(l >= 0)
ocp.subject_to(l <= b)

# ── Conditions initiales ──────────────────────────────────────────────
ocp.subject_to(ocp.at_t0(l)    == 0.0)
ocp.subject_to(ocp.at_t0(ldot) == 0.0)
ocp.subject_to(ocp.at_t0(fx)   == 0.0)
ocp.subject_to(ocp.at_t0(fy)   == 0.0)

# ── Solveur ───────────────────────────────────────────────────────────
ocp.solver('ipopt', {
    'ipopt.print_level': 5,
    'ipopt.max_iter'   : 1000,
    'ipopt.tol'        : 1e-6,
})
ocp.method(MultipleShooting(N=100, intg='rk'))

print("Résolution OCP temps minimal...")
sol = ocp.solve()

T_opt = sol.value(ocp.T)
print(f"Résolu !  T_opt = {T_opt:.4f} s")

# ══════════════════════════════════════════════════════════════════════
# EXTRACTION DE LA SOLUTION
# ══════════════════════════════════════════════════════════════════════
ts,   l_sol    = sol.sample(l,    grid='control')
_,    ldot_sol = sol.sample(ldot, grid='control')
_,    fx_sol   = sol.sample(fx,   grid='control')
_,    fy_sol   = sol.sample(fy,   grid='control')
_,    dfx_sol  = sol.sample(dfx,  grid='control')
_,    dfy_sol  = sol.sample(dfy,  grid='control')

# ── Recalcul des forces de contact ────────────────────────────────────
def contact_forces_np(l, ldot, fx, fy, t):
    if t <= _t1:
        psi, psidot, psiddot = np.pi/2 + 0.5*_acc*t**2, _acc*t, _acc
    elif t <= _t2:
        psi, psidot, psiddot = _p1 + _v1*(t-_t1), _v1, 0.0
    elif t <= T_wall:
        psi  = _p2 + _v1*(t-_t2) - 0.5*_acc*(t-_t2)**2
        psidot, psiddot = _v1 - _acc*(t-_t2), -_acc
    else:
        psi, psidot, psiddot = np.pi, 0.0, 0.0

    eps = 1e-9
    u   = np.clip((l/b)*np.sin(psi), -1+eps, 1-eps)
    k   = 1.0/np.sqrt(1 - u**2 + eps);  k3 = k**3
    delta = psi - np.pi/2 + np.arcsin(u)
    u_l, u_psi = np.sin(psi)/b, (l/b)*np.cos(psi)
    delta_l    = k * u_l
    delta_psi  = 1 + k * u_psi
    nl_ddot    = ((u*k3*u_l**2)*ldot**2
                  + 2*(k*(np.cos(psi)/b) + u*k3*u_l*u_psi)*ldot*psidot
                  + (k*(-(l/b)*np.sin(psi)) + u*k3*u_psi**2)*psidot**2
                  + delta_psi*psiddot)
    cd, sd       = np.cos(delta), np.sin(delta)
    R            = np.array([[cd,-sd],[sd,cd]])
    Rp           = np.array([[-sd,-cd],[cd,-sd]])
    rAO_w, rAB_w = R@rAO_body, R@rAB_body
    Rp_r, R_r    = Rp@rAO_body, R@rAO_body
    ax_coeff  = 1.0 + Rp_r[0]*delta_l;  ay_coeff = Rp_r[1]*delta_l
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
        return z[1], z[2]
    except np.linalg.LinAlgError:
        return 0.0, 0.0

fyA_sol = np.array([contact_forces_np(float(li), float(ldi), float(fxi), float(fyi), float(ti))[0]
                    for li, ldi, fxi, fyi, ti in zip(l_sol, ldot_sol, fx_sol, fy_sol, ts)])
fBn_sol = np.array([contact_forces_np(float(li), float(ldi), float(fxi), float(fyi), float(ti))[1]
                    for li, ldi, fxi, fyi, ti in zip(l_sol, ldot_sol, fx_sol, fy_sol, ts)])

# ══════════════════════════════════════════════════════════════════════
# PLOTS  — une figure par courbe, fond blanc
# ══════════════════════════════════════════════════════════════════════
dF_norm = np.sqrt(dfx_sol**2 + dfy_sol**2)

# ── l(t) ──
fig1, ax = plt.subplots(figsize=(7, 4))
ax.plot(ts, l_sol, color='steelblue', lw=1.8, label='l(t)')
ax.axhline(b, color='seagreen', lw=1, ls=':', alpha=0.8, label=f'target b={b} m')
ax.axvline(T_opt, color='darkorange', lw=1.2, ls='--', alpha=0.8, label=f'T_opt={T_opt:.2f} s')
ax.set_title('l(t)')
ax.set_ylabel('l (m)'); ax.set_xlabel('t (s)')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('l(t)_minT.png', dpi=150, bbox_inches='tight')
plt.show()

# ── ḷ(t) ──
fig2, ax = plt.subplots(figsize=(7, 4))
ax.plot(ts, ldot_sol, color='mediumpurple', lw=1.8)
ax.axhline(0, color='gray', lw=1)
ax.axvline(T_opt, color='darkorange', lw=1.2, ls='--', alpha=0.8, label=f'T_opt={T_opt:.2f} s')
ax.set_title('ḷ(t)')
ax.set_ylabel('ḷ (m/s)'); ax.set_xlabel('t (s)')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ldot(t)_minT.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Commande fx, fy  (zoomé sur les données) ──
fig3, ax = plt.subplots(figsize=(7, 4))
ax.plot(ts, fx_sol, color='steelblue', lw=1.8, label='fx')
ax.plot(ts, fy_sol, color='crimson',   lw=1.8, label='fy')
ax.axhline(0, color='gray', lw=1)
ax.axvline(T_opt, color='darkorange', lw=1.2, ls='--', alpha=0.8, label=f'T_opt={T_opt:.2f} s')
f_min = min(np.min(fx_sol), np.min(fy_sol))
f_max = max(np.max(fx_sol), np.max(fy_sol))
f_margin = (f_max - f_min) * 0.15
ax.set_ylim(f_min - f_margin, f_max + f_margin)
ax.set_title('Control forces [fx, fy]')
ax.set_ylabel('N'); ax.set_xlabel('t (s)')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Command_fx,_fy_minT.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Taux de changement dfx, dfy + norme ──
fig4, ax = plt.subplots(figsize=(7, 4))
ax2_ = ax.twinx()
ax.plot(ts, dfx_sol, color='steelblue', lw=1.4, label='dfx')
ax.plot(ts, dfy_sol, color='crimson',   lw=1.4, label='dfy')
ax2_.plot(ts, dF_norm,  color='seagreen',  lw=1.8, ls='--', label='‖df‖')
ax2_.axhline(dF_max, color='darkorange', lw=1, ls=':', alpha=0.8, label=f'dF_max={dF_max}')
ax2_.set_ylabel('‖df‖ (N/s)')
ax.set_title('Rate of change [dfx, dfy]')
ax.set_ylabel('N/s'); ax.set_xlabel('t (s)')
ax.legend(fontsize=9, loc='upper left'); ax2_.legend(fontsize=9, loc='upper right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('dfx_dfy_minT.png', dpi=150, bbox_inches='tight')
plt.show()

# ── fyA — contact mur 1 ──
fig5, ax = plt.subplots(figsize=(7, 4))
ax.plot(ts, fyA_sol, color='steelblue', lw=1.8, label='fyA')
ax.axhline(0, color='gray', lw=1)
ax.axvline(T_opt, color='darkorange', lw=1.2, ls='--', alpha=0.8, label=f'T_opt={T_opt:.2f} s')
ax.set_title('fyA — contact wall 1')
ax.set_ylabel('N'); ax.set_xlabel('t (s)')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fyA_minT.png', dpi=150, bbox_inches='tight')
plt.show()

# ── fBn — contact mur 2 ──
fig6, ax = plt.subplots(figsize=(7, 4))
ax.plot(ts, fBn_sol, color='darkorange', lw=1.8, label='fBn')
ax.axhline(0, color='crimson', lw=1, ls='--', alpha=0.6, label='limit (0)')
ax.axvline(T_opt, color='seagreen', lw=1.2, ls='--', alpha=0.8, label=f'T_opt={T_opt:.2f} s')
ax.set_title('fBn — contact wall 2')
ax.set_ylabel('N'); ax.set_xlabel('t (s)')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fBn_minT.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nT optimal     = {T_opt:.4f} s")
print(f"l final       = {l_sol[-1]:.4f} m  (target = {b} m)")
print(f"ḷ final       = {ldot_sol[-1]:.4f} m/s  (target = 0)")
print(f"fx max        = {np.max(np.abs(fx_sol)):.2f} N")
print(f"fy max        = {np.max(np.abs(fy_sol)):.2f} N")
print(f"‖df‖ max      = {np.max(np.sqrt(dfx_sol**2 + dfy_sol**2)):.2f} N/s  (limit = {dF_max})")
print(f"fyA min       = {np.min(fyA_sol):.3f} N")
print(f"fBn min       = {np.min(fBn_sol):.3f} N")


# ══════════════════════════════════════════════════════════════════════
# ANIMATION
# ══════════════════════════════════════════════════════════════════════
def wall_profile_np(t):
    if t <= _t1:   return np.pi/2 + 0.5*_acc*t**2, _acc*t, _acc
    elif t <= _t2: return _p1 + _v1*(t - _t1), _v1, 0.0
    elif t <= T_wall: return _p2 + _v1*(t - _t2) - 0.5*_acc*(t - _t2)**2, _v1 - _acc*(t - _t2), -_acc
    else:          return np.pi, 0.0, 0.0

def delta_kinematics_np(l, ldot, psi, psidot):
    eps = 1e-9
    u = np.clip((l / b) * np.sin(psi), -1 + eps, 1 - eps)
    k = 1.0 / np.sqrt(max(eps, 1 - u*u))
    delta     = psi - np.pi/2 + np.arcsin(u)
    delta_l   = k * np.sin(psi) / b
    delta_psi = 1 + k * (l / b) * np.cos(psi)
    delta_dot = delta_l * ldot + delta_psi * psidot
    return delta, delta_dot

def rot_np(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])

psi_anim    = np.array([wall_profile_np(float(t))[0]  for t in ts])
psidot_anim = np.array([wall_profile_np(float(t))[1]  for t in ts])
delta_anim  = np.array([delta_kinematics_np(float(li), float(ldi), float(pi), float(pdi))[0]
                        for li, ldi, pi, pdi in zip(l_sol, ldot_sol, psi_anim, psidot_anim)])

xA_anim = l_sol;  yA_anim = np.zeros(len(ts))
xO_anim = np.zeros(len(ts));  yO_anim = np.zeros(len(ts))
xB_anim = np.zeros(len(ts));  yB_anim = np.zeros(len(ts))
for i in range(len(ts)):
    R = rot_np(delta_anim[i])
    rAO = R @ rAO_body;  rAB = R @ rAB_body
    xO_anim[i] = xA_anim[i] + rAO[0];  yO_anim[i] = yA_anim[i] + rAO[1]
    xB_anim[i] = xA_anim[i] + rAB[0];  yB_anim[i] = yA_anim[i] + rAB[1]

fig_anim, ax2 = plt.subplots(figsize=(8, 7))
ax2.set_xlim(-0.65, 0.65);  ax2.set_ylim(-0.1, 0.65)
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)
ax2.set_xlabel('x (m)')
ax2.set_ylabel('y (m)')
ax2.set_title(f'OCP animation — minimum time')

wall1_line, = ax2.plot([0, 0.6], [0, 0],  color='steelblue',    lw=3,  label='wall 1')
wall2_line, = ax2.plot([], [],             color='darkorange',   lw=3,  label='wall 2')
body_line,  = ax2.plot([], [],             color='dimgray',      lw=2,  label='body')
A_pt,       = ax2.plot([], [], 'o',        color='crimson',      ms=7,  label='A')
B_pt,       = ax2.plot([], [], 'o',        color='seagreen',     ms=7,  label='B')
O_pt,       = ax2.plot([], [], 'o',        color='darkorange',   ms=7,  label='O (COM)')
force_line, = ax2.plot([], [],             color='mediumpurple', lw=2,  label='crane force')
target_pt,  = ax2.plot([b], [0], '*',      color='seagreen',     ms=10, label=f'target l=b={b}')
traj_line,  = ax2.plot([], [],             color='darkorange',   lw=1,  alpha=0.4)

info_txt = ax2.text(0.02, 0.97, '', transform=ax2.transAxes,
                    color='black', fontsize=9, va='top', fontfamily='monospace',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='lightgray'))

ax2.legend(loc='upper right', fontsize=8)

traj_x, traj_y = [], []

def update(i):
    psi_i   = psi_anim[i]
    delta_i = delta_anim[i]
    l_i     = float(l_sol[i])
    t_i     = float(ts[i])
    fx_i    = float(fx_sol[i])
    fy_i    = float(fy_sol[i])

    L2 = 0.55
    wall2_line.set_data([0, L2*np.cos(psi_i)], [0, L2*np.sin(psi_i)])

    R = rot_np(delta_i)
    corners = np.array([[0,0],[a,0],[a,b],[0,b],[0,0]])
    world   = np.array([[xA_anim[i], yA_anim[i]] + R @ c for c in corners])
    body_line.set_data(world[:, 0], world[:, 1])

    A_pt.set_data([xA_anim[i]], [yA_anim[i]])
    B_pt.set_data([xB_anim[i]], [yB_anim[i]])
    O_pt.set_data([xO_anim[i]], [yO_anim[i]])

    scale = 0.02
    force_line.set_data([xO_anim[i], xO_anim[i] + fx_i*scale],
                        [yO_anim[i], yO_anim[i] + fy_i*scale])

    traj_x.append(xO_anim[i]);  traj_y.append(yO_anim[i])
    traj_line.set_data(traj_x, traj_y)

    info_txt.set_text(
        f"t      = {t_i:.2f} s\n"
        f"T_opt  = {T_opt:.2f} s\n"
        f"ψ      = {np.degrees(psi_i):.1f}°\n"
        f"δ      = {np.degrees(delta_i):.1f}°\n"
        f"l      = {l_i:.3f} m\n"
        f"fx     = {fx_i:.1f} N\n"
        f"fy     = {fy_i:.1f} N"
    )
    return wall2_line, body_line, A_pt, B_pt, O_pt, force_line, traj_line, info_txt

dt_anim = float(ts[1] - ts[0])
ani = FuncAnimation(fig_anim, update, frames=len(ts),
                    interval=int(1000 * dt_anim), blit=False)

plt.tight_layout()
print("Saving GIF...")
ani.save('animation_minT.gif', writer='pillow', fps=int(1.0 / dt_anim))
print("GIF saved: animation_minT.gif")
plt.show()
