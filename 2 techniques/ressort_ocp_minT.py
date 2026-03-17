"""Optimal Control Problem — Rockit / IPOPT
   Dynamique : bac + mur 2 dynamique (ressort de torsion).
   État      : (l, ldot, psi, psidot, tau_clock, fx, fy)
   Contrôle  : (dfx, dfy) — taux de variation des forces

   Objectif  : minimiser le temps T (horizon libre)
"""

import numpy as np
import casadi as ca
from rockit import Ocp, MultipleShooting, FreeTime
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ══════════════════════════════════════════════════════════════════════
# PARAMÈTRES PHYSIQUES
# ══════════════════════════════════════════════════════════════════════
m        = 7.0
a        = 0.3
b        = 0.4
I_A      = (m / 3.0) * (a**2 + b**2)
mu       = 0.3

k_spring  = 1.0
J_mur     = 4.0
psi_rest  = np.pi / 2

rAB_body = np.array([0.0, b])
rAO_body = np.array([a/2, b/2])

T_init     = 10.0          # valeur initiale de T pour le warm start
T_min      = 2.0           # borne inférieure sur T
T_max      = 15.0          # borne supérieure sur T
F_max      = 40.0
dF_max     = 20.0
PSI_SWITCH = 140.0 * np.pi / 180
T_RAMP     = 5.0

# ══════════════════════════════════════════════════════════════════════
# DYNAMIQUE
# ══════════════════════════════════════════════════════════════════════
def dynamics_ca(l, ldot, psi, psidot, fx, fy):
    eps = 1e-9
    u    = (l / b) * ca.sin(psi)
    u_c  = ca.fmax(-1 + eps, ca.fmin(1 - eps, u))
    k    = 1.0 / ca.sqrt(1.0 - u_c**2 + eps)
    k3   = k**3

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
    NL = (delta_ll * ldot**2 + 2*delta_lpsi * ldot * psidot
          + delta_psipsi * psidot**2)

    delta = psi - np.pi/2 + ca.arcsin(u_c)
    cd = ca.cos(delta);  sd = ca.sin(delta)
    R  = ca.vertcat(ca.horzcat( cd, -sd), ca.horzcat( sd,  cd))
    Rp = ca.vertcat(ca.horzcat(-sd, -cd), ca.horzcat( cd, -sd))

    rAO_w = ca.mtimes(R,  ca.DM(rAO_body))
    rAB_w = ca.mtimes(R,  ca.DM(rAB_body))
    Rpr   = ca.mtimes(Rp, ca.DM(rAO_body))
    Rr    = ca.mtimes(R,  ca.DM(rAO_body))
    yB_world = rAB_w[1]

    NLx = Rpr[0] * NL - Rr[0] * delta_dot**2
    NLy = Rpr[1] * NL - Rr[1] * delta_dot**2

    tB    = ca.vertcat( ca.cos(psi),  ca.sin(psi))
    nB    = ca.vertcat( ca.sin(psi), -ca.cos(psi))
    fBdir = nB - mu * tB
    xB    = l + rAB_w[0]
    rOB   = ca.vertcat(xB, rAB_w[1])

    def cross2(r, f): return r[0]*f[1] - r[1]*f[0]

    M_mat = ca.vertcat(
        ca.horzcat(m*(1 + Rpr[0]*delta_l),  mu,   -fBdir[0],              -m*Rpr[0]*delta_psi),
        ca.horzcat(m*Rpr[1]*delta_l,        -1.0,  -fBdir[1],              -m*Rpr[1]*delta_psi),
        ca.horzcat(I_A*delta_l,              0.0,  -cross2(rAB_w, fBdir), -I_A*delta_psi     ),
        ca.horzcat(0.0,                      0.0,   cross2(rOB,   fBdir),  J_mur             )
    )
    f_vec = ca.vertcat(fx, fy)
    rhs = ca.vertcat(
        fx - m * NLx,
        fy - m * NLy,
        cross2(rAO_w, f_vec) - I_A * NL,
        -k_spring * (psi - psi_rest)
    )
    z = ca.solve(M_mat, rhs)
    return z[0], z[3], z[1], z[2], yB_world   # lddot, psiddot, fyA, fBn, yB


# ══════════════════════════════════════════════════════════════════════
# OCP — horizon libre
# ══════════════════════════════════════════════════════════════════════
ocp = Ocp(T=FreeTime(T_init))   # T est une variable libre, initialisée à T_init

# ── États ─────────────────────────────────────────────────────────────
l         = ocp.state()
ldot      = ocp.state()
psi       = ocp.state()
psidot    = ocp.state()
tau_clock = ocp.state()
fx        = ocp.state()
fy        = ocp.state()

# ── Contrôles ─────────────────────────────────────────────────────────
dfx = ocp.control()
dfy = ocp.control()

lddot, psiddot, fyA, fBn, yB = dynamics_ca(l, ldot, psi, psidot, fx, fy)

gate_on      = 1.0 / (1.0 + ca.exp(-40.0 * (psi      - PSI_SWITCH)))
gate_started = 1.0 / (1.0 + ca.exp(-40.0 * (tau_clock - 0.1)))

# ── Équations d'état ───────────────────────────────────────────────────
ocp.set_der(l,         ldot)
ocp.set_der(ldot,      lddot)
ocp.set_der(psi,       psidot)
ocp.set_der(psidot,    psiddot)
ocp.set_der(tau_clock, ca.fmax(gate_on, gate_started))
ocp.set_der(fx,        dfx)
ocp.set_der(fy,        dfy)

# ══════════════════════════════════════════════════════════════════════
# FONCTION DE COÛT — minimiser T + régularisations légères
# ══════════════════════════════════════════════════════════════════════
W_T   = 1.0     # poids sur le temps (objectif principal)
W_u   = 1e-3    # régularisation dfx/dfy (lissage léger)
W_c   = 1e2     # pénalité contacts négatifs

ocp.add_objective(W_T * ocp.T)                                          # ← minimiser T
ocp.add_objective(W_u * ocp.integral(dfx**2 + dfy**2))                 # lissage forces
ocp.add_objective(W_c * ocp.integral(ca.fmax(0.0, -fyA)**2))           # fyA >= 0
ocp.add_objective(W_c * ocp.integral(ca.fmax(0.0, -fBn)**2))           # fBn >= 0
ocp.add_objective(1e2 * ocp.integral((l - b)**2))   # pousse l vers b

# ── Contraintes terminales dures ──────────────────────────────────────
ocp.subject_to(ocp.at_tf(l)    == b)
ocp.subject_to(ocp.at_tf(ldot) == 0.0)

# ══════════════════════════════════════════════════════════════════════
# CONTRAINTES
# ══════════════════════════════════════════════════════════════════════
# Bornes sur T
ocp.subject_to(ocp.T >= T_min)
ocp.subject_to(ocp.T <= T_max)

# Forces bornées
ocp.subject_to(fx  >= -F_max);    ocp.subject_to(fx  <= F_max)
ocp.subject_to(fy  >= -F_max);    ocp.subject_to(fy  <= F_max)

# Taux de variation bornés
ocp.subject_to(dfx >= -dF_max);   ocp.subject_to(dfx <= dF_max)
ocp.subject_to(dfy >= -dF_max);   ocp.subject_to(dfy <= dF_max)

ocp.subject_to(l   >= 0.0);       ocp.subject_to(l   <= b)
ocp.subject_to(psi >= np.pi/2);   ocp.subject_to(psi <= np.pi)
ocp.subject_to(ldot    >= 0.0)
ocp.subject_to(tau_clock >= 0.0)
ocp.subject_to(yB  >= 0.0)

# l ne peut avancer que si le mur est ouvert
ocp.subject_to(ldot <= 50.0 * gate_on + 1e-3)

# ── Conditions initiales ───────────────────────────────────────────────
ocp.subject_to(ocp.at_t0(l)         == 0.0)
ocp.subject_to(ocp.at_t0(ldot)      == 0.0)
ocp.subject_to(ocp.at_t0(psi)       == np.pi/2)
ocp.subject_to(ocp.at_t0(psidot)    == 0.0)
ocp.subject_to(ocp.at_t0(tau_clock) == 0.0)
# fx(0) et fy(0) libres

# ══════════════════════════════════════════════════════════════════════
# SOLVEUR
# ══════════════════════════════════════════════════════════════════════
ocp.solver('ipopt', {
    'ipopt.print_level'    : 5,
    'ipopt.max_iter'       : 200,
    'ipopt.tol'            : 1e-4,
    'ipopt.acceptable_tol' : 1e-3,
    'ipopt.acceptable_iter': 15,
    'ipopt.mu_strategy'    : 'adaptive',
})

N = 50
ocp.method(MultipleShooting(N=N, intg='rk'))

# ══════════════════════════════════════════════════════════════════════
# WARM START
# ══════════════════════════════════════════════════════════════════════
t_sw = 0.35
tau  = np.linspace(0.0, 1.0, N + 1)

psi_init = np.where(
    tau <= t_sw,
    np.pi/2 + (PSI_SWITCH - np.pi/2) * (tau / t_sw),
    PSI_SWITCH + (np.pi * 0.94 - PSI_SWITCH) * ((tau - t_sw) / (1.0 - t_sw))
)
l_init = np.where(tau <= t_sw, 0.0, b * (tau - t_sw) / (1.0 - t_sw))
tau_clock_init = np.where(tau <= t_sw, 0.0, (tau - t_sw) * T_init)

dt_ocp      = T_init / N
psidot_init = np.gradient(psi_init, dt_ocp)
ldot_init   = np.maximum(np.gradient(l_init, dt_ocp), 0.0)

F_push        = 15.0
fx_init_state = np.zeros(N + 1)
fy_init_state = np.zeros(N + 1)
for i in range(N + 1):
    psi_i = float(psi_init[i])
    if psi_i < PSI_SWITCH:
        fx_init_state[i] = -F_push * np.sin(psi_i)
        fy_init_state[i] =  F_push * np.cos(psi_i)
    else:
        fx_init_state[i] =  F_push * np.sqrt(2) / 2
        fy_init_state[i] = -F_push * np.sqrt(2) / 2

dfx_init = np.clip(np.diff(fx_init_state) / dt_ocp, -dF_max, dF_max)
dfy_init = np.clip(np.diff(fy_init_state) / dt_ocp, -dF_max, dF_max)

ocp.set_initial(psi,       psi_init)
ocp.set_initial(psidot,    psidot_init)
ocp.set_initial(l,         l_init)
ocp.set_initial(ldot,      ldot_init)
ocp.set_initial(tau_clock, tau_clock_init)
ocp.set_initial(fx,        fx_init_state)
ocp.set_initial(fy,        fy_init_state)
ocp.set_initial(dfx,       dfx_init)
ocp.set_initial(dfy,       dfy_init)

# ══════════════════════════════════════════════════════════════════════
# RÉSOLUTION
# ══════════════════════════════════════════════════════════════════════
print("Résolution OCP (temps minimal)...")
try:
    sol = ocp.solve()
    print("Résolu !")
except Exception as e:
    sol = ocp.non_converged_solution
    print(f"Solution non convergée — dernier itéré affiché.\nErreur : {e}")

# Récupérer T optimal
T_opt = float(sol.value(ocp.T))
print(f"T optimal = {T_opt:.3f} s")

# ══════════════════════════════════════════════════════════════════════
# EXTRACTION
# ══════════════════════════════════════════════════════════════════════
ts,   l_sol         = sol.sample(l,         grid='control')
_,    ldot_sol      = sol.sample(ldot,      grid='control')
_,    psi_sol       = sol.sample(psi,       grid='control')
_,    psidot_sol    = sol.sample(psidot,    grid='control')
_,    fx_sol        = sol.sample(fx,        grid='control')
_,    fy_sol        = sol.sample(fy,        grid='control')
_,    dfx_sol       = sol.sample(dfx,       grid='control')
_,    dfy_sol       = sol.sample(dfy,       grid='control')
_,    tau_clock_sol = sol.sample(tau_clock, grid='control')

l_ref_num = np.minimum(b, b * tau_clock_sol / T_RAMP)

# ══════════════════════════════════════════════════════════════════════
# POST-TRAITEMENT — forces de contact
# ══════════════════════════════════════════════════════════════════════
def dynamics_np_postproc(lv, ldotv, psiv, psidotv, fxv, fyv):
    eps = 1e-9
    u   = np.clip((lv / b) * np.sin(psiv), -1 + eps, 1 - eps)
    k   = 1.0 / np.sqrt(max(eps, 1.0 - u*u))
    k3  = k**3
    u_l      = np.sin(psiv) / b
    u_psi    = (lv / b) * np.cos(psiv)
    u_lpsi   = np.cos(psiv) / b
    u_psipsi = -(lv / b) * np.sin(psiv)
    delta_l      = k * u_l
    delta_psi    = 1.0 + k * u_psi
    delta_ll     = u * k3 * u_l * u_l
    delta_lpsi   = k * u_lpsi   + u * k3 * u_l   * u_psi
    delta_psipsi = k * u_psipsi + u * k3 * u_psi * u_psi
    delta     = psiv - np.pi/2 + np.arcsin(u)
    delta_dot = delta_l * ldotv + delta_psi * psidotv
    NL = delta_ll*ldotv**2 + 2*delta_lpsi*ldotv*psidotv + delta_psipsi*psidotv**2
    cd, sd = np.cos(delta), np.sin(delta)
    R  = np.array([[cd,-sd],[sd,cd]])
    Rp = np.array([[-sd,-cd],[cd,-sd]])
    rAO_w = R @ rAO_body;  rAB_w = R @ rAB_body
    Rpr   = Rp @ rAO_body; Rr    = R  @ rAO_body
    NLx = Rpr[0]*NL - Rr[0]*delta_dot**2
    NLy = Rpr[1]*NL - Rr[1]*delta_dot**2
    tB    = np.array([np.cos(psiv),  np.sin(psiv)])
    nB    = np.array([np.sin(psiv), -np.cos(psiv)])
    fBdir = nB - mu * tB
    rOB   = np.array([lv + rAB_w[0], rAB_w[1]])
    def cross2(r, f): return r[0]*f[1] - r[1]*f[0]
    M = np.array([
        [m*(1 + Rpr[0]*delta_l),  mu,   -fBdir[0],              -m*Rpr[0]*delta_psi],
        [m*Rpr[1]*delta_l,        -1.0, -fBdir[1],              -m*Rpr[1]*delta_psi],
        [I_A*delta_l,              0.0, -cross2(rAB_w, fBdir), -I_A*delta_psi     ],
        [0.0,                      0.0,  cross2(rOB,   fBdir),  J_mur             ]
    ])
    rhs = np.array([
        fxv - m*NLx, fyv - m*NLy,
        cross2(rAO_w, np.array([fxv, fyv])) - I_A*NL,
        -k_spring*(psiv - psi_rest)
    ])
    try:
        z = np.linalg.solve(M, rhs)
        return z[1], z[2]
    except Exception:
        return 0.0, 0.0

fyA_sol = np.array([dynamics_np_postproc(
    float(a_), float(b_), float(c_), float(d_), float(e_), float(f_))[0]
    for a_, b_, c_, d_, e_, f_ in zip(l_sol, ldot_sol, psi_sol, psidot_sol, fx_sol, fy_sol)])
fBn_sol = np.array([dynamics_np_postproc(
    float(a_), float(b_), float(c_), float(d_), float(e_), float(f_))[1]
    for a_, b_, c_, d_, e_, f_ in zip(l_sol, ldot_sol, psi_sol, psidot_sol, fx_sol, fy_sol)])

# ══════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════
psi_switch_deg = np.degrees(PSI_SWITCH)

fig1, ax = plt.subplots(figsize=(7, 4))
ax.plot(ts, l_sol,     color='steelblue',  lw=1.8, label='l(t) OCP')
ax.plot(ts, l_ref_num, color='darkorange', lw=1.4, ls='--', label=f'l_ref(τ)  ramp {T_RAMP}s')
ax.axhline(b, color='seagreen', lw=1, ls=':', label=f'target b={b} m')
ax.set_title(f'l(t)  —  T_opt = {T_opt:.2f} s')
ax.set_ylabel('l (m)'); ax.set_xlabel('t (s)')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('l_t.png', dpi=150, bbox_inches='tight'); plt.show()

fig2, ax = plt.subplots(figsize=(7, 4))
ax.plot(ts, np.degrees(psi_sol), color='darkorange', lw=1.8, label='ψ (wall 2)')
ax.axhline(psi_switch_deg, color='gray', lw=1, ls='--', label=f'ψ_switch ({psi_switch_deg:.0f}°)')
ax.set_title('Wall 2 angle ψ(t)'); ax.set_ylabel('deg'); ax.set_xlabel('t (s)')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('psi_t.png', dpi=150, bbox_inches='tight'); plt.show()

fig3, ax = plt.subplots(figsize=(7, 4))
ax.plot(ts, fx_sol, color='steelblue', lw=1.8, label='fx')
ax.plot(ts, fy_sol, color='crimson',   lw=1.8, label='fy')
ax.axhline(0, color='gray', lw=1)
ax.set_title('Forces [fx, fy]'); ax.set_ylabel('N'); ax.set_xlabel('t (s)')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('forces.png', dpi=150, bbox_inches='tight'); plt.show()

fig4, ax = plt.subplots(figsize=(7, 4))
ax.plot(ts, dfx_sol, color='steelblue', lw=1.8, label='dfx')
ax.plot(ts, dfy_sol, color='crimson',   lw=1.8, label='dfy')
ax.axhline( dF_max, color='gray', lw=1, ls='--', alpha=0.6, label=f'+{dF_max} N/s')
ax.axhline(-dF_max, color='gray', lw=1, ls='--', alpha=0.6, label=f'-{dF_max} N/s')
ax.axhline(0, color='gray', lw=1)
ax.set_title('[dfx, dfy]'); ax.set_ylabel('N/s'); ax.set_xlabel('t (s)')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('dforces.png', dpi=150, bbox_inches='tight'); plt.show()

fig5, ax = plt.subplots(figsize=(7, 4))
ax.plot(ts, tau_clock_sol, color='seagreen', lw=1.8, label='τ_clock')
ax.set_title('tau_clock(t)'); ax.set_ylabel('s'); ax.set_xlabel('t (s)')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('tau_clock.png', dpi=150, bbox_inches='tight'); plt.show()

fig6, ax = plt.subplots(figsize=(7, 4))
ax.plot(ts, fyA_sol, color='steelblue', lw=1.8, label='fyA')
ax.axhline(0, color='gray', lw=1)
ax.set_title('fyA — wall 1 contact'); ax.set_ylabel('N'); ax.set_xlabel('t (s)')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('fyA.png', dpi=150, bbox_inches='tight'); plt.show()

fig7, ax = plt.subplots(figsize=(7, 4))
ax.plot(ts, fBn_sol, color='darkorange', lw=1.8, label='fBn')
ax.axhline(0, color='crimson', lw=1, ls='--', alpha=0.6)
ax.set_title('fBn — wall 2 contact'); ax.set_ylabel('N'); ax.set_xlabel('t (s)')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('fBn.png', dpi=150, bbox_inches='tight'); plt.show()

print(f"\nT optimal        = {T_opt:.3f} s")
print(f"l final          = {l_sol[-1]:.4f} m  (target = {b} m)")
print(f"ldot final       = {ldot_sol[-1]:.4f} m/s")
print(f"ψ final          = {np.degrees(psi_sol[-1]):.2f}°")
print(f"Final error      = {abs(float(l_sol[-1]) - b):.4f} m")
print(f"max fx           = {np.max(np.abs(fx_sol)):.2f} N")
print(f"max fy           = {np.max(np.abs(fy_sol)):.2f} N")
print(f"max dfx          = {np.max(np.abs(dfx_sol)):.2f} N/s  (limit = {dF_max} N/s)")
print(f"max dfy          = {np.max(np.abs(dfy_sol)):.2f} N/s  (limit = {dF_max} N/s)")
print(f"min fyA          = {np.min(fyA_sol):.3f} N")
print(f"min fBn          = {np.min(fBn_sol):.3f} N")

# ══════════════════════════════════════════════════════════════════════
# ANIMATION
# ══════════════════════════════════════════════════════════════════════
def rot_np(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])

def delta_from_state(lv, psiv):
    eps = 1e-9
    u = np.clip((lv / b) * np.sin(psiv), -1 + eps, 1 - eps)
    return psiv - np.pi/2 + np.arcsin(u)

xA_anim    = l_sol
yA_anim    = np.zeros(len(ts))
delta_anim = np.array([delta_from_state(float(li), float(pi))
                        for li, pi in zip(l_sol, psi_sol)])
xO_anim = np.zeros(len(ts));  yO_anim = np.zeros(len(ts))
xB_anim = np.zeros(len(ts));  yB_anim = np.zeros(len(ts))
for i in range(len(ts)):
    R    = rot_np(delta_anim[i])
    rAO  = R @ rAO_body;  rAB = R @ rAB_body
    xO_anim[i] = xA_anim[i] + rAO[0];  yO_anim[i] = yA_anim[i] + rAO[1]
    xB_anim[i] = xA_anim[i] + rAB[0];  yB_anim[i] = yA_anim[i] + rAB[1]

fig_anim, ax2 = plt.subplots(figsize=(8, 7))
ax2.set_xlim(-0.15, 0.65);  ax2.set_ylim(-0.10, 0.65)
ax2.set_aspect('equal');  ax2.grid(True, alpha=0.3)
ax2.set_xlabel('x (m)');  ax2.set_ylabel('y (m)')
ax2.set_title(f'OCP minimum time — T = {T_opt:.2f} s')

wall1_line, = ax2.plot([0, 0.6], [0, 0],  color='steelblue',    lw=3,  label='wall 1')
wall2_line, = ax2.plot([], [],             color='darkorange',   lw=3,  label='wall 2')
spring_arc, = ax2.plot([], [], 'r--',      lw=1.2, alpha=0.7,          label='spring')
body_line,  = ax2.plot([], [],             color='dimgray',      lw=2,  label='crate')
A_pt,       = ax2.plot([], [], 'o',        color='crimson',      ms=7,  label='A')
B_pt,       = ax2.plot([], [], 'o',        color='seagreen',     ms=7,  label='B')
O_pt,       = ax2.plot([], [], 'o',        color='darkorange',   ms=7,  label='COM')
force_line, = ax2.plot([], [],             color='mediumpurple', lw=2,  label='force')
traj_line,  = ax2.plot([], [],             color='steelblue',    lw=1,  alpha=0.4)
info_txt = ax2.text(0.02, 0.97, '', transform=ax2.transAxes, color='black',
                    fontsize=9, va='top', fontfamily='monospace',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='lightgray'))
ax2.legend(loc='upper right', fontsize=8)
traj_x, traj_y = [], []

def update(i):
    psi_i   = float(psi_sol[i])
    delta_i = delta_anim[i]
    fx_i    = float(fx_sol[i])
    fy_i    = float(fy_sol[i])
    t_i     = float(ts[i])
    tau_i   = float(tau_clock_sol[i])
    phase   = "Phase 1 : open wall" if psi_i < PSI_SWITCH else "Phase 2 : slide crate"

    wall2_line.set_data([0, 0.55*np.cos(psi_i)], [0, 0.55*np.sin(psi_i)])
    theta_arc = np.linspace(psi_rest, psi_i, 30)
    spring_arc.set_data(0.10*np.cos(theta_arc), 0.10*np.sin(theta_arc))
    R_i     = rot_np(delta_i)
    corners = np.array([[0,0],[a,0],[a,b],[0,b],[0,0]])
    world   = np.array([[xA_anim[i], yA_anim[i]] + R_i @ c for c in corners])
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
        f"t     = {t_i:.2f} s\n"
        f"T_opt = {T_opt:.2f} s\n"
        f"ψ     = {np.degrees(psi_i):.1f}°\n"
        f"δ     = {np.degrees(delta_i):.1f}°\n"
        f"l     = {float(l_sol[i]):.3f} m\n"
        f"τ_clk = {tau_i:.2f} s\n"
        f"fx    = {fx_i:.1f} N\n"
        f"fy    = {fy_i:.1f} N\n"
        f"{phase}"
    )
    return (wall2_line, spring_arc, body_line, A_pt, B_pt, O_pt,
            force_line, traj_line, info_txt)

dt_anim = float(ts[1] - ts[0])
ani = FuncAnimation(fig_anim, update, frames=len(ts),
                    interval=int(1000*dt_anim), blit=False)
plt.tight_layout()
ani.save('ocp_min_time.gif', writer='pillow', fps=int(1 / dt_anim))
plt.show()
