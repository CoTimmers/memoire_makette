"""feedback.py — OCP nominal + open-loop réel + feedforward-feedback réel
   ────────────────────────────────────────────────────────────────────────
   Bloc 1 : Résolution OCP avec paramètres nominaux  →  trajectoire nominale
   Bloc 2 : Simulation open-loop sur système perturbé  (forces OCP appliquées telles quelles)
   Bloc 3 : Simulation feedforward + feedback sur système perturbé
              fx_cmd = fx_nom + Kp_l*(l_nom - l) + Kd_l*(ldot_nom - ldot)
              fy_cmd = fy_nom + Kp_y*(l_nom - l) + Kd_y*(ldot_nom - ldot)
   Bloc 4 : Comparaison des résultats
"""

import numpy as np
import casadi as ca
from rockit import Ocp, MultipleShooting
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# ══════════════════════════════════════════════════════════════════════
# PARAMÈTRES PHYSIQUES NOMINAUX
# ══════════════════════════════════════════════════════════════════════
m      = 7.0
a      = 0.3
b      = 0.4
I_A    = (m / 3.0) * (a**2 + b**2)
mu     = 0.3
T_wall = 6.0
T_sim  = 10.0
F_max  = 30.0

rAB_body = np.array([0.0, b])
rAO_body = np.array([a/2, b/2])

# ══════════════════════════════════════════════════════════════════════
# GAINS DU FEEDBACK  (réglables)
# ══════════════════════════════════════════════════════════════════════
# Correction de position et vitesse sur fx et fy.
# Raisonnement ordre de grandeur :
#   - tau_corr ~ 0.5 s  →  Kp ~ m/tau² ~ 28 N/m
#   - amortissement critique  →  Kd ~ 2*m/tau ~ 28 N·s/m
# On commence avec des gains modérés pour rester dans les saturations.
Kp_l = 30.0    # gain proportionnel sur fx  [N/m]
Kd_l = 15.0    # gain dérivé       sur fx  [N·s/m]
Kp_y = 15.0    # gain proportionnel sur fy  [N/m]
Kd_y =  8.0    # gain dérivé       sur fy  [N·s/m]

# ══════════════════════════════════════════════════════════════════════
# PROFIL DU MUR  ψ : π/2 → π
# ══════════════════════════════════════════════════════════════════════
_t1   = T_wall * 0.4
_t2   = T_wall * 0.6
_dpsi = np.pi / 2
_acc  = _dpsi / (0.5*_t1**2 + (_t2 - _t1)*_t1 + 0.5*(T_wall - _t2)**2)
_v1   = _acc * _t1
_p1   = np.pi/2 + 0.5*_acc*_t1**2
_p2   = _p1 + _v1*(_t2 - _t1)

_t_sw = _t1 + (np.radians(127) - _p1) / _v1   # instant où ψ = 127°, ≈ 2.7 s

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

def wall_profile_np(t):
    if t <= _t1:
        return np.pi/2 + 0.5*_acc*t**2, _acc*t, _acc
    elif t <= _t2:
        return _p1 + _v1*(t - _t1), _v1, 0.0
    elif t <= T_wall:
        return (_p2 + _v1*(t - _t2) - 0.5*_acc*(t - _t2)**2,
                _v1 - _acc*(t - _t2), -_acc)
    else:
        return np.pi, 0.0, 0.0

# ══════════════════════════════════════════════════════════════════════
# RÉFÉRENCE  l_ref(t)
# ══════════════════════════════════════════════════════════════════════
def l_ref_ca(t):
    ramp = b * (t - _t_sw) / (T_wall - _t_sw)
    return ca.if_else(t < _t_sw,
               0.0,
           ca.if_else(t <= T_wall,
               ramp,
               b))

def l_ref_np(t):
    if t < _t_sw:
        return 0.0
    elif t <= T_wall:
        return b * (t - _t_sw) / (T_wall - _t_sw)
    else:
        return b

# ══════════════════════════════════════════════════════════════════════
# DYNAMIQUE SYMBOLIQUE CasADi  (OCP uniquement)
# ══════════════════════════════════════════════════════════════════════
def dynamics_ca(l, ldot, fx, fy, t):
    psi, psidot, psiddot = wall_profile_ca(t)
    eps  = 1e-9
    u    = (l / b) * ca.sin(psi)
    u_c  = ca.fmax(-1 + eps, ca.fmin(1 - eps, u))
    k    = 1.0 / ca.sqrt(1.0 - u_c**2 + eps);  k3 = k**3
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

    nl_ddot = (delta_ll * ldot**2 + 2*delta_lpsi * ldot*psidot
             + delta_psipsi * psidot**2 + delta_psi * psiddot)

    ax_nl = Rp_r[0]*nl_ddot - R_r[0]*delta_dot**2
    ay_nl = Rp_r[1]*nl_ddot - R_r[1]*delta_dot**2

    tB    = ca.vertcat( ca.cos(psi),  ca.sin(psi))
    nB    = ca.vertcat( ca.sin(psi), -ca.cos(psi))
    fBdir = nB - mu * tB

    def cross2(r, f): return r[0]*f[1] - r[1]*f[0]

    M_mat = ca.vertcat(
        ca.horzcat( m * ax_coeff,    mu,  -fBdir[0]             ),
        ca.horzcat( m * ay_coeff,  -1.0,  -fBdir[1]             ),
        ca.horzcat( I_A * delta_l,   0.0, -cross2(rAB_w, fBdir) )
    )
    rhs = ca.vertcat(
        fx - m * ax_nl,
        fy - m * ay_nl,
        cross2(rAO_w, ca.vertcat(fx, fy)) - I_A * nl_ddot
    )
    z = ca.solve(M_mat, rhs)
    return z[0], z[1], z[2]   # lddot, fyA, fBn

# ══════════════════════════════════════════════════════════════════════
# DYNAMIQUE NUMPY PARAMÉTRÉE  (simulations forward)
# ══════════════════════════════════════════════════════════════════════
def dynamics_np(l, ldot, fx, fy, t, m_p, I_p, mu_p):
    """Retourne (lddot, fyA, fBn) avec les paramètres (m_p, I_p, mu_p)."""
    psi, psidot, psiddot = wall_profile_np(t)
    eps = 1e-9
    u   = np.clip((l / b) * np.sin(psi), -1 + eps, 1 - eps)
    k   = 1.0 / np.sqrt(1 - u**2 + eps);  k3 = k**3

    delta = psi - np.pi/2 + np.arcsin(u)
    u_l       = np.sin(psi) / b
    u_psi     = (l / b) * np.cos(psi)
    u_lpsi    = np.cos(psi) / b
    u_psipsi  = -(l / b) * np.sin(psi)

    delta_l      = k * u_l
    delta_psi    = 1.0 + k * u_psi
    delta_ll     = u * k3 * u_l**2
    delta_lpsi   = k * u_lpsi  + u * k3 * u_l  * u_psi
    delta_psipsi = k * u_psipsi + u * k3 * u_psi**2
    delta_dot    = delta_l * ldot + delta_psi * psidot

    nl_ddot = (delta_ll * ldot**2 + 2*delta_lpsi * ldot*psidot
             + delta_psipsi * psidot**2 + delta_psi * psiddot)

    cd, sd = np.cos(delta), np.sin(delta)
    R  = np.array([[ cd, -sd], [ sd,  cd]])
    Rp = np.array([[-sd, -cd], [ cd, -sd]])

    rAO_w = R  @ rAO_body;  rAB_w = R  @ rAB_body
    Rp_r  = Rp @ rAO_body;  R_r   = R  @ rAO_body

    ax_coeff = 1.0 + Rp_r[0]*delta_l;  ay_coeff = Rp_r[1]*delta_l
    ax_nl = Rp_r[0]*nl_ddot - R_r[0]*delta_dot**2
    ay_nl = Rp_r[1]*nl_ddot - R_r[1]*delta_dot**2

    tB    = np.array([ np.cos(psi),  np.sin(psi)])
    nB    = np.array([ np.sin(psi), -np.cos(psi)])
    fBdir = nB - mu_p * tB

    def c2(r, f): return r[0]*f[1] - r[1]*f[0]

    M = np.array([
        [m_p*ax_coeff,  mu_p,  -fBdir[0]          ],
        [m_p*ay_coeff, -1.0,   -fBdir[1]           ],
        [I_p*delta_l,   0.0,   -c2(rAB_w, fBdir)   ]
    ])
    rhs = np.array([
        fx - m_p*ax_nl,
        fy - m_p*ay_nl,
        c2(rAO_w, np.array([fx, fy])) - I_p*nl_ddot
    ])
    try:
        z = np.linalg.solve(M, rhs)
        return z[0], z[1], z[2]
    except np.linalg.LinAlgError:
        return 0.0, 0.0, 0.0


# ══════════════════════════════════════════════════════════════════════
# ██████████████  BLOC 1 : OCP NOMINAL  ███████████████████████████████
# ══════════════════════════════════════════════════════════════════════
print("=" * 60)
print("BLOC 1 : Résolution OCP nominal...")
print("=" * 60)

ocp  = Ocp(T=T_sim)
l    = ocp.state();  ldot = ocp.state()
fx   = ocp.state();  fy   = ocp.state()

dF_max = 20.0
dfx  = ocp.control();  dfy = ocp.control()

t_sym = ocp.t
lddot_sym, fyA_sym, fBn_sym = dynamics_ca(l, ldot, fx, fy, t_sym)

ocp.set_der(l,    ldot)
ocp.set_der(ldot, lddot_sym)
ocp.set_der(fx,   dfx)
ocp.set_der(fy,   dfy)

l_ref_sym = l_ref_ca(t_sym)
ocp.add_objective(ocp.integral((l - l_ref_sym)**2))
ocp.add_objective(1e2 * (ocp.at_tf(l) - b)**2)
ocp.add_objective(1e1 *  ocp.at_tf(ldot)**2)

W_c = 1e2
ocp.add_objective(W_c * ocp.integral(ca.fmax(0.0, -fyA_sym)**2))
ocp.add_objective(W_c * ocp.integral(ca.fmax(0.0, -fBn_sym)**2))

ocp.subject_to(fx >= -F_max);  ocp.subject_to(fx <=  F_max)
ocp.subject_to(fy >= -F_max);  ocp.subject_to(fy <=  F_max)
ocp.subject_to(dfx**2 + dfy**2 <= dF_max**2)
ocp.subject_to(l >= 0);        ocp.subject_to(l <= b)

ocp.subject_to(ocp.at_t0(l)    == 0.0)
ocp.subject_to(ocp.at_t0(ldot) == 0.0)
ocp.subject_to(ocp.at_t0(fx)   == 0.0)
ocp.subject_to(ocp.at_t0(fy)   == 0.0)

ocp.solver('ipopt', {'ipopt.print_level': 5, 'ipopt.max_iter': 1000, 'ipopt.tol': 1e-6})
ocp.method(MultipleShooting(N=100, intg='rk'))

sol = ocp.solve()
print("OCP résolu !\n")

# ── Extraction des trajectoires nominales ────────────────────────────
ts,   l_nom    = sol.sample(l,    grid='control')
_,    ldot_nom = sol.sample(ldot, grid='control')
_,    fx_nom   = sol.sample(fx,   grid='control')
_,    fy_nom   = sol.sample(fy,   grid='control')
_,    dfx_nom  = sol.sample(dfx,  grid='control')
_,    dfy_nom  = sol.sample(dfy,  grid='control')

l_ref_num = np.array([l_ref_np(float(ti)) for ti in ts])

fyA_nom = np.array([
    dynamics_np(float(li), float(ldi), float(fxi), float(fyi), float(ti), m, I_A, mu)[1]
    for li, ldi, fxi, fyi, ti in zip(l_nom, ldot_nom, fx_nom, fy_nom, ts)
])
fBn_nom = np.array([
    dynamics_np(float(li), float(ldi), float(fxi), float(fyi), float(ti), m, I_A, mu)[2]
    for li, ldi, fxi, fyi, ti in zip(l_nom, ldot_nom, fx_nom, fy_nom, ts)
])

print("── Résultats OCP nominal ──────────────────────────────────")
print(f"  l final    = {l_nom[-1]:.4f} m  (cible = {b} m)")
print(f"  Erreur max = {np.max(np.abs(l_nom - l_ref_num)):.4f} m")
print(f"  fx max     = {np.max(np.abs(fx_nom)):.2f} N")
print(f"  fy max     = {np.max(np.abs(fy_nom)):.2f} N")
print(f"  fyA min    = {np.min(fyA_nom):.3f} N  (≥ 0 ?)")
print(f"  fBn min    = {np.min(fBn_nom):.3f} N  (≥ 0 ?)")


# ══════════════════════════════════════════════════════════════════════
# PARAMÈTRES RÉELS PERTURBÉS  (communs aux blocs 2 et 3)
# ══════════════════════════════════════════════════════════════════════
m_real  = 1.1 * m
I_real  = 1.2 * I_A
mu_real = 0.8 * mu

print(f"\nParamètres réels : m={m_real:.2f} kg (+10%), "
      f"I={I_real:.4f} kg·m² (+20%), mu={mu_real:.3f} (-20%)")

# ── Interpolateurs des trajectoires nominales ─────────────────────────
# Utilisés comme feedforward ET comme référence pour le feedback.
kw = dict(kind='linear', bounds_error=False)
fx_nom_interp   = interp1d(ts, fx_nom,   fill_value=(fx_nom[0],   fx_nom[-1]),   **kw)
fy_nom_interp   = interp1d(ts, fy_nom,   fill_value=(fy_nom[0],   fy_nom[-1]),   **kw)
l_nom_interp    = interp1d(ts, l_nom,    fill_value=(l_nom[0],    l_nom[-1]),    **kw)
ldot_nom_interp = interp1d(ts, ldot_nom, fill_value=(ldot_nom[0], ldot_nom[-1]), **kw)

# ── Grille de temps : phase 1 (l=0 imposé) + phase 2 (dynamique libre) ─
idx_sw   = np.searchsorted(ts, _t_sw)
t_phase1 = ts[:idx_sw]
t_phase2 = ts[idx_sw:]


# ══════════════════════════════════════════════════════════════════════
# ██████████  BLOC 2 : OPEN-LOOP RÉEL  ████████████████████████████████
# ══════════════════════════════════════════════════════════════════════
# Forces nominales fx_nom(t), fy_nom(t) appliquées telles quelles
# sur la dynamique perturbée.
print("\n" + "=" * 60)
print("BLOC 2 : Simulation open-loop sur système perturbé...")
print("=" * 60)

def ode_openloop(t_val, state):
    l_v, ldot_v = state
    fx_v = float(fx_nom_interp(t_val))
    fy_v = float(fy_nom_interp(t_val))
    lddot_v, _, _ = dynamics_np(float(l_v), float(ldot_v),
                                 fx_v, fy_v, t_val, m_real, I_real, mu_real)
    return [ldot_v, float(lddot_v)]

sol_ol = solve_ivp(ode_openloop,
                   t_span=(float(t_phase2[0]), float(t_phase2[-1])),
                   y0=[0.0, 0.0],
                   method='RK45', t_eval=t_phase2,
                   rtol=1e-6, atol=1e-8)
if not sol_ol.success:
    print(f"  ATTENTION intégrateur OL : {sol_ol.message}")

t_ol    = np.concatenate([t_phase1, sol_ol.t])
l_ol    = np.concatenate([np.zeros(len(t_phase1)), sol_ol.y[0]])
ldot_ol = np.concatenate([np.zeros(len(t_phase1)), sol_ol.y[1]])

# Forces appliquées (open-loop = nominales sur toute la grille)
fx_ol_applied = np.array([float(fx_nom_interp(ti)) for ti in t_ol])
fy_ol_applied = np.array([float(fy_nom_interp(ti)) for ti in t_ol])

err_ol = np.abs(l_ol - np.array([l_ref_np(float(ti)) for ti in t_ol]))
print(f"  l final         = {l_ol[-1]:.4f} m")
print(f"  Erreur max OL   = {np.max(err_ol):.4f} m")
print(f"  Erreur finale OL= {abs(l_ol[-1] - b):.4f} m")


# ══════════════════════════════════════════════════════════════════════
# ██████████  BLOC 3 : FEEDFORWARD + FEEDBACK RÉEL  ███████████████████
# ══════════════════════════════════════════════════════════════════════
# La commande est corrigée à chaque instant par l'erreur entre
# l'état réel simulé et la trajectoire nominale.
#
#   fx_cmd = fx_nom(t) + Kp_l*(l_nom(t) - l(t)) + Kd_l*(ldot_nom(t) - ldot(t))
#   fy_cmd = fy_nom(t) + Kp_y*(l_nom(t) - l(t)) + Kd_y*(ldot_nom(t) - ldot(t))
#   avec saturation dans [-F_max, F_max].
print("\n" + "=" * 60)
print("BLOC 3 : Simulation feedforward + feedback sur système perturbé...")
print(f"  Gains : Kp_l={Kp_l}, Kd_l={Kd_l}, Kp_y={Kp_y}, Kd_y={Kd_y}")
print("=" * 60)

def compute_fb_command(t_val, l_v, ldot_v):
    """Calcule et sature la commande feedforward + feedback."""
    ff_x = float(fx_nom_interp(t_val))
    ff_y = float(fy_nom_interp(t_val))
    e_l    = float(l_nom_interp(t_val))    - l_v
    e_ldot = float(ldot_nom_interp(t_val)) - ldot_v
    fx_cmd = np.clip(ff_x + Kp_l*e_l + Kd_l*e_ldot, -F_max, F_max)
    fy_cmd = np.clip(ff_y + Kp_y*e_l + Kd_y*e_ldot, -F_max, F_max)
    return fx_cmd, fy_cmd

def ode_feedback_ctrl(t_val, state):
    l_v, ldot_v = state
    fx_cmd, fy_cmd = compute_fb_command(t_val, float(l_v), float(ldot_v))
    lddot_v, _, _ = dynamics_np(float(l_v), float(ldot_v),
                                 fx_cmd, fy_cmd, t_val, m_real, I_real, mu_real)
    return [ldot_v, float(lddot_v)]

sol_fb = solve_ivp(ode_feedback_ctrl,
                   t_span=(float(t_phase2[0]), float(t_phase2[-1])),
                   y0=[0.0, 0.0],
                   method='RK45', t_eval=t_phase2,
                   rtol=1e-6, atol=1e-8)
if not sol_fb.success:
    print(f"  ATTENTION intégrateur FB : {sol_fb.message}")

t_fb    = np.concatenate([t_phase1, sol_fb.t])
l_fb    = np.concatenate([np.zeros(len(t_phase1)), sol_fb.y[0]])
ldot_fb = np.concatenate([np.zeros(len(t_phase1)), sol_fb.y[1]])

# Reconstruction des commandes réellement appliquées (post-processing)
fx_fb_applied = np.array([
    compute_fb_command(float(ti), float(li), float(ldi))[0]
    for ti, li, ldi in zip(t_fb, l_fb, ldot_fb)
])
fy_fb_applied = np.array([
    compute_fb_command(float(ti), float(li), float(ldi))[1]
    for ti, li, ldi in zip(t_fb, l_fb, ldot_fb)
])

err_fb = np.abs(l_fb - np.array([l_ref_np(float(ti)) for ti in t_fb]))
print(f"  l final          = {l_fb[-1]:.4f} m")
print(f"  Erreur max FB    = {np.max(err_fb):.4f} m")
print(f"  Erreur finale FB = {abs(l_fb[-1] - b):.4f} m")


# ══════════════════════════════════════════════════════════════════════
# ██████████████  BLOC 4 : COMPARAISON  ███████████████████████████████
# ══════════════════════════════════════════════════════════════════════

# ── Figure 1 : trajectoires l(t) ─────────────────────────────────────
fig1, ax = plt.subplots(figsize=(9, 5))
ax.plot(ts,   l_nom,      color='steelblue',   lw=2.2,        label='l_nom(t)  — OCP nominal')
ax.plot(t_ol, l_ol,       color='crimson',     lw=2.0, ls='--', label='l_OL(t)   — open-loop réel')
ax.plot(t_fb, l_fb,       color='seagreen',    lw=2.0, ls='-.',  label='l_FB(t)   — feedforward+feedback réel')
ax.plot(ts,   l_ref_num,  color='darkorange',  lw=1.4, ls=':',  label='l_ref(t)')
ax.axhline(b, color='gray', lw=1, ls=':', alpha=0.7, label=f'cible b = {b} m')
ax.axvline(_t_sw, color='gray', lw=1, ls=':', alpha=0.5)
ax.text(_t_sw + 0.05, 0.01, f't_sw={_t_sw:.1f}s', fontsize=8, color='gray')
ax.set_title('Comparaison trajectoires : nominal, open-loop réel, feedforward+feedback réel')
ax.set_ylabel('l (m)');  ax.set_xlabel('t (s)')
ax.legend(fontsize=9);   ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fb_trajectories.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Figure 2 : erreurs de suivi ───────────────────────────────────────
err_nom_track = np.abs(l_nom - l_ref_num)
fig2, ax = plt.subplots(figsize=(9, 4))
ax.plot(ts,   err_nom_track, color='steelblue',  lw=1.8,       label='erreur nominale (OCP)')
ax.plot(t_ol, err_ol,        color='crimson',    lw=1.8, ls='--', label='erreur open-loop réel')
ax.plot(t_fb, err_fb,        color='seagreen',   lw=1.8, ls='-.', label='erreur feedforward+feedback réel')
ax.set_title('Erreur de suivi |l(t) − l_ref(t)|')
ax.set_ylabel('|e| (m)');  ax.set_xlabel('t (s)')
ax.legend(fontsize=9);     ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fb_tracking_error.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Figure 3 : commandes fx ───────────────────────────────────────────
fig3, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

axes[0].plot(ts,   fx_nom,        color='steelblue',  lw=1.8,       label='fx_nom (OCP)')
axes[0].plot(t_ol, fx_ol_applied, color='crimson',    lw=1.6, ls='--', label='fx_OL (= fx_nom)')
axes[0].plot(t_fb, fx_fb_applied, color='seagreen',   lw=1.6, ls='-.', label='fx_FB (corrigé)')
axes[0].axhline( F_max, color='gray', lw=1, ls=':', alpha=0.6, label=f'±{F_max} N')
axes[0].axhline(-F_max, color='gray', lw=1, ls=':', alpha=0.6)
axes[0].set_title('Commande fx(t)');  axes[0].set_ylabel('N')
axes[0].legend(fontsize=8);           axes[0].grid(True, alpha=0.3)

axes[1].plot(ts,   fy_nom,        color='steelblue',  lw=1.8,       label='fy_nom (OCP)')
axes[1].plot(t_ol, fy_ol_applied, color='crimson',    lw=1.6, ls='--', label='fy_OL (= fy_nom)')
axes[1].plot(t_fb, fy_fb_applied, color='seagreen',   lw=1.6, ls='-.', label='fy_FB (corrigé)')
axes[1].axhline( F_max, color='gray', lw=1, ls=':', alpha=0.6)
axes[1].axhline(-F_max, color='gray', lw=1, ls=':', alpha=0.6)
axes[1].set_title('Commande fy(t)');  axes[1].set_ylabel('N')
axes[1].set_xlabel('t (s)')
axes[1].legend(fontsize=8);           axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fb_commands.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Figure 4 : correction apportée par le feedback ───────────────────
dfx_fb = fx_fb_applied - np.array([float(fx_nom_interp(ti)) for ti in t_fb])
dfy_fb = fy_fb_applied - np.array([float(fy_nom_interp(ti)) for ti in t_fb])
fig4, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
axes[0].plot(t_fb, dfx_fb, color='seagreen', lw=1.8)
axes[0].axhline(0, color='gray', lw=1)
axes[0].set_title('Correction feedback Δfx = fx_FB − fx_nom');  axes[0].set_ylabel('N')
axes[0].grid(True, alpha=0.3)
axes[1].plot(t_fb, dfy_fb, color='mediumpurple', lw=1.8)
axes[1].axhline(0, color='gray', lw=1)
axes[1].set_title('Correction feedback Δfy = fy_FB − fy_nom');  axes[1].set_ylabel('N')
axes[1].set_xlabel('t (s)');  axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fb_correction.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Résumé console ────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("RÉSUMÉ COMPARATIF")
print("=" * 65)
print(f"{'':34s}  {'Nominal':>8s}  {'OL réel':>8s}  {'FB réel':>8s}")
print("-" * 65)

def fmt(v): return f"{v:>8.4f}"

l_ref_final = l_ref_np(float(ts[-1]))
print(f"  {'l final (m)':32s}  {fmt(l_nom[-1])}  {fmt(l_ol[-1])}  {fmt(l_fb[-1])}")
print(f"  {'Erreur finale |l - b| (m)':32s}  {fmt(abs(l_nom[-1]-b))}  {fmt(abs(l_ol[-1]-b))}  {fmt(abs(l_fb[-1]-b))}")
print(f"  {'Erreur max |l - l_ref| (m)':32s}  {fmt(np.max(err_nom_track))}  {fmt(np.max(err_ol))}  {fmt(np.max(err_fb))}")

def ok(v): return "OUI" if v >= b - 1e-3 else "NON"
print(f"  {'Cible b atteinte':32s}  {'OUI':>8s}  {ok(l_ol[-1]):>8s}  {ok(l_fb[-1]):>8s}")
print(f"\n  Gains utilisés : Kp_l={Kp_l}, Kd_l={Kd_l}, Kp_y={Kp_y}, Kd_y={Kd_y}")
print("=" * 65)
