"""close_loop.py — OCP nominal + simulation de robustesse open-loop
   ─────────────────────────────────────────────────────────────────
   Partie 1 : Résolution OCP avec paramètres nominaux (identique à
              moteur_ocp_rockit.py).
   Partie 2 : Simulation forward avec paramètres perturbés.
              Les forces fx(t), fy(t) issues de l'OCP nominal sont
              appliquées telles quelles à un système dont les paramètres
              physiques sont différents (m_real, I_real, mu_real).
              Objectif : évaluer la sensibilité du contrôle open-loop
              aux erreurs de modélisation.
"""

import numpy as np
import casadi as ca
from rockit import Ocp, MultipleShooting
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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


def wall_profile_np(t):
    """Version numpy du profil de mur."""
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
# DYNAMIQUE SYMBOLIQUE  →  (l̈, fyA, fBn)   [paramètres nominaux]
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
# ══════════════════════════════════════════════════════════════════════
_t_sw = _t1 + (np.radians(127) - _p1) / _v1   # ≈ 2.7 s

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
# DYNAMIQUE NUMPY PARAMÉTRÉE  (réutilisée pour nominal et perturbé)
# ══════════════════════════════════════════════════════════════════════
def dynamics_np(l, ldot, fx, fy, t, m_p, I_p, mu_p):
    """
    Calcule lddot, fyA, fBn avec les paramètres physiques (m_p, I_p, mu_p).
    Utilisée aussi bien pour le post-traitement nominal que pour la
    simulation forward perturbée.
    """
    psi, psidot, psiddot = wall_profile_np(t)

    eps = 1e-9
    u   = np.clip((l / b) * np.sin(psi), -1 + eps, 1 - eps)
    k   = 1.0 / np.sqrt(1 - u**2 + eps)
    k3  = k**3

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

    delta_dot = delta_l * ldot + delta_psi * psidot

    nl_ddot = (delta_ll      * ldot**2
             + 2*delta_lpsi  * ldot * psidot
             + delta_psipsi  * psidot**2
             + delta_psi     * psiddot)

    cd, sd = np.cos(delta), np.sin(delta)
    R  = np.array([[ cd, -sd], [ sd,  cd]])
    Rp = np.array([[-sd, -cd], [ cd, -sd]])

    rAO_w = R  @ rAO_body
    rAB_w = R  @ rAB_body
    Rp_r  = Rp @ rAO_body
    R_r   = R  @ rAO_body

    ax_coeff = 1.0 + Rp_r[0] * delta_l
    ay_coeff =       Rp_r[1] * delta_l
    ax_nl = Rp_r[0] * nl_ddot - R_r[0] * delta_dot**2
    ay_nl = Rp_r[1] * nl_ddot - R_r[1] * delta_dot**2

    tB    = np.array([ np.cos(psi),  np.sin(psi)])
    nB    = np.array([ np.sin(psi), -np.cos(psi)])
    fBdir = nB - mu_p * tB

    def c2(r, f): return r[0]*f[1] - r[1]*f[0]

    M = np.array([
        [m_p * ax_coeff,   mu_p,  -fBdir[0]             ],
        [m_p * ay_coeff,  -1.0,   -fBdir[1]             ],
        [I_p * delta_l,    0.0,   -c2(rAB_w, fBdir)     ]
    ])
    rhs = np.array([
        fx - m_p * ax_nl,
        fy - m_p * ay_nl,
        c2(rAO_w, np.array([fx, fy])) - I_p * nl_ddot
    ])
    try:
        z = np.linalg.solve(M, rhs)
        return z[0], z[1], z[2]   # lddot, fyA, fBn
    except np.linalg.LinAlgError:
        return 0.0, 0.0, 0.0


# ══════════════════════════════════════════════════════════════════════
# ██████████████████████   PARTIE 1 : OCP NOMINAL   ██████████████████
# ══════════════════════════════════════════════════════════════════════
ocp  = Ocp(T=T_sim)

l    = ocp.state()
ldot = ocp.state()
fx   = ocp.state()
fy   = ocp.state()

dF_max = 20.0
dfx  = ocp.control()
dfy  = ocp.control()

t = ocp.t

lddot, fyA, fBn = dynamics_ca(l, ldot, fx, fy, t)

ocp.set_der(l,    ldot)
ocp.set_der(ldot, lddot)
ocp.set_der(fx,   dfx)
ocp.set_der(fy,   dfy)

l_ref = l_ref_ca(t)
ocp.add_objective(ocp.integral((l - l_ref)**2))
ocp.add_objective(1e2 * (ocp.at_tf(l) - b)**2)
ocp.add_objective(1e1 *  ocp.at_tf(ldot)**2)

W_c = 1e2
ocp.add_objective(W_c * ocp.integral(ca.fmax(0.0, -fyA)**2))
ocp.add_objective(W_c * ocp.integral(ca.fmax(0.0, -fBn)**2))

ocp.subject_to(fx >= -F_max);  ocp.subject_to(fx <=  F_max)
ocp.subject_to(fy >= -F_max);  ocp.subject_to(fy <=  F_max)
ocp.subject_to(dfx**2 + dfy**2 <= dF_max**2)
ocp.subject_to(l >= 0)

ocp.subject_to(ocp.at_t0(l)    == 0.0)
ocp.subject_to(ocp.at_t0(ldot) == 0.0)
ocp.subject_to(ocp.at_t0(fx)   == 0.0)
ocp.subject_to(ocp.at_t0(fy)   == 0.0)

ocp.solver('ipopt', {
    'ipopt.print_level'  : 5,
    'ipopt.max_iter'     : 1000,
    'ipopt.tol'          : 1e-6,
})
ocp.method(MultipleShooting(N=100, intg='rk'))

print("=" * 60)
print("PARTIE 1 : Résolution OCP nominal...")
print("=" * 60)
sol = ocp.solve()
print("OCP résolu !\n")

# ── Extraction des trajectoires optimales ────────────────────────────
ts,   l_nom    = sol.sample(l,    grid='control')
_,    ldot_nom = sol.sample(ldot, grid='control')
_,    fx_sol   = sol.sample(fx,   grid='control')
_,    fy_sol   = sol.sample(fy,   grid='control')
_,    dfx_sol  = sol.sample(dfx,  grid='control')
_,    dfy_sol  = sol.sample(dfy,  grid='control')

l_ref_num = np.array([l_ref_np(float(ti)) for ti in ts])

# ── Post-traitement forces de contact nominales ───────────────────────
fyA_nom = np.array([
    dynamics_np(float(li), float(ldi), float(fxi), float(fyi), float(ti),
                m, I_A, mu)[1]
    for li, ldi, fxi, fyi, ti in zip(l_nom, ldot_nom, fx_sol, fy_sol, ts)
])
fBn_nom = np.array([
    dynamics_np(float(li), float(ldi), float(fxi), float(fyi), float(ti),
                m, I_A, mu)[2]
    for li, ldi, fxi, fyi, ti in zip(l_nom, ldot_nom, fx_sol, fy_sol, ts)
])

# ── Résultats OCP nominal ─────────────────────────────────────────────
dF_norm = np.sqrt(dfx_sol**2 + dfy_sol**2)
print("── Résultats OCP nominal ──────────────────────────────────")
print(f"  l final       = {l_nom[-1]:.4f} m  (cible = {b} m)")
print(f"  ḷ final       = {ldot_nom[-1]:.4f} m/s")
print(f"  Erreur max    = {np.max(np.abs(l_nom - l_ref_num)):.4f} m")
print(f"  fx max        = {np.max(np.abs(fx_sol)):.2f} N")
print(f"  fy max        = {np.max(np.abs(fy_sol)):.2f} N")
print(f"  ‖df‖ max      = {np.max(dF_norm):.2f} N/s  (limite = {dF_max} N/s)")
print(f"  fyA min       = {np.min(fyA_nom):.3f} N  (doit être ≥ 0)")
print(f"  fBn min       = {np.min(fBn_nom):.3f} N  (doit être ≥ 0)")
idx = np.argmax(l_nom >= b - 1e-3)
if l_nom[idx] >= b - 1e-3:
    print(f"  l atteint b={b} m à t = {ts[idx]:.4f} s")
else:
    print(f"  l n'atteint jamais b (l_max = {np.max(l_nom):.4f} m)")


# ══════════════════════════════════════════════════════════════════════
# ██████████████   PARTIE 2 : SIMULATION PERTURBÉE   ██████████████████
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PARTIE 2 : Simulation forward avec paramètres perturbés...")
print("=" * 60)

# ── Paramètres réels (perturbés) ──────────────────────────────────────
m_real  = 1.1 * m
I_real  = 0.5 * I_A
mu_real = 0.8 * mu

print(f"  m_real  = {m_real:.3f} kg   (nominal = {m:.3f}, +10%)")
print(f"  I_real  = {I_real:.4f} kg·m²  (nominal = {I_A:.4f}, +20%)")
print(f"  mu_real = {mu_real:.3f}      (nominal = {mu:.3f}, -20%)\n")

# ── Interpolateurs des forces open-loop ──────────────────────────────
#    Les forces sont maintenues constantes au-delà de la grille (fill_value).
fx_interp = interp1d(ts, fx_sol, kind='linear',
                     bounds_error=False, fill_value=(fx_sol[0], fx_sol[-1]))
fy_interp = interp1d(ts, fy_sol, kind='linear',
                     bounds_error=False, fill_value=(fy_sol[0], fy_sol[-1]))

# ── ODE  ẋ = f(t, x)  avec x = [l, ldot] ────────────────────────────
def ode_perturbed(t_val, state):
    l_v, ldot_v = state
    fx_v = float(fx_interp(t_val))
    fy_v = float(fy_interp(t_val))
    lddot_v, _, _ = dynamics_np(float(l_v), float(ldot_v), fx_v, fy_v,
                                 t_val, m_real, I_real, mu_real)
    return [ldot_v, float(lddot_v)]

# ── Intégration en deux phases ────────────────────────────────────────
# Phase 1 (t < t_sw) : contrainte physique l = 0
#   La géométrie empêche A de glisser tant que ψ < 127°.
#   On maintient l=0, ldot=0 et on démarre l'intégration libre à t_sw.
# Phase 2 (t >= t_sw) : dynamique libre avec paramètres perturbés.
idx_sw   = np.searchsorted(ts, _t_sw)   # premier indice après t_sw
t_phase1 = ts[:idx_sw]
t_phase2 = ts[idx_sw:]

sol_pert = solve_ivp(
    ode_perturbed,
    t_span=(float(t_phase2[0]), float(t_phase2[-1])),
    y0=[0.0, 0.0],           # l=0, ldot=0 à t_sw
    method='RK45',
    t_eval=t_phase2,
    rtol=1e-6,
    atol=1e-8,
    dense_output=False
)

if not sol_pert.success:
    print(f"  ATTENTION : intégrateur a échoué — {sol_pert.message}")

# Recombinaison des deux phases
t_real    = np.concatenate([t_phase1, sol_pert.t])
l_real    = np.concatenate([np.zeros(len(t_phase1)), sol_pert.y[0]])
ldot_real = np.concatenate([np.zeros(len(t_phase1)), sol_pert.y[1]])
l_ref_real = np.array([l_ref_np(float(ti)) for ti in t_real])

# ── Forces de contact pour le système perturbé ───────────────────────
fyA_real = np.array([
    dynamics_np(float(li), float(ldi),
                float(fx_interp(ti)), float(fy_interp(ti)),
                float(ti), m_real, I_real, mu_real)[1]
    for li, ldi, ti in zip(l_real, ldot_real, t_real)
])
fBn_real = np.array([
    dynamics_np(float(li), float(ldi),
                float(fx_interp(ti)), float(fy_interp(ti)),
                float(ti), m_real, I_real, mu_real)[2]
    for li, ldi, ti in zip(l_real, ldot_real, t_real)
])

# ── Résultats simulation perturbée ────────────────────────────────────
err_real = np.abs(l_real - l_ref_real)
print("── Résultats simulation perturbée ─────────────────────────")
print(f"  l final (perturbé)  = {l_real[-1]:.4f} m  (cible = {b} m)")
print(f"  ḷ final (perturbé)  = {ldot_real[-1]:.4f} m/s")
print(f"  Erreur max tracking = {np.max(err_real):.4f} m")
print(f"  Erreur finale       = {np.abs(l_real[-1] - b):.4f} m")
if l_real[-1] >= b - 1e-3:
    idx_r = np.argmax(l_real >= b - 1e-3)
    print(f"  Cible b={b} m atteinte à t = {t_real[idx_r]:.4f} s  ✓")
else:
    print(f"  Cible b={b} m NON atteinte (l_max = {np.max(l_real):.4f} m)  ✗")
print(f"  fyA min (perturbé)  = {np.min(fyA_real):.3f} N")
print(f"  fBn min (perturbé)  = {np.min(fBn_real):.3f} N")


# ══════════════════════════════════════════════════════════════════════
# PLOTS  — comparaison nominale vs perturbée
# ══════════════════════════════════════════════════════════════════════

# ── 1. l(t) : nominal, perturbé, référence ───────────────────────────
fig1, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(ts,     l_nom,      color='steelblue',  lw=2.0,  label='l_nom(t)  — modèle nominal')
ax.plot(t_real, l_real,     color='crimson',    lw=2.0,  ls='--', label='l_real(t) — modèle perturbé')
ax.plot(ts,     l_ref_num,  color='darkorange', lw=1.4,  ls=':',  label='l_ref(t)')
ax.axhline(b, color='seagreen', lw=1, ls=':', alpha=0.8, label=f'cible b = {b} m')
ax.axvline(_t_sw, color='gray', lw=1, ls=':', alpha=0.6)
ax.text(_t_sw + 0.05, 0.01, f't_sw={_t_sw:.1f}s', fontsize=8, color='gray')
ax.set_title('Comparaison trajectoires : nominal vs perturbé')
ax.set_ylabel('l (m)');  ax.set_xlabel('t (s)')
ax.legend(fontsize=9);   ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('robustness_l_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# ── 2. Erreur de suivi ───────────────────────────────────────────────
err_nom  = np.abs(l_nom  - l_ref_num)
fig2, ax = plt.subplots(figsize=(8, 4))
ax.plot(ts,     err_nom,  color='steelblue', lw=1.8, label='erreur nominale')
ax.plot(t_real, err_real, color='crimson',   lw=1.8, ls='--', label='erreur perturbée')
ax.set_title('Erreur de suivi |l(t) − l_ref(t)|')
ax.set_ylabel('|e| (m)');  ax.set_xlabel('t (s)')
ax.legend(fontsize=9);     ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('robustness_tracking_error.png', dpi=150, bbox_inches='tight')
plt.show()

# ── 3. Forces de commande appliquées (identiques dans les deux cas) ───
fig3, ax = plt.subplots(figsize=(8, 4))
ax.plot(ts, fx_sol, color='steelblue', lw=1.8, label='fx (open-loop)')
ax.plot(ts, fy_sol, color='crimson',   lw=1.8, label='fy (open-loop)')
ax.axhline(0, color='gray', lw=1)
ax.set_title('Forces open-loop appliquées (nominales et réutilisées)')
ax.set_ylabel('N');  ax.set_xlabel('t (s)')
ax.legend(fontsize=9);  ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('robustness_forces.png', dpi=150, bbox_inches='tight')
plt.show()

# ── 4. Forces de contact : nominal vs perturbé ───────────────────────
fig4, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

axes[0].plot(ts,     fyA_nom,  color='steelblue', lw=1.8, label='fyA nominal')
axes[0].plot(t_real, fyA_real, color='crimson',   lw=1.8, ls='--', label='fyA perturbé')
axes[0].axhline(0, color='gray', lw=1, ls='--', alpha=0.6)
axes[0].set_title('Force de contact fyA (mur 1)')
axes[0].set_ylabel('N');  axes[0].legend(fontsize=9);  axes[0].grid(True, alpha=0.3)

axes[1].plot(ts,     fBn_nom,  color='steelblue', lw=1.8, label='fBn nominal')
axes[1].plot(t_real, fBn_real, color='crimson',   lw=1.8, ls='--', label='fBn perturbé')
axes[1].axhline(0, color='gray', lw=1, ls='--', alpha=0.6)
axes[1].set_title('Force de contact fBn (mur 2)')
axes[1].set_ylabel('N');  axes[1].set_xlabel('t (s)')
axes[1].legend(fontsize=9);  axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('robustness_contact_forces.png', dpi=150, bbox_inches='tight')
plt.show()

# ── 5. l(t) et ḷ(t) nominaux (plots individuels, inchangés) ──────────
fig5, ax = plt.subplots(figsize=(7, 4))
ax.plot(ts, l_nom,    color='steelblue',  lw=1.8, label='l(t) OCP')
ax.plot(ts, l_ref_num,color='darkorange', lw=1.4, ls='--', label='l_ref')
ax.axhline(b, color='seagreen', lw=1, ls=':', alpha=0.8, label=f'target b={b} m')
ax.set_title('l(t) — reference tracking (nominal)')
ax.set_ylabel('l (m)');  ax.set_xlabel('t (s)')
ax.legend(fontsize=9);   ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('l(t).png', dpi=150, bbox_inches='tight')
plt.show()

fig6, ax = plt.subplots(figsize=(7, 4))
ax.plot(ts, ldot_nom, color='mediumpurple', lw=1.8)
ax.axhline(0, color='gray', lw=1)
ax.set_title('ḷ(t) — nominal')
ax.set_ylabel('ḷ (m/s)');  ax.set_xlabel('t (s)')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ldot(t).png', dpi=150, bbox_inches='tight')
plt.show()

fig7, ax = plt.subplots(figsize=(7, 4))
ax.plot(ts, fx_sol, color='steelblue', lw=1.8, label='fx')
ax.plot(ts, fy_sol, color='crimson',   lw=1.8, label='fy')
ax.axhline(0, color='gray', lw=1)
ax.set_title('Control forces [fx, fy]')
ax.set_ylabel('N');  ax.set_xlabel('t (s)')
ax.legend(fontsize=9);  ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Command_fx,_fy.png', dpi=150, bbox_inches='tight')
plt.show()

# ── 6. Résumé comparatif final ────────────────────────────────────────
print("\n" + "=" * 60)
print("RÉSUMÉ COMPARATIF")
print("=" * 60)
print(f"{'':30s}  {'Nominal':>12s}  {'Perturbé':>12s}")
print("-" * 60)
print(f"  {'l final (m)':28s}  {l_nom[-1]:>12.4f}  {l_real[-1]:>12.4f}")
print(f"  {'ḷ final (m/s)':28s}  {ldot_nom[-1]:>12.4f}  {ldot_real[-1]:>12.4f}")
print(f"  {'Erreur max |l-l_ref| (m)':28s}  {np.max(err_nom):>12.4f}  {np.max(err_real):>12.4f}")
print(f"  {'Erreur finale |l-b| (m)':28s}  {abs(l_nom[-1]-b):>12.4f}  {abs(l_real[-1]-b):>12.4f}")
print(f"  {'fyA min (N)':28s}  {np.min(fyA_nom):>12.3f}  {np.min(fyA_real):>12.3f}")
print(f"  {'fBn min (N)':28s}  {np.min(fBn_nom):>12.3f}  {np.min(fBn_real):>12.3f}")
nom_ok  = "OUI" if l_nom[-1]  >= b - 1e-3 else "NON"
real_ok = "OUI" if l_real[-1] >= b - 1e-3 else "NON"
print(f"  {'Cible b atteinte':28s}  {nom_ok:>12s}  {real_ok:>12s}")
print("=" * 60)


# ══════════════════════════════════════════════════════════════════════
# ANIMATION  (basée sur la trajectoire nominale OCP)
# ══════════════════════════════════════════════════════════════════════
def delta_kinematics_np(l_v, ldot_v, psi, psidot):
    eps = 1e-9
    u = np.clip((l_v / b) * np.sin(psi), -1 + eps, 1 - eps)
    k = 1.0 / np.sqrt(max(eps, 1 - u*u))
    delta     = psi - np.pi/2 + np.arcsin(u)
    delta_l   = k * np.sin(psi) / b
    delta_psi = 1 + k * (l_v / b) * np.cos(psi)
    delta_dot = delta_l * ldot_v + delta_psi * psidot
    return delta, delta_dot

psi_anim    = np.array([wall_profile_np(float(t))[0]  for t in ts])
psidot_anim = np.array([wall_profile_np(float(t))[1]  for t in ts])
delta_anim  = np.array([
    delta_kinematics_np(float(li), float(ldi), float(pi), float(pdi))[0]
    for li, ldi, pi, pdi in zip(l_nom, ldot_nom, psi_anim, psidot_anim)
])

xA_anim = l_nom
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

fig_anim, ax2 = plt.subplots(figsize=(8, 7))
ax2.set_xlim(-0.65, 0.65);  ax2.set_ylim(-0.1, 0.65)
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)
ax2.set_xlabel('x (m)');     ax2.set_ylabel('y (m)')
ax2.set_title('OCP animation — optimal control (nominal)')

wall1_line, = ax2.plot([0, 0.6], [0, 0],  color='steelblue',  lw=3,  label='wall 1')
wall2_line, = ax2.plot([], [],             color='darkorange', lw=3,  label='wall 2')
body_line,  = ax2.plot([], [],             color='dimgray',    lw=2,  label='body')
A_pt,       = ax2.plot([], [], 'o',        color='crimson',    ms=7,  label='A')
B_pt,       = ax2.plot([], [], 'o',        color='seagreen',   ms=7,  label='B')
O_pt,       = ax2.plot([], [], 'o',        color='darkorange', ms=7,  label='O (COM)')
force_line, = ax2.plot([], [],             color='mediumpurple', lw=2, label='crane force')
lref_line,  = ax2.plot([], [], 's',        color='darkorange', ms=5,  alpha=0.5, label='l_ref')
traj_line,  = ax2.plot([], [],             color='darkorange', lw=1,  alpha=0.4)

info_txt = ax2.text(0.02, 0.97, '', transform=ax2.transAxes,
                    color='black', fontsize=9, va='top', fontfamily='monospace',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='lightgray'))
ax2.legend(loc='upper right', fontsize=8)

traj_x, traj_y = [], []

def update(i):
    psi_i   = psi_anim[i]
    delta_i = delta_anim[i]
    l_i     = float(l_nom[i])
    t_i     = float(ts[i])
    fx_i    = float(fx_sol[i])
    fy_i    = float(fy_sol[i])
    lref_i  = float(l_ref_num[i])

    L2 = 0.55
    wall2_line.set_data([0, L2*np.cos(psi_i)], [0, L2*np.sin(psi_i)])

    R = rot_np(delta_i)
    corners = np.array([[0,0],[a,0],[a,b],[0,b],[0,0]])
    world   = np.array([np.array([xA_anim[i], yA_anim[i]]) + R @ c for c in corners])
    body_line.set_data(world[:, 0], world[:, 1])

    A_pt.set_data([xA_anim[i]], [yA_anim[i]])
    B_pt.set_data([xB_anim[i]], [yB_anim[i]])
    O_pt.set_data([xO_anim[i]], [yO_anim[i]])
    lref_line.set_data([lref_i], [0.0])

    scale = 0.02
    force_line.set_data([xO_anim[i], xO_anim[i] + fx_i*scale],
                        [yO_anim[i], yO_anim[i] + fy_i*scale])

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
ani = FuncAnimation(fig_anim, update, frames=len(ts),
                    interval=int(1000 * dt_anim), blit=False)

plt.tight_layout()
print("\nSauvegarde du GIF nominal...")
ani.save('animation.gif', writer='pillow', fps=int(1.0 / dt_anim))
print("GIF sauvegardé : animation.gif")
plt.show()


# ══════════════════════════════════════════════════════════════════════
# ANIMATION PERTURBÉE — côte à côte nominal vs perturbé
# ══════════════════════════════════════════════════════════════════════

# ── Géométrie du système perturbé ────────────────────────────────────
delta_real_anim = np.array([
    delta_kinematics_np(float(li), float(ldi), float(pi), float(pdi))[0]
    for li, ldi, pi, pdi in zip(l_real, ldot_real, psi_anim, psidot_anim)
])

xA_r = l_real
yA_r = np.zeros(len(t_real))
xO_r = np.zeros(len(t_real))
yO_r = np.zeros(len(t_real))
xB_r = np.zeros(len(t_real))
yB_r = np.zeros(len(t_real))
for i in range(len(t_real)):
    R_r = rot_np(delta_real_anim[i])
    rAO_r = R_r @ rAO_body
    rAB_r = R_r @ rAB_body
    xO_r[i] = xA_r[i] + rAO_r[0];  yO_r[i] = yA_r[i] + rAO_r[1]
    xB_r[i] = xA_r[i] + rAB_r[0];  yB_r[i] = yA_r[i] + rAB_r[1]

# ── Figure côte à côte ────────────────────────────────────────────────
fig_comp, (axL, axR) = plt.subplots(1, 2, figsize=(15, 7))

for ax_i, title_i in [(axL, 'Nominal  (m, I, μ)'),
                       (axR, f'Perturbé  (+10% m, +20% I, -20% μ)')]:
    ax_i.set_xlim(-0.65, 0.65);  ax_i.set_ylim(-0.1, 0.65)
    ax_i.set_aspect('equal')
    ax_i.grid(True, alpha=0.3)
    ax_i.set_xlabel('x (m)');    ax_i.set_ylabel('y (m)')
    ax_i.set_title(title_i)
    ax_i.plot([0, 0.6], [0, 0], color='steelblue', lw=3, label='wall 1')

# ── Éléments animés — côté nominal ───────────────────────────────────
w2L,  = axL.plot([], [], color='darkorange', lw=3,  label='wall 2')
bdL,  = axL.plot([], [], color='dimgray',    lw=2,  label='body')
aL,   = axL.plot([], [], 'o', color='crimson',   ms=7,  label='A')
bpL,  = axL.plot([], [], 'o', color='seagreen',  ms=7,  label='B')
oL,   = axL.plot([], [], 'o', color='darkorange',ms=7,  label='O')
fL,   = axL.plot([], [], color='mediumpurple', lw=2, label='force')
lrL,  = axL.plot([], [], 's', color='darkorange', ms=5, alpha=0.5, label='l_ref')
trL,  = axL.plot([], [], color='steelblue', lw=1, alpha=0.4)
txL   = axL.text(0.02, 0.97, '', transform=axL.transAxes, color='black',
                 fontsize=8, va='top', fontfamily='monospace',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='lightgray'))

# ── Éléments animés — côté perturbé ──────────────────────────────────
w2R,  = axR.plot([], [], color='darkorange', lw=3,  label='wall 2')
bdR,  = axR.plot([], [], color='crimson',    lw=2,  label='body (perturbé)')
aR,   = axR.plot([], [], 'o', color='crimson',   ms=7,  label='A')
bpR,  = axR.plot([], [], 'o', color='seagreen',  ms=7,  label='B')
oR,   = axR.plot([], [], 'o', color='darkorange',ms=7,  label='O')
fR,   = axR.plot([], [], color='mediumpurple', lw=2, label='force')
lrR,  = axR.plot([], [], 's', color='darkorange', ms=5, alpha=0.5, label='l_ref')
trR,  = axR.plot([], [], color='crimson', lw=1, alpha=0.4)
txR   = axR.text(0.02, 0.97, '', transform=axR.transAxes, color='black',
                 fontsize=8, va='top', fontfamily='monospace',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='lightgray'))

for ax_i in (axL, axR):
    ax_i.legend(loc='upper right', fontsize=7)

traj_xL, traj_yL = [], []
traj_xR, traj_yR = [], []

def update_comp(i):
    psi_i  = psi_anim[i]
    L2     = 0.55
    scale  = 0.02
    lref_i = float(l_ref_num[i])

    # ── Mur 2 (commun) ──────────────────────────────────────────────
    wx = L2 * np.cos(psi_i);  wy = L2 * np.sin(psi_i)
    w2L.set_data([0, wx], [0, wy])
    w2R.set_data([0, wx], [0, wy])

    # ── Côté nominal ─────────────────────────────────────────────────
    R_n    = rot_np(delta_anim[i])
    corners = np.array([[0,0],[a,0],[a,b],[0,b],[0,0]])
    wn = np.array([np.array([xA_anim[i], yA_anim[i]]) + R_n @ c for c in corners])
    bdL.set_data(wn[:, 0], wn[:, 1])
    aL.set_data([xA_anim[i]], [yA_anim[i]])
    bpL.set_data([xB_anim[i]], [yB_anim[i]])
    oL.set_data([xO_anim[i]], [yO_anim[i]])
    fx_i = float(fx_sol[i]);  fy_i = float(fy_sol[i])
    fL.set_data([xO_anim[i], xO_anim[i] + fx_i*scale],
                [yO_anim[i], yO_anim[i] + fy_i*scale])
    lrL.set_data([lref_i], [0.0])
    traj_xL.append(xO_anim[i]);  traj_yL.append(yO_anim[i])
    trL.set_data(traj_xL, traj_yL)
    txL.set_text(
        f"t    = {float(ts[i]):.2f} s\n"
        f"ψ    = {np.degrees(psi_i):.1f}°\n"
        f"δ    = {np.degrees(delta_anim[i]):.1f}°\n"
        f"l    = {float(l_nom[i]):.3f} m\n"
        f"lref = {lref_i:.3f} m\n"
        f"fx   = {fx_i:.1f} N\n"
        f"fy   = {fy_i:.1f} N"
    )

    # ── Côté perturbé ─────────────────────────────────────────────────
    R_p    = rot_np(delta_real_anim[i])
    wp = np.array([np.array([xA_r[i], yA_r[i]]) + R_p @ c for c in corners])
    bdR.set_data(wp[:, 0], wp[:, 1])
    aR.set_data([xA_r[i]], [yA_r[i]])
    bpR.set_data([xB_r[i]], [yB_r[i]])
    oR.set_data([xO_r[i]], [yO_r[i]])
    fR.set_data([xO_r[i], xO_r[i] + fx_i*scale],
                [yO_r[i], yO_r[i] + fy_i*scale])
    lrR.set_data([lref_i], [0.0])
    traj_xR.append(xO_r[i]);  traj_yR.append(yO_r[i])
    trR.set_data(traj_xR, traj_yR)
    txR.set_text(
        f"t    = {float(t_real[i]):.2f} s\n"
        f"ψ    = {np.degrees(psi_i):.1f}°\n"
        f"δ    = {np.degrees(delta_real_anim[i]):.1f}°\n"
        f"l    = {float(l_real[i]):.3f} m\n"
        f"lref = {lref_i:.3f} m\n"
        f"fx   = {fx_i:.1f} N\n"
        f"fy   = {fy_i:.1f} N"
    )

    return (w2L, bdL, aL, bpL, oL, fL, lrL, trL, txL,
            w2R, bdR, aR, bpR, oR, fR, lrR, trR, txR)

ani_comp = FuncAnimation(fig_comp, update_comp, frames=len(ts),
                         interval=int(1000 * dt_anim), blit=False)

plt.tight_layout()
print("\nSauvegarde du GIF comparatif nominal vs perturbé...")
ani_comp.save('animation_comparison.gif', writer='pillow', fps=int(1.0 / dt_anim))
print("GIF sauvegardé : animation_comparison.gif")
plt.show()
