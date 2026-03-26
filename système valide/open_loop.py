"""open_loop_1.py — Simulation boucle ouverte : système hybride à 4 modes
   ────────────────────────────────────────────────────────────────────────
   Architecture hybride explicite :

   MODE 1  (cinématique pure, pas d'ODE)
     A fixe, B contraint sur mur 2.
     delta = psi - pi/2,  omega = psi_dot.
     Fin : psi atteint psi_sw_1.

   MODE 2a  (1 DDL : l)
     A glisse sur mur 1 (y_A = 0), B maintenu sur mur 2.
     État ODE : [l, l_dot].  delta calculé par cinématique de contrainte.
     Transition → MODE_2B : lambda_B < 0 (séparation de B du mur 2).
     Transition → MODE_3  : l atteint l_contact (impact de B sur mur 1).

   MODE 2b  (2 DDL : l, delta)
     A glisse sur mur 1, B libre (non-pénétration unilatérale sur mur 2).
     État ODE : [l, delta, l_dot, omega].
     Transition → MODE_3  : l atteint l_contact (impact de B sur mur 1).

   MODE 3  (1 DDL : l, commande PD)
     A et B sur mur 1.  delta = pi/2 par contrainte géométrique.
     État ODE : [l, l_dot].
     Commande PD : fx = -kp*(l - l*) - kd*l_dot,  l* = 0.4 m.

   Conventions :
     phi >= 0  →  admissible ;   phi < 0  →  pénétration interdite.
     Les réactions de contact ne sont PAS calculées explicitement.
     Les contraintes sont traitées par réduction de coordonnées.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
from scipy.optimize import brentq as _brentq

# ══════════════════════════════════════════════════════════════════════
# PARAMÈTRES PHYSIQUES
# ══════════════════════════════════════════════════════════════════════
params = {
    'm': 7.0,
    'a': 0.3,
    'b': 0.4,
    'T_wall': 6.0,
    'psi_sw_1': np.radians(127),   # ψ déclenchant la transition 1 → 2a
    'l_contact': 0.4,              # l déclenchant l'impact 2 → 3  (= b)
}
m = params['m']
a = params['a']
b = params['b']

params['I_O']      = m / 12.0 * (a**2 + b**2)
params['I_A']      = m / 3.0  * (a**2 + b**2)
params['rAO_body'] = np.array([a/2, b/2])
params['rAB_body'] = np.array([0.0, b])

T_SIM = 10.0

# ══════════════════════════════════════════════════════════════════════
# psi_profile(t) — profil du mur pivotant
# ══════════════════════════════════════════════════════════════════════
_T  = params['T_wall']
_t1 = _T * 0.4
_t2 = _T * 0.6
_acc = (np.pi/2) / (0.5*_t1**2 + (_t2 - _t1)*_t1 + 0.5*(_T - _t2)**2)
_v1  = _acc * _t1
_p1  = np.pi/2 + 0.5*_acc*_t1**2
_p2  = _p1 + _v1*(_t2 - _t1)

def psi_profile(t):
    """Retourne (psi, psi_dot, psi_ddot) à l'instant t."""
    if t <= _t1:
        return np.pi/2 + 0.5*_acc*t**2, _acc*t, _acc
    elif t <= _t2:
        return _p1 + _v1*(t - _t1), _v1, 0.0
    elif t <= _T:
        return (_p2 + _v1*(t - _t2) - 0.5*_acc*(t - _t2)**2,
                _v1 - _acc*(t - _t2), -_acc)
    else:
        return np.pi, 0.0, 0.0

# ══════════════════════════════════════════════════════════════════════
# CINÉMATIQUE DE δ (contrainte B sur mur 2 — utilisée en mode 2a)
# ══════════════════════════════════════════════════════════════════════
def delta_kinematics(l, ldot, psi, psidot, psiddot):
    """Retourne (delta, delta_dot, delta_l, delta_psi,
                 delta_ll, delta_lpsi, delta_psipsi)."""
    eps = 1e-9
    u   = np.clip((l / b) * np.sin(psi), -1 + eps, 1 - eps)
    k   = 1.0 / np.sqrt(max(eps, 1 - u*u))
    k3  = k**3

    delta        = psi - 0.5*np.pi + np.arcsin(u)
    u_l          = np.sin(psi) / b
    u_psi        = (l/b) * np.cos(psi)
    u_lpsi       = np.cos(psi) / b
    u_psipsi     = -(l/b) * np.sin(psi)

    delta_l      = k  * u_l
    delta_psi    = 1  + k  * u_psi
    delta_ll     = u  * k3 * u_l    * u_l
    delta_lpsi   = k  * u_lpsi + u  * k3 * u_l  * u_psi
    delta_psipsi = k  * u_psipsi + u * k3 * u_psi * u_psi
    delta_dot    = delta_l * ldot + delta_psi * psidot

    return delta, delta_dot, delta_l, delta_psi, delta_ll, delta_lpsi, delta_psipsi

# ══════════════════════════════════════════════════════════════════════
# COMMANDES PAR MODE
# ══════════════════════════════════════════════════════════════════════
def input_mode1(t):
    """Force maintenant B sur mur 2 (perpendiculaire au mur)."""
    psi, _, _ = psi_profile(t)
    return -5.0 * np.sin(psi), 5.0 * np.cos(psi)

input_phase1 = input_mode1

def input_mode2(t):
    """Force nominale pour faire glisser A le long du mur 1."""
    return 5.0 * 0.2, 5.0 * (-1.0)

input_phase2 = input_mode2

# Mode 3 — commande PD vers la cible l* = 0.4 m
_l_star = 0.4
_kp     = 20.0   # N/m
_kd     = 15.0   # N·s/m

def input_mode3(t, x):
    """Feedback PD : fx = -kp*(l - l*) - kd*l_dot."""
    l, l_dot = x
    fx = -_kp * (l - _l_star) - _kd * l_dot
    return fx, 0.0

# ══════════════════════════════════════════════════════════════════════
# DYNAMIQUES PAR MODE
# ══════════════════════════════════════════════════════════════════════
def dynamics_phase1(t, x, params):
    """Référence — non intégrée (cinématique directe en mode 1)."""
    delta, omega = x
    I_A  = params['I_A']
    a, b = params['a'], params['b']
    fx, fy = input_mode1(t)
    r_AO_x = (a/2)*np.cos(delta) - (b/2)*np.sin(delta)
    r_AO_y = (a/2)*np.sin(delta) + (b/2)*np.cos(delta)
    return [omega, (r_AO_x * fy - r_AO_y * fx) / I_A]

def _phase2a_contact_core(t, x, params):
    """Noyau mode 2a : calcule (l_ddot, lambda_B, delta, delta_dot)."""
    l, l_dot = x
    m_p    = params['m']
    I_A    = params['I_A']
    a, b_p = params['a'], params['b']
    psi, psi_dot, psi_ddot = psi_profile(t)
    fx, fy = input_mode2(t)
    delta, delta_dot, delta_l, delta_psi, delta_ll, delta_lpsi, delta_psipsi = \
        delta_kinematics(l, l_dot, psi, psi_dot, psi_ddot)
    c1 = -(a/2)*np.sin(delta) - (b_p/2)*np.cos(delta)
    c2 =  (a/2)*np.cos(delta) - (b_p/2)*np.sin(delta)
    c3 = -(a/2)*np.cos(delta) + (b_p/2)*np.sin(delta)
    A_kin = (delta_ll*l_dot**2
             + 2*delta_lpsi*l_dot*psi_dot
             + delta_psipsi*psi_dot**2
             + delta_psi*psi_ddot)
    denom  = m_p * (1.0 + c1 * delta_l)
    l_ddot = ((fx - m_p*(c1*A_kin + c3*delta_dot**2)) / denom
              if abs(denom) > 1e-12 else 0.0)
    delta_ddot = delta_l * l_ddot + A_kin
    sin_dp     = np.sin(delta - psi)
    denom_l    = b_p * sin_dp
    lambda_B   = ((m_p*c1*l_ddot + I_A*delta_ddot - c1*fx - c2*fy) / denom_l
                  if abs(denom_l) > 1e-10 else 0.0)
    return l_ddot, lambda_B, delta, delta_dot

def dynamics_mode2a(t, x, params):
    """Mode 2a : A/mur1, B/mur2 (1 DDL). État : [l, l_dot]."""
    l_dot = x[1]
    l_ddot, _, _, _ = _phase2a_contact_core(t, x, params)
    return [l_dot, l_ddot]

def dynamics_mode2b(t, x, params):
    """Mode 2b : A/mur1, B libre (2 DDL). État : [l, delta, l_dot, omega]."""
    l, delta, l_dot, omega = x
    m_p  = params['m']
    I_A  = params['I_A']
    a, b = params['a'], params['b']
    fx, fy = input_mode2(t)
    c1 = -(a/2)*np.sin(delta) - (b/2)*np.cos(delta)
    c2 =  (a/2)*np.cos(delta) - (b/2)*np.sin(delta)
    c3 = -(a/2)*np.cos(delta) + (b/2)*np.sin(delta)
    M_sys = np.array([[m_p,     m_p*c1],
                      [m_p*c1,  I_A   ]])
    rhs   = np.array([fx - m_p*c3*omega**2,
                      c1*fx + c2*fy])
    try:
        sol = np.linalg.solve(M_sys, rhs)
    except np.linalg.LinAlgError:
        sol = np.zeros(2)
    l_ddot, delta_ddot = sol
    return [l_dot, omega, l_ddot, delta_ddot]

def dynamics_mode3(t, x, params):
    """Mode 3 : delta=pi/2, A et B sur mur 1, PD. État : [l, l_dot]."""
    l, l_dot = x
    m_p = params['m']
    fx, _ = input_mode3(t, x)
    return [l_dot, fx / m_p]

# ══════════════════════════════════════════════════════════════════════
# ÉVÉNEMENTS DE TRANSITION
# ══════════════════════════════════════════════════════════════════════
def event_B_separation(t, x, params):
    """lambda_B passe de + à - : B se sépare du mur 2 (mode 2a → 2b)."""
    _, lambda_B, _, _ = _phase2a_contact_core(t, x, params)
    return lambda_B

event_B_separation.terminal  = True
event_B_separation.direction = -1

def event_phase2_to_3(t, x, params):
    """l atteint l_contact : impact de B sur mur 1 (mode 2 → 3)."""
    return x[0] - params['l_contact']

event_phase2_to_3.terminal  = True
event_phase2_to_3.direction = 1

# ══════════════════════════════════════════════════════════════════════
# GAP FUNCTIONS (admissibilité géométrique)
# ══════════════════════════════════════════════════════════════════════
def phi_B_phase1(delta, psi, params):
    """Distance signée B/mur2 — mode 1."""
    b = params['b']
    return -b * np.cos(delta - psi)

def phi_B_phase2(l, delta, psi, params):
    """Distance signée B/mur2 — modes 2a et 2b."""
    b = params['b']
    return l * np.sin(psi) - b * np.cos(delta - psi)

def event_B_penetration_phase2(t, x, params):
    """Surveillance non-terminale : phi_B < 0 en mode 2b."""
    l, delta, _, _ = x
    psi, _, _ = psi_profile(t)
    return phi_B_phase2(l, delta, psi, params)

event_B_penetration_phase2.terminal  = False
event_B_penetration_phase2.direction = -1

# ══════════════════════════════════════════════════════════════════════
# LOI D'IMPACT : mode 2 → mode 3
# Conservation du momentum x du COM (pas de réaction impulsive en x).
# ══════════════════════════════════════════════════════════════════════
def impact_phase2_to_3(x_minus, params):
    """x_minus = [l, delta, l_dot, omega]  →  x_plus = [l, l_dot]."""
    l_minus, delta_minus, l_dot_minus, omega_minus = x_minus
    a   = params['a']
    b_p = params['b']
    c1  = -(a/2)*np.sin(delta_minus) - (b_p/2)*np.cos(delta_minus)
    l_dot_plus = l_dot_minus + c1 * omega_minus
    return np.array([l_minus, l_dot_plus])

# ══════════════════════════════════════════════════════════════════════
# CONSTANTES DE MODE + HELPER ÉVÉNEMENT
# ══════════════════════════════════════════════════════════════════════
MODE_1  = 1   # A fixe,   B/mur2         (cinématique pure)
MODE_2A = 2   # A/mur1,   B/mur2         (1 DDL : l)
MODE_2B = 3   # A/mur1,   B libre        (2 DDL : l, δ)
MODE_3  = 4   # A/mur1,   B/mur1, δ=π/2 (1 DDL : l, PD)

def _make_ev(func, terminal, direction):
    """Wrapper d'événement préservant terminal/direction pour solve_ivp."""
    def ev(t, x):
        return func(t, x, params)
    ev.terminal  = terminal
    ev.direction = direction
    return ev

# ══════════════════════════════════════════════════════════════════════
# SIMULATION
# ══════════════════════════════════════════════════════════════════════

# ── MODE 1 : cinématique directe ──────────────────────────────────────
print("=" * 60)
print("MODE 1 : A fixe, B sur mur 2  →  cinématique directe")
print("=" * 60)

def _psi_minus_target(t):
    return psi_profile(t)[0] - params['psi_sw_1']

_t_lo, _t_hi = 0.0, params['T_wall']
if _psi_minus_target(_t_lo) * _psi_minus_target(_t_hi) < 0:
    t1_end = _brentq(_psi_minus_target, _t_lo, _t_hi, xtol=1e-10)
else:
    t1_end = params['T_wall']

t1_arr  = np.linspace(0.0, t1_end, 300)
l1      = np.zeros_like(t1_arr)
ldot1   = np.zeros_like(t1_arr)
psi1    = np.array([psi_profile(ti)[0] for ti in t1_arr])
psidot1 = np.array([psi_profile(ti)[1] for ti in t1_arr])
delta1  = psi1 - np.pi / 2
omega1  = psidot1
fx1     = np.array([input_mode1(ti)[0] for ti in t1_arr])
fy1     = np.array([input_mode1(ti)[1] for ti in t1_arr])

print(f"  Fin mode 1 : t = {t1_end:.4f} s")
print(f"  δ(t_sw)    = {np.degrees(delta1[-1]):.3f}°  (= ψ - 90°)")
print(f"  ψ(t_sw)    = {np.degrees(psi1[-1]):.3f}°   (cible {np.degrees(params['psi_sw_1']):.1f}°)")

# ── BOUCLE HYBRIDE : modes 2a / 2b / 3 ───────────────────────────────
print("\n" + "=" * 60)
print("BOUCLE HYBRIDE : modes 2a / 2b / 3")
print("=" * 60)

_t_segs, _l_segs, _ldot_segs  = [], [], []
_delta_segs, _omega_segs       = [], []
_psi_segs, _fx_segs, _fy_segs  = [], [], []
_mode_segs                     = []

t_sep    = None   # séparation B/mur2  (2a → 2b)
t_impact = None   # impact B/mur1      (2a ou 2b → 3)

# CI mode 2a : l=0, l_dot=0
mode  = MODE_2A
t_cur = t1_end
x_cur = np.array([0.0, 0.0])   # [l, l_dot]

for _seg in range(20):
    if t_cur >= T_SIM:
        break

    # ── Dynamique + événements ──────────────────────────────────────
    if mode == MODE_2A:
        fun = lambda t, x: dynamics_mode2a(t, x, params)
        events = [
            _make_ev(event_phase2_to_3,  True,  +1),   # → MODE_3
            _make_ev(event_B_separation, True,  -1),   # → MODE_2B
        ]

    elif mode == MODE_2B:
        fun = lambda t, x: dynamics_mode2b(t, x, params)
        events = [
            _make_ev(event_phase2_to_3,          True,  +1),   # → MODE_3
            _make_ev(event_B_penetration_phase2, False, -1),   # surveillance
        ]

    elif mode == MODE_3:
        fun = lambda t, x: dynamics_mode3(t, x, params)
        events = []

    else:
        break

    sol = solve_ivp(fun, (t_cur, T_SIM), x_cur,
                    method='RK45', events=events,
                    dense_output=True, rtol=1e-9, atol=1e-11)

    t_end = float(sol.t[-1])
    n_pts = max(3, int(400 * (t_end - t_cur) / T_SIM))
    t_seg = np.linspace(t_cur, t_end, n_pts)
    y_seg = sol.sol(t_seg)

    # ── Reconstruction des variables du segment ─────────────────────
    if mode == MODE_2A:
        l_seg    = y_seg[0]
        ldot_seg = y_seg[1]
        psi_seg  = np.array([psi_profile(ti)[0] for ti in t_seg])
        d_seg    = np.zeros(n_pts)
        w_seg    = np.zeros(n_pts)
        for i in range(n_pts):
            psi_i, psid_i, psidd_i = psi_profile(t_seg[i])
            dk = delta_kinematics(l_seg[i], ldot_seg[i], psi_i, psid_i, psidd_i)
            d_seg[i] = dk[0]
            w_seg[i] = dk[1]
        fx_seg = np.array([input_mode2(ti)[0] for ti in t_seg])
        fy_seg = np.array([input_mode2(ti)[1] for ti in t_seg])

    elif mode == MODE_2B:
        l_seg    = y_seg[0]
        d_seg    = y_seg[1]
        ldot_seg = y_seg[2]
        w_seg    = y_seg[3]
        psi_seg  = np.array([psi_profile(ti)[0] for ti in t_seg])
        fx_seg   = np.array([input_mode2(ti)[0] for ti in t_seg])
        fy_seg   = np.array([input_mode2(ti)[1] for ti in t_seg])
        if len(sol.t_events[1]) > 0:
            print(f"  ⚠ phi_B < 0 en mode 2b à t = {sol.t_events[1]} s")

    elif mode == MODE_3:
        l_seg    = y_seg[0]
        ldot_seg = y_seg[1]
        d_seg    = np.full(n_pts, np.pi / 2)
        w_seg    = np.zeros(n_pts)
        psi_seg  = np.array([psi_profile(ti)[0] for ti in t_seg])
        fx_seg   = np.array([-_kp*(l_seg[i] - _l_star) - _kd*ldot_seg[i]
                              for i in range(n_pts)])
        fy_seg   = np.zeros(n_pts)

    _t_segs.append(t_seg);     _l_segs.append(l_seg);     _ldot_segs.append(ldot_seg)
    _delta_segs.append(d_seg); _omega_segs.append(w_seg); _psi_segs.append(psi_seg)
    _fx_segs.append(fx_seg);   _fy_segs.append(fy_seg)
    _mode_segs.append(np.full(n_pts, mode, dtype=int))

    # ── Logique de transition ────────────────────────────────────────
    if mode == MODE_2A:
        impact_fired = len(sol.t_events[0]) > 0
        sep_fired    = len(sol.t_events[1]) > 0

        if impact_fired:
            t_impact = t_end
            x_minus  = np.array([l_seg[-1], d_seg[-1], ldot_seg[-1], w_seg[-1]])
            x_cur    = impact_phase2_to_3(x_minus, params)
            t_cur    = t_end
            mode     = MODE_3
            print(f"  MODE_2A → MODE_3  (impact B/mur1) à t = {t_cur:.4f} s")
            print(f"    δ avant   = {np.degrees(d_seg[-1]):.2f}°  (attendu 90°)")
            print(f"    l_dot+    = {float(x_cur[1]):.4f} m/s")

        elif sep_fired:
            t_sep = t_end
            x_cur = np.array([l_seg[-1], d_seg[-1], ldot_seg[-1], w_seg[-1]])
            t_cur = t_end
            mode  = MODE_2B
            print(f"  MODE_2A → MODE_2B (séparation B/mur2) à t = {t_cur:.4f} s")

        else:
            print(f"  MODE_2A : fin sans transition (l_max = {l_seg.max():.4f} m)")
            t_cur = T_SIM

    elif mode == MODE_2B:
        impact_fired = len(sol.t_events[0]) > 0

        if impact_fired:
            t_impact = t_end
            x_minus  = np.array([l_seg[-1], d_seg[-1], ldot_seg[-1], w_seg[-1]])
            x_cur    = impact_phase2_to_3(x_minus, params)
            t_cur    = t_end
            mode     = MODE_3
            print(f"  MODE_2B → MODE_3  (impact B/mur1) à t = {t_cur:.4f} s")
            print(f"    δ avant   = {np.degrees(d_seg[-1]):.2f}°  (attendu ≈ 90°)")
            print(f"    l_dot+    = {float(x_cur[1]):.4f} m/s")
        else:
            print(f"  MODE_2B : fin sans impact (l_max = {l_seg.max():.4f} m)")
            t_cur = T_SIM

    elif mode == MODE_3:
        t_cur = T_SIM

# ── Concaténation des segments hybrides ──────────────────────────────
t2_arr   = np.concatenate(_t_segs)
l2       = np.concatenate(_l_segs)
ldot2    = np.concatenate(_ldot_segs)
delta2   = np.concatenate(_delta_segs)
omega2   = np.concatenate(_omega_segs)
psi2     = np.concatenate(_psi_segs)
fx2      = np.concatenate(_fx_segs)
fy2      = np.concatenate(_fy_segs)
mode2    = np.concatenate(_mode_segs)

p2_ended = (t_impact is not None)
t2_end   = t_impact if p2_ended else t2_arr[-1]

print(f"\n── Résumé transitions ─────────────────────────────────")
if t_sep    is not None: print(f"  Séparation B/mur2  : t = {t_sep:.4f} s")
if t_impact is not None: print(f"  Impact B/mur1      : t = {t_impact:.4f} s")
if not p2_ended:         print("  Pas d'impact avant T_sim")

# ══════════════════════════════════════════════════════════════════════
# CONCATÉNATION GLOBALE + POSITIONS GÉOMÉTRIQUES
# ══════════════════════════════════════════════════════════════════════
t_all     = np.concatenate([t1_arr, t2_arr])
l_all     = np.concatenate([l1,     l2])
ldot_all  = np.concatenate([ldot1,  ldot2])
delta_all = np.concatenate([delta1, delta2])
omega_all = np.concatenate([omega1, omega2])
psi_all   = np.concatenate([psi1,   psi2])
fx_all    = np.concatenate([fx1,    fx2])
fy_all    = np.concatenate([fy1,    fy2])
phase_all = np.concatenate([
    np.full(len(t1_arr), MODE_1, dtype=int),
    mode2
])

rAO_body_p = params['rAO_body']
rAB_body_p = params['rAB_body']

xA_all = l_all
yA_all = np.zeros_like(l_all)
xO_all = np.zeros_like(l_all)
yO_all = np.zeros_like(l_all)
xB_all = np.zeros_like(l_all)
yB_all = np.zeros_like(l_all)

for i in range(len(t_all)):
    cd, sd = np.cos(delta_all[i]), np.sin(delta_all[i])
    R_i    = np.array([[cd, -sd], [sd, cd]])
    rAO_w  = R_i @ rAO_body_p
    rAB_w  = R_i @ rAB_body_p
    xO_all[i] = xA_all[i] + rAO_w[0]
    yO_all[i] = yA_all[i] + rAO_w[1]
    xB_all[i] = xA_all[i] + rAB_w[0]
    yB_all[i] = yA_all[i] + rAB_w[1]

# Gap function φ_B / mur 2 (modes 1, 2a, 2b)
phi_B_all = np.full(len(t_all), np.nan)
for i in range(len(t_all)):
    ph_i  = phase_all[i]
    psi_i = psi_all[i]
    if ph_i == MODE_1:
        phi_B_all[i] = phi_B_phase1(delta_all[i], psi_i, params)
    elif ph_i in (MODE_2A, MODE_2B):
        phi_B_all[i] = phi_B_phase2(l_all[i], delta_all[i], psi_i, params)

# Vérification contraintes mode 3 : y_A = 0, y_B = 0
yA3_all = np.full(len(t_all), np.nan)
yB3_all = np.full(len(t_all), np.nan)
for i in range(len(t_all)):
    if phase_all[i] == MODE_3:
        yA3_all[i] = 0.0
        yB3_all[i] = params['b'] * np.cos(delta_all[i])

print(f"\n── Résumé final ────────────────────────────────────────")
print(f"  l final     = {l_all[-1]:.4f} m"
      f"   (l★ = {_l_star} m,  l_contact = {params['l_contact']} m)")
print(f"  l_dot final = {ldot_all[-1]:.4f} m/s")
print(f"  δ final     = {np.degrees(delta_all[-1]):.2f}°")
print(f"  ψ final     = {np.degrees(psi_all[-1]):.2f}°")

# ══════════════════════════════════════════════════════════════════════
# GRAPHIQUES
# ══════════════════════════════════════════════════════════════════════
PH_COLOR = {
    MODE_1:  'steelblue',
    MODE_2A: 'seagreen',
    MODE_2B: 'darkolivegreen',
    MODE_3:  'darkorange',
}
PH_LABEL = {
    MODE_1:  'Mode 1 — A fixe, B/mur2',
    MODE_2A: 'Mode 2a — A/mur1, B/mur2 (1 DDL)',
    MODE_2B: 'Mode 2b — A/mur1, B libre (2 DDL)',
    MODE_3:  f'Mode 3 — A,B/mur1, δ=π/2, PD (l★={_l_star} m)',
}

def add_vlines(ax):
    ax.axvline(t1_end, color='gray', lw=1, ls=':', alpha=0.7,
               label=f't_sw1 = {t1_end:.2f} s')
    if t_sep is not None:
        ax.axvline(t_sep, color='seagreen', lw=1, ls=':', alpha=0.7,
                   label=f't_sep = {t_sep:.2f} s')
    if t_impact is not None:
        ax.axvline(t_impact, color='crimson', lw=1, ls=':', alpha=0.7,
                   label=f't_impact = {t_impact:.2f} s')

def plot_phases(t, y, ylabel, title, fname, hlines=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    for ph in [MODE_1, MODE_2A, MODE_2B, MODE_3]:
        mask = phase_all == ph
        if mask.any():
            ax.plot(t[mask], y[mask], color=PH_COLOR[ph], lw=2, label=PH_LABEL[ph])
    if hlines:
        for val, lbl, col in hlines:
            ax.axhline(val, color=col, lw=1.2, ls='--', label=lbl)
    add_vlines(ax)
    ax.axhline(0, color='gray', lw=0.7)
    ax.set_title(title)
    ax.set_xlabel('t (s)')
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()

plot_phases(
    t_all, l_all, 'l (m)',
    'l(t) — position de A sur mur 1',
    'ol1_l.png',
    hlines=[
        (params['l_contact'], f"l_contact = {params['l_contact']} m", 'seagreen'),
        (_l_star,             f"l★ = {_l_star} m",                    'darkorange'),
    ]
)

plot_phases(
    t_all, ldot_all, 'ḷ (m/s)',
    'ḷ(t) — vitesse de A',
    'ol1_ldot.png'
)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t_all, np.degrees(psi_all),   color='darkorange', lw=2, label='ψ(t) — mur 2')
ax.plot(t_all, np.degrees(delta_all), color='steelblue',  lw=2, label='δ(t) — bac')
add_vlines(ax)
ax.axhline(0, color='gray', lw=0.7)
ax.set_title('Angles ψ(t) et δ(t)')
ax.set_xlabel('t (s)')
ax.set_ylabel('deg')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ol1_angles.png', dpi=150, bbox_inches='tight')
plt.show()

plot_phases(
    t_all, np.degrees(omega_all), 'δ̇ (deg/s)',
    'δ̇(t) — vitesse angulaire du bac',
    'ol1_deltadot.png'
)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t_all, fx_all, color='steelblue', lw=2, label='fx')
ax.plot(t_all, fy_all, color='crimson',   lw=2, label='fy')
add_vlines(ax)
ax.axhline(0, color='gray', lw=0.7)
ax.set_title('Commandes fx(t) et fy(t)')
ax.set_xlabel('t (s)')
ax.set_ylabel('N')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ol1_commands.png', dpi=150, bbox_inches='tight')
plt.show()

# φ_B(t) sur mur 2 (modes 1, 2a, 2b)
fig, ax = plt.subplots(figsize=(8, 4))
valid = ~np.isnan(phi_B_all)
ax.plot(t_all[valid], phi_B_all[valid], color='steelblue', lw=2,
        label='φ_B(t) — mur 2')
ax.axhline(0, color='crimson', lw=1.5, ls='--', label='φ_B = 0 (contact)')
mask_pen = phi_B_all < 0
if mask_pen.any():
    ax.fill_between(t_all, phi_B_all, 0, where=mask_pen,
                    color='red', alpha=0.3, label='pénétration (non physique)')
add_vlines(ax)
ax.set_title('Distance signée φ_B(t) — contact B / mur 2\n(modes 1, 2a, 2b)')
ax.set_xlabel('t (s)')
ax.set_ylabel('φ_B (m)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ol1_phi_B_wall2.png', dpi=150, bbox_inches='tight')
plt.show()

# Vérification contraintes mode 3 : y_A = 0, y_B = 0
fig, ax = plt.subplots(figsize=(8, 4))
validA = ~np.isnan(yA3_all)
validB = ~np.isnan(yB3_all)
ax.plot(t_all[validA], yA3_all[validA], lw=2, label='y_A (mode 3)')
ax.plot(t_all[validB], yB3_all[validB], lw=2, label='y_B (mode 3)')
ax.axhline(0, color='crimson', lw=1.5, ls='--', label='0')
add_vlines(ax)
ax.set_title('Mode 3 — vérification des contraintes y_A = 0 et y_B = 0')
ax.set_xlabel('t (s)')
ax.set_ylabel('coordonnée y (m)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ol1_y_constraints_phase3.png', dpi=150, bbox_inches='tight')
plt.show()

# ══════════════════════════════════════════════════════════════════════
# ANIMATION
# ══════════════════════════════════════════════════════════════════════
skip   = max(1, len(t_all) // 250)
frames = list(range(0, len(t_all), skip))

fig_a, ax_a = plt.subplots(figsize=(8, 7))
ax_a.set_xlim(-0.60, 0.65)
ax_a.set_ylim(-0.15, 0.65)
ax_a.set_aspect('equal')
ax_a.grid(True, alpha=0.3)
ax_a.set_xlabel('x (m)')
ax_a.set_ylabel('y (m)')
ax_a.set_title('Open-loop simulation — modes 1 / 2a / 2b / 3')
ax_a.plot([0, 0.6], [0, 0], color='black', lw=3, zorder=3, label='mur 1')
ax_a.axvline(_l_star, color='darkorange', lw=1.5, ls='--', alpha=0.6,
             label=f'l★ = {_l_star} m')

wall2_l, = ax_a.plot([], [], color='dimgray',     lw=3)
body_l,  = ax_a.plot([], [], color='steelblue',   lw=2,   label='bac')
A_pt,    = ax_a.plot([], [], 'o', color='crimson',    ms=8, zorder=4, label='A')
B_pt,    = ax_a.plot([], [], 's', color='seagreen',   ms=8, zorder=4, label='B')
O_pt,    = ax_a.plot([], [], '^', color='darkorange', ms=8, zorder=4, label='O (COM)')
force_l, = ax_a.plot([], [], color='mediumpurple', lw=2.5, label='force F')
traj_l,  = ax_a.plot([], [], color='steelblue',   lw=1,   alpha=0.35)
info_txt = ax_a.text(
    0.02, 0.97, '', transform=ax_a.transAxes,
    color='black', fontsize=9, va='top', fontfamily='monospace',
    bbox=dict(facecolor='white', alpha=0.75, edgecolor='lightgray')
)
ax_a.legend(loc='upper right', fontsize=8)

a_p = params['a']
b_p = params['b']
traj_x, traj_y = [], []

def update_anim(idx):
    i       = idx
    psi_i   = psi_all[i]
    delta_i = delta_all[i]
    ph_i    = phase_all[i]
    L2      = 0.55

    wall2_l.set_data([0, L2*np.cos(psi_i)], [0, L2*np.sin(psi_i)])

    cd, sd  = np.cos(delta_i), np.sin(delta_i)
    R_i     = np.array([[cd, -sd], [sd, cd]])
    corners = np.array([[0, 0], [a_p, 0], [a_p, b_p], [0, b_p], [0, 0]])
    world   = np.array([np.array([xA_all[i], yA_all[i]]) + R_i @ c for c in corners])

    body_l.set_data(world[:, 0], world[:, 1])
    body_l.set_color(PH_COLOR[ph_i])

    A_pt.set_data([xA_all[i]], [yA_all[i]])
    B_pt.set_data([xB_all[i]], [yB_all[i]])
    O_pt.set_data([xO_all[i]], [yO_all[i]])

    scale = 0.04
    force_l.set_data([xO_all[i], xO_all[i] + fx_all[i]*scale],
                     [yO_all[i], yO_all[i] + fy_all[i]*scale])

    traj_x.append(xO_all[i])
    traj_y.append(yO_all[i])
    traj_l.set_data(traj_x, traj_y)

    phi_B_i   = phi_B_all[i]
    phi_B_str = "N/A" if np.isnan(phi_B_i) else (
        f"{phi_B_i:+.4f} m" + (" ⚠ PEN" if phi_B_i < 0 else "")
    )

    info_txt.set_text(
        f"{PH_LABEL[ph_i]}\n"
        f"t    = {t_all[i]:.2f} s\n"
        f"ψ    = {np.degrees(psi_i):.1f}°\n"
        f"δ    = {np.degrees(delta_i):.1f}°\n"
        f"δ̇    = {np.degrees(omega_all[i]):.1f} deg/s\n"
        f"l    = {l_all[i]:.3f} m\n"
        f"ḷ    = {ldot_all[i]:.3f} m/s\n"
        f"fx   = {fx_all[i]:.2f} N\n"
        f"fy   = {fy_all[i]:.2f} N\n"
        f"φ_B  = {phi_B_str}"
    )

    return wall2_l, body_l, A_pt, B_pt, O_pt, force_l, traj_l, info_txt

ani = FuncAnimation(fig_a, update_anim, frames=frames, interval=30, blit=False)
plt.tight_layout()
print("\nSauvegarde du GIF...")
ani.save('open_loop_1_animation.gif', writer='pillow', fps=20)
print("GIF sauvegardé : open_loop_1_animation.gif")
plt.show()
