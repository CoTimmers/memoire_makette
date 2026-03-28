import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from matplotlib.animation import FuncAnimation

# ══════════════════════════════════════════════════════════════════════
# PARAMÈTRES
# ══════════════════════════════════════════════════════════════════════
params_ref = {
    "m": 7.0,
    "a": 0.3,
    "b": 0.4,
    "T_wall": 6.0,
    "psi_sw_1": np.radians(100),
    "l_contact": 0.4,
}

# Modèle "réel" différent du nominal
params_real = {
    "m": 5.0,
    "a": 0.3,
    "b": 0.4,
    "T_wall": 6.0,
    "psi_sw_1": np.radians(100),
    "l_contact": 0.4,
}

# Inerties nominales
params_ref["I_O"] = params_ref["m"] / 12.0 * (params_ref["a"]**2 + params_ref["b"]**2)
params_ref["I_A"] = params_ref["m"] /  3.0 * (params_ref["a"]**2 + params_ref["b"]**2)

# Inerties réelles différentes
params_real["I_O"] = 0.7 * params_ref["I_O"]
params_real["I_A"] = 0.7 * params_ref["I_A"]

for p in (params_ref, params_real):
    p["rAO_body"] = np.array([p["a"]/2, p["b"]/2])
    p["rAB_body"] = np.array([0.0, p["b"]])

T_SIM = 10.0

MODE_1  = 1
MODE_2A = 2
MODE_2B = 3
MODE_3  = 4

# ══════════════════════════════════════════════════════════════════════
# PROFIL DU MUR ψ(t)
# ══════════════════════════════════════════════════════════════════════
_T = params_ref["T_wall"]
_t1 = _T * 0.4
_t2 = _T * 0.6
_acc = (np.pi/2) / (0.5*_t1**2 + (_t2 - _t1)*_t1 + 0.5*(_T - _t2)**2)
_v1 = _acc * _t1
_p1 = np.pi/2 + 0.5*_acc*_t1**2
_p2 = _p1 + _v1*(_t2 - _t1)

def psi_profile(t):
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
# GÉOMÉTRIE / CINÉMATIQUE
# ══════════════════════════════════════════════════════════════════════
def phi_B_mode1(delta, psi, params):
    b = params["b"]
    return -b * np.cos(delta - psi)

def phi_B_mode2(l, delta, psi, params):
    b = params["b"]
    return l * np.sin(psi) - b * np.cos(delta - psi)

def delta_kinematics(l, ldot, psi, psidot, psiddot, b):
    eps = 1e-9
    u = np.clip((l / b) * np.sin(psi), -1 + eps, 1 - eps)
    k = 1.0 / np.sqrt(max(eps, 1 - u*u))
    k3 = k**3

    delta = psi - 0.5*np.pi + np.arcsin(u)
    u_l = np.sin(psi) / b
    u_psi = (l/b) * np.cos(psi)
    u_lpsi = np.cos(psi) / b
    u_psipsi = -(l/b) * np.sin(psi)

    delta_l = k * u_l
    delta_psi = 1 + k * u_psi
    delta_ll = u * k3 * u_l * u_l
    delta_lpsi = k * u_lpsi + u * k3 * u_l * u_psi
    delta_psipsi = k * u_psipsi + u * k3 * u_psi * u_psi
    delta_dot = delta_l * ldot + delta_psi * psidot

    return delta, delta_dot, delta_l, delta_psi, delta_ll, delta_lpsi, delta_psipsi

def impact_mode2_to_3(x_minus, params):
    l_minus, delta_minus, l_dot_minus, omega_minus = x_minus
    a = params["a"]
    b = params["b"]
    c1_minus = -(a/2)*np.sin(delta_minus) - (b/2)*np.cos(delta_minus)
    l_dot_plus = l_dot_minus + c1_minus * omega_minus
    return np.array([l_minus, l_dot_plus])

# ══════════════════════════════════════════════════════════════════════
# COMMANDES DE RÉFÉRENCE
# ══════════════════════════════════════════════════════════════════════
def input_mode1_ref(t):
    psi, _, _ = psi_profile(t)
    return -5.0 * np.sin(psi), 5.0 * np.cos(psi)

def input_mode2_ref(t):
    return 1.0, -5.0

_l_star = 0.4
_kp3 = 20.0
_kd3 = 15.0

def input_mode3_ref(t, x):
    l, l_dot = x
    fx = -_kp3 * (l - _l_star) - _kd3 * l_dot
    return fx, 0.0

# ══════════════════════════════════════════════════════════════════════
# FEEDBACKS
# ══════════════════════════════════════════════════════════════════════
# Mode 1 : tracking du COM + orientation avec A*=(0,0)
A_ref = np.array([0.0, 0.0])

_kpx1 = 40.0
_kdx1 = 18.0
_kpy1 = 40.0
_kdy1 = 18.0
_kpdelta1 = 8.0
_kddelta1 = 2.5

def mode1_reference_from_A(t, params, A_ref_xy=A_ref):
    """
    Construit la référence du COM à partir de :
      - A* fixé
      - delta*(t) = psi(t) - pi/2
    Retour :
      x_ref, y_ref, delta_ref, dx_ref, dy_ref, omega_ref, ddx_ref, ddy_ref, alpha_ref
    """
    psi, psi_dot, psi_ddot = psi_profile(t)

    a = params["a"]
    b = params["b"]
    xA_ref, yA_ref = A_ref_xy

    delta_ref = psi - np.pi/2
    omega_ref = psi_dot
    alpha_ref = psi_ddot

    cd = np.cos(delta_ref)
    sd = np.sin(delta_ref)

    # O = A + R(delta) r_AO_body
    x_ref = xA_ref + (a/2)*cd - (b/2)*sd
    y_ref = yA_ref + (a/2)*sd + (b/2)*cd

    # dérivées de O_ref
    dx_ref = (-(a/2)*sd - (b/2)*cd) * omega_ref
    dy_ref = ((a/2)*cd - (b/2)*sd) * omega_ref

    ddx_ref = (-(a/2)*cd + (b/2)*sd) * omega_ref**2 + (-(a/2)*sd - (b/2)*cd) * alpha_ref
    ddy_ref = (-(a/2)*sd - (b/2)*cd) * omega_ref**2 + ((a/2)*cd - (b/2)*sd) * alpha_ref

    return x_ref, y_ref, delta_ref, dx_ref, dy_ref, omega_ref, ddx_ref, ddy_ref, alpha_ref

def input_mode1_ctrl(t, x, params):
    """
    Contrôle en mode 1 basé sur :
      - maintien du point A à A_ref = (0,0)
      - suivi de delta_ref = psi - pi/2

    Etat :
      x = [xO, yO, delta, xOdot, yOdot, omega]

    Retour :
      Fx, Fy, tau
    """
    xO, yO, delta, xOdot, yOdot, omega = x
    m = params["m"]
    I_O = params["I_O"]
    rAO_body = params["rAO_body"]

    # Profil de référence du mur
    psi, psi_dot, psi_ddot = psi_profile(t)

    # Référence voulue
    xA_ref, yA_ref = A_ref
    delta_ref = psi - np.pi/2
    omega_ref = psi_dot
    alpha_ref = psi_ddot

    # Rotation actuelle
    cd = np.cos(delta)
    sd = np.sin(delta)
    R = np.array([[cd, -sd],
                  [sd,  cd]])

    rAO_w = R @ rAO_body

    # Point A reconstruit
    xA = xO - rAO_w[0]
    yA = yO - rAO_w[1]

    # Vitesse du point A
    # vA = vO - omega x rAO_w
    # avec omega x [rx, ry] = [-omega*ry, omega*rx]
    xAdot = xOdot + omega * rAO_w[1]
    yAdot = yOdot - omega * rAO_w[0]

    # Référence de O déduite de A_ref et delta_ref
    cd_ref = np.cos(delta_ref)
    sd_ref = np.sin(delta_ref)
    R_ref = np.array([[cd_ref, -sd_ref],
                      [sd_ref,  cd_ref]])
    rAO_ref_w = R_ref @ rAO_body

    xO_ref = xA_ref + rAO_ref_w[0]
    yO_ref = yA_ref + rAO_ref_w[1]

    # Vitesses de référence du COM induites par la rotation désirée
    # O_ref = A_ref + R(delta_ref) rAO_body
    # vO_ref = omega_ref x rAO_ref_w
    dxO_ref = -omega_ref * rAO_ref_w[1]
    dyO_ref =  omega_ref * rAO_ref_w[0]

    # Accélérations de référence du COM
    # aO_ref = alpha_ref x rAO_ref_w + omega_ref x (omega_ref x rAO_ref_w)
    ddxO_ref = -alpha_ref * rAO_ref_w[1] - omega_ref**2 * rAO_ref_w[0]
    ddyO_ref =  alpha_ref * rAO_ref_w[0] - omega_ref**2 * rAO_ref_w[1]

    # Erreurs sur A
    exA = xA - xA_ref
    eyA = yA - yA_ref
    exAdot = xAdot
    eyAdot = yAdot

    # Erreurs sur delta
    e_delta = delta - delta_ref
    e_omega = omega - omega_ref

    # Commande translation :
    # on garde une base feedforward sur O_ref,
    # mais la correction est faite à partir des erreurs sur A
    Fx = m * ddxO_ref - _kpx1 * exA - _kdx1 * exAdot
    Fy = m * ddyO_ref - _kpy1 * eyA - _kdy1 * eyAdot

    # Commande rotation
    tau = I_O * alpha_ref - _kpdelta1 * e_delta - _kddelta1 * e_omega

    return Fx, Fy, tau

# Mode 2A : suivi de l_ref
_kp_l_2a = 30.0
_kd_l_2a = 12.0

def input_mode2a_ctrl(t, l, l_dot, l_ref, ldot_ref):
    fx_ref, fy_ref = input_mode2_ref(t)
    fx_fb = -_kp_l_2a * (l - l_ref) - _kd_l_2a * (l_dot - ldot_ref)
    fy_fb = 0.0
    return fx_ref + fx_fb, fy_ref + fy_fb

# Mode 2B : suivi de [l, delta, l_dot, omega]
_kp_l_2b     = 20.0
_kd_l_2b     = 8.0
_kp_delta_2b = 14.0
_kd_delta_2b = 6.0

def input_mode2b_ctrl(t, x, x_ref):
    l, delta, l_dot, omega = x
    l_ref, delta_ref, ldot_ref, omega_ref = x_ref

    fx_ref, fy_ref = input_mode2_ref(t)

    fx_fb = -_kp_l_2b * (l - l_ref) - _kd_l_2b * (l_dot - ldot_ref)
    fy_fb = -_kp_delta_2b * (delta - delta_ref) - _kd_delta_2b * (omega - omega_ref)

    return fx_ref + fx_fb, fy_ref + fy_fb

# ══════════════════════════════════════════════════════════════════════
# ÉVÉNEMENTS
# ══════════════════════════════════════════════════════════════════════
def event_mode1_to_2(t, x, params):
    psi, _, _ = psi_profile(t)
    return psi - params["psi_sw_1"]

def event_2a_to_2b(t, x, params, fxfy_fun):
    _, lambda_B, _, _ = mode2a_core(t, x, params, fxfy_fun)
    return lambda_B

def event_2b_to_2a(t, x, params):
    l, delta, _, _ = x
    psi, _, _ = psi_profile(t)
    return phi_B_mode2(l, delta, psi, params)

def event_2_to_3_from_2a(t, x, params):
    return x[0] - params["l_contact"]

def event_2_to_3_from_2b(t, x, params):
    return x[0] - params["l_contact"]

# ══════════════════════════════════════════════════════════════════════
# DYNAMIQUES NOMINALES
# ══════════════════════════════════════════════════════════════════════
def mode2a_core(t, x, params, fxfy_fun):
    l, l_dot = x
    m_p = params["m"]
    I_A = params["I_A"]
    a = params["a"]
    b = params["b"]

    psi, psi_dot, psi_ddot = psi_profile(t)
    fx, fy = fxfy_fun(t, x)

    delta, delta_dot, delta_l, delta_psi, delta_ll, delta_lpsi, delta_psipsi = \
        delta_kinematics(l, l_dot, psi, psi_dot, psi_ddot, b)

    c1 = -(a/2)*np.sin(delta) - (b/2)*np.cos(delta)
    c2 =  (a/2)*np.cos(delta) - (b/2)*np.sin(delta)
    c3 = -(a/2)*np.cos(delta) + (b/2)*np.sin(delta)

    A_kin = (
        delta_ll*l_dot**2
        + 2*delta_lpsi*l_dot*psi_dot
        + delta_psipsi*psi_dot**2
        + delta_psi*psi_ddot
    )

    denom = m_p * (1.0 + c1 * delta_l)
    l_ddot = ((fx - m_p*(c1*A_kin + c3*delta_dot**2)) / denom
              if abs(denom) > 1e-12 else 0.0)

    delta_ddot = delta_l * l_ddot + A_kin

    sin_dp = np.sin(delta - psi)
    denom_l = b * sin_dp
    lambda_B = ((m_p*c1*l_ddot + I_A*delta_ddot - c1*fx - c2*fy) / denom_l
                if abs(denom_l) > 1e-10 else 0.0)

    return l_ddot, lambda_B, delta, delta_dot

def dynamics_mode2a_nominal(t, x, params):
    def fxfy_fun(tt, xx):
        return input_mode2_ref(tt)
    l_dot = x[1]
    l_ddot, _, _, _ = mode2a_core(t, x, params, fxfy_fun)
    return [l_dot, l_ddot]

def dynamics_mode2b_nominal(t, x, params):
    l, delta, l_dot, omega = x
    m_p = params["m"]
    I_A = params["I_A"]
    a = params["a"]
    b = params["b"]

    fx, fy = input_mode2_ref(t)

    c1 = -(a/2)*np.sin(delta) - (b/2)*np.cos(delta)
    c2 =  (a/2)*np.cos(delta) - (b/2)*np.sin(delta)
    c3 = -(a/2)*np.cos(delta) + (b/2)*np.sin(delta)

    M_sys = np.array([[m_p,     m_p*c1],
                      [m_p*c1,  I_A   ]])
    rhs = np.array([fx - m_p*c3*omega**2,
                    c1*fx + c2*fy])

    sol = np.linalg.solve(M_sys, rhs)
    l_ddot, delta_ddot = sol
    return [l_dot, omega, l_ddot, delta_ddot]

def dynamics_mode3(t, x, params):
    l, l_dot = x
    m_p = params["m"]
    fx, _ = input_mode3_ref(t, x)
    return [l_dot, fx/m_p]

# ══════════════════════════════════════════════════════════════════════
# DYNAMIQUES "RÉELLES" CONTRÔLÉES
# ══════════════════════════════════════════════════════════════════════
# Mode 1 réel :
# état = [xO, yO, delta, xOdot, yOdot, omega]
def dynamics_mode1_real(t, x, params):
    xO, yO, delta, xOdot, yOdot, omega = x
    m_p = params["m"]
    I_O = params["I_O"]

    Fx, Fy, tau = input_mode1_ctrl(t, x, params)

    return [
        xOdot,
        yOdot,
        omega,
        Fx / m_p,
        Fy / m_p,
        tau / I_O,
    ]

def dynamics_mode2a_real(t, x, params, l_ref_fun, ldot_ref_fun):
    def fxfy_fun(tt, xx):
        l, l_dot = xx
        l_ref = float(l_ref_fun(tt))
        ldot_ref = float(ldot_ref_fun(tt))
        return input_mode2a_ctrl(tt, l, l_dot, l_ref, ldot_ref)

    l_dot = x[1]
    l_ddot, _, _, _ = mode2a_core(t, x, params, fxfy_fun)
    return [l_dot, l_ddot]

def dynamics_mode2b_real(t, x, params, x_ref_fun):
    l, delta, l_dot, omega = x
    m_p = params["m"]
    I_A = params["I_A"]
    a = params["a"]
    b = params["b"]

    x_ref = x_ref_fun(t)
    fx, fy = input_mode2b_ctrl(t, x, x_ref)

    c1 = -(a/2)*np.sin(delta) - (b/2)*np.cos(delta)
    c2 =  (a/2)*np.cos(delta) - (b/2)*np.sin(delta)
    c3 = -(a/2)*np.cos(delta) + (b/2)*np.sin(delta)

    M_sys = np.array([[m_p,     m_p*c1],
                      [m_p*c1,  I_A   ]])
    rhs = np.array([fx - m_p*c3*omega**2,
                    c1*fx + c2*fy])

    sol = np.linalg.solve(M_sys, rhs)
    l_ddot, delta_ddot = sol
    return [l_dot, omega, l_ddot, delta_ddot]

# ══════════════════════════════════════════════════════════════════════
# SIMULATION NOMINALE
# ══════════════════════════════════════════════════════════════════════
def simulate_nominal(params):
    # Mode 1 nominal imposé
    def _psi_minus_target(t):
        return psi_profile(t)[0] - params["psi_sw_1"]

    t1_end = brentq(_psi_minus_target, 0.0, params["T_wall"])
    t1_arr = np.linspace(0.0, t1_end, 300)

    l1 = np.zeros_like(t1_arr)
    ldot1 = np.zeros_like(t1_arr)
    psi1 = np.array([psi_profile(ti)[0] for ti in t1_arr])
    psidot1 = np.array([psi_profile(ti)[1] for ti in t1_arr])
    delta1 = psi1 - np.pi/2
    omega1 = psidot1
    fx1 = np.array([input_mode1_ref(ti)[0] for ti in t1_arr])
    fy1 = np.array([input_mode1_ref(ti)[1] for ti in t1_arr])
    mode1 = np.full(len(t1_arr), MODE_1, dtype=int)

    # Boucle hybride 2A/2B/3
    t_segs = []
    l_segs = []
    ldot_segs = []
    delta_segs = []
    omega_segs = []
    psi_segs = []
    fx_segs = []
    fy_segs = []
    mode_segs = []

    mode = MODE_2A
    t_cur = t1_end
    x_cur = np.array([0.0, 0.0])  # [l, l_dot] pour 2A

    while t_cur < T_SIM:
        if mode == MODE_2A:
            def ev_impact(t, x):
                return event_2_to_3_from_2a(t, x, params)
            ev_impact.terminal = True
            ev_impact.direction = 1

            def ev_sep(t, x):
                return event_2a_to_2b(t, x, params, lambda tt, xx: input_mode2_ref(tt))
            ev_sep.terminal = True
            ev_sep.direction = -1

            sol = solve_ivp(
                lambda t, x: dynamics_mode2a_nominal(t, x, params),
                (t_cur, T_SIM),
                x_cur,
                dense_output=True,
                events=[ev_impact, ev_sep],
                rtol=1e-9,
                atol=1e-11,
            )

            t_end = float(sol.t[-1])
            t_seg = np.linspace(t_cur, t_end, max(3, int(300 * (t_end - t_cur) / T_SIM) + 3))
            y_seg = sol.sol(t_seg)
            l_seg = y_seg[0]
            ldot_seg = y_seg[1]

            psi_seg = np.array([psi_profile(tt)[0] for tt in t_seg])
            delta_seg = np.zeros_like(t_seg)
            omega_seg = np.zeros_like(t_seg)
            for i, tt in enumerate(t_seg):
                _, _, d_i, w_i = mode2a_core(tt, np.array([l_seg[i], ldot_seg[i]]), params,
                                             lambda ttt, xxx: input_mode2_ref(ttt))
                delta_seg[i] = d_i
                omega_seg[i] = w_i

            fx_seg = np.array([input_mode2_ref(tt)[0] for tt in t_seg])
            fy_seg = np.array([input_mode2_ref(tt)[1] for tt in t_seg])

            t_segs.append(t_seg)
            l_segs.append(l_seg)
            ldot_segs.append(ldot_seg)
            delta_segs.append(delta_seg)
            omega_segs.append(omega_seg)
            psi_segs.append(psi_seg)
            fx_segs.append(fx_seg)
            fy_segs.append(fy_seg)
            mode_segs.append(np.full(len(t_seg), MODE_2A, dtype=int))

            if len(sol.t_events[0]) > 0:
                x_minus = np.array([l_seg[-1], delta_seg[-1], ldot_seg[-1], omega_seg[-1]])
                x_cur = impact_mode2_to_3(x_minus, params)
                t_cur = t_end
                mode = MODE_3
                break
            elif len(sol.t_events[1]) > 0:
                x_cur = np.array([l_seg[-1], delta_seg[-1], ldot_seg[-1], omega_seg[-1]])
                t_cur = t_end
                mode = MODE_2B
            else:
                t_cur = T_SIM

        elif mode == MODE_2B:
            def ev_impact(t, x):
                return event_2_to_3_from_2b(t, x, params)
            ev_impact.terminal = True
            ev_impact.direction = 1

            def ev_recontact(t, x):
                return event_2b_to_2a(t, x, params)
            ev_recontact.terminal = True
            ev_recontact.direction = -1

            sol = solve_ivp(
                lambda t, x: dynamics_mode2b_nominal(t, x, params),
                (t_cur, T_SIM),
                x_cur,
                dense_output=True,
                events=[ev_impact, ev_recontact],
                rtol=1e-9,
                atol=1e-11,
            )

            t_end = float(sol.t[-1])
            t_seg = np.linspace(t_cur, t_end, max(3, int(300 * (t_end - t_cur) / T_SIM) + 3))
            y_seg = sol.sol(t_seg)

            l_seg = y_seg[0]
            delta_seg = y_seg[1]
            ldot_seg = y_seg[2]
            omega_seg = y_seg[3]
            psi_seg = np.array([psi_profile(tt)[0] for tt in t_seg])
            fx_seg = np.array([input_mode2_ref(tt)[0] for tt in t_seg])
            fy_seg = np.array([input_mode2_ref(tt)[1] for tt in t_seg])

            t_segs.append(t_seg)
            l_segs.append(l_seg)
            ldot_segs.append(ldot_seg)
            delta_segs.append(delta_seg)
            omega_segs.append(omega_seg)
            psi_segs.append(psi_seg)
            fx_segs.append(fx_seg)
            fy_segs.append(fy_seg)
            mode_segs.append(np.full(len(t_seg), MODE_2B, dtype=int))

            if len(sol.t_events[0]) > 0:
                x_minus = np.array([l_seg[-1], delta_seg[-1], ldot_seg[-1], omega_seg[-1]])
                x_cur = impact_mode2_to_3(x_minus, params)
                t_cur = t_end
                mode = MODE_3
                break
            elif len(sol.t_events[1]) > 0:
                x_cur = np.array([l_seg[-1], ldot_seg[-1]])
                t_cur = t_end
                mode = MODE_2A
            else:
                t_cur = T_SIM

    if mode == MODE_3 and t_cur < T_SIM:
        sol3 = solve_ivp(
            lambda t, x: dynamics_mode3(t, x, params),
            (t_cur, T_SIM),
            x_cur,
            dense_output=True,
            rtol=1e-9,
            atol=1e-11,
        )
        t3 = np.linspace(t_cur, T_SIM, 300)
        y3 = sol3.sol(t3)
        l3 = y3[0]
        ldot3 = y3[1]
        delta3 = np.full(len(t3), np.pi/2)
        omega3 = np.zeros(len(t3))
        psi3 = np.array([psi_profile(tt)[0] for tt in t3])
        fx3 = np.array([input_mode3_ref(tt, [l3[i], ldot3[i]])[0] for i, tt in enumerate(t3)])
        fy3 = np.zeros(len(t3))

        t_segs.append(t3)
        l_segs.append(l3)
        ldot_segs.append(ldot3)
        delta_segs.append(delta3)
        omega_segs.append(omega3)
        psi_segs.append(psi3)
        fx_segs.append(fx3)
        fy_segs.append(fy3)
        mode_segs.append(np.full(len(t3), MODE_3, dtype=int))

    t_h = np.concatenate(t_segs) if t_segs else np.array([])
    l_h = np.concatenate(l_segs) if t_segs else np.array([])
    ldot_h = np.concatenate(ldot_segs) if t_segs else np.array([])
    delta_h = np.concatenate(delta_segs) if t_segs else np.array([])
    omega_h = np.concatenate(omega_segs) if t_segs else np.array([])
    psi_h = np.concatenate(psi_segs) if t_segs else np.array([])
    fx_h = np.concatenate(fx_segs) if t_segs else np.array([])
    fy_h = np.concatenate(fy_segs) if t_segs else np.array([])
    mode_h = np.concatenate(mode_segs) if t_segs else np.array([], dtype=int)

    return {
        "t": np.concatenate([t1_arr, t_h]),
        "l": np.concatenate([l1, l_h]),
        "ldot": np.concatenate([ldot1, ldot_h]),
        "delta": np.concatenate([delta1, delta_h]),
        "omega": np.concatenate([omega1, omega_h]),
        "psi": np.concatenate([psi1, psi_h]),
        "fx": np.concatenate([fx1, fx_h]),
        "fy": np.concatenate([fy1, fy_h]),
        "mode": np.concatenate([mode1, mode_h]),
        "t1_end": t1_end,
    }

# ══════════════════════════════════════════════════════════════════════
# RÉFÉRENCE NOMINALE
# ══════════════════════════════════════════════════════════════════════
ref = simulate_nominal(params_ref)

# Interpolants de référence
t_ref = ref["t"]
l_ref_fun = interp1d(t_ref, ref["l"], kind="linear", fill_value="extrapolate")
ldot_ref_fun = interp1d(t_ref, ref["ldot"], kind="linear", fill_value="extrapolate")
delta_ref_fun = interp1d(t_ref, ref["delta"], kind="linear", fill_value="extrapolate")
omega_ref_fun = interp1d(t_ref, ref["omega"], kind="linear", fill_value="extrapolate")

def x_ref_mode2b(t):
    return np.array([
        float(l_ref_fun(t)),
        float(delta_ref_fun(t)),
        float(ldot_ref_fun(t)),
        float(omega_ref_fun(t)),
    ])

# ══════════════════════════════════════════════════════════════════════
# SIMULATION RÉELLE AVEC FEEDBACK
# ══════════════════════════════════════════════════════════════════════
def simulate_real(params):
    # Mode 1 réel = tracking du COM + orientation
    def ev_1_to_2(t, x):
        return event_mode1_to_2(t, x, params)
    ev_1_to_2.terminal = True
    ev_1_to_2.direction = 1

    # CI cohérentes avec la référence à t=0
    (x_ref0, y_ref0, delta_ref0,
     dx_ref0, dy_ref0, omega_ref0,
     _, _, _) = mode1_reference_from_A(0.0, params, A_ref)

    x0_1 = np.array([x_ref0, y_ref0, delta_ref0, dx_ref0, dy_ref0, omega_ref0])

    sol1 = solve_ivp(
        lambda t, x: dynamics_mode1_real(t, x, params),
        (0.0, T_SIM),
        x0_1,
        dense_output=True,
        events=[ev_1_to_2],
        rtol=1e-9,
        atol=1e-11,
    )

    t1_end = float(sol1.t[-1])
    t1 = np.linspace(0.0, t1_end, 300)
    y1 = sol1.sol(t1)

    xO1 = y1[0]
    yO1 = y1[1]
    delta1 = y1[2]
    xOdot1 = y1[3]
    yOdot1 = y1[4]
    omega1 = y1[5]

    psi1 = np.array([psi_profile(tt)[0] for tt in t1])

    fx1 = np.zeros(len(t1))
    fy1 = np.zeros(len(t1))
    tau1 = np.zeros(len(t1))
    xA1 = np.zeros(len(t1))
    yA1 = np.zeros(len(t1))
    xAdot1 = np.zeros(len(t1))
    yAdot1 = np.zeros(len(t1))

    rAO_body = params["rAO_body"]

    for i, tt in enumerate(t1):
        Fx_i, Fy_i, Tau_i = input_mode1_ctrl(tt, y1[:, i], params)
        fx1[i] = Fx_i
        fy1[i] = Fy_i
        tau1[i] = Tau_i

        cd = np.cos(delta1[i])
        sd = np.sin(delta1[i])
        R_i = np.array([[cd, -sd], [sd, cd]])
        rAO_w = R_i @ rAO_body

        # A = O - R rAO_body
        xA1[i] = xO1[i] - rAO_w[0]
        yA1[i] = yO1[i] - rAO_w[1]

        # vitesse de A : vA = vO - omega x (R rAO_body)
        # omega x [rx, ry] = [-omega*ry, omega*rx]
        xAdot1[i] = xOdot1[i] + omega1[i] * rAO_w[1]
        yAdot1[i] = yOdot1[i] - omega1[i] * rAO_w[0]

    # Projection vers mode 2A
    # l = x_A, ldot = xdot_A
    mode = MODE_2A
    t_cur = t1_end
    l0 = max(0.0, xA1[-1])
    ldot0 = xAdot1[-1] if l0 > 1e-6 else max(0.0, xAdot1[-1])
    x_cur = np.array([l0, ldot0])

    t_segs = []
    l_segs = []
    ldot_segs = []
    delta_segs = []
    omega_segs = []
    psi_segs = []
    fx_segs = []
    fy_segs = []
    mode_segs = []

    while t_cur < T_SIM:
        if mode == MODE_2A:
            def ev_impact(t, x):
                return event_2_to_3_from_2a(t, x, params)
            ev_impact.terminal = True
            ev_impact.direction = 1

            def ev_sep(t, x):
                return event_2a_to_2b(
                    t, x, params,
                    lambda tt, xx: input_mode2a_ctrl(
                        tt, xx[0], xx[1],
                        float(l_ref_fun(tt)),
                        float(ldot_ref_fun(tt))
                    )
                )
            ev_sep.terminal = True
            ev_sep.direction = -1

            sol = solve_ivp(
                lambda t, x: dynamics_mode2a_real(t, x, params, l_ref_fun, ldot_ref_fun),
                (t_cur, T_SIM),
                x_cur,
                dense_output=True,
                events=[ev_impact, ev_sep],
                rtol=1e-9,
                atol=1e-11,
            )

            t_end = float(sol.t[-1])
            t_seg = np.linspace(t_cur, t_end, max(3, int(300 * (t_end - t_cur) / T_SIM) + 3))
            y_seg = sol.sol(t_seg)

            l_seg = y_seg[0]
            ldot_seg = y_seg[1]
            psi_seg = np.array([psi_profile(tt)[0] for tt in t_seg])

            delta_seg = np.zeros_like(t_seg)
            omega_seg = np.zeros_like(t_seg)
            fx_seg = np.zeros_like(t_seg)
            fy_seg = np.zeros_like(t_seg)

            for i, tt in enumerate(t_seg):
                l_ref = float(l_ref_fun(tt))
                ldot_ref = float(ldot_ref_fun(tt))

                _, _, d_i, w_i = mode2a_core(
                    tt,
                    np.array([l_seg[i], ldot_seg[i]]),
                    params,
                    lambda ttt, xxx: input_mode2a_ctrl(
                        ttt, xxx[0], xxx[1],
                        float(l_ref_fun(ttt)),
                        float(ldot_ref_fun(ttt))
                    )
                )
                delta_seg[i] = d_i
                omega_seg[i] = w_i
                fx_seg[i], fy_seg[i] = input_mode2a_ctrl(tt, l_seg[i], ldot_seg[i], l_ref, ldot_ref)

            t_segs.append(t_seg)
            l_segs.append(l_seg)
            ldot_segs.append(ldot_seg)
            delta_segs.append(delta_seg)
            omega_segs.append(omega_seg)
            psi_segs.append(psi_seg)
            fx_segs.append(fx_seg)
            fy_segs.append(fy_seg)
            mode_segs.append(np.full(len(t_seg), MODE_2A, dtype=int))

            if len(sol.t_events[0]) > 0:
                x_minus = np.array([l_seg[-1], delta_seg[-1], ldot_seg[-1], omega_seg[-1]])
                x_cur = impact_mode2_to_3(x_minus, params)
                t_cur = t_end
                mode = MODE_3
                break
            elif len(sol.t_events[1]) > 0:
                x_cur = np.array([l_seg[-1], delta_seg[-1], ldot_seg[-1], omega_seg[-1]])
                t_cur = t_end
                mode = MODE_2B
            else:
                t_cur = T_SIM

        elif mode == MODE_2B:
            def ev_impact(t, x):
                return event_2_to_3_from_2b(t, x, params)
            ev_impact.terminal = True
            ev_impact.direction = 1

            def ev_recontact(t, x):
                return event_2b_to_2a(t, x, params)
            ev_recontact.terminal = True
            ev_recontact.direction = -1

            sol = solve_ivp(
                lambda t, x: dynamics_mode2b_real(t, x, params, x_ref_mode2b),
                (t_cur, T_SIM),
                x_cur,
                dense_output=True,
                events=[ev_impact, ev_recontact],
                rtol=1e-9,
                atol=1e-11,
            )

            t_end = float(sol.t[-1])
            t_seg = np.linspace(t_cur, t_end, max(3, int(300 * (t_end - t_cur) / T_SIM) + 3))
            y_seg = sol.sol(t_seg)

            l_seg = y_seg[0]
            delta_seg = y_seg[1]
            ldot_seg = y_seg[2]
            omega_seg = y_seg[3]
            psi_seg = np.array([psi_profile(tt)[0] for tt in t_seg])

            fx_seg = np.zeros_like(t_seg)
            fy_seg = np.zeros_like(t_seg)
            for i, tt in enumerate(t_seg):
                fx_seg[i], fy_seg[i] = input_mode2b_ctrl(
                    tt,
                    np.array([l_seg[i], delta_seg[i], ldot_seg[i], omega_seg[i]]),
                    x_ref_mode2b(tt)
                )

            t_segs.append(t_seg)
            l_segs.append(l_seg)
            ldot_segs.append(ldot_seg)
            delta_segs.append(delta_seg)
            omega_segs.append(omega_seg)
            psi_segs.append(psi_seg)
            fx_segs.append(fx_seg)
            fy_segs.append(fy_seg)
            mode_segs.append(np.full(len(t_seg), MODE_2B, dtype=int))

            if len(sol.t_events[0]) > 0:
                x_minus = np.array([l_seg[-1], delta_seg[-1], ldot_seg[-1], omega_seg[-1]])
                x_cur = impact_mode2_to_3(x_minus, params)
                t_cur = t_end
                mode = MODE_3
                break
            elif len(sol.t_events[1]) > 0:
                x_cur = np.array([l_seg[-1], ldot_seg[-1]])
                t_cur = t_end
                mode = MODE_2A
            else:
                t_cur = T_SIM

    if mode == MODE_3 and t_cur < T_SIM:
        sol3 = solve_ivp(
            lambda t, x: dynamics_mode3(t, x, params),
            (t_cur, T_SIM),
            x_cur,
            dense_output=True,
            rtol=1e-9,
            atol=1e-11,
        )

        t3 = np.linspace(t_cur, T_SIM, 300)
        y3 = sol3.sol(t3)
        l3 = y3[0]
        ldot3 = y3[1]
        delta3 = np.full(len(t3), np.pi/2)
        omega3 = np.zeros(len(t3))
        psi3 = np.array([psi_profile(tt)[0] for tt in t3])
        fx3 = np.array([input_mode3_ref(tt, [l3[i], ldot3[i]])[0] for i, tt in enumerate(t3)])
        fy3 = np.zeros(len(t3))

        t_segs.append(t3)
        l_segs.append(l3)
        ldot_segs.append(ldot3)
        delta_segs.append(delta3)
        omega_segs.append(omega3)
        psi_segs.append(psi3)
        fx_segs.append(fx3)
        fy_segs.append(fy3)
        mode_segs.append(np.full(len(t3), MODE_3, dtype=int))

    t_h = np.concatenate(t_segs) if t_segs else np.array([])
    l_h = np.concatenate(l_segs) if t_segs else np.array([])
    ldot_h = np.concatenate(ldot_segs) if t_segs else np.array([])
    delta_h = np.concatenate(delta_segs) if t_segs else np.array([])
    omega_h = np.concatenate(omega_segs) if t_segs else np.array([])
    psi_h = np.concatenate(psi_segs) if t_segs else np.array([])
    fx_h = np.concatenate(fx_segs) if t_segs else np.array([])
    fy_h = np.concatenate(fy_segs) if t_segs else np.array([])
    mode_h = np.concatenate(mode_segs) if t_segs else np.array([], dtype=int)

    return {
        "t": np.concatenate([t1, t_h]),
        "l": np.concatenate([xA1, l_h]),          # en mode 1, on stocke x_A pour affichage cohérent
        "ldot": np.concatenate([xAdot1, ldot_h]), # idem
        "delta": np.concatenate([delta1, delta_h]),
        "omega": np.concatenate([omega1, omega_h]),
        "psi": np.concatenate([psi1, psi_h]),
        "fx": np.concatenate([fx1, fx_h]),
        "fy": np.concatenate([fy1, fy_h]),
        "mode": np.concatenate([np.full(len(t1), MODE_1, dtype=int), mode_h]),
        "xA_mode1": xA1,
        "yA_mode1": yA1,
        "xO_mode1": xO1,
        "yO_mode1": yO1,
        "xAdot_mode1": xAdot1,
        "yAdot_mode1": yAdot1,
        "t_mode1": t1,
    }

real = simulate_real(params_real)

# ══════════════════════════════════════════════════════════════════════
# AFFICHAGES
# ══════════════════════════════════════════════════════════════════════
plt.figure(figsize=(8, 4))
plt.plot(ref["t"], ref["l"], label="l_ref")
plt.plot(real["t"], real["l"], "--", label="l_real")
plt.axhline(_l_star, color="k", linestyle=":", label="l*")
plt.xlabel("t (s)")
plt.ylabel("l (m)")
plt.title("Suivi de l")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(ref["t"], np.degrees(ref["delta"]), label="delta_ref")
plt.plot(real["t"], np.degrees(real["delta"]), "--", label="delta_real")
plt.xlabel("t (s)")
plt.ylabel("delta (deg)")
plt.title("Suivi de l'orientation")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(real["t_mode1"], real["xA_mode1"], label="x_A réel")
plt.plot(real["t_mode1"], real["yA_mode1"], label="y_A réel")
plt.axhline(0.0, color="k", linestyle=":")
plt.xlabel("t (s)")
plt.ylabel("position A (m)")
plt.title("Mode 1 réel — point A reconstruit")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(real["t_mode1"], real["xO_mode1"], label="x_O réel")
plt.plot(real["t_mode1"], real["yO_mode1"], label="y_O réel")
plt.xlabel("t (s)")
plt.ylabel("position COM (m)")
plt.title("Mode 1 réel — suivi du COM")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(ref["t"], ref["fx"], label="fx_ref")
plt.plot(real["t"], real["fx"], "--", label="fx_real")
plt.plot(ref["t"], ref["fy"], label="fy_ref")
plt.plot(real["t"], real["fy"], "--", label="fy_real")
plt.xlabel("t (s)")
plt.ylabel("Force (N)")
plt.title("Forces : référence vs réel contrôlé")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

print("Simulation terminée.")
print(f"Masse nominale  = {params_ref['m']:.3f} kg")
print(f"Masse réelle    = {params_real['m']:.3f} kg")
print(f"I_A nominal     = {params_ref['I_A']:.5f} kg.m²")
print(f"I_A réel        = {params_real['I_A']:.5f} kg.m²")

# ══════════════════════════════════════════════════════════════════════
# RECONSTRUCTION GÉOMÉTRIQUE POUR L'ANIMATION
# ══════════════════════════════════════════════════════════════════════
def reconstruct_geometry(sim, params):
    t_all = sim["t"]
    l_all = sim["l"]
    delta_all = sim["delta"]
    mode_all = sim["mode"]

    xA_all = np.zeros_like(t_all)
    yA_all = np.zeros_like(t_all)
    xO_all = np.zeros_like(t_all)
    yO_all = np.zeros_like(t_all)
    xB_all = np.zeros_like(t_all)
    yB_all = np.zeros_like(t_all)

    rAO_body = params["rAO_body"]
    rAB_body = params["rAB_body"]

    # Pour le mode 1, on reconstruit depuis O stocké
    t_mode1 = sim["t_mode1"]
    xO_mode1 = sim["xO_mode1"]
    yO_mode1 = sim["yO_mode1"]

    for i in range(len(t_all)):
        cd, sd = np.cos(delta_all[i]), np.sin(delta_all[i])
        R_i = np.array([[cd, -sd], [sd, cd]])
        rAO_w = R_i @ rAO_body
        rAB_w = R_i @ rAB_body

        if mode_all[i] == MODE_1 and i < len(t_mode1):
            xO_all[i] = xO_mode1[i]
            yO_all[i] = yO_mode1[i]
            xA_all[i] = xO_all[i] - rAO_w[0]
            yA_all[i] = yO_all[i] - rAO_w[1]
            xB_all[i] = xA_all[i] + rAB_w[0]
            yB_all[i] = yA_all[i] + rAB_w[1]
        else:
            # modes 2A/2B/3 : l = x_A, y_A = 0
            xA_all[i] = l_all[i]
            yA_all[i] = 0.0
            xO_all[i] = xA_all[i] + rAO_w[0]
            yO_all[i] = yA_all[i] + rAO_w[1]
            xB_all[i] = xA_all[i] + rAB_w[0]
            yB_all[i] = yA_all[i] + rAB_w[1]

    return {
        "xA": xA_all, "yA": yA_all,
        "xO": xO_all, "yO": yO_all,
        "xB": xB_all, "yB": yB_all,
    }

geo_real = reconstruct_geometry(real, params_real)

from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════════
# ANIMATION À TEMPS PHYSIQUE RÉEL
# ══════════════════════════════════════════════════════════════════════

PH_COLOR = {
    MODE_1: "steelblue",
    MODE_2A: "seagreen",
    MODE_2B: "darkolivegreen",
    MODE_3: "darkorange",
}
PH_LABEL = {
    MODE_1: "Mode 1",
    MODE_2A: "Mode 2A",
    MODE_2B: "Mode 2B",
    MODE_3: "Mode 3",
}

# Reconstruction géométrique
geo_real = reconstruct_geometry(real, params_real)

# ── IMPORTANT : frames basées sur le temps physique ──────────────────
fps = 20                      # images par seconde du GIF
dt_anim = 1.0 / fps           # pas de temps physique entre 2 frames
t_anim = np.arange(real["t"][0], real["t"][-1], dt_anim)

# pour chaque temps physique, on prend l'indice correspondant dans real["t"]
frames = [np.searchsorted(real["t"], ta, side="left") for ta in t_anim]
frames = np.clip(frames, 0, len(real["t"]) - 1)

# optionnel : supprimer les doublons d'indices
frames = np.unique(frames)

# ── Figure ────────────────────────────────────────────────────────────
fig_a, ax_a = plt.subplots(figsize=(8, 7))
ax_a.set_xlim(-0.60, 0.65)
ax_a.set_ylim(-0.15, 0.65)
ax_a.set_aspect("equal")
ax_a.grid(True, alpha=0.3)
ax_a.set_xlabel("x (m)")
ax_a.set_ylabel("y (m)")
ax_a.set_title("Simulation réelle contrôlée — modes 1 / 2A / 2B / 3")

# mur 1
ax_a.plot([0, 0.6], [0, 0], color="black", lw=3, zorder=3, label="mur 1")

# cible mode 3
ax_a.axvline(_l_star, color="darkorange", lw=1.5, ls="--", alpha=0.6, label=f"l* = {_l_star} m")

wall2_l, = ax_a.plot([], [], color="dimgray", lw=3, label="mur 2")
body_l,  = ax_a.plot([], [], color="steelblue", lw=2, label="bac")
A_pt,    = ax_a.plot([], [], "o", color="crimson", ms=8, zorder=4, label="A")
B_pt,    = ax_a.plot([], [], "s", color="seagreen", ms=8, zorder=4, label="B")
O_pt,    = ax_a.plot([], [], "^", color="darkorange", ms=8, zorder=4, label="O (COM)")
force_l, = ax_a.plot([], [], color="mediumpurple", lw=2.5, label="force F")
traj_l,  = ax_a.plot([], [], color="steelblue", lw=1, alpha=0.35)

info_txt = ax_a.text(
    0.02, 0.97, "", transform=ax_a.transAxes,
    color="black", fontsize=9, va="top", fontfamily="monospace",
    bbox=dict(facecolor="white", alpha=0.75, edgecolor="lightgray")
)

ax_a.legend(loc="upper right", fontsize=8)

traj_x, traj_y = [], []

def update_anim(i):
    t_i = real["t"][i]
    psi_i = real["psi"][i]
    delta_i = real["delta"][i]
    mode_i = real["mode"][i]

    # mur 2
    L2 = 0.55
    wall2_l.set_data([0, L2*np.cos(psi_i)], [0, L2*np.sin(psi_i)])

    # bac
    cd, sd = np.cos(delta_i), np.sin(delta_i)
    R_i = np.array([[cd, -sd], [sd, cd]])
    a = params_real["a"]
    b = params_real["b"]
    corners = np.array([[0, 0], [a, 0], [a, b], [0, b], [0, 0]])

    world = np.array([
        np.array([geo_real["xA"][i], geo_real["yA"][i]]) + R_i @ c
        for c in corners
    ])

    body_l.set_data(world[:, 0], world[:, 1])
    body_l.set_color(PH_COLOR[mode_i])

    A_pt.set_data([geo_real["xA"][i]], [geo_real["yA"][i]])
    B_pt.set_data([geo_real["xB"][i]], [geo_real["yB"][i]])
    O_pt.set_data([geo_real["xO"][i]], [geo_real["yO"][i]])

    # force appliquée
    scale = 0.04
    force_l.set_data(
        [geo_real["xO"][i], geo_real["xO"][i] + real["fx"][i]*scale],
        [geo_real["yO"][i], geo_real["yO"][i] + real["fy"][i]*scale]
    )

    # trajectoire du COM
    traj_x.append(geo_real["xO"][i])
    traj_y.append(geo_real["yO"][i])
    traj_l.set_data(traj_x, traj_y)

    info_txt.set_text(
        f"{PH_LABEL[mode_i]}\n"
        f"t    = {t_i:.2f} s\n"
        f"ψ    = {np.degrees(psi_i):.1f}°\n"
        f"δ    = {np.degrees(delta_i):.1f}°\n"
        f"l    = {real['l'][i]:.3f} m\n"
        f"ḷ    = {real['ldot'][i]:.3f} m/s\n"
        f"fx   = {real['fx'][i]:.2f} N\n"
        f"fy   = {real['fy'][i]:.2f} N"
    )

    return wall2_l, body_l, A_pt, B_pt, O_pt, force_l, traj_l, info_txt

ani = FuncAnimation(
    fig_a,
    update_anim,
    frames=frames,
    interval=1000 / fps,   # en ms, cohérent avec le temps physique
    blit=False
)

plt.tight_layout()
print("\nSauvegarde du GIF...")
ani.save("real_controlled_animation_real_time.gif", writer="pillow", fps=fps)
print("GIF sauvegardé : real_controlled_animation_real_time.gif")
plt.show()