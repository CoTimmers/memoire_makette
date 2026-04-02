import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.interpolate import interp1d

# ══════════════════════════════════════════════════════════════════════
# PARAMÈTRES
# ══════════════════════════════════════════════════════════════════════
params_ref = {
    "m": 7.0,
    "a": 0.3,
    "b": 0.4,
    "T_wall": 6.0,
    "psi_sw_1": np.radians(127),
    "l_contact": 0.4,
}

params_real = {
    "m": 5.0,
    "a": 0.3,
    "b": 0.4,
    "T_wall": 6.0,
    "psi_sw_1": np.radians(127),
    "l_contact": 0.4,
}

params_ref["I_O"] = params_ref["m"] / 12.0 * (params_ref["a"]**2 + params_ref["b"]**2)
params_ref["I_A"] = params_ref["m"] /  3.0 * (params_ref["a"]**2 + params_ref["b"]**2)
params_real["I_O"] = 0.7 * params_ref["I_O"]
params_real["I_A"] = 0.7 * params_ref["I_A"]

for p in (params_ref, params_real):
    p["rAO_body"] = np.array([p["a"]/2, p["b"]/2])
    p["rAB_body"] = np.array([0.0, p["b"]])

T_SIM = 10.0
MODE_1 = 1
MODE_2 = 2
MODE_3 = 3

# ══════════════════════════════════════════════════════════════════════
# PROFIL DU MUR ψ(t)
# ══════════════════════════════════════════════════════════════════════
_T  = params_ref["T_wall"]
_t1 = _T * 0.4
_t2 = _T * 0.6
_acc = (np.pi/2) / (0.5*_t1**2 + (_t2 - _t1)*_t1 + 0.5*(_T - _t2)**2)
_v1  = _acc * _t1
_p1  = np.pi/2 + 0.5*_acc*_t1**2
_p2  = _p1 + _v1*(_t2 - _t1)

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
def phi_B_mode2(l, delta, psi, params):
    b = params["b"]
    return l * np.sin(psi) - b * np.cos(delta - psi)

def delta_kinematics(l, ldot, psi, psidot, psiddot, b):
    eps = 1e-9
    u = np.clip((l / b) * np.sin(psi), -1 + eps, 1 - eps)
    k  = 1.0 / np.sqrt(max(eps, 1 - u*u))
    k3 = k**3
    delta    = psi - 0.5*np.pi + np.arcsin(u)
    u_l      = np.sin(psi) / b
    u_psi    = (l/b) * np.cos(psi)
    u_lpsi   = np.cos(psi) / b
    u_psipsi = -(l/b) * np.sin(psi)
    delta_l      = k * u_l
    delta_psi    = 1 + k * u_psi
    delta_ll     = u * k3 * u_l * u_l
    delta_lpsi   = k * u_lpsi + u * k3 * u_l * u_psi
    delta_psipsi = k * u_psipsi + u * k3 * u_psi * u_psi
    delta_dot    = delta_l * ldot + delta_psi * psidot
    return delta, delta_dot, delta_l, delta_psi, delta_ll, delta_lpsi, delta_psipsi

def mode2a_core(t, x, params, fxfy_fun):
    l, l_dot = x
    m_p = params["m"]
    I_A = params["I_A"]
    a, b = params["a"], params["b"]
    psi, psi_dot, psi_ddot = psi_profile(t)
    fx, fy = fxfy_fun(t, x)
    delta, delta_dot, delta_l, delta_psi, delta_ll, delta_lpsi, delta_psipsi = \
        delta_kinematics(l, l_dot, psi, psi_dot, psi_ddot, b)
    c1 = -(a/2)*np.sin(delta) - (b/2)*np.cos(delta)
    c2 =  (a/2)*np.cos(delta) - (b/2)*np.sin(delta)
    c3 = -(a/2)*np.cos(delta) + (b/2)*np.sin(delta)
    A_kin = (delta_ll*l_dot**2 + 2*delta_lpsi*l_dot*psi_dot
             + delta_psipsi*psi_dot**2 + delta_psi*psi_ddot)
    denom = m_p * (1.0 + c1 * delta_l)
    l_ddot = ((fx - m_p*(c1*A_kin + c3*delta_dot**2)) / denom
              if abs(denom) > 1e-12 else 0.0)
    delta_ddot = delta_l * l_ddot + A_kin
    sin_dp = np.sin(delta - psi)
    denom_l = b * sin_dp
    lambda_B = ((m_p*c1*l_ddot + I_A*delta_ddot - c1*fx - c2*fy) / denom_l
                if abs(denom_l) > 1e-10 else 0.0)
    return l_ddot, lambda_B, delta, delta_dot

def impact_mode2_to_3(x_minus, params):
    l_minus, delta_minus, l_dot_minus, omega_minus = x_minus
    a, b = params["a"], params["b"]
    c1_minus = -(a/2)*np.sin(delta_minus) - (b/2)*np.cos(delta_minus)
    l_dot_plus = l_dot_minus + c1_minus * omega_minus
    return np.array([l_minus, l_dot_plus])

# ══════════════════════════════════════════════════════════════════════
#                     CONTRÔLEURS RESTRUCTURÉS
# ══════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────
#  MODE 1 — Rotation pure autour de A = (0,0)
# ─────────────────────────────────────────────────────────────────────
# Physique : A est le pivot FIXE en contact avec le mur 1.
# État réduit : [δ, ω]  →  I_A · ω̇ = τ_A
# A est exactement à (0,0) par construction → pas de pénétration.
#
# Structure :
#   τ_ff = I_A · α_ref               (feedforward inertiel)
#   τ_fb = −Kp·(δ−δ_ref) − Kd·(ω−ω_ref)   (PD pur, sans terme I_A·α_ref)
#   τ_A  = τ_ff + τ_fb

_KP_DEL1 = 20.0   # proportionnel δ − δ_ref
_KD_DEL1 =  6.0   # dérivé        ω − ω_ref


def reference_mode1(t, params):
    """
    Référence Mode 1.
    Retourne (δ_ref, ω_ref, α_ref).
    """
    psi, psi_dot, psi_ddot = psi_profile(t)
    delta_ref = psi - np.pi/2
    omega_ref = psi_dot
    alpha_ref = psi_ddot
    return delta_ref, omega_ref, alpha_ref


def feedforward_mode1(t, params):
    """
    Feedforward Mode 1.
    τ_ff = I_A · α_ref  — couple nécessaire pour accélérer le bac
    si le modèle est parfait. Aucun terme PD ici.
    """
    _, _, alpha_ref = reference_mode1(t, params)
    tau_ff = params["I_A"] * alpha_ref
    return tau_ff


def feedback_mode1(t, delta, omega, params):
    """
    Feedback Mode 1.
    τ_fb = −Kp·(δ−δ_ref) − Kd·(ω−ω_ref)
    Correction PD pure — pas de terme I_A·α_ref (déjà dans le FF).
    """
    delta_ref, omega_ref, _ = reference_mode1(t, params)
    e_delta = delta - delta_ref
    e_omega = omega - omega_ref
    tau_fb = -_KP_DEL1 * e_delta - _KD_DEL1 * e_omega
    return tau_fb


def control_mode1(t, delta, omega, params):
    """
    Commande totale Mode 1 : τ_A = τ_ff + τ_fb.
    τ_A = I_A·α_ref − Kp·(δ−δ_ref) − Kd·(ω−ω_ref)
    """
    tau_ff = feedforward_mode1(t, params)
    tau_fb = feedback_mode1(t, delta, omega, params)
    return tau_ff + tau_fb


def tau_to_force_mode1(t, tau_A, params):
    """
    Convertit τ_A en (Fx, Fy) pour affichage, dans la direction (−sinψ, cosψ).
    Bras de levier = b/2 = 0.2 m (constant, démontré numériquement).
    """
    psi, _, _ = psi_profile(t)
    levier = params["b"] / 2.0
    F_mag = tau_A / levier if abs(levier) > 1e-9 else 0.0
    return -F_mag * np.sin(psi), F_mag * np.cos(psi)


# ─────────────────────────────────────────────────────────────────────
#  MODE 2 — Glissement de A sur le mur 1
# ─────────────────────────────────────────────────────────────────────
# Commande unifiée : valide que B soit en contact (2A) ou libre (2B).

_l_ref_fun     = None
_ldot_ref_fun  = None
_delta_ref_fun = None
_omega_ref_fun = None

_KP_L2    = 10.0
_KD_L2    = 10.0
_KP_DEL2  = 10.0
_KD_DEL2  =  6.0

_BIAS_AC_ANGLE = np.radians(12.0)
_FF2_MAG       = 6.0


def reference_mode2(t):
    """
    Référence Mode 2 : trajectoire nominale interpolée.
    Retourne (l_ref, δ_ref, ḷ_ref, ω_ref).
    """
    return (float(_l_ref_fun(t)),
            float(_delta_ref_fun(t)),
            float(_ldot_ref_fun(t)),
            float(_omega_ref_fun(t)))


def feedforward_mode2(t, l, delta, params):
    """
    Feedforward Mode 2.

    Direction : approximativement de A vers C (point de contact sur le mur 2),
    avec un biais angulaire pour assurer un moment positif autour de A et une
    composante de translation le long du mur 1.

    On calcule rAC dans le repère monde (≈ rAO, car C ≈ O pour le FF),
    on inverse la direction (de C vers A, force qui pousse le bac dans le coin),
    puis on applique un biais angulaire ±_BIAS_AC_ANGLE.
    Le signe du biais est choisi automatiquement pour garantir :
      - moment autour de A positif (rotation CCW)
      - composante Fx positive (avancement de A)
    """
    a, b = params["a"], params["b"]
    cd, sd = np.cos(delta), np.sin(delta)
    R = np.array([[cd, -sd], [sd,  cd]])

    # Vecteur A→O dans le monde (≈ A→C)
    rAC_world = R @ params["rAO_body"]
    nAC = np.linalg.norm(rAC_world)
    if nAC < 1e-12:
        return 0.0, 0.0

    # Direction de C vers A (force qui pousse le bac vers le coin)
    uCA = -rAC_world / nAC

    def rotate2d(v, angle):
        c, s = np.cos(angle), np.sin(angle)
        return np.array([c*v[0] - s*v[1], s*v[0] + c*v[1]])

    # Essai avec biais négatif (rotation CW de la direction)
    u_bias = rotate2d(uCA, -_BIAS_AC_ANGLE)
    Fx_ff = _FF2_MAG * u_bias[0]
    Fy_ff = _FF2_MAG * u_bias[1]

    # Moment autour de A de cette force appliquée en O
    M_A = rAC_world[0] * Fy_ff - rAC_world[1] * Fx_ff

    # Si le moment est négatif OU Fx est négatif, on inverse le biais
    if M_A <= 0 or Fx_ff <= 0:
        u_bias = rotate2d(uCA, +_BIAS_AC_ANGLE)
        Fx_ff = _FF2_MAG * u_bias[0]
        Fy_ff = _FF2_MAG * u_bias[1]

    return Fx_ff, Fy_ff


def feedback_mode2(t, l, delta, ldot, omega):
    """
    Feedback Mode 2 (PD).
    Fx_fb sur l − l_ref, Fy_fb sur δ − δ_ref.
    """
    l_ref, delta_ref, ldot_ref, omega_ref = reference_mode2(t)
    Fx_fb = -_KP_L2 * (l - l_ref)     - _KD_L2  * (ldot  - ldot_ref)
    Fy_fb = -_KP_DEL2 * (delta - delta_ref) - _KD_DEL2 * (omega - omega_ref)
    return Fx_fb, Fy_fb


def control_mode2(t, l, delta, ldot, omega, params):
    """Commande totale Mode 2 : u = u_ff + u_fb."""
    Fx_ff, Fy_ff = feedforward_mode2(t, l, delta, params)
    Fx_fb, Fy_fb = feedback_mode2(t, l, delta, ldot, omega)
    return Fx_ff + Fx_fb, Fy_ff + Fy_fb


def control_mode2_nominal(t, l, delta, ldot, omega, params):
    """Commande Mode 2 nominale : feedforward uniquement (bootstrap)."""
    return feedforward_mode2(t, l, delta, params)


# ─────────────────────────────────────────────────────────────────────
#  MODE 3 — Régulation finale vers l*
# ─────────────────────────────────────────────────────────────────────

_l_star = 0.4
_KP_L3  = 20.0
_KD_L3  = 15.0


def reference_mode3():
    return _l_star


def feedforward_mode3():
    return 0.0, 0.0


def feedback_mode3(l, ldot):
    Fx_fb = -_KP_L3 * (l - _l_star) - _KD_L3 * ldot
    return Fx_fb, 0.0


def control_mode3(t, x):
    l, ldot = x
    Fx_ff, Fy_ff = feedforward_mode3()
    Fx_fb, Fy_fb = feedback_mode3(l, ldot)
    return Fx_ff + Fx_fb, Fy_ff + Fy_fb


# ══════════════════════════════════════════════════════════════════════
# LISSAGE DES FORCES — filtre passe-bas du 1er ordre
# ══════════════════════════════════════════════════════════════════════
# Pour éviter les sauts de force aux transitions de mode et à l'intérieur
# de chaque mode, on ajoute un état filtré à chaque dynamique :
#   F_filt_dot = (F_cmd − F_filt) / _TAU_F
#   F_appliquée = F_filt   (pas F_cmd directement)
#
# Aux transitions, F_filt est initialisé à la valeur courante de la force
# → la force repart de là où elle était, sans aucun saut.

# _TAU_F = 0.08   # constante de temps du filtre (s) — ~3τ = 0.24s pour atteindre 95%
_TAU_F = 0.15
# _TAU_F = 0.25
# _TAU_F = 0.50

# ══════════════════════════════════════════════════════════════════════
# DYNAMIQUES
# ══════════════════════════════════════════════════════════════════════

def dynamics_mode1(t, x, params):
    """
    Rotation pure autour de A.
    État augmenté : x = [δ, ω, τ_filt]
    τ_filt est le couple filtré actuellement appliqué.
    """
    delta, omega, tau_filt = x
    tau_cmd = control_mode1(t, delta, omega, params)
    tau_dot = (tau_cmd - tau_filt) / _TAU_F
    return [omega, tau_filt / params["I_A"], tau_dot]


def dynamics_mode2_free_B(t, x, params):
    """
    Mode 2 sans contrainte B.
    État augmenté : x = [l, δ, ḷ, ω, Fx_filt, Fy_filt]
    """
    l, delta, ldot, omega, Fx_filt, Fy_filt = x
    m, I_A = params["m"], params["I_A"]
    a, b = params["a"], params["b"]
    Fx_cmd, Fy_cmd = control_mode2(t, l, delta, ldot, omega, params)
    Fx_dot = (Fx_cmd - Fx_filt) / _TAU_F
    Fy_dot = (Fy_cmd - Fy_filt) / _TAU_F
    c1 = -(a/2)*np.sin(delta) - (b/2)*np.cos(delta)
    c2 =  (a/2)*np.cos(delta) - (b/2)*np.sin(delta)
    c3 = -(a/2)*np.cos(delta) + (b/2)*np.sin(delta)
    M_sys = np.array([[m, m*c1], [m*c1, I_A]])
    rhs   = np.array([Fx_filt - m*c3*omega**2, c1*Fx_filt + c2*Fy_filt])
    sol   = np.linalg.solve(M_sys, rhs)
    return [ldot, omega, sol[0], sol[1], Fx_dot, Fy_dot]


def dynamics_mode3(t, x, params):
    """
    Mode 3 : régulation de l.
    État augmenté : x = [l, ḷ, Fx_filt, Fy_filt]
    Fy cible = 0 (décroît vers 0 depuis la valeur héritée du Mode 2).
    """
    l, ldot, Fx_filt, Fy_filt = x
    Fx_cmd, _ = control_mode3(t, [l, ldot])
    Fx_dot = (Fx_cmd  - Fx_filt) / _TAU_F
    Fy_dot = (0.0     - Fy_filt) / _TAU_F   # Fy cible = 0 en Mode 3
    return [ldot, Fx_filt / params["m"], Fx_dot, Fy_dot]


# ══════════════════════════════════════════════════════════════════════
# SIMULATION NOMINALE
# ══════════════════════════════════════════════════════════════════════

def simulate_nominal(params):
    def _psi_target(t):
        return psi_profile(t)[0] - params["psi_sw_1"]
    t1_end = brentq(_psi_target, 0.0, params["T_wall"])
    t1_arr = np.linspace(0.0, t1_end, 300)

    psi1    = np.array([psi_profile(ti)[0] for ti in t1_arr])
    psidot1 = np.array([psi_profile(ti)[1] for ti in t1_arr])
    delta1  = psi1 - np.pi/2
    omega1  = psidot1
    l1      = np.zeros_like(t1_arr)
    ldot1   = np.zeros_like(t1_arr)
    fx1     = np.array([feedforward_mode2(ti, 0.0, delta1[i], params)[0]
                        for i, ti in enumerate(t1_arr)])
    fy1     = np.array([feedforward_mode2(ti, 0.0, delta1[i], params)[1]
                        for i, ti in enumerate(t1_arr)])
    mode1_arr = np.full(len(t1_arr), MODE_1, dtype=int)

    t_segs, l_segs, ldot_segs, delta_segs = [], [], [], []
    omega_segs, psi_segs, fx_segs, fy_segs, mode_segs = [], [], [], [], []

    sub_mode = "2A"
    t_cur = t1_end
    x_cur = np.array([0.0, 0.0])

    while t_cur < T_SIM:
        if sub_mode == "2A":
            psi_cur, _, _ = psi_profile(t_cur)
            b = params["b"]; eps = 1e-9
            u_c = np.clip((x_cur[0]/b)*np.sin(psi_cur), -1+eps, 1-eps)
            delta_cur = psi_cur - np.pi/2 + np.arcsin(u_c)
            omega_cur = 0.0

            fxfy_check = lambda tt, xx: control_mode2_nominal(
                tt, xx[0], delta_cur, xx[1], omega_cur, params)
            _, lB_init, delta_init, omega_init = mode2a_core(
                t_cur, x_cur, params, fxfy_check)
            if lB_init <= 0.0:
                x_cur = np.array([x_cur[0], delta_init, x_cur[1], omega_init])
                sub_mode = "2B"; continue

            def _dyn2a(t, x):
                psi_t, psi_dot_t, _ = psi_profile(t)
                b_p = params["b"]; eps = 1e-9
                u_ = np.clip((x[0]/b_p)*np.sin(psi_t), -1+eps, 1-eps)
                delta_ = psi_t - np.pi/2 + np.arcsin(u_)
                k_ = 1/np.sqrt(max(eps, 1-u_**2))
                delta_l_ = k_*np.sin(psi_t)/b_p
                delta_psi_ = 1 + k_*(x[0]/b_p)*np.cos(psi_t)
                omega_ = delta_l_*x[1] + delta_psi_*psi_dot_t
                fxfy = lambda tt, xx: control_mode2_nominal(
                    tt, xx[0], delta_, xx[1], omega_, params)
                l_dd, _, _, _ = mode2a_core(t, x, params, fxfy)
                return [x[1], l_dd]

            def _ev_2_to_3(t, x): return x[0] - params["l_contact"]
            _ev_2_to_3.terminal = True; _ev_2_to_3.direction = 1

            def _ev_sep(t, x):
                psi_t, psi_dot_t, _ = psi_profile(t)
                b_p = params["b"]; eps = 1e-9
                u_ = np.clip((x[0]/b_p)*np.sin(psi_t), -1+eps, 1-eps)
                delta_ = psi_t - np.pi/2 + np.arcsin(u_)
                k_ = 1/np.sqrt(max(eps, 1-u_**2))
                delta_l_ = k_*np.sin(psi_t)/b_p
                delta_psi_ = 1 + k_*(x[0]/b_p)*np.cos(psi_t)
                omega_ = delta_l_*x[1] + psi_dot_t*delta_psi_
                fxfy = lambda tt, xx: control_mode2_nominal(
                    tt, xx[0], delta_, xx[1], omega_, params)
                _, lB, _, _ = mode2a_core(t, x, params, fxfy)
                return lB
            _ev_sep.terminal = True; _ev_sep.direction = -1

            sol = solve_ivp(_dyn2a, (t_cur, T_SIM), x_cur,
                            dense_output=True, events=[_ev_2_to_3, _ev_sep],
                            rtol=1e-9, atol=1e-11)
            t_end = float(sol.t[-1])
            t_seg = np.linspace(t_cur, t_end,
                                max(3, int(300*(t_end-t_cur)/T_SIM)+3))
            y_seg = sol.sol(t_seg)
            l_seg = y_seg[0]; ld_seg = y_seg[1]

            delta_seg = np.zeros_like(t_seg); omega_seg = np.zeros_like(t_seg)
            fx_seg = np.zeros_like(t_seg);    fy_seg   = np.zeros_like(t_seg)
            psi_seg = np.array([psi_profile(tt)[0] for tt in t_seg])
            for i, tt in enumerate(t_seg):
                psi_t, psi_dot_t, _ = psi_profile(tt)
                b_p = params["b"]; eps = 1e-9
                u_ = np.clip((l_seg[i]/b_p)*np.sin(psi_t), -1+eps, 1-eps)
                delta_seg[i] = psi_t - np.pi/2 + np.arcsin(u_)
                k_ = 1/np.sqrt(max(eps, 1-u_**2))
                delta_l_ = k_*np.sin(psi_t)/b_p
                delta_psi_ = 1 + k_*(l_seg[i]/b_p)*np.cos(psi_t)
                omega_seg[i] = delta_l_*ld_seg[i] + delta_psi_*psi_dot_t
                fx_seg[i], fy_seg[i] = control_mode2_nominal(
                    tt, l_seg[i], delta_seg[i], ld_seg[i], omega_seg[i], params)

            t_segs.append(t_seg);     l_segs.append(l_seg)
            ldot_segs.append(ld_seg); delta_segs.append(delta_seg)
            omega_segs.append(omega_seg); psi_segs.append(psi_seg)
            fx_segs.append(fx_seg);   fy_segs.append(fy_seg)
            mode_segs.append(np.full(len(t_seg), MODE_2, dtype=int))

            if len(sol.t_events[0]) > 0:
                x_minus = np.array([l_seg[-1], delta_seg[-1],
                                    ld_seg[-1], omega_seg[-1]])
                x_cur = impact_mode2_to_3(x_minus, params)
                t_cur = t_end; sub_mode = "3"; break
            elif len(sol.t_events[1]) > 0:
                x_cur = np.array([l_seg[-1], delta_seg[-1],
                                  ld_seg[-1], omega_seg[-1]])
                t_cur = t_end; sub_mode = "2B"
            else:
                t_cur = T_SIM

        elif sub_mode == "2B":
            def _dyn2b(t, x):
                l_, delta_, ldot_, omega_ = x
                m_p, I_A = params["m"], params["I_A"]
                a_, b_ = params["a"], params["b"]
                Fx_, Fy_ = control_mode2_nominal(
                    t, l_, delta_, ldot_, omega_, params)
                c1_ = -(a_/2)*np.sin(delta_) - (b_/2)*np.cos(delta_)
                c2_ =  (a_/2)*np.cos(delta_) - (b_/2)*np.sin(delta_)
                c3_ = -(a_/2)*np.cos(delta_) + (b_/2)*np.sin(delta_)
                M_ = np.array([[m_p, m_p*c1_], [m_p*c1_, I_A]])
                r_ = np.array([Fx_ - m_p*c3_*omega_**2, c1_*Fx_ + c2_*Fy_])
                s_ = np.linalg.solve(M_, r_)
                return [ldot_, omega_, s_[0], s_[1]]

            def _ev_2_to_3b(t, x): return x[0] - params["l_contact"]
            _ev_2_to_3b.terminal = True; _ev_2_to_3b.direction = 1

            def _ev_recontact(t, x):
                l, delta, ldot, omega = x
                psi_t, _, _ = psi_profile(t)
                return phi_B_mode2(l, delta, psi_t, params)
            _ev_recontact.terminal = True; _ev_recontact.direction = -1

            sol = solve_ivp(_dyn2b, (t_cur, T_SIM), x_cur,
                            dense_output=True, events=[_ev_2_to_3b, _ev_recontact],
                            rtol=1e-9, atol=1e-11)
            t_end = float(sol.t[-1])
            t_seg = np.linspace(t_cur, t_end,
                                max(3, int(300*(t_end-t_cur)/T_SIM)+3))
            y_seg = sol.sol(t_seg)
            l_seg = y_seg[0]; delta_seg = y_seg[1]
            ld_seg = y_seg[2]; omega_seg = y_seg[3]
            psi_seg = np.array([psi_profile(tt)[0] for tt in t_seg])
            fx_seg = np.array([
                control_mode2_nominal(tt, l_seg[i], delta_seg[i],
                                      ld_seg[i], omega_seg[i], params)[0]
                for i, tt in enumerate(t_seg)])
            fy_seg = np.array([
                control_mode2_nominal(tt, l_seg[i], delta_seg[i],
                                      ld_seg[i], omega_seg[i], params)[1]
                for i, tt in enumerate(t_seg)])

            t_segs.append(t_seg);     l_segs.append(l_seg)
            ldot_segs.append(ld_seg); delta_segs.append(delta_seg)
            omega_segs.append(omega_seg); psi_segs.append(psi_seg)
            fx_segs.append(fx_seg);   fy_segs.append(fy_seg)
            mode_segs.append(np.full(len(t_seg), MODE_2, dtype=int))

            if len(sol.t_events[0]) > 0:
                x_minus = np.array([l_seg[-1], delta_seg[-1],
                                    ld_seg[-1], omega_seg[-1]])
                x_cur = impact_mode2_to_3(x_minus, params)
                t_cur = t_end; sub_mode = "3"; break
            elif len(sol.t_events[1]) > 0:
                x_cur = np.array([l_seg[-1], ld_seg[-1]])
                t_cur = t_end; sub_mode = "2A"
            else:
                t_cur = T_SIM

    if sub_mode == "3" and t_cur < T_SIM:
        # Nominale : dynamique Mode 3 sans filtre (2 états)
        def _dyn3_nom(t, x):
            l_, ld_ = x
            Fx_, _ = control_mode3(t, [l_, ld_])
            return [ld_, Fx_ / params["m"]]
        sol3 = solve_ivp(_dyn3_nom, (t_cur, T_SIM), x_cur,
                         dense_output=True, rtol=1e-9, atol=1e-11)
        t3 = np.linspace(t_cur, T_SIM, 300)
        y3 = sol3.sol(t3)
        l3 = y3[0]; ld3 = y3[1]
        delta3 = np.full(len(t3), np.pi/2); omega3 = np.zeros(len(t3))
        psi3 = np.array([psi_profile(tt)[0] for tt in t3])
        fx3 = np.array([control_mode3(tt, [l3[i], ld3[i]])[0]
                        for i, tt in enumerate(t3)])
        fy3 = np.zeros(len(t3))
        t_segs.append(t3);      l_segs.append(l3)
        ldot_segs.append(ld3);  delta_segs.append(delta3)
        omega_segs.append(omega3); psi_segs.append(psi3)
        fx_segs.append(fx3);    fy_segs.append(fy3)
        mode_segs.append(np.full(len(t3), MODE_3, dtype=int))

    def _cat(lst): return np.concatenate(lst) if lst else np.array([])
    return {
        "t":     np.concatenate([t1_arr,    _cat(t_segs)]),
        "l":     np.concatenate([l1,        _cat(l_segs)]),
        "ldot":  np.concatenate([ldot1,     _cat(ldot_segs)]),
        "delta": np.concatenate([delta1,    _cat(delta_segs)]),
        "omega": np.concatenate([omega1,    _cat(omega_segs)]),
        "psi":   np.concatenate([psi1,      _cat(psi_segs)]),
        "fx":    np.concatenate([fx1,       _cat(fx_segs)]),
        "fy":    np.concatenate([fy1,       _cat(fy_segs)]),
        "mode":  np.concatenate([mode1_arr, _cat(mode_segs)]),
        "t1_end": t1_end,
    }


# ══════════════════════════════════════════════════════════════════════
# RÉFÉRENCE NOMINALE
# ══════════════════════════════════════════════════════════════════════
print("Simulation nominale en cours...")
ref = simulate_nominal(params_ref)

t_ref = ref["t"]
_l_ref_fun     = interp1d(t_ref, ref["l"],     kind="linear", fill_value="extrapolate")
_ldot_ref_fun  = interp1d(t_ref, ref["ldot"],  kind="linear", fill_value="extrapolate")
_delta_ref_fun = interp1d(t_ref, ref["delta"], kind="linear", fill_value="extrapolate")
_omega_ref_fun = interp1d(t_ref, ref["omega"], kind="linear", fill_value="extrapolate")
print(f"  Simulation nominale terminée. t1_end = {ref['t1_end']:.3f} s")


# ══════════════════════════════════════════════════════════════════════
# SIMULATION RÉELLE AVEC FEEDBACK
# ══════════════════════════════════════════════════════════════════════

def simulate_real(params):
    # ── Mode 1 : rotation pure autour de A = (0,0) ──
    # État augmenté : [δ, ω, τ_filt]
    def ev_1_to_2(t, x):
        psi, _, _ = psi_profile(t)
        return psi - params["psi_sw_1"]
    ev_1_to_2.terminal = True; ev_1_to_2.direction = 1

    psi0, pd0, _ = psi_profile(0.0)
    delta0 = psi0 - np.pi/2
    tau0   = control_mode1(0.0, delta0, pd0, params)   # forcer la CI du filtre = commande initiale
    x0_1   = np.array([delta0, pd0, tau0])

    sol1 = solve_ivp(lambda t, x: dynamics_mode1(t, x, params),
                     (0.0, T_SIM), x0_1, dense_output=True,
                     events=[ev_1_to_2], rtol=1e-9, atol=1e-11)
    t1_end = float(sol1.t[-1])
    t1  = np.linspace(0.0, t1_end, 300)
    y1  = sol1.sol(t1)

    delta1    = y1[0]
    omega1    = y1[1]
    tau_filt1 = y1[2]   # couple filtré effectivement appliqué
    psi1      = np.array([psi_profile(tt)[0] for tt in t1])

    a, b     = params["a"], params["b"]
    rAO_body = params["rAO_body"]
    xA1 = np.zeros(len(t1)); yA1 = np.zeros(len(t1))
    xAdot1 = np.zeros(len(t1)); yAdot1 = np.zeros(len(t1))
    xO1 = np.zeros(len(t1)); yO1 = np.zeros(len(t1))
    xOdot1 = np.zeros(len(t1)); yOdot1 = np.zeros(len(t1))
    fx1 = np.zeros(len(t1)); fy1 = np.zeros(len(t1))

    for i, tt in enumerate(t1):
        cd = np.cos(delta1[i]); sd = np.sin(delta1[i])
        R_i   = np.array([[cd, -sd], [sd, cd]])
        rAO_w = R_i @ rAO_body
        xO1[i] = rAO_w[0]; yO1[i] = rAO_w[1]
        xOdot1[i] = -omega1[i] * rAO_w[1]
        yOdot1[i] =  omega1[i] * rAO_w[0]
        # Force affichée = couple filtré réellement appliqué
        fx1[i], fy1[i] = tau_to_force_mode1(tt, tau_filt1[i], params)

    # ── Mode 2 réel (unifié) ──
    # État augmenté 2A : [l, ḷ, Fx_filt, Fy_filt]
    # État augmenté 2B : [l, δ, ḷ, ω, Fx_filt, Fy_filt]
    t_segs, l_segs, ldot_segs, delta_segs = [], [], [], []
    omega_segs, psi_segs, fx_segs, fy_segs, mode_segs = [], [], [], [], []

    sub_mode = "2A"
    t_cur    = t1_end
    # CI forces filtrées = force de fin de Mode 1, convertie en (Fx, Fy)
    Fx_filt0, Fy_filt0 = fx1[-1], fy1[-1]
    x_cur = np.array([0.0, 0.0, Fx_filt0, Fy_filt0])  # [l, ḷ, Fx_filt, Fy_filt]

    while t_cur < T_SIM:
        if sub_mode == "2A":
            # Extraire les états mécaniques et les états filtrés
            l_c, ld_c, Fxf_c, Fyf_c = x_cur

            psi_cur, psi_dot_cur, _ = psi_profile(t_cur)
            b_p = params["b"]; eps = 1e-9
            u_c = np.clip((l_c/b_p)*np.sin(psi_cur), -1+eps, 1-eps)
            delta_cur = psi_cur - np.pi/2 + np.arcsin(u_c)
            k_c = 1/np.sqrt(max(eps, 1-u_c**2))
            delta_l_c   = k_c*np.sin(psi_cur)/b_p
            delta_psi_c = 1 + k_c*(l_c/b_p)*np.cos(psi_cur)
            omega_cur   = delta_l_c*ld_c + delta_psi_c*psi_dot_cur

            # Vérification admissibilité lambda_B avec force filtrée actuelle
            fxfy_chk = lambda tt, xx: (Fxf_c, Fyf_c)
            _, lB_init, delta_init, omega_init = mode2a_core(
                t_cur, np.array([l_c, ld_c]), params, fxfy_chk)
            if lB_init <= 0.0:
                x_cur = np.array([l_c, delta_init, ld_c, omega_init,
                                  Fxf_c, Fyf_c])
                sub_mode = "2B"; continue

            def _dyn2a_real(t, x):
                l_, ld_, Fxf_, Fyf_ = x
                psi_t, psi_dot_t, _ = psi_profile(t)
                b_p = params["b"]; eps = 1e-9
                u_  = np.clip((l_/b_p)*np.sin(psi_t), -1+eps, 1-eps)
                d_  = psi_t - np.pi/2 + np.arcsin(u_)
                k_  = 1/np.sqrt(max(eps, 1-u_**2))
                dl_ = k_*np.sin(psi_t)/b_p
                dp_ = 1 + k_*(l_/b_p)*np.cos(psi_t)
                w_  = dl_*ld_ + dp_*psi_dot_t
                # Force appliquée = filtrée
                fxfy_filt = lambda tt, xx: (Fxf_, Fyf_)
                l_dd, _, _, _ = mode2a_core(t, np.array([l_, ld_]),
                                            params, fxfy_filt)
                # Commande cible pour le filtre
                Fx_cmd, Fy_cmd = control_mode2(t, l_, d_, ld_, w_, params)
                Fx_dot = (Fx_cmd - Fxf_) / _TAU_F
                Fy_dot = (Fy_cmd - Fyf_) / _TAU_F
                return [ld_, l_dd, Fx_dot, Fy_dot]

            def _ev_2_to_3r(t, x): return x[0] - params["l_contact"]
            _ev_2_to_3r.terminal = True; _ev_2_to_3r.direction = 1

            def _ev_sep_r(t, x):
                l_, ld_, Fxf_, Fyf_ = x
                psi_t, psi_dot_t, _ = psi_profile(t)
                b_p = params["b"]; eps = 1e-9
                u_  = np.clip((l_/b_p)*np.sin(psi_t), -1+eps, 1-eps)
                d_  = psi_t - np.pi/2 + np.arcsin(u_)
                fxfy_filt = lambda tt, xx: (Fxf_, Fyf_)
                _, lB, _, _ = mode2a_core(t, np.array([l_, ld_]),
                                          params, fxfy_filt)
                return lB
            _ev_sep_r.terminal = True; _ev_sep_r.direction = -1

            sol = solve_ivp(_dyn2a_real, (t_cur, T_SIM), x_cur,
                            dense_output=True,
                            events=[_ev_2_to_3r, _ev_sep_r],
                            rtol=1e-9, atol=1e-11)
            t_end = float(sol.t[-1])
            t_seg = np.linspace(t_cur, t_end,
                                max(3, int(300*(t_end-t_cur)/T_SIM)+3))
            y_seg = sol.sol(t_seg)
            l_seg  = y_seg[0]; ld_seg  = y_seg[1]
            Fxf_seg = y_seg[2]; Fyf_seg = y_seg[3]   # forces filtrées

            psi_seg   = np.array([psi_profile(tt)[0] for tt in t_seg])
            delta_seg = np.zeros_like(t_seg); omega_seg = np.zeros_like(t_seg)
            for i, tt in enumerate(t_seg):
                psi_t, psi_dot_t, _ = psi_profile(tt)
                b_p = params["b"]; eps = 1e-9
                u_  = np.clip((l_seg[i]/b_p)*np.sin(psi_t), -1+eps, 1-eps)
                delta_seg[i] = psi_t - np.pi/2 + np.arcsin(u_)
                k_  = 1/np.sqrt(max(eps, 1-u_**2))
                dl_ = k_*np.sin(psi_t)/b_p
                dp_ = 1 + k_*(l_seg[i]/b_p)*np.cos(psi_t)
                omega_seg[i] = dl_*ld_seg[i] + dp_*psi_dot_t

            t_segs.append(t_seg);      l_segs.append(l_seg)
            ldot_segs.append(ld_seg);  delta_segs.append(delta_seg)
            omega_segs.append(omega_seg); psi_segs.append(psi_seg)
            fx_segs.append(Fxf_seg);   fy_segs.append(Fyf_seg)
            mode_segs.append(np.full(len(t_seg), MODE_2, dtype=int))

            if len(sol.t_events[0]) > 0:
                x_minus = np.array([l_seg[-1], delta_seg[-1],
                                    ld_seg[-1], omega_seg[-1]])
                l_plus, ld_plus = impact_mode2_to_3(x_minus, params)
                # Passer les forces filtrées à Mode 3 → pas de saut
                t_cur  = t_end; sub_mode = "3"
                x_cur  = np.array([l_plus, ld_plus,
                                   Fxf_seg[-1], Fyf_seg[-1]])
                break
            elif len(sol.t_events[1]) > 0:
                x_cur = np.array([l_seg[-1], delta_seg[-1],
                                  ld_seg[-1], omega_seg[-1],
                                  Fxf_seg[-1], Fyf_seg[-1]])
                t_cur = t_end; sub_mode = "2B"
            else:
                t_cur = T_SIM

        elif sub_mode == "2B":
            # État augmenté : [l, δ, ḷ, ω, Fx_filt, Fy_filt]
            def _dyn2b_real(t, x):
                l_, d_, ld_, w_, Fxf_, Fyf_ = x
                m, I_A = params["m"], params["I_A"]
                a, b   = params["a"], params["b"]
                # Dynamique mécanique avec force filtrée
                c1 = -(a/2)*np.sin(d_) - (b/2)*np.cos(d_)
                c2 =  (a/2)*np.cos(d_) - (b/2)*np.sin(d_)
                c3 = -(a/2)*np.cos(d_) + (b/2)*np.sin(d_)
                M_ = np.array([[m, m*c1], [m*c1, I_A]])
                r_ = np.array([Fxf_ - m*c3*w_**2, c1*Fxf_ + c2*Fyf_])
                s_ = np.linalg.solve(M_, r_)
                # Dynamique du filtre
                Fx_cmd, Fy_cmd = control_mode2(t, l_, d_, ld_, w_, params)
                return [ld_, w_, s_[0], s_[1],
                        (Fx_cmd - Fxf_) / _TAU_F,
                        (Fy_cmd - Fyf_) / _TAU_F]

            def _ev_2_to_3b_r(t, x): return x[0] - params["l_contact"]
            _ev_2_to_3b_r.terminal = True; _ev_2_to_3b_r.direction = 1

            def _ev_recontact_r(t, x):
                l, d, ld, w, Fxf, Fyf = x
                psi_t, _, _ = psi_profile(t)
                return phi_B_mode2(l, d, psi_t, params)
            _ev_recontact_r.terminal = True; _ev_recontact_r.direction = -1

            sol = solve_ivp(_dyn2b_real, (t_cur, T_SIM), x_cur,
                            dense_output=True,
                            events=[_ev_2_to_3b_r, _ev_recontact_r],
                            rtol=1e-9, atol=1e-11)
            t_end = float(sol.t[-1])
            t_seg = np.linspace(t_cur, t_end,
                                max(3, int(300*(t_end-t_cur)/T_SIM)+3))
            y_seg = sol.sol(t_seg)
            l_seg   = y_seg[0]; delta_seg = y_seg[1]
            ld_seg  = y_seg[2]; omega_seg = y_seg[3]
            Fxf_seg = y_seg[4]; Fyf_seg   = y_seg[5]
            psi_seg = np.array([psi_profile(tt)[0] for tt in t_seg])

            t_segs.append(t_seg);      l_segs.append(l_seg)
            ldot_segs.append(ld_seg);  delta_segs.append(delta_seg)
            omega_segs.append(omega_seg); psi_segs.append(psi_seg)
            fx_segs.append(Fxf_seg);   fy_segs.append(Fyf_seg)
            mode_segs.append(np.full(len(t_seg), MODE_2, dtype=int))

            if len(sol.t_events[0]) > 0:
                x_minus = np.array([l_seg[-1], delta_seg[-1],
                                    ld_seg[-1], omega_seg[-1]])
                l_plus, ld_plus = impact_mode2_to_3(x_minus, params)
                t_cur   = t_end; sub_mode = "3"
                x_cur   = np.array([l_plus, ld_plus,
                                    Fxf_seg[-1], Fyf_seg[-1]])
                break
            elif len(sol.t_events[1]) > 0:
                x_cur = np.array([l_seg[-1], ld_seg[-1],
                                  Fxf_seg[-1], Fyf_seg[-1]])
                t_cur = t_end; sub_mode = "2A"
            else:
                t_cur = T_SIM

    if sub_mode == "3" and t_cur < T_SIM:
        # État augmenté : [l, ḷ, Fx_filt, Fy_filt]
        def _dyn3_real(t, x):
            return dynamics_mode3(t, x, params)

        sol3 = solve_ivp(_dyn3_real, (t_cur, T_SIM), x_cur,
                         dense_output=True, rtol=1e-9, atol=1e-11)
        t3  = np.linspace(t_cur, T_SIM, 300)
        y3  = sol3.sol(t3)
        l3  = y3[0]; ld3 = y3[1]; Fxf3 = y3[2]; Fyf3 = y3[3]
        delta3 = np.full(len(t3), np.pi/2); omega3 = np.zeros(len(t3))
        psi3   = np.array([psi_profile(tt)[0] for tt in t3])

        t_segs.append(t3);      l_segs.append(l3)
        ldot_segs.append(ld3);  delta_segs.append(delta3)
        omega_segs.append(omega3); psi_segs.append(psi3)
        fx_segs.append(Fxf3);   fy_segs.append(Fyf3)
        mode_segs.append(np.full(len(t3), MODE_3, dtype=int))

    def _cat(lst): return np.concatenate(lst) if lst else np.array([])
    return {
        "t":     np.concatenate([t1,                   _cat(t_segs)]),
        "l":     np.concatenate([np.zeros_like(t1),    _cat(l_segs)]),
        "ldot":  np.concatenate([np.zeros_like(t1),    _cat(ldot_segs)]),
        "delta": np.concatenate([delta1,               _cat(delta_segs)]),
        "omega": np.concatenate([omega1,               _cat(omega_segs)]),
        "psi":   np.concatenate([psi1,                 _cat(psi_segs)]),
        "fx":    np.concatenate([fx1,                  _cat(fx_segs)]),
        "fy":    np.concatenate([fy1,                  _cat(fy_segs)]),
        "mode":  np.concatenate([np.full(len(t1), MODE_1, dtype=int),
                                 _cat(mode_segs)]),
        "xA_mode1": xA1, "yA_mode1": yA1,
        "xO_mode1": xO1, "yO_mode1": yO1,
        "xAdot_mode1": xAdot1, "yAdot_mode1": yAdot1,
        "t_mode1": t1,
    }


# ══════════════════════════════════════════════════════════════════════
# EXÉCUTION
# ══════════════════════════════════════════════════════════════════════
print("Simulation réelle avec feedback en cours...")
real = simulate_real(params_real)
print(f"  Simulation réelle terminée. Modes détectés : {np.unique(real['mode'])}")

# ══════════════════════════════════════════════════════════════════════
# AFFICHAGES
# ══════════════════════════════════════════════════════════════════════
MODE_COLOR = {MODE_1: "steelblue", MODE_2: "seagreen", MODE_3: "darkorange"}
MODE_LABEL = {MODE_1: "Mode 1", MODE_2: "Mode 2", MODE_3: "Mode 3"}

fig, axes = plt.subplots(3, 2, figsize=(13, 11))
fig.suptitle("Contrôleur restructuré : FF + FB par mode\n"
             f"(nominal m={params_ref['m']} kg / réel m={params_real['m']} kg)",
             fontsize=12, fontweight="bold")

ax = axes[0, 0]
ax.plot(ref["t"],  ref["l"],  "k-",  lw=1.5, label="l réf (nominal)")
ax.plot(real["t"], real["l"], "b--", lw=1.5, label="l réel")
ax.axhline(_l_star, color="darkorange", ls=":", lw=1.5, label=f"l* = {_l_star} m")
ax.set_xlabel("t (s)"); ax.set_ylabel("l (m)"); ax.set_title("Suivi de l")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[0, 1]
ax.plot(ref["t"],  np.degrees(ref["delta"]),  "k-",  lw=1.5, label="δ réf")
ax.plot(real["t"], np.degrees(real["delta"]), "b--", lw=1.5, label="δ réel")
ax.axhline(90, color="darkorange", ls=":", lw=1.5, label="δ* = 90°")
ax.set_xlabel("t (s)"); ax.set_ylabel("δ (deg)")
ax.set_title("Suivi de l'orientation"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1, 0]
ax.plot(real["t_mode1"],
        np.degrees(np.array([psi_profile(tt)[0] - np.pi/2
                              for tt in real["t_mode1"]])),
        "k-", lw=1.5, label="δ_ref = ψ−90°")
ax.plot(real["t_mode1"],
        np.degrees(real["delta"][:len(real["t_mode1"])]),
        "b--", lw=1.5, label="δ réel")
ax.set_xlabel("t (s)"); ax.set_ylabel("δ (deg)")
ax.set_title("Mode 1 — suivi de l'orientation\n(A = (0,0) exact par construction)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1, 1]
ax.plot(real["t_mode1"], real["xO_mode1"], label="xO réel")
ax.plot(real["t_mode1"], real["yO_mode1"], label="yO réel")
ax.set_xlabel("t (s)"); ax.set_ylabel("pos COM (m)")
ax.set_title("Mode 1 — trajectoire du COM")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[2, 0]
ax.plot(ref["t"],  ref["fx"],  "k-",   lw=1,   label="Fx réf")
ax.plot(real["t"], real["fx"], "b--",  lw=1.5, label="Fx réel")
ax.plot(ref["t"],  ref["fy"],  "gray", lw=1,   ls="-", label="Fy réf")
ax.plot(real["t"], real["fy"], "c--",  lw=1.5, label="Fy réel")
ax.set_xlabel("t (s)"); ax.set_ylabel("Force (N)")
ax.set_title("Forces de commande"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[2, 1]
t_common = np.linspace(0, T_SIM, 2000)
l_ref_i  = interp1d(ref["t"],  ref["l"],  kind="linear",
                    fill_value="extrapolate")(t_common)
l_real_i = interp1d(real["t"], real["l"], kind="linear",
                    fill_value="extrapolate")(t_common)
ax.plot(t_common, l_real_i - l_ref_i, "b-", lw=1.5,
        label="erreur l = l_réel − l_réf")
ax.axhline(0, color="k", ls=":", lw=1)
ax.set_xlabel("t (s)"); ax.set_ylabel("Δl (m)")
ax.set_title("Erreur de suivi sur l"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("controller_redesign_plots.png", dpi=130)
plt.show()

print("\n─── Résultats ───")
print(f"Masse nominale = {params_ref['m']:.1f} kg  |  Masse réelle = {params_real['m']:.1f} kg")
print(f"I_A nominal    = {params_ref['I_A']:.5f} kg·m²  |  I_A réel = {params_real['I_A']:.5f} kg·m²")
print(f"t1_end nominal = {ref['t1_end']:.3f} s")
print(f"l final réel   = {real['l'][-1]:.4f} m  (cible l* = {_l_star})")
print(f"δ final réel   = {np.degrees(real['delta'][-1]):.2f}°  (cible 90°)")
print("Graphiques sauvegardés.")

# ══════════════════════════════════════════════════════════════════════
# ANIMATION
# ══════════════════════════════════════════════════════════════════════
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches

print("\nPréparation de l'animation...")

t_all     = real["t"];    l_all     = real["l"]
delta_all = real["delta"]; psi_all  = real["psi"]
mode_all  = real["mode"]; fx_all   = real["fx"]; fy_all = real["fy"]

_a, _b = params_real["a"], params_real["b"]
_rAO   = params_real["rAO_body"]
_rAB   = params_real["rAB_body"]
_n1    = len(real["t_mode1"])

xA_all = np.zeros_like(t_all); yA_all = np.zeros_like(t_all)
xO_all = np.zeros_like(t_all); yO_all = np.zeros_like(t_all)
xB_all = np.zeros_like(t_all); yB_all = np.zeros_like(t_all)

for _i in range(len(t_all)):
    _cd, _sd = np.cos(delta_all[_i]), np.sin(delta_all[_i])
    _R = np.array([[_cd, -_sd], [_sd, _cd]])
    _rAO_w = _R @ _rAO; _rAB_w = _R @ _rAB
    if mode_all[_i] == MODE_1 and _i < _n1:
        xA_all[_i] = real["xA_mode1"][_i]; yA_all[_i] = real["yA_mode1"][_i]
        xO_all[_i] = real["xO_mode1"][_i]; yO_all[_i] = real["yO_mode1"][_i]
    else:
        xA_all[_i] = l_all[_i];           yA_all[_i] = 0.0
        xO_all[_i] = xA_all[_i] + _rAO_w[0]
        yO_all[_i] = yA_all[_i] + _rAO_w[1]
    xB_all[_i] = xA_all[_i] + (_R @ _rAB)[0]
    yB_all[_i] = yA_all[_i] + (_R @ _rAB)[1]

_MC = {MODE_1: "#2196F3", MODE_2: "#4CAF50", MODE_3: "#FF9800"}
_ML = {MODE_1: "Mode 1 — rotation libre",
       MODE_2: "Mode 2 — glissement",
       MODE_3: "Mode 3 — régulation"}

_fps = 20
_t_anim = np.arange(t_all[0], t_all[-1], 1.0/_fps)
_frames = np.unique(np.clip(
    np.searchsorted(t_all, _t_anim, side="left"), 0, len(t_all)-1))

_fig, _ax = plt.subplots(figsize=(7, 7))
_ax.set_xlim(-0.55, 0.65); _ax.set_ylim(-0.15, 0.60)
_ax.set_aspect("equal"); _ax.grid(True, alpha=0.25)
_ax.set_xlabel("x (m)", fontsize=11); _ax.set_ylabel("y (m)", fontsize=11)
_ax.set_title("Simulation réelle contrôlée\n(FF + FB par mode)", fontsize=11)
_ax.plot([-0.5, 0.65], [0, 0], color="black", lw=3, zorder=3)
_ax.text(0.55, -0.06, "mur 1", fontsize=8, color="black")
_ax.axvline(_l_star, color="#FF9800", lw=1.5, ls="--", alpha=0.7)
_ax.text(_l_star+0.02, 0.54, f"l*={_l_star} m", fontsize=8, color="#FF9800")
_ax.plot(0, 0, "k+", ms=10, zorder=5)

_wall2,  = _ax.plot([], [], color="dimgray",  lw=3,   zorder=3)
_body,   = _ax.plot([], [], lw=2.5, zorder=4)
_traj,   = _ax.plot([], [], color="#1565C0",  lw=1,   alpha=0.35, zorder=2)
_pt_A,   = _ax.plot([], [], "o", color="#E53935", ms=9, zorder=6, label="A")
_pt_B,   = _ax.plot([], [], "s", color="#43A047", ms=9, zorder=6, label="B")
_pt_O,   = _ax.plot([], [], "^", color="#FB8C00", ms=9, zorder=6, label="O (COM)")
_arrow_F,= _ax.plot([], [], "-",  color="#7B1FA2", lw=2.5, zorder=5)
_arrow_h,= _ax.plot([], [], ">",  color="#7B1FA2", ms=7,   zorder=5)
_info    = _ax.text(0.02, 0.98, "", transform=_ax.transAxes, fontsize=8.5,
                    va="top", fontfamily="monospace",
                    bbox=dict(facecolor="white", alpha=0.80,
                              edgecolor="lightgray", boxstyle="round,pad=0.3"))

_leg_modes = _ax.legend(
    handles=[mpatches.Patch(color=_MC[m], label=_ML[m])
             for m in [MODE_1, MODE_2, MODE_3]],
    loc="upper right", fontsize=7.5, framealpha=0.85)
_ax.add_artist(_leg_modes)
_ax.legend(handles=[_pt_A, _pt_B, _pt_O],
           loc="lower right", fontsize=8, framealpha=0.85)

_traj_x, _traj_y = [], []
_F_SCALE = 0.035

def _update(i):
    psi_i = psi_all[i]; delta_i = delta_all[i]
    mode_i = mode_all[i]; t_i = t_all[i]
    xA_i, yA_i = xA_all[i], yA_all[i]
    xO_i, yO_i = xO_all[i], yO_all[i]
    xB_i, yB_i = xB_all[i], yB_all[i]

    _wall2.set_data([0, 0.55*np.cos(psi_i)], [0, 0.55*np.sin(psi_i)])

    _cd, _sd = np.cos(delta_i), np.sin(delta_i)
    _R_i = np.array([[_cd, -_sd], [_sd, _cd]])
    _corners = np.array([[0,0],[_a,0],[_a,_b],[0,_b],[0,0]])
    _world = np.array([[xA_i, yA_i]]) + (_R_i @ _corners.T).T
    _body.set_data(_world[:,0], _world[:,1]); _body.set_color(_MC[mode_i])

    _pt_A.set_data([xA_i], [yA_i])
    _pt_B.set_data([xB_i], [yB_i])
    _pt_O.set_data([xO_i], [yO_i])

    _fxi, _fyi = fx_all[i], fy_all[i]
    _xe = xO_i + _fxi*_F_SCALE; _ye = yO_i + _fyi*_F_SCALE
    _arrow_F.set_data([xO_i, _xe], [yO_i, _ye])
    _arrow_h.set_data([_xe], [_ye])

    _traj_x.append(xO_i); _traj_y.append(yO_i)
    _traj.set_data(_traj_x, _traj_y)

    _info.set_text(
        f"{_ML[mode_i]}\n"
        f"t  = {t_i:5.2f} s\n"
        f"ψ  = {np.degrees(psi_i):6.1f}°\n"
        f"δ  = {np.degrees(delta_i):6.1f}°\n"
        f"l  = {l_all[i]:6.3f} m\n"
        f"Fx = {_fxi:6.2f} N\n"
        f"Fy = {_fyi:6.2f} N")
    return _wall2, _body, _pt_A, _pt_B, _pt_O, _arrow_F, _arrow_h, _traj, _info

_ani = FuncAnimation(_fig, _update, frames=_frames,
                     interval=1000/_fps, blit=False)
plt.tight_layout()
print("Sauvegarde du GIF...")
_ani.save("animation_controller.gif", writer="pillow", fps=_fps)
print("GIF sauvegardé : animation_controller.gif")
plt.show()
