"""
controller_redesign_mode1.py
────────────────────────────
Mode 1 redesigned controller: free rigid body with optimal force allocation.

Physical setup
──────────────
The crate (mass m, inertia I_O about CoM O) is a free rigid body.
Point A (corner in contact with wall 1) is NOT constrained — it may drift.
The single planar force F = [Fx, Fy] is applied at point A (body frame).

Because F acts at A (offset from CoM O by r_{AO}), it simultaneously
produces translational acceleration of A AND angular acceleration α.
This coupling allows a single 2-DOF input to address two objectives:
  1. Stabilise position of A:  p_A → p_A,ref = (0, 0)
  2. Regulate orientation:     δ   → δ_ref(t)

Derivation
──────────
Let r_AO_world = R(δ) @ rAO_body  with components [c2, -c1]:
  c1 = -(a/2)·sin δ - (b/2)·cos δ   (= -r_{AO,world,y})
  c2 =  (a/2)·cos δ - (b/2)·sin δ   (=  r_{AO,world,x})

Newton (F at any body point → CoM):  m · a_O = F
Angular momentum about O (F at A):
  I_O · α = r_{OA} × F = -c2·Fy - c1·Fx  =  -(c1·Fx + c2·Fy)

Acceleration of A (rigid body kinematics):
  a_A = a_O + α·[-c1, -c2]^T + ω²·[c2, -c1]^T

Collecting:  [a_A; α] = J(δ)·F + b(δ,ω)

  J = [[1/m + c1²/I_O ,  c1·c2/I_O  ],
       [c1·c2/I_O      ,  1/m+c2²/I_O],
       [-c1/I_O         ,  -c2/I_O    ]]

  b = [ω²·c2, -ω²·c1, 0]^T

Optimal force (weighted least squares):
  F* = (J^T W J)^{-1} J^T W (d_des - b)

  W = diag(wA, wA, wδ)

  d_des = [a_A,x^des, a_A,y^des, α^des]^T

where
  a_A^des = -KpA·(p_A - p_A,ref) - KdA·(ṗ_A - ṗ_A,ref)
  α^des   = α_ref - Kpδ·(δ - δ_ref) - Kdδ·(ω - ω_ref)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.interpolate import interp1d

# ══════════════════════════════════════════════════════════════════════
# Re-import shared infrastructure from controller_redesign.py
# (parameters, wall profile, geometry, Mode 2/3 controllers)
# ══════════════════════════════════════════════════════════════════════
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

# We import the module but suppress its side-effect prints/simulations
# by temporarily redirecting stdout.
import io, contextlib
_buf = io.StringIO()
with contextlib.redirect_stdout(_buf):
    import controller_redesign as _cr

# ── Shared objects ────────────────────────────────────────────────────
params_ref  = _cr.params_ref
params_real = _cr.params_real
psi_profile = _cr.psi_profile
T_SIM       = _cr.T_SIM
MODE_1, MODE_2, MODE_3 = _cr.MODE_1, _cr.MODE_2, _cr.MODE_3
_TAU_F      = _cr._TAU_F

reference_mode1 = _cr.reference_mode1
control_mode2   = _cr.control_mode2
control_mode3   = _cr.control_mode3
dynamics_mode3  = _cr.dynamics_mode3
phi_B_mode2     = _cr.phi_B_mode2
delta_kinematics = _cr.delta_kinematics
mode2a_core     = _cr.mode2a_core
impact_mode2_to_3 = _cr.impact_mode2_to_3
dynamics_mode2_free_B = _cr.dynamics_mode2_free_B

# ── Nominal reference (already computed inside _cr on import) ─────────
ref = _cr.ref   # simulate_nominal result


# ══════════════════════════════════════════════════════════════════════
# CONTROLLER GAINS
# ══════════════════════════════════════════════════════════════════════

# Position of A — PD gains
_KP_A  = 30.0   # proportional on p_A error  [N/m]
_KD_A  = 10.0   # derivative  on ṗ_A error   [N·s/m]

# Orientation — PD gains (same role as in the original Mode-1 controller)
_KP_DELTA = 20.0  # proportional on δ − δ_ref   [N·m/rad]
_KD_DELTA =  6.0  # derivative  on ω − ω_ref    [N·m·s/rad]

# Optimisation weights
_W_A     = 1.0   # weight on position-of-A objective
_W_DELTA = 0.5   # weight on orientation objective


# ══════════════════════════════════════════════════════════════════════
# GEOMETRY HELPERS
# ══════════════════════════════════════════════════════════════════════

def _c1c2(delta, params):
    """
    Return (c1, c2) — projections of r_AO onto world axes.

    c1 = -r_{AO,world,y}   c2 = r_{AO,world,x}

    With r_AO_world = R(δ) @ rAO_body, rAO_body = [a/2, b/2].
    """
    a, b = params["a"], params["b"]
    cd, sd = np.cos(delta), np.sin(delta)
    c1 = -(a/2)*sd - (b/2)*cd
    c2 =  (a/2)*cd - (b/2)*sd
    return c1, c2


def jacobian_and_bias(delta, omega, params):
    """
    Build the 3×2 Jacobian J and bias b such that

        [a_A; α] = J @ F + b

    for a free rigid body with F applied at point A.

    Parameters
    ----------
    delta : float  — current orientation [rad]
    omega : float  — current angular velocity [rad/s]
    params : dict

    Returns
    -------
    J    : ndarray (3, 2)
    bias : ndarray (3,)
    """
    m   = params["m"]
    I_O = params["I_O"]
    c1, c2 = _c1c2(delta, params)

    J = np.array([
        [1.0/m + c1**2/I_O,   c1*c2/I_O         ],   # a_A,x row
        [c1*c2/I_O,            1.0/m + c2**2/I_O ],   # a_A,y row
        [-c1/I_O,              -c2/I_O            ],   # α     row
    ])

    bias = np.array([omega**2 * c2,
                     -omega**2 * c1,
                     0.0])
    return J, bias


# ══════════════════════════════════════════════════════════════════════
# MODE 1 FREE-BODY CONTROLLER
# ══════════════════════════════════════════════════════════════════════

def optimal_force_mode1(t, xA, yA, delta, xAdot, yAdot, omega, params,
                        kpA=None, kdA=None, kp_delta=None, kd_delta=None,
                        wA=None, w_delta=None,
                        xA_ref=0.0, yA_ref=0.0,
                        xAdot_ref=0.0, yAdot_ref=0.0):
    """
    Compute the optimal force F* = [Fx, Fy] for Mode 1 (free body).

    The force is applied at corner A of the crate.  It is the solution of

        F* = argmin_F  [ wA · ||a_A(F) − a_A^des||²
                       + wδ · (α(F) − α^des)²       ]

    which is solved analytically as a weighted least-squares problem.

    Parameters
    ----------
    t                     : float — current time [s]
    xA, yA                : float — current position of A [m]
    delta                 : float — current orientation [rad]
    xAdot, yAdot          : float — current velocity of A [m/s]
    omega                 : float — current angular velocity [rad/s]
    params                : dict
    kpA, kdA              : PD gains for position of A (default: module globals)
    kp_delta, kd_delta    : PD gains for orientation   (default: module globals)
    wA, w_delta           : optimisation weights        (default: module globals)
    xA_ref, yA_ref        : desired position of A       (default: origin)
    xAdot_ref, yAdot_ref  : desired velocity of A       (default: zero)

    Returns
    -------
    Fx, Fy : float
    """
    if kpA      is None: kpA      = _KP_A
    if kdA      is None: kdA      = _KD_A
    if kp_delta is None: kp_delta = _KP_DELTA
    if kd_delta is None: kd_delta = _KD_DELTA
    if wA       is None: wA       = _W_A
    if w_delta  is None: w_delta  = _W_DELTA

    # ── Desired accelerations ─────────────────────────────────────────
    a_A_des = np.array([
        -kpA * (xA - xA_ref) - kdA * (xAdot - xAdot_ref),
        -kpA * (yA - yA_ref) - kdA * (yAdot - yAdot_ref),
    ])

    delta_ref, omega_ref, alpha_ref = reference_mode1(t, params)
    alpha_des = alpha_ref - kp_delta*(delta - delta_ref) - kd_delta*(omega - omega_ref)

    d_des = np.array([a_A_des[0], a_A_des[1], alpha_des])

    # ── Jacobian and bias ─────────────────────────────────────────────
    J, bias = jacobian_and_bias(delta, omega, params)

    # ── Weighted least squares ────────────────────────────────────────
    # Minimise  ||W^{1/2}(J F − e)||²   with e = d_des − bias
    W  = np.diag([wA, wA, w_delta])
    e  = d_des - bias
    JtW  = J.T @ W          # (2, 3)
    JtWJ = JtW @ J          # (2, 2) — symmetric PD when J has rank 2
    JtWe = JtW @ e          # (2,)

    F_star = np.linalg.solve(JtWJ, JtWe)
    return float(F_star[0]), float(F_star[1])


# ══════════════════════════════════════════════════════════════════════
# MODE 1 FREE-BODY DYNAMICS
# ══════════════════════════════════════════════════════════════════════
# State vector: x = [xA, yA, δ, ẋA, ẏA, ω, Fx_filt, Fy_filt]
#               idx  0   1  2   3    4  5    6        7

def dynamics_mode1_free(t, x, params):
    """
    ODE right-hand side for Mode 1 with A unconstrained.

    State: [xA, yA, δ, ẋA, ẏA, ω, Fx_filt, Fy_filt]

    The filtered force is applied at A; the commanded force comes from
    optimal_force_mode1.  A first-order filter (time constant _TAU_F)
    smooths force transitions.
    """
    xA, yA, delta, xAdot, yAdot, omega, Fx_filt, Fy_filt = x

    m   = params["m"]
    I_O = params["I_O"]
    c1, c2 = _c1c2(delta, params)

    # ── Accelerations from filtered force ─────────────────────────────
    # α = -(c1·Fx + c2·Fy) / I_O
    alpha = -(c1*Fx_filt + c2*Fy_filt) / I_O

    # a_A = F/m + α·[-c1, -c2] + ω²·[c2, -c1]
    xAddot = Fx_filt/m - alpha*c1 + omega**2 * c2
    yAddot = Fy_filt/m - alpha*c2 - omega**2 * c1

    # ── Controller ────────────────────────────────────────────────────
    Fx_cmd, Fy_cmd = optimal_force_mode1(
        t, xA, yA, delta, xAdot, yAdot, omega, params)

    # ── Force filter ──────────────────────────────────────────────────
    Fx_dot = (Fx_cmd - Fx_filt) / _TAU_F
    Fy_dot = (Fy_cmd - Fy_filt) / _TAU_F

    return [xAdot, yAdot, omega, xAddot, yAddot, alpha, Fx_dot, Fy_dot]


# ══════════════════════════════════════════════════════════════════════
# FULL SIMULATION (Mode 1 free  +  existing Mode 2 / Mode 3)
# ══════════════════════════════════════════════════════════════════════

def simulate_real_free(params):
    """
    Simulate the real system with the new Mode-1 free-body controller,
    then hand off to the existing Mode-2 and Mode-3 controllers.

    Returns a dict with keys: t, xA, yA, delta, omega, fx, fy, mode, ...
    """
    # ── Initial conditions for Mode 1 ─────────────────────────────────
    psi0, psi_dot0, _ = psi_profile(0.0)
    delta0 = psi0 - np.pi/2
    omega0 = psi_dot0
    # A starts at origin, at rest
    xA0, yA0     = 0.0, 0.0
    xAdot0, yAdot0 = 0.0, 0.0

    # Initialise filter at the commanded force (no jump at t=0)
    Fx0, Fy0 = optimal_force_mode1(
        0.0, xA0, yA0, delta0, xAdot0, yAdot0, omega0, params)

    x0 = np.array([xA0, yA0, delta0, xAdot0, yAdot0, omega0, Fx0, Fy0])

    # ── Transition event: psi reaches psi_sw_1 ────────────────────────
    def ev_1_to_2(t, x):
        psi, _, _ = psi_profile(t)
        return psi - params["psi_sw_1"]
    ev_1_to_2.terminal  = True
    ev_1_to_2.direction = 1

    sol1 = solve_ivp(
        lambda t, x: dynamics_mode1_free(t, x, params),
        (0.0, T_SIM), x0,
        dense_output=True, events=[ev_1_to_2],
        rtol=1e-8, atol=1e-10)

    t1_end = float(sol1.t[-1])
    t1     = np.linspace(0.0, t1_end, 500)
    y1     = sol1.sol(t1)

    xA1    = y1[0];  yA1    = y1[1]
    delta1 = y1[2]
    xAdot1 = y1[3];  yAdot1 = y1[4]
    omega1 = y1[5]
    Fxf1   = y1[6];  Fyf1   = y1[7]
    psi1   = np.array([psi_profile(tt)[0] for tt in t1])

    # ── Recover CoM and position of A for output ──────────────────────
    xO1 = np.zeros(len(t1)); yO1 = np.zeros(len(t1))
    for i in range(len(t1)):
        cd, sd = np.cos(delta1[i]), np.sin(delta1[i])
        R_i = np.array([[cd, -sd],[sd, cd]])
        rAO_w = R_i @ params["rAO_body"]
        xO1[i] = xA1[i] + rAO_w[0]
        yO1[i] = yA1[i] + rAO_w[1]

    # ── Transition to Mode 2: build initial conditions for Mode 2 ─────
    # A position at transition — project onto wall-1 direction
    psi_sw, _, _ = psi_profile(t1_end)
    wall_dir = np.array([np.cos(psi_sw), np.sin(psi_sw)])

    # l = coordinate of A along wall direction
    pA_end = np.array([xA1[-1], yA1[-1]])
    l_end   = float(np.dot(pA_end, wall_dir))

    # ldot = projection of ẋA onto wall direction
    vA_end = np.array([xAdot1[-1], yAdot1[-1]])
    ldot_end = float(np.dot(vA_end, wall_dir))

    delta_end = delta1[-1]
    omega_end = omega1[-1]
    Fxf_end   = Fxf1[-1]
    Fyf_end   = Fyf1[-1]

    # ── Modes 2 and 3 — reuse existing simulation logic ───────────────
    t_segs, l_segs, ldot_segs, delta_segs = [], [], [], []
    omega_segs, psi_segs, fx_segs, fy_segs, mode_segs = [], [], [], [], []
    xA_segs, yA_segs = [], []

    sub_mode = "2A"
    t_cur    = t1_end
    x_cur    = np.array([l_end, ldot_end, Fxf_end, Fyf_end])

    while t_cur < T_SIM:
        # ── Sub-mode 2A (A constrained to wall, B free or in contact) ─
        if sub_mode == "2A":
            l_c, ld_c, Fxf_c, Fyf_c = x_cur

            psi_cur, psi_dot_cur, _ = psi_profile(t_cur)
            b_p = params["b"]; eps = 1e-9
            u_c = np.clip((l_c/b_p)*np.sin(psi_cur), -1+eps, 1-eps)
            delta_cur = psi_cur - np.pi/2 + np.arcsin(u_c)
            k_c = 1/np.sqrt(max(eps, 1-u_c**2))
            delta_l_c   = k_c*np.sin(psi_cur)/b_p
            delta_psi_c = 1 + k_c*(l_c/b_p)*np.cos(psi_cur)
            omega_cur   = delta_l_c*ld_c + delta_psi_c*psi_dot_cur

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
                fxfy_filt = lambda tt, xx: (Fxf_, Fyf_)
                l_dd, _, _, _ = mode2a_core(t, np.array([l_, ld_]),
                                             params, fxfy_filt)
                Fx_cmd, Fy_cmd = control_mode2(t, l_, d_, ld_, w_, params)
                return [ld_, l_dd,
                        (Fx_cmd - Fxf_) / _TAU_F,
                        (Fy_cmd - Fyf_) / _TAU_F]

            def _ev_2_to_3(t, x): return x[0] - params["l_contact"]
            _ev_2_to_3.terminal = True; _ev_2_to_3.direction = 1

            def _ev_sep(t, x):
                l_, ld_, Fxf_, Fyf_ = x
                psi_t, psi_dot_t, _ = psi_profile(t)
                b_p = params["b"]; eps = 1e-9
                u_  = np.clip((l_/b_p)*np.sin(psi_t), -1+eps, 1-eps)
                d_  = psi_t - np.pi/2 + np.arcsin(u_)
                fxfy_filt = lambda tt, xx: (Fxf_, Fyf_)
                _, lB, _, _ = mode2a_core(t, np.array([l_, ld_]),
                                          params, fxfy_filt)
                return lB
            _ev_sep.terminal = True; _ev_sep.direction = -1

            sol = solve_ivp(_dyn2a_real, (t_cur, T_SIM), x_cur,
                            dense_output=True,
                            events=[_ev_2_to_3, _ev_sep],
                            rtol=1e-8, atol=1e-10)
            t_end = float(sol.t[-1])
            t_seg = np.linspace(t_cur, t_end,
                                max(3, int(300*(t_end-t_cur)/T_SIM)+3))
            y_seg = sol.sol(t_seg)
            l_seg   = y_seg[0]; ld_seg  = y_seg[1]
            Fxf_seg = y_seg[2]; Fyf_seg = y_seg[3]

            psi_seg   = np.array([psi_profile(tt)[0] for tt in t_seg])
            delta_seg = np.zeros_like(t_seg)
            omega_seg = np.zeros_like(t_seg)
            for i, tt in enumerate(t_seg):
                psi_t, psi_dot_t, _ = psi_profile(tt)
                b_p = params["b"]; eps = 1e-9
                u_  = np.clip((l_seg[i]/b_p)*np.sin(psi_t), -1+eps, 1-eps)
                delta_seg[i] = psi_t - np.pi/2 + np.arcsin(u_)
                k_  = 1/np.sqrt(max(eps, 1-u_**2))
                dl_ = k_*np.sin(psi_t)/b_p
                dp_ = 1 + k_*(l_seg[i]/b_p)*np.cos(psi_t)
                omega_seg[i] = dl_*ld_seg[i] + dp_*psi_dot_t

            # Reconstruct A position in world (A on wall 1: x_A = 0 for vert. wall)
            xA_seg_world = np.array([
                l_seg[i]*np.cos(psi_seg[i]) for i in range(len(t_seg))])
            yA_seg_world = np.array([
                l_seg[i]*np.sin(psi_seg[i]) for i in range(len(t_seg))])

            t_segs.append(t_seg); l_segs.append(l_seg)
            ldot_segs.append(ld_seg); delta_segs.append(delta_seg)
            omega_segs.append(omega_seg); psi_segs.append(psi_seg)
            fx_segs.append(Fxf_seg); fy_segs.append(Fyf_seg)
            xA_segs.append(xA_seg_world); yA_segs.append(yA_seg_world)
            mode_segs.append(np.full(len(t_seg), MODE_2, dtype=int))

            if len(sol.t_events[0]) > 0:
                x_minus = np.array([l_seg[-1], delta_seg[-1],
                                    ld_seg[-1], omega_seg[-1]])
                l_plus, ld_plus = impact_mode2_to_3(x_minus, params)
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

        # ── Sub-mode 2B (B free, A free to separate from wall) ────────
        elif sub_mode == "2B":
            def _dyn2b_real(t, x):
                l_, d_, ld_, w_, Fxf_, Fyf_ = x
                m_p, I_A = params["m"], params["I_A"]
                a_, b_   = params["a"], params["b"]
                c1_ = -(a_/2)*np.sin(d_) - (b_/2)*np.cos(d_)
                c2_ =  (a_/2)*np.cos(d_) - (b_/2)*np.sin(d_)
                c3_ = -(a_/2)*np.cos(d_) + (b_/2)*np.sin(d_)
                M_  = np.array([[m_p, m_p*c1_], [m_p*c1_, I_A]])
                r_  = np.array([Fxf_ - m_p*c3_*w_**2,
                                 c1_*Fxf_ + c2_*Fyf_])
                s_  = np.linalg.solve(M_, r_)
                Fx_cmd, Fy_cmd = control_mode2(t, l_, d_, ld_, w_, params)
                return [ld_, w_, s_[0], s_[1],
                        (Fx_cmd - Fxf_) / _TAU_F,
                        (Fy_cmd - Fyf_) / _TAU_F]

            def _ev_2_to_3b(t, x): return x[0] - params["l_contact"]
            _ev_2_to_3b.terminal = True; _ev_2_to_3b.direction = 1

            def _ev_recontact(t, x):
                l, d, ld, w, Fxf, Fyf = x
                psi_t, _, _ = psi_profile(t)
                return phi_B_mode2(l, d, psi_t, params)
            _ev_recontact.terminal = True; _ev_recontact.direction = -1

            sol = solve_ivp(_dyn2b_real, (t_cur, T_SIM), x_cur,
                            dense_output=True,
                            events=[_ev_2_to_3b, _ev_recontact],
                            rtol=1e-8, atol=1e-10)
            t_end = float(sol.t[-1])
            t_seg = np.linspace(t_cur, t_end,
                                max(3, int(300*(t_end-t_cur)/T_SIM)+3))
            y_seg = sol.sol(t_seg)
            l_seg   = y_seg[0]; delta_seg = y_seg[1]
            ld_seg  = y_seg[2]; omega_seg = y_seg[3]
            Fxf_seg = y_seg[4]; Fyf_seg   = y_seg[5]
            psi_seg = np.array([psi_profile(tt)[0] for tt in t_seg])

            # A position in world (2B: A may have left wall 1)
            xA_seg_world = np.zeros(len(t_seg))
            yA_seg_world = l_seg  # approximate: A still near wall x=0

            t_segs.append(t_seg); l_segs.append(l_seg)
            ldot_segs.append(ld_seg); delta_segs.append(delta_seg)
            omega_segs.append(omega_seg); psi_segs.append(psi_seg)
            fx_segs.append(Fxf_seg); fy_segs.append(Fyf_seg)
            xA_segs.append(xA_seg_world); yA_segs.append(yA_seg_world)
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

    # ── Mode 3 ────────────────────────────────────────────────────────
    if sub_mode == "3" and t_cur < T_SIM:
        sol3 = solve_ivp(
            lambda t, x: dynamics_mode3(t, x, params),
            (t_cur, T_SIM), x_cur,
            dense_output=True, rtol=1e-8, atol=1e-10)
        t3   = np.linspace(t_cur, T_SIM, 300)
        y3   = sol3.sol(t3)
        l3   = y3[0]; ld3 = y3[1]; Fxf3 = y3[2]; Fyf3 = y3[3]
        delta3 = np.full(len(t3), np.pi/2)
        omega3 = np.zeros(len(t3))
        psi3   = np.array([psi_profile(tt)[0] for tt in t3])

        t_segs.append(t3); l_segs.append(l3)
        ldot_segs.append(ld3); delta_segs.append(delta3)
        omega_segs.append(omega3); psi_segs.append(psi3)
        fx_segs.append(Fxf3); fy_segs.append(Fyf3)
        xA_segs.append(np.zeros(len(t3))); yA_segs.append(l3)
        mode_segs.append(np.full(len(t3), MODE_3, dtype=int))

    # ── Concatenate all segments ──────────────────────────────────────
    def _cat(lst): return np.concatenate(lst) if lst else np.array([])

    # Mode 1 arrays in output format (l=0 during mode 1, A position tracked)
    t_out     = np.concatenate([t1,                  _cat(t_segs)])
    xA_out    = np.concatenate([xA1,                 _cat(xA_segs)])
    yA_out    = np.concatenate([yA1,                 _cat(yA_segs)])
    delta_out = np.concatenate([delta1,              _cat(delta_segs)])
    omega_out = np.concatenate([omega1,              _cat(omega_segs)])
    psi_out   = np.concatenate([psi1,                _cat(psi_segs)])
    fx_out    = np.concatenate([Fxf1,                _cat(fx_segs)])
    fy_out    = np.concatenate([Fyf1,                _cat(fy_segs)])
    l_out     = np.concatenate([np.zeros(len(t1)),   _cat(l_segs)])
    mode_out  = np.concatenate([np.full(len(t1), MODE_1, dtype=int),
                                 _cat(mode_segs)])

    return {
        "t":      t_out,
        "xA":     xA_out,
        "yA":     yA_out,
        "l":      l_out,
        "delta":  delta_out,
        "omega":  omega_out,
        "psi":    psi_out,
        "fx":     fx_out,
        "fy":     fy_out,
        "mode":   mode_out,
        "t1_end": t1_end,
        # Mode-1 detailed arrays for analysis
        "xA_mode1": xA1, "yA_mode1": yA1,
        "delta_mode1": delta1, "omega_mode1": omega1,
        "t_mode1": t1,
        "Fx_mode1": Fxf1, "Fy_mode1": Fyf1,
        # Reference orientation during mode 1
        "delta_ref_mode1": np.array([reference_mode1(tt, params)[0]
                                     for tt in t1]),
        "omega_ref_mode1": np.array([reference_mode1(tt, params)[1]
                                     for tt in t1]),
    }


# ══════════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════════
print("Simulation with new Mode-1 free-body controller …")
result = simulate_real_free(params_real)
print(f"  Done.  t1_end = {result['t1_end']:.3f} s  |  "
      f"modes: {np.unique(result['mode'])}")

# ── Also run old controller for comparison ────────────────────────────
print("Running old controller for comparison …")
old = _cr.simulate_real(params_real)
print("  Done.")


# ══════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════
MODE_COLOR = {MODE_1: "steelblue", MODE_2: "seagreen", MODE_3: "darkorange"}

t1e   = result["t1_end"]
t_m1  = result["t_mode1"]
mask_m1_old = old["t"] <= old["t_mode1"][-1] if "t_mode1" in old else (
    old["mode"] == MODE_1)

fig, axes = plt.subplots(3, 2, figsize=(13, 11))
fig.suptitle("Mode 1 redesign — free body optimal controller", fontsize=13)

# ── Row 0: Position of A ──────────────────────────────────────────────
ax = axes[0, 0]
ax.plot(t_m1, result["xA_mode1"]*100, label="xA (new)", color="steelblue")
ax.plot(t_m1, result["yA_mode1"]*100, label="yA (new)", color="steelblue",
        ls="--")
ax.axhline(0, color="k", lw=0.7, ls=":")
ax.set_ylabel("Position of A [cm]")
ax.set_xlabel("Time [s]")
ax.set_title("Position of A during Mode 1")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.plot(result["xA_mode1"]*100, result["yA_mode1"]*100,
        color="steelblue", label="trajectory of A (new)")
ax.plot(0, 0, "r*", ms=10, label="A reference (0,0)")
ax.set_xlabel("xA [cm]"); ax.set_ylabel("yA [cm]")
ax.set_title("A trajectory in plane")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_aspect("equal", "datalim")

# ── Row 1: Orientation ────────────────────────────────────────────────
ax = axes[1, 0]
delta_ref_m1 = result["delta_ref_mode1"]
ax.plot(t_m1, np.degrees(result["delta_mode1"]),
        label="δ (new)", color="steelblue")
ax.plot(t_m1, np.degrees(delta_ref_m1),
        label="δ_ref", color="k", ls="--", lw=1.2)
# Old controller orientation (Mode 1 only)
mask_old1 = old["mode"] == MODE_1
ax.plot(old["t"][mask_old1], np.degrees(old["delta"][mask_old1]),
        label="δ (old, fixed-A PD)", color="tomato", ls="-.", lw=1)
ax.set_ylabel("δ [°]")
ax.set_xlabel("Time [s]")
ax.set_title("Orientation during Mode 1")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
e_delta_new = np.degrees(result["delta_mode1"] - delta_ref_m1)
e_delta_old = np.degrees(old["delta"][mask_old1] -
                          np.array([reference_mode1(tt, params_real)[0]
                                    for tt in old["t"][mask_old1]]))
ax.plot(t_m1, e_delta_new, label="δ error (new)", color="steelblue")
ax.plot(old["t"][mask_old1], e_delta_old,
        label="δ error (old)", color="tomato", ls="-.")
ax.axhline(0, color="k", lw=0.7)
ax.set_ylabel("δ − δ_ref [°]")
ax.set_xlabel("Time [s]")
ax.set_title("Orientation error during Mode 1")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ── Row 2: Forces and full simulation ─────────────────────────────────
ax = axes[2, 0]
F_mag_new = np.hypot(result["Fx_mode1"], result["Fy_mode1"])
ax.plot(t_m1, result["Fx_mode1"], label="Fx (new)", color="steelblue")
ax.plot(t_m1, result["Fy_mode1"], label="Fy (new)", color="cornflowerblue",
        ls="--")
ax.plot(t_m1, F_mag_new, label="|F| (new)", color="navy", lw=1.5)
ax.set_ylabel("Force [N]")
ax.set_xlabel("Time [s]")
ax.set_title("Control force during Mode 1")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[2, 1]
# δ over the full simulation (all modes)
for mode in [MODE_1, MODE_2, MODE_3]:
    mask = result["mode"] == mode
    if mask.any():
        ax.plot(result["t"][mask], np.degrees(result["delta"][mask]),
                color=MODE_COLOR[mode],
                label=f"Mode {mode}")
ax.plot(result["t"], np.degrees(result["psi"] - np.pi/2),
        "k--", lw=0.8, label="δ_ref")
ax.set_ylabel("δ [°]")
ax.set_xlabel("Time [s]")
ax.set_title("Orientation — full simulation (new controller)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "..",
                         "controller_redesign_mode1_plots.png"),
            dpi=150, bbox_inches="tight")
print("Plot saved to controller_redesign_mode1_plots.png")
plt.show()
