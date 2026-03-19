"""
dynamics.py — Physics engine for the constrained crate system.

Ported from moteur_dynamique_discrétisee.py and adapted to use SimParams.

Geometry
--------
- Wall 1 : horizontal floor, y = 0.  Corner A slides along it at (l, 0).
- Wall 2 : rotates around origin, angle psi (pi/2 → pi).
- Corner A (bottom-left of crate) is the pivot contact on wall 1.
- Corner B (top-left of crate) is in sliding contact with wall 2.
- delta = orientation angle of crate (computed from constraint: A on wall 1
  AND B on wall 2).

State
-----
  l    : x-position of A along wall 1
  ldot : velocity of l

The system has a single mechanical degree of freedom (l) once both contacts
are active.  Wall 2 position psi(t) is a prescribed kinematic input.
"""
import numpy as np
from state import CrateState, SimParams


# ── Geometric helpers ──────────────────────────────────────────────────────────

def rot(theta: float) -> np.ndarray:
    """2-D rotation matrix."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def rot_prime(theta: float) -> np.ndarray:
    """Derivative of rot w.r.t. theta."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[-s, -c], [c, -s]])


def cross2(r: np.ndarray, f: np.ndarray) -> float:
    """2-D cross product (scalar z-component)."""
    return r[0] * f[1] - r[1] * f[0]


# ── Wall 2 kinematic profile ───────────────────────────────────────────────────

def wall_profile(t: float, params: SimParams):
    """
    Trapezoidal velocity profile for wall 2 angle.

    Returns (psi, psidot, psiddot).
    """
    psi_start = params.psi_start
    psi_end   = params.psi_end
    dpsi      = psi_end - psi_start
    T_wall    = params.T_wall
    t1 = T_wall * params.wall_t1_ratio
    t2 = T_wall * params.wall_t2_ratio

    acc = dpsi / (0.5 * t1**2 + (t2 - t1) * t1 + 0.5 * (T_wall - t2)**2)
    v1  = acc * t1
    p1  = psi_start + 0.5 * acc * t1**2
    p2  = p1 + v1 * (t2 - t1)

    if t <= t1:
        psi, psidot, psiddot = psi_start + 0.5 * acc * t**2, acc * t, acc
    elif t <= t2:
        psi, psidot, psiddot = p1 + v1 * (t - t1), v1, 0.0
    elif t <= T_wall:
        psi     = p2 + v1 * (t - t2) - 0.5 * acc * (t - t2)**2
        psidot  = v1 - acc * (t - t2)
        psiddot = -acc
    else:
        psi, psidot, psiddot = psi_end, 0.0, 0.0

    return psi, psidot, psiddot


# ── Constraint kinematics: (l, psi) → delta ───────────────────────────────────

def delta_kinematics(l: float, ldot: float,
                     psi: float, psidot: float,
                     params: SimParams):
    """
    Compute crate orientation delta and its time-derivatives from the
    double-contact constraint (A on wall 1, B on wall 2).

    Note: psiddot is NOT needed here; it is used only in f_continuous.

    Returns:
        delta, delta_dot,
        delta_l, delta_psi,          (first partial derivatives)
        delta_ll, delta_lpsi, delta_psipsi   (second partial derivatives)
    """
    b   = params.b
    eps = 1e-9

    u  = np.clip((l / b) * np.sin(psi), -1 + eps, 1 - eps)
    k  = 1.0 / np.sqrt(max(eps, 1 - u * u))
    k3 = k ** 3

    delta = psi - 0.5 * np.pi + np.arcsin(u)

    u_l      = (1 / b) * np.sin(psi)
    u_psi    = (l / b) * np.cos(psi)
    u_lpsi   = (1 / b) * np.cos(psi)
    u_psipsi = -(l / b) * np.sin(psi)

    delta_l      = k * u_l
    delta_psi    = 1 + k * u_psi
    delta_ll     = u * k3 * u_l * u_l
    delta_lpsi   = k * u_lpsi   + u * k3 * u_l   * u_psi
    delta_psipsi = k * u_psipsi + u * k3 * u_psi * u_psi

    delta_dot = delta_l * ldot + delta_psi * psidot

    return delta, delta_dot, delta_l, delta_psi, delta_ll, delta_lpsi, delta_psipsi


# ── Continuous Newton-Euler dynamics ──────────────────────────────────────────

def f_continuous(l: float, ldot: float, fx: float, fy: float,
                 t: float, params: SimParams):
    """
    Given the current state (l, ldot), applied force (fx, fy) at the CoM,
    and time t, solve Newton-Euler equations for:
      - lddot   : acceleration of l
      - fyA     : normal reaction from wall 1 on corner A
      - fBn     : normal reaction magnitude from wall 2 on corner B
      - fxB, fyB: Cartesian components of the wall-2 contact force

    The linear system (3×3) is solved at each call:
      [m*ax_coeff   mu    -fBdir_x ] [lddot]   [fx - m*ax_nl            ]
      [m*ay_coeff  -1     -fBdir_y ] [fyA  ] = [fy - m*ay_nl            ]
      [IA*δ_l       0   -r×fBdir  ] [fBn  ]   [r×F - IA*nl_ddot        ]

    Returns: (ldot, lddot, fyA, fBn, fxB, fyB, fBdir, psi)
    """
    psi, psidot, psiddot = wall_profile(t, params)

    (delta, delta_dot, delta_l, delta_psi,
     delta_ll, delta_lpsi, delta_psipsi) = delta_kinematics(
        l, ldot, psi, psidot, params)

    R  = rot(delta)
    Rp = rot_prime(delta)

    rAO_body = params.rAO_body
    rAB_body = params.rAB_body

    rAO_w = R  @ rAO_body
    rAB_w = R  @ rAB_body
    Rp_r  = Rp @ rAO_body
    R_r   = R  @ rAO_body

    # Coefficients of lddot in the acceleration of CoM
    ax_coeff = 1.0 + Rp_r[0] * delta_l
    ay_coeff =       Rp_r[1] * delta_l

    # Non-linear (centripetal + Coriolis) acceleration of CoM
    nl_ddot  = (delta_ll * ldot**2
                + 2 * delta_lpsi * ldot * psidot
                + delta_psipsi * psidot**2
                + delta_psi * psiddot)
    ax_nl = Rp_r[0] * nl_ddot - R_r[0] * delta_dot**2
    ay_nl = Rp_r[1] * nl_ddot - R_r[1] * delta_dot**2

    # Wall 2 contact force direction: normal minus Coulomb friction
    tB    = np.array([ np.cos(psi),  np.sin(psi)])   # tangent along wall 2
    nB    = np.array([ np.sin(psi), -np.cos(psi)])   # inward normal of wall 2
    fBdir = nB - params.mu * tB

    m  = params.m
    IA = params.I_A

    M = np.array([
        [ m * ax_coeff,   params.mu,   -fBdir[0]              ],
        [ m * ay_coeff,  -1.0,         -fBdir[1]              ],
        [ IA * delta_l,   0.0,         -cross2(rAB_w, fBdir)  ]
    ])
    rhs = np.array([
        fx - m * ax_nl,
        fy - m * ay_nl,
        cross2(rAO_w, np.array([fx, fy])) - IA * nl_ddot
    ])

    try:
        sol = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        sol = np.zeros(3)

    lddot = sol[0]
    fyA   = sol[1]
    fBn   = sol[2]
    fxB   = fBn * fBdir[0]
    fyB   = fBn * fBdir[1]

    return ldot, lddot, fyA, fBn, fxB, fyB, fBdir, psi


# ── RK4 integrator ────────────────────────────────────────────────────────────

def rk4_step(l: float, ldot: float, fx: float, fy: float,
             t: float, params: SimParams):
    """
    Single fourth-order Runge-Kutta step for (l, ldot).
    Returns (l_new, ldot_new).
    """
    h = params.dt

    def f_ode(l_, ldot_, t_):
        d, a = f_continuous(l_, ldot_, fx, fy, t_, params)[:2]
        return d, a

    d1, a1 = f_ode(l,            ldot,            t)
    d2, a2 = f_ode(l + h/2*d1,  ldot + h/2*a1,   t + h/2)
    d3, a3 = f_ode(l + h/2*d2,  ldot + h/2*a2,   t + h/2)
    d4, a4 = f_ode(l + h*d3,    ldot + h*a3,      t + h)

    l_new    = l    + (h / 6) * (d1 + 2*d2 + 2*d3 + d4)
    ldot_new = ldot + (h / 6) * (a1 + 2*a2 + 2*a3 + a4)
    return l_new, ldot_new


def integrate_step(state: CrateState, fx: float, fy: float,
                   t: float, params: SimParams) -> None:
    """
    Advance state by one dt in-place, enforcing wall-1 constraint (l ∈ [0, b]).
    """
    l_new, ldot_new = rk4_step(state.l, state.ldot, fx, fy, t, params)

    # Enforce physical bounds: A cannot go past the corner or past length b
    l_new = np.clip(l_new, 0.0, params.b)
    if l_new <= 0.0 and ldot_new < 0:
        ldot_new = 0.0
    if l_new >= params.b and ldot_new > 0:
        ldot_new = 0.0

    state.l    = l_new
    state.ldot = ldot_new
