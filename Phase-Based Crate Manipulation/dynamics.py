"""
dynamics.py — Physics engine for the constrained crate system.

Philosophy (from thesis promoter feedback)
-------------------------------------------
The system has a single mechanical DOF (l = x-position of corner A along
wall 1) once both contacts are active.  The orientation delta is fully
determined by the constraint delta(l, psi).

Approach: Lagrangian / D'Alembert principle
-------------------------------------------
Ideal frictionless contacts are *constraint forces*: they are orthogonal to
any virtual displacement δl, so they do **zero virtual work** and drop out
of the equation of motion automatically.

This avoids computing contact forces (fyA, fBn) inside the integration loop,
which is the right choice because:
  - contact forces depend on micro-contact, local elasticity, and impact
    dynamics that are not modelled and cannot be identified reliably;
  - their exact values are not needed for the control strategy, which works
    at the energy / quasi-static level.

Contact forces can still be *estimated* a posteriori via
`compute_contact_forces()` for visualisation or validation, but they are
clearly marked as a diagnostic, not as a feedback signal.

Geometry
--------
- Wall 1 : horizontal, y = 0.  Corner A slides along it at (l, 0).
- Wall 2 : rotates around origin, angle psi (pi/2 → pi).
- delta = orientation angle of crate (computed from constraint).

State
-----
  l    : x-position of A along wall 1
  ldot : velocity of l

Wall 2 angle psi(t) is a prescribed kinematic input.
"""
import numpy as np
from state import CrateState, SimParams, Mode, rot2, get_corners_world


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


# ── 1-DOF Lagrangian equation of motion ───────────────────────────────────────

def f_continuous(l: float, ldot: float, fx: float, fy: float,
                 t: float, params: SimParams):
    """
    Compute lddot from the 1-DOF Lagrangian equation of motion.

    Derivation (D'Alembert / virtual work principle)
    -------------------------------------------------
    Virtual displacement δl at fixed psi displaces:
        Corner A  :  δrA = [δl, 0]          → wall-1 normal [0, fyA] ⊥ δrA  ✓
        Corner B  :  δrB = [axB, ayB] * δl  → wall-2 normal nB · δrB = 0   ✓
        CoM O     :  δrO = [axc, ayc]  * δl
        Rotation  :  δdelta = delta_l * δl

    Both contact normal forces do zero virtual work → they vanish from the
    equation of motion.  Only the applied force (fx, fy) contributes:

        M_eff * lddot = Q_l - f_nl

    where:
        M_eff = m*(axc² + ayc²) + I_G * delta_l²   (generalised inertia)
        Q_l   = fx*axc + fy*ayc                     (generalised applied force)
        f_nl  = Coriolis / centripetal correction    (from psi(t) driving)
        I_G   = (m/12)*(a² + b²)                    (inertia about CoM)

    Returns: (ldot, lddot)
    """
    psi, psidot, psiddot = wall_profile(t, params)

    (delta, delta_dot, delta_l, delta_psi,
     delta_ll, delta_lpsi, delta_psipsi) = delta_kinematics(
        l, ldot, psi, psidot, params)

    Rp_r = rot_prime(delta) @ params.rAO_body   # R'(delta) @ rAO_body
    R_r  = rot(delta)       @ params.rAO_body   # R(delta)  @ rAO_body

    # Jacobian ∂rO/∂l  (coefficients of lddot in CoM acceleration)
    axc = 1.0 + Rp_r[0] * delta_l
    ayc =       Rp_r[1] * delta_l

    # Nonlinear (Coriolis / centripetal) acceleration of CoM
    nl_ddot = (delta_ll * ldot**2
               + 2.0 * delta_lpsi * ldot * psidot
               + delta_psipsi * psidot**2
               + delta_psi * psiddot)
    ax_nl = Rp_r[0] * nl_ddot - R_r[0] * delta_dot**2
    ay_nl = Rp_r[1] * nl_ddot - R_r[1] * delta_dot**2

    # Generalised inertia and applied force
    I_G   = (params.m / 12.0) * (params.a**2 + params.b**2)
    M_eff = params.m * (axc**2 + ayc**2) + I_G * delta_l**2

    Q_l  = fx * axc + fy * ayc
    f_nl = params.m * (ax_nl * axc + ay_nl * ayc) + I_G * nl_ddot * delta_l

    lddot = (Q_l - f_nl) / max(M_eff, 1e-10)

    return ldot, lddot


# ── Contact force diagnostic (NOT used in integration) ────────────────────────

def compute_contact_forces(l: float, ldot: float, fx: float, fy: float,
                           t: float, params: SimParams):
    """
    Estimate contact forces by solving the Newton-Euler system (3×3).

    *** DIAGNOSTIC ONLY — do NOT use as a feedback signal. ***

    Contact forces (fyA, fBn) depend on micro-contact geometry, local
    elasticity, and impact dynamics that are not modelled here.  Their
    estimated values are qualitatively indicative but not physically
    reliable at the force level.

    The linear system solved is:
      [m*ax_coeff   mu    -fBdir_x ] [lddot]   [fx - m*ax_nl         ]
      [m*ay_coeff  -1     -fBdir_y ] [fyA  ] = [fy - m*ay_nl         ]
      [IA*delta_l   0   -r×fBdir  ] [fBn  ]   [r×F - IA*nl_ddot     ]

    Returns: (fyA, fBn, fxB, fyB, fBdir, psi)
    """
    psi, psidot, psiddot = wall_profile(t, params)

    (delta, delta_dot, delta_l, delta_psi,
     delta_ll, delta_lpsi, delta_psipsi) = delta_kinematics(
        l, ldot, psi, psidot, params)

    R  = rot(delta)
    Rp = rot_prime(delta)

    rAO_w = R  @ params.rAO_body
    rAB_w = R  @ params.rAB_body
    Rp_r  = Rp @ params.rAO_body
    R_r   = R  @ params.rAO_body

    ax_coeff = 1.0 + Rp_r[0] * delta_l
    ay_coeff =       Rp_r[1] * delta_l

    nl_ddot = (delta_ll * ldot**2
               + 2.0 * delta_lpsi * ldot * psidot
               + delta_psipsi * psidot**2
               + delta_psi * psiddot)
    ax_nl = Rp_r[0] * nl_ddot - R_r[0] * delta_dot**2
    ay_nl = Rp_r[1] * nl_ddot - R_r[1] * delta_dot**2

    tB    = np.array([ np.cos(psi),  np.sin(psi)])
    nB    = np.array([ np.sin(psi), -np.cos(psi)])
    fBdir = nB - params.mu * tB

    m  = params.m
    IA = params.I_A

    M = np.array([
        [ m * ax_coeff,   params.mu,   -fBdir[0]             ],
        [ m * ay_coeff,  -1.0,         -fBdir[1]             ],
        [ IA * delta_l,   0.0,         -cross2(rAB_w, fBdir) ]
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

    fyA  = sol[1]
    fBn  = sol[2]
    fxB  = fBn * fBdir[0]
    fyB  = fBn * fBdir[1]

    return fyA, fBn, fxB, fyB, fBdir, psi


# ── RK4 integrator ────────────────────────────────────────────────────────────

def rk4_step(l: float, ldot: float, fx: float, fy: float,
             t: float, params: SimParams):
    """
    Single fourth-order Runge-Kutta step for (l, ldot).
    Returns (l_new, ldot_new).
    """
    h = params.dt

    def f_ode(l_, ldot_, t_):
        return f_continuous(l_, ldot_, fx, fy, t_, params)

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

    l_new = np.clip(l_new, 0.0, params.b)
    if l_new <= 0.0 and ldot_new < 0:
        ldot_new = 0.0
    if l_new >= params.b and ldot_new > 0:
        ldot_new = 0.0

    state.l    = l_new
    state.ldot = ldot_new


# ── APPROACH phase: free flight under control force ────────────────────────────

def approach_step(state: CrateState, params: SimParams) -> None:
    """
    Advance free-floating crate by one dt (in-place).

    Dynamics:
      - Force (0, -F_approach) applied at COM  → pushes box toward wall 1.
      - No torque in free flight → omega stays constant.

    Uses symplectic Euler (velocity first, then position) for energy stability.
    """
    h  = params.dt
    ay = -params.F_approach / params.m   # only y-force

    state.vy    += ay * h
    state.x     += state.vx  * h
    state.y     += state.vy  * h
    state.theta += state.omega * h


# ── CONTACT phase: impact + pivot rotation ────────────────────────────────────

def impact_response(state: CrateState, corner_idx: int,
                    params: SimParams) -> None:
    """
    Apply the instantaneous impulse when corner `corner_idx` hits wall 1 (y = 0).

    Physics (trend-level, coefficient of restitution e):
    -------------------------------------------------------
    Normal impulse J_y at the contact corner satisfies the post-impact
    condition:  v_corner_y_after = -e * v_corner_y_before

    From linear + angular momentum:
        J_y = -(1+e) * v_corner_y_before / (1/m + r_x² / I_G)

    where r_x is the x-component of the vector COM → corner in world frame.

    Updates (vx, vy, omega) and sets pivot_corner_idx / pivot_x in-place.
    Transitions state to Mode.CONTACT.
    """
    R        = rot2(state.theta)
    r_C_body = params.corners_body[corner_idx]   # COM → corner, body frame
    r_C_world = R @ r_C_body                     # COM → corner, world frame
    r_x = r_C_world[0]

    # Velocity of that corner just before impact
    # v_corner = v_COM + omega × r  (2D: v_y = vy + omega * r_x)
    v_corner_y = state.vy + state.omega * r_x

    if v_corner_y >= 0.0:
        # Corner already moving away from wall — no impulse needed
        return

    e   = params.e_restitution
    I_G = params.I_G

    J_y = -(1.0 + e) * v_corner_y / (1.0 / params.m + r_x**2 / I_G)

    state.vy    += J_y / params.m
    state.omega += r_x * J_y / I_G

    # Snap corner exactly to wall 1 to avoid numerical drift
    corners = get_corners_world(state, params)
    state.y -= corners[corner_idx, 1]          # shift COM so corner is at y=0

    state.pivot_corner_idx = corner_idx
    state.pivot_x          = get_corners_world(state, params)[corner_idx, 0]
    state.mode             = Mode.CONTACT


def contact_step(state: CrateState, params: SimParams) -> None:
    """
    Advance crate rotating around its wall-1 pivot corner by one dt (in-place).

    DOF: theta  (1-D rotation around the fixed pivot point on wall 1).

    Equation of motion (Lagrangian about pivot C):
        I_C * theta_ddot = tau_C

    where:
        I_C   = I_G + m * |r_pivot→COM|²   (parallel-axis theorem)
        tau_C = cross2(r_pivot→COM, F)      (torque of control force)
        F     = (0, -F_approach)

    COM position and linear velocity are derived from theta (constraint).
    """
    h          = params.dt
    idx        = state.pivot_corner_idx
    r_C_body   = params.corners_body[idx]        # COM → corner in body frame
    R          = rot2(state.theta)

    # Vector from pivot to COM in world frame  (= - R @ r_C_body)
    r_PC = -(R @ r_C_body)                       # pivot → COM

    # Moment of inertia about pivot
    I_G   = params.I_G
    I_C   = I_G + params.m * (r_PC @ r_PC)

    # Torque of force (0, -F_approach) about pivot
    # tau = r_PC × F  (2D scalar: r_x * F_y - r_y * F_x)
    tau_C = r_PC[0] * (-params.F_approach) - r_PC[1] * 0.0

    alpha = tau_C / I_C   # angular acceleration

    # Symplectic Euler integration
    state.omega += alpha * h
    state.theta += state.omega * h

    # Update COM position from new theta (pivot stays at (pivot_x, 0))
    R_new  = rot2(state.theta)
    r_PC_new = -(R_new @ r_C_body)
    state.x  = state.pivot_x + r_PC_new[0]
    state.y  =            0.0 + r_PC_new[1]

    # Linear velocity of COM from rotation around fixed pivot
    # v_COM = omega × r_PC  (2D: vx = -omega * r_y, vy = omega * r_x)
    state.vx = -state.omega * r_PC_new[1]
    state.vy =  state.omega * r_PC_new[0]
