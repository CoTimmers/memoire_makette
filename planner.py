"""
Trajectory planner with pendulum anti-oscillation (input shaping).

Profile: trapezoidal (or triangular as degenerate case)
  - ta_accel = ta_star = Td/2  (fixed, optimal phase cancellation)
  - v_peak   : optimised
  - tc       : cruise duration (derived from v, Δx, a)
  - ta_decel : optimised (asymmetric profile)

Score = phi_err / phi_ref  +  lambda_time * t_total / t_ref
"""

import math
import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import Literal

G = 9.81  # m/s²


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PendulumParams:
    l: float        # pendulum length [m]
    zeta: float     # damping ratio [-]
    theta_max: float  # max allowed swing angle [rad]


@dataclass
class KinematicLimits:
    vmax: float     # max velocity [m/s]
    amax: float     # max acceleration [m/s²]


@dataclass
class PlannerConfig:
    lambda_time: float = 0.1   # weight: 0 = pure anti-oscillation, 1 = pure speed
    phase_tolerance: float = 0.05  # acceptable normalised phase error (0–1)
    ta_decel_ratio_bounds: tuple = (0.5, 2.0)  # ta_decel in [ratio*ta_star]


@dataclass
class TrajectoryProfile:
    type: Literal["triangular", "trapezoidal"]
    x0: float
    xf: float
    direction: float        # +1 or -1
    ta_accel: float         # acceleration phase duration [s]
    v_peak: float           # peak velocity [m/s]
    tc: float               # cruise duration [s]
    ta_decel: float         # deceleration phase duration [s]
    a_accel: float          # acceleration [m/s²]
    a_decel: float          # deceleration magnitude [m/s²]
    t_total: float          # total duration [s]
    phase_error_norm: float # normalised phase error [0–1]
    score: float


# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------

def pendulum_dynamics(p: PendulumParams):
    """Return (wn, wd, Td, ta_star, Mp, a_allowed)."""
    wn = math.sqrt(G / p.l)
    wd = wn * math.sqrt(1.0 - p.zeta**2)
    Td = 2.0 * math.pi / wd
    ta_star = Td / 2.0
    Mp = math.exp(-p.zeta * math.pi / math.sqrt(1.0 - p.zeta**2))
    a_theta_max = G * p.theta_max / (1.0 + Mp)
    return wn, wd, Td, ta_star, Mp, a_theta_max


def phase_error_normalised(t2: float, Td: float) -> float:
    """
    Normalised phase error in [0, 1].
    t2 = ta_accel + tc  (time at start of deceleration ramp)
    Ideal: t2 is a multiple of Td/2 so decel ramp cancels accel oscillation.
    """
    phi = t2 % Td
    # Distance to nearest multiple of Td/2
    err = min(abs(phi), abs(phi - Td / 2.0), abs(phi - Td))
    return err / (Td / 2.0)  # normalised to [0, 1]


# ---------------------------------------------------------------------------
# Profile builder
# ---------------------------------------------------------------------------

def build_profile(
    x0: float, xf: float,
    v: float, ta_decel: float,
    ta_accel: float, a_allowed: float,
    Td: float,
    t_ref: float,
    config: PlannerConfig,
) -> TrajectoryProfile | None:
    """
    Build a profile for given (v, ta_decel).
    Returns None if constraints are violated.
    """
    direction = math.copysign(1.0, xf - x0)
    dx = abs(xf - x0)

    # Acceleration
    a_accel = v / ta_accel
    if a_accel > a_allowed + 1e-9:
        return None

    # Deceleration
    a_decel = v / ta_decel
    if a_decel > a_allowed + 1e-9:
        return None

    # Cruise distance and duration
    # dx = 0.5*a_accel*ta_accel² + v*tc + 0.5*a_decel*ta_decel²
    d_accel = 0.5 * a_accel * ta_accel**2   # = 0.5 * v * ta_accel
    d_decel = 0.5 * a_decel * ta_decel**2   # = 0.5 * v * ta_decel

    tc_num = dx - d_accel - d_decel
    if tc_num < -1e-9:
        return None
    tc = max(0.0, tc_num) / v

    profile_type: Literal["triangular", "trapezoidal"] = (
        "triangular" if tc < 1e-6 else "trapezoidal"
    )

    t_total = ta_accel + tc + ta_decel
    t2 = ta_accel + tc  # start of decel ramp

    phi_err = phase_error_normalised(t2, Td)
    score = phi_err + config.lambda_time * t_total / t_ref

    return TrajectoryProfile(
        type=profile_type,
        x0=x0, xf=xf,
        direction=direction,
        ta_accel=ta_accel,
        v_peak=v * direction,
        tc=tc,
        ta_decel=ta_decel,
        a_accel=a_accel * direction,
        a_decel=a_decel * direction,
        t_total=t_total,
        phase_error_norm=phi_err,
        score=score,
    )


# ---------------------------------------------------------------------------
# Main planner
# ---------------------------------------------------------------------------

def plan(
    x0: float, xf: float,
    pendulum: PendulumParams,
    limits: KinematicLimits,
    config: PlannerConfig | None = None,
) -> TrajectoryProfile:
    """
    Plan a trajectory from x0 to xf minimising oscillation and travel time.
    theta_max is enforced both analytically (a_allowed) and via simulation penalty.
    """
    if config is None:
        config = PlannerConfig()

    dx = abs(xf - x0)
    if dx < 1e-9:
        raise ValueError("x0 == xf, nothing to plan.")

    # --- Physics ---
    _, _, Td, ta_star, _, a_theta_max = pendulum_dynamics(pendulum)
    a_allowed = min(limits.amax, a_theta_max)

    # --- Reference candidate (symmetric, ta_decel = ta_star) ---
    v_cand = min(limits.vmax, a_allowed * ta_star)
    a_cand = v_cand / ta_star
    dx_min = v_cand**2 / a_cand

    # t_ref: normalisation baseline for time score
    t_ref = 2.0 * ta_star + max(0.0, dx - dx_min) / v_cand

    # ----------------------------------------------------------------
    # Case A — triangular profile (dx <= dx_min)
    # ----------------------------------------------------------------
    if dx <= dx_min:
        a_tri = dx / ta_star**2
        v_tri = a_tri * ta_star
        direction = math.copysign(1.0, xf - x0)
        t2 = ta_star
        phi_err = phase_error_normalised(t2, Td)
        t_total = 2.0 * ta_star
        score = phi_err + config.lambda_time * t_total / t_ref
        return TrajectoryProfile(
            type="triangular",
            x0=x0, xf=xf,
            direction=direction,
            ta_accel=ta_star,
            v_peak=v_tri * direction,
            tc=0.0,
            ta_decel=ta_star,
            a_accel=a_tri * direction,
            a_decel=a_tri * direction,
            t_total=t_total,
            phase_error_norm=phi_err,
            score=score,
        )

    # ----------------------------------------------------------------
    # Case B — trapezoidal profile (dx > dx_min)
    # ----------------------------------------------------------------
    v_low = a_allowed * 1e-3
    v_high = limits.vmax

    ta_decel_low = config.ta_decel_ratio_bounds[0] * ta_star
    ta_decel_high = config.ta_decel_ratio_bounds[1] * ta_star

    def objective(params):
        v, ta_d = params
        prof = build_profile(x0, xf, v, ta_d, ta_star, a_allowed, Td, t_ref, config)
        if prof is None:
            return 1e6
    def objective(params):
        v, ta_d = params
        prof = build_profile(x0, xf, v, ta_d, ta_star, a_allowed, Td, t_ref, config)
        if prof is None:
            return 1e6
        return prof.score

    best_score = math.inf
    best_result = None

    v_starts = np.linspace(v_low * 10, v_high, 6)
    td_starts = np.linspace(ta_decel_low, ta_decel_high, 4)
    bounds = [(v_low, v_high), (ta_decel_low, ta_decel_high)]

    for v0 in v_starts:
        for td0 in td_starts:
            res = minimize(
                objective,
                x0=[v0, td0],
                method="L-BFGS-B",
                bounds=bounds,
                options={"ftol": 1e-10, "gtol": 1e-8, "maxiter": 200},
            )
            if res.fun < best_score:
                best_score = res.fun
                best_result = res

    v_opt, ta_decel_opt = best_result.x
    profile = build_profile(
        x0, xf, v_opt, ta_decel_opt, ta_star, a_allowed, Td, t_ref, config
    )

    if profile is None:
        raise RuntimeError("Optimisation converged to an infeasible point.")

    # --- Post-optimisation theta check ---
    # The analytical a_allowed underestimates swing for asymmetric profiles.
    # If theta_max is exceeded, tighten a_allowed with a safety factor and re-run.
    _, _, theta_reached = simulate_pendulum(profile, pendulum, dt=0.002)
    if theta_reached > pendulum.theta_max * 1.005:  # 0.5% tolerance
        safety = pendulum.theta_max / theta_reached  # < 1
        a_allowed_tight = a_allowed * safety
        v_high_tight = min(v_high, a_allowed_tight * ta_star)

        best_score = math.inf
        best_result = None
        bounds_tight = [(v_low, v_high_tight), (ta_decel_low, ta_decel_high)]

        for v0 in np.linspace(v_low * 10, v_high_tight, 6):
            for td0 in td_starts:
                res = minimize(
                    lambda p: (
                        build_profile(x0, xf, p[0], p[1], ta_star,
                                      a_allowed_tight, Td, t_ref, config) or
                        type('_', (), {'score': 1e6})()
                    ).score,
                    x0=[v0, td0],
                    method="L-BFGS-B",
                    bounds=bounds_tight,
                    options={"ftol": 1e-10, "gtol": 1e-8, "maxiter": 200},
                )
                if res.fun < best_score:
                    best_score = res.fun
                    best_result = res

        v_opt, ta_decel_opt = best_result.x
        profile = build_profile(
            x0, xf, v_opt, ta_decel_opt, ta_star, a_allowed_tight, Td, t_ref, config
        )
        if profile is None:
            raise RuntimeError("Re-optimisation after theta correction failed.")

    return profile


# ---------------------------------------------------------------------------
# Trajectory sampling
# ---------------------------------------------------------------------------

def sample_trajectory(profile: TrajectoryProfile, dt: float = 0.01):
    """
    Sample x(t), v(t), a(t) from a TrajectoryProfile.
    Returns arrays (t, x, v, a).
    """
    d = profile.direction
    ta = profile.ta_accel
    tc = profile.tc
    td = profile.ta_decel
    aa = abs(profile.a_accel)   # acceleration magnitude
    ad = abs(profile.a_decel)   # deceleration magnitude
    vp = abs(profile.v_peak)
    x0 = profile.x0

    t_total = profile.t_total
    t_arr = np.arange(0.0, t_total + dt, dt)
    x_arr = np.zeros_like(t_arr)
    v_arr = np.zeros_like(t_arr)
    a_arr = np.zeros_like(t_arr)

    t1 = ta
    t2 = ta + tc
    t3 = ta + tc + td

    for i, t in enumerate(t_arr):
        if t <= t1:
            # Acceleration ramp
            a = aa
            v = aa * t
            x = 0.5 * aa * t**2
        elif t <= t2:
            # Cruise
            dt_ = t - t1
            a = 0.0
            v = vp
            x = 0.5 * aa * t1**2 + vp * dt_
        else:
            # Deceleration ramp
            dt_ = t - t2
            a = -ad
            v = vp - ad * dt_
            x = (0.5 * aa * t1**2
                 + vp * tc
                 + vp * dt_ - 0.5 * ad * dt_**2)

        x_arr[i] = x0 + d * x
        v_arr[i] = d * v
        a_arr[i] = d * a

    return t_arr, x_arr, v_arr, a_arr


# ---------------------------------------------------------------------------
# Pendulum simulation
# ---------------------------------------------------------------------------

def simulate_pendulum(
    profile: TrajectoryProfile,
    pendulum: PendulumParams,
    dt: float = 0.001,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Simulate the pendulum angle θ(t) driven by the cart acceleration a(t).

    Equation of motion (linearised):
        θ'' + 2ζωn·θ' + ωn²·θ = -a_cart(t) / l

    Returns:
        t_arr   : time array [s]
        theta   : angle array [rad]
        theta_max_reached : max |θ| over the full trajectory [rad]
    """
    wn = math.sqrt(G / pendulum.l)

    t_arr, _, _, a_arr = sample_trajectory(profile, dt=dt)
    n = len(t_arr)

    theta = np.zeros(n)
    dtheta = np.zeros(n)

    for i in range(n - 1):
        # RK4 on state [θ, θ']
        a_cart = a_arr[i]

        def deriv(th, dth, ac):
            ddth = -2 * pendulum.zeta * wn * dth - wn**2 * th - ac / pendulum.l
            return dth, ddth

        th, dth = theta[i], dtheta[i]

        k1_th, k1_dth = deriv(th,              dth,              a_cart)
        k2_th, k2_dth = deriv(th + dt/2*k1_th, dth + dt/2*k1_dth, a_cart)
        k3_th, k3_dth = deriv(th + dt/2*k2_th, dth + dt/2*k2_dth, a_cart)
        k4_th, k4_dth = deriv(th + dt*k3_th,   dth + dt*k3_dth,   a_cart)

        theta[i+1]  = th  + dt/6 * (k1_th  + 2*k2_th  + 2*k3_th  + k4_th)
        dtheta[i+1] = dth + dt/6 * (k1_dth + 2*k2_dth + 2*k3_dth + k4_dth)

    theta_max_reached = float(np.max(np.abs(theta)))
    return t_arr, theta, theta_max_reached


# ---------------------------------------------------------------------------
# Quick summary
# ---------------------------------------------------------------------------

def print_profile(p: TrajectoryProfile, pendulum: PendulumParams | None = None):
    print(f"\n{'─'*50}")
    print(f"  Profile type    : {p.type}")
    print(f"  Direction       : {'→' if p.direction > 0 else '←'}")
    print(f"  ta_accel        : {p.ta_accel:.4f} s")
    print(f"  tc (cruise)     : {p.tc:.4f} s")
    print(f"  ta_decel        : {p.ta_decel:.4f} s")
    print(f"  t_total         : {p.t_total:.4f} s")
    print(f"  v_peak          : {abs(p.v_peak):.4f} m/s")
    print(f"  a_accel         : {abs(p.a_accel):.4f} m/s²")
    print(f"  a_decel         : {abs(p.a_decel):.4f} m/s²")
    print(f"  Phase error     : {p.phase_error_norm*100:.2f} %")
    print(f"  Score           : {p.score:.6f}")
    if pendulum is not None:
        _, _, theta_max_reached = simulate_pendulum(p, pendulum)
        margin = (pendulum.theta_max - theta_max_reached) / pendulum.theta_max * 100
        print(f"  θ_max reached   : {math.degrees(theta_max_reached):.3f}° "
              f"(limit {math.degrees(pendulum.theta_max):.1f}°, margin {margin:.1f}%)")
    print(f"{'─'*50}\n")
