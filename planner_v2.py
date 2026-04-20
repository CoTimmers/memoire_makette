"""
Trajectory planner with pendulum anti-oscillation.

Profile: trapezoidal or triangular, always symmetric:
    ta_accel = ta_decel = ta_star = Td / 2

This guarantees exact phase cancellation of pendulum oscillation.
The only free variable is v (or equivalently tc), maximised for minimum time.

Inputs:
    x0, xf          : start and end positions [m]
    l, zeta          : pendulum length [m] and damping ratio [-]
    theta_max        : max allowed swing angle [rad]
    vmax, amax       : kinematic limits [m/s], [m/s²]
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import Literal

G = 9.81  # m/s²


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PendulumParams:
    l: float            # pendulum length [m]
    zeta: float         # damping ratio [-]
    theta_max: float    # max allowed swing angle [rad]


@dataclass
class KinematicLimits:
    vmax: float         # max velocity [m/s]
    amax: float         # max acceleration [m/s²]


@dataclass
class TrajectoryProfile:
    type: Literal["triangular", "trapezoidal"]
    x0: float
    xf: float
    direction: float    # +1 or -1
    ta: float           # accel = decel duration [s]  (= ta_star)
    v_peak: float       # signed peak velocity [m/s]
    tc: float           # cruise duration [s]
    a: float            # signed acceleration [m/s²]
    t_total: float      # total duration [s]


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------

def pendulum_dynamics(p: PendulumParams):
    """Return (wn, wd, Td, ta_star, Mp, a_allowed)."""
    wn  = math.sqrt(G / p.l)
    wd  = wn * math.sqrt(1.0 - p.zeta**2)
    Td  = 2.0 * math.pi / wd
    ta_star = Td / 2.0
    Mp  = math.exp(-p.zeta * math.pi / math.sqrt(1.0 - p.zeta**2))
    a_theta_max = G * p.theta_max / (1.0 + Mp)
    return wn, wd, Td, ta_star, Mp, a_theta_max


# ---------------------------------------------------------------------------
# Planner  (fully analytical)
# ---------------------------------------------------------------------------

def plan(
    x0: float,
    xf: float,
    pendulum: PendulumParams,
    limits: KinematicLimits,
) -> TrajectoryProfile:
    """
    Compute the time-optimal trajectory from x0 to xf under:
        - v  <= vmax
        - a  <= a_allowed  (= min(amax, a_theta_max))
        - ta_accel = ta_decel = ta_star  (exact anti-oscillation)

    a_theta_max from Mp formula is an approximation; we correct it
    with one simulation pass if theta_max is exceeded.
    """
    dx = abs(xf - x0)
    if dx < 1e-9:
        raise ValueError("x0 == xf, nothing to plan.")

    _, _, _, ta_star, _, a_theta_max = pendulum_dynamics(pendulum)
    a_allowed = min(limits.amax, a_theta_max)

    def _build(a_lim):
        v_opt = min(limits.vmax, a_lim * ta_star)
        a_opt = v_opt / ta_star
        dx_ramps = v_opt * ta_star
        direction = math.copysign(1.0, xf - x0)
        if dx <= dx_ramps:
            a_tri = dx / ta_star**2
            v_tri = a_tri * ta_star
            return TrajectoryProfile(
                type="triangular", x0=x0, xf=xf, direction=direction,
                ta=ta_star, v_peak=direction * v_tri, tc=0.0,
                a=direction * a_tri, t_total=2.0 * ta_star,
            )
        else:
            tc = (dx - dx_ramps) / v_opt
            return TrajectoryProfile(
                type="trapezoidal", x0=x0, xf=xf, direction=direction,
                ta=ta_star, v_peak=direction * v_opt, tc=tc,
                a=direction * a_opt, t_total=2.0 * ta_star + tc,
            )

    profile = _build(a_allowed)

    # --- Simulation-based correction ---
    # Mp formula slightly underestimates swing for damped systems.
    # Iterate until theta_max is respected (converges in 2-3 passes typically).
    for _ in range(10):
        _, _, theta_reached = simulate_pendulum(profile, pendulum, dt=0.002)
        if theta_reached <= pendulum.theta_max * 1.001:
            break
        safety    = pendulum.theta_max / theta_reached
        a_allowed = a_allowed * safety
        profile   = _build(a_allowed)

    return profile


# ---------------------------------------------------------------------------
# Trajectory sampling
# ---------------------------------------------------------------------------

def sample_trajectory(
    profile: TrajectoryProfile,
    dt: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample x(t), v(t), a(t) from a TrajectoryProfile.
    Returns (t, x, v, a).
    """
    d  = profile.direction
    ta = profile.ta
    tc = profile.tc
    vp = abs(profile.v_peak)
    aa = abs(profile.a)

    t_arr = np.arange(0.0, profile.t_total + dt, dt)
    x_arr = np.zeros_like(t_arr)
    v_arr = np.zeros_like(t_arr)
    a_arr = np.zeros_like(t_arr)

    t1 = ta
    t2 = ta + tc

    for i, t in enumerate(t_arr):
        if t <= t1:
            a_ = aa
            v_ = aa * t
            x_ = 0.5 * aa * t**2
        elif t <= t2:
            dt_ = t - t1
            a_ = 0.0
            v_ = vp
            x_ = 0.5 * aa * t1**2 + vp * dt_
        else:
            dt_ = t - t2
            a_ = -aa
            v_ = vp - aa * dt_
            x_ = 0.5 * aa * t1**2 + vp * tc + vp * dt_ - 0.5 * aa * dt_**2

        x_arr[i] = profile.x0 + d * x_
        v_arr[i] = d * v_
        a_arr[i] = d * a_

    return t_arr, x_arr, v_arr, a_arr


# ---------------------------------------------------------------------------
# Pendulum simulation  (RK4)
# ---------------------------------------------------------------------------

def simulate_pendulum(
    profile: TrajectoryProfile,
    pendulum: PendulumParams,
    dt: float = 0.001,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Simulate θ(t) driven by cart acceleration a(t).

    Equation of motion (linearised):
        θ'' + 2ζωn θ' + ωn² θ = -a_cart(t) / l

    Returns (t, theta [rad], theta_max_reached [rad]).
    """
    wn = math.sqrt(G / pendulum.l)

    t_arr, _, _, a_arr = sample_trajectory(profile, dt=dt)
    n = len(t_arr)

    theta  = np.zeros(n)
    dtheta = np.zeros(n)

    def deriv(th, dth, ac):
        ddth = -2.0 * pendulum.zeta * wn * dth - wn**2 * th - ac / pendulum.l
        return dth, ddth

    for i in range(n - 1):
        ac = a_arr[i]
        th, dth = theta[i], dtheta[i]

        k1_t, k1_d = deriv(th,                dth,                ac)
        k2_t, k2_d = deriv(th + dt/2*k1_t,    dth + dt/2*k1_d,    ac)
        k3_t, k3_d = deriv(th + dt/2*k2_t,    dth + dt/2*k2_d,    ac)
        k4_t, k4_d = deriv(th + dt*k3_t,      dth + dt*k3_d,      ac)

        theta[i+1]  = th  + dt/6 * (k1_t + 2*k2_t + 2*k3_t + k4_t)
        dtheta[i+1] = dth + dt/6 * (k1_d + 2*k2_d + 2*k3_d + k4_d)

    theta_max_reached = float(np.max(np.abs(theta)))
    return t_arr, theta, theta_max_reached


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_profile(profile: TrajectoryProfile, pendulum: PendulumParams | None = None):
    p = profile
    print(f"\n{'─'*52}")
    print(f"  type        : {p.type}")
    print(f"  direction   : {'→' if p.direction > 0 else '←'}")
    print(f"  ta          : {p.ta:.4f} s   (= ta_star)")
    print(f"  tc          : {p.tc:.4f} s")
    print(f"  t_total     : {p.t_total:.4f} s")
    print(f"  v_peak      : {abs(p.v_peak):.4f} m/s")
    print(f"  a           : {abs(p.a):.4f} m/s²")
    if pendulum is not None:
        _, _, th_max = simulate_pendulum(p, pendulum)
        margin = (pendulum.theta_max - th_max) / pendulum.theta_max * 100
        print(f"  θ_max       : {math.degrees(th_max):.3f}°  "
              f"(limit {math.degrees(pendulum.theta_max):.1f}°, margin {margin:.1f}%)")
    print(f"{'─'*52}\n")
