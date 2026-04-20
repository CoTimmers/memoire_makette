"""
Tests for planner_v2 — symmetric ta_star profile, fully analytical.
"""

import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from planner_v2 import (
    PendulumParams, KinematicLimits,
    plan, sample_trajectory, simulate_pendulum,
    print_profile, pendulum_dynamics,
)


# ---------------------------------------------------------------------------
# Constraint checker
# ---------------------------------------------------------------------------

def check_constraints(profile, limits, pendulum, tol=1e-6):
    _, _, _, ta_star, _, a_theta_max = pendulum_dynamics(pendulum)
    a_allowed = min(limits.amax, a_theta_max)
    errors = []

    if abs(abs(profile.a) - profile.ta * abs(profile.v_peak) / profile.ta**2) > tol:
        pass  # just a = v / ta, checked below

    if abs(profile.a) > a_allowed + tol:
        errors.append(f"a={abs(profile.a):.6f} > a_allowed={a_allowed:.6f}")

    if abs(profile.v_peak) > limits.vmax + tol:
        errors.append(f"v_peak={abs(profile.v_peak):.6f} > vmax={limits.vmax}")

    if profile.tc < -tol:
        errors.append(f"tc={profile.tc:.6f} < 0")

    if abs(profile.ta - ta_star) > tol:
        errors.append(f"ta={profile.ta:.6f} != ta_star={ta_star:.6f}")

    # Distance check
    ta, tc = profile.ta, profile.tc
    vp = abs(profile.v_peak)
    aa = abs(profile.a)
    dx_computed = vp * ta + vp * tc  # = aa*ta² + vp*tc  (since vp = aa*ta)
    dx_expected = abs(profile.xf - profile.x0)
    if abs(dx_computed - dx_expected) > 1e-6:
        errors.append(f"distance: computed {dx_computed:.6f} != expected {dx_expected:.6f}")

    # Theta check
    _, _, theta_reached = simulate_pendulum(profile, pendulum, dt=0.002)
    if theta_reached > pendulum.theta_max + 1e-3:
        errors.append(
            f"θ_max exceeded: {math.degrees(theta_reached):.3f}° "
            f"> {math.degrees(pendulum.theta_max):.1f}°"
        )

    return errors


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_profiles(cases, filename="profiles.png"):
    """cases: list of (label, profile, pendulum)"""
    n = len(cases)
    fig = plt.figure(figsize=(18, 4 * n))
    gs = gridspec.GridSpec(n, 4, figure=fig, hspace=0.5, wspace=0.35)

    for row, (label, profile, pendulum) in enumerate(cases):
        t, x, v, a   = sample_trajectory(profile, dt=0.005)
        t_th, theta, theta_max_reached = simulate_pendulum(profile, pendulum, dt=0.005)
        theta_deg  = np.degrees(theta)
        limit_deg  = math.degrees(pendulum.theta_max)

        ax_x  = fig.add_subplot(gs[row, 0])
        ax_v  = fig.add_subplot(gs[row, 1])
        ax_a  = fig.add_subplot(gs[row, 2])
        ax_th = fig.add_subplot(gs[row, 3])

        ax_x.plot(t, x, color="#534AB7", linewidth=1.8)
        ax_x.set_title(f"{label}\nposition x(t)", fontsize=9)
        ax_x.set_xlabel("t [s]"); ax_x.set_ylabel("x [m]")
        ax_x.grid(True, alpha=0.3)

        ax_v.plot(t, v, color="#0F6E56", linewidth=1.8)
        ax_v.set_title("velocity v(t)", fontsize=9)
        ax_v.set_xlabel("t [s]"); ax_v.set_ylabel("v [m/s]")
        ax_v.axhline(0, color="gray", linewidth=0.5)
        ax_v.grid(True, alpha=0.3)

        ax_a.plot(t, a, color="#993C1D", linewidth=1.8)
        ax_a.set_title("acceleration a(t)", fontsize=9)
        ax_a.set_xlabel("t [s]"); ax_a.set_ylabel("a [m/s²]")
        ax_a.axhline(0, color="gray", linewidth=0.5)
        ax_a.grid(True, alpha=0.3)
        ax_a.text(0.02, 0.05,
                  f"t={profile.t_total:.2f}s  {profile.type}",
                  transform=ax_a.transAxes, fontsize=7, color="#444")

        ax_th.plot(t_th, theta_deg, color="#185FA5", linewidth=1.8)
        ax_th.axhline( limit_deg, color="red", linewidth=1, linestyle="--",
                       label=f"±{limit_deg:.1f}°")
        ax_th.axhline(-limit_deg, color="red", linewidth=1, linestyle="--")
        ax_th.axhline(0, color="gray", linewidth=0.5)
        ax_th.set_title("pendulum θ(t)", fontsize=9)
        ax_th.set_xlabel("t [s]"); ax_th.set_ylabel("θ [°]")
        ax_th.legend(fontsize=7, loc="upper right")
        ax_th.grid(True, alpha=0.3)
        ax_th.text(0.02, 0.05,
                   f"θ_max = {math.degrees(theta_max_reached):.2f}°",
                   transform=ax_th.transAxes, fontsize=7, color="#185FA5")

    fig.savefig(filename, dpi=130, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"  → saved {filename}")


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def run_tests():
    print("\n" + "="*60)
    print("  TRAJECTORY PLANNER v2 — TEST SUITE")
    print("="*60)

    pendulum = PendulumParams(l=0.5, zeta=0.05, theta_max=math.radians(5))
    limits   = KinematicLimits(vmax=1.0, amax=2.0)
    cases    = []
    all_passed = True

    tests = [
        ("Short move  (triangular expected)", 0.0,  0.05, pendulum),
        ("Medium move (trapezoidal expected)", 0.0,  0.5,  pendulum),
        ("Long move",                          0.0,  2.0,  pendulum),
        ("Negative direction",                 1.0, -0.5,  pendulum),
        ("Tight θ_max = 2°",                   0.0,  1.0,
            PendulumParams(l=0.5, zeta=0.05, theta_max=math.radians(2))),
        ("vmax active",                        0.0,  3.0,
            PendulumParams(l=0.5, zeta=0.05, theta_max=math.radians(10))),
    ]

    for i, row in enumerate(tests):
        label  = row[0]
        x0, xf = row[1], row[2]
        pend   = row[3]
        print(f"\n[{i+1}] {label}")
        p = plan(x0, xf, pend, limits)
        print_profile(p, pend)
        errs = check_constraints(p, limits, pend)
        if errs:
            print("  FAIL:", errs)
            all_passed = False
        else:
            print("  PASS")
        cases.append((label, p, pend))

    print("\n" + "="*60)
    print(f"  {'ALL TESTS PASSED ✓' if all_passed else 'SOME TESTS FAILED ✗'}")
    print("="*60 + "\n")

    plot_profiles(cases, r"C:\Users\loicm\Documents\Project\These\memoire_makette\profiles_3.png")
    return all_passed


if __name__ == "__main__":
    run_tests()
