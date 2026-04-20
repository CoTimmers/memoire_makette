"""
Tests and usage examples for the trajectory planner.
"""

import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from planner import (
    PendulumParams, KinematicLimits, PlannerConfig,
    plan, sample_trajectory, simulate_pendulum, print_profile, pendulum_dynamics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check_constraints(profile, limits, pendulum, tol=1e-3):
    _, _, _, _, _, a_theta_max = pendulum_dynamics(pendulum)
    a_allowed = min(limits.amax, a_theta_max)
    errors = []
    if abs(profile.a_accel) > a_allowed + tol:
        errors.append(f"a_accel {abs(profile.a_accel):.4f} > a_allowed {a_allowed:.4f}")
    if abs(profile.a_decel) > a_allowed + tol:
        errors.append(f"a_decel {abs(profile.a_decel):.4f} > a_allowed {a_allowed:.4f}")
    if abs(profile.v_peak) > limits.vmax + tol:
        errors.append(f"v_peak {abs(profile.v_peak):.4f} > vmax {limits.vmax:.4f}")
    if profile.tc < -tol:
        errors.append(f"tc {profile.tc:.4f} < 0")
    # Check distance
    ta = profile.ta_accel
    tc = profile.tc
    td = profile.ta_decel
    v = abs(profile.v_peak)
    aa = abs(profile.a_accel)
    ad = abs(profile.a_decel)
    dx_computed = 0.5 * aa * ta**2 + v * tc + 0.5 * ad * td**2
    dx_expected = abs(profile.xf - profile.x0)
    if abs(dx_computed - dx_expected) > tol:
        errors.append(f"distance mismatch: computed {dx_computed:.4f} vs expected {dx_expected:.4f}")
    # Check simulated theta_max
    _, _, theta_reached = simulate_pendulum(profile, pendulum)
    if theta_reached > pendulum.theta_max + tol:
        errors.append(
            f"θ_max exceeded: {math.degrees(theta_reached):.3f}° > {math.degrees(pendulum.theta_max):.1f}°"
        )
    return errors


def plot_profiles(cases, filename="profiles.png"):
    n = len(cases)
    fig = plt.figure(figsize=(14, 4 * n))
    gs = gridspec.GridSpec(n, 3, figure=fig, hspace=0.5, wspace=0.35)

    for row, (label, profile) in enumerate(cases):
        t, x, v, a = sample_trajectory(profile, dt=0.005)

        ax_x = fig.add_subplot(gs[row, 0])
        ax_v = fig.add_subplot(gs[row, 1])
        ax_a = fig.add_subplot(gs[row, 2])

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

        # Annotate phase error
        ax_a.text(
            0.02, 0.05,
            f"φ_err={profile.phase_error_norm*100:.1f}%  "
            f"t={profile.t_total:.2f}s  "
            f"{profile.type}",
            transform=ax_a.transAxes, fontsize=7, color="#444"
        )

    fig.savefig(filename, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved {filename}")


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def run_tests():
    print("\n" + "="*60)
    print("  TRAJECTORY PLANNER — TEST SUITE")
    print("="*60)

    # Common pendulum and limits
    pendulum = PendulumParams(l=0.5, zeta=0.05, theta_max=math.radians(5))
    limits   = KinematicLimits(vmax=1.0, amax=2.0)

    cases_to_plot = []
    all_passed = True

    # --- Test 1: short move (should produce triangular profile) ---
    print("\n[1] Short move — expect triangular")
    cfg = PlannerConfig(lambda_time=0.1)
    p = plan(0.0, 0.05, pendulum, limits, cfg)
    print_profile(p, pendulum)
    errs = check_constraints(p, limits, pendulum)
    if errs:
        print("  FAIL:", errs); all_passed = False
    else:
        print("  PASS — all constraints satisfied")
    cases_to_plot.append(("Short move (0→0.05 m)", p))

    # --- Test 2: medium move (trapezoidal expected) ---
    print("\n[2] Medium move — expect trapezoidal")
    p = plan(0.0, 0.5, pendulum, limits, cfg)
    print_profile(p, pendulum)
    errs = check_constraints(p, limits, pendulum)
    if errs:
        print("  FAIL:", errs); all_passed = False
    else:
        print("  PASS — all constraints satisfied")
    cases_to_plot.append(("Medium move (0→0.5 m)", p))

    # --- Test 3: long move ---
    print("\n[3] Long move")
    p = plan(0.0, 2.0, pendulum, limits, cfg)
    print_profile(p, pendulum)
    errs = check_constraints(p, limits, pendulum)
    if errs:
        print("  FAIL:", errs); all_passed = False
    else:
        print("  PASS — all constraints satisfied")
    cases_to_plot.append(("Long move (0→2 m)", p))

    # --- Test 4: negative direction ---
    print("\n[4] Negative direction")
    p = plan(1.0, -0.5, pendulum, limits, cfg)
    print_profile(p, pendulum)
    errs = check_constraints(p, limits, pendulum)
    if errs:
        print("  FAIL:", errs); all_passed = False
    else:
        print("  PASS — all constraints satisfied")
    cases_to_plot.append(("Negative direction (1→-0.5 m)", p))

    # --- Test 5: tight theta_max constraint ---
    print("\n[5] Tight swing constraint (theta_max = 2°)")
    pendulum_strict = PendulumParams(l=0.5, zeta=0.05, theta_max=math.radians(2))
    p = plan(0.0, 1.0, pendulum_strict, limits, cfg)
    print_profile(p, pendulum_strict)
    errs = check_constraints(p, limits, pendulum_strict)
    if errs:
        print("  FAIL:", errs); all_passed = False
    else:
        print("  PASS — all constraints satisfied")
    cases_to_plot.append(("Strict swing (θ_max=2°, 0→1 m)", p))

    # --- Test 6: pure anti-oscillation (lambda_time = 0) ---
    print("\n[6] Pure anti-oscillation mode (λ=0)")
    cfg_anti = PlannerConfig(lambda_time=0.0)
    p = plan(0.0, 1.0, pendulum, limits, cfg_anti)
    print_profile(p, pendulum)
    errs = check_constraints(p, limits, pendulum)
    if errs:
        print("  FAIL:", errs); all_passed = False
    else:
        print("  PASS — all constraints satisfied")
        if p.phase_error_norm < 0.01:
            print("  BONUS — near-zero phase error achieved")
    cases_to_plot.append(("Pure anti-osc λ=0 (0→1 m)", p))

    # --- Test 7: pure speed (lambda_time = 1) ---
    print("\n[7] Pure speed mode (λ=1)")
    cfg_fast = PlannerConfig(lambda_time=1.0)
    p = plan(0.0, 1.0, pendulum, limits, cfg_fast)
    print_profile(p, pendulum)
    errs = check_constraints(p, limits, pendulum)
    if errs:
        print("  FAIL:", errs); all_passed = False
    else:
        print("  PASS — all constraints satisfied")
    cases_to_plot.append(("Pure speed λ=1 (0→1 m)", p))

# --- Summary ---
    print("\n" + "="*60)
    print(f"  {'ALL TESTS PASSED ✓' if all_passed else 'SOME TESTS FAILED ✗'}")
    print("="*60 + "\n")

    plot_profiles(cases_to_plot, r"C:\Users\loicm\Documents\Project\These\memoire_makette\profiles.png")

    return all_passed




if __name__ == "__main__":
    run_tests()
