"""
controller.py — Phase-based controller.

Professor's guidance
--------------------
Do NOT think of the control as "choosing (fx, fy) directly", because in
practice we can never know the exact forces the robot applies.
Instead, think in terms of POTENTIAL ENERGY INJECTION:
  - For each phase, identify which direction of energy injection produces
    the desired motion (rotation, sliding, ...).
  - The force (fx, fy) is the gradient of that potential energy w.r.t.
    the CoM position, i.e. it points in the geometrically meaningful direction.
  - Only the direction matters for the strategy; magnitude is a tuning knob.

Phase strategies
----------------
  APPROACH   : NO force -- camera observes (x, y, theta, omega). Instant transition.

  PUSH       : Force perpendicular to wall 1 (pure -y in world frame).
               This is the simplest potential energy injection: gravity-like
               push toward y = 0.  omega stays constant (free flight, no torque).
               Transition: when a corner reaches wall 1.

  CONTACT    : One corner is the pivot on wall 1. Same -y force continues.
               The force creates a torque about the pivot -> box rotates.
               Direction of rotation depends on which corner is the pivot and
               the sign of omega at impact (trend, not exact).
               Transition: when a full side lies flat on wall 1.

  COINCEMENT : inject energy toward the corner (both walls) to wedge the box.
  PIVOTEMENT : push along wall-2 inward normal (-nB direction).
  GLISSEMENT : push mostly downward with small +x component.
  FINAL      : same energy direction as GLISSEMENT.
"""
import numpy as np
from state import CrateState, SimParams, Mode
from dynamics import wall_profile


def get_command(state: CrateState, t: float, params: SimParams) -> tuple:
    """
    Return (fx, fy) -- force applied at the crate's centre of mass.
    """
    if state.mode == Mode.APPROACH:
        # Observation only -- camera reads the state, no action.
        return 0.0, 0.0

    elif state.mode == Mode.PUSH:
        # Energy injection perpendicular to wall 1 (-y direction in world frame).
        # Wall 1 is horizontal (y = 0), so its inward normal is (0, -1).
        return 0.0, -params.F_approach

    elif state.mode == Mode.CONTACT:
        # Same energy direction as PUSH: keep pressing toward wall 1.
        # The torque this creates about the pivot corner drives the rotation
        # that aligns a side with wall 1.
        return 0.0, -params.F_approach

    elif state.mode == Mode.COINCEMENT:
        psi, _, _ = wall_profile(t, params)
        nB = np.array([np.sin(psi), -np.cos(psi)])
        fx = -params.F_pivot * nB[0]
        fy = -params.F_pivot * nB[1] - params.F_pivot * 0.3
        return fx, fy

    elif state.mode == Mode.PIVOTEMENT:
        psi, _, _ = wall_profile(t, params)
        nB = np.array([np.sin(psi), -np.cos(psi)])
        fx = -params.F_pivot * nB[0]
        fy = -params.F_pivot * nB[1]
        return fx, fy

    elif state.mode == Mode.GLISSEMENT:
        fx = params.F_slide * 0.2
        fy = -params.F_slide
        return fx, fy

    elif state.mode == Mode.FINAL:
        fx = params.F_slide * 0.2
        fy = -params.F_slide
        return fx, fy

    return 0.0, 0.0
