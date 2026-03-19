"""
controller.py — Phase-based controller.

Professor's guidance
--------------------
Do NOT think of the control as "choosing (fx, fy) directly", because in
practice we can never know the exact forces the robot applies.
Instead, think in terms of POTENTIAL ENERGY INJECTION:
  - For each phase, identify which direction of energy injection produces
    the desired motion (rotation, sliding, …).
  - The force (fx, fy) is the gradient of that potential energy w.r.t.
    the CoM position, i.e. it points in the geometrically meaningful direction.
  - Only the direction matters for the strategy; magnitude is a tuning knob.

Phase strategies (from mindmap)
--------------------------------
  APPROACH   : push box toward wall 1 (downward component) to initiate contact.
  COINCEMENT : inject energy toward the corner (both walls) to wedge the box.
  PIVOTEMENT : push along wall-2 inward normal (-nB direction).
               This forces B against wall 2 and lets the wall's rotation do
               work on the box → the box rotates around A.
  GLISSEMENT : push mostly downward with small +x component.
               This maintains wall-1 contact and slides A in +x (away from
               the closing wall 2).
  FINAL      : same energy direction as GLISSEMENT but the monitor decides
               when to stop (wall 2 has closed, box reaches final position).
"""
import numpy as np
from state import CrateState, SimParams, Mode
from dynamics import wall_profile


def get_command(state: CrateState, t: float, params: SimParams) -> tuple:
    """
    Return (fx, fy) — force applied at the crate's centre of mass.

    The direction is chosen from geometric/energy reasoning (see module doc).
    The magnitude is set by SimParams.F_pivot or F_slide.
    """
    psi, _, _ = wall_profile(t, params)

    if state.mode == Mode.APPROACH:
        # Energy injection: push box toward wall 1.
        # Small +x to approach the corner, stronger -y to contact wall 1.
        fx = params.F_pivot * 0.1
        fy = -params.F_pivot * 0.5
        return fx, fy

    elif state.mode == Mode.COINCEMENT:
        # Energy injection: toward the corner formed by wall 1 (y=0) and
        # wall 2 (rotating).  Push in the -nB direction to press B against
        # wall 2, and -y to keep A on wall 1.
        nB = np.array([np.sin(psi), -np.cos(psi)])  # inward normal of wall 2
        fx = -params.F_pivot * nB[0]
        fy = -params.F_pivot * nB[1] - params.F_pivot * 0.3
        return fx, fy

    elif state.mode == Mode.PIVOTEMENT:
        # Energy injection along wall-2 inward normal (-nB).
        # Physical interpretation: this force component, transmitted through
        # the B-wall2 contact, produces a torque around A that rotates the box.
        # (Identical to phase-1 strategy in moteur_dynamique_discrétisee.py)
        nB = np.array([np.sin(psi), -np.cos(psi)])
        fx = -params.F_pivot * nB[0]
        fy = -params.F_pivot * nB[1]
        return fx, fy

    elif state.mode == Mode.GLISSEMENT:
        # Energy injection to slide the box along wall 1 away from wall 2.
        # Small +x (along wall 1) + strong -y (maintain wall-1 contact).
        # (Matches phase-2 strategy in moteur_dynamique_discrétisee.py)
        fx = params.F_slide * 0.2
        fy = -params.F_slide
        return fx, fy

    elif state.mode == Mode.FINAL:
        # Wall 2 is closing; box slides to its final resting position.
        # Same energy direction as GLISSEMENT — the monitor stops the simulation
        # when the geometric target is reached.
        fx = params.F_slide * 0.2
        fy = -params.F_slide
        return fx, fy

    # Default: no force (should not be reached in normal operation)
    return 0.0, 0.0
