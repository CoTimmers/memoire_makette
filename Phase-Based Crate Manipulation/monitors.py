"""
monitors.py — Geometry utilities and phase-transition detection.

Each phase has a monitor that checks whether the transition condition
is satisfied (camera check in the real system, geometric check here).

Phase graph (from mindmap)
--------------------------
  APPROACH
    └─ A touches wall 1 → COINCEMENT
  COINCEMENT
    └─ B touches wall 2 (box wedged) → PIVOTEMENT
  PIVOTEMENT
    └─ psi >= psi_switch (≈ 127°) → GLISSEMENT
  GLISSEMENT
    └─ l >= l_slide_end → FINAL
  FINAL
    └─ l <= l_final_end → None  (simulation done)
"""
import numpy as np
from state import CrateState, SimParams, Mode, get_corners_world
from dynamics import wall_profile, delta_kinematics, rot


# ── Geometry ───────────────────────────────────────────────────────────────────

def rotation_matrix(theta: float) -> np.ndarray:
    """2-D rotation matrix (alias kept for backward compatibility)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def get_box_geometry(state: CrateState, t: float, params: SimParams) -> dict:
    """
    Compute world-frame positions of key crate points at the current state.

    Returns a dict with:
        psi   : wall-2 angle (rad)
        delta : crate orientation (rad)
        A     : corner A position (on wall 1)
        B     : corner B position (on wall 2)
        O     : centre-of-mass position
        corners : (5, 2) array of crate vertices for plotting
    """
    psi, psidot, _ = wall_profile(t, params)
    delta, *_ = delta_kinematics(state.l, state.ldot, psi, psidot, params)
    R = rot(delta)

    xA, yA = state.l, 0.0
    rAO = R @ params.rAO_body
    rAB = R @ params.rAB_body
    xO, yO = xA + rAO[0], yA + rAO[1]
    xB, yB = xA + rAB[0], yA + rAB[1]

    # All 4 corners of the crate (body frame, starting from A)
    a, b = params.a, params.b
    local = np.array([[0, 0], [a, 0], [a, b], [0, b], [0, 0]], dtype=float)
    world_corners = np.array([np.array([xA, yA]) + R @ c for c in local])

    return {
        'psi':     psi,
        'delta':   delta,
        'A':       np.array([xA, yA]),
        'B':       np.array([xB, yB]),
        'O':       np.array([xO, yO]),
        'corners': world_corners,
    }


def dist_point_to_wall2(x: float, y: float, psi: float) -> float:
    """
    Signed distance of point (x, y) from wall 2.

    Wall 2 passes through the origin with inward normal nB = [sin(psi), -cos(psi)].
    Positive value = point is on the crate side (inside the closing angle).
    """
    return x * np.sin(psi) - y * np.cos(psi)


# ── APPROACH / PUSH / CONTACT monitors ────────────────────────────────────────

def first_corner_on_wall1(state: CrateState, params: SimParams) -> int:
    """
    Return the index (0-3) of the lowest corner if it has reached wall 1 (y ≤ 0),
    or -1 if no corner is in contact yet.

    Corner order (matches params.corners_body):
      0: bottom-left  1: bottom-right  2: top-right  3: top-left
    """
    corners = get_corners_world(state, params)
    y_vals  = corners[:, 1]
    idx     = int(np.argmin(y_vals))
    return idx if y_vals[idx] <= 0.0 else -1


def side_flat_on_wall1(state: CrateState, params: SimParams) -> bool:
    """
    Return True when two adjacent corners are both within side_flat_tol of wall 1.
    This signals that a full side is lying on wall 1 → end of CONTACT phase.
    """
    corners = get_corners_world(state, params)
    y_vals  = corners[:, 1]
    tol     = params.side_flat_tol
    for i in range(4):
        j = (i + 1) % 4
        if abs(y_vals[i]) < tol and abs(y_vals[j]) < tol:
            return True
    return False


# ── Phase transition monitors ──────────────────────────────────────────────────

def check_transition(state: CrateState, t: float, params: SimParams):
    """
    Evaluate whether the current phase should transition.

    Returns:
        new_mode (Mode) : the phase to switch to, or
        None            : simulation is complete (FINAL reached target).
    If no transition is needed, returns state.mode.

    Monitor logic (mindmap):
      APPROACH   → COINCEMENT  : camera detects A touching wall 1 (l ≈ 0)
      COINCEMENT → PIVOTEMENT  : camera detects B touching wall 2
      PIVOTEMENT → GLISSEMENT  : psi >= psi_switch (box has rotated ≈ 60°)
      GLISSEMENT → FINAL       : camera detects box slid far enough (l >= threshold)
      FINAL      → done        : camera detects final position reached (l <= threshold)
    """
    psi, _, _ = wall_profile(t, params)

    if state.mode == Mode.APPROACH:
        # Camera observes the crate state (x, y, theta, omega).
        # Transition is immediate: state is always "known" in simulation.
        return Mode.PUSH

    elif state.mode == Mode.PUSH:
        # Detect the first corner touching wall 1.
        idx = first_corner_on_wall1(state, params)
        if idx >= 0:
            return Mode.CONTACT

    elif state.mode == Mode.CONTACT:
        # Detect when a full side is flat on wall 1.
        if side_flat_on_wall1(state, params):
            return Mode.COINCEMENT

    elif state.mode == Mode.COINCEMENT:
        # Monitor: B must be within contact_tol of wall 2
        geo = get_box_geometry(state, t, params)
        xB, yB = geo['B']
        d = abs(dist_point_to_wall2(xB, yB, psi))
        if d < params.contact_tol:
            return Mode.PIVOTEMENT

    elif state.mode == Mode.PIVOTEMENT:
        # Monitor: wall 2 has closed past the pivot threshold
        # (in the real system: camera verifies the box has rotated ≈ 60°)
        if psi >= params.psi_switch:
            return Mode.GLISSEMENT

    elif state.mode == Mode.GLISSEMENT:
        # Monitor: box has slid far enough along wall 1 (camera check)
        if state.l >= params.l_slide_end:
            return Mode.FINAL

    elif state.mode == Mode.FINAL:
        # Monitor: box reached final position (camera + wall constraint check)
        if state.l <= params.l_final_end:
            return None   # simulation complete

    return state.mode
