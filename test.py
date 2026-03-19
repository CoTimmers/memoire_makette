"""
state.py — State definitions for the phase-based crate manipulation system.

Coordinate system
-----------------
  Wall 1 : y = 0  (horizontal, fixed)
  Wall 2 : x = 0  (vertical, fixed)
  The crate lives in x > 0, y > 0.

Body frame convention
---------------------
  The crate is a rectangle (a × b).
  Corner A = bottom-left in body frame  → body coords (0, 0)
  Corner B = top-left                   → body coords (0, b)
  Corner C = top-right                  → body coords (a, b)
  Corner D = bottom-right               → body coords (a, 0)
  CoM O    = centre                     → body coords (a/2, b/2)

  theta = angle of body x-axis w.r.t. world x-axis (rad).

Phase sequence
--------------
  APPROACH  : crate is free in 2-D, no wall contact.
              State: (x, y, theta, vx, vy, omega)
              Command: force (fx, fy) applied at CoM, directed toward wall 1.

  CONTACT   : one corner touches wall 1 (y=0) after an impact.
              The corner becomes a pivot; the crate rotates around it.
              State: (theta, omega)  — pivot position is stored separately.
              Command: force at CoM continues to inject energy.
              Ends when a full side lies on wall 1 → COINCEMENT.
"""

from __future__ import annotations
from enum import Enum, auto
from dataclasses import dataclass, field
import numpy as np


# ── Phase enumeration ──────────────────────────────────────────────────────────

class Mode(Enum):
    """
    Phase sequence.

    APPROACH   : free-flight, no contact.
    CONTACT    : single-corner pivot on wall 1, may include bounces.
    COINCEMENT : full side on wall 1, box wedged toward corner.
    PIVOTEMENT : box rotates ~60° around corner A (A on wall 1, B on wall 2).
    GLISSEMENT : long side on wall 1, box sliding away from wall 2.
    FINAL      : wall 2 closes, box slides to final position.
    """
    APPROACH    = auto()
    CONTACT     = auto()
    COINCEMENT  = auto()
    PIVOTEMENT  = auto()
    GLISSEMENT  = auto()
    FINAL       = auto()


# ── Crate state ────────────────────────────────────────────────────────────────

@dataclass
class CrateState:
    """
    Full rigid-body state of the crate.

    APPROACH phase  →  all 6 fields are active (free body, 3 DOF).
    CONTACT  phase  →  (x, y) is the pivot corner position (fixed on wall 1);
                       only theta and omega evolve.
    Later phases    →  reduced coordinates handled in their own modules.

    Fields
    ------
    x, y      : world-frame CoM position (m)
    theta     : body orientation — angle of body x-axis w.r.t. world x (rad)
    vx, vy    : CoM velocity (m/s)
    omega     : angular velocity (rad/s)
    mode      : current phase
    pivot_idx : index of the corner acting as pivot (0-3), or None
    """
    x:         float
    y:         float
    theta:     float
    vx:        float
    vy:        float
    omega:     float
    mode:      Mode
    pivot_idx: int | None = None   # which corner is the current pivot

    # ── Corner geometry ────────────────────────────────────────────────────────

    def corners_world(self, params: "SimParams") -> np.ndarray:
        """
        Return the 4 corners in world frame as a (4, 2) array.

        Order: A (bottom-left), B (top-left), C (top-right), D (bottom-right)
        in the body frame.  World positions depend on (x, y, theta).
        """
        c, s = np.cos(self.theta), np.sin(self.theta)
        R = np.array([[c, -s],
                      [s,  c]])
        a, b = params.a, params.b
        # corners in body frame (relative to CoM)
        local = np.array([
            [-a / 2, -b / 2],   # A  bottom-left
            [-a / 2,  b / 2],   # B  top-left
            [ a / 2,  b / 2],   # C  top-right
            [ a / 2, -b / 2],   # D  bottom-right
        ])
        com = np.array([self.x, self.y])
        return (R @ local.T).T + com   # (4, 2)

    def corner_velocity_world(self, corner_idx: int,
                               params: "SimParams") -> np.ndarray:
        """
        Velocity of corner `corner_idx` in world frame.

        v_corner = v_CoM + omega × r_CoM_to_corner
        (2-D cross: omega × r = [-omega*ry, omega*rx])
        """
        corners = self.corners_world(params)
        r = corners[corner_idx] - np.array([self.x, self.y])
        v_com = np.array([self.vx, self.vy])
        return v_com + np.array([-self.omega * r[1], self.omega * r[0]])


# ── Simulation parameters ──────────────────────────────────────────────────────

@dataclass
class SimParams:
    # ── Physical parameters ────────────────────────────────────────────────────
    m:  float = 7.0    # crate mass (kg)
    a:  float = 0.3    # crate short side (m)   — body x direction
    b:  float = 0.4    # crate long side (m)    — body y direction
    mu: float = 0.3    # friction coefficient (contacts with walls)

    # ── Impact model ───────────────────────────────────────────────────────────
    e_restitution: float = 0.3   # coefficient of restitution at wall impact
                                  # 0 = perfectly plastic, 1 = elastic

    # ── Initial conditions ────────────────────────────────────────────────────
    # CoM start position (world frame).
    # Must satisfy x0 > a/2, y0 > b/2 to keep crate fully above wall 1.
    x0: float = 0.35   # m
    y0: float = 0.50   # m

    # theta is sampled randomly in make_initial_state(); these bounds define
    # the range (rad).  A "natural" range keeps the crate roughly upright.
    theta_min: float = -np.pi / 6   # -30 deg
    theta_max: float =  np.pi / 6   #  30 deg

    # Angular velocity: random uniform in [omega_min, omega_max] (rad/s)
    omega_min: float = 0.0
    omega_max: float = 0.2

    # ── Control parameters ────────────────────────────────────────────────────
    # APPROACH: force toward wall 1 (mostly -y, small +x to avoid wall 2)
    F_approach_y: float = -30.0   # N  (negative → toward y=0)
    F_approach_x: float =   3.0   # N  (positive → away from x=0)

    # ── Integration ───────────────────────────────────────────────────────────
    dt:         float = 0.002   # RK4 time step (s)
    total_time: float = 10.0    # simulation duration (s)

    # ── Phase-transition tolerances ───────────────────────────────────────────
    contact_tol:   float = 1e-3   # m — corner considered "on wall" if y < tol
    side_flat_tol: float = 5e-3   # m — both corners on wall 1 → COINCEMENT

    # ── Derived geometric quantities ──────────────────────────────────────────
    @property
    def I_G(self) -> float:
        """Moment of inertia about the CoM."""
        return (self.m / 12.0) * (self.a ** 2 + self.b ** 2)

    @property
    def I_A(self) -> float:
        """Moment of inertia about corner A (parallel-axis theorem)."""
        return self.I_G + self.m * (self.a**2 + self.b**2) / 4.0

    @property
    def rAO_body(self) -> np.ndarray:
        """Vector from corner A to CoM in body frame."""
        return np.array([self.a / 2.0, self.b / 2.0])

    @property
    def rAB_body(self) -> np.ndarray:
        """Vector A → B in body frame."""
        return np.array([0.0, self.b])


# ── Initial state factory ──────────────────────────────────────────────────────

def make_initial_state(params: SimParams,
                        rng: np.random.Generator | None = None) -> CrateState:
    """
    Create the initial CrateState for the APPROACH phase.

    - CoM at (params.x0, params.y0)  — deterministic, user-specified.
    - theta  ∈ [theta_min, theta_max]  — sampled uniformly.
    - omega  ∈ [omega_min, omega_max]  — sampled uniformly, sign random.
    - vx = 0, vy = 0  (crate starts at rest, control will accelerate it).

    Parameters
    ----------
    params : SimParams
    rng    : numpy random generator (seeded for reproducibility, or None for
             a fresh random seed each call).

    Returns
    -------
    CrateState in APPROACH mode.
    """
    if rng is None:
        rng = np.random.default_rng()

    theta = rng.uniform(params.theta_min, params.theta_max)
    omega = rng.uniform(params.omega_min, params.omega_max)
    omega *= rng.choice([-1.0, 1.0])   # random sign

    # Safety check: make sure all 4 corners are above wall 1 (y > 0)
    trial = CrateState(
        x=params.x0, y=params.y0,
        theta=theta,
        vx=0.0, vy=0.0,
        omega=omega,
        mode=Mode.APPROACH,
    )
    corners = trial.corners_world(params)
    if np.any(corners[:, 1] < 0):
        # Clamp y0 so the lowest corner is at least 0.05 m above wall 1
        margin = 0.05
        y_min_corner = np.min(corners[:, 1])
        trial.y = params.y0 + (margin - y_min_corner)

    return trial