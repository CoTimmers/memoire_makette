from enum import Enum, auto
from dataclasses import dataclass
import numpy as np


class Mode(Enum):
    """
    Phase sequence from the mindmap:
      APPROACH   : box moving toward wall 1, no contact yet
      COINCEMENT : box wedged in corner (A on wall 1, B on wall 2)
      PIVOTEMENT : box rotating 60° around A (A fixed on wall 1, B sliding on wall 2)
      GLISSEMENT : long side on wall 1, box sliding away from wall 2
      FINAL      : wall 2 closes, box slides to final position
    """
    APPROACH    = auto()
    COINCEMENT  = auto()
    PIVOTEMENT  = auto()
    GLISSEMENT  = auto()
    FINAL       = auto()


@dataclass
class CrateState:
    """
    Constrained state of the crate once A is on wall 1.

    The single degree of freedom is l = x-position of corner A along wall 1.
    The box orientation delta is fully determined by (l, psi) via the contact
    constraint (B on wall 2).  See dynamics.delta_kinematics.
    """
    l:    float   # position of corner A along wall 1 (m), x-axis
    ldot: float   # velocity of A along wall 1 (m/s)
    mode: Mode


@dataclass
class SimParams:
    # ── Physical parameters ──────────────────────────────────────────────────
    m:  float = 7.0    # crate mass (kg)
    a:  float = 0.3    # crate short side (m)
    b:  float = 0.4    # crate long side (m)
    mu: float = 0.3    # friction coefficient at wall 2

    # ── Integration ──────────────────────────────────────────────────────────
    dt:         float = 0.002   # RK4 time step (s)
    total_time: float = 10.0    # simulation duration (s)

    # ── Wall 2 profile (trapezoidal velocity, psi: π/2 → π) ─────────────────
    T_wall:         float = 6.0
    psi_start:      float = np.pi / 2
    psi_end:        float = np.pi
    wall_t1_ratio:  float = 0.4   # ramp-up ends at T_wall * t1_ratio
    wall_t2_ratio:  float = 0.6   # constant velocity ends at T_wall * t2_ratio

    # ── Phase-transition thresholds ──────────────────────────────────────────
    # PIVOTEMENT → GLISSEMENT when psi reaches this angle (≈ 127°)
    psi_switch: float = 127.0 * np.pi / 180.0

    # GLISSEMENT → FINAL when l reaches this value (box slid far enough)
    l_slide_end: float = 0.38    # (m)  — should stay ≤ b

    # FINAL ends when l reaches this value (box near wall 2)
    l_final_end: float = 0.02    # (m)

    # Tolerance for contact detection
    contact_tol: float = 0.01   # (m)

    # ── Control force magnitudes ─────────────────────────────────────────────
    # Professor's guidance: do NOT command arbitrary (fx, fy); instead
    # inject potential energy in a geometrically motivated direction.
    F_pivot: float = 10.0   # force magnitude for PIVOTEMENT (N)
    F_slide: float = 10.0   # force magnitude for GLISSEMENT / FINAL (N)

    # ── Derived geometric quantities (read-only properties) ──────────────────
    @property
    def I_A(self) -> float:
        """Moment of inertia about corner A (parallel-axis from centroid)."""
        return (self.m / 3.0) * (self.a ** 2 + self.b ** 2)

    @property
    def rAB_body(self) -> np.ndarray:
        """Vector A → B in body frame (A is bottom-left, B is top-left)."""
        return np.array([0.0, self.b])

    @property
    def rAO_body(self) -> np.ndarray:
        """Vector A → CoM in body frame."""
        return np.array([self.a / 2.0, self.b / 2.0])
