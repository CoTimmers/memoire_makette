from enum import Enum, auto
from dataclasses import dataclass
import numpy as np


class Mode(Enum):
    """
    Phase sequence from the mindmap:
      APPROACH   : box moving toward wall 1, no contact yet (free flight)
      CONTACT    : one corner on wall 1, box pivots around it
      COINCEMENT : box wedged in corner (A on wall 1, B on wall 2)
      PIVOTEMENT : box rotating 60° around A (A fixed on wall 1, B sliding on wall 2)
      GLISSEMENT : long side on wall 1, box sliding away from wall 2
      FINAL      : wall 2 closes, box slides to final position
    """
    APPROACH    = auto()   # free spin, no force — camera reads state
    PUSH        = auto()   # force toward wall 1, no contact yet
    CONTACT     = auto()   # one corner on wall 1, crate pivots
    COINCEMENT  = auto()
    PIVOTEMENT  = auto()
    GLISSEMENT  = auto()
    FINAL       = auto()


@dataclass
class CrateState:
    """Full 6-DOF state (used for APPROACH and CONTACT phases)."""
    x:     float        # COM x  (m)
    y:     float        # COM y  (m)
    theta: float        # orientation (rad)
    vx:    float        # COM velocity x  (m/s)
    vy:    float        # COM velocity y  (m/s)
    omega: float        # angular velocity (rad/s)
    mode:  Mode

    # Contact info — valid only during CONTACT phase
    pivot_corner_idx: int   = -1    # 0-3, corner index touching wall 1
    pivot_x:         float = 0.0   # x-coordinate of pivot on wall 1


@dataclass
class SimParams:
    # ── Physical parameters ──────────────────────────────────────────────────
    m:  float = 7.0    # crate mass (kg)
    a:  float = 0.3    # crate short side (m)
    b:  float = 0.4    # crate long side (m)
    mu: float = 0.3    # friction coefficient at wall 2

    # ── Integration ──────────────────────────────────────────────────────────
    dt:         float = 0.002   # time step (s)
    total_time: float = 10.0    # simulation duration (s)

    # ── Approach / contact control ───────────────────────────────────────────
    F_approach:    float = 5.0   # magnitude of force toward wall 1 (N)
    e_restitution: float = 0.5   # coefficient of restitution at wall 1 impact
    side_flat_tol: float = 0.008 # tolerance to declare a side flat on wall 1 (m)

    # ── Initial conditions (APPROACH phase) ─────────────────────────────────
    omega_max_init: float = 0.2   # max initial angular velocity (rad/s)
    x_init_min:    float = 0.3
    x_init_max:    float = 0.8
    y_init_min:    float = 0.4
    y_init_max:    float = 0.9

    # ── Wall 2 profile (trapezoidal velocity, psi: π/2 → π) ─────────────────
    T_wall:         float = 6.0
    psi_start:      float = np.pi / 2
    psi_end:        float = np.pi
    wall_t1_ratio:  float = 0.4
    wall_t2_ratio:  float = 0.6

    # ── Derived geometric quantities (read-only properties) ──────────────────
    @property
    def corners_body(self) -> np.ndarray:
        """
        (4, 2) array of corner positions in the body frame (relative to COM).
        Order: bottom-left, bottom-right, top-right, top-left.
        """
        ha, hb = self.a / 2.0, self.b / 2.0
        return np.array([[-ha, -hb],
                         [+ha, -hb],
                         [+ha, +hb],
                         [-ha, +hb]], dtype=float)

    @property
    def I_G(self) -> float:
        """Moment of inertia about COM."""
        return (self.m / 12.0) * (self.a**2 + self.b**2)

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


# ── Geometric helpers (used across all modules) ────────────────────────────────

def rot2(theta: float) -> np.ndarray:
    """2-D rotation matrix."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def get_corners_world(state: CrateState, params: SimParams) -> np.ndarray:
    """
    Return (4, 2) array of corner world-frame positions.
    Corner order matches params.corners_body:
      0: bottom-left, 1: bottom-right, 2: top-right, 3: top-left.
    """
    R = rot2(state.theta)
    com = np.array([state.x, state.y])
    return np.array([com + R @ c for c in params.corners_body])


def init_approach_state(params: SimParams,
                        rng: np.random.Generator | None = None) -> CrateState:
    """
    Create a random initial APPROACH state.

    - COM placed randomly in [x_init_min, x_init_max] × [y_init_min, y_init_max]
      (strictly inside the first quadrant, away from both walls).
    - Orientation theta drawn uniformly in [0, 2π).
    - Angular velocity omega drawn uniformly in [0, omega_max_init].
    - No initial linear velocity.
    """
    if rng is None:
        rng = np.random.default_rng()

    x     = rng.uniform(params.x_init_min, params.x_init_max)
    y     = rng.uniform(params.y_init_min, params.y_init_max)
    theta = rng.uniform(0.0, 2.0 * np.pi)
    omega = rng.uniform(0.0, params.omega_max_init)

    return CrateState(x=x, y=y, theta=theta,
                      vx=0.0, vy=0.0, omega=omega,
                      mode=Mode.APPROACH)
