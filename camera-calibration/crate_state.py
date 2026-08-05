"""
Crate state detection based on ArUco pose estimation.
Checks mode conditions from the thesis and detects valid transitions.

Coordinate frame {W}: origin at the corner (Wall1 / Wall2 junction).
  x_C, y_C : position of the crate center of mass in {W} (meters)
  theta_c   : crate orientation relative to initial position (degrees)
  vx, vy    : linear velocities (m/s)
  omega     : angular velocity (rad/s)
"""

import numpy as np

# ─────────────────────────────────────────────
#  Crate geometry
# ─────────────────────────────────────────────
L_c = 0.400   # crate length  (m)
W_c = 0.300   # crate width   (m)

# ─────────────────────────────────────────────
#  Wall lengths
# ─────────────────────────────────────────────
L_W1 = 0.3    # wall 1 length (m)
L_W2 = 0.3    # wall 2 length (m)

# ─────────────────────────────────────────────
#  Tolerances
# ─────────────────────────────────────────────
EPS_X         = 0.05             # position tolerance x  (m)
EPS_Y         = 0.05             # position tolerance y  (m)
EPS_V         = 0.05             # velocity threshold    (m/s)
THETA_DOT_MAX = 0.6              # angular vel threshold (rad/s)
THETA_MIN     = np.radians(45)   # min rotation for pivoting
EPS_THETA     = np.radians(10)   # angular tolerance

# ─────────────────────────────────────────────
#  Mode names and colors (BGR for OpenCV)
# ─────────────────────────────────────────────
MODES = [
    "UNKNOWN",
    "CORNER_APPROACH",
    "ENGAGING_PIVOTING",
    "SHIFTING_IN",
    "SHIFTING_OUT",
    "DONE",
]

MODE_COLORS = {
    "UNKNOWN":           (128, 128, 128),
    "CORNER_APPROACH":   (255, 165,   0),
    "ENGAGING_PIVOTING": (255,   0, 255),
    "SHIFTING_IN":       (255, 200,   0),
    "SHIFTING_OUT":      (  0, 255,   0),
    "DONE":              (  0, 255,   0),
}

# ─────────────────────────────────────────────
#  Individual state checks
# ─────────────────────────────────────────────

def in_corner_approach(x, y, vx, vy, omega):
    """Crate in positive quadrant, low velocities."""
    return (
        x >= 0 and
        y >= 0 and
        abs(vx)    <= EPS_V and
        abs(vy)    <= EPS_V and
        abs(omega) <= THETA_DOT_MAX
    )


def in_s_long(x, y):
    """Long side aligned with wall 1."""
    return (
        abs(x - L_c / 2) <= EPS_X and
        abs(y - W_c / 2) <= EPS_Y
    )


def in_s_short(x, y):
    """Short side aligned with wall 1."""
    return (
        abs(x - W_c / 2) <= EPS_X and
        abs(y - L_c / 2) <= EPS_Y
    )


def rotation_from_initial(theta_deg):
    """How much the crate has rotated from its nearest rectangular orientation."""
    t = abs(theta_deg) % 180
    return min(t, 180 - t)


def in_engaging_pivoting(x, y, theta_deg):
    """Crate rotated >= 45 degrees and within workspace."""
    delta = rotation_from_initial(theta_deg)
    theta = np.radians(delta)
    return (
        -0.5 <= x <= 0 and
        0 <= y < 0.25 and
        theta >= THETA_MIN
    )


def in_shifting_in(x, y, theta_deg):
    """Long side fully aligned with wall 2, ready to shift."""
    delta = rotation_from_initial(theta_deg)
    theta = np.radians(delta)
    return (
        -L_W2 <= x <= 0 and
        abs(y - W_c / 2) <= EPS_Y and
        theta >= np.radians(90) - EPS_THETA
    )


def in_shifting_out(x, y, theta_deg):
    """Crate shifted to final stacking position."""
    delta = rotation_from_initial(theta_deg)
    theta = np.radians(delta)
    return (
        0 <= x <= L_W1 and
        abs(y - W_c / 2) <= EPS_Y and
        theta >= np.radians(90) - EPS_THETA
    )


# ─────────────────────────────────────────────
#  Main state machine
# ─────────────────────────────────────────────

class CrateStateMachine:
    """
    Tracks the current manipulation mode and checks transition conditions.
    Call update() every frame with the relative pose from ArUco.
    Call confirm_transition() when Enter is pressed.
    """

    MODE_ORDER = [
        "CORNER_APPROACH",
        "ENGAGING_PIVOTING",
        "SHIFTING_IN",
        "SHIFTING_OUT",
        "DONE",
    ]

    def __init__(self):
        self.current_mode        = "UNKNOWN"
        self.can_transition      = False
        self.next_mode           = None
        self.sub_state           = None
        self._transition_latched = False

    def update(self, x, y, theta_deg, vx=0.0, vy=0.0, omega=0.0):
        """
        x, y       : position in world frame {W} (m)
        theta_deg  : relative yaw angle (degrees)
        vx, vy     : linear velocities (m/s)
        omega      : angular velocity (rad/s)
        """
        if not self._transition_latched:
            self.can_transition = False
            self.next_mode      = None
            self.sub_state      = None

        if self.current_mode == "DONE":
            return

        if self.current_mode == "UNKNOWN":
            if in_corner_approach(x, y, vx, vy, omega):
                self.current_mode = "CORNER_APPROACH"
            return

        if self.current_mode == "CORNER_APPROACH":
            if in_s_long(x, y):
                self.sub_state           = "S_LONG"
                self.can_transition      = True
                self.next_mode           = "DONE"
                self._transition_latched = True
            elif in_s_short(x, y):
                self.sub_state           = "S_SHORT"
                self.can_transition      = True
                self.next_mode           = "ENGAGING_PIVOTING"
                self._transition_latched = True
            return

        if self.current_mode == "ENGAGING_PIVOTING":
            if in_engaging_pivoting(x, y, theta_deg):
                self.can_transition      = True
                self.next_mode           = "SHIFTING_IN"
                self._transition_latched = True
            return

        if self.current_mode == "SHIFTING_IN":
            if in_shifting_in(x, y, theta_deg):
                self.can_transition      = True
                self.next_mode           = "SHIFTING_OUT"
                self._transition_latched = True
            return

        if self.current_mode == "SHIFTING_OUT":
            if in_s_long(x, y):
                self.sub_state           = "S_LONG"
                self.can_transition      = True
                self.next_mode           = "CORNER_APPROACH"
                self._transition_latched = True
            return

    def confirm_transition(self):
        """Called when Enter is pressed — advances to next_mode if conditions are met."""
        if self.can_transition and self.next_mode is not None:
            print(f"[FSM] Transition: {self.current_mode} → {self.next_mode}")
            self.current_mode        = self.next_mode
            self.can_transition      = False
            self.next_mode           = None
            self._transition_latched = False
        else:
            print(f"[FSM] Conditions not met — current mode: {self.current_mode}")

    def status_lines(self):
        """Returns list of strings to display on the overlay."""
        color = MODE_COLORS.get(self.current_mode, (255, 255, 255))
        lines = [
            f"Mode: {self.current_mode}"
            + (f"  [{self.sub_state}]" if self.sub_state else ""),
        ]
        if self.can_transition:
            lines.append(f">>> TRANSITION READY -> {self.next_mode}")
        else:
            lines.append("Waiting for conditions...")
        return lines, color