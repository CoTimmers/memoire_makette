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
#  Crate geometry  (adjust to your real crate)
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
EPS_X        = 0.05          # position tolerance x  (m)
EPS_Y        = 0.05          # position tolerance y  (m)
EPS_V        = 0.05          # velocity threshold    (m/s)
THETA_DOT_MAX = 0.6         # angular vel threshold (rad/s)
THETA_MIN    = np.radians(45)  # min rotation for pivoting
EPS_THETA    = np.radians(10)   # angular tolerance


# ─────────────────────────────────────────────
#  Mode names and colors (BGR for OpenCV)
# ─────────────────────────────────────────────
MODES = [
    "UNKNOWN",
    "CORNER_APPROACH",
    "S_LONG",
    "S_SHORT",
    "ENGAGING_PIVOTING",
    "SHIFTING_IN",
    "SHIFTING_OUT",
    "DONE",
]

MODE_COLORS = {
    "UNKNOWN":           (128, 128, 128),
    "CORNER_APPROACH":   (255, 165,   0),
    "S_LONG":            (  0, 200, 255),
    "S_SHORT":           (  0, 200, 255),
    "ENGAGING_PIVOTING": (255,   0, 255),
    "SHIFTING_IN":       (255, 200,   0),
    "SHIFTING_OUT":      (  0, 255,   0),
    "DONE":              (  0, 255,   0),
}

# ─────────────────────────────────────────────
#  Individual state checks
# ─────────────────────────────────────────────

def in_corner_approach(x, y, vx, vy, omega):
    """S_corner: crate in positive quadrant, low velocities."""
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

#Mauvais coté aligné avec le mur 1. 
def in_s_short(x, y):
    """Short side aligned with wall 1."""
    return (
        abs(x - W_c / 2) <= EPS_X and
        abs(y - L_c / 2) <= EPS_Y
    )


# def in_engaging_pivoting(x, y, theta_deg):
#     """S_pivoting: crate rotated ≥ 45° (dans n'importe quel sens) and within workspace."""
#     theta = np.radians(abs(theta_deg))
#     r     = np.sqrt((W_c / 2) ** 2 + (L_c / 2) ** 2)
#     return (
#         -L_W2 * np.cos(np.radians(45)) <= x <= L_W1 and
#         y < r and
#         theta >= THETA_MIN
#     )



def rotation_from_initial(theta_deg):
    """Combien le bac a tourné depuis son orientation rectangulaire la plus proche."""
    t = abs(theta_deg) % 180
    return min(t, 180 - t)

def in_engaging_pivoting(x, y, theta_deg):
    delta = rotation_from_initial(theta_deg)
    theta = np.radians(delta)
    r = np.sqrt((W_c / 2) ** 2 + (L_c / 2) ** 2)
    return (
        -0.5 <= x <= 0 and
        0 <= y < 0.25 and
        theta >= THETA_MIN  # 45°
    )


# def in_shifting_in(x, y, theta_deg):
#     """Long side fully aligned with wall 2, ready to shift."""
#     theta = np.radians(abs(theta_deg))
#     return (
#         -L_W2 <= x <= L_W1 and
#         abs(y - W_c / 2) <= EPS_Y and
#         theta >= np.radians(90) - EPS_THETA
#     )
    
def in_shifting_in(x, y, theta_deg):
    """Long side fully aligned with wall 2, ready to shift."""
    delta = rotation_from_initial(theta_deg)
    theta = np.radians(delta)
    return (
        -L_W2 <= x <= 0 and
        abs(y - W_c / 2) <= EPS_Y and
        theta >= np.radians(90) - EPS_THETA
    )


# def in_shifting_out(x, y, theta_deg):
#     """Final stacking position reached."""
#     theta = np.radians(abs(theta_deg))
#     r     = np.sqrt((W_c / 2) ** 2 + (L_c / 2) ** 2)
#     return (
#         abs(x - L_c / 2) <= EPS_X and
#         y < r and
#         theta >= THETA_MIN
#     )

def in_shifting_out(x, y, theta_deg):
    """Final stacking position reached."""
    delta = rotation_from_initial(theta_deg)
    theta = np.radians(delta)
    r     = np.sqrt((W_c / 2) ** 2 + (L_c / 2) ** 2)
    return (
        0 <= x <= L_W1 and
        abs(y - W_c / 2) <= EPS_Y and
        theta >= THETA_MIN
    )


# ─────────────────────────────────────────────
#  Main state machine
# ─────────────────────────────────────────────

class CrateStateMachine:
    """
    Tracks the current manipulation mode and checks transition conditions.
    Call update() every frame with the relative pose from ArUco.
    """

    MODE_ORDER = [
        "CORNER_APPROACH",
        "ENGAGING_PIVOTING",
        "SHIFTING_IN",
        "DONE",
    ]

    def __init__(self):
        self.current_mode   = "UNKNOWN"
        self.can_transition = False
        self.next_mode      = None
        self.sub_state      = None   # S_LONG or S_SHORT inside CORNER_APPROACH

    def update(self, x, y, theta_deg, vx=0.0, vy=0.0, omega=0.0):
        """
        x, y       : position in world frame {W} (m)
        theta_deg  : relative yaw angle (degrees)
        vx, vy     : linear velocities (m/s)
        omega      : angular velocity (rad/s)
        """
        self.can_transition = False
        self.next_mode      = None
        self.sub_state      = None

        # ── DONE → etat terminal, rien a verifier ──
        if self.current_mode == "DONE":
            return

        # ── SHIFTING_IN → seul mode qui peut atteindre DONE ──
        if self.current_mode == "SHIFTING_IN":
            if in_shifting_out(x, y, theta_deg):
                self.can_transition = True
                self.next_mode      = "DONE"
            return

        # ── ENGAGING_PIVOTING → check S_pivoting ──
        if self.current_mode == "ENGAGING_PIVOTING":
            if in_engaging_pivoting(x, y, theta_deg):
                self.can_transition = True
                self.next_mode      = "SHIFTING_IN"
            if in_shifting_in(x, y, theta_deg):
                self.current_mode   = "SHIFTING_IN"
                self.can_transition = False
            return

        # ── CORNER_APPROACH → check S_long or S_short ──
        if self.current_mode == "CORNER_APPROACH":
            if in_s_long(x, y):
                self.sub_state      = "S_LONG"
                self.can_transition = True
                self.next_mode      = "DONE"   # long side → stacking complete
            elif in_s_short(x, y):
                self.sub_state      = "S_SHORT"
                self.can_transition = True
                self.next_mode      = "ENGAGING_PIVOTING"
            return

        # ── UNKNOWN → detect current mode ──
        if in_corner_approach(x, y, vx, vy, omega):
            self.current_mode = "CORNER_APPROACH"
        elif in_engaging_pivoting(x, y, theta_deg):
            self.current_mode = "ENGAGING_PIVOTING"
        elif in_shifting_in(x, y, theta_deg):
            self.current_mode = "SHIFTING_IN"
        else:
            self.current_mode = "UNKNOWN"

    def status_lines(self):
        """Returns list of strings to display on the overlay."""
        color = MODE_COLORS.get(self.current_mode, (255, 255, 255))
        lines = [
            f"Mode: {self.current_mode}"
            + (f"  [{self.sub_state}]" if self.sub_state else ""),
        ]
        if self.can_transition:
            lines.append(f">>> TRANSITION OK -> {self.next_mode}")
        else:
            lines.append("En attente des conditions...")
        return lines, color
