#!/usr/bin/env python3
"""
Planar rigid box (top-down) against 2 walls with friction, simulated with CasADi.

CONTROL (optimization-based, copy-paste runnable)
------------------------------------------------
At each time step we compute the force u = [Fx, Fy] applied at the CoM by solving a
tiny (2x2) quadratic optimization problem (closed-form solve, no external QP solver):

Primary objective (HIGH weight):
  - produce a desired torque about the LOWER-LEFT contact point (pivot) in the corner:
      tau(u) = (q - r_p) x u

Secondary objective (LOW weight):
  - keep the pivot near the corner (intersection of walls) by pushing along wall normals
    proportionally to gaps at the pivot.

Then we convert u to magnitude+direction:
  F = ||u||, alpha = atan2(Fy, Fx)

Requires:
  pip install casadi numpy matplotlib
"""

import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# -----------------------
# USER PARAMETERS
# -----------------------
box_width  = 2.0
box_height = 1.0

m = 7.0
I = (m/12.0) * (box_width**2 + box_height**2)

# Contact/friction parameters (need not be tiny if you want strong contact torques)
k_n   = 2000.0
c_n   = 60.0
mu    = 0.3
eps_v = 0.05
beta  = 200.0

# wall2 motion
theta0 = 1.0 * np.pi / 2
Omega  = np.pi/12
L      = 6.0

# simulation/animation
fps = 60
dt  = 1.0 / fps
# dt = 0.1
# fps = int(1.0/dt)

# initial state [x, y, psi, vx, vy, omega]
x0 = np.array([box_width/2+1e-6, box_height/2+1e-6, 0, 0.0, 0.0, 0.0], dtype=float)

# plot bounds
xlim = (-4, 4)
ylim = (-2, 4)

S_np = np.array([[0.0, -1.0],
                 [1.0,  0.0]], dtype=float)


# -----------------------
# GEOMETRY HELPERS
# -----------------------
def rot_np(psi: float) -> np.ndarray:
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s],
                     [s,  c]], dtype=float)

def corners_body_4() -> np.ndarray:
    return np.array([
        [-box_width/2, -box_height/2],  # lower-left (pivot)
        [ box_width/2, -box_height/2],
        [ box_width/2,  box_height/2],
        [-box_width/2,  box_height/2],
    ], dtype=float)

def get_rectangle_corners_position(x, y, psi) -> np.ndarray:
    corners = np.array([
        [-box_width/2, -box_height/2],
        [ box_width/2, -box_height/2],
        [ box_width/2,  box_height/2],
        [-box_width/2,  box_height/2],
        [-box_width/2, -box_height/2],
    ], dtype=float)
    return (rot_np(psi) @ corners.T).T + np.array([x, y], dtype=float)

def wall2_direction(theta: float) -> np.ndarray:
    return np.array([np.cos(theta), np.sin(theta)], dtype=float)

def wrap_angle(a: float) -> float:
    return (a + np.pi) % (2*np.pi) - np.pi

def pos(z: float) -> float:
    return max(0.0, z)


# -----------------------
# OPTIMIZATION-BASED CONTROLLER (CLOSED-FORM QP)
# -----------------------
def controller_opt_force(state: np.ndarray, theta: float):
    """
    Computes u=[Fx,Fy] by minimizing a weighted quadratic cost:

      J(u) = w_tau * (a^T u - tau_des)^2
           + w_c   * || N u - b ||^2
           + w_r   * ||u||^2

    where:
      - a encodes torque about the pivot: tau(u) = a^T u
      - N stacks wall normals (n1^T; n2^T) to encourage pushing toward the corner
      - b is proportional to the (positive-part) gaps at the pivot

    Returns: (F, alpha, u_vec, tau_des, tau_u_about_pivot)
    """
    x, y, psi, vx, vy, w = state
    q = np.array([x, y], dtype=float)

    # pivot = lower-left corner
    p_ll = np.array([-box_width/2, -box_height/2], dtype=float)
    R = rot_np(psi)
    r_p = q + R @ p_ll                       # pivot world position

    # lever from pivot to CoM
    ell = q - r_p                            # 2D

    # Torque about pivot from force u:
    # tau = ell_x * u_y - ell_y * u_x = [-ell_y, ell_x] dot [u_x,u_y]
    a = np.array([-ell[1], ell[0]], dtype=float)   # 2D

    # wall normals
    n1 = np.array([0.0, 1.0], dtype=float)         # wall1 y=0, free space y>=0
    n2 = np.array([-np.sin(theta), np.cos(theta)], dtype=float)  # wall2 through origin
    # orient n2 so that CoM is in its free-space
    if (n2 @ q) < 0.0:
        n2 = -n2

    # gaps at pivot (positive means "away from wall" in free space)
    g1 = float(r_p[1])          # y coordinate
    g2 = float(n2 @ r_p)

    # --- Desired torque (PRIMARY) to align psi -> theta ---
    epsi = wrap_angle(psi - theta)
    Kpsi = 80.0
    Kw   = 20.0
    tau_des = -Kpsi * epsi - Kw * w

    # --- Corner objective (SECONDARY): push toward the corner only when away ---
    Kc = 50.0
    b = np.array([Kc * pos(g1), Kc * pos(g2)], dtype=float)       # desired normal push
    N = np.vstack([n1, n2])                                       # 2x2

    # --- Regularization (keeps u reasonable, prevents crazy solutions when a ~ 0) ---
    # Weights: rotation dominates
    w_tau = 100.0
    w_c   = 3.0
    w_r   = 0.2

    # Solve argmin_u J(u): closed form linear system
    # J = w_tau (a^T u - tau_des)^2 + w_c ||N u - b||^2 + w_r ||u||^2
    # => (w_tau a a^T + w_c N^T N + w_r I) u = w_tau tau_des a + w_c N^T b
    H = (w_tau * np.outer(a, a)) + (w_c * (N.T @ N)) + (w_r * np.eye(2))
    rhs = (w_tau * tau_des * a) + (w_c * (N.T @ b))

    # If ell is tiny, a is tiny -> H still invertible thanks to w_r I
    u = np.linalg.solve(H, rhs)

    # saturate
    F_max = 100.0
    F = float(np.linalg.norm(u))
    if F > F_max:
        u = (F_max / F) * u
        F = F_max

    alpha = float(np.arctan2(u[1], u[0])) if F > 1e-9 else 0.0
    tau_u = float(a @ u)  # achieved torque about pivot (from applied force)

    # alpha = theta +1.1*np.pi/2
    alpha = float(np.arctan2(u[1], u[0])) if F > 1e-9 else 0.0
    return F, alpha, u, float(tau_des), tau_u


# -----------------------
# CASADI MODEL (smooth contact + friction)
# -----------------------
def softplus(z, beta_):
    return (1.0/beta_) * ca.log(1 + ca.exp(beta_ * z))

def smooth_neg(z, beta_):
    # ~ max(0, -z)
    return softplus(-z, beta_)

def build_casadi_rhs():
    # state
    x = ca.SX.sym("x")
    y = ca.SX.sym("y")
    psi = ca.SX.sym("psi")
    vx = ca.SX.sym("vx")
    vy = ca.SX.sym("vy")
    w  = ca.SX.sym("w")
    X  = ca.vertcat(x, y, psi, vx, vy, w)

    # parameters
    th   = ca.SX.sym("theta")
    Fp   = ca.SX.sym("F")
    alp  = ca.SX.sym("alpha")

    # rotation
    c = ca.cos(psi); s = ca.sin(psi)
    R = ca.vertcat(
        ca.horzcat(c, -s),
        ca.horzcat(s,  c)
    )
    S = ca.DM([[0.0, -1.0],
               [1.0,  0.0]])

    q = ca.vertcat(x, y)
    v_com = ca.vertcat(vx, vy)

    # wall1: y=0
    n1 = ca.DM([0.0, 1.0])
    t1 = ca.DM([1.0, 0.0])
    b1 = 0.0

    # wall2: through origin at angle th
    t2 = ca.vertcat(ca.cos(th), ca.sin(th))
    n2 = ca.vertcat(-ca.sin(th), ca.cos(th))
    b2 = 0.0

    # orient normal so CoM starts in free-space
    g_com = ca.dot(n2, q) - b2
    flip = ca.if_else(g_com < 0, -1.0, 1.0)
    n2 = flip * n2
    t2 = flip * t2

    # input force at CoM
    f_u = ca.vertcat(Fp * ca.cos(alp), Fp * ca.sin(alp))
    f_total = f_u
    tau_total = 0.0

    corners = corners_body_4()
    gaps_w1 = []
    gaps_w2 = []

    for i in range(4):
        p_b = ca.DM(corners[i, :]).reshape((2, 1))
        p_b = ca.vertcat(p_b[0], p_b[1])

        # corner pos/vel
        r = q + R @ p_b
        v = v_com + w * (S @ (R @ p_b))

        # gaps
        g1 = ca.dot(n1, r) - b1
        g2 = ca.dot(n2, r) - b2
        gaps_w1.append(g1)
        gaps_w2.append(g2)

        # normal/tangent velocities
        vn1 = ca.dot(n1, v); vt1 = ca.dot(t1, v)
        vn2 = ca.dot(n2, v); vt2 = ca.dot(t2, v)

        # normal forces (penalty + approach damping)
        pen1  = smooth_neg(g1, beta)
        inw1  = smooth_neg(vn1, beta)
        lamn1 = k_n * pen1 + c_n * inw1

        pen2  = smooth_neg(g2, beta)
        inw2  = smooth_neg(vn2, beta)
        lamn2 = k_n * pen2 + c_n * inw2

        # friction
        lamt1 = -mu * lamn1 * ca.tanh(vt1 / eps_v)
        lamt2 = -mu * lamn2 * ca.tanh(vt2 / eps_v)

        f1 = lamn1 * n1 + lamt1 * t1
        f2 = lamn2 * n2 + lamt2 * t2
        f_c = f1 + f2

        f_total = f_total + f_c

        # torque about CoM from contacts
        rp = R @ p_b
        tau_total = tau_total + (rp[0]*f_c[1] - rp[1]*f_c[0])

    # accelerations
    ax = f_total[0] / m
    ay = f_total[1] / m
    wdot = tau_total / I

    Xdot = ca.vertcat(vx, vy, w, ax, ay, wdot)

    f_rhs = ca.Function("f_rhs", [X, th, Fp, alp], [Xdot],
                        ["X", "theta", "F", "alpha"], ["Xdot"])
    f_diag = ca.Function("f_diag", [X, th], [ca.vertcat(*gaps_w1, *gaps_w2)],
                         ["X", "theta"], ["gaps"])
    return f_rhs, f_diag


def rk4_step(f_rhs, Xk, theta, F, alpha, h):
    k1 = f_rhs(Xk, theta, F, alpha)
    k2 = f_rhs(Xk + 0.5*h*k1, theta, F, alpha)
    k3 = f_rhs(Xk + 0.5*h*k2, theta, F, alpha)
    k4 = f_rhs(Xk + h*k3, theta, F, alpha)
    return Xk + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)


def main():
    f_rhs, f_diag = build_casadi_rhs()

    T = 8.0
    N = int(T / dt)

    traj    = np.zeros((N+1, 6), dtype=float)
    gaps    = np.zeros((N+1, 8), dtype=float)
    thetas  = np.zeros(N+1, dtype=float)

    ctrl    = np.zeros((N+1, 2), dtype=float)  # [F, alpha]
    forces  = np.zeros((N+1, 2), dtype=float)  # [Fx, Fy]
    taus    = np.zeros((N+1, 2), dtype=float)  # [tau_des, tau_achieved]

    X = ca.DM(x0)
    traj[0, :] = np.array(X).reshape(-1)
    thetas[0] = theta0
    gaps[0, :] = np.array(f_diag(X, theta0)).reshape(-1)

    t = 0.0
    for k in range(N):
        theta = min(theta0 + Omega * t, np.pi)
        thetas[k] = theta

        x_np = np.array(X).reshape(-1)
        Fk, ak, uvec, tau_des, tau_u = controller_opt_force(x_np, theta)

        ctrl[k, :] = [Fk, ak]
        forces[k, :] = uvec
        taus[k, :] = [tau_des, tau_u]

        X = rk4_step(f_rhs, X, theta, Fk, ak, dt)

        traj[k+1, :] = np.array(X).reshape(-1)
        gaps[k+1, :] = np.array(f_diag(X, theta)).reshape(-1)
        thetas[k+1] = theta

        t += dt

    ctrl[-1, :] = ctrl[-2, :]
    forces[-1, :] = forces[-2, :]
    taus[-1, :] = taus[-2, :]

    # -----------------------
    # ANIMATION
    # -----------------------
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal", "box")
    ax.set_title("Optimization-based force control: torque-about-pivot + corner (secondary)")

    # wall1
    ax.plot([xlim[0]-10, xlim[1]+10], [0, 0], lw=2, color="grey")

    wall2_line, = ax.plot([], [], lw=3, color="grey")
    box_line, = ax.plot([], [], lw=2, color="tab:blue")
    corners_scatter = ax.scatter([], [], s=60)

    info_text = ax.text(0.02, 0.98, "", transform=ax.transAxes,
                        va="top", ha="left", family="monospace")

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)


    force_arrow = None
    pivot_scatter = ax.scatter([], [], s=80)

    def update(frame):
        nonlocal force_arrow

        x, y, psi = traj[frame, 0], traj[frame, 1], traj[frame, 2]
        theta = thetas[frame]

        # wall2
        d = wall2_direction(theta)
        wall2_line.set_data([0.0, L*d[0]], [0.0, L*d[1]])

        # box
        poly = get_rectangle_corners_position(x, y, psi)
        box_line.set_data(poly[:, 0], poly[:, 1])

        # corners contact coloring
        corners_world = poly[:-1]
        g_w1 = gaps[frame, 0:4]
        g_w2 = gaps[frame, 4:8]
        active = (g_w1 < 0.0) | (g_w2 < 0.0)
        colors = ["tab:red" if active[i] else "tab:green" for i in range(4)]
        corners_scatter.set_offsets(corners_world)
        corners_scatter.set_color(colors)

        # pivot marker (lower-left corner is corners_world[0] in our ordering)
        pivot = corners_world[0]
        pivot_scatter.set_offsets([pivot])
        pivot_scatter.set_color(["tab:purple"])

        # force arrow
        Fx, Fy = forces[frame, 0], forces[frame, 1]
        if force_arrow is not None:
            force_arrow.remove()
        force_arrow = ax.arrow(x, y, 0.25*Fx, 0.25*Fy,
                               head_width=0.08, length_includes_head=True,
                               color="tab:orange")

        Fk, ak = ctrl[frame, 0], ctrl[frame, 1]
        tau_des, tau_u = taus[frame, 0], taus[frame, 1]

        info_text.set_text(
            f"x={x:+.3f} y={y:+.3f} psi={psi:+.3f}\n"
            f"vx={traj[frame,3]:+.3f} vy={traj[frame,4]:+.3f} w={traj[frame,5]:+.3f}\n"
            f"theta={theta:+.3f}  (psi-theta={wrap_angle(psi-theta):+.3f})\n"
            f"F={Fk:+.3f}  alpha={np.rad2deg(ak):+.1f} deg\n"
            f"Fx={Fx:+.3f} Fy={Fy:+.3f}\n"
            f"tau_des={tau_des:+.3f}  tau_u(pivot)={tau_u:+.3f}\n"
            f"contacts(red corners): {int(active.sum())}/4"
        )

        return wall2_line, box_line, corners_scatter, pivot_scatter, info_text, force_arrow

    anim = FuncAnimation(fig, update, interval=int(1000 / fps), blit=False, cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    print("Running opt_controller_pivot_torque.py")
    main()
