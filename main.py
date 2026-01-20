#!/usr/bin/env python3
"""
Constrained rigid-body simulation: rectangle stuck (no-slip) to a rotating wall.

- World origin at the intersection/pivot of two walls.
- Wall 2 rotates with prescribed theta(t) = theta0 + Omega t.
- A fixed body point r_c on the rectangle is "glued" to Wall 2:
    1) normal constraint: n(theta)^T p_c(q) = 0
    2) tangential constraint: t(theta)^T p_c(q) = s0   (no sliding; fixed address)
- Control input: crane force applied at COM:
    u(t,X) = (F, alpha) -> f = F [cos(alpha), sin(alpha)]
  (No direct torque from crane because applied at COM.)
- Constraint forces (lambda_t, lambda_n) are solved each step via a 5x5 linear system.
- Includes simple projection to reduce numerical drift.

Outputs:
- rectangle_stuck_rotating_wall.gif
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


# ----------------------------
# Math helpers
# ----------------------------

def rot(psi: float) -> np.ndarray:
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s],
                     [s,  c]])

S = np.array([[0.0, -1.0],
              [1.0,  0.0]])

def wall_basis(theta: float):
    """Unit tangent t and normal n for wall angle theta."""
    t = np.array([np.cos(theta), np.sin(theta)])
    n = np.array([-np.sin(theta), np.cos(theta)])
    return t, n


# ----------------------------
# Constrained dynamics core
# ----------------------------

def solve_ddq_lambda(X, ttime, u, params):
    """
    Solve for accelerations ddq=[ax, ay, ddpsi] and multipliers lambda=[lambda_t, lambda_n]
    enforcing sticking constraints.
    State X = [x,y,psi,vx,vy,omega].
    """
    x, y, psi, vx, vy, omega = X
    m = params["m"]
    I = params["I"]
    theta0 = params["theta0"]
    Omega = params["Omega"]
    rc = params["r_c"]
    s0 = params["s0"]

    theta = theta0 + Omega * ttime
    tvec, nvec = wall_basis(theta)

    R = rot(psi)
    a = R @ rc            # COM -> contact in world
    b = R @ (S @ rc)      # a rotated 90deg = a^perp

    pc = np.array([x, y]) + a
    pcdot = np.array([vx, vy]) + omega * b

    # Jacobian J of constraints phi = [t^T pc - s0, n^T pc]^T
    # J is 2x3 mapping [x,y,psi] to constraint values
    J = np.zeros((2, 3))
    J[0, 0:2] = tvec
    J[1, 0:2] = nvec
    J[0, 2] = tvec @ b
    J[1, 2] = nvec @ b

    # Input generalized force (force at COM, no torque)
    Fmag, alpha = u
    fxy = Fmag * np.array([np.cos(alpha), np.sin(alpha)])
    Q = np.array([fxy[0], fxy[1], 0.0])

    # Mass/inertia matrix
    D = np.diag([m, m, I])

    # Acceleration-level constraint RHS beta:
    #
    # This enforces that the glued point follows the wall rotation kinematics.
    # Derived by differentiating the constraints twice with theta(t)=theta0+Omega t.
    beta1 = (Omega**2) * (tvec @ pc) - 2.0 * Omega * (nvec @ pcdot) + (omega**2) * (tvec @ a)
    beta2 = (Omega**2) * (nvec @ pc) + 2.0 * Omega * (tvec @ pcdot) + (omega**2) * (nvec @ a)
    beta = np.array([beta1, beta2])

    # Solve 5x5 block system:
    # [D  -J^T][ddq]   [Q]
    # [J   0  ][lam] = [beta]
    A = np.zeros((5, 5))
    A[0:3, 0:3] = D
    A[0:3, 3:5] = -J.T
    A[3:5, 0:3] = J

    bvec = np.zeros(5)
    bvec[0:3] = Q
    bvec[3:5] = beta

    sol = np.linalg.solve(A, bvec)
    ddq = sol[0:3]   # [ax, ay, ddpsi]
    lam = sol[3:5]   # [lambda_t, lambda_n]
    return ddq, lam


# ----------------------------
# Drift reduction (projection)
# ----------------------------

def project_position(q, ttime, params, iters=3):
    """
    Project (x,y,psi) onto the constraint manifold phi(q,t)=0 using
    least-norm correction: deltaq = -J^T (J J^T)^{-1} phi.
    """
    x, y, psi = q
    theta0 = params["theta0"]
    Omega = params["Omega"]
    rc = params["r_c"]
    s0 = params["s0"]

    theta = theta0 + Omega * ttime
    tvec, nvec = wall_basis(theta)

    for _ in range(iters):
        R = rot(psi)
        a = R @ rc
        b = R @ (S @ rc)
        pc = np.array([x, y]) + a

        phi = np.array([
            tvec @ pc - s0,
            nvec @ pc
        ])

        J = np.zeros((2, 3))
        J[0, 0:2] = tvec
        J[1, 0:2] = nvec
        J[0, 2] = tvec @ b
        J[1, 2] = nvec @ b

        JJt = J @ J.T
        deltaq = - J.T @ np.linalg.solve(JJt, phi)

        x += deltaq[0]
        y += deltaq[1]
        psi += deltaq[2]

    return np.array([x, y, psi])


def project_velocity(q, v, ttime, params):
    """
    Project velocity v=[vx,vy,omega] so that velocity-level constraints are satisfied.

    Constraints:
      c1 = t^T pc - s0 = 0
      c2 = n^T pc      = 0

    Differentiate once (theta-dot = Omega):
      d/dt c1 = t^T pcdot + Omega * n^T pc = 0
      d/dt c2 = n^T pcdot - Omega * t^T pc = 0

    But J v = [t^T pcdot, n^T pcdot]^T.
    So we require: J v = rhs, where rhs = [-Omega n^T pc, +Omega t^T pc]^T.
    """
    x, y, psi = q
    vx, vy, omega = v

    theta0 = params["theta0"]
    Omega = params["Omega"]
    rc = params["r_c"]

    theta = theta0 + Omega * ttime
    tvec, nvec = wall_basis(theta)

    R = rot(psi)
    a = R @ rc
    b = R @ (S @ rc)
    pc = np.array([x, y]) + a

    J = np.zeros((2, 3))
    J[0, 0:2] = tvec
    J[1, 0:2] = nvec
    J[0, 2] = tvec @ b
    J[1, 2] = nvec @ b

    rhs = np.array([
        -Omega * (nvec @ pc),
         Omega * (tvec @ pc)
    ])

    vvec = np.array([vx, vy, omega])
    resid = rhs - (J @ vvec)

    JJt = J @ J.T
    deltav = J.T @ np.linalg.solve(JJt, resid)
    return vvec + deltav


# ----------------------------
# Time integration (RK4)
# ----------------------------

def dynamics_rhs(X, ttime, u_func, params):
    u = u_func(ttime, X)
    ddq, _ = solve_ddq_lambda(X, ttime, u, params)

    x, y, psi, vx, vy, omega = X
    ax, ay, ddpsi = ddq

    return np.array([vx, vy, omega, ax, ay, ddpsi])


def rk4_step(X, ttime, dt, u_func, params):
    k1 = dynamics_rhs(X, ttime, u_func, params)
    k2 = dynamics_rhs(X + 0.5*dt*k1, ttime + 0.5*dt, u_func, params)
    k3 = dynamics_rhs(X + 0.5*dt*k2, ttime + 0.5*dt, u_func, params)
    k4 = dynamics_rhs(X + dt*k3,     ttime + dt,     u_func, params)

    Xn = X + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

    # Project to reduce drift
    qn = project_position(Xn[0:3], ttime + dt, params, iters=2)
    vn = project_velocity(qn, Xn[3:6], ttime + dt, params)

    Xn[0:3] = qn
    Xn[3:6] = vn
    return Xn


# ----------------------------
# Visualization helpers
# ----------------------------

def rectangle_poly(x, y, psi, w, h):
    corners = np.array([
        [-w/2, -h/2],
        [ w/2, -h/2],
        [ w/2,  h/2],
        [-w/2,  h/2],
        [-w/2, -h/2],
    ])
    R = rot(psi)
    pts = corners @ R.T + np.array([x, y])
    return pts


# ----------------------------
# Main
# ----------------------------

def main():
    # --- parameters you can tweak ---
    w, h = 0.6, 0.3

    params = {
        "m": 1.0,
        "I": 0.02,          # tune as needed
        "theta0": np.pi/2,  # wall starts vertical
        "Omega": 0.6,       # rad/s (constant)
        "w": w,
        "h": h,
        # glued contact point in body frame (top-left corner)
        "r_c": np.array([-w/2, h/2]),
        # "address" along the wall where the point is glued
        "s0": 0.9,
    }

    # --- control input (edit this!) ---
    # u(t,X) = (F, alpha). alpha=0 -> +x direction
    def u_func(t, X):
        F = 0.0
        alpha = 0.0
        return np.array([F, alpha])

    # --- initial condition consistent with constraints ---
    psi0 = 0.0
    theta_init = params["theta0"]
    t0, _ = wall_basis(theta_init)

    # want contact point at pc0 = s0 * t0 (on the wall, at distance s0)
    pc0 = params["s0"] * t0

    # COM so that pc = [x;y] + R(psi) rc => [x;y] = pc - R rc
    a0 = rot(psi0) @ params["r_c"]
    xy0 = pc0 - a0

    X0 = np.array([xy0[0], xy0[1], psi0, 0.0, 0.0, 0.0])

    # project to be safe
    X0[0:3] = project_position(X0[0:3], 0.0, params, iters=5)
    X0[3:6] = project_velocity(X0[0:3], X0[3:6], 0.0, params)

    # --- simulate ---
    T = 10.0
    dt = 0.005
    N = int(T/dt)

    times = np.linspace(0.0, T, N+1)
    traj = np.zeros((N+1, 6))
    traj[0] = X0

    X = X0.copy()
    for k in range(N):
        X = rk4_step(X, times[k], dt, u_func, params)
        traj[k+1] = X

    # --- animate ---
    fps = 30
    step = max(1, int(1/(fps*dt)))
    idx = np.arange(0, len(times), step)
    times_a = times[idx]
    traj_a = traj[idx]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal", "box")
    ax.set_title("Rectangle stuck to a rotating wall (constraint-based)")

    rect_line, = ax.plot([], [], lw=2)
    contact_dot, = ax.plot([], [], marker="o", markersize=6)
    wall2_line, = ax.plot([], [], lw=2)
    wall1_line, = ax.plot([], [], lw=2)
    time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")

    # fixed wall1 as x-axis (y=0)
    L = 2.0
    wall1_line.set_data([-L, L], [0.0, 0.0])

    # limits
    pad = 0.6
    xmin = traj_a[:, 0].min() - pad
    xmax = traj_a[:, 0].max() + pad
    ymin = traj_a[:, 1].min() - pad
    ymax = traj_a[:, 1].max() + pad
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    def init():
        rect_line.set_data([], [])
        contact_dot.set_data([], [])
        wall2_line.set_data([], [])
        time_text.set_text("")
        return rect_line, contact_dot, wall2_line, wall1_line, time_text

    def animate(i):
        tnow = times_a[i]
        x, y, psi, vx, vy, omega = traj_a[i]

        theta = params["theta0"] + params["Omega"] * tnow
        tvec, _ = wall_basis(theta)

        # wall2 line through origin along tvec
        wall2_line.set_data([-L*tvec[0], L*tvec[0]],
                            [-L*tvec[1], L*tvec[1]])

        # rectangle polygon
        poly = rectangle_poly(x, y, psi, params["w"], params["h"])
        rect_line.set_data(poly[:, 0], poly[:, 1])

        # contact point
        pc = np.array([x, y]) + rot(psi) @ params["r_c"]
        contact_dot.set_data([pc[0]], [pc[1]])

        time_text.set_text(f"t = {tnow:5.2f} s")
        return rect_line, contact_dot, wall2_line, wall1_line, time_text

    ani = animation.FuncAnimation(
        fig, animate, frames=len(times_a), init_func=init, interval=1000/fps, blit=True
    )

    out_gif = "rectangle_stuck_rotating_wall.gif"
    ani.save(out_gif, writer=animation.PillowWriter(fps=fps))
    print(f"Saved animation to: {out_gif}")

    # Uncomment to display an interactive window instead of only saving:
    # plt.show()

if __name__ == "__main__":
    main()
