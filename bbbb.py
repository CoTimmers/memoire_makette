import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import casadi as ca


# =========================
# PARAMETERS
# =========================
m = 7.0
a = 0.3
b = 0.4
I = (m/12.0) * (a**2 + b**2)
I_A = (m/3.0) * (a**2 + b**2)
mu = 0.3

r_corner_to_com = ca.vertcat(a/2, b/2)

rA_body  = ca.vertcat(-a/2, -b/2)
rB_body  = ca.vertcat(-a/2,  +b/2)
rAB_body = ca.vertcat(0.0, b)
rAO_body = ca.vertcat(a/2, b/2)

T_wall = 6.0


# =========================
# WALL PROFILE
# =========================
def wall_profile_casadi(t):
    psi_start = ca.pi/2
    psi_end   = ca.pi
    delta_psi = psi_end - psi_start

    t1 = T_wall * 0.4
    t2 = T_wall * 0.6

    acc = delta_psi / (0.5*t1**2 + (t2-t1)*t1 + 0.5*(T_wall - t2)**2)

    v1 = acc*t1
    p1 = psi_start + 0.5*acc*t1**2
    p2 = p1 + v1*(t2 - t1)

    psi_p1 = psi_start + 0.5*acc*t**2
    psi_p2 = p1 + v1*(t - t1)
    psi_p3 = p2 + v1*(t - t2) - 0.5*acc*(t - t2)**2

    psi = ca.if_else(t <= t1, psi_p1,
          ca.if_else(t <= t2, psi_p2,
            ca.if_else(t <= T_wall, psi_p3, psi_end)))

    psidot_p1 = acc*t
    psidot_p2 = v1
    psidot_p3 = v1 - acc*(t - t2)
    psidot = ca.if_else(t <= t1, psidot_p1,
            ca.if_else(t <= t2, psidot_p2,
              ca.if_else(t <= T_wall, psidot_p3, 0.0)))

    psiddot = ca.if_else(t <= t1, acc,
             ca.if_else(t <= t2, 0.0,
               ca.if_else(t <= T_wall, -acc, 0.0)))

    return psi, psidot, psiddot


def wall_profile_numeric(t):
    psi, psidot, psiddot = wall_profile_casadi(ca.DM(t))
    return float(psi), float(psidot), float(psiddot)


# =========================
# GEOMETRY TOOLS
# =========================
def rot(theta):
    return ca.vertcat(
        ca.horzcat(ca.cos(theta), -ca.sin(theta)),
        ca.horzcat(ca.sin(theta),  ca.cos(theta))
    )


def rot_prime(theta):
    return ca.vertcat(
        ca.horzcat(-ca.sin(theta), -ca.cos(theta)),
        ca.horzcat( ca.cos(theta), -ca.sin(theta))
    )


def cross2(u, v):
    return u[0]*v[1] - u[1]*v[0]


# =========================
# KINEMATICS
# =========================
def delta_kinematics_casadi(l, ldot, lddot, psi, psidot, psiddot, b, r_vec):
    eps = 1e-9

    u_raw = (l / b) * ca.sin(psi)
    u = ca.fmin(1.0 - 1e-9, ca.fmax(-1.0 + 1e-9, u_raw))

    one_minus = ca.fmax(eps, 1 - u*u)
    k  = 1 / ca.sqrt(one_minus)
    k3 = k**3

    delta = psi - 0.5*ca.pi + ca.asin(u)

    u_l   = (1 / b) * ca.sin(psi)
    u_psi = (l / b) * ca.cos(psi)

    delta_l   = k * u_l
    delta_psi = 1 + k * u_psi

    u_ll     = 0
    u_lpsi   = (1 / b) * ca.cos(psi)
    u_psipsi = -(l / b) * ca.sin(psi)

    delta_ll     = k*u_ll     + u*k3*u_l*u_l
    delta_lpsi   = k*u_lpsi   + u*k3*u_l*u_psi
    delta_psipsi = k*u_psipsi + u*k3*u_psi*u_psi

    delta_dot = delta_l*ldot + delta_psi*psidot

    delta_ddot = (
        delta_ll*(ldot**2)
        + 2*delta_lpsi*ldot*psidot
        + delta_psipsi*(psidot**2)
        + delta_l*lddot
        + delta_psi*psiddot
    )

    R  = rot(delta)
    Rp = rot_prime(delta)

    a_vec = ca.vertcat(lddot, 0) + (Rp @ r_vec) * delta_ddot - (R @ r_vec) * (delta_dot**2)
    return delta, delta_dot, delta_ddot, a_vec[0], a_vec[1]


# =========================
# CONTINUOUS DYNAMICS
# x = [l, ldot]
# u = [fx, fy]
# z = [lddot, fya, fbn] solved from mechanics
# =========================
def build_continuous_model(h_crane=1.0, g0=9.81):
    x = ca.SX.sym("x", 2)
    l = x[0]
    ldot = x[1]

    u = ca.SX.sym("u", 2)
    fx = u[0]
    fy = u[1]

    t = ca.SX.sym("t")

    # algebraic unknowns
    z = ca.SX.sym("z", 3)
    lddot = z[0]
    fya   = z[1]
    fbn   = z[2]

    psi, psidot, psiddot = wall_profile_casadi(t)

    delta, delta_dot, delta_ddot, ax, ay = delta_kinematics_casadi(
        l, ldot, lddot, psi, psidot, psiddot, b, r_corner_to_com
    )

    tB = ca.vertcat(ca.cos(psi), ca.sin(psi))
    nB = ca.vertcat(ca.sin(psi), -ca.cos(psi))

    fxa = -mu * fya
    fA = ca.vertcat(fxa, fya)

    fbt = -mu * fbn
    fB = fbn * nB + fbt * tB
 
    fc = ca.vertcat(fx, fy)

    R = rot(delta)
    rAB_w = R @ rAB_body
    rAO_w = R @ rAO_body

    pA = ca.vertcat(l, 0)
    pO = pA + rAO_w
    pB = pA + rAB_w

    xc = pO[0]
    yc = pO[1]
    yB = pB[1]

    eq1 = (fA[0] + fB[0] + fx) - m * ax
    eq2 = (fA[1] + fB[1] + fy) - m * ay
    eq3 = cross2(rAO_w, fc) + cross2(rAB_w, fB) - I_A * delta_ddot

    g_alg = ca.vertcat(eq1, eq2, eq3)

    # solve the 3x3 linear system in z = [lddot, fya, fbn]
    A = ca.jacobian(g_alg, z)
    g0_alg = ca.substitute(g_alg, z, ca.DM.zeros(3, 1))
    z_sol = ca.solve(A, -g0_alg)

    lddot_sol = z_sol[0]
    fya_sol   = z_sol[1]
    fbn_sol   = z_sol[2]

    # recompute useful quantities with solved lddot
    delta2, delta_dot2, delta_ddot2, ax2, ay2 = delta_kinematics_casadi(
        l, ldot, lddot_sol, psi, psidot, psiddot, b, r_corner_to_com
    )

    tB2 = ca.vertcat(ca.cos(psi), ca.sin(psi))
    nB2 = ca.vertcat(ca.sin(psi), -ca.cos(psi))

    fxa2 = -mu * fya_sol
    fbt2 = -mu * fbn_sol

    fA2 = ca.vertcat(fxa2, fya_sol)
    fB2 = fbn_sol * nB2 + fbt2 * tB2

    R2 = rot(delta2)
    rAB_w2 = R2 @ rAB_body
    rAO_w2 = R2 @ rAO_body

    pA2 = ca.vertcat(l, 0)
    pO2 = pA2 + rAO_w2
    pB2 = pA2 + rAB_w2

    xc2 = pO2[0]
    yc2 = pO2[1]
    yB2 = pB2[1]

    # crane geometry from (fx, fy)
    eps = 1e-9
    Fh = ca.sqrt(fx**2 + fy**2 + eps)
    sigma = ca.atan(m*g0 / Fh)
    d = h_crane / ca.tan(sigma)

    ex = fx / Fh
    ey = fy / Fh

    xg = xc2 + d * ex
    yg = yc2 + d * ey

    xdot = ca.vertcat(ldot, lddot_sol)

    extra = ca.vertcat(
        lddot_sol,          # 0
        fya_sol,            # 1
        fbn_sol,            # 2
        delta2,             # 3
        fxa2,               # 4
        fbt2,               # 5
        fA2[0], fA2[1],     # 6,7
        fB2[0], fB2[1],     # 8,9
        xc2, yc2,           # 10,11
        xg, yg,             # 12,13
        Fh,                 # 14
        yB2                 # 15
    )

    f_dyn = ca.Function("f_dyn", [x, u, t], [xdot, extra], ["x", "u", "t"], ["xdot", "extra"])
    return f_dyn


# =========================
# RK4 INTEGRATOR
# =========================
def build_rk4_step(dt=0.002, h_crane=1.0, g0=9.81):
    f_dyn = build_continuous_model(h_crane=h_crane, g0=g0)

    x = ca.SX.sym("x", 2)
    u = ca.SX.sym("u", 2)
    t = ca.SX.sym("t")

    k1, _ = f_dyn(x,               u, t)
    k2, _ = f_dyn(x + 0.5*dt*k1,   u, t + 0.5*dt)
    k3, _ = f_dyn(x + 0.5*dt*k2,   u, t + 0.5*dt)
    k4, _ = f_dyn(x + dt*k3,       u, t + dt)

    x_next = x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

    rk4_step = ca.Function("rk4_step", [x, u, t], [x_next], ["x", "u", "t"], ["x_next"])
    return f_dyn, rk4_step


# =========================
# SIMPLE CONTROL LAW
# Replace this later by Rockit / OCP result if needed
# =========================
def control_law(t, x):
    l = x[0]
    ldot = x[1]

    # Example smooth heuristic
    fx = 8.0 + 18.0 * max(0.0, (b - l))
    fy = 12.0 - 2.5 * ldot

    fx = np.clip(fx, -50.0, 50.0)
    fy = np.clip(fy, -50.0, 50.0)
    return np.array([fx, fy], dtype=float)


# =========================
# ANIMATION
# =========================
def animate_walls_AB_crane(hist, b=0.4, L1=0.5, L2=0.5, dt=0.002, skip=10):
    t_vals     = np.array(hist["t"])
    l_vals     = np.array(hist["l"])
    psi_vals   = np.array(hist["psi"])
    delta_vals = np.array(hist["delta"])

    xc_vals = np.array(hist["xc"])
    yc_vals = np.array(hist["yc"])
    xg_vals = np.array(hist["xg"])
    yg_vals = np.array(hist["yg"])

    def rot_np(theta):
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([[c, -s],
                         [s,  c]])

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.4)

    xmax = max(L1, float(np.max(l_vals)) + 0.1, b + 0.1,
               float(np.max(xg_vals)) + 0.1, float(np.max(xc_vals)) + 0.1)
    xmin = min(-0.2, float(np.min(xg_vals)) - 0.1, float(np.min(xc_vals)) - 0.1)
    ymax = max(0.6, float(np.max(yg_vals)) + 0.1, float(np.max(yc_vals)) + 0.1)
    ymin = min(-0.2, float(np.min(yg_vals)) - 0.1, float(np.min(yc_vals)) - 0.1)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    wall1_line, = ax.plot([], [], lw=3, label="wall1 (fixed)")
    wall2_line, = ax.plot([], [], lw=3, label="wall2 (rotating)")
    seg_line,   = ax.plot([], [], lw=2, label="AB")
    A_pt,       = ax.plot([], [], "o", label="A")
    B_pt,       = ax.plot([], [], "o", label="B")

    COM_pt,     = ax.plot([], [], "o", label="COM")
    crane_pt,   = ax.plot([], [], "o", label="crane")
    cable_line, = ax.plot([], [], "--", lw=1.5, label="cable (COM->crane)")

    txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")
    ax.legend(loc="upper right")

    def update(i):
        t = t_vals[i]
        l = l_vals[i]
        psi = psi_vals[i]
        delta = delta_vals[i]

        wall1_line.set_data([0, L1], [0, 0])
        wall2_line.set_data([0, L2*np.cos(psi)], [0, L2*np.sin(psi)])

        R = rot_np(delta)
        A = np.array([l, 0.0])
        Bp = A + (R @ np.array([0.0, b]))

        seg_line.set_data([A[0], Bp[0]], [A[1], Bp[1]])
        A_pt.set_data([A[0]], [A[1]])
        B_pt.set_data([Bp[0]], [Bp[1]])

        xc, yc = xc_vals[i], yc_vals[i]
        xg, yg = xg_vals[i], yg_vals[i]

        COM_pt.set_data([xc], [yc])
        crane_pt.set_data([xg], [yg])
        cable_line.set_data([xc, xg], [yc, yg])

        AB = np.linalg.norm(Bp - A)
        cable_h = np.sqrt((xg - xc)**2 + (yg - yc)**2)

        txt.set_text(
            f"t={t:.2f}s\n"
            f"psi={psi:.2f} rad\n"
            f"l={l:.3f} m\n"
            f"|AB|={AB:.3f} m\n"
            f"cable_h={cable_h:.3f} m"
        )

        return wall1_line, wall2_line, seg_line, A_pt, B_pt, COM_pt, crane_pt, cable_line, txt

    frame_idx = np.arange(0, len(t_vals), skip)
    interval = int(1000 * dt * skip)
    ani = FuncAnimation(fig, update, frames=frame_idx, interval=interval, blit=False)
    plt.show()
    return ani


# =========================
# MAIN SIMULATION
# =========================
if __name__ == "__main__":
    dt = 0.002
    T_sim = 10.0
    N = int(T_sim / dt)

    f_dyn, rk4_step = build_rk4_step(dt=dt, h_crane=1.0, g0=9.81)

    # state x = [l, ldot]
    x = np.array([0.0, 0.0], dtype=float)

    hist = {
        "t": [], "l": [], "ldot": [], "lddot": [],
        "psi": [], "delta": [],
        "fx": [], "fy": [],
        "fxa": [], "fya": [],
        "fbn": [], "fbt": [],
        "fBx": [], "fBy": [],
        "xc": [], "yc": [],
        "xg": [], "yg": [],
        "Fh": [], "yB": []
    }

    for k in range(N + 1):
        t = k * dt
        l, ldot = float(x[0]), float(x[1])

        u = control_law(t, x)
        fx, fy = float(u[0]), float(u[1])

        out = f_dyn(x=x, u=u, t=t)
        xdot_val = np.array(out["xdot"]).squeeze()
        extra = np.array(out["extra"]).squeeze()

        lddot = float(extra[0])
        fya   = float(extra[1])
        fbn   = float(extra[2])
        delta = float(extra[3])
        fxa   = float(extra[4])
        fbt   = float(extra[5])
        fBx   = float(extra[8])
        fBy   = float(extra[9])
        xc    = float(extra[10])
        yc    = float(extra[11])
        xg    = float(extra[12])
        yg    = float(extra[13])
        Fh    = float(extra[14])
        yB    = float(extra[15])

        psi, psidot, psiddot = wall_profile_numeric(t)

        hist["t"].append(t)
        hist["l"].append(l)
        hist["ldot"].append(ldot)
        hist["lddot"].append(lddot)

        hist["psi"].append(float(psi))
        hist["delta"].append(delta)

        hist["fx"].append(fx)
        hist["fy"].append(fy)

        hist["fxa"].append(fxa)
        hist["fya"].append(fya)

        hist["fbn"].append(fbn)
        hist["fbt"].append(fbt)
        hist["fBx"].append(fBx)
        hist["fBy"].append(fBy)

        hist["xc"].append(xc)
        hist["yc"].append(yc)
        hist["xg"].append(xg)
        hist["yg"].append(yg)

        hist["Fh"].append(Fh)
        hist["yB"].append(yB)

        # RK4 integration
        x_next = rk4_step(x=x, u=u, t=t)["x_next"]
        x = np.array(x_next).squeeze()

        # clamp l in [0,b]
        if x[0] < 0.0:
            x[0] = 0.0
            if x[1] < 0.0:
                x[1] = 0.0

        if x[0] > b:
            x[0] = b
            if x[1] > 0.0:
                x[1] = 0.0

    print("Final l =", hist["l"][-1])
    print("Final t =", hist["t"][-1])

    # =========================
    # SAME PLOTS AS YOUR CODE
    # =========================
    plt.figure(figsize=(10, 5))
    plt.step(hist["t"], hist["l"], where="post", label="l(t)", lw=2)
    plt.title("Evolution of l over time")
    plt.xlabel("Time (s)")
    plt.ylabel("l (m)")
    plt.grid(True, alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.step(hist["t"], hist["fx"], where="post", label="fx(t)", lw=2)
    plt.step(hist["t"], hist["fy"], where="post", label="fy(t)", lw=2)
    plt.title("Evolution of control forces")
    plt.xlabel("Time (s)")
    plt.ylabel("Force (N)")
    plt.grid(True, alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(hist["t"], hist["fxa"], label="fxa (friction A)")
    plt.plot(hist["t"], hist["fya"], label="fya (normal A)")
    plt.plot(hist["t"], hist["fbn"], label="fbn (normal B local)")
    plt.plot(hist["t"], hist["fbt"], label="fbt (tangent B local)")
    plt.xlabel("Time (s)")
    plt.ylabel("Forces (N)")
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(hist["t"], hist["l"], label="l")
    plt.plot(hist["t"], hist["psi"], label="psi")
    plt.plot(hist["t"], hist["delta"], label="delta")
    plt.xlabel("Time (s)")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(hist["xg"], hist["yg"], label="crane")
    plt.plot(hist["xc"], hist["yc"], label="COM")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.show()

    animate_walls_AB_crane(hist, b=b, L1=0.5, L2=0.5, dt=dt, skip=10)