import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import casadi as ca


m = 7.0
a = 0.3
b = 0.4
I = (m/12.0) * (a**2 + b**2)
I_A = (m/3.0) * (a**2 + b**2)
mu = 0.3

r_corner_to_com = ca.vertcat(a/2, b/2)

rA_body = ca.vertcat(-a/2, -b/2)   # O -> A in body frame
rB_body = ca.vertcat(-a/2,  +b/2)  # O -> B in body frame
rAB_body = ca.vertcat(0.0, b)      # A -> B in body frame
rAO_body = ca.vertcat(a/2, b/2)    # A -> O in body frame

T_wall = 6.0


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


def wall_profile_numeric(t):
    psi, psidot, psiddot = wall_profile_casadi(ca.DM(t))
    return float(psi), float(psidot), float(psiddot)


def build_step_solver_implicit_no_newton(dt=0.002, h_crane=1.0, g0=9.81, amax=2.0, x_wall=0.0):
    # ---------- parameters ----------
    t    = ca.SX.sym("t")
    l    = ca.SX.sym("l")
    ldot = ca.SX.sym("ldot")
    p_state = ca.vertcat(t, l, ldot)

    # ---------- decision variables ----------
    fx    = ca.SX.sym("fx")
    fy    = ca.SX.sym("fy")
    lddot = ca.SX.sym("lddot")
    fya   = ca.SX.sym("fya")   # normal at A
    fbn   = ca.SX.sym("fbn")   # normal at B (local normal)

    x = ca.vertcat(fx, fy, lddot, fya, fbn)

    # ---------- wall profile ----------
    psi, psidot, psiddot = wall_profile_casadi(t)

    # ---------- kinematics ----------
    delta, delta_dot, delta_ddot, ax, ay = delta_kinematics_casadi(
        l, ldot, lddot, psi, psidot, psiddot, b, r_corner_to_com
    )

    # ---------- local basis at B ----------
    # tangent follows the wall orientation
    tB = ca.vertcat(ca.cos(psi), ca.sin(psi))
    # normal chosen so that:
    # psi = pi/2 -> nB = [1, 0] (wall vertical, pushing to the right)
    # psi = pi   -> nB = [0, 1] (wall horizontal, pushing upward)
    nB = ca.vertcat(ca.sin(psi), -ca.cos(psi))

    # ---------- contact forces ----------
    # A: horizontal ground
    fxa = -mu * fya
    fA = ca.vertcat(fxa, fya)

    # B: rotating wall, local Coulomb
    fbt = -mu * fbn
    fB = fbn * nB + fbt * tB

    # crane force
    fc = ca.vertcat(fx, fy)

    # ---------- geometry in world frame ----------
    R = rot(delta)

    rAB_w = R @ rAB_body   # A -> B
    rAO_w = R @ rAO_body   # A -> O

    # COM position from A
    pA = ca.vertcat(l, 0)
    pO = pA + rAO_w
    xc = pO[0]
    yc = pO[1]

    pB = pA + rAB_w
    yB = pB[1]

    # ---------- dynamics ----------
    eq1 = (fA[0] + fB[0] + fx) - m * ax
    eq2 = (fA[1] + fB[1] + fy) - m * ay
    eq3 = cross2(rAO_w, fc) + cross2(rAB_w, fB) - I_A * delta_ddot

    F = ca.vertcat(eq1, eq2, eq3)

    # ---------- crane geometry from (fx, fy) ----------
    eps = 1e-9
    Fh = ca.sqrt(fx**2 + fy**2 + eps)
    sigma = ca.atan(m*g0 / Fh)
    d = h_crane / ca.tan(sigma)

    ex = fx / Fh
    ey = fy / Fh

    xg = xc + d * ex
    yg = yc + d * ey

    # ---------- previous-step parameters ----------
    fx_prev = ca.SX.sym("fx_prev")
    fy_prev = ca.SX.sym("fy_prev")
    xg_prev = ca.SX.sym("xg_prev")
    yg_prev = ca.SX.sym("yg_prev")
    v_prev  = ca.SX.sym("v_prev")

    p = ca.vertcat(p_state, fx_prev, fy_prev, xg_prev, yg_prev, v_prev)

    # ---------- simple placeholder cost ----------
    l_next_pred = l + dt*ldot + 0.5*(dt**2)*lddot
    ldot_next = ldot + dt*lddot
    e = b - l_next_pred
    alpha = 1 / (1 + (e/0.01)**2)

    J = -l_next_pred + 1e-1 * alpha * ldot_next**2

    # ---------- constraints ----------
    g_lstop = l_next_pred - b
    g = ca.vertcat(
        F,      # 3 equalities
        fya,    # >= 0
        fbn,
        g_lstop,
        delta
    )

    nlp = {"x": x, "p": p, "f": J, "g": g}
    opts = {
        "ipopt.print_level": 0,
        "print_time": 0,
        "ipopt.bound_relax_factor": 0.0,
    }
    S = ca.nlpsol("S_no_newton", "ipopt", nlp, opts)

    lbg = [0.0, 0.0, 0.0,   0.0, 0.0, -ca.inf, 0.0]
    ubg = [0.0, 0.0, 0.0,   ca.inf, ca.inf, 0.0, ca.pi/2]

    # x = [fx, fy, lddot, fya, fbn]
    lbx = [-50.0, -50.0, -1e4, 0.0, 0.0]
    ubx = [ 50.0,  50.0,  1e4, 1e6, 1e6]

    crane_fun = ca.Function("crane_fun_no_newton", [x, p_state], [xc, yc, xg, yg, yB])
    contact_fun = ca.Function("contact_fun", [x, p_state], [fA, fB, nB, tB, delta])

    def solve_step(tv, lv, ldotv,
                   x0=None,
                   fx_prev_v=0.0, fy_prev_v=0.0,
                   xg_prev_v=0.0, yg_prev_v=0.0, v_prev_v=0.0):
        if x0 is None:
            x0 = [0.0, 0.0, 0.0, 1.0, 1.0]

        sol = S(
            x0=x0,
            p=[tv, lv, ldotv, fx_prev_v, fy_prev_v, xg_prev_v, yg_prev_v, v_prev_v],
            lbx=lbx, ubx=ubx,
            lbg=lbg, ubg=ubg
        )

        x_opt = sol["x"].full().squeeze()

        fx_opt, fy_opt, lddot_opt, fya_opt, fbn_opt = x_opt

        cvals = crane_fun(ca.DM(x_opt), ca.DM([tv, lv, ldotv]))
        xc_v = float(cvals[0])
        yc_v = float(cvals[1])
        xg_v = float(cvals[2])
        yg_v = float(cvals[3])
        yB_v = float(cvals[4])

        cf = contact_fun(ca.DM(x_opt), ca.DM([tv, lv, ldotv]))
        fA_v = np.array(cf[0].full()).squeeze()
        fB_v = np.array(cf[1].full()).squeeze()
        nB_v = np.array(cf[2].full()).squeeze()
        tB_v = np.array(cf[3].full()).squeeze()
        delta_v = float(cf[4])

        u_opt = np.array([fx_opt, fy_opt], dtype=float)
        y_opt = np.array([lddot_opt, fya_opt, fbn_opt], dtype=float)

        extra = {
            "xc": xc_v, "yc": yc_v, "xg": xg_v, "yg": yg_v, "yB": yB_v,
            "fA": fA_v, "fB": fB_v, "nB": nB_v, "tB": tB_v, "delta": delta_v
        }

        return u_opt, y_opt, extra, x_opt

    return solve_step


if __name__ == "__main__":
    dt = 0.002
    T_sim = 10.0
    N = int(T_sim / dt)

    solver = build_step_solver_implicit_no_newton(dt=dt, h_crane=1.0, g0=9.81, amax=2.0, x_wall=0.0)

    l = 0.0
    ldot = 0.0

    # x = [fx, fy, lddot, fya, fbn]
    x0 = [0.0, 0.0, 0.0, 1.0, 1.0]

    fx_prev = 0.0
    fy_prev = 0.0
    xg_prev = 0.0
    yg_prev = 0.0
    v_prev  = 0.0

    hist = {
        "t": [], "l": [], "ldot": [], "lddot": [],
        "psi": [], "delta": [],
        "fx": [], "fy": [],
        "fxa": [], "fya": [],
        "fbn": [], "fbt": [],
        "fBx": [], "fBy": [],
        "xc": [], "yc": [],
        "xg": [], "yg": [],
        "Fh": [],
        "yB": []
    }

    for k in range(N + 1):
        t = k * dt
        psi, psidot, psiddot = wall_profile_numeric(t)

        u_opt, y_opt, extra, x_opt = solver(
            t, l, ldot,
            x0=x0,
            fx_prev_v=fx_prev, fy_prev_v=fy_prev,
            xg_prev_v=xg_prev, yg_prev_v=yg_prev, v_prev_v=v_prev
        )

        fx, fy = float(u_opt[0]), float(u_opt[1])
        lddot, fya, fbn = float(y_opt[0]), float(y_opt[1]), float(y_opt[2])

        fxa = -mu * fya
        fbt = -mu * fbn
        fBx = float(extra["fB"][0])
        fBy = float(extra["fB"][1])

        eps = 1e-9
        Fh = np.sqrt(fx*fx + fy*fy + eps)

        hist["t"].append(t)
        hist["l"].append(l)
        hist["ldot"].append(ldot)
        hist["lddot"].append(lddot)

        hist["psi"].append(float(psi))
        hist["delta"].append(float(extra["delta"]))

        hist["fx"].append(fx)
        hist["fy"].append(fy)

        hist["fxa"].append(fxa)
        hist["fya"].append(fya)

        hist["fbn"].append(fbn)
        hist["fbt"].append(fbt)
        hist["fBx"].append(fBx)
        hist["fBy"].append(fBy)

        hist["xc"].append(float(extra["xc"]))
        hist["yc"].append(float(extra["yc"]))
        hist["xg"].append(float(extra["xg"]))
        hist["yg"].append(float(extra["yg"]))

        hist["Fh"].append(float(Fh))
        hist["yB"].append(float(extra["yB"]))

        # semi-implicit Euler
        ldot = ldot + dt * lddot
        l    = l    + dt * ldot

        if l < 0.0:
            l = 0.0
            if ldot < 0.0:
                ldot = 0.0

        dx = float(extra["xg"]) - xg_prev
        dy = float(extra["yg"]) - yg_prev
        v_prev = np.sqrt(dx*dx + dy*dy) / dt

        xg_prev, yg_prev = float(extra["xg"]), float(extra["yg"])
        fx_prev, fy_prev = fx, fy
        x0 = x_opt

    print("Final l =", hist["l"][-1])
    print("Final t =", hist["t"][-1])

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

    def animate_walls_AB_crane(hist, b=0.4, L1=0.5, L2=0.5, dt=0.002, skip=10):
        t_vals    = np.array(hist["t"])
        l_vals    = np.array(hist["l"])
        psi_vals  = np.array(hist["psi"])
        delta_vals= np.array(hist["delta"])

        xc_vals = np.array(hist["xc"])
        yc_vals = np.array(hist["yc"])
        xg_vals = np.array(hist["xg"])
        yg_vals = np.array(hist["yg"])

        def rot_np(theta):
            c = np.cos(theta); s = np.sin(theta)
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
            B = A + (R @ np.array([0.0, b]))

            seg_line.set_data([A[0], B[0]], [A[1], B[1]])
            A_pt.set_data([A[0]], [A[1]])
            B_pt.set_data([B[0]], [B[1]])

            xc, yc = xc_vals[i], yc_vals[i]
            xg, yg = xg_vals[i], yg_vals[i]

            COM_pt.set_data([xc], [yc])
            crane_pt.set_data([xg], [yg])
            cable_line.set_data([xc, xg], [yc, yg])

            AB = np.linalg.norm(B - A)
            cable_h = np.sqrt((xg - xc)**2 + (yg - yc)**2)

            txt.set_text(f"t={t:.2f}s\npsi={psi:.2f} rad\nl={l:.3f} m\n|AB|={AB:.3f} m\ncable_h={cable_h:.3f} m")
            return wall1_line, wall2_line, seg_line, A_pt, B_pt, COM_pt, crane_pt, cable_line, txt

        frame_idx = np.arange(0, len(t_vals), skip)
        interval = int(1000 * dt * skip)
        ani = FuncAnimation(fig, update, frames=frame_idx, interval=interval, blit=False)
        plt.show()
        return ani

    animate_walls_AB_crane(hist, b=b, L1=0.5, L2=0.5, dt=dt, skip=10)