
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

rA_body = ca.vertcat(-a/2, -b/2)
rB_body = ca.vertcat(-a/2,  +b/2)
rAB = ca.vertcat(0.0, b)
rAO = ca.vertcat(a/2, b/2)

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

#get the acceleration in function of delta's 
import casadi as ca

def delta_kinematics_casadi(l, ldot, lddot, psi, psidot, psiddot, b, r_vec):
    eps = 1e-9

    # ---- New formula ----
    # delta = psi - pi/2 + asin( (l/b) * sin(psi) )
    u = (l / b) * ca.sin(psi)

    one_minus = ca.fmax(eps, 1 - u*u)
    k  = 1 / ca.sqrt(one_minus)   # d/dx asin(u) = k * du/dx
    k3 = k**3

    delta = psi - 0.5*ca.pi + ca.asin(u)

    # ---- First derivatives ----
    u_l   = (1 / b) * ca.sin(psi)
    u_psi = (l / b) * ca.cos(psi)

    delta_l   = k * u_l
    delta_psi = 1 + k * u_psi

    # ---- Second derivatives of u ----
    u_ll     = 0
    u_lpsi   = (1 / b) * ca.cos(psi)
    u_psipsi = -(l / b) * ca.sin(psi)

    # ---- Second derivatives of delta ----
    # Using: d2/dxdy asin(u) = k*u_xy + u*k^3*u_x*u_y
    delta_ll     = k*u_ll     + u*k3*u_l*u_l
    delta_lpsi   = k*u_lpsi   + u*k3*u_l*u_psi
    delta_psipsi = k*u_psipsi + u*k3*u_psi*u_psi

    # ---- Time derivatives ----
    delta_dot = delta_l*ldot + delta_psi*psidot

    delta_ddot = (
        delta_ll*(ldot**2)
        + 2*delta_lpsi*ldot*psidot
        + delta_psipsi*(psidot**2)
        + delta_l*lddot
        + delta_psi*psiddot
    )

    # ---- Kinematics part unchanged ----
    R  = rot(delta)
    Rp = rot_prime(delta)

    a_vec = ca.vertcat(lddot, 0) + (Rp @ r_vec) * delta_ddot - (R @ r_vec) * (delta_dot**2)
    return delta, delta_dot, delta_ddot, a_vec[0], a_vec[1]


#for ploting, we want to call the wall profile with normal floats, not casadi DM
def wall_profile_numeric(t):
    psi, psidot, psiddot = wall_profile_casadi(ca.DM(t))
    return float(psi), float(psidot), float(psiddot)


def build_step_solver_implicit_no_newton(dt=0.002, h_crane=1.0, g0=9.81, amax=2.0, x_wall=0.0):
    # ---- parameters (given each step)
    t    = ca.SX.sym("t")
    l    = ca.SX.sym("l")
    ldot = ca.SX.sym("ldot")
    p_state = ca.vertcat(t, l, ldot)

    # ---- decision variables (ALL in IPOPT)
    fx    = ca.SX.sym("fx")
    fy    = ca.SX.sym("fy")
    lddot = ca.SX.sym("lddot")
    fya   = ca.SX.sym("fya")  # = Na >= 0
    fxb   = ca.SX.sym("fxb")  # = Nb >= 0

    x = ca.vertcat(fx, fy, lddot, fya, fxb)

    # wall profile
    psi, psidot, psiddot = wall_profile_casadi(t)

    # kinematics (depends on lddot now directly)
    delta, delta_dot, delta_ddot, ax, ay = delta_kinematics_casadi(
        l, ldot, lddot, psi, psidot, psiddot, b, r_corner_to_com
    )

    NA  = fya                  # >= 0 (ta variable de décision)
    TxA = -mu * NA             # ou +mu*NA selon le sens
    fA  = ca.vertcat(TxA, NA)

    # B: mur pivote avec psi -> base tangente/normal
    NB = fxb                   # >= 0 (ta variable de décision)

    tB = ca.vertcat(ca.cos(psi), ca.sin(psi))          # tangente du mur
    nB = ca.vertcat(-ca.sin(psi), ca.cos(psi))         # normale (orthogonale)

    # si la normale pousse dans le mauvais sens, fais: nB = -nB

    TB = -mu * NB              # friction saturée le long de la tangente
    fB = NB*nB + TB*tB
    fc  = ca.vertcat(fx, fy)
    
         # normal  (1,0)

    # lever arms in world
    R   = rot(delta)
    rAO_R = R @ rA_body
    rAB_R = R @ rB_body
    

    # dynamics equalities (must be 0)
    eq1 = (fA[0] + fB[0] + fx) - m*ax
    eq2 = (fA[1] + fB[1] + fy) - m*ay
    # eq3 = (cross2(rOA, fA) + cross2(rOB, fB)) - I*delta_ddot
    eq3 = (cross2(rAO_R, fc) + cross2(rAB_R, fB)) - I_A*delta_ddot
    F   = ca.vertcat(eq1, eq2, eq3)   # = 0

    # ---- crane geometry from (fx,fy)  (still computed for logging)
    R   = rot(delta)
    pA = ca.vertcat(l, 0)
    pC = pA - (R @ rA_body)
    xc = pC[0]
    yc = pC[1]

    eps = 1e-9
    Fh = ca.sqrt(fx**2 + fy**2 + eps)
    sigma = ca.atan(m*g0 / Fh)
    d = h_crane / ca.tan(sigma)
    ex = fx / Fh
    ey = fy / Fh
    xg = xc + d*ex
    yg = yc + d*ey

    # ---- parameters for cost + constraints
    fx_prev = ca.SX.sym("fx_prev")
    fy_prev = ca.SX.sym("fy_prev")
    p = ca.vertcat(p_state, fx_prev, fy_prev)

    # ---- 1-step prediction to target l=b
    l_next_pred    = l + dt*ldot + 0.5*(dt**2)*lddot
    ldot_next_pred = ldot + dt*lddot

    eps_gate = 0.1
    # gate: ~0 loin, ~1 près de l=b
    e = l_next_pred - b
    alpha = 1 / (1 + (e/eps_gate)**2)


    # ---- objective
    w_u     = 1e-2
    w_du    = 1e-2
    w_l = 1e2
    w_v = 1e2
    
    over = ca.fmax(0, l_next_pred - b)
    w_over = 1e1
    w_a = 1e1
    J = w_du*((fx - fx_prev)**2 + (fy - fy_prev)**2) + w_over * over**2 + w_v*alpha*(ldot_next_pred)**2 + w_a * (lddot**2) + w_l * (l_next_pred - b)**2



    # ---- constraints
    # equalities: F == 0
    # inequalities: Na>=0, Nb>=0
    g = ca.vertcat(
        F,          # 3 eq
        fya, fxb    # >= 0
    )

    nlp = {"x": x, "p": p, "f": J, "g": g}
    opts = {
        "ipopt.print_level": 0,
        "print_time": 0,
        "ipopt.bound_relax_factor": 0.0,
    }
    S = ca.nlpsol("S_no_newton", "ipopt", nlp, opts)

    # bounds for g: [eq1,eq2,eq3, Na, Nb]
    lbg = [0.0, 0.0, 0.0,  0.0, 0.0]
    ubg = [0.0, 0.0, 0.0,  ca.inf, ca.inf]

    # bounds for x = [fx, fy, lddot, Na, Nb]
    lbx = [-50.0, -50.0, -1e4, 0.0, 0.0]
    ubx = [ 50.0,  50.0,  1e4, 1e6, 1e6]

    # for logging crane position after solve
    crane_fun = ca.Function("crane_fun_no_newton", [x, p_state], [xc, yc, xg, yg])

    def solve_step(tv, lv, ldotv,
                   x0=None,                 # initial guess for ALL vars
                   fx_prev_v=0.0, fy_prev_v=0.0):
        if x0 is None:
            # [fx, fy, lddot, Na, Nb]
            x0 = [0.0, 0.0, 0.0, 0.0, 0.0]

        sol = S(
            x0=x0,
            p=[tv, lv, ldotv, fx_prev_v, fy_prev_v],
            lbx=lbx, ubx=ubx,
            lbg=lbg, ubg=ubg
        )

        x_opt = sol["x"].full().squeeze()

        fx_opt, fy_opt, lddot_opt, Na_opt, Nb_opt = x_opt

        cvals = crane_fun(ca.DM(x_opt), ca.DM([tv, lv, ldotv]))
        xc_v = float(cvals[0])
        yc_v = float(cvals[1])
        xg_v = float(cvals[2])
        yg_v = float(cvals[3])

        u_opt = np.array([fx_opt, fy_opt], dtype=float)
        y_opt = np.array([lddot_opt, Na_opt, Nb_opt], dtype=float)

        return u_opt, y_opt, (xc_v, yc_v, xg_v, yg_v), x_opt

    return solve_step


if __name__ == "__main__":

    dt = 0.002
    T_sim = 7.0
    N = int(T_sim / dt)

    solver = build_step_solver_implicit_no_newton(dt=dt, h_crane=1.0, g0=9.81, amax=2.0, x_wall=0.0)

    # state
    l = 0.0
    ldot = 0.0

    # warm start for IPOPT decision vector: x = [fx, fy, lddot, Na, Nb]
    x0 = [0.0, 0.0, 0.0, 0.0, 0.0]

    # previous control for smoothness
    fx_prev = 0.0
    fy_prev = 0.0

    hist = {
        "t": [], "l": [], "ldot": [], "lddot": [],
        "psi": [], "delta": [],
        "fx": [], "fy": [],
        "Na": [], "Nb": [],
        "Ta": [], "Tb": [],
        "fAx": [], "fAy": [],
        "fBx": [], "fBy": [],
        "xc": [], "yc": [],
        "xg": [], "yg": [],
        "Fh": []
    }

    for k in range(N + 1):
        t = k * dt

        # wall profile (log)
        psi, psidot, psiddot = wall_profile_numeric(t)

        # solve 1 step (NO teleport): returns u_opt, y_opt, crane_tuple, x_opt
        u_opt, y_opt, (xc, yc, xg, yg), x_opt = solver(
            t, l, ldot,
            x0=x0,
            fx_prev_v=fx_prev, fy_prev_v=fy_prev
        )

        fx, fy = float(u_opt[0]), float(u_opt[1])
        lddot, Na, Nb = float(y_opt[0]), float(y_opt[1]), float(y_opt[2])

        # force magnitude
        eps = 1e-9
        Fh = np.sqrt(fx*fx + fy*fy + eps)

        # delta and delta_dot for logging
        delta_dm, delta_dot_dm, _, _, _ = delta_kinematics_casadi(
            ca.DM(l), ca.DM(ldot), ca.DM(lddot),
            ca.DM(psi), ca.DM(psidot), ca.DM(psiddot),
            b, r_corner_to_com
        )
        delta = float(delta_dm)
        delta_dot = float(delta_dot_dm)

        # rebuild contact forces for logging (same as in solver)
        t1 = np.array([1.0, 0.0])
        n1 = np.array([0.0, 1.0])
        t2 = np.array([np.cos(psi), np.sin(psi)])
        n2 = np.array([-np.sin(psi), np.cos(psi)])

        # slip velocities (approx numeric)
        vA = np.array([ldot, 0.0])
        vtA = float(t1 @ vA)

        # vB = vA + R'(delta)*[0,b]*delta_dot
        Rp = np.array([[-np.sin(delta), -np.cos(delta)],
                       [ np.cos(delta), -np.sin(delta)]])
        vB = vA + (Rp @ np.array([0.0, b])) * delta_dot
        vtB = float(t2 @ vB)

        v_eps = 1e-3
        sA = np.tanh(vtA / v_eps)
        sB = np.tanh(vtB / v_eps)

        Ta = -mu * Na * sA
        Tb = -mu * Nb * sB

        fA = Na*n1 + Ta*t1
        fB = Nb*n2 + Tb*t2

        # log
        hist["t"].append(t)
        hist["l"].append(l)
        hist["ldot"].append(ldot)
        hist["lddot"].append(lddot)
        hist["psi"].append(float(psi))
        hist["delta"].append(delta)

        hist["fx"].append(fx)
        hist["fy"].append(fy)
        hist["Na"].append(Na)
        hist["Nb"].append(Nb)
        hist["Ta"].append(Ta)
        hist["Tb"].append(Tb)
        hist["fAx"].append(float(fA[0]))
        hist["fAy"].append(float(fA[1]))
        hist["fBx"].append(float(fB[0]))
        hist["fBy"].append(float(fB[1]))

        hist["xc"].append(float(xc))
        hist["yc"].append(float(yc))
        hist["xg"].append(float(xg))
        hist["yg"].append(float(yg))
        hist["Fh"].append(float(Fh))

        # integrate (semi-implicit Euler)
        ldot = ldot + dt * lddot
        l    = l    + dt * ldot

        # unilateral constraint l >= 0
        if l < 0.0:
            l = 0.0
            if ldot < 0.0:
                ldot = 0.0

        # update previous values
        fx_prev, fy_prev = fx, fy

        # warm-start next IPOPT
        x0 = x_opt

    print("Final l =", hist["l"][-1])
    print("Final t =", hist["t"][-1])

    # -------------------------
    # Plots
    # -------------------------
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
    plt.title("Evolution of control forces (chosen by solver)")
    plt.xlabel("Time (s)")
    plt.ylabel("Force (N)")
    plt.grid(True, alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(hist["t"], hist["Na"], label="Na (normal A)")
    plt.plot(hist["t"], hist["Nb"], label="Nb (normal B)")
    plt.plot(hist["t"], hist["Ta"], label="Ta (friction A)")
    plt.plot(hist["t"], hist["Tb"], label="Tb (friction B)")
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

    # -------------------------
    # Animation (walls + AB + crane + cable)
    # -------------------------
    def animate_walls_AB_crane(hist, b=0.4, L1=0.5, L2=0.5, dt=0.002, skip=10):
        t_vals    = np.array(hist["t"])
        l_vals    = np.array(hist["l"])
        psi_vals  = np.array(hist["psi"])
        delta_vals= np.array(hist["delta"])

        xc_vals  = np.array(hist["xc"])
        yc_vals  = np.array(hist["yc"])
        xg_vals  = np.array(hist["xg"])
        yg_vals  = np.array(hist["yg"])

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
        # crane_pt,   = ax.plot([], [], "o", label="crane")
        # cable_line, = ax.plot([], [], "--", lw=1.5, label="cable (COM->crane)")

        txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")
        ax.legend(loc="upper right")

        def update(i):
            t = t_vals[i]
            l = l_vals[i]
            psi = psi_vals[i]
            delta = delta_vals[i]

            wall1_line.set_data([0, L1], [0, 0])
            wall2_line.set_data([0, L2*np.cos(psi)], [0, L2*np.sin(psi)])

            A = np.array([l, 0.0])
            R = rot_np(-delta)  # keep the sign you settled on
            # B = A + R @ np.array([0.0, b])
            B = (b - l) * np.array([np.cos(psi), np.sin(psi)])

            seg_line.set_data([A[0], B[0]], [A[1], B[1]])
            A_pt.set_data([A[0]], [A[1]])
            B_pt.set_data([B[0]], [B[1]])

            xc, yc = xc_vals[i], yc_vals[i]
            xg, yg = xg_vals[i], yg_vals[i]

            COM_pt.set_data([xc], [yc])
            # crane_pt.set_data([xg], [yg])
            # cable_line.set_data([xc, xg], [yc, yg])

            AB = np.linalg.norm(B - A)
            cable_h = np.sqrt((xg - xc)**2 + (yg - yc)**2)
            txt.set_text(f"t={t:.2f}s\npsi={psi:.2f} rad\ndelta={delta:.2f} rad\nl={l:.3f} m\n|AB|={AB:.3f} m\ncable_h={cable_h:.3f} m")

            return wall1_line, wall2_line, seg_line, A_pt, B_pt, COM_pt, txt

        frame_idx = np.arange(0, len(t_vals), skip)
        interval = int(1000 * dt * skip)
        ani = FuncAnimation(fig, update, frames=frame_idx, interval=interval, blit=False)
        plt.show()
        return ani

    ani = animate_walls_AB_crane(hist, b=b, L1=0.5, L2=0.5, dt=dt, skip=10)