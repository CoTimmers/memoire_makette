"Mur 2 pivote vers l'extérieur"
"le bac commence dans le coin entre les 2 murs et doit finir dans le coin mais pivoté de 90°"
"goal: optimiser le temps de rotation du bac pour arriver dans la position finale"
""
"le mur 2 pivote avec une vitesse constante"
"le glissement sur les surfaces est autorisé mais on doit garder le contacte avec les murs"

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import casadi as ca

# Parameters
m = 7
a = 0.3
b = 0.4
I = (m/12.0) * (a**2 + b**2)
mu = 0.3

N = 50  # steps

a_local = ca.vertcat([-a/2, -b/2])
b_local = ca.vertcat([-a/2, +b/2])
c_local = ca.vertcat([+a/2, -b/2])
d_local = ca.vertcat([ a/2,  b/2])

def rotation_matrix(theta):
    return ca.vertcat(
        ca.horzcat(ca.cos(theta), -ca.sin(theta)),
        ca.horzcat(ca.sin(theta),  ca.cos(theta))
    )

T_wall = 10.0
def wall_state_casadi(t):
    psi_start = ca.pi / 2
    psi_end = ca.pi
    delta_psi = psi_end - psi_start

    t1 = T_wall * 0.4
    t2 = T_wall * 0.6

    accel = delta_psi / ( (t1**2)/2 + (t2-t1)*t1 + ((T_wall-t2)**2)/2 )

    psi_p1 = psi_start + 0.5 * accel * t**2

    v_at_t1 = accel * t1
    p_at_t1 = psi_start + 0.5 * accel * t1**2
    psi_p2 = p_at_t1 + v_at_t1 * (t - t1)

    p_at_t2 = p_at_t1 + v_at_t1 * (t2 - t1)
    psi_p3 = p_at_t2 + v_at_t1 * (t - t2) - 0.5 * accel * (t - t2)**2

    psi = ca.if_else(t <= t1, psi_p1,
          ca.if_else(t <= t2, psi_p2,
            ca.if_else(t <= T_wall, psi_p3, psi_end)))

    return psi

# State x = [x, y, theta, vx, vy, omega]
nx = 6
x_sym = ca.MX.sym('x', nx)
ax_sym = ca.MX.sym('ax')
ay_sym = ca.MX.sym('ay')
alpha_sym = ca.MX.sym('alpha')

x_dot = ca.vertcat(x_sym[3], x_sym[4], x_sym[5], ax_sym, ay_sym, alpha_sym)
f_dynamic = ca.Function('f_dyn', [x_sym, ax_sym, ay_sym, alpha_sym], [x_dot])

psi_sym = ca.MX.sym('psi')
n1_sym = ca.vertcat(0, 1)
n2_sym = ca.vertcat(ca.sin(psi_sym), -ca.cos(psi_sym))

R_bac = rotation_matrix(x_sym[2])
P_A = x_sym[0:2] + R_bac @ a_local
P_B = x_sym[0:2] + R_bac @ b_local
P_C = x_sym[0:2] + R_bac @ c_local
P_D = x_sym[0:2] + R_bac @ d_local

g_col = ca.vertcat(
    ca.dot(P_A, n1_sym), ca.dot(P_B, n1_sym), ca.dot(P_C, n1_sym), ca.dot(P_D, n1_sym),
    ca.dot(P_A, n2_sym), ca.dot(P_B, n2_sym), ca.dot(P_C, n2_sym), ca.dot(P_D, n2_sym)
)
f_collision = ca.Function('f_col', [x_sym, psi_sym], [g_col])

# ---- OPTI ----
opti = ca.Opti()

T_total = opti.variable()
X = opti.variable(nx, N+1)

# Force strategy: U = Fn * n_AB + Ft * t_AB
Fn = opti.variable(1, N)
Ft = opti.variable(1, N)
opti.subject_to(opti.bounded(0, Ft, 50))
opti.subject_to(opti.bounded(-30, Ft, 30))

dt = T_total / N

# Contact forces (still decision variables)
lambda_A_wall2 = opti.variable(1, N)
lambda_B_wall2 = opti.variable(1, N)
lambda_A_wall1 = opti.variable(1, N)
lambda_B_wall1 = opti.variable(1, N)

lambda_A_wall2_t = opti.variable(1, N)
lambda_B_wall2_t = opti.variable(1, N)
lambda_A_wall1_t = opti.variable(1, N)
lambda_B_wall1_t = opti.variable(1, N)

# friction cones
opti.subject_to(lambda_A_wall1_t**2 <= (mu * lambda_A_wall1)**2)
opti.subject_to(lambda_B_wall1_t**2 <= (mu * lambda_B_wall1)**2)
opti.subject_to(lambda_A_wall2_t**2 <= (mu * lambda_A_wall2)**2)
opti.subject_to(lambda_B_wall2_t**2 <= (mu * lambda_B_wall2)**2)

lam_cost = 0
ctrl_cost = 0

for k in range(N):
    st = X[:, k]
    theta_k = st[2]

    # 1) Mur 2
    current_t = k * dt
    psi_k = wall_state_casadi(current_t)

    n1 = ca.vertcat(0, 1)
    t1 = ca.vertcat(-1, 0)

    n2_k = ca.vertcat(ca.sin(psi_k), -ca.cos(psi_k))
    t2_k = ca.vertcat(ca.cos(psi_k),  ca.sin(psi_k))

    # 2) Géométrie bac : COM + frame AB
    C_k = st[0:2]
    R_k = rotation_matrix(theta_k)

    P_A_k = C_k + R_k @ a_local
    P_B_k = C_k + R_k @ b_local

    dAB = P_B_k - P_A_k
    t_AB = dAB / (ca.sqrt(ca.sumsqr(dAB)) + 1e-8)
    n_AB = ca.vertcat(-t_AB[1], t_AB[0])  # perpendiculaire à AB

    # 3) Force appliquée (décomposée normal/tangent à AB)
    U_k = Fn[0, k] * n_AB + Ft[0, k] * t_AB
    Fx_k = U_k[0]
    Fy_k = U_k[1]

    # 4) Réactions de contact (normales)
    RA_wall2 = lambda_A_wall2[k] * n2_k
    RB_wall2 = lambda_B_wall2[k] * n2_k
    RA_wall1 = lambda_A_wall1[k] * n1
    RB_wall1 = lambda_B_wall1[k] * n1

    # 5) Frictions (tangentielles aux murs)
    RA_wall2_t = lambda_A_wall2_t[k] * t2_k
    RB_wall2_t = lambda_B_wall2_t[k] * t2_k
    RA_wall1_t = lambda_A_wall1_t[k] * t1
    RB_wall1_t = lambda_B_wall1_t[k] * t1

    # 6) Accélération COM
    F_tot = ca.vertcat(Fx_k, Fy_k) \
            + RA_wall2 + RB_wall2 + RA_wall1 + RB_wall1 \
            + RA_wall2_t + RB_wall2_t + RA_wall1_t + RB_wall1_t

    ax_k = F_tot[0] / m
    ay_k = F_tot[1] / m

    # 7) Moment autour du COM
    rA_world = R_k @ a_local
    rB_world = R_k @ b_local

    def moment_z(r, F):
        return r[0]*F[1] - r[1]*F[0]

    alpha_phys = (
        moment_z(rA_world, RA_wall2)   + moment_z(rB_world, RB_wall2) +
        moment_z(rA_world, RA_wall1)   + moment_z(rB_world, RB_wall1) +
        moment_z(rA_world, RA_wall2_t) + moment_z(rB_world, RB_wall2_t) +
        moment_z(rA_world, RA_wall1_t) + moment_z(rB_world, RB_wall1_t)
    ) / I

    # 8) Intégration RK4
    k1 = f_dynamic(st, ax_k, ay_k, alpha_phys)
    k2 = f_dynamic(st + dt/2*k1, ax_k, ay_k, alpha_phys)
    k3 = f_dynamic(st + dt/2*k2, ax_k, ay_k, alpha_phys)
    k4 = f_dynamic(st + dt*k3,   ax_k, ay_k, alpha_phys)
    opti.subject_to(X[:, k+1] == st + dt/6*(k1 + 2*k2 + 2*k3 + k4))

    # 9) Distances aux murs (non-interpénétration)
    g_val = f_collision(st, psi_k)
    dist_A_wall1 = g_val[0]
    dist_B_wall1 = g_val[1]
    dist_A_wall2 = g_val[4]
    dist_B_wall2 = g_val[5]

    opti.subject_to(dist_A_wall1 >= 0)
    opti.subject_to(dist_B_wall1 >= 0)
    opti.subject_to(dist_A_wall2 >= 0)
    opti.subject_to(dist_B_wall2 >= 0)

    opti.subject_to(lambda_A_wall1[k] >= 0)
    opti.subject_to(lambda_B_wall1[k] >= 0)
    opti.subject_to(lambda_A_wall2[k] >= 0)
    opti.subject_to(lambda_B_wall2[k] >= 0)

    # (OPTIONNEL mais conseillé) Passivité anti-éjection
    eps_v = 1e-6
    omega_k = st[5]
    vCOM = st[3:5]
    def point_velocity(r_local):
        r_world = R_k @ r_local
        omega_cross_r = ca.vertcat(-omega_k * r_world[1], omega_k * r_world[0])
        return vCOM + omega_cross_r
    vA = point_velocity(a_local)
    vB = point_velocity(b_local)
    gdot_A_wall1 = ca.dot(vA, n1)
    gdot_B_wall1 = ca.dot(vB, n1)
    gdot_A_wall2 = ca.dot(vA, n2_k)
    gdot_B_wall2 = ca.dot(vB, n2_k)
    opti.subject_to(lambda_A_wall1[k] * gdot_A_wall1 <= eps_v)
    opti.subject_to(lambda_B_wall1[k] * gdot_B_wall1 <= eps_v)
    opti.subject_to(lambda_A_wall2[k] * gdot_A_wall2 <= eps_v)
    opti.subject_to(lambda_B_wall2[k] * gdot_B_wall2 <= eps_v)

    # 10) Coûts (si tu veux régulariser)
    lam_cost += (lambda_A_wall1[k]**2 + lambda_B_wall1[k]**2 +
                 lambda_A_wall2[k]**2 + lambda_B_wall2[k]**2 +
                 lambda_A_wall1_t[k]**2 + lambda_B_wall1_t[k]**2 +
                 lambda_A_wall2_t[k]**2 + lambda_B_wall2_t[k]**2)

    ctrl_cost += Fn[0, k]**2 + Ft[0, k]**2


dFt_max = 50.0  # N/s à ajuster
dFn_max = 50.0 

for k in range(N-1):
    opti.subject_to((Fn[0,k+1] - Fn[0,k]) * N <=  dFn_max * T_total)
    opti.subject_to((Fn[0,k+1] - Fn[0,k]) * N >= -dFn_max * T_total)
    opti.subject_to((Ft[0,k+1] - Ft[0,k]) * N <=  dFt_max * T_total)
    opti.subject_to((Ft[0,k+1] - Ft[0,k]) * N >= -dFt_max * T_total)





# Objective: time-optimal + small regularization
wL = 1e-4
wF = 1e-4
opti.minimize(T_total + wL*lam_cost + wF*ctrl_cost)

# Boundary conditions
opti.subject_to(X[:, 0] == ca.vertcat(a/2 + 0.001, b/2, 0, 0, 0, 0))
opti.subject_to(X[0, N] == b/2)
opti.subject_to(X[1, N] == a/2)
opti.subject_to(X[2, N] == ca.pi/2)
opti.subject_to(X[3:6, N] == 0)

opti.subject_to(T_total >= 0.1)

# Initial guesses
opti.set_initial(T_total, 5.0)
opti.set_initial(X, 0.1)
opti.set_initial(Ft, 0.0)

opti.solver('ipopt')
sol = opti.solve()

t_opt = float(sol.value(T_total))
Ft_opt = ca.DM(sol.value(Ft))  # forme (1,N) ou (N,1) selon cas
Fn_opt = ca.DM(sol.value(Fn))
X_opt  = ca.DM(sol.value(X))

# a_local/b_local en DM (pas MX)
a_loc = ca.DM([-a/2, -b/2])
b_loc = ca.DM([-a/2,  b/2])

# force reconstruite (DM 2xN)
U_rec = ca.DM.zeros(2, N)

for k in range(N):
    xk = X_opt[:, k]
    theta_k = xk[2]
    C_k = xk[0:2]

    # Rotation 2x2 en DM
    c = ca.cos(theta_k)
    s = ca.sin(theta_k)
    R = ca.vertcat(
        ca.horzcat(c, -s),
        ca.horzcat(s,  c)
    )

    P_A_k = C_k + R @ a_loc
    P_B_k = C_k + R @ b_loc

    dAB = P_B_k - P_A_k
    t_AB = dAB / (ca.sqrt(ca.dot(dAB, dAB)) + 1e-8)
    n_AB = ca.vertcat(-t_AB[1], t_AB[0])

    # gérer la forme de Fn_opt/Ft_opt (1xN ou Nx1)
    Fn_k = Fn_opt[0, k] if Fn_opt.size2() == N else Fn_opt[k]
    Ft_k = Ft_opt[0, k] if Ft_opt.size2() == N else Ft_opt[k]

    U_k = Fn_k * n_AB + Ft_k * t_AB
    U_rec[:, k] = U_k

# Si tu veux un numpy pour matplotlib :
u_opt = np.array(U_rec)

dt_opt = t_opt / N
time_steps = np.linspace(0, t_opt - dt_opt, N)

# ---- Plot U ----
plt.figure(figsize=(10, 5))
plt.step(time_steps, u_opt[0, :], where='post', label='Fx', lw=2)
plt.step(time_steps, u_opt[1, :], where='post', label='Fy', lw=2)
plt.title(f"Forces appliquées (Fn_mean={np.mean(Fn_opt):.1f}N, T_total={t_opt:.2f}s)")
plt.xlabel("Temps (s)")
plt.ylabel("Force (N)")
plt.grid(True, alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

lam_A1 = sol.value(lambda_A_wall1).ravel()
lam_B1 = sol.value(lambda_B_wall1).ravel()
lam_A2 = sol.value(lambda_A_wall2).ravel()
lam_B2 = sol.value(lambda_B_wall2).ravel()


plt.figure(figsize=(10, 5))

plt.step(time_steps, lam_A1, where='post', label='lambda A mur1', lw=2)
plt.step(time_steps, lam_B1, where='post', label='lambda B mur1', lw=2)
plt.step(time_steps, lam_A2, where='post', label='lambda A mur2', lw=2)
plt.step(time_steps, lam_B2, where='post', label='lambda B mur2', lw=2)

plt.title(f"Forces appliquées (Fn_mean={np.mean(Fn_opt):.1f}N, T_total={t_opt:.2f}s)")
plt.xlabel("Temps (s)")
plt.ylabel("Lambdas (N)")
plt.grid(True, alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()



###animation 

def animate_solution(sol_X, sol_T):
    N_steps = sol_X.shape[1]
    dt = sol_T / (N_steps - 1)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-0.2, 1.0)
    ax.set_ylim(-0.2, 1.0)
    ax.set_aspect('equal')
    ax.grid(True)

    # Éléments graphiques
    line_wall1, = ax.plot([], [], 'k-', lw=4, label="Mur 1 (Sol)")
    line_wall2, = ax.plot([], [], 'r-', lw=4, label="Mur 2 (Pivot)")
    rect = plt.Rectangle((0, 0), a, b, angle=0, color='blue', alpha=0.6)
    ax.add_patch(rect)

    def update(k):
        # 1. Mise à jour du mur 2
        t_curr = k * dt
        psi_k = sol.value(wall_state_casadi(t_curr))
        
        # Mur 1 (statique sur X)
        line_wall1.set_data([0, 1], [0, 0])
        
        # Mur 2 (pivotant depuis l'origine)
        x_w2 = [0, ca.cos(psi_k)]
        y_w2 = [0, ca.sin(psi_k)]
        line_wall2.set_data(x_w2, y_w2)

        # 2. Mise à jour du bac
        xc_k = sol_X[0, k]
        yc_k = sol_X[1, k]
        theta_k = sol_X[2, k]
        
        # Calcul du coin bas-gauche pour le dessin du rectangle
        # Le rectangle plt.Rectangle se définit par son coin (x,y) et son angle
        # On doit retrouver le coin local (-a/2, -b/2) dans le monde
        R = np.array([[np.cos(theta_k), -np.sin(theta_k)],
                      [np.sin(theta_k),  np.cos(theta_k)]])
        corner_local = np.array([-a/2, -b/2])
        corner_world = np.array([xc_k, yc_k]) + R @ corner_local
        
        rect.set_xy(corner_world)
        rect.angle = np.degrees(theta_k)
        
        return line_wall1, line_wall2, rect

    ani = animation.FuncAnimation(fig, update, frames=N_steps, interval=50, blit=True)
    plt.legend()
    plt.show()
    return ani

# Appel de l'animation après le sol = opti.solve()
animate_solution(sol.value(X), sol.value(T_total))