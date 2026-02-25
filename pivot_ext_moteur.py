"Mur 2 pivote vers l'extérieur"
"le bac commence dans le coin entre les 2 murs et doit finir dans le coin mais pivoté de 90°"
"goal: optimiser le temps de rotation du bac pour arriver dans la position finale"
""
"le mur 2 pivote avec une vitesse constante"
"le glissement sur les surfaces est autorisé mais on doit garder le contacte avec les murs"


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from IPython.display import HTML
import casadi as ca

#Parameters
m = 7
a = 0.3
b = 0.4
I = (m/12.0) * (a**2 + b**2)
mu = 0.3

T = 15
psi0 = 0
psi_demi = np.pi/4
psiF = np.pi/2
psi_dot_dot_acc = (2 * (psi_demi - psi0)) / (T/2)**2
N = 50 #pas

a_local = ca.vertcat([-a/2, -b/2])
b_local = ca.vertcat([-a/2, +b/2])
c_local = ca.vertcat([+a/2, -b/2])
d_local = ca.vertcat( a/2,  b/2) 

def rotation_matrix(theta):
    """Matrice de rotation 2D compatible CasADi"""
    return ca.vertcat(
        ca.horzcat(ca.cos(theta), -ca.sin(theta)),
        ca.horzcat(ca.sin(theta),  ca.cos(theta))
    )


def wall_state_casadi(t, T_total):
    # Paramètres de transition
    psi_start = ca.pi / 2  # Vertical (90°)
    psi_end = ca.pi        # Horizontal vers l'extérieur (180°)
    delta_psi = psi_end - psi_start
    
    t1 = T_total * 0.4  # Fin de l'accélération
    t2 = T_total * 0.6  # Début du freinage
    
    accel = delta_psi / ( (t1**2)/2 + (t2-t1)*t1 + ( (T_total-t2)**2 )/2 ) # Facteur correctif

    # Phase 1 : Accélération
    psi_p1 = psi_start + 0.5 * accel * t**2
    
    # Phase 2 : Vitesse constante (raccord à t1)
    v_at_t1 = accel * t1
    p_at_t1 = psi_start + 0.5 * accel * t1**2
    psi_p2 = p_at_t1 + v_at_t1 * (t - t1)
    
    # Phase 3 : Freinage (raccord à t2)
    p_at_t2 = p_at_t1 + v_at_t1 * (t2 - t1)
    psi_p3 = p_at_t2 + v_at_t1 * (t - t2) - 0.5 * accel * (t - t2)**2
    
    # Assemblage sans sauts
    psi = ca.if_else(t <= t1, psi_p1,
            ca.if_else(t <= t2, psi_p2, psi_p3))
            
    return psi



# x = [x, y, theta, vx, vy, omega]
nx = 6
x_sym = ca.MX.sym('x', nx)
# u = [ax, ay, alpha]
nu = 3
u_sym = ca.MX.sym('u', nu)

# Dynamic : x_dot = f(x, u)

rhs = ca.vertcat(x_sym[3], x_sym[4], x_sym[5], u_sym[0], u_sym[1], u_sym[2])
f_dynamic = ca.Function('f_dyn', [x_sym, u_sym], [rhs])


psi_sym = ca.MX.sym('psi')
n1 = ca.vertcat(0, 1)
n2 = ca.vertcat(ca.sin(psi_sym), -ca.cos(psi_sym))


R_bac = rotation_matrix(x_sym[2])
P_A = x_sym[0:2] + R_bac @ a_local
P_B = x_sym[0:2] + R_bac @ b_local
P_C = x_sym[0:2] + R_bac @ c_local
P_D = x_sym[0:2] + R_bac @ d_local

# Distances aux murs 
g_col = ca.vertcat(
    ca.dot(P_A, n1), ca.dot(P_B, n1), ca.dot(P_C, n1), ca.dot(P_D, n1), 
    ca.dot(P_A, n2), ca.dot(P_B, n2), ca.dot(P_C, n2), ca.dot(P_D, n2)  
)
f_collision = ca.Function('f_col', [x_sym, psi_sym], [g_col])


opti = ca.Opti()

# Variables de décision
T_total = opti.variable()
X = opti.variable(nx, N+1)
U = opti.variable(nu, N)

# Objectif : Minimiser le temps total
opti.minimize(T_total)

dt = T_total / N

for k in range(N):
    
    opti.subject_to(X[:, k+1] == X[:, k] + f_dynamic(X[:, k], U[:, k]) * dt)

    # Contraintes de non-pénétration
    current_t = k * dt
    psi_k = wall_state_casadi(current_t, T_total)
    opti.subject_to(f_collision(X[:, k], psi_k) >= 0)

    # Distances calculées par f_collision
    g_val = f_collision(X[:, k], psi_k)
    dist_sol = ca.fmin(ca.fmin(g_val[0], g_val[1]), ca.fmin(g_val[2], g_val[3]))
    dist_mur = ca.fmin(ca.fmin(g_val[4], g_val[5]), ca.fmin(g_val[6], g_val[7]))

    opti.subject_to(g_val >= 0)


    # t_k = k * dt
    # psi_k = wall_state_casadi(t_k, T_total)
    # n2_k = ca.vertcat(ca.sin(psi_k), -ca.cos(psi_k))
    # accel_vec = U[0:2, k]
    # opti.subject_to(ca.dot(accel_vec, n2_k) >= 0)
    # v_bac = X[3:5, k]





opti.subject_to(X[:, 0] == ca.vertcat(a/2, b/2, 0, 0, 0, 0))

# Arrivée : Bac pivoté de 90° (theta=pi/2) dans le nouveau coin
# Le centre de masse final dépend de la position du mur à T_total (pi/2)
opti.subject_to(X[0, N] == b/2)
opti.subject_to(X[1, N] == a/2)
opti.subject_to(X[2, N] == ca.pi/2) # Angle final
opti.subject_to(X[3:6, N] == 0)    # Vitesse finale nulle

# Bornes sur le temps et les contrôles
opti.subject_to(T_total >= 0.1)
opti.subject_to(opti.bounded(-2, U, 2)) # Accélérations limitées


opti.set_initial(T_total, 5.0)  # On suppose que ça prendra 5s au début
opti.set_initial(X, 0.1)


# --- 7. Résolution ---
opti.solver('ipopt')
sol = opti.solve()

# --- 8. Récupération des résultats ---
t_opt = sol.value(T_total)
x_opt = sol.value(X)
print(f"Temps de rotation optimal : {t_opt:.2f} secondes")







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
        psi_k = sol.value(wall_state_casadi(t_curr, T_total))
        
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