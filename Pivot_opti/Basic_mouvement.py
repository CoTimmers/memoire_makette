import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

#Parameters
m = 7
a = 0.3
b = 0.4
I = (m/12.0) * (a**2 + b**2)
mu = 0.3

psi0 = 0
psiF = np.pi/2
psi_demi = np.pi/4
T = 6
psi_dot = (psiF - psi0)/T
psi_dot_dot_1 = (2 * (psi_demi - psi0)) / (T/2)**2
psi_dot_dot_2 = (2 * (psiF - psi_demi)) / (T/2)**2


N = 60
dt = 6/N
t_grid = np.linspace(0, 6, N)








xc = 0.15
yc = 0.20
vcx0 = 0.0
vcy0 = 0.0

pivot_world = np.array([0.0, 0.0], dtype=float)

#définir les points A & B
a_local = np.array([-a/2, -b/2], dtype=float)
b_local = np.array([-a/2, +b/2], dtype=float)

def rotation_matrix(theta):
    r = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return r

def wall_state(t):
    t_mid = T / 2
    
    if t <= 0:
        return psi0, 0.0, 0.0
    
    if t <= t_mid:
        # Phase 1 : Accélération
        accel = psi_dot_dot_1
        psi = psi0 + 0.5 * accel * t**2
        psi_dot = accel * t
    elif t <= T:
        # Phase 2 : Décélération
        v_mid = psi_dot_dot_1 * t_mid
        p_mid = psi0 + 0.5 * psi_dot_dot_1 * t_mid**2
        
        dt_phase2 = t - t_mid
        accel = -psi_dot_dot_2
        psi = p_mid + v_mid * dt_phase2 + 0.5 * accel * dt_phase2**2
        psi_dot = v_mid + accel * dt_phase2
    else:
        # Phase 3 : Fixe à la fin
        psi = psiF
        psi_dot = 0.0
        accel = 0.0
        
    return psi, psi_dot, accel

def normal_vector(t):
    psi = psi0 + psi_dot * t
    n1 = np.array([0.0, 1.0], dtype=float)
    n2 = np.array([np.cos(psi), np.sin(psi)], dtype=float)
    t1 = np.array([1.0, 0.0], dtype=float)
    t2 = np.array([-np.sin(psi), np.cos(psi)], dtype=float)
    return n1, n2, t1, t2

def corners_position(xc, yc, theta) -> np.ndarray: # exprimer la position d'un point dans le monde en fonction du COM
    corners = np.array([
        [-a/2, -b/2], #A
        [-a/2,  b/2], #B
        [ a/2,  b/2], #C
        [ a/2, -b/2]  #D
    ], dtype=float)
    return (rotation_matrix(theta) @ corners.T).T + np.array([xc, yc], dtype=float)

# print(corners_position(-0.15,-0.20, np.pi/2))

def distance(t, xc, yc, theta):
    n1, n2 = normal_vector(t)
    corners = corners_position(xc, yc, theta)
    da = corners[0]
    db = corners[1]
    dist_wall_2_a = np.dot(n2, da - pivot_world)
    dist_wall_2_b = np.dot(n2, db - pivot_world)
    return dist_wall_2_a, dist_wall_2_b

# print(distance(6, 0.15, 0.20, np.pi/2))

def intensity_induced_force(psi, psi_dot, state): #on dit qu'on veut coller le coin B à chaque time-step, donc on doit --> tau = f x r
    x, y, theta, vx, vy, theta_dot = state
    Kp = 50
    Kd = 50
    e = psi - theta
    ew =  psi_dot - theta_dot  
    alpha = Kp * e + Kd * ew
    I = (m/3) * (a**2 + b**2) # moment of inertia depuis le coin du bac
    f = (I * alpha )/(b/2)
    return f


def orientation_induced_force(f, state):
    x, y, theta, vx, vy, theta_dot = state
    Fc = np.array([-f, 0])
    orientation = rotation_matrix(theta) @ Fc
    return orientation


def wrap_angle(a):
    return (a + np.pi) % (2*np.pi) - np.pi


def simulate_phase1(state0):
    dt = T / (N - 1)
    state = np.array(state0, dtype=float)
    state_hist = np.zeros((N, 6), dtype=float)
    force_hist = np.zeros((N, 2), dtype=float)
    psi_hist   = np.zeros(N, dtype=float)
    n2_hist = np.zeros((N, 2), dtype=float)
    force_norm_hist = np.zeros(N)


    B_local = np.array([-a/2,  b/2], dtype=float)
    A_to_com = np.array([a/2, b/2])

    # matrice 90° pour v = v_com + w * S*(r_rel)
    S = np.array([[0.0, -1.0],
                  [1.0,  0.0]], dtype=float)

    for k in range(N):
        t = k * dt

        psi = psi0 + psi_dot * t
        current_psi_ddot = psi_dot_dot_1 if psi < np.pi/4 else -psi_dot_dot_2

        n1, n2 , t1, t2= normal_vector(t)
        n2 = n2 / (np.linalg.norm(n2) + 1e-12)
        t2 = t2 / (np.linalg.norm(t2) + 1e-12)

        x, y, theta, vx, vy, theta_dot = state
        #Force nécessaire pour faire tourner le bac à la vitesse du mur

        Fy = (I * current_psi_ddot )/ (b/2)

        #Position et vitesse du bac au time-step
        theta_dot += current_psi_ddot * dt
        theta += theta_dot * dt

        #Projeter la norme de la force nécessaire sur le vecteur normal
        F = -Fy * n2
        psi = psi0 + psi_dot * t
        r_mur = b
        a_n_mur = r_mur * psi_dot
        a_t_mur = r_mur * current_psi_ddot
        a_mur = np.sqrt(a_n_mur**2 + a_t_mur**2)

        #calcul pour la plaque qui veut suivre la vitesse du mur
        r_bac = np.sqrt((a/2)**2+(b/2)**2)
        a_n_bac = r_bac * psi_dot
        a_t_bac = r_bac * current_psi_ddot
        a_bac = np.sqrt(a_n_bac**2 + a_t_bac**2)

        # 1. Vecteur du COM vers le pivot
        r_vec = np.array([pivot_world[0]- x, pivot_world[1] - y])
        r_norm = np.linalg.norm(r_vec) + 1e-12
        u_r = r_vec / r_norm
        u_t = np.array([-u_r[1], u_r[0]])

        an = r_norm * (theta_dot**2)      
        at = r_norm * current_psi_ddot  
        a_world = an * u_r - at * u_t

        vx += a_world[0] * dt
        vy += a_world[1] * dt
        x += vx * dt
        y += vy * dt

        state = np.array([x, y, theta, vx, vy, theta_dot])
        state_hist[k, :] = state
        force_hist[k, :] = F
        psi_hist[k]  = psi
        n2_hist[k, :] = n2
        force_norm_hist[k] = np.linalg.norm(m * (an + at))


    return state_hist, force_norm_hist, psi_hist, n2_hist

state0 = [xc, yc, 0.0, 0.0, 0.0, 0.0]  # x,y,theta,vx,vy,theta_dot
state_hist, force_norm_hist, psi_hist, n2_hist = simulate_phase1(
    state0
)

print(state_hist[N-1, 0], state_hist[N-1, 1], np.rad2deg(state_hist[N-1, 2]))
# --- Animation et Graphique ---
# On crée deux lignes : une pour l'anim, une pour le graph
fig, (ax, ax_force) = plt.subplots(2, 1, figsize=(7, 10), gridspec_kw={'height_ratios': [2, 1]})

# 1. Setup Animation (DÉZOOMÉ vers la gauche)
ax.set_xlim(-0.6, 0.6) # On élargit à gauche
ax.set_ylim(-0.1, 0.7)
ax.set_aspect('equal')
ax.grid(True, linestyle=':')

line_plaque, = ax.plot([], [], 'b-', lw=3, label='Bac')
line_mur,    = ax.plot([], [], 'r-', lw=4, label='Mur 2')
point_B,     = ax.plot([], [], 'go', markersize=8)

# 2. Setup Graphique Force
ax_force.set_xlim(0, T)
ax_force.set_ylim(0, np.max(force_norm_hist) * 1.2)
ax_force.set_xlabel("Temps (s)")
ax_force.set_ylabel("Force (N)")
ax_force.set_title("Force exercée par le mur sur le bac")
force_plot, = ax_force.plot([], [], 'orange', lw=2)
time_text = ax_force.text(0.05, 0.9, '', transform=ax_force.transAxes)

def init():
    line_plaque.set_data([], [])
    line_mur.set_data([], [])
    point_B.set_data([], [])
    force_plot.set_data([], [])
    return line_plaque, line_mur, point_B, force_plot

def animate(k):
    x, y, theta, _, _, _ = state_hist[k]
    t = k * dt
    
    # Bac
    R = rotation_matrix(theta)
    corners = np.array([[-a/2, -b/2], [a/2, -b/2], [a/2, b/2], [-a/2, b/2], [-a/2, -b/2]]).T
    w_corners = (R @ corners).T + np.array([x, y])
    line_plaque.set_data(w_corners[:, 0], w_corners[:, 1])

    # Mur
    n1, n2, t1, t2 = normal_vector(t)
    mur_end = pivot_world + t2 * 0.5
    line_mur.set_data([pivot_world[0], mur_end[0]], [pivot_world[1], mur_end[1]])

    # Point B
    rB = np.array([x, y]) + R @ np.array([-a/2, b/2])
    point_B.set_data([rB[0]], [rB[1]])

    # Graph Force
    force_plot.set_data(np.linspace(0, t, k+1), force_norm_hist[:k+1])
    time_text.set_text(f't = {t:.2f}s')

    return line_plaque, line_mur, point_B, force_plot

ani = FuncAnimation(fig, animate, frames=N, init_func=init, blit=True, interval=50)

plt.tight_layout()
plt.show()