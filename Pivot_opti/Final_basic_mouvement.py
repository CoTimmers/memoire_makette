"je veux créer un code basic pour montrer c'est quoi le mouvement basic pour atteindre la position finale"
"La plaque rectangulaire commence collé au mur 1 et 2, avec le long coté  collé au mur 2"
"Mur 1: y=0"
"Mur 2: x=0"
"Mur 2 pivote de la position verticale, à vitesse constante pour atteindre + 90°"

"Première phase: Appliquer une force au COM pour rester coller au mur pendant le pivotement."
"Deuxième phase: Tirer le bac vers x+ pour que le coin supérieur gauche dépasse le pivot"
"Troisème phase: Fermer le mur qui pivote"
"Quatrième phase: Tirer le bac vers x- pour le positionner dans le coin"


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from matplotlib.animation import FFMpegWriter

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

#acc & decell
psi_dot_dot_1 = (2 * (psi_demi - psi0)) / (T/2)**2
psi_dot_dot_2 = (2 * (psiF - psi_demi)) / (T/2)**2


N = 60
dt = T/(N-1)
t_grid = np.linspace(0, T, N)

#position final crane phase 2
pf = [0.25, 0.03]


xc = 0.15
yc = 0.20
vcx0 = 0.0
vcy0 = 0.0

pivot_world = np.array([0.0, 0.0], dtype=float)
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
        # Phase 1: Acceleration
        accel = psi_dot_dot_1
        psi = psi0 + 0.5 * accel * t**2
        psi_dot = accel * t
    elif t <= T:
        # Phase 2: Braking
        v_mid = psi_dot_dot_1 * t_mid
        p_mid = psi0 + 0.5 * psi_dot_dot_1 * t_mid**2
        
        dt_phase2 = t - t_mid
        accel = -psi_dot_dot_2
        psi = p_mid + v_mid * dt_phase2 + 0.5 * accel * dt_phase2**2
        psi_dot = v_mid + accel * dt_phase2
    else:
        # Phase 3: wall fixed
        psi = psiF
        psi_dot = 0.0
        accel = 0.0
        
    return psi, psi_dot, accel

def normal_vector(t):
    psi, _,_= wall_state(t)
    # psi = psi0 + psi_dot * t
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


def distance(t, xc, yc, theta):
    n1, n2 = normal_vector(t)
    corners = corners_position(xc, yc, theta)
    da = corners[0]
    db = corners[1]
    dist_wall_2_a = np.dot(n2, da - pivot_world)
    dist_wall_2_b = np.dot(n2, db - pivot_world)
    return dist_wall_2_a, dist_wall_2_b



def intensity_induced_force(psi, psi_dot, state): #on dit qu'on veut coller le coin B à chaque time-step, donc on doit --> tau = f x r
    x, y, theta, vx, vy, theta_dot = state
    Kp = 50
    Kd = 50
    e = psi - theta
    ew =  psi_dot - theta_dot  
    alpha = Kp * e + Kd * ew
    I = (m/12) * (a**2 + b**2) 
    f = (I * alpha )/(b/2)
    return f


def orientation_induced_force(f, state):
    x, y, theta, vx, vy, theta_dot = state
    Fc = np.array([-f, 0])
    orientation = rotation_matrix(theta) @ Fc
    return orientation


def wrap_angle(a):
    return (a + np.pi) % (2*np.pi) - np.pi

#Phase 1: stay close to the pivoting wall
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

    S = np.array([[0.0, -1.0],
                  [1.0,  0.0]], dtype=float)
    
    

    for k in range(N):
        t = k * dt
        psi, psi_dot, psi_dot_dot = wall_state(t)
        # psi = psi0 + psi_dot * t
        # current_psi_ddot = psi_dot_dot_1 if psi < np.pi/4 else -psi_dot_dot_2

        n1, n2 , t1, t2= normal_vector(t)
        n2 = n2 / (np.linalg.norm(n2) + 1e-12)
        t2 = t2 / (np.linalg.norm(t2) + 1e-12)

        x, y, theta, vx, vy, theta_dot = state
        
        #Acceleration of the wall: psi_dot_dot, used in F=m*a to find F. 
        Fy = (I * psi_dot_dot )/ (b/2)

        #Position et vitesse du bac au time-step
        theta_dot += psi_dot_dot * dt
        theta += theta_dot * dt

        #Projeter la norme de la force nécessaire sur le vecteur normal
        F = -Fy * n2
        psi = psi0 + psi_dot * t
        r_mur = b
        a_n_mur = r_mur * psi_dot
        a_t_mur = r_mur * psi_dot_dot
        a_mur = np.sqrt(a_n_mur**2 + a_t_mur**2)

        #calcul pour la plaque qui va suivre le mur avec la meme vitesse
        r_bac = np.sqrt((a/2)**2+(b/2)**2)
        a_n_bac = r_bac * psi_dot
        a_t_bac = r_bac * psi_dot_dot
        a_bac = np.sqrt(a_n_bac**2 + a_t_bac**2)

        # 1. Vecteur du COM vers le pivot
        r_vec = np.array([pivot_world[0]- x, pivot_world[1] - y])
        r_norm = np.linalg.norm(r_vec) + 1e-12
        u_r = r_vec / r_norm
        u_t = np.array([-u_r[1], u_r[0]])

        an = r_norm * (theta_dot**2)      
        at = r_norm * psi_dot_dot
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


state0 = [xc, yc, 0.0, 0.0, 0.0, 0.0] 
state_hist, force_norm_hist, psi_hist, n2_hist = simulate_phase1(state0)


#Phase 2: tranlate the crate horizontally at the right of the pivot
def crane_position(state_hist, force_norm_hist, n2_hist, kd = 0.5, cable_length=1.0):
    "ajouter que la grue ne oeut pass se teleporter pour changer de force."
    "je doiss ajoute une vitessse max"

    num_steps = state_hist.shape[0]
    crane_pos_hist = np.zeros((num_steps, 2))

    
    for k in range(num_steps):
        # Position actuelle du COM du bac
        xc, yc = state_hist[k, 0], state_hist[k, 1]
        com_pos = np.array([xc, yc])
    
        n2 = n2_hist[k]

        dist = ((force_norm_hist[k]) / kd )
        
        crane_pos = com_pos -n2 * (cable_length + dist)

        crane_pos_hist[k] = crane_pos
        
    return crane_pos_hist

crane_pos_hist = crane_position(state_hist, force_norm_hist, n2_hist, kd = 10.0, cable_length=0.1)



def simulate_phase2(crane_pos_hist, pf, dt, N):
    state_crane0 = np.asarray(crane_pos_hist[-1], float).reshape(2,)
    pf = np.asarray(pf, float).reshape(2,)

    d = pf - state_crane0
    L = np.linalg.norm(d)
    crane_hist = np.zeros((N, 2), float)

    if L < 1e-12:
        crane_hist[:] = state_crane0
        return crane_hist

    u = d / L
    a = 4 * L / (T**2)  # accel scalaire pour acc puis dec et v_fin=0

    s = 0.0
    v = 0.0
    k_mid = (N - 1) // 2

    for k in range(N):
        acc = a if k <= k_mid else -a
        s = s + v*dt + 0.5*acc*dt**2
        v = v + acc*dt

        # clamp sécurité
        s = np.clip(s, 0.0, L)
        crane_hist[k] = state_crane0 + u*s

    crane_hist[-1] = pf
    return crane_hist

crane_pos = simulate_phase2(crane_pos_hist, pf, dt, N)

p0 = state_hist[N-1, 0:2]    
v0 = state_hist[N-1, 3:5]   

def pos_crate(state_crane_hist, p0, v0, kd, m, dt, c_damp = 20.0, y_floor = 0.15, restitution = 0.0, v_eps=1e-3):
    N = state_crane_hist.shape[0]
    p_hist = np.zeros((N, 2), float)
    v_hist = np.zeros((N, 2), float)


    p = np.asarray(p0, dtype=float).reshape(2,)
    v = np.asarray(v0, dtype=float).reshape(2,)

    for k in range(N):
        c = state_crane_hist[k]
        d = c - p                        

        dist = np.linalg.norm(d)
        
        F = kd * (c - p) - c_damp * v
    
        a = F / m 

        p = p + v*dt + 0.5*a*dt**2
        v = v + a*dt

        if p[1] < y_floor:
            p[1] = y_floor
            if v[1] < 0.0:
                v[1] = -restitution * v[1] 

        p_hist[k] = p
        v_hist[k] = v
    
    return p_hist, v_hist


p_hist2, v_hist2 = pos_crate(crane_pos, p0, v0, 10.0, m, dt, c_damp = 10.0, y_floor = 0.15, restitution = 0.0)



#Phase 3: after the wall is closed, the crate is brought in the corner between both walls. 
def simulate_phase3(crane_pos_hist, pf, dt, N):
    state_crane0 = np.asarray(crane_pos_hist, float).reshape(2,)
    pf = np.asarray(pf, float).reshape(2,)

    d = pf - state_crane0
    L = np.linalg.norm(d)
    crane_hist = np.zeros((N, 2), float)

    if L < 1e-12:
        crane_hist[:] = state_crane0
        return crane_hist

    u = d / L
    a = 4 * L / (T**2)  # accel scalaire pour acc puis dec et v_fin=0

    s = 0.0
    v = 0.0
    k_mid = (N - 1) // 2

    for k in range(N):
        acc = a if k <= k_mid else -a
        s = s + v*dt + 0.5*acc*dt**2
        v = v + acc*dt

        
        s = np.clip(s, 0.0, L)
        crane_hist[k] = state_crane0 + u*s

    crane_hist[-1] = pf
    return crane_hist





extra_seconds = 5.0
extra_steps = int(extra_seconds / dt)

# grue reste à la fin de phase2
crane_hold = np.repeat(crane_pos[-1][None, :], extra_steps, axis=0)     # (extra_steps,2)

# trajectoire grue phase2 étendue
crane_phase2_ext = np.vstack([crane_pos, crane_hold])                  # (N+extra_steps,2)

# simuler le bac pendant phase2 + hold
p_hist2_ext, v_hist2_ext = pos_crate(crane_phase2_ext, p0, v0, 10.0, m, dt,
                                     c_damp=10.0, y_floor=0.15, restitution=0.0)

# état initial phase3 = fin du hold
p0_3 = p_hist2_ext[-1]
v0_3 = v_hist2_ext[-1]

# =========================
# 2) PHASE 3 (grue bouge vers pf_phase3)
# =========================
pf_phase3 = np.array([0.22, 0.03], float)

# IMPORTANT: phase3 démarre depuis la position finale de phase2 (pf)
crane_pos_phase_3 = simulate_phase3(crane_pos[-1], pf_phase3, dt, N)

# simuler le bac pendant phase3
p_hist3, v_hist3 = pos_crate(crane_pos_phase_3, p0_3, v0_3, 10.0, m, dt,
                            c_damp=10.0, y_floor=0.15, restitution=0.0)

# =========================
# 3) CONCAT pour animation (phase1 + phase2+hold + phase3)
# =========================
bac_xy_all = np.vstack([
    state_hist[:, 0:2],   # phase1
    p_hist2_ext,          # phase2 + hold
    p_hist3               # phase3
])

theta_all = np.concatenate([
    state_hist[:, 2],
    # np.full(p_hist2_ext.shape[0] + p_hist3.shape[0], state_hist[-1, 2])
    np.full(p_hist2_ext.shape[0] + p_hist3.shape[0], 1.57)
])

crane_all = np.vstack([
    crane_pos_hist,       # phase1
    crane_phase2_ext,     # phase2 + hold
    crane_pos_phase_3     # phase3
])

force_all = np.concatenate([
    force_norm_hist,
    np.zeros(p_hist2_ext.shape[0] + p_hist3.shape[0])
])

frames_total = bac_xy_all.shape[0]
t_total = frames_total * dt

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# =========================
# FIGURE ANIMATION SEULE
# =========================
fig, ax = plt.subplots(figsize=(7, 7))

ax.set_xlim(-0.75, 0.75)
ax.set_ylim(-0.20, 0.75)
ax.set_aspect('equal')
ax.grid(True, linestyle=':')

line_plaque, = ax.plot([], [], 'b-', lw=3, label='Bac')
line_mur,    = ax.plot([], [], 'r-', lw=4, label='Mur 2')
line_mur_horiz, = ax.plot([0.0, 0.5], [0.0, 0.0], 'k-', lw=4, label='Mur 1')
point_B,     = ax.plot([], [], 'go', markersize=8)

line_cable,  = ax.plot([], [], 'k--', lw=1, label='Câble')
point_grue,  = ax.plot([], [], 'ro', markersize=6, label='Grue')

time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, va='top')

ax.legend(loc="lower right")

def smoothstep(tau):
    tau = np.clip(tau, 0.0, 1.0)
    return 3*tau**2 - 2*tau**3

def init():
    line_plaque.set_data([], [])
    line_mur.set_data([], [])
    line_mur_horiz.set_data([0.0, 0.5], [0.0, 0.0])
    point_B.set_data([], [])
    line_cable.set_data([], [])
    point_grue.set_data([], [])
    time_text.set_text('')
    return line_plaque, line_mur, line_mur_horiz, point_B, line_cable, point_grue, time_text

def animate(k):
    t = k * dt

    # Bac
    x, y = bac_xy_all[k]
    theta = theta_all[k]
    R = rotation_matrix(theta)

    corners = np.array([[-a/2, -b/2],
                        [ a/2, -b/2],
                        [ a/2,  b/2],
                        [-a/2,  b/2],
                        [-a/2, -b/2]]).T
    w_corners = (R @ corners).T + np.array([x, y])
    line_plaque.set_data(w_corners[:, 0], w_corners[:, 1])

    # Mur 2 : planning d'affichage indépendant (comme tu faisais)
    T1 = T
    T2 = T
    Thold = extra_seconds

    t_global = t

    if t_global <= T1:
        tau = t_global / T1 if T1 > 0 else 1.0
        psi_draw = psi0 + (psiF - psi0) * smoothstep(tau)

    elif t_global <= T1 + T2:
        psi_draw = psiF

    elif t_global <= T1 + T2 + Thold:
        tau = (t_global - (T1 + T2)) / Thold if Thold > 0 else 1.0
        psi_draw = psiF + (psi0 - psiF) * smoothstep(tau)

    else:
        psi_draw = psi0

    t2 = np.array([-np.sin(psi_draw), np.cos(psi_draw)], dtype=float)
    mur_end = pivot_world + t2 * 0.5
    line_mur.set_data([pivot_world[0], mur_end[0]], [pivot_world[1], mur_end[1]])

    # Point B (coin haut-gauche)
    rB = np.array([x, y]) + R @ np.array([-a/2, b/2])
    point_B.set_data([rB[0]], [rB[1]])

    # Grue + câble
    crane_x, crane_y = crane_all[k]
    line_cable.set_data([x, crane_x], [y, crane_y])
    point_grue.set_data([crane_x], [crane_y])

    time_text.set_text(f"t = {t:.2f} s")

    return line_plaque, line_mur, line_mur_horiz, point_B, line_cable, point_grue, time_text

ani = FuncAnimation(
    fig, animate, frames=frames_total, init_func=init,
    blit=True, interval=50
)

plt.tight_layout()
plt.show()


t_axis = np.arange(crane_all.shape[0]) * dt
xg = crane_all[:, 0]
yg = crane_all[:, 1]

plt.figure(figsize=(6,6))
plt.plot(xg, yg, lw=2, label="Crane trajectory")

# Départ (rouge)
plt.scatter([xg[0]], [yg[0]], s=120, marker="o", color="red", edgecolors="k", zorder=5, label="Start")

# Fin (style “damier” ≈ marqueur X)
plt.scatter([xg[-1]], [yg[-1]], s=160, marker="X", color="white", edgecolors="k", linewidths=2, zorder=6, label="End")

plt.xlabel("x crane (m)")
plt.ylabel("y crane (m)")
plt.grid(True, linestyle=":")
plt.axis("equal")
plt.legend()
plt.tight_layout()
plt.show()




kd = 10.0
c_damp = 10.0

# =========================
# Tension phase 1
# =========================
T_phase1 = force_norm_hist.copy()   # déjà dans ton code (ce que tu plottais)

# =========================
# Tension phase 2 + hold
# =========================
F2 = kd * (crane_phase2_ext - p_hist2_ext) - c_damp * v_hist2_ext
T_phase2 = np.linalg.norm(F2, axis=1)

# =========================
# Tension phase 3
# =========================
F3 = kd * (crane_pos_phase_3 - p_hist3) - c_damp * v_hist3
T_phase3 = np.linalg.norm(F3, axis=1)

# =========================
# CONCAT total + temps
# =========================
Tension_all = np.concatenate([T_phase1, T_phase2, T_phase3])
t_axis = np.arange(Tension_all.shape[0]) * dt

# =========================
# PLOT
# =========================
plt.figure(figsize=(9,4))
plt.plot(t_axis, Tension_all, lw=2)
plt.xlabel("Time (s)")
plt.ylabel("Rope tension (N)")
plt.grid(True, linestyle=":")
plt.tight_layout()
plt.show()



# =========================
# SAUVEGARDE EN GIF
# =========================
# Nécessite: pip install pillow
gif_name = "animation_bac.gif"
ani.save(gif_name, writer=PillowWriter(fps=20))
print(f"GIF sauvegardé: {gif_name}")

mp4_name = "animation_bac2.mp4"
ani.save(mp4_name, writer=FFMpegWriter(fps=30, bitrate=1800))
print(f"MP4 sauvegardé: {mp4_name}")