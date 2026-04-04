import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# PARAMETRES PHYSIQUES / DYNAMIQUES
# ==========================================
gamma = 1.0   # coefficient dynamique rotationnel
# theta_ddot = -gamma * u

# ==========================================
# PARAMETRES DU CONTROLEUR
# ==========================================
Kp = 4.0
Ki = 1.0

# Kp = 2.0
# Ki = 0.7

# ==========================================
# CONDITIONS INITIALES
# ==========================================
theta0_deg = 40.0
theta0 = np.deg2rad(theta0_deg)   # angle initial [rad]
omega0 = 0.0                      # vitesse angulaire initiale [rad/s]
I0 = 0.0                          # état intégral initial

# ==========================================
# PARAMETRES DE SIMULATION
# ==========================================
dt = 0.001
t_final = 8.0
time = np.arange(0, t_final + dt, dt)

# seuil de convergence
theta_tol_deg = 0.5
theta_tol = np.deg2rad(theta_tol_deg)

# ==========================================
# STOCKAGE
# ==========================================
theta = np.zeros_like(time)
omega = np.zeros_like(time)
I = np.zeros_like(time)
u = np.zeros_like(time)
alpha_ang = np.zeros_like(time)   # accélération angulaire

# ==========================================
# INITIALISATION
# ==========================================
theta[0] = theta0
omega[0] = omega0
I[0] = I0

# ==========================================
# SIMULATION AVEC BUTEE PHYSIQUE A theta = 0
# ==========================================
for k in range(len(time) - 1):
    # Si le bac a déjà atteint le mur final, on le bloque
    if theta[k] <= 0:
        theta[k] = 0.0
        omega[k] = 0.0
        u[k] = 0.0
        alpha_ang[k] = 0.0

        theta[k + 1] = 0.0
        omega[k + 1] = 0.0
        I[k + 1] = I[k]   # on fige l'intégrateur
        continue

    # Commande PI
    # erreur e(t) = theta(t), car theta* = 0
    u[k] = Kp * theta[k] + Ki * I[k]

    # Dynamique de rotation
    # theta_ddot = -gamma * u
    alpha_ang[k] = -gamma * u[k]

    # Intégration Euler
    omega[k + 1] = omega[k] + dt * alpha_ang[k]
    theta[k + 1] = theta[k] + dt * omega[k]

    # Intégrateur
    I[k + 1] = I[k] + dt * theta[k]

    # Butée physique : le mur empêche theta < 0
    if theta[k + 1] < 0:
        theta[k + 1] = 0.0
        omega[k + 1] = 0.0
        I[k + 1] = I[k]   # on peut aussi figer l'intégrale au contact

# Dernier point
if theta[-1] > 0:
    u[-1] = Kp * theta[-1] + Ki * I[-1]
    alpha_ang[-1] = -gamma * u[-1]
else:
    theta[-1] = 0.0
    omega[-1] = 0.0
    u[-1] = 0.0
    alpha_ang[-1] = 0.0

# ==========================================
# RESULTATS
# ==========================================
idx_conv = np.where(theta <= theta_tol)[0]
conv_time = time[idx_conv[0]] if len(idx_conv) > 0 else None

u_max = np.max(np.abs(u))
omega_max = np.max(np.abs(omega))
alpha_ang_max = np.max(np.abs(alpha_ang))

print("===== PARAMETRES =====")
print(f"gamma = {gamma}")
print(f"Kp = {Kp}")
print(f"Ki = {Ki}")
print(f"theta0 = {theta0_deg} deg")
print(f"theta* = 0 deg")
print()

print("===== RESULTATS =====")
if conv_time is not None:
    print(f"Temps d'arrivée à la butée angulaire ~ {conv_time:.3f} s")
else:
    print("L'angle n'atteint pas le seuil sur la durée simulée.")

print(f"Commande maximale |u|_max = {u_max:.3f}")
print(f"Vitesse angulaire maximale |omega|_max = {omega_max:.3f} rad/s")
print(f"Accélération angulaire maximale |theta_ddot|_max = {alpha_ang_max:.3f} rad/s²")

# ==========================================
# AFFICHAGE
# ==========================================
plt.figure(figsize=(10, 5))
plt.plot(time, np.rad2deg(theta), label="theta(t)")
plt.axhline(theta_tol_deg, color='gray', linestyle='--', linewidth=1, label='tolérance')
plt.axhline(0.0, color='red', linestyle='--', linewidth=1, label='butée physique')
plt.xlabel("Temps [s]")
plt.ylabel("Angle theta(t) [deg]")
plt.title("Evolution de l'angle avec butée physique")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.figure(figsize=(10, 5))
plt.plot(time, omega, label="omega(t)")
plt.xlabel("Temps [s]")
plt.ylabel("Vitesse angulaire omega(t) [rad/s]")
plt.title("Evolution de la vitesse angulaire")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.figure(figsize=(10, 5))
plt.plot(time, u, label="u(t)")
plt.xlabel("Temps [s]")
plt.ylabel("Commande u(t)")
plt.title("Evolution de la commande PI")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.figure(figsize=(10, 5))
plt.plot(time, I, label="I(t)")
plt.xlabel("Temps [s]")
plt.ylabel("Etat intégral I(t)")
plt.title("Evolution de l'intégrale de l'erreur angulaire")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.figure(figsize=(10, 5))
plt.plot(time, alpha_ang, label="theta_ddot(t)")
plt.xlabel("Temps [s]")
plt.ylabel("Accélération angulaire [rad/s²]")
plt.title("Evolution de l'accélération angulaire")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()