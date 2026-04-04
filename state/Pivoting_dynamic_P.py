import numpy as np
import matplotlib.pyplot as plt

gamma = 1.0
Kp = 1.0

dt = 0.001
t_final = 8.0
time = np.arange(0, t_final + dt, dt)

theta0 = 0.0
omega0 = 0.0

psi_dot_const = np.deg2rad(5.0)   # 5 deg/s

def psi_t(t):
    return psi_dot_const * t

theta = np.zeros_like(time)
omega = np.zeros_like(time)
psi = np.zeros_like(time)
error = np.zeros_like(time)
u = np.zeros_like(time)
alpha_ang = np.zeros_like(time)

theta[0] = theta0
omega[0] = omega0
psi[0] = psi_t(time[0])

for k in range(len(time) - 1):
    psi[k] = psi_t(time[k])

    error[k] = psi[k] - theta[k]
    if error[k] < 0:
        error[k] = 0.0

    u[k] = Kp * error[k]
    alpha_ang[k] = gamma * u[k]

    omega[k + 1] = omega[k] + dt * alpha_ang[k]
    theta[k + 1] = theta[k] + dt * omega[k]

    psi_next = psi_t(time[k + 1])

    # contrainte physique : pas de dépassement
    if theta[k + 1] > psi_next:
        theta[k + 1] = psi_next
        omega[k + 1] = 0.0   # ou autre choix physique

psi[-1] = psi_t(time[-1])
error[-1] = max(psi[-1] - theta[-1], 0.0)
u[-1] = Kp * error[-1]
alpha_ang[-1] = gamma * u[-1]

plt.figure(figsize=(10,5))
plt.plot(time, np.rad2deg(theta), label='theta(t) : bac')
plt.plot(time, np.rad2deg(psi), '--', label='psi(t) : mur')
plt.xlabel('Temps [s]')
plt.ylabel('Angle [deg]')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.figure(figsize=(10,5))
plt.plot(time, np.rad2deg(error), label='e(t)=psi-theta')
plt.xlabel('Temps [s]')
plt.ylabel('Erreur [deg]')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.figure(figsize=(10,5))
plt.plot(time, u, label='u(t)')
plt.xlabel('Temps [s]')
plt.ylabel('Commande')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()