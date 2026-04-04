import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# PARAMETRES PHYSIQUES / DYNAMIQUES
# ==========================================
gamma = 1.0   # coefficient dynamique
# theta_ddot = gamma * u

# ==========================================
# PARAMETRES DU CONTROLEUR PD
# ==========================================
Kp = 4.0
Kd = 3.0

# ==========================================
# CONDITIONS INITIALES
# ==========================================
theta0_deg = 0.0
theta0 = np.deg2rad(theta0_deg)   # angle initial du bac [rad]
omega0 = 0.0                      # vitesse angulaire initiale [rad/s]

# ==========================================
# PARAMETRES DE SIMULATION
# ==========================================
dt = 0.001
t_final = 8.0
time = np.arange(0, t_final + dt, dt)

# ==========================================
# CONSIGNE : MUR EN ROTATION A VITESSE CONSTANTE
# ==========================================
psi_dot_const_deg = 5.0
psi_dot_const = np.deg2rad(psi_dot_const_deg)   # [rad/s]

def psi_t(t):
    return psi_dot_const * t

def psi_dot_t(t):
    return psi_dot_const

# ==========================================
# STOCKAGE
# ==========================================
theta = np.zeros_like(time)
omega = np.zeros_like(time)
psi = np.zeros_like(time)
psi_dot = np.zeros_like(time)

error = np.zeros_like(time)
error_dot = np.zeros_like(time)

u = np.zeros_like(time)
alpha_ang = np.zeros_like(time)

contact = np.zeros_like(time, dtype=bool)

# ==========================================
# INITIALISATION
# ==========================================
theta[0] = theta0
omega[0] = omega0
psi[0] = psi_t(time[0])
psi_dot[0] = psi_dot_t(time[0])

# Vérification cohérence initiale
if theta[0] > psi[0]:
    theta[0] = psi[0]
    omega[0] = psi_dot[0]
    contact[0] = True

# ==========================================
# SIMULATION
# ==========================================
for k in range(len(time) - 1):
    # Consigne mur
    psi[k] = psi_t(time[k])
    psi_dot[k] = psi_dot_t(time[k])

    # Erreurs
    error[k] = psi[k] - theta[k]
    error_dot[k] = psi_dot[k] - omega[k]

    # Si contact déjà actif : le bac suit le mur
    if contact[k]:
        theta[k] = psi[k]
        omega[k] = psi_dot[k]

        u[k] = 0.0
        alpha_ang[k] = 0.0

        psi_next = psi_t(time[k + 1])
        psi_dot_next = psi_dot_t(time[k + 1])

        theta[k + 1] = psi_next
        omega[k + 1] = psi_dot_next
        contact[k + 1] = True
        continue

    # ==========================================
    # PHASE LIBRE : CONTROLE PD
    # ==========================================
    u[k] = Kp * error[k] + Kd * error_dot[k]

    # Dynamique
    alpha_ang[k] = gamma * u[k]

    # Intégration Euler
    omega_free = omega[k] + dt * alpha_ang[k]
    theta_free = theta[k] + dt * omega[k]

    # Consigne à l'instant suivant
    psi_next = psi_t(time[k + 1])
    psi_dot_next = psi_dot_t(time[k + 1])

    # ==========================================
    # TEST DE CONTACT
    # ==========================================
    # Si le mouvement libre dépasse le mur, on active le contact
    if theta_free >= psi_next:
        theta[k + 1] = psi_next
        omega[k + 1] = psi_dot_next
        contact[k + 1] = True
    else:
        theta[k + 1] = theta_free
        omega[k + 1] = omega_free
        contact[k + 1] = False

# ==========================================
# DERNIER POINT
# ==========================================
psi[-1] = psi_t(time[-1])
psi_dot[-1] = psi_dot_t(time[-1])

error[-1] = psi[-1] - theta[-1]
error_dot[-1] = psi_dot[-1] - omega[-1]

if contact[-1]:
    theta[-1] = psi[-1]
    omega[-1] = psi_dot[-1]
    u[-1] = 0.0
    alpha_ang[-1] = 0.0
else:
    u[-1] = Kp * error[-1] + Kd * error_dot[-1]
    alpha_ang[-1] = gamma * u[-1]

# ==========================================
# RESULTATS
# ==========================================
u_max = np.max(np.abs(u))
omega_max = np.max(np.abs(omega))
alpha_ang_max = np.max(np.abs(alpha_ang))

print("===== PARAMETRES =====")
print(f"gamma = {gamma}")
print(f"Kp = {Kp}")
print(f"Kd = {Kd}")
print(f"theta0 = {theta0_deg} deg")
print(f"psi_dot = {psi_dot_const_deg} deg/s")
print()

print("===== RESULTATS =====")
print(f"Commande maximale |u|_max = {u_max:.3f}")
print(f"Vitesse angulaire maximale |omega|_max = {omega_max:.3f} rad/s")
print(f"Accélération angulaire maximale |theta_ddot|_max = {alpha_ang_max:.3f} rad/s²")

if np.any(contact):
    first_contact_idx = np.argmax(contact)
    print(f"Premier contact à t ~ {time[first_contact_idx]:.3f} s")
else:
    print("Aucun contact détecté.")

# ==========================================
# AFFICHAGE
# ==========================================
plt.figure(figsize=(10, 5))
plt.plot(time, np.rad2deg(theta), label="theta(t) : bac")
plt.plot(time, np.rad2deg(psi), '--', label="psi(t) : mur")
plt.xlabel("Temps [s]")
plt.ylabel("Angle [deg]")
plt.title("Suivi du mur avec contrôleur PD")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.figure(figsize=(10, 5))
plt.plot(time, np.rad2deg(error), label="e(t) = psi - theta")
plt.xlabel("Temps [s]")
plt.ylabel("Erreur [deg]")
plt.title("Erreur de suivi")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.figure(figsize=(10, 5))
plt.plot(time, np.rad2deg(error_dot), label="e_dot(t) = psi_dot - omega")
plt.xlabel("Temps [s]")
plt.ylabel("Erreur de vitesse [deg/s]")
plt.title("Erreur de vitesse angulaire")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.figure(figsize=(10, 5))
plt.plot(time, u, label="u(t)")
plt.xlabel("Temps [s]")
plt.ylabel("Commande")
plt.title("Commande PD")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.figure(figsize=(10, 5))
plt.plot(time, np.rad2deg(omega), label="omega(t)")
plt.plot(time, np.rad2deg(psi_dot), '--', label="psi_dot(t) : mur")
plt.xlabel("Temps [s]")
plt.ylabel("Vitesse angulaire [deg/s]")
plt.title("Vitesse angulaire du bac et du mur")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.figure(figsize=(10, 5))
plt.plot(time, alpha_ang, label="theta_ddot(t)")
plt.xlabel("Temps [s]")
plt.ylabel("Accélération angulaire [rad/s²]")
plt.title("Accélération angulaire")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.figure(figsize=(10, 5))
plt.plot(time, contact.astype(int), label="contact")
plt.xlabel("Temps [s]")
plt.ylabel("0 = libre, 1 = contact")
plt.title("Etat de contact")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()