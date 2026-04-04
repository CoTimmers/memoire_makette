import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# PARAMETRES PHYSIQUES / DYNAMIQUES
# ==========================================
gamma = 1.0   # coefficient dynamique
# theta_ddot = gamma * u

# ==========================================
# PARAMETRES DU CONTROLEUR
# ==========================================
Kp = 4.0
Kd = 3.0
u0 = 0.05   # biais positif pour l'option 1

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
# CONSIGNE : MUR QUI TOURNE A VITESSE CONSTANTE
# jusqu'à 40 deg
# ==========================================
psi_dot_const_deg = 5.0
psi_dot_const = np.deg2rad(psi_dot_const_deg)   # [rad/s]

psi_final_deg = 40.0
psi_final = np.deg2rad(psi_final_deg)

def psi_t(t):
    return min(psi_dot_const * t, psi_final)

def psi_dot_t(t):
    if psi_t(t) < psi_final:
        return psi_dot_const
    else:
        return 0.0

# ==========================================
# FONCTION DE SIMULATION GENERIQUE
# ==========================================
def simulate_controller(controller_type="P_bias"):
    theta = np.zeros_like(time)
    omega = np.zeros_like(time)
    psi = np.zeros_like(time)
    psi_dot = np.zeros_like(time)

    error = np.zeros_like(time)
    error_dot = np.zeros_like(time)

    u = np.zeros_like(time)
    alpha_ang = np.zeros_like(time)

    theta[0] = theta0
    omega[0] = omega0
    psi[0] = psi_t(time[0])
    psi_dot[0] = psi_dot_t(time[0])

    # cohérence physique initiale
    if theta[0] > psi[0]:
        theta[0] = psi[0]

    for k in range(len(time) - 1):
        psi[k] = psi_t(time[k])
        psi_dot[k] = psi_dot_t(time[k])

        # erreur de position
        error[k] = psi[k] - theta[k]
        if error[k] < 0:
            error[k] = 0.0

        # erreur de vitesse
        error_dot[k] = psi_dot[k] - omega[k]

        # ==========================================
        # CHOIX DU CONTROLEUR
        # ==========================================
        if controller_type == "P_bias":
            # Option 1 : u = u0 + Kp*(psi-theta)
            # on garde une commande positive tant que le mouvement n'est pas fini
            if psi[k] < psi_final:
                u[k] = u0 + Kp * error[k]
            else:
                u[k] = Kp * error[k]

        elif controller_type == "unilateral_PD":
            # Option 2 : u = Kp*(psi-theta) + Kd*max(0, psi_dot-omega)
            u[k] = Kp * error[k] + Kd * max(0.0, error_dot[k])

        else:
            raise ValueError("controller_type doit être 'P_bias' ou 'unilateral_PD'")

        # sécurité physique : commande toujours positive
        if u[k] < 0:
            u[k] = 0.0

        # dynamique
        alpha_ang[k] = gamma * u[k]

        # intégration Euler
        omega[k + 1] = omega[k] + dt * alpha_ang[k]
        theta[k + 1] = theta[k] + dt * omega[k]

        # contrainte physique : theta <= psi
        psi_next = psi_t(time[k + 1])
        psi_dot_next = psi_dot_t(time[k + 1])

        if theta[k + 1] > psi_next:
            theta[k + 1] = psi_next

            # on évite d'avoir une vitesse du bac supérieure à celle du mur
            omega[k + 1] = min(omega[k + 1], psi_dot_next)

    # dernier point
    psi[-1] = psi_t(time[-1])
    psi_dot[-1] = psi_dot_t(time[-1])

    error[-1] = psi[-1] - theta[-1]
    if error[-1] < 0:
        error[-1] = 0.0

    error_dot[-1] = psi_dot[-1] - omega[-1]

    if controller_type == "P_bias":
        if psi[-1] < psi_final:
            u[-1] = u0 + Kp * error[-1]
        else:
            u[-1] = Kp * error[-1]
    elif controller_type == "unilateral_PD":
        u[-1] = Kp * error[-1] + Kd * max(0.0, error_dot[-1])

    if u[-1] < 0:
        u[-1] = 0.0

    alpha_ang[-1] = gamma * u[-1]

    return {
        "theta": theta,
        "omega": omega,
        "psi": psi,
        "psi_dot": psi_dot,
        "error": error,
        "error_dot": error_dot,
        "u": u,
        "alpha_ang": alpha_ang
    }

# ==========================================
# SIMULATIONS
# ==========================================
res_P_bias = simulate_controller("P_bias")
res_unilateral_PD = simulate_controller("unilateral_PD")

# ==========================================
# RESULTATS NUMERIQUES
# ==========================================
def print_results(name, res):
    print(f"===== {name} =====")
    print(f"Erreur max [deg] = {np.max(np.rad2deg(res['error'])):.3f}")
    print(f"Commande max = {np.max(res['u']):.3f}")
    print(f"Vitesse max [deg/s] = {np.max(np.rad2deg(res['omega'])):.3f}")
    print(f"Accélération max [rad/s²] = {np.max(res['alpha_ang']):.3f}")
    print()

print("===== PARAMETRES =====")
print(f"gamma = {gamma}")
print(f"Kp = {Kp}")
print(f"Kd = {Kd}")
print(f"u0 = {u0}")
print(f"psi_dot = {psi_dot_const_deg} deg/s")
print(f"psi_final = {psi_final_deg} deg")
print()

print_results("OPTION 1 : P + biais positif", res_P_bias)
print_results("OPTION 2 : PD unilateral", res_unilateral_PD)

# ==========================================
# AFFICHAGE
# ==========================================
plt.figure(figsize=(10, 5))
plt.plot(time, np.rad2deg(res_P_bias["theta"]), label="theta(t) - P+biais")
plt.plot(time, np.rad2deg(res_unilateral_PD["theta"]), label="theta(t) - PD unilatéral")
plt.plot(time, np.rad2deg(res_P_bias["psi"]), '--', label="psi(t) : mur")
plt.xlabel("Temps [s]")
plt.ylabel("Angle [deg]")
plt.title("Comparaison des angles")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.figure(figsize=(10, 5))
plt.plot(time, np.rad2deg(res_P_bias["error"]), label="erreur - P+biais")
plt.plot(time, np.rad2deg(res_unilateral_PD["error"]), label="erreur - PD unilatéral")
plt.xlabel("Temps [s]")
plt.ylabel("Erreur [deg]")
plt.title("Comparaison de l'erreur de suivi")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.figure(figsize=(10, 5))
plt.plot(time, res_P_bias["u"], label="u(t) - P+biais")
plt.plot(time, res_unilateral_PD["u"], label="u(t) - PD unilatéral")
plt.xlabel("Temps [s]")
plt.ylabel("Commande")
plt.title("Comparaison des commandes")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.figure(figsize=(10, 5))
plt.plot(time, np.rad2deg(res_P_bias["omega"]), label="omega(t) - P+biais")
plt.plot(time, np.rad2deg(res_unilateral_PD["omega"]), label="omega(t) - PD unilatéral")
plt.plot(time, np.rad2deg(res_P_bias["psi_dot"]), '--', label="psi_dot(t) : mur")
plt.xlabel("Temps [s]")
plt.ylabel("Vitesse angulaire [deg/s]")
plt.title("Comparaison des vitesses angulaires")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.figure(figsize=(10, 5))
plt.plot(time, res_P_bias["alpha_ang"], label="theta_ddot(t) - P+biais")
plt.plot(time, res_unilateral_PD["alpha_ang"], label="theta_ddot(t) - PD unilatéral")
plt.xlabel("Temps [s]")
plt.ylabel("Accélération angulaire [rad/s²]")
plt.title("Comparaison des accélérations angulaires")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()