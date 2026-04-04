import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# PARAMETRES PHYSIQUES
# ==========================================
m = 7.0                  # masse [kg]
mu = 0.2                 # coefficient de friction
theta_deg = 34.0         # angle de la force [deg]
theta = np.deg2rad(theta_deg)

# facteur utile : cos(theta) - mu*sin(theta)
c = np.cos(theta) - mu * np.sin(theta)

# ==========================================
# PARAMETRES DU CONTROLEUR
# ==========================================
# Kp = 20.0
# Ki = 8.0

# Kp = 10.0
# Ki = 4.0

Kp = 2.8
Ki = 0.7

# ==========================================
# CONDITIONS INITIALES
# ==========================================
l0 = 0.20                # distance initiale au mur [m]
v0 = 0.0                 # vitesse initiale [m/s]
I0 = 0.0                 # état intégral initial

# ==========================================
# PARAMETRES DE SIMULATION
# ==========================================
dt = 0.001               # pas de temps [s]
t_final = 8.0            # durée totale [s]
time = np.arange(0, t_final + dt, dt)

# ==========================================
# VECTEURS DE STOCKAGE
# ==========================================
l = np.zeros_like(time)   # distance au mur
v = np.zeros_like(time)   # vitesse
I = np.zeros_like(time)   # intégrale de l'erreur
u = np.zeros_like(time)   # commande
a = np.zeros_like(time)   # accélération

# ==========================================
# INITIALISATION
# ==========================================
l[0] = l0
v[0] = v0
I[0] = I0

# ==========================================
# SIMULATION (Euler explicite)
# ==========================================
for k in range(len(time) - 1):
    # Si le bac est arrivé au mur, on le bloque
    if l[k] <= 0:
        l[k] = 0.0
        v[k] = 0.0
        u[k] = 0.0
        a[k] = 0.0

        l[k + 1] = 0.0
        v[k + 1] = 0.0
        I[k + 1] = I[k]   # on fige l'intégrale
        continue

    # Commande PI
    # erreur e(t) = l(t), car l* = 0
    u[k] = Kp * l[k] + Ki * I[k]

    # Dynamique :
    # m * l_ddot = -u * (cos(theta) - mu*sin(theta))
    a[k] = -(u[k] * c) / m

    # Intégration Euler
    v[k + 1] = v[k] + dt * a[k]
    l[k + 1] = l[k] + dt * v[k]
    I[k + 1] = I[k] + dt * l[k]

    # Butée physique : on empêche l de devenir négatif
    if l[k + 1] < 0:
        l[k + 1] = 0.0
        v[k + 1] = 0.0

# Dernier point
if l[-1] > 0:
    u[-1] = Kp * l[-1] + Ki * I[-1]
    a[-1] = -(u[-1] * c) / m
else:
    u[-1] = 0.0
    a[-1] = 0.0
    v[-1] = 0.0
    l[-1] = 0.0

# ==========================================
# INFOS UTILES
# ==========================================
idx_arrival = np.where(l <= 1e-4)[0]
arrival_time = time[idx_arrival[0]] if len(idx_arrival) > 0 else None

u_max = np.max(u)
v_max = np.max(np.abs(v))
a_max = np.max(np.abs(a))

print("===== PARAMETRES =====")
print(f"m = {m} kg")
print(f"mu = {mu}")
print(f"theta = {theta_deg} deg")
print(f"facteur utile c = cos(theta) - mu sin(theta) = {c:.4f}")
print(f"Kp = {Kp}")
print(f"Ki = {Ki}")
print()

print("===== RESULTATS =====")
if arrival_time is not None:
    print(f"Temps d'arrivée au mur ~ {arrival_time:.3f} s")
else:
    print("Le bac n'atteint pas le mur sur la durée simulée.")

print(f"Commande maximale u_max = {u_max:.3f}")
print(f"Vitesse maximale |v|_max = {v_max:.3f} m/s")
print(f"Acceleration maximale |a|_max = {a_max:.3f} m/s^2")

# ==========================================
# AFFICHAGE
# ==========================================
plt.figure(figsize=(10, 5))
plt.plot(time, l, label="l(t)")
plt.xlabel("Temps [s]")
plt.ylabel("Distance au mur l(t) [m]")
plt.title("Distance restante au mur")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.figure(figsize=(10, 5))
plt.plot(time, v, label="v(t)")
plt.xlabel("Temps [s]")
plt.ylabel("Vitesse v(t) [m/s]")
plt.title("Vitesse du bac")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.figure(figsize=(10, 5))
plt.plot(time, u, label="u(t)")
plt.xlabel("Temps [s]")
plt.ylabel("Commande u(t)")
plt.title("Commande PI")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.figure(figsize=(10, 5))
plt.plot(time, I, label="I(t)")
plt.xlabel("Temps [s]")
plt.ylabel("Etat intégral I(t)")
plt.title("Intégrale de l'erreur")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.figure(figsize=(10, 5))
plt.plot(time, a, label="a(t)")
plt.xlabel("Temps [s]")
plt.ylabel("Accélération a(t) [m/s²]")
plt.title("Accélération du bac")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()