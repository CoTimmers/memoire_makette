import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# PARAMETRES PHYSIQUES
# ==========================================
m = 7.0                  # masse [kg]
beta = 1.0 / m           # gain dynamique simplifié

# ==========================================
# PARAMETRES DU CONTROLEUR
# ==========================================
#Temps= 3s:
# Kp = 18.67
# Ki = 12.44

#2s
# Kp = 28
# Ki = 28

# Tu peux ensuite tester par exemple :
Kp = 3.0
Ki = 3.0

# ==========================================
# CONDITIONS INITIALES
# ==========================================
d0 = 0.20                # distance initiale [m]
v0 = 0.0                 # vitesse initiale [m/s]
I0 = 0.0                 # état intégral initial

# ==========================================
# PARAMETRES DE SIMULATION
# ==========================================
dt = 0.001
t_final = 8.0
time = np.arange(0, t_final + dt, dt)

# ==========================================
# STOCKAGE
# ==========================================
d = np.zeros_like(time)
v = np.zeros_like(time)
I = np.zeros_like(time)
u = np.zeros_like(time)
a = np.zeros_like(time)

# Initialisation
d[0] = d0
v[0] = v0
I[0] = I0

# ==========================================
# SIMULATION
# ==========================================
for k in range(len(time) - 1):
    # Si le bac a atteint le mur
    if d[k] <= 0:
        d[k] = 0.0
        v[k] = 0.0
        u[k] = 0.0
        a[k] = 0.0

        d[k + 1] = 0.0
        v[k + 1] = 0.0
        I[k + 1] = I[k]
        continue

    # Commande PI
    u[k] = Kp * d[k] + Ki * I[k]

    # Dynamique
    a[k] = -beta * u[k]

    # Intégration Euler
    v[k + 1] = v[k] + dt * a[k]
    d[k + 1] = d[k] + dt * v[k]
    I[k + 1] = I[k] + dt * d[k]

    # Butée physique
    if d[k + 1] < 0:
        d[k + 1] = 0.0
        v[k + 1] = 0.0

# Dernier point
if d[-1] > 0:
    u[-1] = Kp * d[-1] + Ki * I[-1]
    a[-1] = -beta * u[-1]
else:
    d[-1] = 0.0
    v[-1] = 0.0
    u[-1] = 0.0
    a[-1] = 0.0

# ==========================================
# RESULTATS
# ==========================================
idx_arrival = np.where(d <= 1e-4)[0]
arrival_time = time[idx_arrival[0]] if len(idx_arrival) > 0 else None

u_max = np.max(u)
v_max = np.max(np.abs(v))
a_max = np.max(np.abs(a))

print("===== PARAMETRES =====")
print(f"m = {m} kg")
print(f"beta = {beta:.4f}")
print(f"Kp = {Kp}")
print(f"Ki = {Ki}\n")

print("===== RESULTATS =====")
if arrival_time is not None:
    print(f"Temps d'arrivée ~ {arrival_time:.3f} s")
else:
    print("La cible n'est pas atteinte pendant la simulation.")

print(f"Commande maximale u_max = {u_max:.3f}")
print(f"Vitesse maximale |v|_max = {v_max:.3f} m/s")
print(f"Accélération maximale |a|_max = {a_max:.3f} m/s²")

# ==========================================
# AFFICHAGE
# ==========================================
plt.figure(figsize=(10, 5))
plt.plot(time, d, label="d(t)")
plt.xlabel("Temps [s]")
plt.ylabel("Distance d(t) [m]")
plt.title("Evolution de la distance")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.figure(figsize=(10, 5))
plt.plot(time, v, label="v(t)")
plt.xlabel("Temps [s]")
plt.ylabel("Vitesse v(t) [m/s]")
plt.title("Evolution de la vitesse")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.figure(figsize=(10, 5))
plt.plot(time, u, label="u(t)")
plt.xlabel("Temps [s]")
plt.ylabel("Commande u(t)")
plt.title("Evolution de la commande")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.figure(figsize=(10, 5))
plt.plot(time, I, label="I(t)")
plt.xlabel("Temps [s]")
plt.ylabel("Etat intégral I(t)")
plt.title("Evolution de l'intégrale")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.figure(figsize=(10, 5))
plt.plot(time, a, label="a(t)")
plt.xlabel("Temps [s]")
plt.ylabel("Accélération a(t) [m/s²]")
plt.title("Evolution de l'accélération")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()