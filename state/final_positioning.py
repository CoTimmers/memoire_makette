import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# PARAMETRES PHYSIQUES
# ==========================================
mu = 0.2
phi_deg = 36.86
phi = np.deg2rad(phi_deg)

# coefficient quasi-statique
alpha = 1.0

# beta = alpha * (cos(phi) - mu*sin(phi))
beta = alpha * (np.cos(phi) - mu * np.sin(phi))

# ==========================================
# SPECIFICATIONS DESIREES
# ==========================================
Ts = 3.0      # temps de convergence souhaité [s]
zeta = 1.0    # amortissement critique
wn = 4.0 / (zeta * Ts)

# Gains PI pour le modèle quasi-statique
Kp = 2 * zeta * wn / beta
Ki = wn**2 / beta

# ==========================================
# CONSIGNE
# ==========================================
l_star = 0.4

# ==========================================
# CONDITIONS INITIALES
# ==========================================
l0 = 0.2
I0 = 0.0

# ==========================================
# PARAMETRES DE SIMULATION
# ==========================================
dt = 0.001
t_final = 8.0
time = np.arange(0, t_final + dt, dt)

tol = 1e-3

# ==========================================
# STOCKAGE
# ==========================================
l = np.zeros_like(time)
I = np.zeros_like(time)
u = np.zeros_like(time)
e = np.zeros_like(time)
l_dot = np.zeros_like(time)

# ==========================================
# INITIALISATION
# ==========================================
l[0] = l0
I[0] = I0
e[0] = l_star - l0

# ==========================================
# SIMULATION
# ==========================================
for k in range(len(time) - 1):
    # erreur signée
    e[k] = l_star - l[k]

    # commande PI
    u[k] = Kp * e[k] + Ki * I[k]

    # dynamique quasi-statique
    l_dot[k] = beta * u[k]

    # intégration Euler
    l[k + 1] = l[k] + dt * l_dot[k]
    I[k + 1] = I[k] + dt * e[k]

# dernier point
e[-1] = l_star - l[-1]
u[-1] = Kp * e[-1] + Ki * I[-1]
l_dot[-1] = beta * u[-1]

# ==========================================
# RESULTATS
# ==========================================
idx_conv = np.where(np.abs(l - l_star) <= tol)[0]
conv_time = time[idx_conv[0]] if len(idx_conv) > 0 else None

u_max = np.max(np.abs(u))
ldot_max = np.max(np.abs(l_dot))

print("===== PARAMETRES =====")
print(f"mu = {mu}")
print(f"phi = {phi_deg} deg")
print(f"alpha = {alpha}")
print(f"beta = {beta:.5f}")
print(f"Ts voulu = {Ts} s")
print(f"zeta = {zeta}")
print(f"wn = {wn:.5f}")
print(f"Kp = {Kp:.3f}")
print(f"Ki = {Ki:.3f}")
print(f"l* = {l_star} m")
print()

print("===== RESULTATS =====")
if conv_time is not None:
    print(f"Temps de convergence ~ {conv_time:.3f} s")
else:
    print("La cible n'est pas atteinte dans la tolérance choisie.")

print(f"Commande maximale |u|_max = {u_max:.3f}")
print(f"Vitesse maximale |l_dot|_max = {ldot_max:.3f} m/s")

# ==========================================
# AFFICHAGE
# ==========================================
plt.figure(figsize=(10, 5))
plt.plot(time, l, label="l(t)")
plt.axhline(l_star, color="red", linestyle="--", label="l*")
plt.xlabel("Temps [s]")
plt.ylabel("Position l(t) [m]")
plt.title("Evolution de la position")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.figure(figsize=(10, 5))
plt.plot(time, e, label="e(t) = l* - l")
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Temps [s]")
plt.ylabel("Erreur e(t) [m]")
plt.title("Evolution de l'erreur")
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
plt.title("Evolution de l'intégrale de l'erreur")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.figure(figsize=(10, 5))
plt.plot(time, l_dot, label="l_dot(t)")
plt.xlabel("Temps [s]")
plt.ylabel("Vitesse quasi-statique [m/s]")
plt.title("Evolution de la vitesse")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()