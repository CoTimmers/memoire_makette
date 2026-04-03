import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# Paramètres physiques
# ==========================================
theta_deg = 34
theta = np.deg2rad(theta_deg)
mu = 0.2

# facteur utile dans la dynamique
geom_factor = np.cos(theta) - mu * np.sin(theta)

print(f"theta = {theta_deg} deg")
print(f"mu = {mu}")
print(f"geom_factor = {geom_factor:.4f}")

# ==========================================
# Paramètres du contrôleur FIXES
# ==========================================
Kp = 5.0
Ki = 3.0

# ==========================================
# Paramètres de simulation
# ==========================================
dt = 0.001
t_final = 6.0
time = np.arange(0, t_final + dt, dt)

# condition initiale
l0 = 0.20   # distance initiale [m]
I0 = 0.0

# valeurs de alpha à tester
alpha_values = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]

# ==========================================
# Stockage des résultats
# ==========================================
results = []

for alpha in alpha_values:
    l = np.zeros_like(time)
    I = np.zeros_like(time)
    u = np.zeros_like(time)

    l[0] = l0
    I[0] = I0

    for k in range(len(time) - 1):
        # commande PI
        u[k] = Kp * l[k] + Ki * I[k]

        # butée finale
        if l[k] <= 0:
            l[k] = 0.0
            u[k] = 0.0
            l[k + 1] = 0.0
            I[k + 1] = I[k]
            continue

        # dynamique
        # l_dot = - alpha * u * (cos(theta) - mu sin(theta))
        l_dot = -alpha * u[k] * geom_factor

        # intégrateur
        I_dot = l[k]

        # intégration Euler
        l[k + 1] = l[k] + dt * l_dot
        I[k + 1] = I[k] + dt * I_dot

        # éviter l < 0
        if l[k + 1] < 0:
            l[k + 1] = 0.0

    # dernière valeur de commande
    if l[-1] > 0:
        u[-1] = Kp * l[-1] + Ki * I[-1]
    else:
        u[-1] = 0.0

    # estimation du temps d'arrivée
    idx_zero = np.where(l <= 1e-4)[0]
    if len(idx_zero) > 0:
        arrival_time = time[idx_zero[0]]
    else:
        arrival_time = None

    results.append({
        "alpha": alpha,
        "l": l,
        "I": I,
        "u": u,
        "arrival_time": arrival_time
    })

# ==========================================
# Graphe 1 : distance restante
# ==========================================
plt.figure(figsize=(10, 5))
for res in results:
    label = f'alpha={res["alpha"]}'
    if res["arrival_time"] is not None:
        label += f', t_arr={res["arrival_time"]:.2f}s'
    plt.plot(time, res["l"], label=label)

plt.xlabel("Temps [s]")
plt.ylabel("Distance restante l(t) [m]")
plt.title("Evolution de la distance restante pour différents alpha")
plt.grid(True)
plt.legend()
plt.tight_layout()

# ==========================================
# Graphe 2 : commande
# ==========================================
plt.figure(figsize=(10, 5))
for res in results:
    plt.plot(time, res["u"], label=f'alpha={res["alpha"]}')

plt.xlabel("Temps [s]")
plt.ylabel("Commande u(t)")
plt.title("Evolution de la commande pour différents alpha")
plt.grid(True)
plt.legend()
plt.tight_layout()

# ==========================================
# Graphe 3 : état intégral
# ==========================================
plt.figure(figsize=(10, 5))
for res in results:
    plt.plot(time, res["I"], label=f'alpha={res["alpha"]}')

plt.xlabel("Temps [s]")
plt.ylabel("Etat intégral I(t)")
plt.title("Evolution de l'intégrale de l'erreur")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()

# ==========================================
# Résumé console
# ==========================================
print("\nRésumé :")
print(f"Kp = {Kp}, Ki = {Ki}\n")

for res in results:
    if res["arrival_time"] is not None:
        print(f"alpha = {res['alpha']:.2f} -> arrivée vers t = {res['arrival_time']:.3f} s")
    else:
        print(f"alpha = {res['alpha']:.2f} -> pas arrivé à la cible pendant la simulation")