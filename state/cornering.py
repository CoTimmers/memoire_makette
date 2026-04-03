import numpy as np
import matplotlib.pyplot as plt

# =========================
# Paramètres physiques
# =========================
theta_deg = 34
theta = np.deg2rad(theta_deg)
mu = 0.2

# facteur géométrie + friction
geom_factor = np.cos(theta) - mu * np.sin(theta)

# =========================
# Spécifications désirées
# =========================
Ts = 3.0          # temps de convergence souhaité [s]
zeta = 1.0        # amortissement critique
wn = 4 / (zeta * Ts)

# =========================
# Simulation
# =========================
dt = 0.001
t_final = 6.0
time = np.arange(0, t_final + dt, dt)

# condition initiale
l0 = 0.20   # distance initiale [m] par exemple

# différentes valeurs de alpha à tester
alpha_values = [0.2, 0.5, 1.0, 2.0, 5.0]

# stockage pour affichage
results = []

for alpha in alpha_values:
    beta = alpha * geom_factor

    # gains calculés à partir de la synthèse
    Kp = 2 * zeta * wn / beta
    Ki = wn**2 / beta

    # états
    l = np.zeros_like(time)
    I = np.zeros_like(time)
    u = np.zeros_like(time)

    l[0] = l0
    I[0] = 0.0

    for k in range(len(time) - 1):
        # commande PI
        u[k] = Kp * l[k] + Ki * I[k]

        # si le bac est arrivé, on bloque tout
        if l[k] <= 0:
            l[k] = 0
            u[k] = 0
            I[k+1] = I[k]
            l[k+1] = 0
            continue

        # dynamique :
        # l_dot = - alpha * u * (cos(theta) - mu sin(theta))
        l_dot = -alpha * u[k] * geom_factor

        # intégrateur
        I_dot = l[k]

        # Euler explicite
        l[k+1] = l[k] + dt * l_dot
        I[k+1] = I[k] + dt * I_dot

        # butée physique finale
        if l[k+1] < 0:
            l[k+1] = 0

    u[-1] = Kp * l[-1] + Ki * I[-1] if l[-1] > 0 else 0

    results.append({
        "alpha": alpha,
        "beta": beta,
        "Kp": Kp,
        "Ki": Ki,
        "l": l,
        "I": I,
        "u": u
    })

# =========================
# Affichage des résultats
# =========================
plt.figure(figsize=(10, 5))
for res in results:
    plt.plot(time, res["l"], label=f'alpha={res["alpha"]}, Kp={res["Kp"]:.2f}, Ki={res["Ki"]:.2f}')
plt.xlabel("Temps [s]")
plt.ylabel("l(t) [m]")
plt.title("Evolution de la distance restante l(t)")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.figure(figsize=(10, 5))
for res in results:
    plt.plot(time, res["u"], label=f'alpha={res["alpha"]}')
plt.xlabel("Temps [s]")
plt.ylabel("u(t) [N] (ou unité équivalente)")
plt.title("Evolution de la commande u(t)")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()

# =========================
# Affichage console
# =========================
print(f"theta = {theta_deg} deg")
print(f"mu = {mu}")
print(f"geom_factor = cos(theta) - mu sin(theta) = {geom_factor:.4f}")
print(f"wn = {wn:.4f} rad/s\n")

for res in results:
    print(f"alpha = {res['alpha']:.3f} -> beta = {res['beta']:.4f}, "
          f"Kp = {res['Kp']:.4f}, Ki = {res['Ki']:.4f}")