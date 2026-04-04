import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# PARAMETRES PHYSIQUES
# ==========================================
m = 7.0
beta = 1.0 / m

# ==========================================
# CONDITIONS INITIALES
# ==========================================
d0 = 0.20
v0 = 0.0
I0 = 0.0

# ==========================================
# PARAMETRES DE SIMULATION
# ==========================================
dt = 0.001
t_final = 8.0
time = np.arange(0, t_final + dt, dt)

# ==========================================
# PARAMETRES PI
# ==========================================
Kp = 4.0
Ki = 1.0

# ==========================================
# PARAMETRE FORCE CONSTANTE
# ==========================================
F0 = 0.85

# ==========================================
# FONCTION DE SIMULATION PI
# ==========================================
def simulate_pi(Kp, Ki, beta, d0, v0, I0, time, dt):
    d = np.zeros_like(time)
    v = np.zeros_like(time)
    I = np.zeros_like(time)
    u = np.zeros_like(time)
    a = np.zeros_like(time)

    d[0] = d0
    v[0] = v0
    I[0] = I0

    for k in range(len(time) - 1):
        if d[k] <= 0:
            d[k] = 0.0
            v[k] = 0.0
            u[k] = 0.0
            a[k] = 0.0

            d[k + 1] = 0.0
            v[k + 1] = 0.0
            I[k + 1] = I[k]
            continue

        u[k] = Kp * d[k] + Ki * I[k]
        a[k] = -beta * u[k]

        v[k + 1] = v[k] + dt * a[k]
        d[k + 1] = d[k] + dt * v[k]
        I[k + 1] = I[k] + dt * d[k]

        if d[k + 1] < 0:
            d[k + 1] = 0.0
            v[k + 1] = 0.0

    if d[-1] > 0:
        u[-1] = Kp * d[-1] + Ki * I[-1]
        a[-1] = -beta * u[-1]
    else:
        d[-1] = 0.0
        v[-1] = 0.0
        u[-1] = 0.0
        a[-1] = 0.0

    return d, v, I, u, a

# ==========================================
# FONCTION DE SIMULATION FORCE CONSTANTE
# ==========================================
def simulate_constant_force(F0, beta, d0, v0, time, dt):
    d = np.zeros_like(time)
    v = np.zeros_like(time)
    u = np.zeros_like(time)
    a = np.zeros_like(time)

    d[0] = d0
    v[0] = v0

    for k in range(len(time) - 1):
        if d[k] <= 0:
            d[k] = 0.0
            v[k] = 0.0
            u[k] = 0.0
            a[k] = 0.0

            d[k + 1] = 0.0
            v[k + 1] = 0.0
            continue

        u[k] = F0
        a[k] = -beta * u[k]

        v[k + 1] = v[k] + dt * a[k]
        d[k + 1] = d[k] + dt * v[k]

        if d[k + 1] < 0:
            d[k + 1] = 0.0
            v[k + 1] = 0.0

    if d[-1] > 0:
        u[-1] = F0
        a[-1] = -beta * u[-1]
    else:
        d[-1] = 0.0
        v[-1] = 0.0
        u[-1] = 0.0
        a[-1] = 0.0

    return d, v, u, a

# ==========================================
# SIMULATIONS
# ==========================================
d_pi, v_pi, I_pi, u_pi, a_pi = simulate_pi(Kp, Ki, beta, d0, v0, I0, time, dt)
d_cst, v_cst, u_cst, a_cst = simulate_constant_force(F0, beta, d0, v0, time, dt)

# ==========================================
# EXTRACTION DES INDICATEURS
# ==========================================
def get_arrival_time(d, time, threshold=1e-4):
    idx = np.where(d <= threshold)[0]
    return time[idx[0]] if len(idx) > 0 else None

arrival_pi = get_arrival_time(d_pi, time)
arrival_cst = get_arrival_time(d_cst, time)

print("===== COMPARAISON =====")
print("\n--- PI ---")
print(f"Kp = {Kp}, Ki = {Ki}")
print(f"Temps d'arrivée = {arrival_pi:.3f} s" if arrival_pi is not None else "Pas d'arrivée")
print(f"u_max = {np.max(u_pi):.3f}")
print(f"|v|_max = {np.max(np.abs(v_pi)):.3f} m/s")
print(f"|a|_max = {np.max(np.abs(a_pi)):.3f} m/s²")

print("\n--- Force constante ---")
print(f"F0 = {F0}")
print(f"Temps d'arrivée = {arrival_cst:.3f} s" if arrival_cst is not None else "Pas d'arrivée")
print(f"u_max = {np.max(u_cst):.3f}")
print(f"|v|_max = {np.max(np.abs(v_cst)):.3f} m/s")
print(f"|a|_max = {np.max(np.abs(a_cst)):.3f} m/s²")

# ==========================================
# GRAPHES
# ==========================================
plt.figure(figsize=(10, 5))
plt.plot(time, d_pi, label="PI")
plt.plot(time, d_cst, label="Force constante", linestyle="--")
plt.xlabel("Temps [s]")
plt.ylabel("Distance d(t) [m]")
plt.title("Comparaison de la distance")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.figure(figsize=(10, 5))
plt.plot(time, v_pi, label="PI")
plt.plot(time, v_cst, label="Force constante", linestyle="--")
plt.xlabel("Temps [s]")
plt.ylabel("Vitesse v(t) [m/s]")
plt.title("Comparaison de la vitesse")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.figure(figsize=(10, 5))
plt.plot(time, u_pi, label="PI")
plt.plot(time, u_cst, label="Force constante", linestyle="--")
plt.xlabel("Temps [s]")
plt.ylabel("Commande u(t)")
plt.title("Comparaison de la commande")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.figure(figsize=(10, 5))
plt.plot(time, a_pi, label="PI")
plt.plot(time, a_cst, label="Force constante", linestyle="--")
plt.xlabel("Temps [s]")
plt.ylabel("Accélération a(t) [m/s²]")
plt.title("Comparaison de l'accélération")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()