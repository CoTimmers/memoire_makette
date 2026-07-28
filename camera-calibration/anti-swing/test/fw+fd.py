# -*- coding: utf-8 -*-
"""
Architecture a deux couches : input shaping (feedforward) + retour d'etat (feedback).
Compare, sur le modele non lineaire (l = 3 m) :
   A. feedback seul
   B. shaper + commande de position seule (pas de termes anti-ballant)
   C. shaper + feedback complet
Une perturbation (choc sur le bac) a t = 14 s montre la limite du feedforward.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Physique ---
g, l = 9.81, 3.0
zeta_n = 0.00228
omega = np.sqrt(g / l)

# --- Controleur ---
K_full = np.array([1.000, 2.774, -15.19, -5.272])
K_pos = np.array([1.000, 2.774, 0.0, 0.0])
A_MAX = 1.0

# --- Shaper ZV : 2 impulsions espacees d'une demi-periode ---
Kz = np.exp(-zeta_n * np.pi / np.sqrt(1 - zeta_n**2))
A1, A2 = 1 / (1 + Kz), Kz / (1 + Kz)
Td = np.pi / (omega * np.sqrt(1 - zeta_n**2))
print(f"Shaper ZV : A1 = {A1:.4f} a t = 0, A2 = {A2:.4f} a t = {Td:.3f} s")

# --- Profil de reference trapezoidal (position et vitesse) ---
X_REF, V_MAX, A_REF, T0 = 1.0, 0.25, 0.5, 0.5
t_acc = V_MAX / A_REF
d_acc = 0.5 * A_REF * t_acc**2
t_cst = (X_REF - 2 * d_acc) / V_MAX


def trapeze(tt):
    """Position, vitesse et acceleration du profil de base."""
    tau = tt - T0
    if tau <= 0:
        return 0.0, 0.0, 0.0
    if tau <= t_acc:
        return 0.5 * A_REF * tau**2, A_REF * tau, A_REF
    if tau <= t_acc + t_cst:
        return d_acc + V_MAX * (tau - t_acc), V_MAX, 0.0
    if tau <= 2 * t_acc + t_cst:
        tr = tau - t_acc - t_cst
        return (d_acc + V_MAX * t_cst + V_MAX * tr - 0.5 * A_REF * tr**2,
                V_MAX - A_REF * tr, -A_REF)
    return X_REF, 0.0, 0.0


def reference(tt, shaped):
    """Consigne (position, vitesse, acceleration), avec ou sans mise en forme."""
    if not shaped:
        return trapeze(tt)
    p1, v1, a1 = trapeze(tt)
    p2, v2, a2 = trapeze(tt - Td)
    return A1 * p1 + A2 * p2, A1 * v1 + A2 * v2, A1 * a1 + A2 * a2


# --- Perturbation : choc sur le bac a t = 14 s ---
def perturbation(tt):
    return 0.35 if 14.0 <= tt <= 14.3 else 0.0     # rad/s^2 sur theta


def dynamique(z, a, tt):
    x, xd, th, thd = z
    thdd = (-a * np.cos(th) - g * np.sin(th)) / l - 2 * zeta_n * omega * thd + perturbation(tt)
    return np.array([xd, a, thd, thdd])


dt, t_end = 0.002, 25.0
t = np.arange(0, t_end, dt)


def simuler(K, shaped):
    z = np.zeros(4)
    Z = np.zeros((len(t), 4))
    U = np.zeros(len(t))
    for i, ti in enumerate(t):
        p_ref, v_ref, a_ref = reference(ti, shaped)
        # feedforward (acceleration de reference) + feedback (ecart d'etat)
        u = a_ref - float(K @ (z - np.array([p_ref, v_ref, 0.0, 0.0])))
        u = float(np.clip(u, -A_MAX, A_MAX))
        k1 = dynamique(z, u, ti)
        k2 = dynamique(z + dt/2 * k1, u, ti)
        k3 = dynamique(z + dt/2 * k2, u, ti)
        k4 = dynamique(z + dt * k3, u, ti)
        z = z + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        Z[i], U[i] = z, u
    return Z, U


cas = {
    "A. Feedback only":        (simuler(K_full, False), "0.55", "--"),
    "B. Shaping only":         (simuler(K_pos,  True),  "0.35", ":"),
    "C. Shaping + feedback":   (simuler(K_full, True),  "k",    "-"),
}

print("\n--- Metriques ---")
for nom, ((Z, U), _, _) in cas.items():
    th = np.degrees(Z[:, 2])
    transfert = th[t < 13]
    apres_choc = th[t > 18]
    print(f"{nom:24s} pic pendant transfert {np.max(np.abs(transfert)):5.2f} deg | "
          f"residuel 4 s apres le choc {np.max(np.abs(apres_choc)):5.2f} deg")

fig, ax = plt.subplots(2, 1, figsize=(7.2, 5.4), sharex=True)
for nom, ((Z, U), col, ls) in cas.items():
    ax[0].plot(t, Z[:, 0], color=col, ls=ls, lw=1.3, label=nom)
    ax[1].plot(t, np.degrees(Z[:, 2]), color=col, ls=ls, lw=1.3)
ax[0].axhline(X_REF, color="0.8", ls=":", lw=0.8)
ax[0].set_ylabel(r"$x$ [m]")
ax[0].legend(fontsize=8, loc="lower right", frameon=False)
ax[0].grid(alpha=0.25)
ax[1].axvspan(14, 14.3, color="0.85")
ax[1].annotate("disturbance", (14.5, 2.4), fontsize=8, color="0.4")
ax[1].set_ylabel(r"$\theta$ [deg]")
ax[1].set_xlabel("time [s]")
ax[1].grid(alpha=0.25)
plt.tight_layout()
plt.savefig("shaper_feedback.pdf")
plt.savefig("shaper_feedback.png", dpi=140)
print("\nFigures ecrites : shaper_feedback.pdf / .png")