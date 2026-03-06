# import numpy as np
# from scipy.optimize import root
# import matplotlib.pyplot as plt

# m = 7.0
# a = 0.3
# b = 0.4

# I = (m/12.0) * (a**2 + b**2)                 # inertie CM rectangle
# I_A = (m/3.0)  * (a**2 + b**2)               # si A est un coin: I_A = I + m*((a/2)^2+(b/2)^2) = (m/3)(a^2+b^2)
# mu = 0.3
# T_wall = 6.0

# # Vecteurs dans le repère lié à A (2D)
# rAB = np.array([0.0, b], dtype=float)        # A -> B
# rAO = np.array([a/2, b/2], dtype=float)      # A -> centre (O)

# psi_switch = np.pi/2 + np.pi/4

# def wall_profile_casadi(t):
#     psi_start = np.pi/2
#     psi_end   = np.pi
#     delta_psi = psi_end - psi_start

#     t1 = T_wall * 0.4
#     t2 = T_wall * 0.6

#     acc = delta_psi / (0.5*t1**2 + (t2-t1)*t1 + 0.5*(T_wall - t2)**2)

#     v1 = acc*t1
#     p1 = psi_start + 0.5*acc*t1**2
#     p2 = p1 + v1*(t2 - t1)

#     if t <= t1:
#         psi = psi_start + 0.5*acc*t**2
#         psidot = acc*t
#         psiddot = acc
#     elif t <= t2:
#         psi = p1 + v1*(t - t1)
#         psidot = v1
#         psiddot = 0.0
#     elif t <= T_wall:
#         psi = p2 + v1*(t - t2) - 0.5*acc*(t - t2)**2
#         psidot = v1 - acc*(t - t2)
#         psiddot = -acc
#     else:
#         psi = psi_end
#         psidot = 0.0
#         psiddot = 0.0

#     return float(psi), float(psidot), float(psiddot)

# def rot(theta):
#     c = np.cos(theta); s = np.sin(theta)
#     return np.array([[c, -s],
#                      [s,  c]], dtype=float)

# def rot_prime(theta):
#     c = np.cos(theta); s = np.sin(theta)
#     return np.array([[-s, -c],
#                      [ c, -s]], dtype=float)

# def delta_kinematics(l, ldot, lddot, psi, psidot, psiddot, b, r_vec):
#     eps = 1e-9

#     u_raw = (l / b) * np.sin(psi)
#     u = np.clip(u_raw, -1.0 + 1e-9, 1.0 - 1e-9)

#     one_minus = max(eps, 1 - u * u)
#     k = 1 / np.sqrt(one_minus)
#     k3 = k**3

#     delta = psi - 0.5 * np.pi + np.arcsin(u)

#     u_l   = (1 / b) * np.sin(psi)
#     u_psi = (l / b) * np.cos(psi)

#     delta_l   = k * u_l
#     delta_psi = 1 + k * u_psi

#     u_ll     = 0
#     u_lpsi   = (1 / b) * np.cos(psi)
#     u_psipsi = -(l / b) * np.sin(psi)

#     delta_ll     = k * u_ll     + u * k3 * u_l * u_l
#     delta_lpsi   = k * u_lpsi   + u * k3 * u_l * u_psi
#     delta_psipsi = k * u_psipsi + u * k3 * u_psi * u_psi

#     delta_dot = delta_l * ldot + delta_psi * psidot

#     delta_ddot = (
#         delta_ll * (ldot**2)
#         + 2 * delta_lpsi * ldot * psidot
#         + delta_psipsi * (psidot**2)
#         + delta_l * lddot
#         + delta_psi * psiddot
#     )

#     R  = rot(delta)
#     Rp = rot_prime(delta)

#     a_vec = np.array([lddot, 0.0]) + (Rp @ r_vec) * delta_ddot - (R @ r_vec) * (delta_dot**2)

#     return delta, delta_dot, delta_ddot, a_vec[0], a_vec[1]


# def FxFy_intuitive(psi, l, b):
#     if psi < psi_switch:
#         return -10.0, -1.0
#     if l < b:
#         return -10.0, -10.0
#     return -10.0, -10.0

# def cross2(r, f):
#     r = np.asarray(r, dtype=float).reshape(2,)
#     f = np.asarray(f, dtype=float).reshape(2,)
#     return r[0]*f[1] - r[1]*f[0]

# def equations(u, t, l, ldot):
#     # inconnues
#     fAx, fBx, lddot = u

#     # point dont tu prends l'accélération (ici centre)
#     r_vec = np.array([a/2, b/2], dtype=float)

#     # 1) psi(t)
#     psi, psidot, psiddot = wall_profile_casadi(t)

#     # 2) trajectoire intuitive Fx,Fy
#     Fx, Fy = FxFy_intuitive(psi, l, b)

#     # 3) cinématique
#     delta, delta_dot, delta_ddot, ax, ay = delta_kinematics(
#         l, ldot, lddot, psi, psidot, psiddot, b, r_vec
#     )

#     # 4) frottement Coulomb en glissement
#     fAy = -mu * fAx
#     fBy = -mu * fBx

#     fA = np.array([fAx, fAy], dtype=float)
#     fB = np.array([fBx, fBy], dtype=float)

#     fc = np.array([Fx, Fy], dtype=float)

#     # rotation
#     R = rot(delta)

#     # moments autour de A (vecteurs dans le monde)
#     rAO_world = R @ rAO
#     rAB_world = R @ rAB

#     # 5) équations
#     eq1 = (fAx + fBx + Fx) - m*ax
#     eq2 = (fAy + fBy + Fy) - m*ay
#     eq3 = (cross2(rAO_world, fc) + cross2(rAB_world, fB)) - I_A*delta_ddot

#     return [eq1, eq2, eq3]

# # ----------------------------
# # Intégration Euler
# # ----------------------------
# dt = 0.01
# T = np.arange(0.0, T_wall + 1e-12, dt)

# l = 0.00
# ldot = 0.0

# sol = []
# residuals = []
# traj = []  # (t,l,ldot)

# u_guess = np.array([5, 5, 0.0], dtype=float)

# for t in T:
#     if l >= b - 1e-6:
#         break

#     res = root(equations, u_guess, args=(float(t), float(l), float(ldot)), method="hybr")
#     if not res.success:
#         raise RuntimeError(f"Solver failed at t={t:.4f}: {res.message}")

#     fAx, fBx, lddot = res.x

#     sol.append(res.x.copy())
#     residuals.append(np.linalg.norm(equations(res.x, float(t), float(l), float(ldot))))
#     traj.append((t, l, ldot))

#     # Euler explicite
#     ldot_next = ldot + lddot * dt
#     l_next    = l    + ldot_next * dt

#     # condition de fin de course
#     if l_next >= b:
#         l = b
#         ldot = 0.0
#         break
#     else:
#         l = l_next
#         ldot = ldot_next

#     u_guess = res.x

# sol = np.array(sol)
# residuals = np.array(residuals)
# traj = np.array(traj)

# print("finished at t =", traj[-1,0] if len(traj) > 0 else None)
# print("final l =", l, "final ldot =", ldot)
# print("max residual:", residuals.max() if len(residuals) > 0 else None)
# print("min fAx, fBx:", sol[:,0].min(), sol[:,1].min())


# t_arr  = traj[:, 0]
# fAx    = sol[:, 0]
# fBx    = sol[:, 1]


# plt.figure()
# plt.plot(t_arr, fAx, label="fAx")
# plt.plot(t_arr, fBx, label="fBx")
# plt.xlabel("Time [s]")
# plt.ylabel("Force [N]")
# plt.title("fAx and fBx vs time")
# plt.legend()
# plt.show()

# fAy = -mu * fAx
# fBy = -mu * fBx

# plt.figure()
# plt.plot(t_arr, fAy, label="fAy")
# plt.plot(t_arr, fBy, label="fBy")
# plt.xlabel("Time [s]")
# plt.ylabel("Force [N]")
# plt.title("fAy and fBy vs time")
# plt.legend()
# plt.show()


import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt

m = 7.0
a = 0.3
b = 0.4

I = (m/12.0) * (a**2 + b**2)                 # inertie CM rectangle
I_A = (m/3.0)  * (a**2 + b**2)               # si A est un coin: I_A = I + m*((a/2)^2+(b/2)^2) = (m/3)(a^2+b^2)
mu = 0.3
T_wall = 6.0

# Vecteurs dans le repère lié à A (2D)
rAB = np.array([0.0, b], dtype=float)        # A -> B
rAO = np.array([a/2, b/2], dtype=float)      # A -> centre (O)

psi_switch = np.pi/2 + np.pi/4


def rot(theta):
    c = np.cos(theta); s = np.sin(theta)
    return np.array([[c, -s],
                     [s,  c]], dtype=float)

def rot_prime(theta):
    c = np.cos(theta); s = np.sin(theta)
    return np.array([[-s, -c],
                     [ c, -s]], dtype=float)
    
def cross2(r, f):
    r = np.asarray(r, dtype=float).reshape(2,)
    f = np.asarray(f, dtype=float).reshape(2,)
    return r[0]*f[1] - r[1]*f[0]

def delta_kinematics(l, ldot, lddot, psi, psidot, psiddot, b, r_vec):
    eps = 1e-9

    u_raw = (l / b) * np.sin(psi)
    u = np.clip(u_raw, -1.0 + 1e-9, 1.0 - 1e-9)

    one_minus = max(eps, 1 - u * u)
    k = 1 / np.sqrt(one_minus)
    k3 = k**3

    delta = psi - 0.5 * np.pi + np.arcsin(u)

    u_l   = (1 / b) * np.sin(psi)
    u_psi = (l / b) * np.cos(psi)

    delta_l   = k * u_l
    delta_psi = 1 + k * u_psi

    u_ll     = 0
    u_lpsi   = (1 / b) * np.cos(psi)
    u_psipsi = -(l / b) * np.sin(psi)

    delta_ll     = k * u_ll     + u * k3 * u_l * u_l
    delta_lpsi   = k * u_lpsi   + u * k3 * u_l * u_psi
    delta_psipsi = k * u_psipsi + u * k3 * u_psi * u_psi

    delta_dot = delta_l * ldot + delta_psi * psidot

    delta_ddot = (
        delta_ll * (ldot**2)
        + 2 * delta_lpsi * ldot * psidot
        + delta_psipsi * (psidot**2)
        + delta_l * lddot
        + delta_psi * psiddot
    )

    R  = rot(delta)
    Rp = rot_prime(delta)

    a_vec = np.array([lddot, 0.0]) + (Rp @ r_vec) * delta_ddot - (R @ r_vec) * (delta_dot**2)

    return delta, delta_dot, delta_ddot, a_vec[0], a_vec[1]

from scipy.optimize import least_squares

def solve_instant(l, ldot, psi, psidot, psiddot, Fx, Fy, u_guess=None):
    if u_guess is None:
        u_guess = np.array([10.0, 10.0, 0.0])  # [fAy, fBn, lddot]

    def eqs(u):
        fAy, fBn, lddot = u

        tB = np.array([np.cos(psi), np.sin(psi)])
        nB = np.array([np.sin(psi), -np.cos(psi)])

        fAx = -mu * fAy
        fBt = -mu * fBn

        fA = np.array([fAx, fAy], dtype=float)
        fB = fBn * nB + fBt * tB
        fc = np.array([Fx, Fy], dtype=float)

        r_vec = np.array([a/2, b/2], dtype=float)

        delta, delta_dot, delta_ddot, ax, ay = delta_kinematics(
            l, ldot, lddot, psi, psidot, psiddot, b, r_vec
        )

        R = rot(delta)

        rAB = R @ np.array([0.0, b], dtype=float)
        rAO = R @ np.array([a/2, b/2], dtype=float)

        eq1 = (fA[0] + fB[0] + Fx) - m * ax
        eq2 = (fA[1] + fB[1] + Fy) - m * ay
        eq3 = cross2(rAO, fc) + cross2(rAB, fB) - I_A * delta_ddot

        return np.array([eq1, eq2, eq3], dtype=float)

    res = least_squares(
        eqs,
        x0=u_guess,
        bounds=([0.0, 0.0, -np.inf], [np.inf, np.inf, np.inf])
    )
    
    print("res.fun =", res.fun)
    print("norme =", np.linalg.norm(res.fun))
    return res



psi = 3*np.pi/4
psidot = 0.2
psiddot = 0.0
l = 0.2
ldot = 0.1
Fx = -5.0
Fy = -2.0

res = solve_instant(l, ldot, psi, psidot, psiddot, Fx, Fy)

if res.success:
    fAy, fBx, lddot = res.x
    print("solution:", res.x)
else:
    print("échec:", res.message)
    
