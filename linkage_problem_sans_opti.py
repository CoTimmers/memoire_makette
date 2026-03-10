import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt

m = 7.0
a = 0.3
b = 0.4

I = (m/12.0) * (a**2 + b**2)                
I_A = (m/3.0)  * (a**2 + b**2)               
mu = 0.3
T_wall = 6.0


rAB = np.array([0.0, b], dtype=float)        
rAO = np.array([a/2, b/2], dtype=float)     

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
    
