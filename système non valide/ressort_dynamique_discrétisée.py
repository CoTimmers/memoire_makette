"""Simulation avant — système mur 2 dynamique (ressort de torsion).
   Pas d'optimiseur : commande intuitive manuelle.
   État : (l, ldot, psi, psidot)
   Contrôle : (fx, fy) appliqué au COM du bac
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ══════════════════════════════════════════════════════════════════════
# PARAMÈTRES PHYSIQUES  (identiques à ocp_corrige.py)
# ══════════════════════════════════════════════════════════════════════
m        = 7.0
a        = 0.3
b        = 0.4
I_A      = (m / 3.0) * (a**2 + b**2)
mu       = 0.3

k_spring = 5.0        # raideur ressort de torsion (N·m/rad)
J_mur    = 4.0        # moment d'inertie mur 2 (kg·m²)
psi_rest = np.pi / 2  # position repos du mur (90°)

rAB_body = np.array([0.0, b])
rAO_body = np.array([a/2, b/2])

T_sim = 10.0

# ══════════════════════════════════════════════════════════════════════
# UTILITAIRES
# ══════════════════════════════════════════════════════════════════════
def rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])

def rot_prime(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[-s, -c], [c, -s]])

def cross2(r, f):
    return r[0]*f[1] - r[1]*f[0]

# ══════════════════════════════════════════════════════════════════════
# DYNAMIQUE CONTINUE 4×4
#   Inconnues z = [lddot, fyA, fBn, psiddot]
# ══════════════════════════════════════════════════════════════════════
def dynamics_np(l, ldot, psi, psidot, fx, fy):
    eps = 1e-9

    # ── Cinématique de δ ──────────────────────────────────────────────
    u   = np.clip((l / b) * np.sin(psi), -1 + eps, 1 - eps)
    k   = 1.0 / np.sqrt(max(eps, 1.0 - u*u))
    k3  = k**3

    delta = psi - np.pi/2 + np.arcsin(u)

    u_l      = np.sin(psi) / b
    u_psi    = (l / b) * np.cos(psi)
    u_lpsi   = np.cos(psi) / b
    u_psipsi = -(l / b) * np.sin(psi)

    delta_l      = k * u_l
    delta_psi    = 1.0 + k * u_psi
    delta_ll     = u * k3 * u_l * u_l
    delta_lpsi   = k * u_lpsi   + u * k3 * u_l   * u_psi
    delta_psipsi = k * u_psipsi + u * k3 * u_psi * u_psi

    delta_dot = delta_l * ldot + delta_psi * psidot
    NL = (delta_ll      * ldot**2
        + 2*delta_lpsi  * ldot * psidot
        + delta_psipsi  * psidot**2)

    # ── Matrices de rotation ─────────────────────────────────────────
    R  = rot(delta)
    Rp = rot_prime(delta)

    rAO_w = R  @ rAO_body
    rAB_w = R  @ rAB_body
    Rpr   = Rp @ rAO_body
    Rr    = R  @ rAO_body

    # ── Termes non-linéaires ─────────────────────────────────────────
    NLx = Rpr[0] * NL - Rr[0] * delta_dot**2
    NLy = Rpr[1] * NL - Rr[1] * delta_dot**2

    # ── Contact en B ─────────────────────────────────────────────────
    tB    = np.array([ np.cos(psi),  np.sin(psi)])
    nB    = np.array([ np.sin(psi), -np.cos(psi)])
    fBdir = nB - mu * tB

    xB  = l + rAB_w[0]
    yB  = rAB_w[1]
    rOB = np.array([xB, yB])

    # ── Système 4×4 ──────────────────────────────────────────────────
    M_mat = np.array([
        [m*(1 + Rpr[0]*delta_l),  mu,  -fBdir[0],             -m*Rpr[0]*delta_psi ],
        [m*Rpr[1]*delta_l,        -1.0, -fBdir[1],             -m*Rpr[1]*delta_psi ],
        [I_A*delta_l,              0.0, -cross2(rAB_w, fBdir), -I_A*delta_psi      ],
        [0.0,                      0.0,  cross2(rOB,   fBdir),  J_mur              ]
    ])

    f_vec = np.array([fx, fy])
    rhs = np.array([
        fx - m * NLx,
        fy - m * NLy,
        cross2(rAO_w, f_vec) - I_A * NL,
        -k_spring * (psi - psi_rest)
    ])

    try:
        z = np.linalg.solve(M_mat, rhs)
    except np.linalg.LinAlgError:
        z = np.zeros(4)

    lddot    = z[0]
    fyA      = z[1]
    fBn      = z[2]
    psiddot  = z[3]

    # Contrainte unilatérale en A : mur 1 ne peut pas tirer (fyA >= 0)
    if fyA < 0:
        # A décolle → on re-résout un système 3×3 sans fyA
        # On garde les équations : Newton-x bac, moment bac, moment mur 2
        M3 = np.array([
            [m*(1 + Rpr[0]*delta_l),  -fBdir[0],              -m*Rpr[0]*delta_psi ],
            [I_A*delta_l,             -cross2(rAB_w, fBdir),  -I_A*delta_psi      ],
            [0.0,                      cross2(rOB,   fBdir),   J_mur              ]
        ])
        rhs3 = np.array([
            fx - m * NLx,
            cross2(rAO_w, f_vec) - I_A * NL,
            -k_spring * (psi - psi_rest)
        ])
        try:
            z3 = np.linalg.solve(M3, rhs3)
        except np.linalg.LinAlgError:
            z3 = np.zeros(3)
        lddot   = z3[0]
        fBn     = z3[1]
        psiddot = z3[2]
        fyA     = 0.0

    fxB = fBn * fBdir[0]
    fyB = fBn * fBdir[1]
    tau_spring = -k_spring * (psi - psi_rest)

    return lddot, psiddot, fyA, fBn, fxB, fyB, tau_spring, delta, delta_dot

# ══════════════════════════════════════════════════════════════════════
# INTÉGRATION RK4  — état = [l, ldot, psi, psidot]
# ══════════════════════════════════════════════════════════════════════
def f_ode(l, ldot, psi, psidot, fx, fy):
    lddot, psiddot, *_ = dynamics_np(l, ldot, psi, psidot, fx, fy)
    return ldot, lddot, psidot, psiddot

def rk4_step(l, ldot, psi, psidot, fx, fy, h):
    d1l, d1ld, d1p, d1pd = f_ode(l,              ldot,              psi,              psidot,              fx, fy)
    d2l, d2ld, d2p, d2pd = f_ode(l+h/2*d1l,      ldot+h/2*d1ld,    psi+h/2*d1p,      psidot+h/2*d1pd,     fx, fy)
    d3l, d3ld, d3p, d3pd = f_ode(l+h/2*d2l,      ldot+h/2*d2ld,    psi+h/2*d2p,      psidot+h/2*d2pd,     fx, fy)
    d4l, d4ld, d4p, d4pd = f_ode(l+h*d3l,        ldot+h*d3ld,      psi+h*d3p,        psidot+h*d3pd,       fx, fy)

    l_new      = l      + (h/6) * (d1l  + 2*d2l  + 2*d3l  + d4l )
    ldot_new   = ldot   + (h/6) * (d1ld + 2*d2ld + 2*d3ld + d4ld)
    psi_new    = psi    + (h/6) * (d1p  + 2*d2p  + 2*d3p  + d4p )
    psidot_new = psidot + (h/6) * (d1pd + 2*d2pd + 2*d3pd + d4pd)
    return l_new, ldot_new, psi_new, psidot_new

# ══════════════════════════════════════════════════════════════════════
# COMMANDE INTUITIVE
#   Phase 1 (psi < PSI_SWITCH) : pousser contre mur 2 pour l'ouvrir
#   Phase 2 (psi >= PSI_SWITCH) : pousser le bac le long de mur 1
# ══════════════════════════════════════════════════════════════════════
# PSI_SWITCH = 127 * np.pi / 180   # ~2.22 rad
PSI_SWITCH = 140 * np.pi / 180   # ~2.22 rad
F_PUSH     = 15.0                 # amplitude de la force (N)

def get_command(psi):
    if psi < PSI_SWITCH:
        # Pousser dans la direction normale du mur 2 (−nB) pour l'ouvrir
        nB = np.array([np.sin(psi), -np.cos(psi)])
        return -F_PUSH * nB[0], -F_PUSH * nB[1]
    else:
        
        return F_PUSH*np.sqrt(2)/2, -F_PUSH * np.sqrt(2)/2

# ══════════════════════════════════════════════════════════════════════
# SIMULATION
# ══════════════════════════════════════════════════════════════════════
dt  = 0.002
N   = int(T_sim / dt)

l      = 0.0
ldot   = 0.0
psi    = np.pi / 2   # mur au repos : 90°
psidot = 0.0

hist = {k: [] for k in ['t', 'l', 'ldot', 'psi', 'psidot', 'delta',
                         'fx', 'fy', 'xA', 'yA', 'xB', 'yB', 'xO', 'yO',
                         'fya', 'fxb', 'fyb', 'tau_spring']}

for k in range(N + 1):
    t = k * dt

    # Géométrie
    eps   = 1e-9
    u_i   = np.clip((l / b) * np.sin(psi), -1 + eps, 1 - eps)
    delta = psi - np.pi/2 + np.arcsin(u_i)
    R_i   = rot(delta)
    xA, yA = l, 0.0
    rAO   = R_i @ rAO_body
    rAB   = R_i @ rAB_body
    xO, yO = xA + rAO[0], yA + rAO[1]
    xB, yB = xA + rAB[0], yA + rAB[1]

    # Commande
    fx, fy = get_command(psi)

    # Réactions internes
    _, _, fyA_i, _, fxB_i, fyB_i, tau_i, _, _ = dynamics_np(l, ldot, psi, psidot, fx, fy)

    # Enregistrement
    for key, val in zip(hist.keys(),
                        [t, l, ldot, psi, psidot, delta,
                         fx, fy, xA, yA, xB, yB, xO, yO,
                         fyA_i, fxB_i, fyB_i, tau_i]):
        hist[key].append(val)

    # Intégration RK4
    l_new, ldot_new, psi_new, psidot_new = rk4_step(l, ldot, psi, psidot, fx, fy, dt)

    # Clamp l ∈ [0, b]
    l_new = np.clip(l_new, 0.0, b)
    if l_new <= 0.0 and ldot_new < 0: ldot_new = 0.0
    if l_new >= b   and ldot_new > 0: ldot_new = 0.0

    # Clamp psi ∈ [π/2, π]
    psi_new = np.clip(psi_new, np.pi/2, np.pi)
    if psi_new <= np.pi/2 and psidot_new < 0: psidot_new = 0.0
    if psi_new >= np.pi   and psidot_new > 0: psidot_new = 0.0

    l, ldot, psi, psidot = l_new, ldot_new, psi_new, psidot_new

# Instant du switch
PSW_T = next((hist['t'][i] for i, p in enumerate(hist['psi'])
              if p >= PSI_SWITCH), None)

print(f"Switch at    t = {PSW_T:.3f} s  (psi = 127 deg)")
print(f"l at switch    = {hist['l'][int(PSW_T/dt)]:.4f} m")
print(f"l final        = {hist['l'][-1]:.4f} m  (cible = {b} m)")
print(f"ldot final     = {hist['ldot'][-1]:.4f} m/s")
print(f"psi final      = {np.degrees(hist['psi'][-1]):.2f} deg")
print(f"Max |ldot|     = {max(abs(v) for v in hist['ldot']):.4f} m/s")

# ══════════════════════════════════════════════════════════════════════
# PLOTS STATIQUES
# ══════════════════════════════════════════════════════════════════════
plots = [
    ('l(t)',
     hist['t'],
     [(hist['l'],                              'steelblue',  'l(t)'),
      (b * np.ones(len(hist['t'])),            'seagreen',   f'target b = {b} m', '--')],
     'l (m)'),

    ('ldot(t)',
     hist['t'],
     [(hist['ldot'],                           'mediumpurple', 'ldot(t)')],
     'ldot (m/s)'),

    ('Angles',
     hist['t'],
     [(np.degrees(hist['psi']),                'darkorange', 'psi (mur 2)'),
      (np.degrees(hist['delta']),              'seagreen',   'delta (bac)')],
     'deg'),

    ('Command fx, fy',
     hist['t'],
     [(hist['fx'],                             'steelblue', 'fx'),
      (hist['fy'],                             'tomato',    'fy')],
     'N'),

    ('fyA contact wall 1',
     hist['t'],
     [(hist['fya'],                            'steelblue', 'fyA')],
     'N'),

    ('fxB, fyB contact wall 2',
     hist['t'],
     [(hist['fxb'],                            'seagreen',   'fxB'),
      (hist['fyb'],                            'darkorange', 'fyB')],
     'N'),

    ('tau_spring',
     hist['t'],
     [(hist['tau_spring'],                     'tomato', 'τ_spring (N·m)')],
     'N·m'),
]

for title, t_arr, curves, ylabel in plots:
    fig, ax = plt.subplots(figsize=(7, 4))
    for curve in curves:
        y, color, label = curve[0], curve[1], curve[2]
        ls = curve[3] if len(curve) > 3 else '-'
        ax.plot(t_arr, y, color=color, lw=1.5, ls=ls, label=label)
    ax.axhline(0, color='gray', lw=0.8)
    if PSW_T:
        ax.axvline(PSW_T, color='gray', lw=1.2, ls='--', label='psi=127 deg')
    ax.set_title(title)
    ax.set_xlabel('t (s)')
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    plt.tight_layout()
    fname = (title.replace(' ', '_')
                  .replace('[', '').replace(']', '')
                  .replace('/', '').replace(',', '')
                  .strip() + '_sim.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()

# ══════════════════════════════════════════════════════════════════════
# ANIMATION
# ══════════════════════════════════════════════════════════════════════
skip   = 10
frames = np.arange(0, len(hist['t']), skip)

fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.set_xlim(-0.65, 0.65)
ax2.set_ylim(-0.10, 0.65)
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)
ax2.set_xlabel('x (m)')
ax2.set_ylabel('y (m)')
ax2.set_title('Simulation with spring')

wall1_line, = ax2.plot([0, 0.6], [0, 0], 'k',   lw=2,   label='wall 1 (fixed)')
wall2_line, = ax2.plot([], [],             'k',   lw=2,   label='wall 2 (with spring)')
spring_arc, = ax2.plot([], [],             'r--', lw=1.2, alpha=0.7, label='spring')
body_line,  = ax2.plot([], [],             'b',   lw=1.5, label='crate')
A_pt,       = ax2.plot([], [], 'ko', ms=6, label='A')
B_pt,       = ax2.plot([], [], 'ks', ms=6, label='B')
O_pt,       = ax2.plot([], [], 'k^', ms=6, label='COM')
force_line, = ax2.plot([], [], 'r',   lw=2,   label='force')
traj_line,  = ax2.plot([], [], 'b--', lw=1,   alpha=0.4)

info_txt = ax2.text(0.02, 0.97, '', transform=ax2.transAxes,
                    fontsize=9, va='top', fontfamily='monospace',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

ax2.legend(loc='upper right', fontsize=8)
traj_x, traj_y = [], []

def update(i):
    psi_i   = hist['psi'][i]
    delta_i = hist['delta'][i]
    t_i     = hist['t'][i]
    fx_i    = hist['fx'][i]
    fy_i    = hist['fy'][i]
    tau_i   = hist['tau_spring'][i]

    L2 = 0.55
    wall2_line.set_data([0, L2*np.cos(psi_i)], [0, L2*np.sin(psi_i)])

    r_arc = 0.10
    theta_arc = np.linspace(psi_rest, psi_i, 30)
    spring_arc.set_data(r_arc * np.cos(theta_arc), r_arc * np.sin(theta_arc))

    R_i     = rot(delta_i)
    xA_i, yA_i = hist['xA'][i], hist['yA'][i]
    corners = np.array([[0,0],[a,0],[a,b],[0,b],[0,0]])
    world   = np.array([[xA_i, yA_i] + R_i @ c for c in corners])
    body_line.set_data(world[:, 0], world[:, 1])

    A_pt.set_data([hist['xA'][i]], [hist['yA'][i]])
    B_pt.set_data([hist['xB'][i]], [hist['yB'][i]])
    O_pt.set_data([hist['xO'][i]], [hist['yO'][i]])

    xO_i, yO_i = hist['xO'][i], hist['yO'][i]
    scale = 0.025
    force_line.set_data([xO_i, xO_i + fx_i*scale],
                        [yO_i, yO_i + fy_i*scale])

    traj_x.append(xO_i); traj_y.append(yO_i)
    traj_line.set_data(traj_x, traj_y)

    phase = "Phase 1 : open wall" if psi_i < PSI_SWITCH else "Phase 2 : slide crate"
    info_txt.set_text(
        f"t          = {t_i:.2f} s\n"
        f"psi        = {np.degrees(psi_i):.1f} deg\n"
        f"delta      = {np.degrees(delta_i):.1f} deg\n"
        f"l          = {hist['l'][i]:.3f} m\n"
        f"ldot       = {hist['ldot'][i]:.3f} m/s\n"
        f"tau_spring= {tau_i:.2f} N.m\n"
        f"fx         = {fx_i:.1f} N\n"
        f"fy         = {fy_i:.1f} N\n"
        f"{phase}"
    )
    return wall2_line, spring_arc, body_line, A_pt, B_pt, O_pt, force_line, traj_line, info_txt

ani = FuncAnimation(fig2, update, frames=frames,
                    interval=int(1000 * dt * skip), blit=False)

plt.tight_layout()
ani.save('animation_sim_ressort.gif', writer='pillow', fps=int(1 / (dt * skip)))
plt.show()
