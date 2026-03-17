import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ══════════════════════════════════════════════════════════════════════
# PARAMÈTRES PHYSIQUES
# ══════════════════════════════════════════════════════════════════════
m      = 7.0
a      = 0.3
b      = 0.4
I_A    = (m / 3.0) * (a**2 + b**2)
mu     = 0.3
T_wall = 6.0

rAB_body = np.array([0.0, b])
rAO_body = np.array([a/2, b/2])

# ══════════════════════════════════════════════════════════════════════
# PROFIL DU MUR 2
# ══════════════════════════════════════════════════════════════════════
def wall_profile(t):
    psi_start = np.pi / 2
    psi_end   = np.pi
    dpsi      = psi_end - psi_start
    t1 = T_wall * 0.4
    t2 = T_wall * 0.6
    acc = dpsi / (0.5*t1**2 + (t2 - t1)*t1 + 0.5*(T_wall - t2)**2)
    v1  = acc * t1
    p1  = psi_start + 0.5*acc*t1**2
    p2  = p1 + v1*(t2 - t1)
    if t <= t1:
        psi, psidot, psiddot = psi_start + 0.5*acc*t**2, acc*t, acc
    elif t <= t2:
        psi, psidot, psiddot = p1 + v1*(t - t1), v1, 0.0
    elif t <= T_wall:
        psi     = p2 + v1*(t - t2) - 0.5*acc*(t - t2)**2
        psidot  = v1 - acc*(t - t2)
        psiddot = -acc
    else:
        psi, psidot, psiddot = psi_end, 0.0, 0.0
    return psi, psidot, psiddot

# ══════════════════════════════════════════════════════════════════════
# UTILITAIRES GÉOMÉTRIQUES
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
# CINÉMATIQUE DE δ
# ══════════════════════════════════════════════════════════════════════
def delta_kinematics(l, ldot, psi, psidot, psiddot):
    eps = 1e-9
    u  = np.clip((l / b) * np.sin(psi), -1 + eps, 1 - eps)
    k  = 1.0 / np.sqrt(max(eps, 1 - u*u))
    k3 = k**3
    delta        = psi - 0.5*np.pi + np.arcsin(u)
    u_l          = (1/b) * np.sin(psi)
    u_psi        = (l/b) * np.cos(psi)
    u_lpsi       = (1/b) * np.cos(psi)
    u_psipsi     = -(l/b) * np.sin(psi)
    delta_l      = k * u_l
    delta_psi    = 1 + k * u_psi
    delta_ll     = u * k3 * u_l * u_l
    delta_lpsi   = k * u_lpsi   + u * k3 * u_l   * u_psi
    delta_psipsi = k * u_psipsi + u * k3 * u_psi * u_psi
    delta_dot    = delta_l * ldot + delta_psi * psidot
    return delta, delta_dot, delta_l, delta_psi, delta_ll, delta_lpsi, delta_psipsi

# ══════════════════════════════════════════════════════════════════════
# DYNAMIQUE CONTINUE
# ══════════════════════════════════════════════════════════════════════
def f_continuous(l, ldot, fx, fy, t):
    psi, psidot, psiddot = wall_profile(t)
    (delta, delta_dot, delta_l, delta_psi,
     delta_ll, delta_lpsi, delta_psipsi) = delta_kinematics(l, ldot, psi, psidot, psiddot)
    R  = rot(delta)
    Rp = rot_prime(delta)
    rAO_w = R  @ rAO_body
    rAB_w = R  @ rAB_body
    Rp_r  = Rp @ rAO_body
    R_r   = R  @ rAO_body
    ax_coeff = 1.0 + Rp_r[0] * delta_l
    ay_coeff =       Rp_r[1] * delta_l
    nl_ddot  = (delta_ll * ldot**2 + 2*delta_lpsi * ldot * psidot
                + delta_psipsi * psidot**2 + delta_psi * psiddot)
    ax_nl = Rp_r[0] * nl_ddot - R_r[0] * delta_dot**2
    ay_nl = Rp_r[1] * nl_ddot - R_r[1] * delta_dot**2
    tB    = np.array([ np.cos(psi),  np.sin(psi)])
    nB    = np.array([ np.sin(psi), -np.cos(psi)])
    fBdir = nB - mu * tB
    M = np.array([
        [ m * ax_coeff,   mu,  -fBdir[0]             ],
        [ m * ay_coeff,  -1.0, -fBdir[1]             ],
        [ I_A * delta_l,  0.0, -cross2(rAB_w, fBdir) ]
    ])
    rhs = np.array([
        fx - m * ax_nl,
        fy - m * ay_nl,
        cross2(rAO_w, np.array([fx, fy])) - I_A * nl_ddot
    ])
    try:
        sol = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        sol = np.zeros(3)
    lddot = sol[0]
    fyA   = sol[1]
    fBn   = sol[2]
    fxB   = fBn * fBdir[0]
    fyB   = fBn * fBdir[1]
    return ldot, lddot, fyA, fBn, fxB, fyB, fBdir, psi

# ══════════════════════════════════════════════════════════════════════
# INTÉGRATION RK4
# ══════════════════════════════════════════════════════════════════════
def f_ode(l, ldot, fx, fy, t):
    d, a = f_continuous(l, ldot, fx, fy, t)[:2]
    return d, a

def rk4_step(l, ldot, fx, fy, t, h):
    d1, a1 = f_ode(l,           ldot,           fx, fy, t)
    d2, a2 = f_ode(l + h/2*d1,  ldot + h/2*a1, fx, fy, t + h/2)
    d3, a3 = f_ode(l + h/2*d2,  ldot + h/2*a2, fx, fy, t + h/2)
    d4, a4 = f_ode(l + h*d3,    ldot + h*a3,   fx, fy, t + h)
    l_new    = l    + (h/6) * (d1 + 2*d2 + 2*d3 + d4)
    ldot_new = ldot + (h/6) * (a1 + 2*a2 + 2*a3 + a4)
    return l_new, ldot_new

# ══════════════════════════════════════════════════════════════════════
# COMMANDE INTUITIVE
# ══════════════════════════════════════════════════════════════════════
PSI_SWITCH = 127 * np.pi / 180
F1 = 10.0

def get_command(t):
    psi, _, _ = wall_profile(t)
    if psi < PSI_SWITCH:
        nB = np.array([np.sin(psi), -np.cos(psi)])
        return -F1 * nB[0], -F1 * nB[1]
    else:
        return 10.0 * 0.2, 10.0 * (-1.0)

# ══════════════════════════════════════════════════════════════════════
# SIMULATION
# ══════════════════════════════════════════════════════════════════════
dt    = 0.002
T_sim = 10.0
N     = int(T_sim / dt)
l     = 0.0
ldot  = 0.0

hist = {k: [] for k in ['t', 'l', 'ldot', 'psi', 'delta',
                         'fx', 'fy', 'xA', 'yA', 'xB', 'yB', 'xO', 'yO',
                         'fya', 'fxb', 'fyb']}

for k in range(N + 1):
    t = k * dt
    psi, psidot, psiddot = wall_profile(t)
    delta = delta_kinematics(l, ldot, psi, psidot, psiddot)[0]
    R     = rot(delta)
    xA, yA = l, 0.0
    rAO   = R @ rAO_body
    rAB   = R @ rAB_body
    xO, yO = xA + rAO[0], yA + rAO[1]
    xB, yB = xA + rAB[0], yA + rAB[1]
    fx, fy = get_command(t)
    _, _, fyA_i, _, fxB_i, fyB_i, _, _ = f_continuous(l, ldot, fx, fy, t)
    for key, val in zip(hist.keys(),
                        [t, l, ldot, psi, delta, fx, fy,
                         xA, yA, xB, yB, xO, yO,
                         fyA_i, fxB_i, fyB_i]):
        hist[key].append(val)
    l_new, ldot_new = rk4_step(l, ldot, fx, fy, t, dt)
    l    = np.clip(l_new, 0.0, b)
    ldot = ldot_new
    if l <= 0.0 and ldot < 0: ldot = 0.0
    if l >= b   and ldot > 0: ldot = 0.0

PSW_T = next((hist['t'][i] for i, p in enumerate(hist['psi'])
              if p >= PSI_SWITCH), None)

print(f"Switch at  t = {PSW_T:.3f} s  (psi = 127 deg)")
print(f"l at switch  = {hist['l'][int(PSW_T/dt)]:.4f} m")
print(f"l final      = {hist['l'][-1]:.4f} m  (target = {b} m)")
print(f"ldot final   = {hist['ldot'][-1]:.4f} m/s  (ideal = 0)")
print(f"Max |ldot|   = {max(abs(v) for v in hist['ldot']):.4f} m/s")

# ══════════════════════════════════════════════════════════════════════
# PLOTS STATIQUES
# ══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

def vline(ax, label=True):
    if PSW_T:
        ax.axvline(PSW_T, color='gray', lw=1.2, ls='--',
                   label='psi=127 deg' if label else None)


plots = [
    ('l(t)',              hist['t'], [(hist['l'],    'steelblue', 'l(t)'),
                                      (b * np.ones(len(hist['t'])), 'seagreen', f'target b = {b} m', '--')],
     'l (m)'),
    ('ldot(t)',           hist['t'], [(hist['ldot'], 'mediumpurple', 'ldot(t)')],
     'ldot (m/s)'),
    ('Angles',            hist['t'], [(np.degrees(hist['psi']),   'darkorange', 'psi (wall 2)'),
                                      (np.degrees(hist['delta']), 'seagreen',   'delta (box)')],
     'deg'),
    ('Command [fx, fy]',  hist['t'], [(hist['fx'], 'steelblue', 'fx'),
                                      (hist['fy'], 'tomato',    'fy')],
     'N'),
    ('fyA — contact wall 1', hist['t'], [(hist['fya'], 'steelblue', 'fyA')],
     'N'),
    ('fxB, fyB — contact wall 2', hist['t'], [(hist['fxb'], 'seagreen',   'fxB'),
                                               (hist['fyb'], 'darkorange', 'fyB')],
     'N'),
]

for title, t, curves, ylabel in plots:
    fig, ax = plt.subplots(figsize=(7, 4))
    for curve in curves:
        y, color, label = curve[0], curve[1], curve[2]
        ls = curve[3] if len(curve) > 3 else '-'
        ax.plot(t, y, color=color, lw=1.5, ls=ls, label=label)
    ax.axhline(0, color='gray', lw=0.8)
    if PSW_T:
        ax.axvline(PSW_T, color='gray', lw=1.2, ls='--', label='psi=127 deg')
    ax.set_title(title)
    ax.set_xlabel('t (s)')
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    plt.tight_layout()
    filename = title.replace(' ', '_').replace('[', '').replace(']', '').replace('/', '').replace('—', '').strip() + '.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
# ══════════════════════════════════════════════════════════════════════
# ANIMATION
# ══════════════════════════════════════════════════════════════════════
skip   = 10
frames = np.arange(0, len(hist['t']), skip)

fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.set_xlim(-0.50, 0.65); ax2.set_ylim(-0.1, 0.65)
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)
ax2.set_xlabel('x (m)')
ax2.set_ylabel('y (m)')
ax2.set_title('System animation')

wall1_line, = ax2.plot([0, 0.6], [0, 0], 'k',  lw=2,  label='wall 1 (fixed)')
wall2_line, = ax2.plot([], [],             'k',  lw=2,  label='wall 2 (moving)')
body_line,  = ax2.plot([], [],             'b',  lw=1.5,label='box')
A_pt,       = ax2.plot([], [], 'ko', ms=6, label='A')
B_pt,       = ax2.plot([], [], 'ks', ms=6, label='B')
O_pt,       = ax2.plot([], [], 'k^', ms=6, label='C (COM)')
force_line, = ax2.plot([], [], 'r',  lw=2,  label='force')
traj_line,  = ax2.plot([], [], 'b--',lw=1,  alpha=0.4)

info_txt = ax2.text(0.02, 0.97, '', transform=ax2.transAxes,
                    fontsize=9, va='top', fontfamily='monospace',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

ax2.legend(loc='upper right', fontsize=8)

traj_x, traj_y = [], []

def update(i):
    psi   = hist['psi'][i]
    delta = hist['delta'][i]
    l_i   = hist['l'][i]
    t_i   = hist['t'][i]
    fx_i  = hist['fx'][i]
    fy_i  = hist['fy'][i]

    L2 = 0.55
    wall2_line.set_data([0, L2*np.cos(psi)], [0, L2*np.sin(psi)])

    R = rot(delta)
    xA_i, yA_i = hist['xA'][i], hist['yA'][i]
    corners = np.array([[0,0],[a,0],[a,b],[0,b],[0,0]])
    world   = np.array([[xA_i, yA_i] + R @ c for c in corners])
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

    phase = "Phase 1" if psi < PSI_SWITCH else "Phase 2"
    info_txt.set_text(
        f"t     = {t_i:.2f} s\n"
        f"psi   = {np.degrees(psi):.1f} deg\n"
        f"delta = {np.degrees(delta):.1f} deg\n"
        f"l     = {l_i:.3f} m\n"
        f"fx    = {fx_i:.1f} N\n"
        f"fy    = {fy_i:.1f} N\n"
        f"{phase}"
    )
    return wall2_line, body_line, A_pt, B_pt, O_pt, force_line, traj_line, info_txt

ani = FuncAnimation(fig2, update, frames=frames,
                    interval=int(1000 * dt * skip), blit=False)

plt.tight_layout()
ani.save('animation.gif', writer='pillow', fps=int(1 / (dt * skip)))
plt.show()