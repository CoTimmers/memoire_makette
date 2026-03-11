"probleme non optimisé, avec des forces intuitives"

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ══════════════════════════════════════════════════════════════════════
# PARAMÈTRES PHYSIQUES
# ══════════════════════════════════════════════════════════════════════
m     = 7.0
a     = 0.3
b     = 0.4
I_A   = (m / 3.0) * (a**2 + b**2)
mu    = 0.3
T_wall = 6.0

rAB_body = np.array([0.0, b])
rAO_body = np.array([a/2, b/2])

# ══════════════════════════════════════════════════════════════════════
# PROFIL DU MUR 2 : ψ va de π/2 à π avec profil trapézoïdal en vitesse
# ══════════════════════════════════════════════════════════════════════
def wall_profile(t):
    """
    Retourne (psi, psidot, psiddot) à l'instant t.
    psi : π/2 → π  (anti-horaire)
    Profil vitesse trapézoïdal : accélération / vitesse constante / décélération
    """
    psi_start = np.pi / 2
    psi_end   = np.pi
    dpsi      = psi_end - psi_start

    t1 = T_wall * 0.4   # fin de la phase d'accélération
    t2 = T_wall * 0.6   # début de la phase de décélération

    acc = dpsi / (0.5*t1**2 + (t2 - t1)*t1 + 0.5*(T_wall - t2)**2)
    v1  = acc * t1
    p1  = psi_start + 0.5*acc*t1**2
    p2  = p1 + v1*(t2 - t1)

    if t <= t1:
        psi     = psi_start + 0.5*acc*t**2
        psidot  = acc * t
        psiddot = acc
    elif t <= t2:
        psi     = p1 + v1*(t - t1)
        psidot  = v1
        psiddot = 0.0
    elif t <= T_wall:
        psi     = p2 + v1*(t - t2) - 0.5*acc*(t - t2)**2
        psidot  = v1 - acc*(t - t2)
        psiddot = -acc
    else:
        psi     = psi_end
        psidot  = 0.0
        psiddot = 0.0

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
    """
    Calcule delta et ses dérivées à partir de la contrainte géométrique
    (B reste sur le mur 2).

    Retourne :
        delta, delta_dot,
        delta_l, delta_psi,
        delta_ll, delta_lpsi, delta_psipsi
    """
    eps = 1e-9

    u = np.clip((l / b) * np.sin(psi), -1 + eps, 1 - eps)
    k  = 1.0 / np.sqrt(max(eps, 1 - u*u))
    k3 = k**3

    delta = psi - 0.5*np.pi + np.arcsin(u)

    # Dérivées partielles de u
    u_l      = (1/b) * np.sin(psi)
    u_psi    = (l/b) * np.cos(psi)
    u_lpsi   = (1/b) * np.cos(psi)
    u_psipsi = -(l/b) * np.sin(psi)

    # Dérivées partielles de delta
    delta_l      = k * u_l
    delta_psi    = 1 + k * u_psi
    delta_ll     = u * k3 * u_l * u_l
    delta_lpsi   = k * u_lpsi   + u * k3 * u_l   * u_psi
    delta_psipsi = k * u_psipsi + u * k3 * u_psi * u_psi

    # Vitesse angulaire de delta
    delta_dot = delta_l * ldot + delta_psi * psidot

    return delta, delta_dot, delta_l, delta_psi, delta_ll, delta_lpsi, delta_psipsi


# ══════════════════════════════════════════════════════════════════════
# DYNAMIQUE CONTINUE  ẋ = f(x, u)
# ══════════════════════════════════════════════════════════════════════
def f_continuous(l, ldot, fx, fy, t):
    """
    Résout le système linéaire 3×3 en [l̈, fyA, fBn]
    et retourne (ldot, lddot).

    Équations de Newton :
        eq1 : fA[0] + fB[0] + fx = m * ax
        eq2 : fA[1] + fB[1] + fy = m * ay
        eq3 : cross(rAO, fc) + cross(rAB, fB) = I_A * delta_ddot

    Modèle de contact :
        fA = [-mu*fyA, fyA]                  (glissement sur mur 1)
        fB = fBn * nB + (-mu*fBn) * tB       (contact sur mur 2)
    """
    psi, psidot, psiddot = wall_profile(t)

    (delta, delta_dot, delta_l, delta_psi,
     delta_ll, delta_lpsi, delta_psipsi) = delta_kinematics(l, ldot, psi, psidot, psiddot)

    R  = rot(delta)
    Rp = rot_prime(delta)

    rAO_w = R  @ rAO_body
    rAB_w = R  @ rAB_body
    Rp_r  = Rp @ rAO_body
    R_r   = R  @ rAO_body

    # ── Coefficients linéaires de l̈ dans (ax, ay, δ̈) ──────────────
    ax_coeff = 1.0 + Rp_r[0] * delta_l
    ay_coeff =       Rp_r[1] * delta_l

    # ── Termes non-linéaires (connus, dépendent de ldot et psidot) ──
    nl_ddot = (delta_ll      * ldot**2
               + 2*delta_lpsi   * ldot * psidot
               + delta_psipsi   * psidot**2
               + delta_psi      * psiddot)

    ax_nl = Rp_r[0] * nl_ddot - R_r[0] * delta_dot**2
    ay_nl = Rp_r[1] * nl_ddot - R_r[1] * delta_dot**2

    # ── Vecteurs de contact en B ─────────────────────────────────────
    tB    = np.array([ np.cos(psi),  np.sin(psi)])
    nB    = np.array([ np.sin(psi), -np.cos(psi)])
    fBdir = nB - mu * tB   # direction de fB par unité de fBn

    # ── Matrice M 3×3 — inconnues z = [l̈, fyA, fBn] ────────────────
    #
    # eq1 → m*(ax_nl + ax_coeff*l̈) = fx + (-mu*fyA) + fBn*fBdir[0]
    #      → m*ax_coeff*l̈ + mu*fyA - fBn*fBdir[0] = fx - m*ax_nl
    #
    # eq2 → m*ay_coeff*l̈ - fyA - fBn*fBdir[1] = fy - m*ay_nl
    #
    # eq3 → I_A*(nl_ddot + delta_l*l̈) - fBn*cross(rAB,fBdir)
    #              = cross(rAO, [fx,fy]) - I_A*nl_ddot
    M = np.array([
        [ m * ax_coeff,   mu,  -fBdir[0]               ],
        [ m * ay_coeff,  -1.0, -fBdir[1]               ],
        [ I_A * delta_l,  0.0, -cross2(rAB_w, fBdir)   ]
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
    fyA   = sol[1]   # force normale en A (mur 1)
    fBn   = sol[2]   # force normale en B (mur 2)

    fxB = fBn * fBdir[0]
    fyB = fBn * fBdir[1]

    return ldot, lddot, fyA, fBn, fxB, fyB, fBdir, psi


# ══════════════════════════════════════════════════════════════════════
# INTÉGRATION RK4
# ══════════════════════════════════════════════════════════════════════
def f_ode(l, ldot, fx, fy, t):
    """Version allégée de f_continuous pour RK4 (retourne seulement ldot, lddot)."""
    d, a = f_continuous(l, ldot, fx, fy, t)[:2]
    return d, a

def rk4_step(l, ldot, fx, fy, t, h):
    """Un pas RK4 de longueur h."""
    d1, a1 = f_ode(l,             ldot,             fx, fy, t)
    d2, a2 = f_ode(l + h/2*d1,    ldot + h/2*a1,   fx, fy, t + h/2)
    d3, a3 = f_ode(l + h/2*d2,    ldot + h/2*a2,   fx, fy, t + h/2)
    d4, a4 = f_ode(l + h*d3,      ldot + h*a3,     fx, fy, t + h)

    l_new    = l    + (h/6) * (d1 + 2*d2 + 2*d3 + d4)
    ldot_new = ldot + (h/6) * (a1 + 2*a2 + 2*a3 + a4)
    return l_new, ldot_new


# ══════════════════════════════════════════════════════════════════════
# COMMANDE INTUITIVE PAR MORCEAUX
# ══════════════════════════════════════════════════════════════════════
PSI_SWITCH = 3 * np.pi / 4   # 135° — seuil de changement de phase
F1 = 10.0                     # amplitude phase 1

def get_command(t):
    """
    Phase 1 (psi < 135°) : f = -nB = (-sin(psi), cos(psi))
        Force perpendiculaire au mur 2, plaquant le bac contre lui.
        Maintient l ≈ 0 pendant la rotation.

    Phase 2 (psi >= 135°) : f = 10 * (0.5, -1)
        Force diagonale bas-droite pour faire glisser l → b.
    """
    psi, _, _ = wall_profile(t)

    if psi < PSI_SWITCH:
        # nB = (sin(psi), -cos(psi)) → -nB = (-sin(psi), cos(psi))
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

l    = 0.0
ldot = 0.0

hist = {k: [] for k in ['t', 'l', 'ldot', 'psi', 'delta',
                         'fx', 'fy', 'xA', 'yA', 'xB', 'yB', 'xO', 'yO',
                         'fya', 'fxb', 'fyb']}

for k in range(N + 1):
    t = k * dt

    psi, psidot, psiddot = wall_profile(t)
    delta = delta_kinematics(l, ldot, psi, psidot, psiddot)[0]

    R    = rot(delta)
    xA, yA = l, 0.0
    rAO  = R @ rAO_body
    rAB  = R @ rAB_body
    xO, yO = xA + rAO[0], yA + rAO[1]
    xB, yB = xA + rAB[0], yA + rAB[1]

    fx, fy = get_command(t)

    # Forces de contact (calculées au point courant)
    _, _, fyA_i, _, fxB_i, fyB_i, _, _ = f_continuous(l, ldot, fx, fy, t)

    for key, val in zip(hist.keys(),
                        [t, l, ldot, psi, delta, fx, fy,
                         xA, yA, xB, yB, xO, yO,
                         fyA_i, fxB_i, fyB_i]):
        hist[key].append(val)

    # Intégration RK4
    l_new, ldot_new = rk4_step(l, ldot, fx, fy, t, dt)

    # Contraintes sur l
    l    = np.clip(l_new, 0.0, b)
    ldot = ldot_new
    if l <= 0.0 and ldot < 0: ldot = 0.0
    if l >= b   and ldot > 0: ldot = 0.0

# Instant du switch de phase
PSW_T = next((hist['t'][i] for i, p in enumerate(hist['psi'])
              if p >= PSI_SWITCH), None)

print(f"Switch de phase à  t = {PSW_T:.3f} s  (ψ = 135°)")
print(f"l à t_switch       = {hist['l'][int(PSW_T/dt)]:.4f} m")
print(f"l final            = {hist['l'][-1]:.4f} m  (cible = {b} m)")
print(f"ḷ final            = {hist['ldot'][-1]:.4f} m/s  (idéal = 0)")
print(f"Max |ḷ|            = {max(abs(v) for v in hist['ldot']):.4f} m/s")


# ══════════════════════════════════════════════════════════════════════
# PLOTS STATIQUES
# ══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.patch.set_facecolor('#0d0f14')

def style(ax):
    ax.set_facecolor('#13161e')
    ax.tick_params(colors='#8899bb')
    ax.grid(True, alpha=0.2, color='#232840')
    for sp in ax.spines.values():
        sp.set_edgecolor('#232840')
    ax.xaxis.label.set_color('#8899bb')
    ax.yaxis.label.set_color('#8899bb')
    ax.title.set_color('#c8d4f0')

for ax in axes.flat:
    style(ax)

def vline(ax, label=True):
    if PSW_T:
        ax.axvline(PSW_T, color='#ffb347', lw=1.2, ls='--', alpha=0.9,
                   label='ψ=135°' if label else None)

def phase_labels(ax):
    """Annotations Phase 1 / Phase 2 sous le graphe."""
    if PSW_T:
        ylo, yhi = ax.get_ylim()
        y = ylo + 0.04 * (yhi - ylo)
        ax.text(PSW_T / 2, y, 'Phase 1 : f = −nB',
                color='#5ee7ff', fontsize=7, ha='center', alpha=0.8)
        ax.text((PSW_T + T_sim) / 2, y, 'Phase 2 : f = 10·(0.2,−1)',
                color='#56f0a0', fontsize=7, ha='center', alpha=0.8)

# ── l(t) ──
axes[0,0].plot(hist['t'], hist['l'], color='#5ee7ff', lw=1.8, label='l(t)')
axes[0,0].axhline(b, color='#56f0a0', lw=1, ls=':', alpha=0.7, label=f'cible b = {b} m')
vline(axes[0,0])
axes[0,0].set_title('l(t)')
axes[0,0].set_ylabel('l (m)'); axes[0,0].set_xlabel('t (s)')
axes[0,0].legend(fontsize=8, facecolor='#13161e', labelcolor='#c8d4f0')
phase_labels(axes[0,0])

# ── ḷ(t) ──
axes[0,1].plot(hist['t'], hist['ldot'], color='#b57aff', lw=1.8)
axes[0,1].axhline(0, color='#3a4560', lw=1)
vline(axes[0,1], False)
axes[0,1].set_title('ḷ(t)')
axes[0,1].set_ylabel('ḷ (m/s)'); axes[0,1].set_xlabel('t (s)')
phase_labels(axes[0,1])

# ── Angles ──
axes[0,2].plot(hist['t'], np.degrees(hist['psi']),   color='#ffb347', lw=1.8, label='ψ (mur 2)')
axes[0,2].plot(hist['t'], np.degrees(hist['delta']), color='#56f0a0', lw=1.8, label='δ (bac)')
vline(axes[0,2], False)
axes[0,2].set_title('Angles')
axes[0,2].set_ylabel('degrés'); axes[0,2].set_xlabel('t (s)')
axes[0,2].legend(fontsize=8, facecolor='#13161e', labelcolor='#c8d4f0')
phase_labels(axes[0,2])

# ── Commande ──
axes[1,0].plot(hist['t'], hist['fx'], color='#5ee7ff', lw=1.8, label='fx')
axes[1,0].plot(hist['t'], hist['fy'], color='#ff5f6d', lw=1.8, label='fy')
axes[1,0].axhline(0, color='#3a4560', lw=1)
vline(axes[1,0], False)
axes[1,0].set_title('Commande [fx, fy]')
axes[1,0].set_ylabel('N'); axes[1,0].set_xlabel('t (s)')
axes[1,0].legend(fontsize=8, facecolor='#13161e', labelcolor='#c8d4f0')
phase_labels(axes[1,0])

# ── fyA — force normale en A (mur 1) ──
axes[1,1].plot(hist['t'], hist['fya'], color='#5ee7ff', lw=1.8, label='fyA')
axes[1,1].axhline(0, color='#3a4560', lw=1)
vline(axes[1,1], False)
axes[1,1].set_title('fyA — contact mur 1')
axes[1,1].set_ylabel('N'); axes[1,1].set_xlabel('t (s)')
axes[1,1].legend(fontsize=8, facecolor='#13161e', labelcolor='#c8d4f0')
phase_labels(axes[1,1])

# ── fxB, fyB — force de contact en B (mur 2) ──
axes[1,2].plot(hist['t'], hist['fxb'], color='#56f0a0', lw=1.8, label='fxB')
axes[1,2].plot(hist['t'], hist['fyb'], color='#ffb347', lw=1.8, label='fyB')
axes[1,2].axhline(0, color='#3a4560', lw=1)
vline(axes[1,2], False)
axes[1,2].set_title('fxB, fyB — contact mur 2')
axes[1,2].set_ylabel('N'); axes[1,2].set_xlabel('t (s)')
axes[1,2].legend(fontsize=8, facecolor='#13161e', labelcolor='#c8d4f0')
phase_labels(axes[1,2])



plt.suptitle('Simulation — commande intuitive par morceaux',
             color='#c8d4f0', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('simulation_plots.png', dpi=150, bbox_inches='tight')
plt.show()


# ══════════════════════════════════════════════════════════════════════
# ANIMATION
# ══════════════════════════════════════════════════════════════════════
skip = 10
frames = np.arange(0, len(hist['t']), skip)

fig2, ax2 = plt.subplots(figsize=(10, 7))
fig2.patch.set_facecolor('#0d0f14')
ax2.set_facecolor('#13161e')
ax2.set_xlim(-0.50, 0.65); ax2.set_ylim(-0.1, 0.65)
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.2, color='#232840')
ax2.tick_params(colors='#8899bb')
for sp in ax2.spines.values():
    sp.set_edgecolor('#232840')
ax2.set_xlabel('x (m)', color='#8899bb')
ax2.set_ylabel('y (m)', color='#8899bb')
ax2.set_title('Animation du système', color='#c8d4f0')

# Éléments graphiques
wall1_line, = ax2.plot([0, 0.6], [0, 0],   color='#5ee7ff', lw=3,   label='mur 1 (fixe)')
wall2_line, = ax2.plot([], [],               color="#ffad3b", lw=3,   label='mur 2 (mobile)')
body_line,  = ax2.plot([], [],               color='#c8d4f0', lw=2,   label='bac')
A_pt,       = ax2.plot([], [], 'o',          color='#ff5f6d', ms=7,   label='A')
B_pt,       = ax2.plot([], [], 'o',          color='#56f0a0', ms=7,   label='B')
O_pt,       = ax2.plot([], [], 'o',          color='#ffb347', ms=7,   label='O (COM)')
force_line, = ax2.plot([], [],               color='#b57aff', lw=2,   label='force grue')
traj_line,  = ax2.plot([], [],               color='#ffb347', lw=1,   alpha=0.4)

info_txt = ax2.text(0.02, 0.97, '', transform=ax2.transAxes,
                    color='#c8d4f0', fontsize=9, va='top',
                    fontfamily='monospace',
                    bbox=dict(facecolor='#13161e', alpha=0.7, edgecolor='#232840'))

ax2.legend(loc='upper right', fontsize=8,
           facecolor='#13161e', labelcolor='#c8d4f0', edgecolor='#232840')

traj_x, traj_y = [], []

def update(i):
    psi   = hist['psi'][i]
    delta = hist['delta'][i]
    l_i   = hist['l'][i]
    t_i   = hist['t'][i]
    fx_i  = hist['fx'][i]
    fy_i  = hist['fy'][i]

    # Mur 2
    L2 = 0.55
    wall2_line.set_data([0, L2*np.cos(psi)], [0, L2*np.sin(psi)])

    # Corps rigide : 4 coins du rectangle
    R = rot(delta)
    xA_i, yA_i = hist['xA'][i], hist['yA'][i]
    corners = np.array([[0,0],[a,0],[a,b],[0,b],[0,0]])
    world   = np.array([[xA_i, yA_i] + R @ c for c in corners])
    body_line.set_data(world[:, 0], world[:, 1])

    A_pt.set_data([hist['xA'][i]], [hist['yA'][i]])
    B_pt.set_data([hist['xB'][i]], [hist['yB'][i]])
    O_pt.set_data([hist['xO'][i]], [hist['yO'][i]])

    # Flèche de force
    xO_i, yO_i = hist['xO'][i], hist['yO'][i]
    scale = 0.025
    force_line.set_data([xO_i, xO_i + fx_i*scale],
                        [yO_i, yO_i + fy_i*scale])

    # Trace du COM
    traj_x.append(xO_i); traj_y.append(yO_i)
    traj_line.set_data(traj_x, traj_y)

    # Texte
    phase = "Phase 1 : f = −nB" if psi < PSI_SWITCH else "Phase 2 : f = 10·(0.2,−1)"
    info_txt.set_text(
        f"t    = {t_i:.2f} s\n"
        f"ψ    = {np.degrees(psi):.1f}°\n"
        f"δ    = {np.degrees(delta):.1f}°\n"
        f"l    = {l_i:.3f} m\n"
        f"fx   = {fx_i:.1f} N\n"
        f"fy   = {fy_i:.1f} N\n"
        f"{phase}"
    )

    return wall2_line, body_line, A_pt, B_pt, O_pt, force_line, traj_line, info_txt

ani = FuncAnimation(fig2, update, frames=frames,
                    interval=int(1000 * dt * skip), blit=False)

plt.tight_layout()
plt.show()