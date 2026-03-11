import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

"perte contacte à b, alors que non !"
# ══════════════════════════════════════════════════════════════════════
# PARAMÈTRES PHYSIQUES
# ══════════════════════════════════════════════════════════════════════
m      = 7.0
a      = 0.3
b      = 0.4
I_A    = (m / 3.0) * (a**2 + b**2)
mu     = 0.3
T_wall = 6.0
T_sim  = 10.0

rAB_body = ca.vertcat(0.0, b)
rAO_body = ca.vertcat(a/2, b/2)


# ══════════════════════════════════════════════════════════════════════
# PROFIL DU MUR 2
# ══════════════════════════════════════════════════════════════════════
def wall_profile_ca(t):
    psi_start = ca.pi / 2
    psi_end   = ca.pi
    dpsi      = psi_end - psi_start
    t1  = T_wall * 0.4
    t2  = T_wall * 0.6
    acc = dpsi / (0.5*t1**2 + (t2-t1)*t1 + 0.5*(T_wall-t2)**2)
    v1  = acc * t1
    p1  = psi_start + 0.5*acc*t1**2
    p2  = p1 + v1*(t2 - t1)

    psi = ca.if_else(t <= t1,
            psi_start + 0.5*acc*t**2,
          ca.if_else(t <= t2,
            p1 + v1*(t - t1),
          ca.if_else(t <= T_wall,
            p2 + v1*(t - t2) - 0.5*acc*(t - t2)**2,
            psi_end)))

    psidot = ca.if_else(t <= t1, acc*t,
             ca.if_else(t <= t2, v1,
             ca.if_else(t <= T_wall, v1 - acc*(t - t2), 0.0)))

    psiddot = ca.if_else(t <= t1,     acc,
              ca.if_else(t <= t2,     0.0,
              ca.if_else(t <= T_wall, -acc, 0.0)))

    return psi, psidot, psiddot


# ══════════════════════════════════════════════════════════════════════
# GÉOMÉTRIE
# ══════════════════════════════════════════════════════════════════════
def rot_ca(theta):
    return ca.vertcat(
        ca.horzcat(ca.cos(theta), -ca.sin(theta)),
        ca.horzcat(ca.sin(theta),  ca.cos(theta))
    )

def rot_prime_ca(theta):
    return ca.vertcat(
        ca.horzcat(-ca.sin(theta), -ca.cos(theta)),
        ca.horzcat( ca.cos(theta), -ca.sin(theta))
    )

def cross2(r, f):
    return r[0]*f[1] - r[1]*f[0]


# ══════════════════════════════════════════════════════════════════════
# CINÉMATIQUE DE δ  —  lddot est une entrée (variable d'optim)
# ══════════════════════════════════════════════════════════════════════
def delta_kinematics_ca(l, ldot, lddot, psi, psidot, psiddot):
    eps = 1e-9
    u_raw = (l / b) * ca.sin(psi)
    u = ca.fmin(1.0 - eps, ca.fmax(-1.0 + eps, u_raw))

    one_minus = ca.fmax(eps, 1 - u*u)
    k  = 1.0 / ca.sqrt(one_minus)
    k3 = k**3

    delta = psi - 0.5*ca.pi + ca.asin(u)

    u_l      = (1/b) * ca.sin(psi)
    u_psi    = (l/b) * ca.cos(psi)
    u_lpsi   = (1/b) * ca.cos(psi)
    u_psipsi = -(l/b) * ca.sin(psi)

    delta_l      = k * u_l
    delta_psi    = 1 + k * u_psi
    delta_ll     = u * k3 * u_l * u_l
    delta_lpsi   = k * u_lpsi   + u * k3 * u_l   * u_psi
    delta_psipsi = k * u_psipsi + u * k3 * u_psi * u_psi

    delta_dot = delta_l * ldot + delta_psi * psidot

    delta_ddot = (
        delta_ll     * ldot**2
      + 2*delta_lpsi * ldot * psidot
      + delta_psipsi * psidot**2
      + delta_l      * lddot
      + delta_psi    * psiddot
    )

    R  = rot_ca(delta)
    Rp = rot_prime_ca(delta)
    r_w  = R  @ rAO_body
    rp_w = Rp @ rAO_body

    ax = lddot + rp_w[0] * delta_ddot - r_w[0] * delta_dot**2
    ay =         rp_w[1] * delta_ddot - r_w[1] * delta_dot**2

    return delta, delta_dot, delta_ddot, ax, ay


# ══════════════════════════════════════════════════════════════════════
# CONSTRUCTION DU SOLVEUR NLP (1 seul sous-pas)
#
# Variables : x = [fx, fy, lddot, fyA, fBn]   (5 variables)
# Paramètres: p = [t, l, ldot]
#
# Contraintes d'égalité  : F(x, p) = 0   (3 équations de Newton)
# Contraintes d'inégalité: fyA >= 0, fBn >= 0
#
# Ce solveur est appelé 4 fois par pas RK4 avec des (t, l, ldot)
# intermédiaires différents — exactement comme f_continuous dans
# simulation.py, mais avec (fx, fy) optimisés plutôt que fixés.
# ══════════════════════════════════════════════════════════════════════
def build_nlp_solver():
    # ── Variables ──
    fx    = ca.SX.sym("fx")
    fy    = ca.SX.sym("fy")
    lddot = ca.SX.sym("lddot")
    fyA   = ca.SX.sym("fyA")
    fBn   = ca.SX.sym("fBn")
    x_var = ca.vertcat(fx, fy, lddot, fyA, fBn)

    # ── Paramètres ──
    t_p    = ca.SX.sym("t")
    l_p    = ca.SX.sym("l")
    ldot_p = ca.SX.sym("ldot")
    p_var  = ca.vertcat(t_p, l_p, ldot_p)

    # ── Cinématique ──
    psi, psidot, psiddot = wall_profile_ca(t_p)

    delta, delta_dot, delta_ddot, ax, ay = delta_kinematics_ca(
        l_p, ldot_p, lddot, psi, psidot, psiddot
    )

    # ── Forces de contact ──
    tB = ca.vertcat( ca.cos(psi),  ca.sin(psi))
    nB = ca.vertcat( ca.sin(psi), -ca.cos(psi))

    fA = ca.vertcat(-mu * fyA, fyA)
    fB = fBn * nB + (-mu * fBn) * tB
    fc = ca.vertcat(fx, fy)

    R     = rot_ca(delta)
    rAO_w = R @ rAO_body
    rAB_w = R @ rAB_body

    # ── Équations de Newton ──
    eq1 = (fA[0] + fB[0] + fx) - m * ax
    eq2 = (fA[1] + fB[1] + fy) - m * ay
    eq3 = cross2(rAO_w, fc) + cross2(rAB_w, fB) - I_A * delta_ddot
    F_eq = ca.vertcat(eq1, eq2, eq3)

    # ── Coût : uniquement sur le sous-pas k1 (t, l, ldot) ──
    # Le coût prédictif n'est évalué qu'au premier appel (k1)
    # Pour k2, k3, k4 on minimise juste ||u||² pour la régularisation
    T_pred    = 1.0
    l_pred    = l_p    + T_pred*ldot_p + 0.5*T_pred**2 * lddot
    ldot_pred = ldot_p + T_pred*lddot
    s_pred    = ca.fmin(1.0, ca.fmax(0.0, (t_p + T_pred) / T_sim))
    l_ref_pred    = b * (3*s_pred**2 - 2*s_pred**3)

    l_brake = 0.7 * b
    w_vel   = 1.0 + 50.0 * ca.fmax(0.0, (l_p - l_brake) / (b - l_brake))**2

    q_l    = 100.0
    q_ldot = 10.0
    r_f    = 0.001
    J = (  q_l    * (l_pred - l_ref_pred)**2
         + q_ldot * w_vel * ldot_pred**2
         + r_f    * (fx**2 + fy**2) )

    # ── Contraintes ──
    g = ca.vertcat(F_eq, fyA, fBn)

    nlp  = {"x": x_var, "p": p_var, "f": J, "g": g}
    opts = {
        "ipopt.print_level": 0,
        "print_time":        0,
        "ipopt.max_iter":    200,
        "ipopt.tol":         1e-6,
    }
    S = ca.nlpsol("nlp_substep", "ipopt", nlp, opts)

    # g = [eq1=0, eq2=0, eq3=0, fyA>=0, fBn>=0]
    lbg = [0., 0., 0.,   0., 0.]
    ubg = [0., 0., 0.,   ca.inf, ca.inf]

    # x = [fx, fy, lddot, fyA, fBn]
    lbx = [-100., -100., -1e4, 0., 0.]
    ubx = [ 100.,  100.,  1e4, 1e6, 1e6]

    extras_fn = ca.Function("extras", [x_var, p_var],
                            [delta, psi, rAO_w, rAB_w])

    def solve_substep(tv, lv, ldotv, x0=None):
        """
        Résout le NLP pour un état (tv, lv, ldotv).
        Retourne (lddot, fyA, fBn, fx, fy, x_opt).
        """
        if x0 is None:
            x0 = [0., 0., 0., 1., 1.]

        sol = S(
            x0=x0,
            p=[tv, lv, ldotv],
            lbx=lbx, ubx=ubx,
            lbg=lbg, ubg=ubg
        )
        x_opt = sol["x"].full().squeeze()
        return x_opt   # [fx, fy, lddot, fyA, fBn]

    return solve_substep, extras_fn


# ══════════════════════════════════════════════════════════════════════
# SIMULATION MPC + RK4
#
# À chaque pas de temps dt :
#   k1 = NLP(t,       l,              ldot)
#   k2 = NLP(t+dt/2,  l+dt/2*ldot,   ldot+dt/2*k1_lddot)
#   k3 = NLP(t+dt/2,  l+dt/2*ldot,   ldot+dt/2*k2_lddot)  [même t que k2]
#   k4 = NLP(t+dt,    l+dt*k3_ldot,  ldot+dt*k3_lddot)
#
#   l_next    = l    + dt/6*(k1_ldot  + 2*k2_ldot  + 2*k3_ldot  + k4_ldot)
#   ldot_next = ldot + dt/6*(k1_lddot + 2*k2_lddot + 2*k3_lddot + k4_lddot)
#
# La commande (fx, fy) utilisée pour l'état est celle de k1.
# Les 3 autres appels NLP servent à trouver le lddot cohérent
# avec l'état intermédiaire — la commande peut légèrement varier.
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    dt  = 0.002
    N   = int(T_sim / dt)

    print("Construction du solveur NLP...")
    solve_substep, extras_fn = build_nlp_solver()

    l    = 0.0
    ldot = 0.0

    # Warm-start partagé entre les sous-pas
    x0_k1 = [0., 0., 0., 1., 1.]
    x0_k2 = [0., 0., 0., 1., 1.]
    x0_k3 = [0., 0., 0., 1., 1.]
    x0_k4 = [0., 0., 0., 1., 1.]

    hist = {k: [] for k in [
        "t", "l", "ldot", "lddot",
        "psi", "delta",
        "fx", "fy",
        "fyA", "fBn",
        "xO", "yO", "xB", "yB"
    ]}

    print(f"Simulation MPC + RK4 sur {T_sim}s ({N} pas, 4 NLP/pas)...")
    for k in range(N + 1):
        t = k * dt

        # ── k1 : état courant (t, l, ldot) ──
        sol1 = solve_substep(t,        l,                ldot,              x0_k1)
        fx1, fy1, lddot1, fyA1, fBn1 = sol1
        x0_k1 = sol1.tolist()

        # ── k2 : état intermédiaire à t+dt/2 ──
        l2    = l    + (dt/2)*ldot   + (dt/2)**2/2 * lddot1
        ldot2 = ldot + (dt/2)*lddot1
        sol2  = solve_substep(t+dt/2,  l2,               ldot2,             x0_k2)
        fx2, fy2, lddot2, fyA2, fBn2 = sol2
        x0_k2 = sol2.tolist()

        # ── k3 : même instant t+dt/2, mais état issu de k2 ──
        l3    = l    + (dt/2)*ldot   + (dt/2)**2/2 * lddot2
        ldot3 = ldot + (dt/2)*lddot2
        sol3  = solve_substep(t+dt/2,  l3,               ldot3,             x0_k3)
        fx3, fy3, lddot3, fyA3, fBn3 = sol3
        x0_k3 = sol3.tolist()

        # ── k4 : état à t+dt ──
        l4    = l    + dt*ldot       + dt**2/2 * lddot3
        ldot4 = ldot + dt*lddot3
        sol4  = solve_substep(t+dt,    l4,               ldot4,             x0_k4)
        fx4, fy4, lddot4, fyA4, fBn4 = sol4
        x0_k4 = sol4.tolist()

        # ── Intégration RK4 ──
        l_new    = l    + (dt/6)*(ldot  + 2*ldot2  + 2*ldot3  + ldot4)
        ldot_new = ldot + (dt/6)*(lddot1 + 2*lddot2 + 2*lddot3 + lddot4)

        # ── Grandeurs à enregistrer (depuis k1) ──
        ev = extras_fn(ca.DM(sol1), ca.DM([t, l, ldot]))
        delta_v = float(ev[0])
        psi_v   = float(ev[1])
        rAO_v   = np.array(ev[2].full()).squeeze()
        rAB_v   = np.array(ev[3].full()).squeeze()

        hist["t"].append(t)
        hist["l"].append(l)
        hist["ldot"].append(ldot)
        hist["lddot"].append(lddot1)
        hist["psi"].append(psi_v)
        hist["delta"].append(delta_v)
        hist["fx"].append(fx1)
        hist["fy"].append(fy1)
        hist["fyA"].append(fyA1)
        hist["fBn"].append(fBn1)
        hist["xO"].append(l + rAO_v[0])
        hist["yO"].append(rAO_v[1])
        hist["xB"].append(l + rAB_v[0])
        hist["yB"].append(rAB_v[1])

        # ── Avancer l'état ──
        l    = l_new
        ldot = ldot_new

        if l < 0.0:
            l = 0.0
            if ldot < 0.0: ldot = 0.0
        if l > b:
            l = b
            if ldot > 0.0: ldot = 0.0

        if k % 500 == 0:
            print(f"  t={t:.2f}s  l={l:.4f}m  ldot={ldot:.4f}m/s  "
                  f"fx={fx1:.2f}N  fy={fy1:.2f}N  "
                  f"fyA={fyA1:.3f}  fBn={fBn1:.3f}")

    print(f"\n── Résultats ────────────────────────────")
    print(f"l final    = {hist['l'][-1]:.4f} m   (cible = {b} m)")
    print(f"ldot final = {hist['ldot'][-1]:.4f} m/s (cible = 0)")
    print(f"Min fyA    = {min(hist['fyA']):.4f} N  (doit être >= 0)")
    print(f"Min fBn    = {min(hist['fBn']):.4f} N  (doit être >= 0)")

    # ══════════════════════════════════════════════════════════════════
    # PLOTS
    # ══════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.patch.set_facecolor('#0d0f14')

    def style(ax):
        ax.set_facecolor('#13161e')
        ax.tick_params(colors='#8899bb')
        ax.grid(True, alpha=0.2, color='#232840')
        for sp in ax.spines.values(): sp.set_edgecolor('#232840')
        ax.xaxis.label.set_color('#8899bb')
        ax.yaxis.label.set_color('#8899bb')
        ax.title.set_color('#c8d4f0')

    for ax in axes.flat: style(ax)

    t_arr = np.array(hist["t"])
    s_ref = np.clip(t_arr / T_sim, 0, 1)
    l_ref = b * (3*s_ref**2 - 2*s_ref**3)

    axes[0,0].plot(t_arr, hist["l"],  color='#5ee7ff', lw=2,   label='l(t)')
    axes[0,0].plot(t_arr, l_ref,      color='#56f0a0', lw=1.5, ls='--', label='l_ref')
    axes[0,0].axhline(b, color='#ffb347', lw=1, ls=':', alpha=0.6)
    axes[0,0].set_title('l(t)'); axes[0,0].set_ylabel('l (m)'); axes[0,0].set_xlabel('t (s)')
    axes[0,0].legend(fontsize=8, facecolor='#13161e', labelcolor='#c8d4f0')

    axes[0,1].plot(t_arr, hist["ldot"], color='#b57aff', lw=2)
    axes[0,1].axhline(0, color='#3a4560', lw=1)
    axes[0,1].set_title('ḷ(t)'); axes[0,1].set_ylabel('ḷ (m/s)'); axes[0,1].set_xlabel('t (s)')

    axes[0,2].plot(t_arr, hist["fyA"], color='#56f0a0', lw=1.8, label='fyA')
    axes[0,2].plot(t_arr, hist["fBn"], color='#ffb347', lw=1.8, label='fBn')
    axes[0,2].axhline(0, color='#ff5f6d', lw=1, ls='--', alpha=0.8)
    axes[0,2].set_title('Forces de contact')
    axes[0,2].set_ylabel('N'); axes[0,2].set_xlabel('t (s)')
    axes[0,2].legend(fontsize=8, facecolor='#13161e', labelcolor='#c8d4f0')

    axes[1,0].plot(t_arr, hist["fx"], color='#5ee7ff', lw=1.8, label='fx')
    axes[1,0].plot(t_arr, hist["fy"], color='#ff5f6d', lw=1.8, label='fy')
    axes[1,0].axhline(0, color='#3a4560', lw=1)
    axes[1,0].set_title('Commande [fx, fy]')
    axes[1,0].set_ylabel('N'); axes[1,0].set_xlabel('t (s)')
    axes[1,0].legend(fontsize=8, facecolor='#13161e', labelcolor='#c8d4f0')

    axes[1,1].plot(t_arr, np.degrees(hist["psi"]),   color='#ffb347', lw=1.8, label='ψ')
    axes[1,1].plot(t_arr, np.degrees(hist["delta"]), color='#56f0a0', lw=1.8, label='δ')
    axes[1,1].set_title('Angles'); axes[1,1].set_ylabel('°'); axes[1,1].set_xlabel('t (s)')
    axes[1,1].legend(fontsize=8, facecolor='#13161e', labelcolor='#c8d4f0')

    axes[1,2].plot(hist["l"], hist["ldot"], color='#5ee7ff', lw=1.8)
    axes[1,2].scatter([hist["l"][0]],  [hist["ldot"][0]],  color='#56f0a0', s=60, zorder=5, label='début')
    axes[1,2].scatter([hist["l"][-1]], [hist["ldot"][-1]], color='#ff5f6d', s=60, zorder=5, label='fin')
    axes[1,2].set_title('Portrait de phase')
    axes[1,2].set_xlabel('l (m)'); axes[1,2].set_ylabel('ḷ (m/s)')
    axes[1,2].legend(fontsize=8, facecolor='#13161e', labelcolor='#c8d4f0')

    plt.suptitle('MPC + RK4 — NLP résolu aux 4 sous-pas',
                 color='#c8d4f0', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('mpc_rk4_results.png', dpi=150, bbox_inches='tight')
    plt.show()

    # ══════════════════════════════════════════════════════════════════
    # ANIMATION DU MÉCANISME
    # ══════════════════════════════════════════════════════════════════
    from matplotlib.animation import FuncAnimation

    skip   = 10
    frames = np.arange(0, len(hist["t"]), skip)

    corners_body = np.array([[0, 0], [a, 0], [a, b], [0, b], [0, 0]])

    PSI_SWITCH = 3 * np.pi / 4

    fig_anim, ax_a = plt.subplots(figsize=(7, 7))
    fig_anim.patch.set_facecolor('#0d0f14')
    ax_a.set_facecolor('#13161e')
    ax_a.set_xlim(-0.15, 0.75)
    ax_a.set_ylim(-0.15, 0.65)
    ax_a.set_aspect('equal')
    ax_a.tick_params(colors='#8899bb')
    ax_a.grid(True, alpha=0.2, color='#232840')
    for sp in ax_a.spines.values():
        sp.set_edgecolor('#232840')
    ax_a.xaxis.label.set_color('#8899bb')
    ax_a.yaxis.label.set_color('#8899bb')
    ax_a.set_title('Animation du mécanisme (MPC + RK4)', color='#c8d4f0', fontsize=13)

    # Sol et pivot fixes
    ax_a.plot([0, 0.6], [0, 0], color='#5ee7ff', lw=2.5, label='Sol (mur 1)')
    ax_a.plot(0, 0, 'o', color='#8899bb', ms=6)

    wall_line,  = ax_a.plot([], [], color='#ffb347', lw=2.5, label='Mur 2 (ψ)')
    body_patch, = ax_a.plot([], [], color='#c8d4f0', lw=2.0, label='Corps',
                             solid_capstyle='round')
    traj_line,  = ax_a.plot([], [], color='#ffb347', lw=1.0, alpha=0.4, ls='--')
    force_line, = ax_a.plot([], [], color='#b57aff', lw=2.0, label='Force [fx,fy]')
    point_A,    = ax_a.plot([], [], 'o', color='#ff5f6d', ms=8, label='A', zorder=6)
    point_B,    = ax_a.plot([], [], 'o', color='#56f0a0', ms=8, label='B', zorder=6)
    point_O,    = ax_a.plot([], [], 'o', color='#ffb347', ms=7, label='O (COM)', zorder=6)
    info_txt    = ax_a.text(0.02, 0.97, '', transform=ax_a.transAxes,
                            color='#c8d4f0', fontsize=9, va='top',
                            fontfamily='monospace',
                            bbox=dict(facecolor='#13161e', alpha=0.7,
                                      edgecolor='#232840'))
    ax_a.legend(loc='upper right', fontsize=8,
                facecolor='#13161e', labelcolor='#c8d4f0', edgecolor='#232840')

    traj_x, traj_y = [], []

    def update(fi):
        psi_v = hist["psi"][fi]
        d_v   = hist["delta"][fi]
        l_v   = hist["l"][fi]
        t_v   = hist["t"][fi]
        fx_v  = hist["fx"][fi]
        fy_v  = hist["fy"][fi]
        fBn_v = hist["fBn"][fi]
        xO_v  = hist["xO"][fi];  yO_v = hist["yO"][fi]
        xB_v  = hist["xB"][fi];  yB_v = hist["yB"][fi]

        # Mur 2
        L2 = 0.6
        wall_line.set_data([0, L2*np.cos(psi_v)], [0, L2*np.sin(psi_v)])

        # Corps
        R = np.array([[np.cos(d_v), -np.sin(d_v)],
                      [np.sin(d_v),  np.cos(d_v)]])
        world = np.array([[l_v, 0.0] + R @ c for c in corners_body])
        body_patch.set_data(world[:, 0], world[:, 1])

        # Points
        point_A.set_data([l_v],  [0.0])
        point_B.set_data([xB_v], [yB_v])
        point_O.set_data([xO_v], [yO_v])

        # Flèche de force (scale 0.02 m/N)
        scale = 0.02
        force_line.set_data([xO_v, xO_v + fx_v*scale],
                            [yO_v, yO_v + fy_v*scale])

        # Trace du COM
        traj_x.append(xO_v); traj_y.append(yO_v)
        traj_line.set_data(traj_x, traj_y)

        phase = "Phase 1 : f = −nB" if psi_v < PSI_SWITCH else "Phase 2 : glissement"
        contact_B = "CONTACT" if fBn_v > 0.1 else "décollé !"
        info_txt.set_text(
            f"t   = {t_v:.2f} s\n"
            f"ψ   = {np.degrees(psi_v):.1f}°\n"
            f"δ   = {np.degrees(d_v):.1f}°\n"
            f"l   = {l_v:.3f} m\n"
            f"fx  = {fx_v:.1f} N\n"
            f"fy  = {fy_v:.1f} N\n"
            f"fBn = {fBn_v:.1f} N  [{contact_B}]\n"
            f"{phase}"
        )
        return (wall_line, body_patch, traj_line, force_line,
                point_A, point_B, point_O, info_txt)

    ani = FuncAnimation(fig_anim, update, frames=frames,
                        interval=int(1000 * dt * skip), blit=False)
    plt.tight_layout()
    plt.show()