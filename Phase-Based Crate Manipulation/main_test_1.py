"""
main_single.py — Simulation phase-based de la manipulation du bac.

Vue du dessus. Le bac est suspendu par un câble de grue accroché au CoM.
La commande est une force appliquée au CoM dans le repère monde.

Phases :
  APPROACH   : bac en vol libre, pas de force — observation caméra
  PUSH       : force constante vers mur 1, rebonds successifs
               → convergence vers minimum d énergie potentielle (côté plat)
  COINCEMENT : côté plat sur mur 1, glissement vers mur 2
  PIVOTEMENT : A sur mur 1, B sur mur 2, pivotement 90 deg
  GLISSEMENT : côté long sur mur 1, glissement le long des murs
  FINAL      : mur 2 se referme, retour au minimum en x
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from enum import Enum, auto
from dataclasses import dataclass


# ══════════════════════════════════════════════════════════════════════════════
# 1. STATE
# ══════════════════════════════════════════════════════════════════════════════

class Mode(Enum):
    APPROACH    = auto()
    PUSH        = auto()
    COINCEMENT  = auto()
    PIVOTEMENT  = auto()
    STABILISATION = auto()   # contact B-mur2 etabli, attendre stabilisation
    GLISSEMENT  = auto()
    FINAL       = auto()


@dataclass
class CrateState:
    x:     float
    y:     float
    theta: float
    vx:    float
    vy:    float
    omega: float
    mode:  Mode
    pivot_x:   float = 0.0
    pivot_y:   float = 0.0
    pivot_idx: int   = 0

    x_pivot_mur2:  float = 0.0
    stab_counter:  int   = 0
    l:             float = 0.0   # coord generalisee PIVOTEMENT (xA sur mur 1)
    ldot:          float = 0.0


@dataclass
class SimParams:
    m:  float = 7.0
    a:  float = 0.3
    b:  float = 0.4
    mu: float = 0.3

    dt:         float = 0.002
    total_time: float = 15.0

    # PUSH
    F_approach:    float = 3.0
    e_restitution: float = 0.4
    damping_omega: float = 0.8
    side_flat_tol: float = 0.012

    # COINCEMENT
    F_coincement:  float = 1.0
    contact_tol:   float = 0.015
    stab_steps:    int   = 200    # nombre de timesteps d attente apres contact B-mur2

    # PIVOTEMENT
    F_pivot:       float = 8.0
    pivot_bias:    float = 0.5

    # GLISSEMENT
    F_slide:       float = 1.0
    l_slide_end:   float = 0.35

    # FINAL
    l_final_end:   float = 0.05

    # Conditions initiales
    x_init:        float = 0.45
    y_init:        float = 0.50
    theta_min:     float = 0.0
    theta_max:     float = 2*np.pi
    omega_min:     float = 0.0
    omega_max:     float = 0.2

    # Mur 2 : psi va de pi/2 (vertical) a pi (apres 90 deg anti-horaire)
    T_wall:        float = 6.0
    psi_start:     float = np.pi / 2
    psi_end:       float = np.pi          # 90 deg anti-horaire : vertical -> horizontal
    wall_t1_ratio: float = 0.3
    wall_t2_ratio: float = 0.7

    @property
    def corners_body(self) -> np.ndarray:
        ha, hb = self.a/2.0, self.b/2.0
        return np.array([[-ha,-hb],[+ha,-hb],[+ha,+hb],[-ha,+hb]], dtype=float)

    @property
    def I_G(self) -> float:
        return (self.m/12.0) * (self.a**2 + self.b**2)


def rot2(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c,-s],[s,c]])


def get_corners_world(state: CrateState, params: SimParams) -> np.ndarray:
    R   = rot2(state.theta)
    com = np.array([state.x, state.y])
    return np.array([com + R @ c for c in params.corners_body])


def init_state(params: SimParams, rng=None) -> CrateState:
    if rng is None:
        rng = np.random.default_rng()
    theta = np.radians(100.0)  # theta fixe a 100 deg pour les tests
    omega = rng.uniform(params.omega_min, params.omega_max) * rng.choice([-1.0, 1.0])
    state = CrateState(x=params.x_init, y=params.y_init,
                       theta=theta, vx=0.0, vy=0.0,
                       omega=omega, mode=Mode.APPROACH)
    corners = get_corners_world(state, params)
    min_y   = corners[:,1].min()
    if min_y < 0.05:
        state.y += (0.05 - min_y)
    return state


# ══════════════════════════════════════════════════════════════════════════════
# 2. DYNAMICS
# ══════════════════════════════════════════════════════════════════════════════

def wall_profile(t: float, params: SimParams, t_pivot_start: float = -1.0):
    """
    Profil trapezodal du mur 2.
    t_pivot_start : instant ou le pivotement demarre (debut phase PIVOTEMENT).
                    Si negatif, le mur ne tourne pas encore (reste a psi_start).
    """
    if t_pivot_start < 0 or t < t_pivot_start:
        return params.psi_start, 0.0, 0.0

    t_rel = t - t_pivot_start   # temps relatif depuis debut pivotement
    p0, p1 = params.psi_start, params.psi_end
    dpsi   = p1 - p0
    T      = params.T_wall
    t1, t2 = T*params.wall_t1_ratio, T*params.wall_t2_ratio
    acc    = dpsi / (0.5*t1**2 + (t2-t1)*t1 + 0.5*(T-t2)**2)
    v1     = acc*t1
    p_1    = p0 + 0.5*acc*t1**2
    p_2    = p_1 + v1*(t2-t1)
    if t_rel <= t1:
        return p0+0.5*acc*t_rel**2, acc*t_rel, acc
    elif t_rel <= t2:
        return p_1+v1*(t_rel-t1), v1, 0.0
    elif t_rel <= T:
        return p_2+v1*(t_rel-t2)-0.5*acc*(t_rel-t2)**2, v1-acc*(t_rel-t2), -acc
    else:
        return p1, 0.0, 0.0


def push_step(state: CrateState, fx: float, fy: float,
              params: SimParams) -> None:
    """
    Euler symplectique — corps libre sous force au CoM.

    Newton : m*ax=fx, m*ay=fy, tau=0 (force au CoM).
    Amortissement sur omega : I_G * d(omega)/dt = -damping * omega
    Simule la dissipation du cable — pas de calcul exact de tension.
    """
    h = params.dt
    state.vx    += (fx / params.m) * h
    state.vy    += (fy / params.m) * h
    state.x     += state.vx * h
    state.y     += state.vy * h
    alpha        = -params.damping_omega * state.omega / params.I_G
    state.omega += alpha * h
    state.theta += state.omega * h


def apply_impact(state: CrateState, corner_idx: int,
                 params: SimParams) -> None:
    """
    Impulsion instantanee quand coin touche mur 1 (y=0).

    J = -(1+e) * v_corner_y / (1/m + rx^2/I_G)
    Delta_vy    = J/m
    Delta_omega = rx*J/I_G

    On ne calcule pas la reaction normale exacte.
    Apres impact : bac reste en PUSH (rebond libre).
    """
    R        = rot2(state.theta)
    r_corner = R @ params.corners_body[corner_idx]
    rx       = r_corner[0]

    v_corner_y = state.vy + state.omega * rx
    if v_corner_y >= 0.0:
        return

    e   = params.e_restitution
    J   = -(1.0+e)*v_corner_y / (1.0/params.m + rx**2/params.I_G)
    state.vy    += J / params.m
    state.omega += rx * J / params.I_G
    corners      = get_corners_world(state, params)
    state.y     -= corners[corner_idx, 1]


def coincement_step(state: CrateState, fx: float, fy: float,
                    params: SimParams,
                    t_pivot_start: float = -1.0) -> None:
    """
    COINCEMENT : theta fixe, glissement en x sous la force de commande.
    Contraintes : vy=0, omega=0 (cote plat sur mur 1).
    Contrainte mur 2 : aucun coin ne peut passer x < 0 (mur 2 vertical).
    """
    state.vx    += (fx / params.m) * params.dt
    state.x     += state.vx * params.dt
    state.vy     = 0.0
    state.omega  = 0.0

    # Contrainte mur 2 (vertical, x=0) : empecher penetration
    corners = get_corners_world(state, params)
    min_x   = corners[:, 0].min()
    if min_x < 0.0:
        state.x  -= min_x   # ramener le coin le plus gauche a x=0
        state.vx  = 0.0     # stopper le glissement




def delta_kinematics(l: float, ldot: float, psi: float, psidot: float,
                     params):
    """
    Contrainte geometrique double contact (notre geometrie) :
      A = coin pivot sur mur 1, xA = l
      B = coin adjacent (cote court a) sur mur 2
      r_AB_corps = (a, 0)

    Contrainte : sin(psi - theta) = l*sin(psi)/a
    -> theta = psi - arcsin(l*sin(psi)/a)
    """
    a   = params.a
    eps = 1e-9
    u   = np.clip(l * np.sin(psi) / a, -1+eps, 1-eps)
    k   = 1.0 / np.sqrt(max(eps, 1 - u*u))
    k3  = k**3

    delta        = psi - np.arcsin(u)
    u_l          =  np.sin(psi) / a
    u_psi        =  l * np.cos(psi) / a
    u_lpsi       =  np.cos(psi) / a
    u_psipsi     = -l * np.sin(psi) / a
    delta_l      = -k * u_l
    delta_psi    =  1 - k * u_psi
    delta_ll     = -u * k3 * u_l * u_l
    delta_lpsi   = -k * u_lpsi   - u * k3 * u_l   * u_psi
    delta_psipsi = -k * u_psipsi - u * k3 * u_psi * u_psi
    delta_dot    =  delta_l * ldot + delta_psi * psidot

    return delta, delta_dot, delta_l, delta_psi, delta_ll, delta_lpsi, delta_psipsi


def lagrange_1dof(l, ldot, fx, fy, t, params, t_pivot_start):
    """
    EOM Lagrangien 1-DDL — phase PIVOTEMENT.
    Coordonnee generalisee : l = xA (position de A sur mur 1).
    rAO_corps = (a/2, -b/2)  [A=coin3=(-a/2,b/2), CoM=(0,0)]
    M_eff * lddot = Q_l - f_nl
    """
    psi, psidot, psiddot = wall_profile(t, params, t_pivot_start)
    (delta, delta_dot, delta_l, delta_psi,
     delta_ll, delta_lpsi, delta_psipsi) = delta_kinematics(l, ldot, psi, psidot, params)

    c, s = np.cos(delta), np.sin(delta)
    rAO  = np.array([params.a/2.0, -params.b/2.0])
    Rp_r = np.array([-s*rAO[0] - c*rAO[1],
                      c*rAO[0] - s*rAO[1]])
    R_r  = np.array([ c*rAO[0] - s*rAO[1],
                      s*rAO[0] + c*rAO[1]])

    axc = 1.0 + Rp_r[0] * delta_l
    ayc =       Rp_r[1] * delta_l

    nl_ddot = (delta_ll * ldot**2
               + 2.0 * delta_lpsi * ldot * psidot
               + delta_psipsi * psidot**2
               + delta_psi * psiddot)
    ax_nl = Rp_r[0] * nl_ddot - R_r[0] * delta_dot**2
    ay_nl = Rp_r[1] * nl_ddot - R_r[1] * delta_dot**2

    I_G   = params.I_G
    M_eff = params.m * (axc**2 + ayc**2) + I_G * delta_l**2
    Q_l   = fx * axc + fy * ayc
    f_nl  = params.m * (ax_nl * axc + ay_nl * ayc) + I_G * nl_ddot * delta_l
    lddot = (Q_l - f_nl) / max(M_eff, 1e-10)
    return ldot, lddot


def pivotement_lagrange_step(state, fx, fy, t, params, t_pivot_start):
    """
    RK4 sur (l, ldot). Reconstruit (x, y, theta) depuis la contrainte.
    """
    h = params.dt
    l, ldot = state.l, state.ldot

    def f(l_, ld_, t_):
        return lagrange_1dof(l_, ld_, fx, fy, t_, params, t_pivot_start)

    d1,a1 = f(l,          ldot,       t)
    d2,a2 = f(l+h/2*d1,  ldot+h/2*a1, t+h/2)
    d3,a3 = f(l+h/2*d2,  ldot+h/2*a2, t+h/2)
    d4,a4 = f(l+h*d3,    ldot+h*a3,   t+h)

    state.l    = np.clip(l    + (h/6)*(d1+2*d2+2*d3+d4), 0.0, params.a)
    state.ldot =         ldot + (h/6)*(a1+2*a2+2*a3+a4)

    psi, psidot, _ = wall_profile(t, params, t_pivot_start)
    delta, *_ = delta_kinematics(state.l, state.ldot, psi, psidot, params)
    state.theta = delta

    rAO = np.array([params.a/2.0, -params.b/2.0])
    c, s = np.cos(delta), np.sin(delta)
    state.x = state.l + c*rAO[0] - s*rAO[1]
    state.y =           s*rAO[0] + c*rAO[1]


def enforce_wall_constraints(state: CrateState, params: SimParams,
                              t: float = 0.0,
                              t_pivot_start: float = -1.0) -> None:
    """
    Contraintes de non-penetration pour tous les coins.

    Mur 1 : y = 0  (fixe)
    Mur 2 : angle psi (tourne pendant PIVOTEMENT)

    Pour chaque violation : translation du CoM + annulation
    de la composante de vitesse vers le mur.
    Le bac est libre — il suit le mur parce que la force l y pousse
    et parce qu il ne peut pas traverser.
    """
    psi, _, _ = wall_profile(t, params, t_pivot_start)

    # ── Mur 1 : y >= 0 ───────────────────────────────────────────────────────
    corners = get_corners_world(state, params)
    min_y   = corners[:, 1].min()
    if min_y < 0.0:
        state.y -= min_y
        if state.vy < 0.0:
            state.vy = 0.0

    # ── Mur 2 : dist_to_wall2 >= 0 ───────────────────────────────────────────
    # Normale interne (vers l interieur, cote bac)
    nB_in_x =  np.sin(psi)
    nB_in_y = -np.cos(psi)

    corners = get_corners_world(state, params)
    dists   = np.array([dist_to_wall2(cx, cy, psi) for cx,cy in corners])
    min_d   = dists.min()
    if min_d < 0.0:
        # Translater le CoM pour sortir le coin le plus penetrant
        state.x -= min_d * nB_in_x
        state.y -= min_d * nB_in_y
        # Annuler la composante de vitesse qui va vers mur 2
        v_toward = state.vx * nB_in_x + state.vy * nB_in_y
        if v_toward < 0.0:
            state.vx -= v_toward * nB_in_x
            state.vy -= v_toward * nB_in_y

def pivotement_step(state: CrateState, fx: float, fy: float,
                    params: SimParams) -> None:
    """
    PIVOTEMENT — Lagrangien autour du pivot A.

    A = coin le plus proche du coin (0,0), recalcule a chaque step
    depuis l etat courant. Pas de position hardcodee.

    I_A * alpha = tau = r_AO x F
    puis reconstruction de (x,y) du CoM depuis A.
    Contrainte : A ne peut pas penetrer les murs (x>=0, y>=0).
    """
    h = params.dt

    # Identifier A depuis l etat courant
    corners = get_corners_world(state, params)
    A_idx   = int(np.argmin([cx**2 + cy**2 for cx,cy in corners]))
    xA      = max(0.0, corners[A_idx, 0])   # contraint a x>=0
    yA      = max(0.0, corners[A_idx, 1])   # contraint a y>=0

    # r_AO (A -> CoM) dans le repere monde
    rx = state.x - xA
    ry = state.y - yA

    # Moment d inertie autour de A (axes paralleles)
    I_A   = params.I_G + params.m * (rx**2 + ry**2)

    # Couple autour de A
    tau   = rx * fy - ry * fx
    alpha = tau / I_A

    # Integration omega et theta
    state.omega += alpha * h
    state.theta += state.omega * h

    # Reconstruction CoM depuis A (fixe sur les murs)
    bc    = params.corners_body[A_idx]
    r_b   = np.array([-bc[0], -bc[1]])   # A->CoM en corps
    c, s  = np.cos(state.theta), np.sin(state.theta)
    rx_n  = c*r_b[0] - s*r_b[1]
    ry_n  = s*r_b[0] + c*r_b[1]
    state.x = xA + rx_n
    state.y = yA + ry_n

    # Vitesses coherentes
    state.vx = -state.omega * ry_n
    state.vy =  state.omega * rx_n


# ══════════════════════════════════════════════════════════════════════════════
# 3. MONITORS
# ══════════════════════════════════════════════════════════════════════════════

def dist_to_wall2(x: float, y: float, psi: float) -> float:
    return x*np.sin(psi) - y*np.cos(psi)


def lowest_corner(state: CrateState, params: SimParams) -> tuple:
    corners = get_corners_world(state, params)
    idx     = int(np.argmin(corners[:,1]))
    return idx, float(corners[idx,1])


def side_flat_on_wall1(state: CrateState, params: SimParams) -> bool:
    corners = get_corners_world(state, params)
    y       = corners[:,1]
    tol     = params.side_flat_tol
    for i in range(4):
        j = (i+1) % 4
        if y[i] <= tol and y[j] <= tol:
            return True
    return False


def corner_B_on_wall2(state: CrateState, t: float,
                      params: SimParams,
                      t_pivot_start: float = -1.0) -> bool:
    """
    Detecte qu un coin du bac touche mur 2.
    On cherche le coin le plus proche de mur 2 (pas forcement coin 3).
    Le coin de contact devient le pivot B pour la phase PIVOTEMENT.
    """
    psi, _, _ = wall_profile(t, params, t_pivot_start)
    corners   = get_corners_world(state, params)
    dists     = [dist_to_wall2(cx, cy, psi) for cx, cy in corners]
    min_dist  = min(dists)
    return min_dist < params.contact_tol


def check_transition(state: CrateState, t: float, params: SimParams,
                     t_pivot_start: float = -1.0,
                     theta_piv_start: float = 0.0):
    if state.mode == Mode.APPROACH:
        return Mode.PUSH
    elif state.mode == Mode.PUSH:
        if side_flat_on_wall1(state, params):
            return Mode.COINCEMENT
    elif state.mode == Mode.COINCEMENT:
        if corner_B_on_wall2(state, t, params, t_pivot_start):
            return Mode.STABILISATION

    elif state.mode == Mode.STABILISATION:
        # Attendre N timesteps apres le contact B-mur2
        state.stab_counter += 1
        if state.stab_counter >= params.stab_steps:
            return Mode.PIVOTEMENT

    elif state.mode == Mode.PIVOTEMENT:
        delta_theta = abs(state.theta - theta_piv_start)
        delta_theta = delta_theta % (2*np.pi)
        if delta_theta > np.pi:
            delta_theta = 2*np.pi - delta_theta
        if delta_theta >= np.radians(88):
            return Mode.GLISSEMENT

    elif state.mode == Mode.GLISSEMENT:
        # Fin quand le coin inférieur-gauche a dépassé x=0 de plus de 5cm.
        # Le coin inférieur-gauche = coin avec le plus petit x parmi les 4.
        corners = get_corners_world(state, params)
        x_min_corner = float(corners[:, 0].min())
        if x_min_corner > 0.05:
            return None   # simulation terminée

    return state.mode


# ══════════════════════════════════════════════════════════════════════════════
# 4. CONTROLLER
# ══════════════════════════════════════════════════════════════════════════════

def get_command(state: CrateState, t: float,
                params: SimParams,
                t_pivot_start: float = -1.0) -> tuple:
    if state.mode == Mode.APPROACH:
        return 0.0, 0.0

    elif state.mode == Mode.PUSH:
        return params.F_approach * 0.1, -params.F_approach

    elif state.mode == Mode.COINCEMENT:
        psi, _, _ = wall_profile(t, params, t_pivot_start)
        fx = -params.F_coincement * np.sin(psi)
        fy = -params.F_coincement * 0.3
        return fx, fy

    elif state.mode == Mode.PIVOTEMENT:
        psi, _, _ = wall_profile(t, params, t_pivot_start)
        nB_out_x = -np.sin(psi)
        nB_out_y =  np.cos(psi)
        corners  = get_corners_world(state, params)
        A_idx    = int(np.argmin([cx**2+cy**2 for cx,cy in corners]))
        C_idx    = (A_idx + 2) % 4
        xA, yA   = corners[A_idx]
        xC, yC   = corners[C_idx]
        CAx = xA-xC; CAy = yA-yC
        mag = np.hypot(CAx, CAy); CAx/=mag; CAy/=mag
        rx = state.x - xA; ry = state.y - yA
        r_mag = np.hypot(rx, ry)
        if r_mag < 1e-9:
            return params.F_approach*nB_out_x, params.F_approach*nB_out_y
        bx = nB_out_x + CAx; by = nB_out_y + CAy
        mag = np.hypot(bx, by)
        if mag > 1e-9: bx/=mag; by/=mag
        tau = rx*by - ry*bx
        if tau <= 0.0:
            bx = -ry/r_mag; by = rx/r_mag
        tau_nB = rx*nB_out_y - ry*nB_out_x
        if tau_nB >= 0:
            bx2 = (1-params.pivot_bias)*CAx + params.pivot_bias*nB_out_x
            by2 = (1-params.pivot_bias)*CAy + params.pivot_bias*nB_out_y
            mag = np.hypot(bx2, by2)
            if mag > 1e-9: bx2/=mag; by2/=mag
            if rx*by2 - ry*bx2 > 0:
                bx, by = bx2, by2
        return params.F_approach*bx, params.F_approach*by

    elif state.mode == Mode.STABILISATION:
        return 0.0, 0.0   # pas de force, laisser le bac se stabiliser

    elif state.mode == Mode.GLISSEMENT:
        return params.F_slide, -params.F_slide * 0.2

    return 0.0, 0.0

# ══════════════════════════════════════════════════════════════════════════════
# 5. SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

def simulate(params=None, seed=None):
    if params is None:
        params = SimParams()
    rng   = np.random.default_rng(seed)
    state = init_state(params, rng=rng)

    print(f"Init: theta={np.degrees(state.theta):.1f} deg  "
          f"omega={state.omega:.3f} rad/s  "
          f"CoM=({state.x:.3f},{state.y:.3f})")

    keys = ['t','x','y','theta','vx','vy','omega','fx','fy','mode','psi']
    hist = {k: [] for k in keys}
    N    = int(params.total_time / params.dt)

    # t_pivot_start : instant de declenchement du mur 2
    # Reste negatif jusqu'a la transition COINCEMENT -> PIVOTEMENT
    t_pivot_start   = -1.0
    theta_piv_start = 0.0   # theta au debut du pivotement

    for k in range(N+1):
        t = k * params.dt

        new_mode = check_transition(state, t, params, t_pivot_start, theta_piv_start)
        if new_mode is None:
            print(f"[t={t:.3f}s] Simulation terminee.")
            break
        if new_mode != state.mode:
            print(f"[t={t:.3f}s] {state.mode.name} -> {new_mode.name}")
            # Declencher le mur au moment de la transition vers PIVOTEMENT
            if new_mode == Mode.STABILISATION:
                state.stab_counter = 0
                print(f"  Contact B-mur2 etabli, attente {params.stab_steps} steps ({params.stab_steps*params.dt:.2f}s)...")

            if new_mode == Mode.GLISSEMENT:
                # Mémoriser la position x du pivot de mur 2 (= xA final)
                corners = get_corners_world(state, params)
                A_idx   = int(np.argmin([cx**2+cy**2 for cx,cy in corners]))
                state.x_pivot_mur2 = float(corners[A_idx, 0])
                # Vitesse initiale nulle pour démarrer proprement
                state.vx = 0.0; state.vy = 0.0; state.omega = 0.0
                print(f"  x_pivot_mur2={state.x_pivot_mur2:.4f}  cible x>{state.x_pivot_mur2+0.05:.4f}")

            if new_mode == Mode.PIVOTEMENT:
                t_pivot_start   = t
                theta_piv_start = state.theta
                corners  = get_corners_world(state, params)
                A_idx    = int(np.argmin([cx**2+cy**2 for cx,cy in corners]))
                state.pivot_x   = float(corners[A_idx, 0])
                state.pivot_y   = 0.0
                state.pivot_idx = A_idx

                # Initialiser l = a*sin(psi - theta) / sin(psi)
                psi0 = params.psi_start
                sin_psi0 = np.sin(psi0)
                if abs(sin_psi0) > 1e-6:
                    l0 = params.a * np.sin(psi0 - state.theta) / sin_psi0
                    state.l = float(np.clip(l0, 0.0, params.a))
                else:
                    state.l = 0.0
                state.ldot = 0.0
                # Absorber le choc : annuler les vitesses
                state.vx = 0.0; state.vy = 0.0; state.omega = 0.0

                print(f"  Pivot t={t:.3f}s  coin{A_idx} "
                      f"({state.pivot_x:.4f},{state.pivot_y:.4f})"
                      f"  theta={np.degrees(theta_piv_start):.1f} deg")
            state.mode = new_mode

        psi, _, _ = wall_profile(t, params, t_pivot_start)
        fx, fy    = get_command(state, t, params, t_pivot_start)

        for key, val in zip(keys, [t, state.x, state.y, state.theta,
                                    state.vx, state.vy, state.omega,
                                    fx, fy, state.mode.name, psi]):
            hist[key].append(val)

        if state.mode in (Mode.APPROACH, Mode.PUSH):
            push_step(state, fx, fy, params)
            idx, y_min = lowest_corner(state, params)
            if y_min <= 0.0:
                apply_impact(state, idx, params)
            enforce_wall_constraints(state, params, t, t_pivot_start)

        elif state.mode == Mode.COINCEMENT:
            coincement_step(state, fx, fy, params, t_pivot_start)
            enforce_wall_constraints(state, params, t, t_pivot_start)

        elif state.mode == Mode.PIVOTEMENT:
            pivotement_lagrange_step(state, fx, fy, t, params, t_pivot_start)

        elif state.mode == Mode.STABILISATION:
            # Meme dynamique que COINCEMENT mais on attend la stabilisation
            # Force nulle — on laisse le bac amortir naturellement
            coincement_step(state, 0.0, 0.0, params, t_pivot_start)
            # Impact sur mur 2 si B penetre (x_B < 0)
            psi_s, _, _ = wall_profile(t, params, t_pivot_start)
            corners_s = get_corners_world(state, params)
            for j,(cx,cy) in enumerate(corners_s):
                if dist_to_wall2(cx, cy, psi_s) < 0:
                    # Annuler la composante de vitesse vers mur 2
                    state.vx = max(0.0, state.vx)
                    break

        elif state.mode == Mode.GLISSEMENT:
            push_step(state, fx, fy, params)
            # Mur 1 seulement (y>=0) — pas de contrainte mur 2 en glissement
            corners = get_corners_world(state, params)
            min_y = corners[:, 1].min()
            if min_y < 0.0:
                state.y -= min_y
                if state.vy < 0.0:
                    state.vy = 0.0

    return hist, params


# ══════════════════════════════════════════════════════════════════════════════
# 6. VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

_COLORS = {
    'APPROACH':'#888780','PUSH':'#378ADD','COINCEMENT':'#1D9E75',
    'STABILISATION':'#9FE1CB',
    'PIVOTEMENT':'#7F77DD','GLISSEMENT':'#D85A30','FINAL':'#BA7517',
}


def plot_results(hist, params):
    t    = hist['t']
    cols = [_COLORS.get(m,'gray') for m in hist['mode']]

    def cplot(ax, data, ylabel, title):
        for i in range(len(t)-1):
            ax.plot(t[i:i+2], data[i:i+2], color=cols[i], lw=1.5)
        ax.axhline(0, color='gray', lw=0.7)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('t (s)'); ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    fig.suptitle('Phase-based crate simulation', fontsize=12)
    cplot(axes[0,0], hist['y'],   'y (m)',   'CoM y')
    cplot(axes[0,1], hist['x'],   'x (m)',   'CoM x')
    cplot(axes[0,2], [np.degrees(v) for v in hist['theta']], 'theta (deg)', 'Orientation')
    cplot(axes[1,0], hist['vy'],  'vy (m/s)','vy')
    cplot(axes[1,1], [np.degrees(v) for v in hist['omega']], 'omega (deg/s)','omega')
    cplot(axes[1,2], hist['fy'],  'fy (N)',  'Commande fy')

    from matplotlib.patches import Patch
    patches = [Patch(color=v, label=k) for k,v in _COLORS.items()
               if k in set(hist['mode'])]
    fig.legend(handles=patches, loc='upper right', fontsize=8, ncol=2)
    plt.tight_layout(); plt.show()


def animate(hist, params):
    skip   = max(1, int(0.025/params.dt))
    frames = list(range(0, len(hist['t']), skip))
    fig, ax = plt.subplots(figsize=(7,7))
    ax.set_xlim(-0.15, 0.7); ax.set_ylim(-0.15, 0.7)
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    ax.set_title('Crate manipulation')
    ax.plot([-0.15,0.7],[0,0],'k',lw=3)
    wall2_ln, = ax.plot([],[],'b-',lw=3)
    body_ln,  = ax.plot([],[],'b-',lw=2)
    com_pt,   = ax.plot([],'k^',ms=7)
    force_ln, = ax.plot([],'r-',lw=2)
    traj_ln,  = ax.plot([],'b--',lw=1,alpha=0.4)
    info      = ax.text(0.02,0.98,'',transform=ax.transAxes,fontsize=9,
                        va='top',fontfamily='monospace',
                        bbox=dict(facecolor='white',alpha=0.78,edgecolor='gray'))
    tx, ty = [], []

    def update(fi):
        i   = frames[fi]
        t_i = hist['t'][i]
        col = _COLORS.get(hist['mode'][i],'gray')
        # Pour l animation, on recalcule t_pivot_start depuis l historique
        psi_i = hist['psi'][i]
        wall2_ln.set_data([0,0.6*np.cos(psi_i)],[0,0.6*np.sin(psi_i)])
        fake = CrateState(x=hist['x'][i],y=hist['y'][i],theta=hist['theta'][i],
                          vx=0,vy=0,omega=0,mode=Mode.APPROACH)
        c4 = get_corners_world(fake, params)
        c5 = np.vstack([c4,c4[0]])
        body_ln.set_data(c5[:,0],c5[:,1]); body_ln.set_color(col)
        xO,yO = hist['x'][i],hist['y'][i]
        com_pt.set_data([xO],[yO])
        sc=0.012
        force_ln.set_data([xO,xO+hist['fx'][i]*sc],[yO,yO+hist['fy'][i]*sc])
        tx.append(xO); ty.append(yO)
        traj_ln.set_data(tx,ty)
        info.set_text(f"t={t_i:.3f}s  {hist['mode'][i]}\n"
                      f"theta={np.degrees(hist['theta'][i]):.1f} deg\n"
                      f"omega={np.degrees(hist['omega'][i]):.1f} deg/s\n"
                      f"y={hist['y'][i]:.4f}m")
        return wall2_ln,body_ln,com_pt,force_ln,traj_ln,info

    ani = FuncAnimation(fig,update,frames=len(frames),
                        interval=int(1000*params.dt*skip),blit=False)
    plt.tight_layout(); plt.show()
    return ani


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--seed',    type=int, default=None)  # None = seed aleatoire
    p.add_argument('--no-anim',action='store_true')
    args = p.parse_args()

    params = SimParams()
    hist, params = simulate(params, seed=args.seed)
    modes = []
    prev  = None
    for m in hist['mode']:
        if m != prev: modes.append(m); prev=m
    print(f"Phases : {' -> '.join(modes)}")
    print(f"theta  : {np.degrees(hist['theta'][-1]):.1f} deg")
    print(f"x      : {hist['x'][-1]:.4f} m")
    print(f"y      : {hist['y'][-1]:.4f} m")
    plot_results(hist, params)
    if not args.no_anim:
        ani = animate(hist, params)
