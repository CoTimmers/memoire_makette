"""
simulation.py — Main simulation loop and visualisation.

Usage
-----
    python simulation.py

The simulation starts with the box in PIVOTEMENT (A on wall 1 at l=0,
B on wall 2) which matches the initial condition of moteur_dynamique_discrétisee.py.
APPROACH and COINCEMENT are included as phases but the box is initialised
directly at the wedged position for validation purposes.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from state import CrateState, SimParams, Mode
from dynamics import f_continuous, integrate_step
from controller import get_command
from monitors import check_transition, get_box_geometry


# ── Core simulation ────────────────────────────────────────────────────────────

def simulate(params: SimParams | None = None, initial_mode: Mode = Mode.PIVOTEMENT):
    """
    Run the full phase-based simulation.

    Parameters
    ----------
    params       : SimParams (default: SimParams())
    initial_mode : starting phase (default PIVOTEMENT to match reference model)

    Returns
    -------
    hist   : dict of time-series arrays
    params : SimParams used
    """
    if params is None:
        params = SimParams()

    # Initial state: A at origin (l=0), box at rest
    state = CrateState(l=0.0, ldot=0.0, mode=initial_mode)

    keys = ['t', 'l', 'ldot', 'psi', 'delta',
            'fx', 'fy', 'xA', 'yA', 'xB', 'yB', 'xO', 'yO',
            'fyA', 'fBn', 'fxB', 'fyB', 'mode']
    hist = {k: [] for k in keys}

    N = int(params.total_time / params.dt)

    for k in range(N + 1):
        t = k * params.dt

        # ── Phase transition check (monitor) ──────────────────────────────
        result = check_transition(state, t, params)
        if result is None:
            print(f"[t={t:.3f}s] Simulation complete — FINAL position reached.")
            break
        if result != state.mode:
            print(f"[t={t:.3f}s] {state.mode.name} → {result.name}")
            state.mode = result

        # ── Geometry ──────────────────────────────────────────────────────
        geo = get_box_geometry(state, t, params)
        psi   = geo['psi']
        delta = geo['delta']

        # ── Control (potential energy injection) ──────────────────────────
        fx, fy = get_command(state, t, params)

        # ── Contact forces (for logging) ──────────────────────────────────
        _, _, fyA_i, fBn_i, fxB_i, fyB_i, _, _ = f_continuous(
            state.l, state.ldot, fx, fy, t, params)

        # ── Record ────────────────────────────────────────────────────────
        row = [t, state.l, state.ldot, psi, delta,
               fx, fy,
               geo['A'][0], geo['A'][1],
               geo['B'][0], geo['B'][1],
               geo['O'][0], geo['O'][1],
               fyA_i, fBn_i, fxB_i, fyB_i,
               state.mode.name]
        for key, val in zip(keys, row):
            hist[key].append(val)

        # ── Integrate ─────────────────────────────────────────────────────
        integrate_step(state, fx, fy, t, params)

    return hist, params


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_results(hist: dict, params: SimParams) -> None:
    """Static plots of key simulation variables."""
    t   = hist['t']
    psi_switch_t = next(
        (hist['t'][i] for i, p in enumerate(hist['psi']) if p >= params.psi_switch),
        None)

    def vline(ax):
        if psi_switch_t is not None:
            ax.axvline(psi_switch_t, color='gray', lw=1.2, ls='--',
                       label=f'psi={np.degrees(params.psi_switch):.0f}°')

    plots = [
        ('l(t)',
         [(hist['l'],    'steelblue', 'l(t)'),
          ([params.b]*len(t), 'seagreen', f'target b={params.b} m', '--')],
         'l (m)'),

        ('ldot(t)',
         [(hist['ldot'], 'mediumpurple', 'ldot(t)')],
         'ldot (m/s)'),

        ('Angles',
         [(np.degrees(hist['psi']),   'darkorange', 'psi  wall 2'),
          (np.degrees(hist['delta']), 'seagreen',   'delta  box')],
         'deg'),

        ('Command (fx, fy)',
         [(hist['fx'], 'steelblue', 'fx'),
          (hist['fy'], 'tomato',    'fy')],
         'N'),

        ('fyA — wall 1 reaction',
         [(hist['fyA'], 'steelblue', 'fyA')],
         'N'),

        ('fxB, fyB — wall 2 reaction',
         [(hist['fxB'], 'seagreen',   'fxB'),
          (hist['fyB'], 'darkorange', 'fyB')],
         'N'),
    ]

    for title, curves, ylabel in plots:
        fig, ax = plt.subplots(figsize=(7, 4))
        for curve in curves:
            y, color, label = curve[0], curve[1], curve[2]
            ls = curve[3] if len(curve) > 3 else '-'
            ax.plot(t, y, color=color, lw=1.5, ls=ls, label=label)
        ax.axhline(0, color='gray', lw=0.8)
        vline(ax)
        ax.set_title(title)
        ax.set_xlabel('t (s)')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.show()


def animate(hist: dict, params: SimParams) -> None:
    """Animated visualisation of the crate and walls."""
    skip   = 5
    frames = np.arange(0, len(hist['t']), skip)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(-0.10, 0.65)
    ax.set_ylim(-0.10, 0.65)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Phase-based crate manipulation')

    ax.plot([-0.1, 0.65], [0, 0], 'k', lw=2, label='wall 1')  # static — no handle needed
    wall2_line, = ax.plot([], [],                    'k',  lw=2,   label='wall 2')
    body_line,  = ax.plot([], [],                    'b',  lw=1.5, label='crate')
    A_pt,       = ax.plot([], [], 'ko', ms=6, label='A')
    B_pt,       = ax.plot([], [], 'ks', ms=6, label='B')
    O_pt,       = ax.plot([], [], 'k^', ms=6, label='CoM')
    force_line, = ax.plot([], [], 'r',  lw=2,  label='force')
    traj_line,  = ax.plot([], [], 'b--', lw=1, alpha=0.4)
    info_txt = ax.text(0.02, 0.97, '', transform=ax.transAxes,
                       fontsize=9, va='top', fontfamily='monospace',
                       bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))
    ax.legend(loc='upper right', fontsize=8)

    traj_x, traj_y = [], []

    from dynamics import rot

    def update(i):
        psi_i   = hist['psi'][i]
        delta_i = hist['delta'][i]
        t_i     = hist['t'][i]
        fx_i    = hist['fx'][i]
        fy_i    = hist['fy'][i]
        xA_i    = hist['xA'][i]
        yA_i    = hist['yA'][i]
        xO_i    = hist['xO'][i]
        yO_i    = hist['yO'][i]
        mode_i  = hist['mode'][i]

        L2 = 0.55
        wall2_line.set_data([0, L2 * np.cos(psi_i)], [0, L2 * np.sin(psi_i)])

        R = rot(delta_i)
        a, b = params.a, params.b
        local = np.array([[0, 0], [a, 0], [a, b], [0, b], [0, 0]], dtype=float)
        world = np.array([np.array([xA_i, yA_i]) + R @ c for c in local])
        body_line.set_data(world[:, 0], world[:, 1])

        A_pt.set_data([hist['xA'][i]], [hist['yA'][i]])
        B_pt.set_data([hist['xB'][i]], [hist['yB'][i]])
        O_pt.set_data([xO_i], [yO_i])

        scale = 0.02
        force_line.set_data([xO_i, xO_i + fx_i * scale],
                            [yO_i, yO_i + fy_i * scale])

        traj_x.append(xO_i)
        traj_y.append(yO_i)
        traj_line.set_data(traj_x, traj_y)

        info_txt.set_text(
            f"t     = {t_i:.2f} s\n"
            f"psi   = {np.degrees(psi_i):.1f} deg\n"
            f"delta = {np.degrees(delta_i):.1f} deg\n"
            f"l     = {hist['l'][i]:.3f} m\n"
            f"fx    = {fx_i:.1f} N\n"
            f"fy    = {fy_i:.1f} N\n"
            f"phase = {mode_i}"
        )
        return wall2_line, body_line, A_pt, B_pt, O_pt, force_line, traj_line, info_txt

    ani = FuncAnimation(fig, update, frames=frames,
                        interval=int(1000 * params.dt * skip), blit=False)
    plt.tight_layout()
    plt.show()
    return ani


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    params = SimParams()
    hist, params = simulate(params)

    psi_sw_t = next(
        (hist['t'][i] for i, p in enumerate(hist['psi']) if p >= params.psi_switch),
        None)
    print(f"\nSwitch at   t = {psi_sw_t:.3f} s  (psi = {np.degrees(params.psi_switch):.0f} deg)"
          if psi_sw_t else "\nSwitch not reached.")
    print(f"l final       = {hist['l'][-1]:.4f} m  (target b = {params.b} m)")
    print(f"ldot final    = {hist['ldot'][-1]:.4f} m/s  (ideal = 0)")
    print(f"Max |ldot|    = {max(abs(v) for v in hist['ldot']):.4f} m/s")

    plot_results(hist, params)
    ani = animate(hist, params)
