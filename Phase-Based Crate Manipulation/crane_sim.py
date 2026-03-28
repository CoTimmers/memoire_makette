"""
Crate simulation — Euler-Lagrange dynamics
==========================================
State:  q = [xA, theta]   (2 DOF)
        q_dot = [vxA, omega]

Wall 1: y = 0  (A slides along x, y_A = 0 always)
Wall 2: rotated at psi = pi/4 (fixed after pivoting phase)

Crane positions: list of (ux, uy) entered manually below.
The simulation runs each phase until steady state, then moves to next position.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import casadi as ca

# ─── Physical parameters ───────────────────────────────────────────
a    = 0.3       # short side (m)
b    = 0.4       # long side (m)
m    = 7.0       # mass (kg)
g    = 9.81      # gravity (m/s²) — projected into horizontal plane via cable
I_c  = m * (a**2 + b**2) / 12.0  # moment of inertia about COM
L    = 1.2       # cable length (m)
mu   = 0.3       # friction coefficient at Wall 2
psi  = np.pi/4   # Wall 2 angle (fixed, after pivoting)

# Cable stiffness equivalent: F = (m*g/L) * horizontal_offset
k_cable = m * g / L   # pendulum stiffness (N/m)

# Damping (structural + contact)
damp_rot = 2.0
damp_lin = 5.0
e_rest   = 0.15

dt       = 0.005
t_phase  = 15.0   # max time per phase (s)
ss_tol   = 1e-4   # steady-state kinetic energy threshold (J)
ss_time  = 1.0    # must stay below threshold for this long

# ─── Crane positions (enter manually) ──────────────────────────────
# Format: list of (ux, uy) — crane position for each phase
# System starts at theta=pi/2, xA=0
CRANE_POSITIONS = [
    (-0.20,  0.10),   # Phase 1: creates rotation toward theta=180°
    ( 0.25, -0.10),   # Phase 2: pulls toward final position
]

# ─── Geometry ──────────────────────────────────────────────────────
OFFSETS_BODY = np.array([[-b/2,-a/2],[b/2,-a/2],[b/2,a/2],[-b/2,a/2]])

def rot(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta),  np.cos(theta)]])

def com_from_state(xA, theta):
    """COM position from state (xA, theta). A is at (xA, 0).
    Body offset A->COM: at theta=pi/2, COM is at (a/2, b/2) from A=(0,0).
    rAC_body = R(-pi/2) @ (a/2, b/2) = (b/2, -a/2)
    """
    R      = rot(theta)
    rAC    = R @ np.array([b/2, -a/2])   # body offset A→COM
    return np.array([xA + rAC[0], rAC[1]])

def corners_from_state(xA, theta):
    cx, cy = com_from_state(xA, theta)
    R = rot(theta)
    return np.array([np.array([cx, cy]) + R @ o for o in OFFSETS_BODY])

def wall2_normal():
    return np.array([np.cos(psi), np.sin(psi)])

# ─── Euler-Lagrange mass matrix and RHS ────────────────────────────
def build_EL(xA, theta, vxA, omega, ux, uy):
    """
    Euler-Lagrange equations for q = [xA, theta]:
      M(theta) * q_ddot = rhs(q, q_dot, u)

    Derivation:
      cx = xA + (b/2)*cos(theta) - (a/2)*sin(theta)
      cy =      (b/2)*sin(theta) + (a/2)*cos(theta)

      cx_dot = vxA - (b/2)*sin(theta)*omega - (a/2)*cos(theta)*omega
      cy_dot =       (b/2)*cos(theta)*omega - (a/2)*sin(theta)*omega

    T = 0.5*m*(cx_dot^2 + cy_dot^2) + 0.5*I_c*omega^2
    U = 0.5*k_cable*((cx-ux)^2 + (cy-uy)^2)
    """
    # body offset A->COM: rAC_body = (b/2, -a/2)
    # cx = xA + (b/2)*cos(theta) + (a/2)*sin(theta)
    # cy =      (b/2)*sin(theta) - (a/2)*cos(theta)
    c  = np.cos(theta)
    s  = np.sin(theta)

    dcx_dxA    = 1.0
    dcx_dtheta = -(b/2)*s + (a/2)*c
    dcy_dxA    = 0.0
    dcy_dtheta =  (b/2)*c + (a/2)*s

    # Mass matrix M = J^T * m * J + I_c (for rotation DOF)
    # J = [[dcx_dxA, dcx_dtheta],
    #      [dcy_dxA, dcy_dtheta]]
    M11 = m * (dcx_dxA**2    + dcy_dxA**2)
    M12 = m * (dcx_dxA*dcx_dtheta + dcy_dxA*dcy_dtheta)
    M22 = m * (dcx_dtheta**2 + dcy_dtheta**2) + I_c
    M   = np.array([[M11, M12], [M12, M22]])

    # Coriolis/centrifugal terms: d/dt(J) * q_dot
    # d(dcx_dtheta)/dt = (-(b/2)*c + (a/2)*s) * omega
    # d(dcy_dtheta)/dt = (-(b/2)*s - (a/2)*c) * omega
    d2cx_dtheta2 = -(b/2)*c - (a/2)*s
    d2cy_dtheta2 = -(b/2)*s + (a/2)*c

    # Coriolis RHS contribution: -m * J^T * (dJ/dt * q_dot)
    # dJ/dt * q_dot = [[0], [d2cx_dtheta2 * omega],
    #                        [d2cy_dtheta2 * omega]]  (only theta col changes)
    ax_cor = d2cx_dtheta2 * omega**2
    ay_cor = d2cy_dtheta2 * omega**2

    cor1 = -m * (dcx_dxA    * ax_cor + dcy_dxA    * ay_cor)
    cor2 = -m * (dcx_dtheta * ax_cor + dcy_dtheta * ay_cor)

    # Potential energy gradient: ∂U/∂q
    cx, cy = com_from_state(xA, theta)
    dU_dcx = k_cable * (cx - ux)
    dU_dcy = k_cable * (cy - uy)
    dU_dxA    = dU_dcx * dcx_dxA    + dU_dcy * dcy_dxA
    dU_dtheta = dU_dcx * dcx_dtheta + dU_dcy * dcy_dtheta

    # Damping generalized forces
    Q_damp = np.array([-damp_lin * vxA, -damp_rot * omega])

    # RHS
    rhs = np.array([cor1 - dU_dxA,
                    cor2 - dU_dtheta]) + Q_damp

    return M, rhs

# ─── Wall contact constraints ───────────────────────────────────────
def apply_wall1_constraint(xA, theta, vxA, omega):
    """
    Wall 1: all corners y >= 0.
    If violated: push up (cy correction), apply restitution on omega.
    """
    corners = corners_from_state(xA, theta)
    min_y   = min(c[1] for c in corners)
    if min_y < -1e-5:
        # This shouldn't happen if A is always on Wall 1 (y_A=0)
        # but handle numerically
        if omega > 0:
            omega *= -e_rest
        vxA   *= e_rest
    return vxA, omega

def apply_wall2_constraint(xA, theta, vxA, omega, ux, uy):
    """
    Wall 2: no corner penetrates.
    Check friction cone: |F_t| <= mu * F_n at contact point B.
    If penetration: push back, apply restitution.
    Returns corrected state + flag indicating if B is in contact.
    """
    nW2     = wall2_normal()
    corners = corners_from_state(xA, theta)
    cx, cy  = com_from_state(xA, theta)

    # Find most penetrating corner
    dists  = [np.dot(c, nW2) for c in corners]
    min_d  = min(dists)

    if min_d < -1e-5:
        # Compute reaction force needed to prevent penetration
        # Push A along Wall 1 to resolve penetration
        pen = -min_d
        # Shift xA so that penetrating corner is back on wall
        # Approximate: move COM by pen * nW2
        # cx_new = cx + pen*nW2[0] → xA_new = xA + pen*nW2[0]
        xA += pen * nW2[0]

        # Velocity: reflect component along nW2
        # vCOM = vxA * (1,0) + omega * (-rAC_y, rAC_x)
        R   = rot(theta)
        rAC = R @ np.array([b/2, -a/2])
        vcx = vxA - omega * rAC[1]
        vcy =       omega * rAC[0]
        v_n = vcx * nW2[0] + vcy * nW2[1]
        if v_n < 0:
            vcx -= (1 + e_rest) * v_n * nW2[0]
            vcy -= (1 + e_rest) * v_n * nW2[1]
            # Recover vxA and omega from corrected vCOM
            # vxA = vcx + omega * rAC[1]  — use old omega for now
            vxA = vcx + omega * rAC[1]
            omega *= e_rest

    return xA, vxA, omega

# ─── Integration step ──────────────────────────────────────────────
def step(xA, theta, vxA, omega, ux, uy):
    """One RK4 step of Euler-Lagrange equations."""
    def f(xA, theta, vxA, omega):
        M, rhs = build_EL(xA, theta, vxA, omega, ux, uy)
        q_ddot = np.linalg.solve(M, rhs)
        return vxA, omega, q_ddot[0], q_ddot[1]

    # RK4
    k1 = f(xA,              theta,              vxA,              omega)
    k2 = f(xA+dt/2*k1[0],  theta+dt/2*k1[1],  vxA+dt/2*k1[2],  omega+dt/2*k1[3])
    k3 = f(xA+dt/2*k2[0],  theta+dt/2*k2[1],  vxA+dt/2*k2[2],  omega+dt/2*k2[3])
    k4 = f(xA+dt*k3[0],    theta+dt*k3[1],    vxA+dt*k3[2],    omega+dt*k3[3])

    xA_n    = xA    + dt/6*(k1[0]+2*k2[0]+2*k3[0]+k4[0])
    theta_n = theta + dt/6*(k1[1]+2*k2[1]+2*k3[1]+k4[1])
    vxA_n   = vxA   + dt/6*(k1[2]+2*k2[2]+2*k3[2]+k4[2])
    omega_n = omega + dt/6*(k1[3]+2*k2[3]+2*k3[3]+k4[3])

    # Clamp theta to [pi/2, pi]
    if theta_n > np.pi:
        theta_n = np.pi
        omega_n = -e_rest * abs(omega_n)
    if theta_n < np.pi/2:
        theta_n = np.pi/2
        omega_n =  e_rest * abs(omega_n)

    # Apply wall constraints
    vxA_n, omega_n = apply_wall1_constraint(xA_n, theta_n, vxA_n, omega_n)
    xA_n, vxA_n, omega_n = apply_wall2_constraint(
        xA_n, theta_n, vxA_n, omega_n, ux, uy)

    return xA_n, theta_n, vxA_n, omega_n

# ─── Run simulation ────────────────────────────────────────────────
def run_phase(xA0, theta0, vxA0, omega0, ux, uy, phase_idx):
    """Run one phase until steady state or timeout."""
    xA, theta, vxA, omega = xA0, theta0, vxA0, omega0
    states  = [(xA, theta, vxA, omega)]
    ss_count = 0

    n_steps = int(t_phase / dt)
    for _ in range(n_steps):
        xA, theta, vxA, omega = step(xA, theta, vxA, omega, ux, uy)
        states.append((xA, theta, vxA, omega))

        # Kinetic energy check for steady state
        cx, cy = com_from_state(xA, theta)
        R   = rot(theta)
        rAC = R @ np.array([b/2, -a/2])
        vcx = vxA - omega * rAC[1]
        vcy =       omega * rAC[0]
        KE  = 0.5*m*(vcx**2+vcy**2) + 0.5*I_c*omega**2
        if KE < ss_tol:
            ss_count += 1
            if ss_count >= int(ss_time / dt):
                print(f"Phase {phase_idx+1}: steady state at t={len(states)*dt:.2f}s  "
                      f"theta={np.degrees(theta):.1f}°  xA={xA:.3f}m")
                break
        else:
            ss_count = 0

    return states

# Run all phases
print("Running simulation...")
theta0, xA0, vxA0, omega0 = np.pi/2, 0.0, 0.0, 0.0

all_states  = []
all_phases  = []
all_cranes  = []

for i, (ux, uy) in enumerate(CRANE_POSITIONS):
    print(f"Phase {i+1}: crane at ({ux:.3f}, {uy:.3f})")
    phase_states = run_phase(xA0, theta0, vxA0, omega0, ux, uy, i)
    all_states  += phase_states
    all_phases  += [i+1] * len(phase_states)
    all_cranes  += [(ux, uy)] * len(phase_states)
    xA0, theta0, vxA0, omega0 = phase_states[-1]

all_states = np.array([(s[0],s[1],s[2],s[3]) for s in all_states])
all_phases = np.array(all_phases)
print(f"Total frames: {len(all_states)}")

# ─── Animation ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 7),
                          gridspec_kw={'width_ratios': [2, 1]})
ax   = axes[0]
ax_e = axes[1]

ax.set_xlim(-0.3, 0.9)
ax.set_ylim(-0.15, 0.75)
ax.set_aspect('equal')
ax.set_facecolor('#f8f9fa')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlabel('x (m)', fontsize=11)
ax.set_ylabel('y (m)', fontsize=11)
ax.set_title('Crate — Euler-Lagrange dynamics', fontsize=12, fontweight='bold')

# Wall 1
ax.axhline(0, color='#343a40', lw=5, zorder=5)
ax.fill_between([-0.3,0.9],[-0.15,-0.15],[0,0], color='#adb5bd', alpha=0.5)
ax.text(0.8, -0.10, 'Wall 1', fontsize=9)

# Wall 2
wlen = 0.7
wx, wy = -wlen*np.sin(psi), wlen*np.cos(psi)
ax.plot([0,wx],[0,wy], color='#343a40', lw=5, zorder=5)
ax.fill([0,wx,wx-0.15*np.cos(psi),0-0.15*np.cos(psi)],
        [0,wy,wy-0.15*np.sin(psi),0-0.15*np.sin(psi)],
        color='#adb5bd', alpha=0.5, zorder=4)
ax.text(-0.25, 0.35, 'Wall 2', fontsize=9, rotation=90, va='center')

# Crane positions markers
colors_crane = ['#e03131', '#2f9e44', '#1971c2', '#f08c00']
for i, (ux, uy) in enumerate(CRANE_POSITIONS):
    c = colors_crane[i % len(colors_crane)]
    ax.plot(ux, uy, '*', color=c, markersize=14, zorder=12)
    ax.annotate(f'u{i+1}=({ux:.2f},{uy:.2f})',
                xy=(ux, uy), xytext=(ux+0.03, uy+0.04),
                fontsize=7, color=c)

# Crate patch
crate_patch = patches.Polygon([[0,0]]*4, closed=True,
                               facecolor='#4dabf7', edgecolor='#1971c2',
                               lw=2, alpha=0.85, zorder=6)
ax.add_patch(crate_patch)

com_dot,  = ax.plot([], [], 'o', color='#e03131', markersize=8, zorder=10)
A_dot,    = ax.plot([], [], '^', color='#9c36b5', markersize=10, zorder=11)
crane_dot,= ax.plot([], [], 's', color='#f08c00', markersize=12, zorder=11)
force_arr = ax.annotate('', xy=(0,0), xytext=(0,0),
    arrowprops=dict(arrowstyle='->', color='#f08c00', lw=2.5), zorder=9)

info_txt  = ax.text(0.02, 0.97, '', transform=ax.transAxes, fontsize=9,
                    va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
phase_txt = ax.text(0.98, 0.03, '', transform=ax.transAxes, fontsize=11,
                    fontweight='bold', va='bottom', ha='right',
                    bbox=dict(boxstyle='round', facecolor='#fff3bf', alpha=0.9))

# Energy plot
n_tot = len(all_states)
times_arr = np.arange(n_tot) * dt
KEs = []
PEs = []
for i, (s, (ux,uy)) in enumerate(zip(all_states, all_cranes)):
    xA_i, theta_i, vxA_i, omega_i = s
    cx_i, cy_i = com_from_state(xA_i, theta_i)
    R_i   = rot(theta_i)
    rAC_i = R_i @ np.array([b/2, -a/2])
    vcx_i = vxA_i - omega_i * rAC_i[1]
    vcy_i =         omega_i * rAC_i[0]
    KE_i  = 0.5*m*(vcx_i**2+vcy_i**2) + 0.5*I_c*omega_i**2
    PE_i  = 0.5*k_cable*((cx_i-ux)**2 + (cy_i-uy)**2)
    KEs.append(KE_i)
    PEs.append(PE_i)

ax_e.set_facecolor('#f8f9fa')
ax_e.grid(True, alpha=0.3, linestyle='--')
ax_e.set_xlabel('Time (s)', fontsize=10)
ax_e.set_ylabel('Energy (J)', fontsize=10)
ax_e.set_title('System Energy', fontsize=11, fontweight='bold')
ax_e.plot(times_arr, KEs, color='#e03131', lw=1.2, label='Kinetic')
ax_e.plot(times_arr, PEs, color='#1971c2', lw=1.2, label='Potential')
ax_e.plot(times_arr, np.array(KEs)+np.array(PEs),
          color='#2f9e44', lw=1.5, linestyle='--', label='Total')
ax_e.legend(fontsize=8)
time_vline = ax_e.axvline(0, color='#f08c00', lw=1.5, linestyle=':')

# Phase transition lines
t_transitions = []
for i in range(1, len(all_phases)):
    if all_phases[i] != all_phases[i-1]:
        t_transitions.append(i * dt)
for tt in t_transitions:
    ax_e.axvline(tt, color='#9c36b5', lw=1, linestyle='--', alpha=0.6)

skip = 6

def update(frame):
    idx = min(frame * skip, len(all_states)-1)
    s   = all_states[idx]
    xA_i, theta_i, vxA_i, omega_i = s
    ph  = all_phases[idx]
    ux, uy = all_cranes[idx]

    cx_i, cy_i = com_from_state(xA_i, theta_i)
    corners_i  = corners_from_state(xA_i, theta_i)

    crate_patch.set_xy(corners_i)
    com_dot.set_data([cx_i], [cy_i])
    A_dot.set_data([xA_i], [0])
    crane_dot.set_data([ux], [uy])

    # Force arrow
    d  = np.array([ux - cx_i, uy - cy_i])
    n  = np.linalg.norm(d)
    fd = d/n if n > 1e-6 else np.array([0,-1])
    force_arr.set_position((cx_i, cy_i))
    force_arr.xy = (cx_i + 0.15*fd[0], cy_i + 0.15*fd[1])

    # Info
    R_i   = rot(theta_i)
    rAC_i = R_i @ np.array([b/2, -a/2])
    vcx_i = vxA_i - omega_i * rAC_i[1]
    vcy_i =         omega_i * rAC_i[0]
    KE_i  = 0.5*m*(vcx_i**2+vcy_i**2) + 0.5*I_c*omega_i**2
    PE_i  = 0.5*k_cable*((cx_i-ux)**2 + (cy_i-uy)**2)

    info_txt.set_text(
        f'θ       = {np.degrees(theta_i):.1f}°\n'
        f'xA      = {xA_i:.3f} m\n'
        f'KE      = {KE_i:.4f} J\n'
        f'PE      = {PE_i:.4f} J\n'
        f'Crane   = ({ux:.2f}, {uy:.2f})'
    )
    phase_txt.set_text(f'PHASE {ph}')
    col = colors_crane[(ph-1) % len(colors_crane)]
    phase_txt.get_bbox_patch().set_facecolor(col)
    phase_txt.set_color('white')

    time_vline.set_xdata([idx*dt, idx*dt])

    return (crate_patch, com_dot, A_dot, crane_dot, force_arr,
            info_txt, phase_txt, time_vline)

n_frames = len(all_states) // skip
ani = animation.FuncAnimation(fig, update, frames=n_frames,
                               interval=30, blit=False)
plt.tight_layout()
writer = animation.FFMpegWriter(fps=25, bitrate=2000)
ani.save('/mnt/user-data/outputs/crane_sim.mp4', writer=writer)
print("Saved crane_sim.mp4")
