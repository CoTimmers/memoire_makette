import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

# ─── Parameters ───────────────────────────────────────────────────────────────
a = 0.3          # short side (m)
b = 0.4          # long side (m)
m = 7.0          # mass (kg)
I = m * (a**2 + b**2) / 12.0   # moment of inertia about COM

F_crane      = 5.0   # crane force phase 1 (N)
F_crane_p2   = 10.0    # crane force phase 2 (N)
damp_rot     = 1.5    # rotational damping phase 1
damp_rot_p2  = 3.0    # rotational damping phase 2 — higher to converge
damp_lin     = 6.0    # linear damping on xA phase 2 — higher to converge
e_rest       = 0.2    # restitution coefficient — lower = less bouncing
TRANSITION_DUR = 10.0  # seconds to blend force direction

wall2_speed = np.radians(5)    # rad/s anticlockwise
wall2_max   = np.radians(50)   # wall2 stops at 45°

CRANE_TARGET = np.array([0.3, -0.1])   # fixed aim point for phase 2

dt    = 0.005
t_end = 25.0
times = np.arange(0, t_end, dt)

# ─── Geometry ─────────────────────────────────────────────────────────────────
OFFSETS_BODY = np.array([[-b/2,-a/2],[b/2,-a/2],[b/2,a/2],[-b/2,a/2]])

def rot_matrix(theta):
    return np.array([[np.cos(theta),-np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])

def crate_corners(cx, cy, theta):
    R = rot_matrix(theta)
    return np.array([np.array([cx,cy]) + R @ o for o in OFFSETS_BODY])

def pivot_corner(cx, cy, theta):
    """Corner closest to origin."""
    corners = crate_corners(cx, cy, theta)
    return corners[np.argmin(np.linalg.norm(corners, axis=1))]

def wall2_normal(psi):
    """Inward normal of wall2 (pointing toward crate)."""
    return np.array([np.cos(psi), np.sin(psi)])

def wall1_contact(theta, fixed_offset=None):
    """
    Returns (rCA, cy_when_A_at_y0).
    If fixed_offset is given (body-frame offset of A from COM),
    uses that corner always — prevents pivot switching.
    Otherwise picks the lowest corner dynamically.
    """
    R = rot_matrix(theta)
    if fixed_offset is not None:
        world_offset = R @ fixed_offset   # COM→A in world frame
        rCA = -world_offset               # A→COM
        cy  = -world_offset[1]            # shift so A is at y=0
    else:
        ys  = np.array([(R @ o)[1] for o in OFFSETS_BODY])
        idx = np.argmin(ys)
        rCA = -(R @ OFFSETS_BODY[idx])
        cy  = -ys[idx]
    return rCA, cy

# ─── Sector and force direction ───────────────────────────────────────────────
def sector_angles(cx, cy, theta, psi):
    nB  = wall2_normal(psi)
    lo  = np.arctan2(-nB[1], -nB[0])
    A   = pivot_corner(cx, cy, theta)
    rAC = np.array([cx,cy]) - A
    hi  = np.arctan2(-rAC[1], -rAC[0])
    while hi <= lo: hi += 2*np.pi
    while hi - lo >= np.pi: hi -= 2*np.pi
    if hi <= lo: hi += 2*np.pi
    return lo, hi

def crane_force_dir_p1(cx, cy, theta, psi):
    lo, hi = sector_angles(cx, cy, theta, psi)
    mid = (lo+hi)/2
    return np.array([np.cos(mid), np.sin(mid)])

# Fixed direction computed once at t=0
theta0   = np.pi/2
cx0, cy0 = a/2, b/2
A0       = np.array([0., 0.])
FIXED_DIR = crane_force_dir_p1(cx0, cy0, theta0, 0.0)
print(f"Phase 1 fixed force direction: {np.degrees(np.arctan2(FIXED_DIR[1],FIXED_DIR[0])):.1f}°")

def force_dir_p2(cx, cy):
    d = CRANE_TARGET - np.array([cx,cy])
    n = np.linalg.norm(d)
    return d/n if n > 1e-6 else np.array([1.,0.])

def blended_dir(cx, cy, t_p2):
    alpha = min(t_p2 / TRANSITION_DUR, 1.0)
    d = (1-alpha)*FIXED_DIR + alpha*force_dir_p2(cx,cy)
    n = np.linalg.norm(d)
    return d/n if n > 1e-6 else force_dir_p2(cx,cy)

# ─── Phase 1 step: pure rotation about fixed A=(0,0) ─────────────────────────
def step_p1(s, psi):
    cx, cy, theta, vx, vy, om = s
    F   = F_crane * FIXED_DIR
    rAC = np.array([cx,cy]) - A0
    I_A = I + m*np.dot(rAC,rAC)
    tau = rAC[0]*F[1] - rAC[1]*F[0] - damp_rot*om
    alpha     = tau / I_A
    om_new    = om + alpha*dt
    theta_new = theta + om_new*dt

    # Clamp to [pi/2, pi]
    if theta_new > np.pi:
        theta_new = np.pi
        om_new = -e_rest*abs(om_new)
    if theta_new < np.pi/2:
        theta_new = np.pi/2
        om_new = 0.0

    dth   = theta_new - theta
    Rd    = rot_matrix(dth)
    rAC_n = Rd @ rAC
    cx_n  = A0[0] + rAC_n[0]
    cy_n  = A0[1] + rAC_n[1]

    # Wall2 contact: binary search to prevent penetration
    nW2 = wall2_normal(psi)
    if min(np.dot(c, nW2) for c in crate_corners(cx_n, cy_n, theta_new)) < -1e-5:
        lo_s, hi_s = theta, theta_new
        for _ in range(20):
            tm  = (lo_s+hi_s)/2
            ct  = A0 + rot_matrix(tm-theta) @ rAC
            pen = min(np.dot(c,nW2) for c in crate_corners(ct[0],ct[1],tm))
            if pen < -1e-5: hi_s = tm
            else:           lo_s = tm
        theta_new = lo_s
        Rd    = rot_matrix(theta_new-theta)
        rAC_n = Rd @ rAC
        cx_n  = A0[0] + rAC_n[0]
        cy_n  = A0[1] + rAC_n[1]
        om_new = -e_rest*abs(om_new)

    vx_n = -om_new*rAC_n[1]
    vy_n =  om_new*rAC_n[0]
    return np.array([cx_n, cy_n, theta_new, vx_n, vy_n, om_new])

# ─── Phase 2 step: A slides on wall1 (y_A=0), 2 DOF: xA, theta ──────────────
def step_p2(s, psi, t_p2, a_body_offset):
    cx, cy, theta, vx, vy, om = s

    # Use FIXED body-frame offset of A — never changes corner
    rCA, cy_implied = wall1_contact(theta, fixed_offset=a_body_offset)
    xA  = cx - rCA[0]

    vxA = vx + om*rCA[1]

    F = F_crane_p2 * blended_dir(cx, cy, t_p2)

    I_A   = I + m*np.dot(rCA,rCA)
    tau_A = rCA[0]*F[1] - rCA[1]*F[0]
    alpha = (tau_A - damp_rot_p2*om) / I_A
    a_Ax  = (F[0] + m*alpha*rCA[1] - damp_lin*vxA) / m

    om_new    = om    + alpha*dt
    theta_new = theta + om_new*dt
    vxA_new   = vxA  + a_Ax*dt
    xA_new    = xA   + vxA_new*dt

    rCA_n, cy_n = wall1_contact(theta_new, fixed_offset=a_body_offset)
    cx_n = xA_new + rCA_n[0]

    # ── Wall 1 non-penetration: all corners y ≥ 0 ──
    corners = crate_corners(cx_n, cy_n, theta_new)
    min_y   = min(c[1] for c in corners)
    if min_y < -1e-5:
        # Shift entire crate up by penetration amount — cy_n changes, cx_n unchanged
        cy_n   -= min_y
        # xA stays same (A slides along x), only cy of COM changes
        # Damp omega and vxA on impact
        if om_new > 0:    # rotating in direction that pushes corner down
            om_new  *= -e_rest
        vxA_new *= e_rest

    # ── Wall 2 non-penetration: all corners dist ≥ 0 ──
    nW2 = wall2_normal(psi)
    corners = crate_corners(cx_n, cy_n, theta_new)
    pen = min(np.dot(c, nW2) for c in corners)
    if pen < -1e-5:
        cx_n   -= pen * nW2[0]
        cy_n   -= pen * nW2[1]
        xA_new  = cx_n - rCA_n[0]
        v_n     = vxA_new * nW2[0]
        if v_n < 0:
            vxA_new -= (1 + e_rest) * v_n * nW2[0]
        om_new *= e_rest

    vx_n = vxA_new - om_new*rCA_n[1]
    vy_n =           om_new*rCA_n[0]
    return np.array([cx_n, cy_n, theta_new, vx_n, vy_n, om_new])

# ─── Run simulation ───────────────────────────────────────────────────────────
state      = np.array([cx0, cy0, theta0, 0., 0., 0.])
states     = [state.copy()]
wall2_angs = [0.0]
phases     = [1]
verified   = False
t_p2       = 0.0
A_BODY_OFFSET = None   # body-frame offset of pivot A, fixed at phase 2 start

for i in range(1, len(times)):
    s   = states[-1].copy()
    psi = wall2_angs[-1]
    ph  = phases[-1]

    if ph == 1:
        psi_new = min(psi + wall2_speed*dt, wall2_max)
        if psi_new >= wall2_max - 1e-4 and not verified:
            verified = True
            ph = 2
            t_p2 = 0.0
            # Fix body-frame offset of A at this exact moment
            cx_s, cy_s, theta_s = s[0], s[1], s[2]
            R_s = rot_matrix(theta_s)
            ys  = np.array([(R_s @ o)[1] for o in OFFSETS_BODY])
            A_BODY_OFFSET = OFFSETS_BODY[np.argmin(ys)].copy()
        s_new = step_p1(s, psi_new)
    else:
        psi_new = psi
        t_p2   += dt
        s_new   = step_p2(s, psi_new, t_p2, A_BODY_OFFSET)

    states.append(s_new.copy())
    wall2_angs.append(psi_new)
    phases.append(ph)

states     = np.array(states)
wall2_angs = np.array(wall2_angs)
phases     = np.array(phases)

# ─── Animation ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 9))
ax.set_xlim(-0.55, 0.85)
ax.set_ylim(-0.15, 0.85)
ax.set_aspect('equal')
ax.set_facecolor('#f8f9fa')
fig.patch.set_facecolor('#f0f2f5')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlabel('x (m)', fontsize=11)
ax.set_ylabel('y (m)', fontsize=11)
ax.set_title('Pivotement du bac — Vue du dessus', fontsize=13, fontweight='bold')

# Wall 1
ax.axhline(0, color='#343a40', lw=5, zorder=5)
ax.fill_between([-0.55,0.85], [-0.15,-0.15], [0,0], color='#adb5bd', alpha=0.5)
ax.text(0.75, -0.11, 'Wall 1', fontsize=9, color='#343a40')

# Wall 2 (dynamic)
wall2_line, = ax.plot([], [], color='#343a40', lw=5, zorder=5)
wall2_fill  = ax.fill([], [], color='#adb5bd', alpha=0.5, zorder=4)[0]
ax.text(-0.45, 0.4, 'Wall 2', fontsize=9, color='#343a40', rotation=90, va='center')

# Crate
crate_patch = patches.Polygon([[0,0]]*4, closed=True,
                               facecolor='#4dabf7', edgecolor='#1971c2',
                               lw=2, alpha=0.85, zorder=6)
ax.add_patch(crate_patch)

# COM
com_dot, = ax.plot([], [], 'o', color='#e03131', markersize=9, zorder=10)

# Crane target
ax.plot(*CRANE_TARGET, 'x', color='#f08c00', markersize=14, markeredgewidth=2.5, zorder=12)
ax.annotate('Cible grue\n(phase 2)', xy=CRANE_TARGET,
            xytext=(CRANE_TARGET[0]+0.05, CRANE_TARGET[1]+0.07),
            fontsize=8, color='#f08c00')

# Pivot A
A_dot, = ax.plot([], [], '^', color='#9c36b5', markersize=11, zorder=11)

# Sector lines + fill
SL = 0.20
sector_lo, = ax.plot([], [], color='#2f9e44', lw=1.8, zorder=7)
sector_hi, = ax.plot([], [], color='#e03131', lw=1.8, zorder=7)
sector_fan  = ax.fill([], [], color='#69db7c', alpha=0.3, zorder=4)[0]

# Force arrow
FA = 0.15
force_line, = ax.plot([], [], color='#f08c00', lw=2.2, linestyle='-.', zorder=8)
force_arr   = ax.annotate('', xy=(0,0), xytext=(0,0),
    arrowprops=dict(arrowstyle='->', color='#f08c00', lw=2.5), zorder=9)

# Info
info_txt  = ax.text(0.02, 0.97, '', transform=ax.transAxes, fontsize=9, va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
phase_txt = ax.text(0.98, 0.03, '', transform=ax.transAxes, fontsize=11,
                    fontweight='bold', va='bottom', ha='right',
                    bbox=dict(boxstyle='round', facecolor='#fff3bf', alpha=0.9))

from matplotlib.lines import Line2D
ax.legend(handles=[
    Line2D([0],[0], color='#2f9e44', lw=2,   label='Limite basse (−n_B)'),
    Line2D([0],[0], color='#e03131', lw=2,   label='Limite haute (C→A)'),
    Line2D([0],[0], color='#f08c00', lw=2, linestyle='-.', label='Force grue'),
    patches.Patch(facecolor='#69db7c', alpha=0.4, label='Secteur acceptable'),
    Line2D([0],[0], marker='^', color='#9c36b5', lw=0, markersize=9, label='Pivot A'),
], loc='upper right', fontsize=8, framealpha=0.9)

# Compute sector angles ONCE at initial position — fixed for phase 1
LO_FIXED, HI_FIXED = sector_angles(cx0, cy0, theta0, 0.0)

skip = 8

def update(frame):
    idx = min(frame*skip, len(states)-1)
    s   = states[idx]
    psi = wall2_angs[idx]
    ph  = phases[idx]
    cx, cy, theta = s[0], s[1], s[2]

    # Wall 2
    wlen = 0.75
    wx, wy = -wlen*np.sin(psi), wlen*np.cos(psi)
    wall2_line.set_data([0,wx],[0,wy])
    perp = np.array([-np.cos(psi),-np.sin(psi)])*0.18
    wall2_fill.set_xy([[0,0],[wx,wy],[wx+perp[0],wy+perp[1]],[perp[0],perp[1]]])

    # Crate + COM
    crate_patch.set_xy(crate_corners(cx,cy,theta))
    com_dot.set_data([cx],[cy])

    # Pivot A
    A = pivot_corner(cx,cy,theta)
    A_dot.set_data([A[0]],[A[1]])

    # Sector — fixed initial orientation in phase 1, hidden in phase 2
    if ph == 1:
        lo, hi = LO_FIXED, HI_FIXED
        sector_lo.set_data([cx, cx+SL*np.cos(lo)],[cy, cy+SL*np.sin(lo)])
        sector_hi.set_data([cx, cx+SL*np.cos(hi)],[cy, cy+SL*np.sin(hi)])
        fan_a = np.linspace(lo, hi, 30)
        sector_fan.set_xy(list(zip(
            [cx]+list(cx+SL*np.cos(fan_a))+[cx],
            [cy]+list(cy+SL*np.sin(fan_a))+[cy]
        )))
    else:
        # Hide sector in phase 2
        sector_lo.set_data([], [])
        sector_hi.set_data([], [])
        sector_fan.set_xy([[0,0]])

    # Force direction
    t_elapsed = np.sum(np.array(phases[:idx+1])==2) * dt
    if ph == 1:
        f_dir = FIXED_DIR
    else:
        f_dir = blended_dir(cx, cy, t_elapsed)
    fa = np.arctan2(f_dir[1], f_dir[0])
    force_line.set_data([cx, cx+SL*np.cos(fa)],[cy, cy+SL*np.sin(fa)])
    force_arr.set_position((cx,cy))
    force_arr.xy = (cx+FA*np.cos(fa), cy+FA*np.sin(fa))

    # Info
    lo_d, hi_d = sector_angles(cx, cy, theta, psi)
    sw = np.degrees(hi_d - lo_d)
    om = s[5]
    rCA, _ = wall1_contact(theta)
    I_A = I + m*np.dot(rCA,rCA)
    KE  = 0.5*I_A*om**2
    info_txt.set_text(
        f'θ (bac)   = {np.degrees(theta):.1f}°\n'
        f'ψ (Wall2) = {np.degrees(psi):.1f}°\n'
        f'Secteur   = {sw:.1f}°\n'
        f'E. cin.   = {KE:.3f} J'
    )
    if ph == 1:
        phase_txt.set_text('PHASE 1\nWall 2 → 45°')
        phase_txt.get_bbox_patch().set_facecolor('#fff3bf')
    else:
        alpha_blend = min(t_elapsed/TRANSITION_DUR, 1.0)*100
        phase_txt.set_text(f'PHASE 2\nTransition {alpha_blend:.0f}%')
        phase_txt.get_bbox_patch().set_facecolor('#d3f9d8')

    return (crate_patch, com_dot, wall2_line, wall2_fill,
            sector_lo, sector_hi, sector_fan, force_line,
            A_dot, info_txt, phase_txt)

n_frames = len(states)//skip
ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=40, blit=False)
plt.tight_layout()
writer = animation.FFMpegWriter(fps=25, bitrate=2000)
ani.save('pivotement_simu1.mp4', writer=writer)
print("Done.")
