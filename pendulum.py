import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ============================================================
# PARAMETERS
# ============================================================

g = 9.81          # gravity [m/s^2]
L = 2.2           # cable length [m]

a_max = 0.48     # maximum acceleration [m/s^2]
v_max = 0.30      # target velocity [m/s]

t_j = 0.4        # smooth rise/fall duration [s]
dt = 0.002
t_end = 5.0
N = int(t_end / dt)

# ============================================================
# MOTION PROFILE DESIGN
# ============================================================
# We want:
# 1) smooth rise 0 -> a_max over t_j
# 2) constant a_max over t_const
# 3) smooth fall a_max -> 0 over t_j
# 4) then constant velocity
#
# The total velocity gain must be v_max.

# Velocity gained during one smooth rise:
# integral of 0.5*a_max*(1-cos(pi t/t_j)) dt from 0 to t_j = 0.5*a_max*t_j
dv_rise = 0.5 * a_max * t_j

# Same for smooth fall
dv_fall = 0.5 * a_max * t_j

# Remaining velocity gain must come from constant acceleration plateau
dv_const = v_max - dv_rise - dv_fall

if dv_const < 0:
    raise ValueError("v_max too small for the chosen a_max and t_j. Reduce t_j or increase v_max.")

t_const = dv_const / a_max

t1 = t_j
t2 = t1 + t_const
t3 = t2 + t_j

print(f"t_rise   = {t_j:.3f} s")
print(f"t_const  = {t_const:.3f} s")
print(f"t_fall   = {t_j:.3f} s")
print(f"t_end_acc= {t3:.3f} s")

# ============================================================
# ACCELERATION PROFILE
# ============================================================

def acceleration_profile(t):
    # Phase 1: smooth rise 0 -> a_max
    if 0 <= t < t1:
        return 0.5 * a_max * (1 - np.cos(np.pi * t / t_j))

    # Phase 2: constant acceleration
    elif t1 <= t < t2:
        return a_max

    # Phase 3: smooth fall a_max -> 0
    elif t2 <= t < t3:
        tau = t - t2
        return 0.5 * a_max * (1 + np.cos(np.pi * tau / t_j))

    # Phase 4: zero acceleration
    else:
        return 0.0

# ============================================================
# INITIAL CONDITIONS
# ============================================================

x = 0.0
v = 0.0
theta = 0.0
theta_dot = 0.0

# ============================================================
# STORAGE
# ============================================================

T = []
X = []
V = []
A = []
TH = []
X_LOAD = []

# ============================================================
# SIMULATION
# ============================================================

for k in range(N):
    t = k * dt

    a = acceleration_profile(t)

    # Trolley dynamics
    v = v + a * dt
    x = x + v * dt

    # Correct pendulum dynamics:
    # theta_ddot + (g/L) theta = -a/L
    theta_ddot = -(g / L) * theta - a / L
    theta_dot = theta_dot + theta_ddot * dt
    theta = theta + theta_dot * dt

    # Load horizontal position
    x_load = x + L * np.sin(theta)

    T.append(t)
    X.append(x)
    V.append(v)
    A.append(a)
    TH.append(theta)
    X_LOAD.append(x_load)

T = np.array(T)
X = np.array(X)
V = np.array(V)
A = np.array(A)
TH = np.array(TH)
X_LOAD = np.array(X_LOAD)

print(f"Final speed      = {V[-1]:.3f} m/s")
print(f"Max |theta|      = {np.rad2deg(np.max(np.abs(TH))):.3f} deg")
print(f"Final trolley x  = {X[-1]:.3f} m")
print(f"Final load x     = {X_LOAD[-1]:.3f} m")

# ============================================================
# PLOTS
# ============================================================

plt.figure(figsize=(8,4))
plt.plot(T, A, linewidth=2)
plt.axhline(a_max, linestyle="--", color="red", label=r"$a_{max}$")
plt.xlabel("Time [s]")
plt.ylabel("Acceleration [m/s²]")
plt.title("Smooth acceleration profile")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,4))
plt.plot(T, V, linewidth=2)
plt.axhline(v_max, linestyle="--", color="green", label=r"$v_{max}$")
plt.xlabel("Time [s]")
plt.ylabel("Velocity [m/s]")
plt.title("Trolley velocity")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,4))
plt.plot(T, np.rad2deg(TH), linewidth=2)
plt.xlabel("Time [s]")
plt.ylabel("Angle [deg]")
plt.title("Pendulum angle")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,4))
plt.plot(T, X, label="Trolley", linewidth=2)
plt.plot(T, X_LOAD, label="Load", linewidth=2)
plt.xlabel("Time [s]")
plt.ylabel("Horizontal position [m]")
plt.title("Trolley and load positions")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================
# ANIMATION
# ============================================================

fig, ax = plt.subplots(figsize=(10,4))
ax.set_xlim(-0.5, max(np.max(X) + 0.5, 1.0))
ax.set_ylim(-L - 0.3, 0.5)
ax.set_aspect('equal')
ax.grid(True)
ax.set_title("Crane with smooth acceleration and smooth release")

# rail
ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [0, 0], color="black", linewidth=2)

trolley_width = 0.12
trolley_line, = ax.plot([], [], linewidth=6)
cable_line, = ax.plot([], [], linewidth=2)
load_point, = ax.plot([], [], 'o', markersize=12)

time_text = ax.text(0.02, 0.92, "", transform=ax.transAxes)
theta_text = ax.text(0.02, 0.85, "", transform=ax.transAxes)

def init():
    trolley_line.set_data([], [])
    cable_line.set_data([], [])
    load_point.set_data([], [])
    time_text.set_text("")
    theta_text.set_text("")
    return trolley_line, cable_line, load_point, time_text, theta_text

def update(i):
    x_t = X[i]
    th = TH[i]

    x_pivot = x_t
    y_pivot = 0.0

    x_l = x_t + L * np.sin(th)
    y_l = -L * np.cos(th)

    trolley_line.set_data(
        [x_t - trolley_width/2, x_t + trolley_width/2],
        [0.0, 0.0]
    )
    cable_line.set_data([x_pivot, x_l], [y_pivot, y_l])
    load_point.set_data([x_l], [y_l])

    time_text.set_text(f"t = {T[i]:.2f} s")
    theta_text.set_text(f"theta = {np.rad2deg(th):.2f} deg")

    return trolley_line, cable_line, load_point, time_text, theta_text

step = 10
real_interval = dt * step * 1000   # 20 ms

ani = FuncAnimation(
    fig,
    update,
    frames=range(0, len(T), step),
    init_func=init,
    interval=real_interval,
    blit=True
)

plt.tight_layout()
plt.show()

# Optional save:
# ani.save("smooth_accel_release.gif", writer="pillow", fps=30)