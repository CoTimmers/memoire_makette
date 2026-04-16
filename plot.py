import numpy as np
import matplotlib.pyplot as plt

# =========================
# Parameters
# =========================
v_fast = 0.5   # [m/s]
v_slow = 0.3   # [m/s]
a_max  = 0.085  # [m/s^2]

d_total = 1.50  # [m]
d_near  = 0.05  # [m]

dt = 0.01       # [s]

# =========================
# Distances needed for speed transitions
# =========================
d_fs = (v_fast**2 - v_slow**2) / (2 * a_max)   # fast -> slow
d_s0 = (v_slow**2) / (2 * a_max)               # slow -> stop

print(f"Distance fast -> slow : {d_fs:.4f} m")
print(f"Distance slow -> stop : {d_s0:.4f} m")

# =========================
# Check feasibility
# =========================
if d_total < d_near + d_fs:
    print("Warning: Not enough distance to reach and maintain v_fast before the near zone.")

if d_near < d_s0:
    print("Warning: Near zone is too short to stop smoothly from v_slow.")

# =========================
# Key positions
# =========================
x1 = d_fs                    # end of acceleration to v_fast
x2 = d_total - d_near - d_fs # start of deceleration fast -> slow
x3 = d_total - d_s0          # start of deceleration slow -> 0
x4 = d_total                 # final target

x1 = max(x1, 0)
x2 = max(x2, x1)
x3 = max(x3, x2)
x4 = max(x4, x3)

# =========================
# Build profile by simulation
# =========================
t = 0.0
x = 0.0
v = 0.0

T = [t]
X = [x]
V = [v]

while x < d_total:
    # Desired speed according to current position
    if x < x1:
        v_ref = v_fast
    elif x < x2:
        v_ref = v_fast
    elif x < x3:
        v_ref = v_slow
    else:
        v_ref = 0.0

    # Acceleration-limited update
    if v < v_ref:
        v = min(v + a_max * dt, v_ref)
    elif v > v_ref:
        v = max(v - a_max * dt, v_ref)

    # Position update
    x = x + v * dt
    t = t + dt

    T.append(t)
    X.append(x)
    V.append(v)

    if t > 200:
        print("Warning: Simulation stopped because it took too long.")
        break

T = np.array(T)
X = np.array(X)
V = np.array(V)

# =========================
# Figure 1: velocity vs time
# =========================
plt.figure(figsize=(8, 5), facecolor="white")
plt.plot(T, V * 1000, linewidth=2, label="Velocity")
plt.axhline(v_fast * 1000, linestyle="--", color="red", label=r"$v_{fast}$")
plt.axhline(v_slow * 1000, linestyle="--", color="green", label=r"$v_{slow}$")

plt.xlabel("Time [s]")
plt.ylabel("Velocity [mm/s]")
plt.title("Velocity profile vs time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("path_3.png", dpi=300, facecolor="white", bbox_inches="tight")
plt.show()

# =========================
# Figure 2: velocity vs position
# =========================
plt.figure(figsize=(8, 5), facecolor="white")
plt.plot(X, V * 1000, linewidth=2)
plt.axvline(d_total - d_near, linestyle="--", label="Near zone begins")
plt.axvline(d_total, linestyle="--", label="Target")

plt.xlabel("Position [m]")
plt.ylabel("Velocity [mm/s]")
plt.title("Velocity profile as a function of position")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("velocity_vs_position.png", dpi=300, facecolor="white", bbox_inches="tight")
plt.show()

# =========================
# Figure 3: position vs time
# =========================
plt.figure(figsize=(8, 5), facecolor="white")
plt.plot(T, X, linewidth=2)

plt.xlabel("Time [s]")
plt.ylabel("Position [m]")
plt.title("Position evolution")
plt.grid(True)
plt.tight_layout()
plt.savefig("position_vs_time.png", dpi=300, facecolor="white", bbox_inches="tight")
plt.show()