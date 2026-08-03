import numpy as np
import matplotlib.pyplot as plt

# Parameters
v0 = 0.08
d_acc = 0.20
d2 = 0.40

# Acceleration phase
s_acc = np.linspace(0, d_acc, 100)
v_acc = v0 * s_acc / d_acc

# Constant-speed phase
s_const = np.linspace(d_acc, d2, 100)
v_const = np.full_like(s_const, v0)

# Add the vertical drop at d2
s = np.concatenate([s_acc, s_const, [d2, d2]])
v = np.concatenate([v_acc, v_const, [v0, 0]])

# Plot
plt.figure(figsize=(12, 7))
plt.plot(s, v, linewidth=2.5, label=r"$v_{\mathrm{crane}}(s)$")
plt.axvline(d2, color='red', linestyle='--', linewidth=2, label=r"$d_2$")
plt.axhline(v0, color='gray', linestyle=':', linewidth=2, label=r"$v_0$")

# Text
plt.text(0.02, v0 + 0.002, r"$v_0 = 0.08\ \mathrm{m/s}$", fontsize=16)

# Labels and style
plt.xlabel("Distance traveled by the crane [m]", fontsize=16)
plt.ylabel("Crane velocity [m/s]", fontsize=16)
plt.xlim(-0.02, 0.42)
plt.ylim(0, 0.10)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=16, loc="center left")

plt.tight_layout()
plt.savefig("crane_velocity_profile.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.show()