import pandas as pd
import matplotlib.pyplot as plt

blue = pd.read_csv("drop_30-30_2_aligned.csv")
orange = pd.read_csv("drop_50-20_aligned.csv")

# --- Height relative to the successful stacked position ---
plt.figure(figsize=(9, 5))
plt.plot(blue["t"], blue["height_relative_mm"], label="(20, 30)mm drop)", color="tab:blue")
plt.plot(orange["t"], orange["height_relative_mm"], label="(50, 20)mm drop)", color="tab:orange")
plt.axhline(0, linewidth=1, linestyle="--", color="black")
plt.xlabel("Time [s]")
plt.ylabel("Crate height [mm]")
plt.title("Drop test - Crate height")
plt.grid(True, alpha=0.3)
plt.legend()
plt.ylim(-15, 40)
plt.tight_layout()
plt.show()

# --- Yaw angle ---
plt.figure(figsize=(9, 5))
plt.plot(blue["t"], blue["yaw_deg"], label="(20, 30)mm drop)", color="tab:blue")
plt.plot(orange["t"], orange["yaw_deg"], label="(50, 20)mm drop)", color="tab:orange")
plt.xlabel("Time [s]")
plt.ylabel("Yaw [deg]")
plt.title("Drop test - Crate Yaw angle")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
