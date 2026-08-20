import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

 
tests = {
    "Nominal": "essaie_58_target_fixed.csv",
    "Mass +1 kg": "essaie+56_target_fixed.csv",
    "Cable +5 %": "essaie_54_2_target_fixed.csv",
}

data = {}

for name, filename in tests.items():

    df = pd.read_csv(filename)

    # Absolute position errors
    df["ex_abs"] = np.abs(df["ex"])
    df["ey_abs"] = np.abs(df["ey"])

    # Total position error
    df["e_norm"] = np.sqrt(
        df["ex"]**2 + df["ey"]**2
    )

    # Total commanded velocity
    df["vcmd_norm"] = np.sqrt(
        df["vcmd_x"]**2 + df["vcmd_y"]**2
    )

    data[name] = df


# --------------------------------------------------
# 1. Position errors
# --------------------------------------------------

plt.figure(figsize=(9, 5))

for name, df in data.items():
    plt.plot(
        df["t"],
        df["e_norm"],
        label=name
    )

plt.xlabel("Time [s]")
plt.ylabel("Position error [m]")
plt.title("Crate Position Error")
plt.xlim(0, 10)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 2. Commanded crane velocity
# --------------------------------------------------

plt.figure(figsize=(9, 5))

for name, df in data.items():
    plt.plot(
        df["t"],
        df["vcmd_norm"],
        label=name
    )

plt.xlabel("Time [s]")
plt.ylabel("Commanded velocity [m/s]")
plt.title("Crane Commanded Velocity")
plt.xlim(0, 10)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 3. Target detection
# --------------------------------------------------

plt.figure(figsize=(9, 5))

for name, df in data.items():
    plt.step(
        df["t"],
        df["dans_cible"],
        where="post",
        label=name
    )

plt.xlabel("Time [s]")
plt.ylabel("Target state")
plt.yticks([0, 1], ["Outside", "Inside"])
plt.ylim(-0.1, 1.1)
plt.xlim(0, 10)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()





# --------------------------------------------------
# 4. Crate error trajectory + TCP trajectory
# --------------------------------------------------

plt.figure(figsize=(8, 7))

for name, df in data.items():

    # Crate trajectory based on position error
    ex = 1000 * df["ex"]
    ey = 1000 * df["ey"]

    plt.plot(
        ex,
        ey,
        label=f"{name} - COM Position"
    )

    # TCP displacement relative to its initial position
    tcp_x = 1000 * (df["tcp_x"] - df["tcp_x"].iloc[0])
    tcp_y = 1000 * (df["tcp_y"] - df["tcp_y"].iloc[0])

    plt.plot(
        tcp_x,
        tcp_y,
        linestyle="--",
        linewidth=0.8,
        label=f"{name} - TCP"
    )


plt.xlabel("X [mm]")
plt.ylabel("Y [mm]")
plt.title("Crate and TCP XY Trajectories")

plt.xlim(-600, 200)
plt.ylim(-300, 250)

plt.gca().set_aspect("equal", adjustable="box")

plt.grid(True, alpha=0.3)
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()