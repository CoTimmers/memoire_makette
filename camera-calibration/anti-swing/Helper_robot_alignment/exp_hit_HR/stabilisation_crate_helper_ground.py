import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# Files to compare
# --------------------------------------------------

tests = {
    "Nominal": "essaie_840_z07_03.csv",
    "Mass +1 kg": "essaie_850_Lvrai107_9kg.csv",
    "Cable + 5%": "essaie_850_Lvrai107.csv",
    # "Test 4": "essaie_4.csv",
    # "Test 5": "essaie_5.csv",
}

data = {}

for name, filename in tests.items():

    df = pd.read_csv(filename)

    # Absolute crate position errors
    df["ex_abs"] = np.abs(df["ex"])
    df["ey_abs"] = np.abs(df["ey"])

    # Total crate position error
    df["e_norm"] = np.sqrt(
        df["ex"]**2 + df["ey"]**2
    )

    # Helper rotation in degrees
    df["helper_yaw_deg"] = np.degrees(df["hdyaw"])

    data[name] = df


# --------------------------------------------------
# 1. Crate position error
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

plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 2. Target reached
# --------------------------------------------------

plt.figure(figsize=(9, 4))

for name, df in data.items():
    plt.step(
        df["t"],
        df["dans_cible"],
        where="post",
        label=name
    )

plt.xlabel("Time [s]")
plt.ylabel("Target state")
plt.title("Target Detection")

plt.yticks([0, 1], ["Outside", "Inside"])
plt.ylim(-0.1, 1.1)

plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 3. Helper robot rotation
# --------------------------------------------------

plt.figure(figsize=(9, 5))

for name, df in data.items():
    plt.plot(
        df["t"],
        df["helper_yaw_deg"],
        label=name
    )

# Validation criterion
plt.axhline(
    1,
    linestyle="--",
    color="black",
    label="+1° limit"
)

plt.axhline(
    -1,
    linestyle="--",
    color="black",
    label="-1° limit"
)

plt.xlabel("Time [s]")
plt.ylabel("Rotation [deg]")
plt.title("Helper Robot Rotation")

plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()