"""Analyse the motion of the helper during a run of 8_main.py.

    python analyse_helper.py run8_ff+fb_v150_143022.csv
    python analyse_helper.py a.csv b.csv          -> superimposes several runs
    python analyse_helper.py                      -> asks for the names

What the file contains and what has to be done with it
------------------------------------------------------
For every cycle of the control loop, 8_main.py logs:

    hx, hy      helper pose in the world frame               [m]
    hdx, hdy    displacement from the reference pose         [m]
    hdyaw       rotation from the reference pose             [rad]
    hbouge      1 as soon as |hd| exceeds SEUIL_BOUGE

The speed is not in there, it has to be differentiated. Careful: the loop runs
at 500 Hz but the camera only delivers a fresh pose at the video rate, around
30 Hz. The helper columns are therefore staircases, each step repeated a dozen
or so times. Differentiating row by row gives a spike at every step and zero in
between, which means nothing physically. So only the instants where the
measurement really changes are differentiated, and the result is then smoothed.

Smoothing has a cost: it spreads the impact front. The half-width of the window
is displayed, and it is the time resolution of the speed. Taking it too wide
clips the very speed peak that is being measured.

The impact instant kept is the first rising edge of hbouge, that is, the first
crossing of SEUIL_BOUGE. It is a threshold on displacement, so it necessarily
comes slightly after the physical contact: the helper first has to move by
SEUIL_BOUGE. The lag is roughly SEUIL_BOUGE divided by the initial speed of the
helper.
"""

import sys
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

# half-width of the speed smoothing window, in vision samples
SMOOTHING = 2

HELPER_COLS = ("hx", "hy", "hdx", "hdy", "hdyaw", "hbouge")


def load(path):
    d = np.genfromtxt(path, delimiter=",", names=True)
    if d.size == 0:
        return None
    missing = [c for c in HELPER_COLS if c not in d.dtype.names]
    if missing:
        print(f"{path}: columns {missing} absent, run skipped "
              f"(log produced by an earlier version of 8_main.py)")
        return None
    return d


def helper_speed(t, x, y, half=SMOOTHING):
    """Differentiate the helper pose, keeping only the real vision samples.

    Returns (t_meas, vx, vy) at the measurement instants. The differences are
    centred, hence valid even if the video rate is not perfectly regular, which
    is the case as soon as the detection misses a frame.
    """
    # a fresh measurement = the pose changed since the previous row
    fresh = np.ones(len(t), dtype=bool)
    fresh[1:] = (np.diff(x) != 0) | (np.diff(y) != 0)
    idx = np.flatnonzero(fresh)
    if len(idx) < 3:
        return np.array([]), np.array([]), np.array([])

    tm, xm, ym = t[idx], x[idx], y[idx]

    if half > 0 and len(tm) > 2 * half + 1:
        kernel = np.ones(2 * half + 1) / (2 * half + 1)
        xm = np.convolve(xm, kernel, mode="same")
        ym = np.convolve(ym, kernel, mode="same")
        # the edges are biased by the truncated kernel, cut them off
        tm, xm, ym = tm[half:-half], xm[half:-half], ym[half:-half]

    vx = np.gradient(xm, tm)
    vy = np.gradient(ym, tm)
    return tm, vx, vy


def impact_time(t, hbouge):
    """First rising edge of hbouge, or None if the threshold is never crossed."""
    i = np.flatnonzero(hbouge > 0.5)
    return float(t[i[0]]) if len(i) else None


# ---------------- input ----------------
files = sys.argv[1:]
if not files:
    typed = input("CSV files to analyse (space separated, "
                  "Enter = all): ").strip()
    files = typed.split() if typed else sorted(glob.glob("*.csv"))

files = [f for f in files if os.path.exists(f)]
if not files:
    raise SystemExit("no file found.")

runs = []
for f in files:
    d = load(f)
    if d is not None:
        runs.append((os.path.splitext(os.path.basename(f))[0], d))
if not runs:
    raise SystemExit("no usable run.")

# ---------------- figure ----------------
fig, ax = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]

header = (f"\n{'run':22s} {'t imp':>7s} {'|d| max':>9s} {'|d| end':>9s} "
          f"{'dx end':>7s} {'dy end':>7s} {'yaw end':>8s} {'|v| max':>8s} "
          f"{'n vis':>6s} {'f vis':>6s}")
print(header)
print("-" * (len(header) - 1))

for k, (name, d) in enumerate(runs):
    c = colours[k % len(colours)]
    t = d["t"]
    dx, dy = 1000 * d["hdx"], 1000 * d["hdy"]       # mm
    dn = np.hypot(dx, dy)
    yaw = np.degrees(d["hdyaw"])

    tm, vx, vy = helper_speed(t, d["hx"], d["hy"])
    vn = np.hypot(vx, vy) if len(tm) else np.array([])

    t_imp = impact_time(t, d["hbouge"])

    # actual video rate, useful to judge the time resolution
    fresh = np.ones(len(t), dtype=bool)
    fresh[1:] = (np.diff(d["hx"]) != 0) | (np.diff(d["hy"]) != 0)
    n_vis = int(fresh.sum())
    f_vis = n_vis / t[-1] if t[-1] > 0 else 0.0

    # ---- panel 1: displacement ----
    ax[0].plot(t, dn, color=c, lw=1.4, label=f"{name}  |d|")
    ax[0].plot(t, dx, color=c, lw=0.9, ls="--", alpha=.7, label=f"{name}  x")
    ax[0].plot(t, dy, color=c, lw=0.9, ls=":", alpha=.7, label=f"{name}  y")

    # ---- panel 2: yaw ----
    ax[1].plot(t, yaw, color=c, lw=1.3, label=name)

    # ---- panel 3: helper speed ----
    if len(tm):
        ax[2].plot(tm, 1000 * vn, color=c, lw=1.4, label=f"{name}  |v|")
        ax[2].plot(tm, 1000 * vx, color=c, lw=0.9, ls="--", alpha=.7)
        ax[2].plot(tm, 1000 * vy, color=c, lw=0.9, ls=":", alpha=.7)

    # ---- panel 4: commanded speed ----
    ax[3].plot(t, 1000 * np.hypot(d["vcmd_x"], d["vcmd_y"]),
               color=c, lw=1.2, label=f"{name}  commanded speed")

    # impact mark on every panel
    if t_imp is not None:
        for a in ax:
            a.axvline(t_imp, color=c, lw=1.0, ls="-.", alpha=.6)

    v_max = 1000 * vn.max() if len(vn) else 0.0
    print(f"{name:22s} "
          f"{(f'{t_imp:6.2f}s' if t_imp is not None else '     --'):>7s} "
          f"{dn.max():7.1f}mm {dn[-1]:7.1f}mm {dx[-1]:+6.1f} {dy[-1]:+6.1f} "
          f"{yaw[-1]:+7.2f}d {v_max:7.1f} {n_vis:6d} {f_vis:5.1f}Hz")

ax[0].set_ylabel("helper displacement [mm]")
ax[0].axhline(0, color="k", lw=.5)
ax[0].legend(fontsize=7, ncol=max(1, len(runs)))
ax[1].set_ylabel("helper rotation [deg]")
ax[1].axhline(0, color="k", lw=.5)
ax[1].legend(fontsize=8)
ax[2].set_ylabel("helper speed [mm/s]")
ax[2].axhline(0, color="k", lw=.5)
ax[2].legend(fontsize=7)
ax[3].set_ylabel("commanded crane speed [mm/s]")
ax[3].set_xlabel("t [s]")
ax[3].legend(fontsize=8)
for a in ax:
    a.grid(alpha=.3)

fig.suptitle(f"Helper: displacement, rotation, speed   "
             f"(speed smoothing: +/-{SMOOTHING} video samples)")
plt.tight_layout()

out = "helper_" + "_".join(name for name, _ in runs[:3]) + ".png"
plt.savefig(out, dpi=130)
print(f"\ndash-dotted vertical line = first crossing of SEUIL_BOUGE")
print(f"figure: {out}")
plt.show()