"""Analyse one or several runs of 8_main.py.

    python analyse_run.py run1.csv
    python analyse_run.py a.csv b.csv        -> superimposes several runs
    python analyse_run.py                    -> asks for the names
    python analyse_run.py *.csv              -> all of them

Four quantities are plotted, one per panel:

    1. sway speed        L * theta_dot, compared to a limit
    2. helper travel     displacement of the helper since its reference pose
    3. helper speed      derivative of the helper pose
    4. commanded speed   what the control law asked of the crane

Sway speed
----------
The load hangs from the cable, so its speed is the trolley speed plus what the
oscillation adds:

    v_load = v_tcp + L * theta_dot

The first term is the intended transport, the second is the sway. The sway term
is the one worth bounding: it survives once the trolley has stopped, and it is
what sets the residual impact speed.

    v_sway = L * theta_dot          [m/s], two components

No subtraction is needed to isolate it. theta_dot is measured relative to the
trolley, so L * theta_dot is already the sway contribution alone; computing
v_load - v_tcp would return exactly the same thing.

theta_dot comes from the estimator, not from the camera: it is a filter state,
already smoothed, and it inherits the free-pendulum model. After a contact the
model is wrong for a few tenths of a second, so theta_dot should be read with
caution over that window.

Helper speed
------------
The log does not carry it, it has to be differentiated. The control loop runs at
500 Hz but the camera only delivers a fresh pose at the video rate, around 30 Hz.
The helper columns are therefore staircases, each step repeated a dozen or so
times. Differentiating row by row would give a spike at every step and zero in
between, which means nothing physically. Only the instants where the measurement
actually changes are kept, and the result is then smoothed.

Smoothing has a cost: it spreads the impact front. The half-width of the window
is printed in the figure title, and it is the time resolution of the speed.
Taking it too wide clips the very peak being measured.

Impact instant
--------------
The first rising edge of hbouge, i.e. the first crossing of SEUIL_BOUGE. That is
a threshold on displacement, so it necessarily lands slightly after the physical
contact: the helper must first move by SEUIL_BOUGE. The lag is roughly
SEUIL_BOUGE divided by the initial helper speed.

Trolley speed
-------------
Read from the log when the run recorded it (columns vtcp_x, vtcp_y), and
differentiated from the logged TCP position otherwise. Prefer logging it: at
500 Hz the raw derivative of an encoder position is very noisy, and the command
is not a substitute since it is what was asked, not what happened. In 8_main.py,
the header

    w.writerow([..., "tcp_x", "tcp_y", "vtcp_x", "vtcp_y", ...])

and, in the loop, *v_tcp next to *tcp. The variable already exists.
"""

import sys
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

# ---------------- settings ----------------
L_CABLE       = 1.10    # cable length [m], must match L_VRAI of the run
SWAY_LIMIT    = 0.095   # sway speed not to be exceeded [m/s]
T_IGNORE      = 0.3     # start-up transient, excluded from the statistics [s]
HELPER_SMOOTH = 2       # half-width of the helper speed window [vision samples]
TCP_SMOOTH    = 25      # half-width of the TCP position window [samples]

SWAY_COLS   = ("t", "th_x", "th_y", "thd_x", "thd_y")
HELPER_COLS = ("hx", "hy", "hdx", "hdy", "hdyaw", "hbouge")


# ---------------- helpers ----------------
def load(path):
    """Read one CSV. Returns None if it is unusable."""
    d = np.genfromtxt(path, delimiter=",", names=True, encoding="utf-8")
    if d.size == 0:
        print(f"{path}: empty, skipped")
        return None
    missing = [c for c in SWAY_COLS if c not in d.dtype.names]
    if missing:
        print(f"{path}: columns {missing} absent, skipped")
        return None
    return d


def smooth(x, half):
    """Moving average, correctly normalised at the edges."""
    n = 2 * half + 1
    if half < 1 or len(x) < n:
        return x
    kernel = np.ones(n)
    weight = np.convolve(np.ones_like(x, dtype=float), kernel, mode="same")
    return np.convolve(x, kernel, mode="same") / weight


def tcp_speed(d):
    """Trolley speed: from the log if present, differentiated otherwise."""
    names = d.dtype.names
    if "vtcp_x" in names and "vtcp_y" in names:
        return d["vtcp_x"], d["vtcp_y"], "encoder"
    if "tcp_x" not in names or "tcp_y" not in names:
        return None, None, "absent"
    t = d["t"]
    vx = np.gradient(smooth(d["tcp_x"], TCP_SMOOTH), t)
    vy = np.gradient(smooth(d["tcp_y"], TCP_SMOOTH), t)
    return vx, vy, "derived"


def fresh_samples(x, y):
    """Boolean mask of the rows where the vision pose actually changed."""
    fresh = np.ones(len(x), dtype=bool)
    fresh[1:] = (np.diff(x) != 0) | (np.diff(y) != 0)
    return fresh


def helper_speed(t, x, y, half=HELPER_SMOOTH):
    """Differentiate the helper pose, keeping only the real vision samples.

    Returns (t_meas, vx, vy) at the measurement instants. The differences are
    centred, hence valid even when the video rate is not perfectly regular,
    which happens as soon as the detection misses a frame.
    """
    idx = np.flatnonzero(fresh_samples(x, y))
    if len(idx) < 3:
        return np.array([]), np.array([]), np.array([])

    tm, xm, ym = t[idx], x[idx], y[idx]
    if half > 0 and len(tm) > 2 * half + 1:
        xm, ym = smooth(xm, half), smooth(ym, half)
        # the edges are biased by the truncated kernel, cut them off
        tm, xm, ym = tm[half:-half], xm[half:-half], ym[half:-half]

    return tm, np.gradient(xm, tm), np.gradient(ym, tm)


def impact_time(t, hbouge):
    """First rising edge of hbouge, or (None, None) if never crossed."""
    i = np.flatnonzero(hbouge > 0.5)
    return (float(t[i[0]]), int(i[0])) if len(i) else (None, None)


def fmt(value, spec, width):
    """Format a value, or a dash of the same width when it is missing."""
    return format("--" if value is None else format(value, spec), f">{width}s")


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
fig, ax = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
i_sway, i_travel, i_hspeed, i_cmd = 0, 1, 2, 3
colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]

header = (f"\n{'run':20s} {'t imp':>7s} {'sway max':>9s} {'sway imp':>9s} "
          f"{'% over':>7s} {'|d| max':>8s} {'|d| end':>8s} {'yaw end':>8s} "
          f"{'|vh| max':>9s} {'cmd max':>8s} {'vision':>7s}")
print(header)
print("-" * (len(header) - 1))

tcp_all_logged = True

for k, (name, d) in enumerate(runs):
    c = colours[k % len(colours)]
    t = d["t"]
    useful = t > T_IGNORE

    # ---- sway speed ----
    vs_x = L_CABLE * d["thd_x"]
    vs_y = L_CABLE * d["thd_y"]
    vs = np.hypot(vs_x, vs_y)
    vs_u = vs[useful]
    over = 100.0 * np.mean(vs_u > SWAY_LIMIT)

    # ---- trolley speed, kept as a sanity check on the magnitudes ----
    vt_x, vt_y, source = tcp_speed(d)
    tcp_all_logged = tcp_all_logged and source == "encoder"

    # ---- helper ----
    has_helper = all(c_ in d.dtype.names for c_ in HELPER_COLS)
    if has_helper:
        dx, dy = 1000 * d["hdx"], 1000 * d["hdy"]        # mm
        dn = np.hypot(dx, dy)
        yaw = np.degrees(d["hdyaw"])
        tm, vhx, vhy = helper_speed(t, d["hx"], d["hy"])
        vh = np.hypot(vhx, vhy) if len(tm) else np.array([])
        t_imp, i_imp = impact_time(t, d["hbouge"])
        n_vis = int(fresh_samples(d["hx"], d["hy"]).sum())
        f_vis = n_vis / t[-1] if t[-1] > 0 else 0.0
    else:
        print(f"{name}: helper columns absent, panels 2 and 3 skipped")
        dn = yaw = vh = np.array([])
        tm = np.array([])
        t_imp, i_imp, f_vis = None, None, None

    # ---- panel 1: sway speed ----
    ax[i_sway].plot(t, 1000 * vs, color=c, lw=1.5, label=f"{name}  |v sway|")
    ax[i_sway].plot(t, 1000 * vs_x, color=c, lw=0.8, ls="--", alpha=.6)
    ax[i_sway].plot(t, 1000 * vs_y, color=c, lw=0.8, ls=":", alpha=.6)

    # ---- panel 2: helper travel ----
    if len(dn):
        ax[i_travel].plot(t, dn, color=c, lw=1.4, label=f"{name}  |d|")
        ax[i_travel].plot(t, dx, color=c, lw=0.9, ls="--", alpha=.7)
        ax[i_travel].plot(t, dy, color=c, lw=0.9, ls=":", alpha=.7)

    # ---- panel 3: helper speed ----
    if len(tm):
        ax[i_hspeed].plot(tm, 1000 * vh, color=c, lw=1.4, label=f"{name}  |v|")
        ax[i_hspeed].plot(tm, 1000 * vhx, color=c, lw=0.9, ls="--", alpha=.7)
        ax[i_hspeed].plot(tm, 1000 * vhy, color=c, lw=0.9, ls=":", alpha=.7)

    # ---- panel 4: commanded speed ----
    v_cmd = np.hypot(d["vcmd_x"], d["vcmd_y"])
    ax[i_cmd].plot(t, 1000 * v_cmd, color=c, lw=1.3, label=f"{name}  commanded")
    if vt_x is not None:
        ax[i_cmd].plot(t, 1000 * np.hypot(vt_x, vt_y), color=c, lw=0.9,
                       ls="--", alpha=.6, label=f"{name}  actual ({source})")

    # ---- impact marker on every panel ----
    if t_imp is not None:
        for a in ax:
            a.axvline(t_imp, color=c, lw=1.0, ls="-.", alpha=.6)

    print(f"{name:20s} "
          f"{fmt(t_imp, '6.2f', 7)} "
          f"{1000 * vs_u.max():8.1f}m "
          f"{fmt(None if i_imp is None else 1000 * vs[i_imp], '8.1f', 9)} "
          f"{over:6.1f}% "
          f"{fmt(dn.max() if len(dn) else None, '7.1f', 8)} "
          f"{fmt(dn[-1] if len(dn) else None, '7.1f', 8)} "
          f"{fmt(yaw[-1] if len(yaw) else None, '+7.2f', 8)} "
          f"{fmt(1000 * vh.max() if len(vh) else None, '8.1f', 9)} "
          f"{1000 * v_cmd.max():7.1f} "
          f"{fmt(f_vis, '5.1f', 6)}Hz")

# ---------------- cosmetics ----------------
ax[i_sway].axhline(1000 * SWAY_LIMIT, color="r", lw=1.4, ls="--",
                   label=f"limit {1000 * SWAY_LIMIT:.0f} mm/s")
ax[i_sway].set_ylabel("sway speed L·θ̇ [mm/s]")
ax[i_sway].legend(fontsize=7, ncol=2)
ax[i_travel].set_ylabel("helper travel [mm]")
ax[i_travel].legend(fontsize=7, ncol=max(1, len(runs)))
ax[i_hspeed].set_ylabel("helper speed [mm/s]")
ax[i_hspeed].legend(fontsize=7)
ax[i_cmd].set_ylabel("crane speed [mm/s]")
ax[i_cmd].set_xlabel("t [s]")
ax[i_cmd].legend(fontsize=7)
for a in ax:
    a.grid(alpha=.3)
    a.axhline(0, color="k", lw=.5)

fig.suptitle(f"Sway and helper   L = {L_CABLE:.2f} m   "
             f"limit {1000 * SWAY_LIMIT:.0f} mm/s   "
             f"(first {T_IGNORE:.1f} s ignored, helper speed smoothed over "
             f"±{HELPER_SMOOTH} vision samples)")
plt.tight_layout()

out = "run_" + "_".join(name for name, _ in runs[:3]) + ".png"
plt.savefig(out, dpi=130)

print("\nsway speed = L * theta_dot, the part of the load speed coming from the")
print("oscillation alone. Dash-dotted vertical line = first crossing of")
print("SEUIL_BOUGE, i.e. contact detected on the helper displacement.")
if not tcp_all_logged:
    print("NOTE: vtcp absent from at least one run, tool speed differentiated "
          "from the TCP position (noisier). Add vtcp_x, vtcp_y to the log.")
print(f"figure: {out}")
plt.show()