import numpy as np
import matplotlib.pyplot as plt


def overshoot_ratio(zeta: float) -> float:
    """
    Compute the standard second-order overshoot ratio:
        M_p = exp(-zeta*pi / sqrt(1-zeta^2))

    Parameters
    ----------
    zeta : float
        Damping ratio, must satisfy 0 < zeta < 1.

    Returns
    -------
    float
        Overshoot ratio M_p.
    """
    if not (0 < zeta < 1):
        raise ValueError("zeta must be between 0 and 1 (exclusive).")
    return np.exp(-zeta * np.pi / np.sqrt(1.0 - zeta**2))


def acceleration_limit_from_angle(
    theta_lim_deg: float,
    zeta: float,
    g: float = 9.81,
    n_transitions: int = 4,
) -> float:
    """
    Compute the maximum acceleration from the worst-case sway-angle bound:

        theta_peak,worst ≈ n_transitions * theta_eq * (1 + M_p)

    with:
        theta_eq = a / g

    Hence:
        a_max = g * theta_lim / (n_transitions * (1 + M_p))

    Parameters
    ----------
    theta_lim_deg : float
        Maximum allowed angle in degrees.
    zeta : float
        Damping ratio.
    g : float
        Gravitational acceleration.
    n_transitions : int
        Number of acceleration transitions. Default is 4 for trapezoidal profile.

    Returns
    -------
    float
        Maximum allowed acceleration [m/s^2].
    """
    theta_lim_rad = np.deg2rad(theta_lim_deg)
    Mp = overshoot_ratio(zeta)
    a_max = g * theta_lim_rad / (n_transitions * (1.0 + Mp))
    return a_max


def generate_motion_profile(
    D: float,
    v_max: float = 0.1,
    theta_lim_deg: float = 4.0,
    zeta: float = 0.02,
    dt: float = 0.01,
    g: float = 9.81,
):
    """
    Generate a 1D motion profile over distance D using the vibration-limited
    acceleration and a maximum velocity constraint.

    The profile is:
    - trapezoidal if D >= v_max^2 / a_max
    - triangular otherwise

    Parameters
    ----------
    D : float
        Travel distance [m]. Must be positive.
    v_max : float
        Maximum allowed velocity [m/s].
    theta_lim_deg : float
        Maximum allowed sway angle [deg].
    zeta : float
        Damping ratio.
    dt : float
        Sampling time [s].
    g : float
        Gravitational acceleration [m/s^2].

    Returns
    -------
    dict
        Dictionary containing:
        - "time"
        - "acceleration"
        - "velocity"
        - "position"
        - "a_max"
        - "profile_type"
        - "t_acc"
        - "t_const"
        - "t_total"
        - "v_peak"
    """
    if D <= 0:
        raise ValueError("D must be strictly positive.")
    if v_max <= 0:
        raise ValueError("v_max must be strictly positive.")
    if dt <= 0:
        raise ValueError("dt must be strictly positive.")

    # Vibration-limited acceleration
    a_max = acceleration_limit_from_angle(
        theta_lim_deg=theta_lim_deg,
        zeta=zeta,
        g=g,
        n_transitions=4,
    )

    # Distance needed to reach v_max with accel + decel
    D_min_trap = v_max**2 / a_max

    if D >= D_min_trap:
        # Trapezoidal profile
        profile_type = "trapezoidal"

        t_acc = v_max / a_max
        t_const = (D - D_min_trap) / v_max
        t_total = 2.0 * t_acc + t_const
        v_peak = v_max

        t1 = t_acc
        t2 = t_acc + t_const
        tf = t_total

        time = np.arange(0.0, tf + dt, dt)
        acc = np.zeros_like(time)
        vel = np.zeros_like(time)
        pos = np.zeros_like(time)

        x1 = 0.5 * a_max * t1**2
        x2 = x1 + v_max * (t2 - t1)

        for i, t in enumerate(time):
            if t <= t1:
                # Acceleration phase
                acc[i] = a_max
                vel[i] = a_max * t
                pos[i] = 0.5 * a_max * t**2
            elif t <= t2:
                # Constant velocity phase
                acc[i] = 0.0
                vel[i] = v_max
                pos[i] = x1 + v_max * (t - t1)
            else:
                # Deceleration phase
                tau = t - t2
                acc[i] = -a_max
                vel[i] = v_max - a_max * tau
                pos[i] = x2 + v_max * tau - 0.5 * a_max * tau**2

        # Force exact final values numerically
        vel[-1] = 0.0
        pos[-1] = D

    else:
        # Triangular profile
        profile_type = "triangular"

        # Use maximum allowed acceleration and compute reachable peak velocity
        a = a_max
        t_acc = np.sqrt(D / a)
        t_const = 0.0
        t_total = 2.0 * t_acc
        v_peak = a * t_acc

        t1 = t_acc
        tf = t_total

        time = np.arange(0.0, tf + dt, dt)
        acc = np.zeros_like(time)
        vel = np.zeros_like(time)
        pos = np.zeros_like(time)

        x_mid = 0.5 * a * t1**2

        for i, t in enumerate(time):
            if t <= t1:
                # Acceleration phase
                acc[i] = a
                vel[i] = a * t
                pos[i] = 0.5 * a * t**2
            else:
                # Deceleration phase
                tau = t - t1
                acc[i] = -a
                vel[i] = v_peak - a * tau
                pos[i] = x_mid + v_peak * tau - 0.5 * a * tau**2

        # Force exact final values numerically
        vel[-1] = 0.0
        pos[-1] = D

    return {
        "time": time,
        "acceleration": acc,
        "velocity": vel,
        "position": pos,
        "a_max": a_max,
        "profile_type": profile_type,
        "t_acc": t_acc,
        "t_const": t_const,
        "t_total": t_total,
        "v_peak": v_peak,
    }


def plot_profile(profile: dict) -> None:
    """
    Plot acceleration, velocity, and position versus time.
    """
    t = profile["time"]
    a = profile["acceleration"]
    v = profile["velocity"]
    x = profile["position"]

    plt.figure(figsize=(10, 4))
    plt.plot(t, a)
    plt.xlabel("Time [s]")
    plt.ylabel("Acceleration [m/s²]")
    plt.title("Acceleration profile")
    plt.grid(True)
    plt.tight_layout()

    plt.figure(figsize=(10, 4))
    plt.plot(t, v)
    plt.xlabel("Time [s]")
    plt.ylabel("Velocity [m/s]")
    plt.title("Velocity profile")
    plt.grid(True)
    plt.tight_layout()

    plt.figure(figsize=(10, 4))
    plt.plot(t, x)
    plt.xlabel("Time [s]")
    plt.ylabel("Position [m]")
    plt.title("Position profile")
    plt.grid(True)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    # ===== USER INPUT =====
    D = 0.10              # distance to travel [m]
    v_max = 0.1          # maximum velocity [m/s]
    theta_lim_deg = 4.0  # maximum sway angle [deg]
    zeta = 0.02          # damping ratio
    dt = 0.01            # time step [s]

    profile = generate_motion_profile(
        D=D,
        v_max=v_max,
        theta_lim_deg=theta_lim_deg,
        zeta=zeta,
        dt=dt,
    )

    print("Profile type      :", profile["profile_type"])
    print("a_max [m/s^2]     :", profile["a_max"])
    print("v_peak [m/s]      :", profile["v_peak"])
    print("t_acc [s]         :", profile["t_acc"])
    print("t_const [s]       :", profile["t_const"])
    print("t_total [s]       :", profile["t_total"])

    # Velocity data at each time-step
    print("\nVelocity samples:")
    for t, v in zip(profile["time"], profile["velocity"]):
        print(f"t = {t:7.3f} s | v = {v:8.5f} m/s")

    plot_profile(profile)