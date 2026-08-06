"""Anti-sway control law with a velocity command, and command limiting.

Structure, per horizontal axis:

    v_cmd = v_ref + v_fb
    v_fb  = K_x (x_ref - x) + K_theta theta_hat + K_thetadot theta_dot_hat

This is a feedforward trajectory with state feedback; it is not an LQR unless the
gains are obtained by minimising a quadratic cost. Here they come from pole
placement, in closed form.

Why the gains have these values
-------------------------------
The pendulum is driven by the acceleration, and the command is a velocity, so
substituting v into a = v_dot gives

    (1 + K_thetadot/L) theta_ddot
  + (2 zeta omega_n + K_theta/L) theta_dot
  +  omega_n^2 theta
  = -(1/L)(v_ref_dot + K_x e_x_dot)

Hence, and this is specific to a velocity command:

  - K_theta adds damping, because differentiating the velocity turns theta into
    theta_dot. It is the anti-sway term.
  - K_thetadot changes the effective inertia, hence the natural frequency.
    Keeping the natural frequency means K_thetadot = 0.
  - K_x sets the position loop, which is first order with a velocity command:
    x_dot = K_x (x_ref - x), so the pole sits at -K_x and K_x = omega_t.

Keeping omega_n and imposing a damping ratio zeta_cl therefore gives

    K_x        = omega_t
    K_theta    = 2 L omega_n (zeta_cl - zeta)
    K_thetadot = 0

Sign convention, consistent with frames.py and estimator.py: theta > 0 when the
load is displaced towards positive x relative to the trolley. K_theta > 0 then
makes the trolley move towards the side the load has swung to, which is what
removes energy from the pendulum.

Preliminary values, to be validated experimentally: zeta_cl = 0.7, omega_t = 1.
"""

from __future__ import annotations

import numpy as np

G = 9.81


class Gains:
    """Feedback gains, computed once from the cable length."""

    def __init__(self,
                 L: float = 1.11,          # cable length [m]
                 zeta: float = 0.00228,      # measured natural damping [-]
                 zeta_cl: float = 0.7,       # damping wanted in closed loop [-]
                 omega_t: float = 1.0,       # position loop bandwidth [rad/s]
                 omega_cl: float | None = None,   # None keeps the natural frequency
                 verbose: bool = True):
        if L <= 0:
            raise ValueError("L must be positive.")
        self.L, self.zeta = L, zeta
        self.omega_n = np.sqrt(G / L)

        if omega_cl is None or abs(omega_cl - self.omega_n) < 1e-9:
            self.K_thetadot = 0.0
            omega_cl = self.omega_n
        else:
            # changing the frequency costs a permanent restoring term
            self.K_thetadot = L * ((self.omega_n / omega_cl) ** 2 - 1.0)

        f = 1.0 + self.K_thetadot / L
        self.K_theta = L * (2 * zeta_cl * omega_cl * f - 2 * zeta * self.omega_n)
        self.K_x = omega_t
        self.omega_cl, self.zeta_cl = omega_cl, zeta_cl

        if verbose:
            print(f"L = {L:.4f} m   omega_n = {self.omega_n:.4f} rad/s   "
                  f"T_n = {2*np.pi/self.omega_n:.4f} s")
            # print(f"K_x = {self.K_x:.4f} 1/s        position pole at "
            #       f"{-omega_t:.2f} rad/s, tau = {1/omega_t:.2f} s")
            tau = f"tau = {1/omega_t:.2f} s" if omega_t > 1e-9 else "pas de boucle de position"
            print(f"K_x = {self.K_x:.4f} 1/s        position pole at "
                  f"{-omega_t:.2f} rad/s, {tau}")
            print(f"K_theta = {self.K_theta:.4f} m/s per rad   "
                  f"({np.radians(1)*self.K_theta*1000:.1f} mm/s per degree)")
            print(f"K_thetadot = {self.K_thetadot:.4f} m per rad")
            print(f"closed-loop sway: zeta = {zeta_cl:.2f}, "
                  f"omega = {omega_cl:.3f} rad/s, "
                  f"half-life = {np.log(2)/(zeta_cl*omega_cl):.2f} s")

    def as_tuple(self):
        return self.K_x, self.K_theta, self.K_thetadot


def feedback(erreur, theta, theta_dot, gains: Gains, v_fb_max: float) -> np.ndarray:
    """Feedback velocity on both axes.

    erreur     [ex, ey]   position error x_ref - x, base axes   [m]
    theta      [thx, thy] estimated sway angles                 [rad]
    theta_dot  [.., ..]   estimated sway rates                  [rad/s]
    """
    v = (gains.K_x * np.asarray(erreur, float)
         + gains.K_theta * np.asarray(theta, float)
         + gains.K_thetadot * np.asarray(theta_dot, float))
    return np.clip(v, -v_fb_max, v_fb_max)


class CommandLimiter:
    """Enforce velocity, acceleration and jerk limits on the velocity command."""

    def __init__(self, Ts: float, v_max: float, a_max: float,
                 jerk_max: float | None = None, n_axes: int = 2):
        self.Ts, self.v_max, self.a_max, self.jerk_max = Ts, v_max, a_max, jerk_max
        self.v_prev = np.zeros(n_axes)
        self.a_prev = np.zeros(n_axes)
        self.n_sat = 0
        self.n_appels = 0

    def __call__(self, v_raw) -> np.ndarray:
        v_raw = np.asarray(v_raw, float)
        self.n_appels += 1

        a = (v_raw - self.v_prev) / self.Ts
        if self.jerk_max is not None:
            j = (a - self.a_prev) / self.Ts
            a = self.a_prev + np.clip(j, -self.jerk_max, self.jerk_max) * self.Ts
        a = np.clip(a, -self.a_max, self.a_max)

        v = np.clip(self.v_prev + a * self.Ts, -self.v_max, self.v_max)
        if np.any(np.abs(v - v_raw) > 1e-9):
            self.n_sat += 1

        self.a_prev = (v - self.v_prev) / self.Ts
        self.v_prev = v
        return v

    def regle(self, a_max: float | None = None, jerk_max: float | None = ...,
              v_max: float | None = None) -> None:
        """Change the limits between phases.

        Experiment 7 needs one deceleration for transport and a much harsher one
        for the deliberate stop that launches the load. The limiter has to know
        about it, not just the robot: a_applied feeds the Kalman prediction, so a
        limiter still modelling 0.8 m/s2 while the tool brakes at 4 would leave
        the filter with the wrong input exactly when the sway is created.
        """
        if a_max is not None:
            self.a_max = a_max
        if jerk_max is not Ellipsis:
            self.jerk_max = jerk_max
        if v_max is not None:
            self.v_max = v_max

    @property
    def a_applied(self) -> np.ndarray:
        """Acceleration actually applied, to be fed to the Kalman filter."""
        return self.a_prev

    @property
    def taux_saturation(self) -> float:
        return self.n_sat / max(self.n_appels, 1)


class FinDeMouvement:
    """Dwell-time test on the whole state.

    Preliminary thresholds, to be validated experimentally.
    """

    def __init__(self, eps_x=0.010, eps_theta=np.radians(1.0),
                 eps_theta_dot=0.010, eps_v=0.005, T_dwell=0.5):
        self.eps = (eps_x, eps_theta, eps_theta_dot, eps_v)
        self.T_dwell = T_dwell
        self.t_ok = None

    def __call__(self, erreur, theta, theta_dot, vitesse, t: float) -> bool:
        ex, eth, ethd, ev = self.eps
        ok = (np.linalg.norm(erreur) < ex
              and np.max(np.abs(theta)) < eth
              and np.max(np.abs(theta_dot)) < ethd
              and np.linalg.norm(vitesse) < ev)
        if not ok:
            self.t_ok = None
            return False
        if self.t_ok is None:
            self.t_ok = t
        return (t - self.t_ok) >= self.T_dwell

    def reset(self):
        self.t_ok = None


class Integrateur:
    """Integral term on the position error, to overcome static friction.

    A proportional law leaves a steady-state error whenever a constant
    resistance opposes the motion: at equilibrium the command K_x * e is too
    small to break the friction and the load stops short of the target. The
    integral accumulates that error until the command is large enough.

    It is deliberately not active during the planned motion: the feedforward
    already produces the displacement there, and accumulating would cause an
    overshoot. Friction only matters at the end of the move, at low speed.
    """

    def __init__(self, K_i: float = 0.15, v_max: float = 0.05, Ts: float = 1 / 500):
        self.K_i, self.v_max, self.Ts = K_i, v_max, Ts
        self.I = np.zeros(2)
        self.actif = False

    def __call__(self, erreur, actif: bool) -> np.ndarray:
        if not actif:
            self.actif = False
            return np.zeros(2)
        if not self.actif:          # first activation: start from zero
            self.I = np.zeros(2)
            self.actif = True
        self.I += np.asarray(erreur, float) * self.Ts
        borne = self.v_max / self.K_i if self.K_i > 1e-12 else 0.0
        self.I = np.clip(self.I, -borne, borne)
        return self.K_i * self.I

    def reset(self):
        self.I = np.zeros(2)
        self.actif = False


# ---------------------------------------------------------------- self-test
if __name__ == "__main__":
    L, zeta, Ts = 0.5225, 0.00228, 1 / 125
    g = Gains(L=L, zeta=zeta, zeta_cl=0.7, omega_t=1.0)

    # closed-loop check on the nonlinear pendulum, released at 8 deg
    dt, T = 1e-4, 6.0
    n = int(T / dt)
    x, v_prev, th, thd = 0.0, 0.0, np.radians(8.0), 0.0
    Kx, Kth, Kthd = g.as_tuple()
    th0 = abs(th)
    traj = np.zeros(n)
    for k in range(n):
        v = Kx * (0.0 - x) + Kth * th + Kthd * thd
        a = (v - v_prev) / dt
        v_prev = v
        thdd = -(G / L) * np.sin(th) - 2 * zeta * g.omega_n * thd - (a / L) * np.cos(th)
        thd += dt * thdd
        th += dt * thd
        x += dt * v
        traj[k] = abs(th)

    # time after which the sway stays below 50 % and 5 % of its initial value
    def temps_sous(frac):
        seuil = frac * th0
        idx = np.where(traj > seuil)[0]
        return idx[-1] * dt if len(idx) else 0.0

    t50, t95 = temps_sous(0.5), temps_sous(0.05)
    print(f"\nsway decay, nonlinear simulation from {np.degrees(th0):.0f} deg")
    print(f"  half-life   measured {t50:.3f} s   expected "
          f"{np.log(2)/(g.zeta_cl*g.omega_cl):.3f} s")
    print(f"  95 % decay  measured {t95:.3f} s   expected "
          f"{3.0/(g.zeta_cl*g.omega_cl):.3f} s")
    print(f"  residual after 6 s: {np.degrees(traj[-1]):.4f} deg")
    print(f"  trolley excursion : {x:.4f} m")

    lim = CommandLimiter(Ts, v_max=0.15, a_max=0.5, jerk_max=5.0)
    for _ in range(5):
        v = lim([1.0, 0.0])          # ask far beyond the limits
    print(f"\nlimiter after 5 steps at v_raw = 1 m/s: v = {np.round(lim.v_prev, 4)}"
          f"   a_applied = {np.round(lim.a_applied, 3)}"
          f"   saturated {100*lim.taux_saturation:.0f} % of calls")