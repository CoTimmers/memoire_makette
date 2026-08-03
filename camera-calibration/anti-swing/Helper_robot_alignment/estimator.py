"""Kalman estimation of the sway angle and rate, with camera delay compensation.

Model, per sway plane, small angles:

    theta_ddot + 2 zeta omega_n theta_dot + omega_n^2 theta = -a / L
    omega_n = sqrt(g / L)

State z = [theta, theta_dot]^T, input a = trolley acceleration, output y = theta.

    A = [[0, 1], [-omega_n^2, -2 zeta omega_n]]      B = [[0], [-1/L]]
    C = [1, 0]

Discretisation is exact (matrix exponential), not Euler.

Sign convention, consistent with frames.py: theta > 0 when the load is displaced
towards positive x relative to the trolley. Accelerating towards +x therefore
pushes the load backwards, hence the minus sign in B.

Delay handling. A camera measurement carries the timestamp of the instant the
image was taken, which is earlier than the instant it is processed. The filter
keeps a history of predicted states, covariances and applied accelerations; when
a delayed measurement arrives, the correction is applied at the matching past
instant and the state is then repropagated to the present using the stored
accelerations.

Preliminary values, to be tuned and validated experimentally:
    sigma_a     = 0.05 m/s^2   uncertainty on the applied acceleration
    sigma_theta = 0.2 deg      camera noise on the angle
    P0          = diag(0.01^2, 0.1^2)

Typical use, one control cycle:

    est.predict(a_applied, t_now)          # every cycle, 125 Hz
    if new_frame:
        est.update(theta_measured, t_frame)   # only when an image arrives
    theta, theta_dot = est.state
"""

from __future__ import annotations

import numpy as np
from collections import deque
from scipy.linalg import expm

G = 9.81


def _discretise(A: np.ndarray, B: np.ndarray, Ts: float):
    """Exact zero-order-hold discretisation via the matrix exponential."""
    n = A.shape[0]
    M = np.zeros((n + B.shape[1], n + B.shape[1]))
    M[:n, :n] = A
    M[:n, n:] = B
    Md = expm(M * Ts)
    return Md[:n, :n], Md[:n, n:]


def _process_noise(A: np.ndarray, B: np.ndarray, sigma_a: float, Ts: float):
    """Discrete process noise from a white acceleration uncertainty (Van Loan)."""
    n = A.shape[0]
    Qc = B @ B.T * sigma_a ** 2
    M = np.zeros((2 * n, 2 * n))
    M[:n, :n] = -A
    M[:n, n:] = Qc
    M[n:, n:] = A.T
    E = expm(M * Ts)
    Ad = E[n:, n:].T
    return Ad @ E[:n, n:]


class KalmanSway:
    """Kalman filter for one sway plane, with delayed-measurement handling."""

    def __init__(self,
                 L: float = 0.5225,          # cable length [m]
                 zeta: float = 0.00228,      # measured natural damping [-]
                 Ts: float = 1 / 125,        # control period [s], CB3
                 sigma_a: float = 0.05,      # acceleration uncertainty [m/s^2]
                 sigma_theta: float = np.radians(0.2),   # camera noise [rad]
                 P0: np.ndarray | None = None,
                 profondeur: float = 1.0):   # history depth [s]
        if L <= 0 or Ts <= 0:
            raise ValueError("L and Ts must be positive.")

        self.L, self.zeta, self.Ts = L, zeta, Ts
        self.omega_n = np.sqrt(G / L)

        self.A = np.array([[0.0, 1.0],
                           [-self.omega_n ** 2, -2 * zeta * self.omega_n]])
        self.B = np.array([[0.0], [-1.0 / L]])
        self.C = np.array([[1.0, 0.0]])

        self.Ad, self.Bd = _discretise(self.A, self.B, Ts)
        self.Q = _process_noise(self.A, self.B, sigma_a, Ts)
        self.R = np.array([[sigma_theta ** 2]])

        self.z = np.zeros((2, 1))
        # P0: theta is unknown until the first image, theta_dot even more so.
        self.P0 = (np.diag([np.radians(10.0) ** 2, 0.5 ** 2])
                   if P0 is None else np.array(P0, float))
        self.P = self.P0.copy()

        self.histo: deque = deque(maxlen=int(profondeur / Ts))
        self.innovation = 0.0
        self.gain = np.zeros(2)
        self.n_rejets = 0
        self.initialise = False

    # ---------- accessors ----------
    @property
    def state(self) -> tuple[float, float]:
        """(theta, theta_dot) in rad and rad/s."""
        return float(self.z[0, 0]), float(self.z[1, 0])

    @property
    def sigma(self) -> tuple[float, float]:
        """Current standard deviations of the estimate, rad and rad/s."""
        return float(np.sqrt(self.P[0, 0])), float(np.sqrt(self.P[1, 1]))

    # ---------- prediction, every control cycle ----------
    def predict(self, a_applied: float, t: float) -> None:
        """Propagate one step with the acceleration actually applied."""
        if not np.isfinite(a_applied):
            a_applied = 0.0
        self.z = self.Ad @ self.z + self.Bd * a_applied
        self.P = self.Ad @ self.P @ self.Ad.T + self.Q
        self.P = 0.5 * (self.P + self.P.T)            # keep it symmetric
        self.histo.append({"t": t,
                           "z": self.z.copy(),
                           "P": self.P.copy(),
                           "a": float(a_applied)})

    # ---------- correction, only when an image arrives ----------
    def update(self, theta_mes: float, t_mes: float,
               seuil_nis: float = 25.0) -> bool:
        """Correct with a measurement taken at t_mes (possibly in the past).

        Returns True if the measurement was accepted. Outliers are rejected on a
        normalised innovation squared test, which also protects against a marker
        briefly detected at the wrong place.
        """
        if not np.isfinite(theta_mes) or not self.histo:
            return False

        # locate the history entry matching the measurement instant
        i = min(range(len(self.histo)),
                key=lambda j: abs(self.histo[j]["t"] - t_mes))
        z, P = self.histo[i]["z"].copy(), self.histo[i]["P"].copy()

        # first image: start from it instead of from zero, otherwise the
        # outlier test would reject the only information able to correct the
        # initial guess.
        if not self.initialise:
            z = np.array([[theta_mes], [0.0]])
            P = np.diag([self.R[0, 0], self.P0[1, 1]])
            self.initialise = True
            self.histo[i]["z"], self.histo[i]["P"] = z.copy(), P.copy()
            self._repropage(i, z, P)
            return True

        y = np.array([[theta_mes]]) - self.C @ z
        S = self.C @ P @ self.C.T + self.R
        nis = (y.T @ np.linalg.inv(S) @ y).item()
        if nis > seuil_nis:                            # implausible measurement
            self.n_rejets += 1
            return False

        K = P @ self.C.T @ np.linalg.inv(S)
        z = z + K @ y
        I_KC = np.eye(2) - K @ self.C
        P = I_KC @ P @ I_KC.T + K @ self.R @ K.T       # Joseph form
        P = 0.5 * (P + P.T)

        self.innovation = float(y[0, 0])
        self.gain = K.flatten()

        self.histo[i]["z"], self.histo[i]["P"] = z.copy(), P.copy()
        self._repropage(i, z, P)
        return True

    def _repropage(self, i: int, z: np.ndarray, P: np.ndarray) -> None:
        """Carry a correction made at index i forward to the present."""
        for j in range(i + 1, len(self.histo)):
            a = self.histo[j - 1]["a"]
            z = self.Ad @ z + self.Bd * a
            P = self.Ad @ P @ self.Ad.T + self.Q
            P = 0.5 * (P + P.T)
            self.histo[j]["z"], self.histo[j]["P"] = z.copy(), P.copy()
        self.z, self.P = z, P


class SwayEstimator2D:
    """Two independent filters, one per horizontal axis."""

    def __init__(self, **kwargs):
        self.x = KalmanSway(**kwargs)
        self.y = KalmanSway(**kwargs)

    def predict(self, a_applied, t: float) -> None:
        self.x.predict(float(a_applied[0]), t)
        self.y.predict(float(a_applied[1]), t)

    def update(self, theta_mes, t_mes: float) -> tuple[bool, bool]:
        return (self.x.update(float(theta_mes[0]), t_mes),
                self.y.update(float(theta_mes[1]), t_mes))

    @property
    def theta(self) -> np.ndarray:
        return np.array([self.x.state[0], self.y.state[0]])

    @property
    def theta_dot(self) -> np.ndarray:
        return np.array([self.x.state[1], self.y.state[1]])


# ---------------------------------------------------------------- self-test
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    L, zeta, Ts = 0.5225, 0.00228, 1 / 125
    est = KalmanSway(L=L, zeta=zeta, Ts=Ts)

    print(f"omega_n = {est.omega_n:.4f} rad/s   T_n = {2*np.pi/est.omega_n:.4f} s")
    print(f"A_d =\n{np.round(est.Ad, 6)}")
    print(f"B_d = {np.round(est.Bd.flatten(), 8)}")
    print(f"Q   =\n{est.Q}")
    print(f"R   = {est.R[0,0]:.4e} rad^2   (sigma = "
          f"{np.degrees(np.sqrt(est.R[0,0])):.2f} deg)\n")

    # truth: free pendulum released at 8 deg, no command
    dt, T = Ts, 6.0
    n = int(T / dt)
    th, thd = np.radians(8.0), 0.0
    f_cam, retard, sig = 30.0, 0.035, np.radians(0.2)
    t_next = 0.0
    err_th, err_thd = [], []
    tampon = []

    for k in range(n):
        t = k * dt
        # plant, semi-implicit integration: no artificial energy on an oscillator
        thdd = -(G / L) * np.sin(th) - 2 * zeta * est.omega_n * thd
        thd += dt * thdd
        th += dt * thd          # thd already updated -> symplectic

        est.predict(0.0, t)

        if t >= t_next:                       # a frame is taken now
            tampon.append((t, th + rng.normal(0, sig)))
            t_next += 1.0 / f_cam
        # it becomes available 'retard' later
        while tampon and t >= tampon[0][0] + retard:
            t_img, y = tampon.pop(0)
            est.update(y, t_img)

        if t > 1.0:                           # skip the convergence transient
            err_th.append(est.state[0] - th)
            err_thd.append(est.state[1] - thd)

    print(f"RMS error on theta     : {np.degrees(np.std(err_th)):.4f} deg")
    print(f"RMS error on theta_dot : {np.std(err_thd):.4f} rad/s")
    print(f"final sigma            : {np.degrees(est.sigma[0]):.4f} deg, "
          f"{est.sigma[1]:.4f} rad/s")
    print(f"rejected measurements  : {est.n_rejets}")