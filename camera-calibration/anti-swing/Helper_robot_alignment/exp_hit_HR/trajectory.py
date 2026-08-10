"""Trapezoidal velocity profile whose ramps are shaped to the pendulum period.

Principle. A ramp of the velocity profile is a rectangular pulse of acceleration:
a step +a at its beginning, a step -a at its end. Each step excites the pendulum.
For the two to cancel, the delay between them must be a full damped period.

Proof, for the undamped case. The response to a step +a applied at t = 0 is

    theta_1(t) = -(a / L omega^2) [1 - cos(omega t)]

and the response to the step -a applied at t1 is the same with opposite sign and
delayed. After t1 their sum is

    theta(t) = (a / L omega^2) [cos(omega t) - cos(omega (t - t1))]

With t1 = T_d the two cosines are equal and the sway vanishes exactly. With
t1 = T_d / 2 the second cosine changes sign, the terms add, and the amplitude is
doubled: half a period is the worst possible choice, not the best.

    t1 = T_d = 2 pi / omega_d          ramp duration

Sizing, following the two cases:

    a_eff  = v_max / t1                       effective acceleration
    d_ramp = 0.5 a_eff t1^2                   distance covered by one ramp

    if |d| > 2 d_ramp   the cruise phase exists, v_peak = v_max
    else                no cruise phase, and the ramp duration is kept:
                        a_eff = |d| / t1^2,  v_peak = a_eff t1 = |d| / t1

So for short moves the peak velocity is reduced automatically, rather than the
ramp being shortened, which is what preserves the cancellation.
"""

from __future__ import annotations

import numpy as np


class Trajectoire:
    """Straight move with ramps lasting one damped period.

        traj = Trajectoire([dx, dy], v_max=0.10, t1=1.45)
        p, v, a = traj(t)          # 2D vectors, relative to the start point
    """

    def __init__(self, deplacement, v_max: float, t1: float,
                 t0: float = 0.0, accorde: bool = True, a_max: float = 0.5):
        d = np.asarray(deplacement, float)
        self.D = float(np.linalg.norm(d))
        self.direction = d / self.D if self.D > 1e-9 else np.zeros_like(d)
        self.t0, self.t1 = t0, t1

        if not accorde:                       # plain trapezoid, for comparison
            self.v = min(v_max, np.sqrt(self.D * a_max))
            self.a = a_max
            self.t_ramp = self.v / a_max
        else:
            self.t_ramp = t1
            a_eff = v_max / t1
            d_ramp = 0.5 * a_eff * t1 ** 2
            if self.D > 2 * d_ramp:           # long move: cruise phase exists
                self.a, self.v = a_eff, v_max
            else:                             # short move: lower the peak speed
                self.a = self.D / t1 ** 2
                self.v = self.a * t1

        self.d_ramp = 0.5 * self.a * self.t_ramp ** 2
        self.t_cst = ((self.D - 2 * self.d_ramp) / self.v) if self.v > 1e-12 else 0.0
        self.t_cst = max(0.0, self.t_cst)
        self.duree = 2 * self.t_ramp + self.t_cst

    def __call__(self, t: float):
        """Position, velocity and acceleration at time t, as 2D vectors."""
        tau = t - self.t0
        if tau <= 0:
            p = v = a = 0.0
        elif tau <= self.t_ramp:                                  # acceleration
            p, v, a = 0.5 * self.a * tau ** 2, self.a * tau, self.a
        elif tau <= self.t_ramp + self.t_cst:                     # cruise
            p = self.d_ramp + self.v * (tau - self.t_ramp)
            v, a = self.v, 0.0
        elif tau <= self.duree:                                   # deceleration
            tr = tau - self.t_ramp - self.t_cst
            p = (self.d_ramp + self.v * self.t_cst
                 + self.v * tr - 0.5 * self.a * tr ** 2)
            v, a = self.v - self.a * tr, -self.a
        else:                                                     # arrived
            p, v, a = self.D, 0.0, 0.0
        return p * self.direction, v * self.direction, a * self.direction

    def terminee(self, t: float) -> bool:
        return t >= self.t0 + self.duree

    def __repr__(self):
        phase = "with cruise" if self.t_cst > 1e-6 else "no cruise"
        return (f"<Trajectoire {1000*self.D:.0f} mm in {self.duree:.2f} s | "
                f"ramp {self.t_ramp:.3f} s, a={self.a:.4f} m/s2, "
                f"v={self.v:.3f} m/s, {phase}>")


# ---------------------------------------------------------------- self-test
if __name__ == "__main__":
    G = 9.81
    for L in (0.5225, 3.0):
        wn = np.sqrt(G / L)
        T = 2 * np.pi / wn
        print(f"\n=== L = {L} m   omega_n = {wn:.4f} rad/s   T_d = {T:.4f} s ===")

        def residuel(traj):
            dt = 1e-4
            n = int((traj.duree + 10.0) / dt)
            th = thd = v_prev = 0.0
            pic = res = 0.0
            for k in range(n):
                t = k * dt
                _, vv, _ = traj(t)
                vx = float(vv[0])
                a = (vx - v_prev) / dt
                v_prev = vx
                thdd = -(G / L) * np.sin(th) - (a / L) * np.cos(th)
                thd += dt * thdd
                th += dt * thd
                pic = max(pic, abs(th))
                if t > traj.duree + 1.0:
                    res = max(res, abs(th))
            return np.degrees(pic), np.degrees(res)

        for nom, kw in [("ramp = T_d/2 (wrong)", dict(t1=T / 2)),
                        ("ramp = T_d  (correct)", dict(t1=T)),
                        ("plain trapezoid      ", dict(t1=T, accorde=False))]:
            tr = Trajectoire([0.30, 0.10], v_max=0.10, **kw)
            p, r = residuel(tr)
            print(f"  {nom}  peak {p:5.2f} deg   residual {r:6.3f} deg   "
                  f"{tr.duree:5.2f} s")

        # short move: the peak speed is lowered, the ramp is preserved
        tr = Trajectoire([0.05, 0.0], v_max=0.10, t1=T)
        print(f"  short move            {tr}")
        p, r = residuel(tr)
        print(f"                          peak {p:5.2f} deg   residual {r:6.3f} deg")