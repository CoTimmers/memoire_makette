"""Trapezoidal velocity profile whose ramps are shaped to the pendulum period,
plus the open-ended variant used to launch the load deliberately.

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

sans_decel
----------
Experiment 7 needs the opposite of a clean arrival: the crane must reach the end
of the plan at full cruise speed and then be stopped dead, so that the load
carries on and swings into the crate. With sans_decel=True the profile is a
shaped ramp followed by cruise, and it simply stops being defined at |d|; the
caller is responsible for cutting the command. The shaped ramp still matters,
because it guarantees the load is hanging straight at the moment of the stop,
which is what makes the launch repeatable.

What the stop produces. Before it, trolley and load share the velocity v and the
cable is vertical. After it the trolley is fixed and the load still carries v, so
the pendulum starts from theta = 0 with theta_dot = v / L. Its reach is

    R = v sqrt(L / g)          horizontal, measured from the stopping point
    theta_max = v / sqrt(g L)

and if an obstacle stands at a distance d < R along the direction of travel, the
load meets it at

    v_contact = v sqrt(1 - (d / R)^2)

Note how fast that collapses as d approaches R: contact right at the top of the
arc happens at nearly zero speed, and a few millimetres further out it does not
happen at all.
"""

from __future__ import annotations

import numpy as np

G = 9.81


def portee_ballant(v: float, L: float) -> float:
    """Horizontal reach of the load after the trolley is stopped dead, [m]."""
    return v * np.sqrt(L / G)


def angle_ballant(v: float, L: float) -> float:
    """Peak sway angle after the same stop, [rad]."""
    return v / np.sqrt(G * L)


def vitesse_contact(v: float, L: float, d: float) -> float:
    """Speed at which the load meets an obstacle d ahead of the stop, [m/s].

    Returns 0.0 when the load cannot reach that far.
    """
    R = portee_ballant(v, L)
    if R <= 1e-12 or d >= R:
        return 0.0
    return float(v * np.sqrt(1.0 - (d / R) ** 2))


def vitesse_pour_contact(v_contact: float, L: float, d: float) -> float:
    """Cruise speed needed to meet an obstacle d ahead at v_contact, [m/s]."""
    k = np.sqrt(L / G)
    return float(np.hypot(v_contact, d / k))


class Trajectoire:
    """Straight move with ramps lasting one damped period.

        traj = Trajectoire([dx, dy], v_max=0.10, t1=1.45)
        p, v, a = traj(t)          # 2D vectors, relative to the start point

    With sans_decel=True there is no braking ramp: the profile holds v_max from
    the end of the acceleration ramp onwards, and terminee() marks the instant
    the planned distance has been covered.
    """

    def __init__(self, deplacement, v_max: float, t1: float,
                 t0: float = 0.0, accorde: bool = True, a_max: float = 0.5,
                 sans_decel: bool = False):
        d = np.asarray(deplacement, float)
        self.D = float(np.linalg.norm(d))
        self.direction = d / self.D if self.D > 1e-9 else np.zeros_like(d)
        self.t0, self.t1 = t0, t1
        self.sans_decel = sans_decel

        if not accorde:                       # plain trapezoid, for comparison
            self.v = min(v_max, np.sqrt(self.D * a_max))
            self.a = a_max
            self.t_ramp = self.v / a_max
        elif sans_decel:
            # One ramp only. The peak speed is what the launch depends on, so it
            # is never lowered; a move too short for the ramp is an error.
            self.t_ramp = t1
            self.a = v_max / t1
            self.v = v_max
            if self.D < 0.5 * self.a * t1 ** 2:
                raise ValueError(
                    f"move of {1000*self.D:.0f} mm too short: the shaped ramp "
                    f"alone covers {500*self.a*t1**2:.0f} mm. Lower v_max or "
                    f"start further away.")
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
        n_rampes = 1 if sans_decel else 2
        self.t_cst = ((self.D - n_rampes * self.d_ramp) / self.v
                      if self.v > 1e-12 else 0.0)
        self.t_cst = max(0.0, self.t_cst)
        self.duree = n_rampes * self.t_ramp + self.t_cst

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
        elif self.sans_decel:
            # past the planned distance: hold the cruise speed, so that being a
            # cycle late on the cut does not create a spurious deceleration
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
        if self.sans_decel:
            return (f"<Trajectoire {1000*self.D:.0f} mm in {self.duree:.2f} s | "
                    f"ramp {self.t_ramp:.3f} s, a={self.a:.4f} m/s2, "
                    f"cruise {self.t_cst:.2f} s, ends at v={self.v:.3f} m/s>")
        phase = "with cruise" if self.t_cst > 1e-6 else "no cruise"
        return (f"<Trajectoire {1000*self.D:.0f} mm in {self.duree:.2f} s | "
                f"ramp {self.t_ramp:.3f} s, a={self.a:.4f} m/s2, "
                f"v={self.v:.3f} m/s, {phase}>")


# ---------------------------------------------------------------- self-test
if __name__ == "__main__":

    def simule(traj, L, t_fin, t_arret=None, zeta=0.00228):
        """Nonlinear pendulum driven by the profile. Returns t, theta, x_charge.

        t_arret, if given, is the instant the velocity command is cut to zero.
        Integration is semi-implicit, so no artificial energy appears.
        """
        dt = 1e-4
        n = int(t_fin / dt)
        wn = np.sqrt(G / L)
        th = thd = 0.0
        v_prev = 0.0
        x_tr = 0.0
        T = np.zeros(n); TH = np.zeros(n); XC = np.zeros(n)
        for k in range(n):
            t = k * dt
            _, vv, _ = traj(t)
            vx = float(vv[0])
            if t_arret is not None and t >= t_arret:
                vx = 0.0
            a = (vx - v_prev) / dt
            v_prev = vx
            x_tr += dt * vx
            thdd = -(G / L) * np.sin(th) - 2 * zeta * wn * thd - (a / L) * np.cos(th)
            thd += dt * thdd
            th += dt * thd                       # thd already updated
            T[k], TH[k], XC[k] = t, th, x_tr + L * np.sin(th)
        return T, TH, XC

    # ---- part 1: the classical result, ramp = T_d cancels the sway ----
    for L in (0.5225, 1.11):
        wn = np.sqrt(G / L)
        T_d = 2 * np.pi / wn
        print(f"\n=== L = {L} m   omega_n = {wn:.4f} rad/s   T_d = {T_d:.4f} s ===")
        for nom, kw in [("ramp = T_d/2 (wrong)", dict(t1=T_d / 2)),
                        ("ramp = T_d  (correct)", dict(t1=T_d)),
                        ("plain trapezoid      ", dict(t1=T_d, accorde=False))]:
            tr = Trajectoire([0.30, 0.10], v_max=0.10, **kw)
            t, th, _ = simule(tr, L, tr.duree + 5.0)
            res = np.degrees(np.max(np.abs(th[t > tr.duree + 1.0])))
            print(f"  {nom}  peak {np.degrees(np.abs(th).max()):5.2f} deg   "
                  f"residual {res:6.3f} deg   {tr.duree:5.2f} s")

    # ---- part 2: experiment 7, launch by stopping dead ----
    L, V, D_MOVE, D3 = 1.11, 0.15, 0.35, 0.050
    T_d = 2 * np.pi * np.sqrt(L / G)
    print(f"\n=== experiment 7: L = {L} m, cruise {V} m/s, move {1000*D_MOVE:.0f} mm ===")
    tr = Trajectoire([D_MOVE, 0.0], v_max=V, t1=T_d, sans_decel=True)
    print(f"  {tr}")

    R = portee_ballant(V, L)
    print(f"  predicted reach      {1000*R:6.1f} mm   "
          f"theta_max {np.degrees(angle_ballant(V, L)):5.2f} deg")
    print(f"  predicted contact at {1000*D3:.0f} mm: "
          f"{1000*vitesse_contact(V, L, D3):5.1f} mm/s")

    t, th, xc = simule(tr, L, tr.duree + 4.0, t_arret=tr.duree)
    avant = t < tr.duree
    apres = t >= tr.duree
    x_arret = xc[apres][0]
    print(f"  sway during transport {np.degrees(np.abs(th[avant]).max()):5.2f} deg"
          f"   (shaping working if near zero)")
    print(f"  simulated reach       {1000*(xc[apres].max() - x_arret):6.1f} mm")
    i_pic = np.argmax(xc[apres])
    print(f"  time to the top       {t[apres][i_pic] - tr.duree:6.3f} s   "
          f"(T_d/4 = {T_d/4:.3f} s)")

    # ---- part 3: choosing the geometry ----
    print(f"\n  v      reach    theta_max   contact at {1000*D3:.0f} mm")
    for v in (0.10, 0.12, 0.15, 0.18, 0.20, 0.25):
        r = portee_ballant(v, L)
        vc = vitesse_contact(v, L, D3)
        etat = f"{1000*vc:6.1f} mm/s" if vc > 0 else "  no contact"
        print(f"  {v:.2f}  {1000*r:7.1f} mm   {np.degrees(angle_ballant(v, L)):5.2f} deg"
              f"   {etat}")

    print(f"\n  d3 needed for a chosen contact speed, at v = {V} m/s")
    for frac in (0.3, 0.5, 0.7, 0.9):
        d = R * np.sqrt(1 - frac ** 2)
        print(f"    contact at {1000*frac*V:6.1f} mm/s  ->  d3 = {1000*d:5.1f} mm")