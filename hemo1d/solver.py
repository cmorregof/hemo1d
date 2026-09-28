"""
Network solver: explicit two-step Lax-Wendroff (Richtmyer) in the interior of every vessel,
characteristic-based treatment of boundaries (prescribed inflow, 3-element Windkessel outflow,
non-reflecting outflow) and of bifurcations (conservation of mass + continuity of total pressure).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import brentq

from .model import Vessel


def newton_fd(fun, z0, tol=1e-11, max_iter=50, h=1e-7):
    """Newton's method with a forward-difference Jacobian and residual-based stopping."""
    z = np.asarray(z0, dtype=float).copy()
    r = fun(z)
    for _ in range(max_iter):
        if np.max(np.abs(r)) < tol:
            return z
        J = np.empty((z.size, z.size))
        for k in range(z.size):
            zp = z.copy()
            zp[k] += h
            J[:, k] = (fun(zp) - r) / h
        z = z - np.linalg.solve(J, r)
        r = fun(z)
    if np.max(np.abs(r)) < 1e3 * tol:
        return z
    raise RuntimeError(f"Newton did not converge, max residual {np.max(np.abs(r)):.3e}")


# ----------------------------------------------------------------------------- boundary objects
@dataclass
class Windkessel3:
    """Three-element (RCR) Windkessel: p1D - p_c = R1 Q ;  C dp_c/dt = Q - (p_c - p_out)/R2"""

    R1: float          # [Pa s / m^3]
    C: float           # [m^3 / Pa]
    R2: float          # [Pa s / m^3]
    p_out: float = 0.0  # [Pa]
    p_c: float = 0.0    # state: pressure across the capacitor

    def total_resistance(self):
        return self.R1 + self.R2

    def reset(self, p_c=0.0):
        self.p_c = p_c


@dataclass
class Junction:
    parent: str
    daughters: List[str]


# ----------------------------------------------------------------------------- network
class Network:
    """A tree of vessels with one inflow, Windkessel/non-reflecting outlets and bifurcations."""

    def __init__(self, vessels: List[Vessel], inflow: Callable[[float], float],
                 outlets: Dict[str, Optional[Windkessel3]], junctions: List[Junction] = (),
                 cfl: float = 0.5):
        self.vessels = {v.name: v for v in vessels}
        self.root = vessels[0].name
        self.inflow = inflow
        self.outlets = outlets        # vessel name -> Windkessel3 or None (non-reflecting)
        self.junctions = list(junctions)
        self.cfl = cfl
        self.t = 0.0
        # boundary states (A, Q) at the left/right ends of each vessel, at current time
        self.left: Dict[str, Tuple[float, float]] = {}
        self.right: Dict[str, Tuple[float, float]] = {}
        for v in vessels:
            self.left[v.name] = (v.A0, 0.0)
            self.right[v.name] = (v.A0, 0.0)

    # ---- time step ------------------------------------------------------------------
    def stable_dt(self):
        return self.cfl * min(v.dx / v.max_speed() for v in self.vessels.values())

    def step(self, dt: float):
        # 1) interior update, Richtmyer two-step Lax-Wendroff with ghost boundary states
        for name, v in self.vessels.items():
            AL, QL = self.left[name]
            AR, QR = self.right[name]
            A = np.concatenate([[AL], v.A, [AR]])
            Q = np.concatenate([[QL], v.Q, [QR]])
            U = np.stack([A, Q])
            F = v.flux(A, Q)
            S = v.source(A, Q)
            r = dt / v.dx
            # half step at cell interfaces
            Uh = 0.5 * (U[:, 1:] + U[:, :-1]) - 0.5 * r * (F[:, 1:] - F[:, :-1]) + 0.25 * dt * (S[:, 1:] + S[:, :-1])
            Fh = v.flux(Uh[0], Uh[1])
            Sh = v.source(Uh[0], Uh[1])
            Un = U[:, 1:-1] - r * (Fh[:, 1:] - Fh[:, :-1]) + 0.5 * dt * (Sh[:, 1:] + Sh[:, :-1])
            v.A, v.Q = Un[0].copy(), Un[1].copy()
        self.t += dt
        # 2) boundary states at the new time from characteristic extrapolation
        self._update_inlet(dt)
        for j in self.junctions:
            self._update_junction(j, dt)
        for name, wk in self.outlets.items():
            self._update_outlet(name, wk, dt)

    # ---- characteristic extrapolation --------------------------------------------------
    def _incoming_W2_left(self, v: Vessel, dt):
        """W2 (travelling leftwards) reaching x=0 at t+dt: foot of the characteristic inside the domain."""
        AL, QL = self.left[v.name]
        _, W2b = v.riemann(AL, QL)
        _, W2c = v.riemann(v.A[:2], v.Q[:2])
        lam = abs(QL / AL - v.wave_speed(AL))
        xf = lam * dt
        return np.interp(xf, [0.0, 0.5 * v.dx, 1.5 * v.dx], [W2b, W2c[0], W2c[1]])

    def _outgoing_W1_right(self, v: Vessel, dt):
        """W1 (travelling rightwards) reaching x=L at t+dt."""
        AR, QR = self.right[v.name]
        W1b, _ = v.riemann(AR, QR)
        W1c, _ = v.riemann(v.A[-2:], v.Q[-2:])
        lam = abs(QR / AR + v.wave_speed(AR))
        xf = lam * dt
        return np.interp(xf, [0.0, 0.5 * v.dx, 1.5 * v.dx], [W1b, W1c[-1], W1c[-2]])

    # ---- inlet: prescribed flow rate -------------------------------------------------------
    def _update_inlet(self, dt):
        v = self.vessels[self.root]
        Qin = self.inflow(self.t)
        W2 = self._incoming_W2_left(v, dt)

        def g(A):
            return Qin / A - 4.0 * (v.wave_speed(A) - v.c0) - W2

        A = brentq(g, 0.2 * v.A0, 5.0 * v.A0, xtol=1e-16)
        self.left[v.name] = (A, Qin)

    # ---- outlets --------------------------------------------------------------------------
    def _update_outlet(self, name, wk: Optional[Windkessel3], dt):
        v = self.vessels[name]
        W1 = self._outgoing_W1_right(v, dt)
        if wk is None:  # non-reflecting: W2 keeps its undisturbed value 0
            A, Q = v.from_riemann(W1, 0.0)
            self.right[name] = (A, Q)
            return

        def Q_of_A(A):
            return A * (W1 - 4.0 * (v.wave_speed(A) - v.c0))

        # trapezoidal (Crank-Nicolson) update of the Windkessel ODE, implicit in Q^{n+1}
        Q_old = self.right[name][1]
        a = dt / (2.0 * wk.C * wk.R2)

        def p_c_new(Q):
            return ((1.0 - a) * wk.p_c + dt / (2.0 * wk.C) * (Q + Q_old) + 2.0 * a * wk.p_out) / (1.0 + a)

        def g(A):
            Q = Q_of_A(A)
            return v.pressure(A) - wk.R1 * Q - p_c_new(Q)

        A = brentq(g, 0.2 * v.A0, 5.0 * v.A0, xtol=1e-16)
        Q = Q_of_A(A)
        wk.p_c = p_c_new(Q)
        self.right[name] = (A, Q)

    # ---- bifurcation ---------------------------------------------------------------------
    def _update_junction(self, j: Junction, dt):
        """Six unknowns (A, u at the parent outlet and at both daughter inlets), six equations:
        mass conservation, continuity of total pressure p + rho u^2/2 (x2), outgoing
        characteristic of the parent, incoming characteristic of each daughter."""
        p = self.vessels[j.parent]
        ds = [self.vessels[d] for d in j.daughters]
        W1p = self._outgoing_W1_right(p, dt)
        W2d = [self._incoming_W2_left(d, dt) for d in ds]
        vs = [p] + ds
        rho = p.rho
        pref = p.beta / p.A0 * np.sqrt(p.A0)      # pressure scale
        qref = p.A0 * p.c0                        # flow scale

        def unpack(z):
            A = [z[2 * k] * vs[k].A0 for k in range(3)]
            u = [z[2 * k + 1] * vs[k].c0 for k in range(3)]
            return A, u

        def residual(z):
            A, u = unpack(z)
            res = np.empty(6)
            res[0] = (A[0] * u[0] - A[1] * u[1] - A[2] * u[2]) / qref
            pt = [vs[k].pressure(A[k]) + 0.5 * rho * u[k] ** 2 for k in range(3)]
            res[1] = (pt[0] - pt[1]) / pref
            res[2] = (pt[0] - pt[2]) / pref
            res[3] = (u[0] + 4.0 * (p.wave_speed(A[0]) - p.c0) - W1p) / p.c0
            for k in (1, 2):
                res[3 + k] = (u[k] - 4.0 * (vs[k].wave_speed(A[k]) - vs[k].c0) - W2d[k - 1]) / vs[k].c0
            return res

        AR, QR = self.right[p.name]
        z0 = [AR / p.A0, QR / AR / p.c0]
        for d in ds:
            AL, QL = self.left[d.name]
            z0 += [AL / d.A0, QL / AL / d.c0]
        z = newton_fd(residual, np.array(z0))
        A, u = unpack(z)
        self.right[p.name] = (A[0], A[0] * u[0])
        for k, d in enumerate(ds):
            self.left[d.name] = (A[k + 1], A[k + 1] * u[k + 1])

    # ---- driver ---------------------------------------------------------------------------
    def run(self, t_end: float, probes: Dict[str, List[float]], dt: Optional[float] = None,
            sample_every: int = 1, t_start_record: float = 0.0):
        """Advance to t_end, recording (t, p, Q, A) at probe locations {vessel: [x positions in m]}."""
        rec = {"t": []}
        for name, xs in probes.items():
            for xk in xs:
                rec[(name, xk)] = {"p": [], "Q": [], "A": []}
        it = 0
        while self.t < t_end - 1e-12:
            h = self.stable_dt() if dt is None else dt
            h = min(h, t_end - self.t)
            self.step(h)
            it += 1
            if it % sample_every == 0 and self.t >= t_start_record:
                rec["t"].append(self.t)
                for name, xs in probes.items():
                    v = self.vessels[name]
                    for xk in xs:
                        if xk <= 0.0:
                            A, Q = self.left[name]
                        elif xk >= v.length:
                            A, Q = self.right[name]
                        else:
                            A = np.interp(xk, v.x, v.A)
                            Q = np.interp(xk, v.x, v.Q)
                        rec[(name, xk)]["p"].append(v.pressure(A))
                        rec[(name, xk)]["Q"].append(Q)
                        rec[(name, xk)]["A"].append(A)
        rec["t"] = np.asarray(rec["t"])
        for k, d in rec.items():
            if k != "t":
                for kk in d:
                    d[kk] = np.asarray(d[kk])
        return rec


# ----------------------------------------------------------------------------- 0D reference
def windkessel_0d(inflow: Callable[[float], float], wk: Windkessel3, t_end: float, dt: float,
                  C_extra: float = 0.0, p0: float = 0.0):
    """Pure lumped (0D) model: the whole vessel collapsed into the Windkessel, with its wall
    compliance C_extra added in parallel. Returns (t, p_inlet)."""
    n = int(round(t_end / dt))
    t = np.arange(n + 1) * dt
    p_c = np.empty(n + 1)
    p_c[0] = p0
    C = wk.C + C_extra
    for k in range(n):
        Q = inflow(t[k])
        p_c[k + 1] = p_c[k] + dt / C * (Q - (p_c[k] - wk.p_out) / wk.R2)
    Q = np.array([inflow(tk) for tk in t])
    return t, p_c + wk.R1 * Q
