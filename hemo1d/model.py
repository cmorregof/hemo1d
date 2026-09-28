"""
One-dimensional model of blood flow in a compliant vessel.

Unknowns (per vessel, per cross-section x):
    A(x, t)  cross-sectional area          [m^2]
    Q(x, t)  volumetric flow rate           [m^3/s]

Governing equations (Sherwin et al. 2003, Alastruey et al. 2011, Boileau et al. 2015):

    dA/dt + dQ/dx                                 = 0
    dQ/dt + d(alpha Q^2 / A)/dx + (A/rho) dp/dx   = -K_R Q / A

Tube law (linear elastic thin wall, constant reference area A0):

    p(A) = p_ext + beta/A0 * (sqrt(A) - sqrt(A0)),   beta = 4/3 sqrt(pi) E h

Friction:  K_R = 2 (zeta + 2) pi mu / rho   (zeta = 9 -> Womersley-like profile, zeta = 2 -> Poiseuille)

With alpha = 1 the system is strictly hyperbolic with wave speed c(A) = sqrt(beta/(2 rho A0)) A^{1/4}
and Riemann invariants  W_{1,2} = u +/- 4 (c - c0),  u = Q/A.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

RHO = 1060.0  # blood density [kg/m^3]
MU = 4.0e-3   # blood dynamic viscosity [Pa s]


@dataclass
class Vessel:
    """Geometry, material and discretisation of a single straight compliant vessel."""

    name: str
    length: float          # [m]
    radius: float          # reference radius [m]
    thickness: float       # wall thickness [m]
    young: float           # Young modulus [Pa]
    n_cells: int = 100
    rho: float = RHO
    mu: float = MU
    zeta: float = 9.0      # velocity-profile parameter (9: Boileau et al. 2015)
    p_ext: float = 0.0     # external pressure [Pa]

    # state (filled in by reset)
    x: np.ndarray = field(init=False, repr=False)
    A: np.ndarray = field(init=False, repr=False)
    Q: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.A0 = np.pi * self.radius**2
        self.beta = (4.0 / 3.0) * np.sqrt(np.pi) * self.young * self.thickness
        self.K_R = 2.0 * (self.zeta + 2.0) * np.pi * self.mu / self.rho
        self.dx = self.length / self.n_cells
        self.x = (np.arange(self.n_cells) + 0.5) * self.dx
        self.c0 = self.wave_speed(self.A0)
        self.reset()

    # ---- constitutive relations ------------------------------------------------
    def pressure(self, A):
        return self.p_ext + self.beta / self.A0 * (np.sqrt(A) - np.sqrt(self.A0))

    def area_from_pressure(self, p):
        return (np.sqrt(self.A0) + (p - self.p_ext) * self.A0 / self.beta) ** 2

    def wave_speed(self, A):
        return np.sqrt(self.beta / (2.0 * self.rho * self.A0)) * A**0.25

    def riemann(self, A, Q):
        u = Q / A
        c = self.wave_speed(A)
        return u + 4.0 * (c - self.c0), u - 4.0 * (c - self.c0)

    def from_riemann(self, W1, W2):
        c = self.c0 + (W1 - W2) / 8.0
        A = (c / np.sqrt(self.beta / (2.0 * self.rho * self.A0))) ** 4
        u = 0.5 * (W1 + W2)
        return A, u * A

    # ---- flux and source -----------------------------------------------------
    def flux(self, A, Q):
        return np.stack([Q, Q**2 / A + self.beta / (3.0 * self.rho * self.A0) * A**1.5])

    def source(self, A, Q):
        return np.stack([np.zeros_like(A), -self.K_R * Q / A])

    def reset(self):
        self.A = np.full(self.n_cells, self.A0)
        self.Q = np.zeros(self.n_cells)

    def max_speed(self):
        return np.max(np.abs(self.Q / self.A) + self.wave_speed(self.A))


def blood_vessel(name, length_cm, radius_cm, thickness_mm, young_kPa, n_cells=100, **kw) -> Vessel:
    """Convenience constructor in the units usually quoted in the literature."""
    return Vessel(name, length_cm * 1e-2, radius_cm * 1e-2, thickness_mm * 1e-3, young_kPa * 1e3, n_cells, **kw)
