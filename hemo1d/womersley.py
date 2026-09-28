"""
Full-order viscous model of the axial flow in a straight rigid vessel: the axisymmetric
Navier-Stokes equations for fully developed pulsatile flow reduce exactly to the linear problem

    du/dt = -(1/rho) dp/dz(t) + nu * (1/r) d/dr ( r du/dr ),     u(R, t) = 0,  du/dr(0, t) = 0,

whose solution resolves the radial velocity profile that the 1D model replaces by a fixed shape
(the parameter zeta in the friction term -K_R Q/A). Two solvers are provided:

  * the analytical Womersley (1955) solution, harmonic by harmonic (Bessel functions);
  * a finite-difference Crank-Nicolson solver in r, verified against the former.

Both give, for a prescribed flow waveform Q(t), the wall shear stress and hence the viscous
friction per unit length that the 1D momentum equation must model - which is what lets us
compare the reduced closure (zeta) with the full-order profile.
"""
from __future__ import annotations

import numpy as np
from scipy.special import jv


def j0(z):
    return jv(0, z)


def j1(z):
    return jv(1, z)

from .model import MU, RHO


# ----------------------------------------------------------------------------- analytical (Womersley)
def womersley_number(R, omega, nu):
    return R * np.sqrt(omega / nu)


def harmonic_profile(r, R, omega, G, rho=RHO, mu=MU):
    """Complex amplitude of u(r) for a pressure gradient -dp/dz = Re(G e^{i omega t}).
    For omega = 0 returns the Poiseuille profile."""
    nu = mu / rho
    if omega == 0.0:
        return G / (4.0 * mu) * (R**2 - r**2) + 0j
    lam = womersley_number(R, omega, nu) * 1j**1.5
    return G / (1j * omega * rho) * (1.0 - j0(lam * r / R) / j0(lam))


def harmonic_flow(R, omega, G, rho=RHO, mu=MU):
    """Complex amplitude of Q for a pressure gradient amplitude G."""
    nu = mu / rho
    if omega == 0.0:
        return np.pi * R**4 * G / (8.0 * mu) + 0j
    lam = womersley_number(R, omega, nu) * 1j**1.5
    return np.pi * R**2 * G / (1j * omega * rho) * (1.0 - 2.0 * j1(lam) / (lam * j0(lam)))


def harmonic_wall_shear(R, omega, G, rho=RHO, mu=MU):
    """Complex amplitude of the wall shear stress tau_w = -mu du/dr|_R (positive along the flow)."""
    nu = mu / rho
    if omega == 0.0:
        return G * R / 2.0 + 0j
    lam = womersley_number(R, omega, nu) * 1j**1.5
    # du/dr at R:  G/(i w rho) * (-(lam/R) * (-J1(lam)) / J0(lam)) = G/(i w rho) * lam J1(lam) / (R J0(lam))
    return -mu * G / (1j * omega * rho) * lam * j1(lam) / (R * j0(lam))


def womersley_from_flow(t, Q, R, rho=RHO, mu=MU, n_harmonics=20, r=None):
    """Given one period of a flow waveform Q(t) (uniformly sampled, t[0]=0, period = t[-1]+dt),
    reconstruct by Fourier synthesis: the pressure gradient -dp/dz(t), the wall shear tau_w(t),
    and (optionally) the velocity field u(r, t). Returns dict."""
    n = len(t)
    dt = t[1] - t[0]
    T = n * dt
    Qh = np.fft.rfft(Q) / n
    omegas = 2.0 * np.pi * np.arange(len(Qh)) / T
    G = np.zeros_like(Qh)
    tau = np.zeros_like(Qh)
    U = None if r is None else np.zeros((len(Qh), len(r)), dtype=complex)
    for k, om in enumerate(omegas):
        if k > n_harmonics:
            break
        G[k] = Qh[k] / harmonic_flow(R, om, 1.0, rho, mu)
        tau[k] = harmonic_wall_shear(R, om, G[k], rho, mu)
        if U is not None:
            U[k] = harmonic_profile(r, R, om, G[k], rho, mu)
    out = {"dpdz": -np.fft.irfft(G * n, n=n), "tau_w": np.fft.irfft(tau * n, n=n), "t": t}
    if U is not None:
        # u(r, t) = sum_k Re(U_k e^{i w_k t}), with the irfft convention
        out["u"] = np.fft.irfft(U * n, n=n, axis=0)
        out["r"] = r
    return out


# ----------------------------------------------------------------------------- finite differences
def axisymmetric_fd(R, dpdz, t_end, nr=80, dt=1e-3, rho=RHO, mu=MU):
    """Crank-Nicolson finite-difference solution of du/dt = -(1/rho) dpdz(t) + nu (u_rr + u_r / r)
    on r in [0, R] with u(R) = 0 and symmetry at r = 0 (L'Hopital: (1/r) u_r -> u_rr at r=0).
    dpdz is a callable returning dp/dz at time t (negative drives flow in +z).
    Returns t, r, u[t, r], Q[t], tau_w[t]."""
    nu = mu / rho
    r = np.linspace(0.0, R, nr + 1)
    dr = r[1] - r[0]
    N = nr  # unknowns u_0..u_{nr-1}; u_nr = 0
    # Laplacian operator L (N x N)
    L = np.zeros((N, N))
    L[0, 0], L[0, 1] = -4.0 / dr**2, 4.0 / dr**2            # r=0: 2 u_rr with symmetric ghost
    for i in range(1, N):
        L[i, i - 1] = 1.0 / dr**2 - 1.0 / (2.0 * r[i] * dr)
        L[i, i] = -2.0 / dr**2
        if i + 1 < N:
            L[i, i + 1] = 1.0 / dr**2 + 1.0 / (2.0 * r[i] * dr)
    I = np.eye(N)
    A = I - 0.5 * dt * nu * L
    B = I + 0.5 * dt * nu * L
    Ainv = np.linalg.inv(A)
    nt = int(round(t_end / dt))
    u = np.zeros(N)
    ts = np.arange(nt + 1) * dt
    U = np.zeros((nt + 1, nr + 1))
    for k in range(nt):
        f = -0.5 * dt / rho * (dpdz(ts[k]) + dpdz(ts[k + 1]))
        u = Ainv @ (B @ u + f)
        U[k + 1, :N] = u
    Q = 2.0 * np.pi * np.trapezoid(U * r, r, axis=1)
    # second-order one-sided derivative at the wall
    tau_w = -mu * (3.0 * U[:, -1] - 4.0 * U[:, -2] + U[:, -3]) / (2.0 * dr)
    return ts, r, U, Q, tau_w


# ----------------------------------------------------------------------------- 1D closure
def friction_1d(Q, A, zeta, rho=RHO, mu=MU):
    """Viscous friction per unit length, divided by rho, as used in the 1D momentum equation:
    f = -K_R Q / A, K_R = 2 (zeta + 2) pi mu / rho."""
    return -2.0 * (zeta + 2.0) * np.pi * mu / rho * Q / A


def effective_zeta(Q, tau_w, R, mu=MU):
    """Least-squares zeta that best matches the full-order friction 2 pi R tau_w with (zeta+2) 2 pi mu Q / A."""
    A = np.pi * R**2
    f_full = 2.0 * np.pi * R * tau_w          # force per unit length opposing the flow
    basis = 2.0 * np.pi * mu * Q / A
    k = np.dot(f_full, basis) / np.dot(basis, basis)
    return k - 2.0
