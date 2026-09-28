"""Verification of the full-order axisymmetric model (hemo1d.womersley)."""
import numpy as np

from hemo1d.model import MU, RHO
from hemo1d.womersley import (axisymmetric_fd, effective_zeta, harmonic_flow, harmonic_profile,
                              harmonic_wall_shear, womersley_from_flow)

R, T = 3e-3, 1.1
OM = 2 * np.pi / T


def test_fd_matches_womersley_analytical():
    """Crank-Nicolson FD solution vs Womersley's Bessel-function solution for one harmonic."""
    G = 200.0
    ts, r, U, Q, tau = axisymmetric_fd(R, lambda t: -G * np.cos(OM * t), t_end=6 * T, nr=80, dt=5e-4)
    m = ts >= 5 * T
    e = np.exp(1j * OM * ts[m])
    ua = np.real(harmonic_profile(r, R, OM, G)[None, :] * e[:, None])
    Qa = np.real(harmonic_flow(R, OM, G) * e)
    taua = np.real(harmonic_wall_shear(R, OM, G) * e)
    assert np.abs(U[m] - ua).max() / np.abs(ua).max() < 1e-3
    assert np.abs(Q[m] - Qa).max() / np.abs(Qa).max() < 1e-3
    assert np.abs(tau[m] - taua).max() / np.abs(taua).max() < 2e-3


def test_fd_steady_poiseuille():
    G = 200.0
    ts, r, U, Q, tau = axisymmetric_fd(R, lambda t: -G, t_end=3.0, nr=80, dt=2e-3)
    assert abs(Q[-1] / (np.pi * R**4 * G / (8 * MU)) - 1) < 2e-3
    assert abs(tau[-1] / (G * R / 2) - 1) < 2e-3


def test_fourier_reconstruction_and_zeta_limits():
    """Reconstructing u(r,t) from a flow waveform returns that waveform; for slow flow the effective
    profile parameter tends to the Poiseuille value zeta = 2."""
    t = np.linspace(0, 20.0, 200, endpoint=False)          # very slow cycle: alpha ~ 0.9
    Q = 6e-6 + 3e-6 * np.sin(2 * np.pi * t / 20.0)
    r = np.linspace(0, R, 41)
    w = womersley_from_flow(t, Q, R, r=r)
    Qrec = 2 * np.pi * np.trapezoid(w["u"] * r, r, axis=1)
    assert np.abs(Qrec - Q).max() / Q.max() < 5e-3
    assert abs(effective_zeta(Q, w["tau_w"], R) - 2.0) < 0.1
