"""
Test cases with the geometric, material and Windkessel parameters of the benchmark
    Boileau, Pan, Sazonov, Vignon-Clementel, Alastruey et al. (2015), "A benchmark study of
    numerical schemes for one-dimensional arterial blood flow modelling", IJNMBE 31(10).
Geometry is given at zero transmural pressure (openBF convention, equivalent to the benchmark's
diastolic reference) and the inflow waveforms are the benchmark's tabulated data (data/).
"""
from __future__ import annotations

import os

import numpy as np

from .model import blood_vessel
from .solver import Junction, Network, Windkessel3

MMHG = 133.322  # Pa


def half_sine_inflow(q_peak, q_base=0.0, t_sys=0.3, period=1.1):
    """Periodic flow rate [m^3/s]: baseline q_base plus a half-sine of amplitude q_peak-q_base."""
    def q(t):
        tau = t % period
        if tau < t_sys:
            return q_base + (q_peak - q_base) * np.sin(np.pi * tau / t_sys)
        return q_base
    q.period = period
    return q


# ----------------------------------------------------------------------------- benchmark inflows
DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def tabulated_inflow(path):
    """Periodic inflow Q(t) [m^3/s] interpolated from a two-column (t, Q) file spanning one period.
    The files in data/ are the inflow waveforms of the Boileau et al. (2015) benchmark, as
    distributed with openBF (INSIGNEO, Apache-2.0)."""
    d = np.loadtxt(path)
    t, Q = d[:, 0], d[:, 1]
    period = t[-1]

    def q(tt):
        return float(np.interp(tt % period, t, Q))
    q.period = period
    q.table = d
    return q


# ----------------------------------------------------------------------------- case 1: carotid
def common_carotid(n_cells=100, cfl=0.5, benchmark_inflow=True):
    """Boileau et al. 2015, common carotid artery, with the zero-pressure reference geometry used by
    openBF: L=126 mm, R0=2.6485 mm, h0=0.24 mm, E=700 kPa, zeta=2 (Poiseuille profile),
    R1=2.4875e8, C=1.7529e-10, R2=1.8697e9 (SI), p_out=0, and the benchmark inflow waveform
    (mean 6.48 ml/s, period 1.1 s)."""
    v = blood_vessel("carotid", 12.6, 0.26485, 0.24, 700.0, n_cells=n_cells, zeta=2.0)
    wk = Windkessel3(R1=2.4875e8, C=1.7529e-10, R2=1.8697e9, p_out=0.0)
    if benchmark_inflow:
        q = tabulated_inflow(os.path.join(DATA, "cca_inlet.dat"))
    else:
        q = half_sine_inflow(q_peak=14e-6, q_base=4e-6, t_sys=0.3, period=1.1)
    net = Network([v], q, outlets={"carotid": wk}, cfl=cfl)
    return net, v, wk, q


# ----------------------------------------------------------------------------- case 2: aortic bifurcation
def aortic_bifurcation(n_cells=100, cfl=0.5, benchmark_inflow=True):
    """Boileau et al. 2015, aortic (iliac) bifurcation, zero-pressure reference geometry as in openBF:
    parent L=8.6 cm, R0=7.581 mm, h0=0.9 mm, E=500 kPa; daughters L=8.5 cm, R0=5.492 mm,
    h0=0.68 mm, E=700 kPa; zeta=9; R1=6.8123e7, C=3.6664e-10, R2=3.1013e9 at each outlet;
    benchmark inflow (mean 7.9 ml/s, peak 87 ml/s, with backflow, period 1.1 s)."""
    aorta = blood_vessel("aorta", 8.6, 0.7581, 0.9, 500.0, n_cells=n_cells)
    il1 = blood_vessel("iliac_L", 8.5, 0.5492, 0.68, 700.0, n_cells=n_cells)
    il2 = blood_vessel("iliac_R", 8.5, 0.5492, 0.68, 700.0, n_cells=n_cells)
    wk = {n: Windkessel3(R1=6.8123e7, C=3.6664e-10, R2=3.1013e9, p_out=0.0) for n in ("iliac_L", "iliac_R")}
    if benchmark_inflow:
        q = tabulated_inflow(os.path.join(DATA, "ibif_inlet.dat"))
    else:
        q = half_sine_inflow(q_peak=40e-6, q_base=0.0, t_sys=0.3, period=1.1)
    net = Network([aorta, il1, il2], q, outlets=wk,
                  junctions=[Junction("aorta", ["iliac_L", "iliac_R"])], cfl=cfl)
    return net, (aorta, il1, il2), wk, q


def run_cycles(net: Network, probes, n_cycles=6, period=1.1, record_last=1, sample_every=5):
    """Run n_cycles heart beats and record the last `record_last` ones."""
    return net.run(n_cycles * period, probes, sample_every=sample_every,
                   t_start_record=(n_cycles - record_last) * period)
