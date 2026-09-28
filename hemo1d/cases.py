"""
Test cases with the geometric, material and Windkessel parameters of the benchmark
    Boileau, Pan, Sazonov, Vignon-Clementel, Alastruey et al. (2015), "A benchmark study of
    numerical schemes for one-dimensional arterial blood flow modelling", IJNMBE 31(10).
The inflow waveforms of the benchmark are tabulated data; here they are replaced by smooth
analytic waveforms (half-sine systole over a diastolic baseline) whose mean flow matches the
mean pressure ~100 mmHg implied by the Windkessel total resistance. The shapes are therefore
illustrative; the wave physics, wall parameters and outflow models are the benchmark's.
"""
from __future__ import annotations

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


# ----------------------------------------------------------------------------- case 1: carotid
def common_carotid(n_cells=100, cfl=0.5):
    """Boileau et al. 2015, test 3 (common carotid artery): L=126 mm, r=3 mm, h=0.3 mm, E=700 kPa,
    R1=2.4875e8, C=1.7529e-10, R2=1.8697e9 (SI), p_out = 0."""
    v = blood_vessel("carotid", 12.6, 0.3, 0.3, 700.0, n_cells=n_cells)
    wk = Windkessel3(R1=2.4875e8, C=1.7529e-10, R2=1.8697e9, p_out=0.0)
    q = half_sine_inflow(q_peak=14e-6, q_base=4e-6, t_sys=0.3, period=1.1)
    net = Network([v], q, outlets={"carotid": wk}, cfl=cfl)
    return net, v, wk, q


# ----------------------------------------------------------------------------- case 2: aortic bifurcation
def aortic_bifurcation(n_cells=100, cfl=0.5):
    """Boileau et al. 2015, test 4 (aortic bifurcation): abdominal aorta L=8.6 cm, r=0.86 cm,
    h=1.032 mm, E=500 kPa; two iliacs L=8.5 cm, r=0.6 cm, h=0.72 mm, E=700 kPa;
    R1=6.8123e7, C=3.6664e-10, R2=3.1013e9 at each iliac outlet."""
    aorta = blood_vessel("aorta", 8.6, 0.86, 1.032, 500.0, n_cells=n_cells)
    il1 = blood_vessel("iliac_L", 8.5, 0.6, 0.72, 700.0, n_cells=n_cells)
    il2 = blood_vessel("iliac_R", 8.5, 0.6, 0.72, 700.0, n_cells=n_cells)
    wk = {n: Windkessel3(R1=6.8123e7, C=3.6664e-10, R2=3.1013e9, p_out=0.0) for n in ("iliac_L", "iliac_R")}
    q = half_sine_inflow(q_peak=40e-6, q_base=0.0, t_sys=0.3, period=1.1)
    net = Network([aorta, il1, il2], q, outlets=wk,
                  junctions=[Junction("aorta", ["iliac_L", "iliac_R"])], cfl=cfl)
    return net, (aorta, il1, il2), wk, q


def run_cycles(net: Network, probes, n_cycles=6, period=1.1, record_last=1, sample_every=5):
    """Run n_cycles heart beats and record the last `record_last` ones."""
    return net.run(n_cycles * period, probes, sample_every=sample_every,
                   t_start_record=(n_cycles - record_last) * period)
