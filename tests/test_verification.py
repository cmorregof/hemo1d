"""
Verification of the 1D solver against analytical results.
Run with:  python -m pytest tests -q
"""
import numpy as np
import pytest

from hemo1d import Network, Windkessel3, Junction, blood_vessel, common_carotid, run_cycles
from hemo1d.model import MU


def test_poiseuille_pressure_drop():
    """Steady flow with a parabolic profile (zeta=2) must reproduce Poiseuille's law
    dp = 8 mu L Q / (pi r^4) up to the small area change along the vessel."""
    v = blood_vessel("tube", 20.0, 0.3, 0.3, 700.0, n_cells=100, zeta=2.0)
    Q0 = 5e-6

    def q(t):
        return Q0 * min(t / 0.05, 1.0)  # smooth ramp to a constant flow

    net = Network([v], q, outlets={"tube": None})
    net.run(1.0, {"tube": []})
    p_in = v.pressure(net.left["tube"][0])
    p_out = v.pressure(net.right["tube"][0])
    dp_poiseuille = 8.0 * MU * v.length * Q0 / (np.pi * v.radius**4)
    # the vessel is pressurised, so its lumen is slightly wider than r: use the actual mean area
    dp_poiseuille_A = 8.0 * np.pi * MU * v.length * Q0 / np.mean(v.A) ** 2
    assert abs((p_in - p_out) - dp_poiseuille) / dp_poiseuille < 0.06
    assert abs((p_in - p_out) - dp_poiseuille_A) / dp_poiseuille_A < 0.005


def test_pulse_wave_speed():
    """A small pulse must travel at c0 = sqrt(beta/(2 rho A0)) A0^{1/4} (Moens-Korteweg form)."""
    v = blood_vessel("tube", 50.0, 0.3, 0.3, 700.0, n_cells=400)

    def q(t):
        return 1e-7 * np.exp(-((t - 0.01) / 0.003) ** 2)

    net = Network([v], q, outlets={"tube": None})
    rec = net.run(0.15, {"tube": [v.length]}, sample_every=1)
    t_arr = rec["t"][np.argmax(rec[("tube", v.length)]["Q"])]
    t_expected = 0.01 + v.length / v.c0
    assert abs(t_arr - t_expected) / t_expected < 0.02


def test_junction_conservation():
    """Flow leaving the parent equals the sum of flows entering the daughters (to Newton tolerance)."""
    from hemo1d import aortic_bifurcation
    net, (ao, il1, il2), wk, q = aortic_bifurcation(n_cells=30)
    rec = run_cycles(net, {"aorta": [ao.length], "iliac_L": [0.0], "iliac_R": [0.0]}, n_cycles=2, sample_every=1)
    Qa = rec[("aorta", ao.length)]["Q"]
    Qd = rec[("iliac_L", 0.0)]["Q"] + rec[("iliac_R", 0.0)]["Q"]
    assert np.max(np.abs(Qa - Qd)) < 1e-9 * max(1.0, np.max(np.abs(Qa)) * 1e6)


def test_mesh_convergence():
    """Mesh/time-step convergence of the mid-vessel pressure against a fine reference.
    Lax-Wendroff is second order in the interior, but the characteristic treatment of the
    boundaries (linear extrapolation of the Riemann invariants + ghost states frozen at t^n)
    is first order, and the measured global rate in a cycle-averaged L2 norm is ~1.1-1.2.
    The test asserts monotone convergence with rate > 1; raising the boundary treatment to
    second order is a documented next step."""
    ref, v, _, _ = common_carotid(n_cells=320)
    probes = lambda vv: {"carotid": [vv.length / 2]}
    rec_ref = run_cycles(ref, probes(v), n_cycles=3, sample_every=1)
    errs = []
    for n in (20, 40, 80):
        net, vv, _, _ = common_carotid(n_cells=n)
        rec = run_cycles(net, probes(vv), n_cycles=3, sample_every=1)
        p_ref = np.interp(rec["t"], rec_ref["t"], rec_ref[("carotid", v.length / 2)]["p"])
        errs.append(np.sqrt(np.mean((rec[("carotid", vv.length / 2)]["p"] - p_ref) ** 2)))
    rates = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert rates.min() > 1.0, f"errors {errs}, rates {rates}"
    assert errs[-1] < 5.0  # Pa: below 0.05 mmHg on the finest mesh


def test_agreement_with_openbf_carotid():
    """Cross-code check: on the Boileau et al. (2015) common-carotid case with identical inputs,
    inlet pressure must agree with openBF's reference waveform to within 2% of the pulse pressure."""
    import os
    net, v, _, q = common_carotid(n_cells=63)
    rec = run_cycles(net, {"carotid": [0.0]}, n_cycles=8, period=q.period, sample_every=2)
    ref = np.loadtxt(os.path.join(os.path.dirname(__file__), "..", "data", "openbf_ref_cca_carotid_P.dat"))
    tt = (rec["t"] - rec["t"][0]) % q.period
    order = np.argsort(tt)
    p = np.interp(ref[:, 0] % q.period, tt[order], rec[("carotid", 0.0)]["p"][order])
    rms = np.sqrt(np.mean((p - ref[:, 1]) ** 2))
    assert rms < 0.02 * (ref[:, 1].max() - ref[:, 1].min()), f"RMS {rms:.0f} Pa"
