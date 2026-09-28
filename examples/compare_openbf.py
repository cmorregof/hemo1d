"""
Cross-code comparison on the Boileau et al. (2015) benchmark cases: hemo1d (Lax-Wendroff, this repo)
against the reference waveforms of openBF (finite-volume MUSCL, Benemerito et al. 2024), computed by
openBF's own test suite on the same inflow, geometry, wall and Windkessel parameters (data/).
Writes figures/fig5_vs_openbf.png and prints RMS / peak differences.
Run: python examples/compare_openbf.py  (~2 min)
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hemo1d import aortic_bifurcation, common_carotid, run_cycles  # noqa: E402

C1, C2, MUTED, GRID = "#2a78d6", "#eb6834", "#6b7280", "#e5e7eb"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "lines.linewidth": 2})
DATA = os.path.join(os.path.dirname(__file__), "..", "data")
OUT = os.path.join(os.path.dirname(__file__), "..", "figures")
COLS = {"x0": 1, "xL2": 3, "xL": 5}


def load_ref(case, vessel, field):
    return np.loadtxt(os.path.join(DATA, f"openbf_ref_{case}_{vessel}_{field}.dat"))


def align(t_ref, y_ref, t, y, period):
    """Both series cover one cycle but may start at different phases: shift hemo1d's time so that
    its cycle starts where the reference's does (inflow is periodic, so phase = t mod period)."""
    tt = (t - t[0]) % period
    order = np.argsort(tt)
    return np.interp(t_ref % period, tt[order], y[order])


def compare(name, rec, key_of, ref_P, ref_Q, period, axes, title):
    stats = {}
    for lab, col in COLS.items():
        p_ref, q_ref = ref_P[:, col], ref_Q[:, col]
        p = align(ref_P[:, 0], p_ref, rec["t"], rec[key_of(lab)]["p"], period)
        q = align(ref_Q[:, 0], q_ref, rec["t"], rec[key_of(lab)]["Q"], period)
        stats[lab] = dict(
            p_rms=np.sqrt(np.mean((p - p_ref) ** 2)), p_rel=np.sqrt(np.mean((p - p_ref) ** 2)) / (p_ref.max() - p_ref.min()),
            p_sys=(p.max() - p_ref.max()), p_dia=(p.min() - p_ref.min()),
            q_rms=np.sqrt(np.mean((q - q_ref) ** 2)), q_rel=np.sqrt(np.mean((q - q_ref) ** 2)) / (q_ref.max() - q_ref.min()),
        )
    ax_p, ax_q = axes
    t = ref_P[:, 0]
    for lab, ls in (("x0", "-"), ("xL", "--")):
        col = COLS[lab]
        ax_p.plot(t, ref_P[:, col] / 1e3, ls, color=C2, label=f"openBF, {'inlet' if lab=='x0' else 'outlet'}")
        ax_p.plot(t, align(t, ref_P[:, col], rec["t"], rec[key_of(lab)]["p"], period) / 1e3, ls, color=C1, label=f"hemo1d, {'inlet' if lab=='x0' else 'outlet'}")
        ax_q.plot(t, ref_Q[:, col] * 1e6, ls, color=C2)
        ax_q.plot(t, align(t, ref_Q[:, col], rec["t"], rec[key_of(lab)]["Q"], period) * 1e6, ls, color=C1)
    ax_p.set(title=f"{title}: pressure", xlabel="time in cycle [s]", ylabel="p [kPa]")
    ax_q.set(title=f"{title}: flow", xlabel="time in cycle [s]", ylabel="Q [ml/s]")
    return stats


if __name__ == "__main__":
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.8))

    net, v, wk, q = common_carotid(n_cells=126)
    rec = run_cycles(net, {"carotid": [0.0, v.length / 2, v.length]}, n_cycles=10, period=q.period, sample_every=2)
    key = lambda lab: ("carotid", {"x0": 0.0, "xL2": v.length / 2, "xL": v.length}[lab])
    s1 = compare("cca", rec, key, load_ref("cca", "carotid", "P"), load_ref("cca", "carotid", "Q"), q.period, axes[0], "Common carotid")

    net, (ao, il1, il2), wk, q = aortic_bifurcation(n_cells=86)
    rec = run_cycles(net, {"aorta": [0.0, ao.length / 2, ao.length]}, n_cycles=10, period=q.period, sample_every=2)
    key = lambda lab: ("aorta", {"x0": 0.0, "xL2": ao.length / 2, "xL": ao.length}[lab])
    s2 = compare("ibif", rec, key, load_ref("ibif", "aorta", "P"), load_ref("ibif", "aorta", "Q"), q.period, axes[1], "Aortic bifurcation (parent)")

    axes[0][0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_vs_openbf.png"), dpi=160)

    for name, s in (("common carotid", s1), ("aortic bifurcation, parent vessel", s2)):
        print(f"\n{name} — hemo1d vs openBF")
        for lab, d in s.items():
            print(f"  {lab:4s}  p: RMS {d['p_rms']:.0f} Pa ({100*d['p_rel']:.1f}% of pulse pressure), "
                  f"systolic {d['p_sys']:+.0f} Pa, diastolic {d['p_dia']:+.0f} Pa | "
                  f"Q: RMS {d['q_rms']*1e6:.2f} ml/s ({100*d['q_rel']:.1f}% of range)")
