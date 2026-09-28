"""
Reproduces the figures in figures/:
  fig1_carotid.png        pressure and flow along the common carotid (Boileau et al. 2015 parameters)
  fig2_1d_vs_0d.png       model-complexity comparison: 1D vessel + RCR vs pure 0D lumped model
  fig3_bifurcation.png    aortic bifurcation: waves through the junction
  fig4_convergence.png    mesh convergence of the mid-vessel pressure
Run:  python examples/make_figures.py   (about 3 minutes)
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hemo1d import MMHG, aortic_bifurcation, common_carotid, run_cycles, windkessel_0d  # noqa: E402

C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"   # categorical slots, fixed order
INK, MUTED, GRID = "#1f2937", "#6b7280", "#e5e7eb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "lines.linewidth": 2})
OUT = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUT, exist_ok=True)


def fig_carotid():
    net, v, wk, q = common_carotid(n_cells=100)
    xs = [0.0, v.length / 2, v.length]
    rec = run_cycles(net, {"carotid": xs}, n_cycles=6, sample_every=4)
    t = rec["t"] - rec["t"][0]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for xk, c, lab in zip(xs, (C1, C2, C3), ("inlet", "mid-vessel", "outlet")):
        ax[0].plot(t, rec[("carotid", xk)]["p"] / MMHG, color=c, label=lab)
        ax[1].plot(t, rec[("carotid", xk)]["Q"] * 1e6, color=c, label=lab)
    ax[0].set(xlabel="time in cycle [s]", ylabel="pressure [mmHg]", title="Common carotid: pressure")
    ax[1].set(xlabel="time in cycle [s]", ylabel="flow rate [ml/s]", title="Common carotid: flow")
    ax[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_carotid.png"), dpi=160)
    return rec, v, wk, q


def fig_1d_vs_0d(rec, v, wk, q):
    # lumped model: vessel wall compliance C_v = L dA/dp|A0 added to the Windkessel capacitor
    C_v = v.length * 2.0 * v.A0 * np.sqrt(v.A0) / v.beta
    period = q.period
    t0, p0 = windkessel_0d(q, wk, 6 * period, 1e-4, C_extra=C_v)
    m = t0 >= 5 * period
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(rec["t"] - rec["t"][0], rec[("carotid", 0.0)]["p"] / MMHG, color=C1, label="1D vessel + RCR")
    ax.plot(t0[m] - 5 * period, p0[m] / MMHG, color=C2, label="0D lumped (RCR + wall compliance)")
    ax.set(xlabel="time in cycle [s]", ylabel="inlet pressure [mmHg]",
           title="Model complexity: 1D vs 0D at the carotid inlet")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_1d_vs_0d.png"), dpi=160)
    p1 = rec[("carotid", 0.0)]["p"] / MMHG
    p0c = np.interp(rec["t"] - rec["t"][0], t0[m] - 5 * period, p0[m] / MMHG)
    print(f"1D vs 0D inlet pressure: systolic {p1.max():.1f} vs {p0c.max():.1f} mmHg, "
          f"diastolic {p1.min():.1f} vs {p0c.min():.1f} mmHg, RMS difference {np.sqrt(np.mean((p1-p0c)**2)):.2f} mmHg")


def fig_bifurcation():
    net, (ao, il1, il2), wk, q = aortic_bifurcation(n_cells=50)
    probes = {"aorta": [0.0, ao.length], "iliac_L": [il1.length]}
    rec = run_cycles(net, probes, n_cycles=6, sample_every=4)
    t = rec["t"] - rec["t"][0]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for key, c, lab in zip([("aorta", 0.0), ("aorta", ao.length), ("iliac_L", il1.length)],
                           (C1, C2, C3), ("aorta inlet", "bifurcation", "iliac outlet")):
        ax[0].plot(t, rec[key]["p"] / MMHG, color=c, label=lab)
        ax[1].plot(t, rec[key]["Q"] * 1e6, color=c, label=lab)
    ax[0].set(xlabel="time in cycle [s]", ylabel="pressure [mmHg]", title="Aortic bifurcation: pressure")
    ax[1].set(xlabel="time in cycle [s]", ylabel="flow rate [ml/s]", title="Aortic bifurcation: flow")
    ax[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_bifurcation.png"), dpi=160)


def fig_convergence():
    ref, v, _, _ = common_carotid(n_cells=320)
    rec_ref = run_cycles(ref, {"carotid": [v.length / 2]}, n_cycles=3, sample_every=1)
    ns, errs = (20, 40, 80, 160), []
    for n in ns:
        net, vv, _, _ = common_carotid(n_cells=n)
        rec = run_cycles(net, {"carotid": [vv.length / 2]}, n_cycles=3, sample_every=1)
        p_ref = np.interp(rec["t"], rec_ref["t"], rec_ref[("carotid", v.length / 2)]["p"])
        errs.append(np.sqrt(np.mean((rec[("carotid", vv.length / 2)]["p"] - p_ref) ** 2)))
    h = v.length / np.array(ns)
    rates = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    print("convergence errors [Pa]:", np.round(errs, 3), " rates:", np.round(rates, 2))
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.loglog(h * 1e3, errs, "o-", color=C1, label="measured L2 error of p (mid-vessel)")
    ax.loglog(h * 1e3, errs[0] * (h / h[0]) ** 1, "--", color=MUTED, lw=1, label="slope 1")
    ax.loglog(h * 1e3, errs[0] * (h / h[0]) ** 2, ":", color=MUTED, lw=1, label="slope 2")
    ax.set(xlabel="mesh size h [mm]", ylabel="error [Pa]", title="Mesh convergence (reference: 320 cells)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_convergence.png"), dpi=160)


if __name__ == "__main__":
    rec, v, wk, q = fig_carotid()
    fig_1d_vs_0d(rec, v, wk, q)
    fig_bifurcation()
    fig_convergence()
    print("figures written to", os.path.abspath(OUT))
