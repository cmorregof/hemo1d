"""
Full-order (axisymmetric Navier-Stokes, Womersley) vs reduced (1D friction closure) comparison on
the two benchmark vessels. For the benchmark flow waveform Q(t) in each vessel, the full-order model
gives the radial velocity profile and the wall shear stress; the 1D model replaces both by a fixed
profile parameter zeta. Writes figures/fig6_full_vs_reduced.png.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hemo1d.model import MU, RHO  # noqa: E402
from hemo1d.womersley import effective_zeta, friction_1d, womersley_from_flow, womersley_number  # noqa: E402

C1, C2, C3, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#6b7280", "#e5e7eb"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "lines.linewidth": 2})
DATA = os.path.join(os.path.dirname(__file__), "..", "data")
OUT = os.path.join(os.path.dirname(__file__), "..", "figures")

CASES = [  # name, inflow file, radius at the reference state [m], zeta used by the benchmark
    ("Common carotid", "cca_inlet.dat", 3.0e-3, 2.0),
    ("Abdominal aorta", "ibif_inlet.dat", 8.6e-3, 9.0),
]

if __name__ == "__main__":
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.8))
    for row, (name, fn, R, zeta_bench) in enumerate(CASES):
        d = np.loadtxt(os.path.join(DATA, fn))
        t, Q = d[:-1, 0], d[:-1, 1]
        T = d[-1, 0]
        r = np.linspace(0.0, R, 61)
        w = womersley_from_flow(t, Q, R, r=r)
        A = np.pi * R**2
        alpha = womersley_number(R, 2 * np.pi / T, MU / RHO)
        z_eff = effective_zeta(Q, w["tau_w"], R)

        # --- velocity profiles at four instants
        ax = axes[row][0]
        idx = [np.argmax(Q), np.argmin(Q), int(0.5 * len(t)), int(0.8 * len(t))]
        labels = ["peak flow", "min flow", "mid-diastole", "late diastole"]
        for k, (i, lab, c) in enumerate(zip(idx, labels, (C1, C2, C3, MUTED))):
            ax.plot(r / R, w["u"][i] * 100, color=c, label=lab)
            # 1D assumed profile with the same Q: u = (zeta+2)/zeta * Qbar (1 - (r/R)^zeta)
            ubar = Q[i] / A
            ax.plot(r / R, (zeta_bench + 2) / zeta_bench * ubar * (1 - (r / R) ** zeta_bench) * 100, ":", color=c, lw=1.2)
        ax.set(title=f"{name}: velocity profiles (α = {alpha:.1f})", xlabel="r / R", ylabel="u [cm/s]")
        ax.legend(frameon=False, fontsize=8, title=f"solid: full-order · dotted: 1D, ζ = {zeta_bench:g}", title_fontsize=8)

        # --- friction per unit length over the cycle
        ax = axes[row][1]
        f_full = -2.0 * np.pi * R * w["tau_w"] / RHO
        ax.plot(t, f_full * 1e3, color=C1, label="full-order (Womersley)")
        ax.plot(t, friction_1d(Q, A, zeta_bench) * 1e3, color=C2, label=f"1D, ζ = {zeta_bench:g} (benchmark)")
        other = 9.0 if zeta_bench == 2.0 else 2.0
        ax.plot(t, friction_1d(Q, A, other) * 1e3, "--", color=C3, label=f"1D, ζ = {other:g}")
        ax.set(title=f"{name}: viscous friction term  (ζ_eff = {z_eff:.1f})", xlabel="time in cycle [s]", ylabel="−K_R Q/A  [10⁻³ m²/s²]")
        ax.legend(frameon=False, fontsize=8)

        rms = lambda f: np.sqrt(np.mean((f - f_full) ** 2)) / (f_full.max() - f_full.min())
        print(f"{name}: alpha={alpha:.2f}, zeta_eff={z_eff:.2f}; RMS mismatch of friction vs full-order: "
              f"zeta={zeta_bench:g}: {100*rms(friction_1d(Q, A, zeta_bench)):.1f}%, zeta={other:g}: {100*rms(friction_1d(Q, A, other)):.1f}%, "
              f"zeta_eff: {100*rms(friction_1d(Q, A, z_eff)):.1f}%")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig6_full_vs_reduced.png"), dpi=160)
