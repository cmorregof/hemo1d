"""
Interactive explorer for the 1D blood-flow model.   Run:  streamlit run app.py
"""
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from hemo1d import MMHG, Network, Windkessel3, blood_vessel, half_sine_inflow, run_cycles, windkessel_0d
from hemo1d.model import MU, RHO
from hemo1d.womersley import effective_zeta, friction_1d, womersley_from_flow, womersley_number

C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e5e7eb", "lines.linewidth": 2})

st.set_page_config(page_title="hemo1d explorer", layout="wide")
st.title("hemo1d — reduced-order blood flow in a compliant artery")
st.caption("Three rungs of the model hierarchy for one vessel: full-order radial viscous profile (axisymmetric "
           "Navier–Stokes, Womersley), 1D area–flow model (Lax–Wendroff) with a 3-element Windkessel outlet, and a 0D "
           "lumped model. Default parameters: common carotid of Boileau et al. (2015).")

with st.sidebar:
    st.header("Vessel")
    L = st.slider("length [cm]", 5.0, 40.0, 12.6, 0.1)
    r = st.slider("radius [mm]", 1.0, 10.0, 3.0, 0.1)
    h = st.slider("wall thickness [mm]", 0.1, 2.0, 0.3, 0.05)
    E = st.slider("Young modulus [kPa]", 100.0, 2000.0, 700.0, 10.0)
    st.header("Windkessel outlet (RCR)")
    R1 = st.slider("R1 [1e8 Pa s/m³]", 0.1, 10.0, 2.4875, 0.05)
    C = st.slider("C [1e-10 m³/Pa]", 0.2, 10.0, 1.7529, 0.05)
    R2 = st.slider("R2 [1e9 Pa s/m³]", 0.2, 6.0, 1.8697, 0.05)
    st.header("Inflow")
    qpk = st.slider("peak flow [ml/s]", 2.0, 60.0, 14.0, 0.5)
    qb = st.slider("baseline flow [ml/s]", 0.0, 20.0, 4.0, 0.5)
    T = st.slider("period [s]", 0.6, 1.5, 1.1, 0.05)
    st.header("Numerics")
    n = st.select_slider("cells", [20, 40, 80, 160], 40)
    cycles = st.slider("cycles", 2, 8, 4)
    zeta = st.select_slider("1D profile parameter ζ", [2.0, 4.0, 9.0], 2.0)


@st.cache_data(show_spinner="integrating…")
def simulate(L, r, h, E, R1, C, R2, qpk, qb, T, n, cycles, zeta):
    v = blood_vessel("v", L, r / 10.0, h, E, n_cells=n, zeta=zeta)
    wk = Windkessel3(R1=R1 * 1e8, C=C * 1e-10, R2=R2 * 1e9)
    q = half_sine_inflow(qpk * 1e-6, qb * 1e-6, t_sys=0.3 * T / 1.1, period=T)
    net = Network([v], q, outlets={"v": wk})
    xs = [0.0, v.length / 2, v.length]
    rec = run_cycles(net, {"v": xs}, n_cycles=cycles, period=T, sample_every=3)
    C_v = v.length * 2.0 * v.A0 * np.sqrt(v.A0) / v.beta
    wk0 = Windkessel3(R1=R1 * 1e8, C=C * 1e-10, R2=R2 * 1e9)
    t0, p0 = windkessel_0d(q, wk0, cycles * T, 1e-4, C_extra=C_v)
    m = t0 >= (cycles - 1) * T
    out = {"t": rec["t"] - rec["t"][0], "t0": t0[m] - (cycles - 1) * T, "p0": p0[m] / MMHG,
           "c0": v.c0, "A0": v.A0, "Cv": C_v}
    for xk, lab in zip(xs, ("inlet", "mid", "outlet")):
        out[lab] = (rec[("v", xk)]["p"] / MMHG, rec[("v", xk)]["Q"] * 1e6)
    return out


res = simulate(L, r, h, E, R1, C, R2, qpk, qb, T, n, cycles, zeta)

k1, k2, k3, k4 = st.columns(4)
k1.metric("pulse wave speed c₀", f"{res['c0']:.2f} m/s")
k2.metric("systolic / diastolic (inlet)", f"{res['inlet'][0].max():.0f} / {res['inlet'][0].min():.0f} mmHg")
k3.metric("pulse pressure", f"{res['inlet'][0].max() - res['inlet'][0].min():.1f} mmHg")
k4.metric("wall compliance L·dA/dp", f"{res['Cv']*1e10:.2f} ×1e-10 m³/Pa")

c1, c2 = st.columns(2)
fig, ax = plt.subplots(figsize=(5.5, 3.4))
for lab, c in zip(("inlet", "mid", "outlet"), (C1, C2, C3)):
    ax.plot(res["t"], res[lab][0], color=c, label=lab)
ax.set(xlabel="time in cycle [s]", ylabel="pressure [mmHg]", title="Pressure along the vessel (1D)")
ax.legend(frameon=False)
c1.pyplot(fig, use_container_width=True)

fig, ax = plt.subplots(figsize=(5.5, 3.4))
for lab, c in zip(("inlet", "mid", "outlet"), (C1, C2, C3)):
    ax.plot(res["t"], res[lab][1], color=c, label=lab)
ax.set(xlabel="time in cycle [s]", ylabel="flow rate [ml/s]", title="Flow along the vessel (1D)")
ax.legend(frameon=False)
c2.pyplot(fig, use_container_width=True)

fig, ax = plt.subplots(figsize=(11, 3.4))
ax.plot(res["t"], res["inlet"][0], color=C1, label="1D vessel + RCR")
ax.plot(res["t0"], res["p0"], color=C2, label="0D lumped (RCR + wall compliance)")
ax.set(xlabel="time in cycle [s]", ylabel="inlet pressure [mmHg]",
       title="Model complexity: does the 1D wave propagation matter for the inlet pressure?")
ax.legend(frameon=False)
st.pyplot(fig, use_container_width=True)

# ---- full-order radial profile vs 1D closure
st.subheader("Full-order vs reduced: the viscous closure")
Rm = r * 1e-3
tt = np.linspace(0, T, 200, endpoint=False)
Qt = np.array([half_sine_inflow(qpk * 1e-6, qb * 1e-6, t_sys=0.3 * T / 1.1, period=T)(x) for x in tt])
rr = np.linspace(0, Rm, 61)
w = womersley_from_flow(tt, Qt, Rm, r=rr)
z_eff = effective_zeta(Qt, w["tau_w"], Rm)
alpha = womersley_number(Rm, 2 * np.pi / T, MU / RHO)
k1, k2 = st.columns(2)
k1.metric("Womersley number α", f"{alpha:.2f}")
k2.metric("effective ζ from the full-order model", f"{z_eff:.2f}", help="ζ = 2 is Poiseuille; ζ → ∞ is a flat profile")
c1, c2 = st.columns(2)
fig, ax = plt.subplots(figsize=(5.5, 3.4))
Am = np.pi * Rm**2
for i, lab, c in zip([np.argmax(Qt), int(0.5 * len(tt)), int(0.85 * len(tt))], ("peak flow", "mid cycle", "late cycle"), (C1, C2, C3)):
    ax.plot(rr / Rm, w["u"][i] * 100, color=c, label=lab)
    ax.plot(rr / Rm, (zeta + 2) / zeta * Qt[i] / Am * (1 - (rr / Rm) ** zeta) * 100, ":", color=c, lw=1.2)
ax.set(xlabel="r / R", ylabel="u [cm/s]", title=f"Velocity profiles: full-order (solid) vs 1D ζ = {zeta:g} (dotted)")
ax.legend(frameon=False)
c1.pyplot(fig, use_container_width=True)
fig, ax = plt.subplots(figsize=(5.5, 3.4))
ax.plot(tt, -2 * np.pi * Rm * w["tau_w"] / RHO * 1e3, color=C1, label="full-order (Womersley)")
ax.plot(tt, friction_1d(Qt, Am, zeta) * 1e3, color=C2, label=f"1D closure, ζ = {zeta:g}")
ax.set(xlabel="time in cycle [s]", ylabel="−K_R Q/A [10⁻³ m²/s²]", title="Viscous friction term in the momentum equation")
ax.legend(frameon=False)
c2.pyplot(fig, use_container_width=True)

with st.expander("Model"):
    st.latex(r"\partial_t A + \partial_x Q = 0, \qquad \partial_t Q + \partial_x\!\left(\frac{Q^2}{A}\right) + \frac{A}{\rho}\,\partial_x p = -K_R \frac{Q}{A}")
    st.latex(r"p(A) = \frac{\beta}{A_0}\left(\sqrt{A}-\sqrt{A_0}\right),\quad \beta = \tfrac{4}{3}\sqrt{\pi}\,E h,\quad c(A)=\sqrt{\tfrac{\beta}{2\rho A_0}}\,A^{1/4}")
    st.markdown("Interior: two-step Lax–Wendroff. Boundaries: Riemann invariants $W_{1,2}=u\\pm4(c-c_0)$; "
                "inlet with prescribed $Q(t)$, outlet coupled to an RCR Windkessel solved implicitly. "
                "0D model: the vessel's wall compliance $L\\,dA/dp$ is lumped into the Windkessel capacitor.")
