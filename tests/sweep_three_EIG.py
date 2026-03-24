import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import settings
from functions import f_vec, sf_vec
from analysis import oscillation_metric, phase_difference

"""
Parameter sweep for the three-neuron circuit:
  - Excitatory synapse : Neuron 0 --> Neuron 1  (G_syne)
  - Inhibitory synapse : Neuron 1 --> Neuron 0  (G_syni)
  - Gap junction       : Neuron 1 <-> Neuron 2  (G_gap)

State vector: [V0, V1, V2, SF0, SF1]
  SF0/SF1 are synaptic fatigue variables for the two cells with synaptic outputs.

Sweep structure
---------------
  Outer loop : G_gap  (a small set of fixed values → one figure row each)
  Inner grid : G_syne x G_syni  (2-D heatmap)

Metrics plotted per (G_syne, G_syni, G_gap) combination:
  1. Neuron 0 oscillates   (bool → 0/1)
  2. Neuron 1 oscillates   (bool → 0/1)
  3. CV of ISI – Neuron 0  (regularity; lower = more regular)
  4. Phase difference 0↔1  (0 = in-phase, 0.5 = antiphase)
  5. Phase difference 1↔2  (how well the gap junction couples neurons 1 and 2)
"""

# ── Biophysical parameters (Yuval model) ─────────────────────────────────────
settings.C   = 7.0
settings.g   = 1.0
settings.L   = -70.0
settings.T   = -45.0
settings.H   = -35.0
settings.m1  = 0.7
settings.m2  = 1 / 81.0
settings.m3  = -1 / 30.0
settings.m4  = 0.17

settings.E_syne = 0.0
settings.E_syni = -100.0
settings.k_syn  = 0.5   # sigmoid steepness
settings.V_th   = -52.0   # mV half-activation voltage
settings.a      = 0.000035
settings.b      = 0.005
settings.numCells = 3

# ── Simulation parameters ─────────────────────────────────────────────────────
Iapp  = 0.32    # pA applied to Neuron 0 throughout
V0    = -75.0  # initial voltage for all three neurons (mV)
tf    = 5000   # ms
t     = np.linspace(0, tf, int(tf))

inits = np.array([V0, V0, V0, 0.0, 0.0])  # [V0, V1, V2, SF0, SF1]

MEDIA_DIR = os.path.join(os.path.dirname(__file__), '..', 'media')

# ── Sweep ranges ──────────────────────────────────────────────────────────────
G_syne_vals = np.logspace(-2.0, 0.0, 11)   # excitatory conductance
G_syni_vals = np.logspace(-2.0, 0.0, 11)   # inhibitory conductance
G_gap_slices = [0.0, 0.01, 0.05, 0.1]     # fixed gap-junction values to slice at


# ── ODE ───────────────────────────────────────────────────────────────────────
def make_ode(G_syne, G_syni, G_gap):
    def ode(t, y):
        V0, V1, V2, SF0, SF1 = y

        fv = f_vec(np.array([V0, V1, V2]))

        # Sigmoid gating for cells with synaptic output
        s0 = 1.0 / (1.0 + np.exp(-settings.k_syn * (V0 - settings.V_th)))
        s1 = 1.0 / (1.0 + np.exp(-settings.k_syn * (V1 - settings.V_th)))

        # Synaptic fatigue dynamics
        SF0_dot = sf_vec(np.array([V0]), np.array([SF0]))[0]
        SF1_dot = sf_vec(np.array([V1]), np.array([SF1]))[0]

        # Synaptic currents
        I_exc1 = G_syne * s0 * SF0 * (V1 - settings.E_syne)  # excitation into N1 from N0
        I_inh0 = G_syni * s1 * SF1 * (V0 - settings.E_syni)  # inhibition into N0 from N1

        # Gap junction currents (bidirectional 1 <-> 2)
        I_gap1 = G_gap * (V1 - V2)
        I_gap2 = G_gap * (V2 - V1)

        dV0 = (settings.g * fv[0] - I_inh0 + Iapp) / settings.C
        dV1 = (settings.g * fv[1] - I_exc1 - I_gap1) / settings.C
        dV2 = (settings.g * fv[2] - I_gap2) / settings.C

        return [dV0, dV1, dV2, SF0_dot, SF1_dot]

    return ode


def run_sim(G_syne, G_syni, G_gap):
    settings.G_syne = G_syne
    settings.G_syni = G_syni
    settings.G_gap  = G_gap
    return solve_ivp(make_ode(G_syne, G_syni, G_gap), [0.0, tf], inits,
                     method='BDF', t_eval=t, max_step=5.0, rtol=1e-3, atol=1e-5)


# ── Main sweep ────────────────────────────────────────────────────────────────
def run(save=True):
    n_e = len(G_syne_vals)
    n_i = len(G_syni_vals)
    n_metrics = 5

    # results[gap_idx, metric_idx, i_syni, j_syne]
    results = np.full((len(G_gap_slices), n_metrics, n_i, n_e), np.nan)

    total = len(G_gap_slices) * n_e * n_i
    count = 0

    for g_idx, G_gap in enumerate(G_gap_slices):
        for j, G_syne in enumerate(G_syne_vals):
            for i, G_syni in enumerate(G_syni_vals):
                count += 1
                print(f"  [{count}/{total}]  G_gap={G_gap:.3f}  "
                      f"G_syne={G_syne:.2f}  G_syni={G_syni:.2f}", end='\r')

                sol = run_sim(G_syne, G_syni, G_gap)

                m0 = oscillation_metric(sol.t, sol.y[0])
                m1 = oscillation_metric(sol.t, sol.y[1])
                ph01 = phase_difference(sol.t, sol.y[0], sol.y[1])
                ph12 = phase_difference(sol.t, sol.y[1], sol.y[2])

                results[g_idx, 0, i, j] = float(m0['oscillates'])
                results[g_idx, 1, i, j] = float(m1['oscillates'])
                results[g_idx, 2, i, j] = m0['cv_isi']
                results[g_idx, 3, i, j] = ph01
                results[g_idx, 4, i, j] = ph12

    print()

    metric_labels = [
        "N0 oscillates",
        "N1 oscillates",
        "CV ISI – N0\n(lower = regular)",
        "Phase diff N0↔N1\n(0.5 = antiphase)",
        "Phase diff N1↔N2\n(0.5 = antiphase)",
    ]
    cmaps = ['RdYlGn', 'RdYlGn', 'viridis_r', 'coolwarm', 'coolwarm']
    vranges = [(0, 1), (0, 1), (None, None), (0, 0.5), (0, 0.5)]

    extent = [G_syne_vals[0], G_syne_vals[-1], G_syni_vals[0], G_syni_vals[-1]]

    fig, axes = plt.subplots(
        len(G_gap_slices), n_metrics,
        figsize=(4 * n_metrics, 3.5 * len(G_gap_slices)),
        squeeze=False
    )
    fig.suptitle(
        f"Three-Neuron Sweep  (Iapp={Iapp} pA, V0={V0} mV)\n"
        f"Circuit: N0 -[exc]→ N1 -[inh]→ N0,  N1 ↔[gap]↔ N2",
        fontsize=13
    )

    for g_idx, G_gap in enumerate(G_gap_slices):
        for m_idx in range(n_metrics):
            ax = axes[g_idx, m_idx]
            data = results[g_idx, m_idx]
            vmin, vmax = vranges[m_idx]

            im = ax.imshow(
                data, origin='lower', aspect='auto', extent=extent,
                cmap=cmaps[m_idx],
                vmin=vmin, vmax=vmax
            )
            fig.colorbar(im, ax=ax, shrink=0.8)

            if g_idx == 0:
                ax.set_title(metric_labels[m_idx], fontsize=9)
            if m_idx == 0:
                ax.set_ylabel(f"G_gap={G_gap:.3f}\nG_syni", fontsize=8)
            else:
                ax.set_ylabel("G_syni", fontsize=8)
            ax.set_xlabel("G_syne", fontsize=8)

    fig.tight_layout()

    if save:
        out = os.path.join(MEDIA_DIR, f"sweep_three_EIG_Iapp{Iapp}.png")
        plt.savefig(out, dpi=150)
        print(f"Saved → {out}")

    plt.show()


if __name__ == "__main__":
    run()
