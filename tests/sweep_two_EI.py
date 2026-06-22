import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import settings
from functions import f_vec, w_inf_vec
from analysis import oscillation_metric, phase_difference

"""
Parameter sweep for the two-neuron excitatory-inhibitory circuit — no synaptic fatigue.
  - Excitatory synapse : N0 --> N1  (G_syne)
  - Inhibitory synapse : N1 --> N0  (G_syni)

N0 receives constant Iapp. N1 receives no direct input.
State vector: [V0, V1, w0, w1]

Sweep structure
---------------
  Rows   : Iapp  (linearly spaced)
  Grid   : G_syne (x) × G_syni (y), both log-spaced

Metrics:
  1. N0 oscillates   (bool)
  2. N1 oscillates   (bool)
  3. CV ISI – N0     (regularity; lower = more regular)
  4. Phase diff N0↔N1
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
settings.k_syn  = 0.25
settings.V_th   = -52.0
settings.beta   = 0.2
settings.numCells = 2

tau_w = 200.0

# ── Simulation settings ───────────────────────────────────────────────────────
tf    = 5000
t     = np.linspace(0, tf, int(tf))
inits = [settings.T, settings.L - 0.5, 0.0, 0.0]  # [V0, V1, w0, w1]

MEDIA_DIR = os.path.join(os.path.dirname(__file__), '..', 'media')

Iapp_slices = np.linspace(2, 5, 7)    # 2.0 → 5.0 pA


# ── ODE ───────────────────────────────────────────────────────────────────────
def make_ode(G_syne, G_syni, Iapp):
    def ode(t, y):
        V0, V1, w0, w1 = y

        fv   = f_vec(np.array([V0, V1]))
        winf = w_inf_vec(np.array([V0, V1]))

        s0 = 1.0 / (1.0 + np.exp(-settings.k_syn * (V0 - settings.V_th)))
        s1 = 1.0 / (1.0 + np.exp(-settings.k_syn * (V1 - settings.V_th)))

        I_exc1 = G_syne * s0 * (V1 - settings.E_syne)
        I_inh0 = G_syni * s1 * (V0 - settings.E_syni)

        dV0 = (settings.g * fv[0] - w0 - I_inh0 + Iapp) / settings.C
        dV1 = (settings.g * fv[1] - w1 - I_exc1) / settings.C
        dw0 = (winf[0] - w0) / tau_w
        dw1 = (winf[1] - w1) / tau_w

        return [dV0, dV1, dw0, dw1]

    return ode


def run_sim(G_syne, G_syni, Iapp):
    return solve_ivp(
        make_ode(G_syne, G_syni, Iapp),
        [0.0, tf], inits, method='BDF', t_eval=t,
        rtol=1e-3, atol=1e-5
    )


# ── Main sweep ────────────────────────────────────────────────────────────────
def run(save=False):
    G_syne_vals = np.logspace(-2, np.log10(1.0), 11)   # 0.01 → 1.0 nS
    G_syni_vals = np.logspace(-2, np.log10(1.0), 11)   # 0.01 → 1.0 nS

    n_e       = len(G_syne_vals)
    n_i       = len(G_syni_vals)
    n_Iapp    = len(Iapp_slices)
    n_metrics = 4

    # results[Iapp_idx, metric_idx, syni_idx, syne_idx]
    results = np.full((n_Iapp, n_metrics, n_i, n_e), np.nan)

    total = n_Iapp * n_e * n_i
    count = 0

    for a_idx, Iapp in enumerate(Iapp_slices):
        for j, G_syne in enumerate(G_syne_vals):
            for i, G_syni in enumerate(G_syni_vals):
                count += 1
                print(f"  [{count}/{total}]  Iapp={Iapp:.2f}  "
                      f"G_syne={G_syne:.3f}  G_syni={G_syni:.3f}", end='\r')

                sol = run_sim(G_syne, G_syni, Iapp)

                m0   = oscillation_metric(sol.t, sol.y[0])
                m1   = oscillation_metric(sol.t, sol.y[1])
                ph01 = phase_difference(sol.t, sol.y[0], sol.y[1])

                results[a_idx, 0, i, j] = float(m0['oscillates'])
                results[a_idx, 1, i, j] = float(m1['oscillates'])
                results[a_idx, 2, i, j] = m0['cv_isi']
                results[a_idx, 3, i, j] = ph01

    print()

    metric_labels = [
        "N0 oscillates",
        "N1 oscillates",
        "CV ISI – N0\n(lower = regular)",
        "Phase N0↔N1\n(0.5 = antiphase)",
    ]
    cmaps   = ['RdYlGn', 'RdYlGn', 'viridis_r', 'coolwarm']
    vranges = [(0, 1),   (0, 1),   (None, None), (0, 0.5)]

    fig, axes = plt.subplots(
        n_Iapp, n_metrics,
        figsize=(4 * n_metrics, 3.2 * n_Iapp),
        squeeze=False
    )
    fig.suptitle(
        f"Two-Neuron Exc-Inh Sweep — No Synaptic Fatigue  [Yuval]\n"
        f"Circuit: N0 –[exc]→ N1 –[inh]→ N0   (constant Iapp on N0 only)\n"
        f"Axes: G_syne (x) × G_syni (y), log scale  [0.01 – 1.0 nS]  |  "
        f"k_syn={settings.k_syn},  beta={settings.beta},  tau_w={tau_w} ms",
        fontsize=10
    )

    for a_idx, Iapp in enumerate(Iapp_slices):
        for m_idx in range(n_metrics):
            ax = axes[a_idx, m_idx]
            Z  = results[a_idx, m_idx]
            vmin, vmax = vranges[m_idx]

            im = ax.imshow(
                Z, origin='lower', aspect='auto',
                extent=[0, n_e, 0, n_i],
                cmap=cmaps[m_idx], vmin=vmin, vmax=vmax
            )
            fig.colorbar(im, ax=ax, shrink=0.8)

            xticks = np.arange(n_e)
            yticks = np.arange(n_i)
            ax.set_xticks(xticks[::2] + 0.5)
            ax.set_xticklabels([f"{v:.3f}" for v in G_syne_vals[::2]], fontsize=6, rotation=45)
            ax.set_yticks(yticks[::2] + 0.5)
            ax.set_yticklabels([f"{v:.3f}" for v in G_syni_vals[::2]], fontsize=6)

            if a_idx == 0:
                ax.set_title(metric_labels[m_idx], fontsize=9)
            if m_idx == 0:
                ax.set_ylabel(f"Iapp={Iapp:.2f} pA\nG_syni", fontsize=8)
            else:
                ax.set_ylabel("G_syni", fontsize=8)
            ax.set_xlabel("G_syne", fontsize=8)

    fig.tight_layout()

    if save:
        out = os.path.join(
            MEDIA_DIR,
            f"sweep_two_EI_no_sf_yuval_beta{settings.beta}_tauw{tau_w}.png"
        )
        plt.savefig(out, dpi=150)
        print(f"Saved → {out}")

    plt.show()


if __name__ == "__main__":
    run()
