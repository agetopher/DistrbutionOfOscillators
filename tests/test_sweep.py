import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import settings
from functions import *
from analysis import oscillation_metric, phase_difference

"""
Parameter sweep over G_syni x G_gap for the 2-cell mutual inhibition model.
Produces heatmaps of: whether the network oscillates, ISI regularity, and phase difference.
"""

# For Yuval Model
settings.C = 7.0
settings.g = 1.0
settings.L = -70.0
settings.T = -45.0
settings.H = -35.0
settings.m1 = 0.7
settings.m2 = 1/81.0
settings.m3 = -1/30.0
settings.m4 = 0.17

settings.G_syne = 0
settings.E_syne = 0
settings.E_syni = -100.0
settings.k_syn  = 0.125
settings.V_th   = -52.0
settings.a      = 0.000035
settings.b      = 0.005
settings.numCells = 2

MEDIA_DIR = os.path.join(os.path.dirname(__file__), '..', 'media')

# Fixed simulation parameters
Iapp        = 3
V0          = -45.0
sf0_init    = 0.3
tf          = 5000
dt          = 1
t           = np.linspace(0, tf, int(tf / dt))
V_init      = np.array([V0, settings.L - 0.5])
F_init      = np.array([sf0_init, 0.0])
inits       = np.append(V_init, F_init)


def run_sim(G_syni, G_gap):
    settings.G_syni = G_syni
    settings.G_gap  = G_gap

    def ode(t, y):
        fv = f_vec(y[0:2].reshape(2, 1))
        sf = sf_vec(y[0:2], y[2:])

        s0 = 1.0 / (1.0 + np.exp(-settings.k_syn * (y[0] - settings.V_th)))
        s1 = 1.0 / (1.0 + np.exp(-settings.k_syn * (y[1] - settings.V_th)))

        I_inh0 = settings.G_syni * s1 * y[3] * (y[0] - settings.E_syni)
        I_inh1 = settings.G_syni * s0 * y[2] * (y[1] - settings.E_syni)

        I_gap0 = settings.G_gap * (y[0] - y[1])
        I_gap1 = settings.G_gap * (y[1] - y[0])

        z = np.empty(4)
        z[0] = (settings.g * fv[0] - I_inh0 - I_gap0 + Iapp) / settings.C
        z[1] = (settings.g * fv[1] - I_inh1 - I_gap1 + Iapp) / settings.C
        z[2] = sf[0]
        z[3] = sf[1]
        return z

    return solve_ivp(ode, [0.0, tf], inits, method='BDF', t_eval=t)


def run(save=True):
    G_syni_vals = np.linspace(0.1, 1.0, 10)
    G_gap_vals  = np.linspace(0.0, 0.2, 10)

    osc_map   = np.zeros((len(G_gap_vals), len(G_syni_vals)))
    cv_map    = np.full((len(G_gap_vals), len(G_syni_vals)), np.nan)
    phase_map = np.full((len(G_gap_vals), len(G_syni_vals)), np.nan)

    total = len(G_syni_vals) * len(G_gap_vals)
    count = 0
    for j, gs in enumerate(G_syni_vals):
        for i, gg in enumerate(G_gap_vals):
            count += 1
            print(f"  [{count}/{total}]  G_syni={gs:.2f}  G_gap={gg:.3f}", end='\r')

            sol = run_sim(gs, gg)
            m   = oscillation_metric(sol.t, sol.y[0])
            ph  = phase_difference(sol.t, sol.y[0], sol.y[1])

            osc_map[i, j]   = float(m['oscillates'])
            cv_map[i, j]    = m['cv_isi']
            phase_map[i, j] = ph

    print()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle(f"2-Cell Sweep  (Iapp={Iapp}, sf0={sf0_init})", fontsize=13)

    extent = [G_syni_vals[0], G_syni_vals[-1], G_gap_vals[0], G_gap_vals[-1]]

    im0 = axes[0].imshow(osc_map,   origin='lower', aspect='auto', extent=extent, cmap='RdYlGn',   vmin=0, vmax=1)
    axes[0].set_title("Oscillates")
    axes[0].set_xlabel("G_syni");  axes[0].set_ylabel("G_gap")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(cv_map,    origin='lower', aspect='auto', extent=extent, cmap='viridis_r')
    axes[1].set_title("CV of ISI (lower = more regular)")
    axes[1].set_xlabel("G_syni");  axes[1].set_ylabel("G_gap")
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(phase_map, origin='lower', aspect='auto', extent=extent, cmap='coolwarm', vmin=0, vmax=0.5)
    axes[2].set_title("Phase difference (0.5 = antiphase)")
    axes[2].set_xlabel("G_syni");  axes[2].set_ylabel("G_gap")
    fig.colorbar(im2, ax=axes[2])

    fig.tight_layout()

    if save:
        plt.savefig(os.path.join(MEDIA_DIR, f"sweep_Iapp{Iapp}_sf{sf0_init}.png"))

    plt.show()


if __name__ == "__main__":
    run()
