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
Parameter sweep over G_syni for the 2-cell mutual inhibition model.
Produces plots of: whether the network oscillates, ISI regularity, and phase difference.
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


def run_sim(G_syni, k_syn, V_th):
    def ode(t, y):
        fv = f_vec(y[0:2].reshape(2, 1))
        sf = sf_vec(y[0:2], y[2:])

        s0 = 1.0 / (1.0 + np.exp(-k_syn * (y[0] - V_th)))
        s1 = 1.0 / (1.0 + np.exp(-k_syn * (y[1] - V_th)))

        I_inh0 = G_syni * s1 * y[3] * (y[0] - settings.E_syni)
        I_inh1 = G_syni * s0 * y[2] * (y[1] - settings.E_syni)

        z = np.empty(4)
        z[0] = (settings.g * fv[0] - I_inh0 + Iapp) / settings.C
        z[1] = (settings.g * fv[1] - I_inh1 + Iapp) / settings.C
        z[2] = sf[0]
        z[3] = sf[1]
        return z

    return solve_ivp(ode, [0.0, tf], inits, method='BDF', t_eval=t)


def run(save=True, synapse_config='yuval'):
    # Boyle: step-like sigmoid at rest — effective at much lower conductances
    # Yuval: shallow sigmoid near threshold — needs higher conductances
    if synapse_config == 'boyle':
        k_syn       = 100.0
        V_th        = -70.0
        G_syni_vals = np.linspace(0.0001, 0.5, 300)
    else:
        k_syn       = settings.k_syn
        V_th        = settings.V_th
        G_syni_vals = np.linspace(0.1, 1.0, 30)

    osc_arr   = np.zeros(len(G_syni_vals))
    cv_arr    = np.full(len(G_syni_vals), np.nan)
    phase_arr = np.full(len(G_syni_vals), np.nan)

    for j, gs in enumerate(G_syni_vals):
        print(f"  [{j+1}/{len(G_syni_vals)}]  G_syni={gs:.4f}", end='\r')

        sol = run_sim(gs, k_syn, V_th)
        m   = oscillation_metric(sol.t, sol.y[0])
        ph  = phase_difference(sol.t, sol.y[0], sol.y[1])

        osc_arr[j]   = float(m['oscillates'])
        cv_arr[j]    = m['cv_isi']
        phase_arr[j] = ph

    print()

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    fig.suptitle(f"2-Cell Mutual Inhibition Sweep  (Iapp={Iapp}, sf0={sf0_init}, [{synapse_config} synapses])", fontsize=13)

    axes[0].plot(G_syni_vals, osc_arr, color='green', marker='o', markersize=4)
    axes[0].set_ylabel("Oscillates")
    axes[0].set_ylim(-0.1, 1.1)
    axes[0].set_yticks([0, 1])

    axes[1].plot(G_syni_vals, cv_arr, color='steelblue', marker='o', markersize=4)
    axes[1].set_ylabel("CV of ISI")

    axes[2].plot(G_syni_vals, phase_arr, color='tomato', marker='o', markersize=4)
    axes[2].axhline(0.5, color='gray', linestyle='--', linewidth=0.8)
    axes[2].set_ylabel("Phase difference")
    axes[2].set_xlabel("G_syni (nS)")
    axes[2].set_ylim(0, 1)

    fig.tight_layout()

    if save:
        plt.savefig(os.path.join(MEDIA_DIR, f"sweep_two_II_Iapp{Iapp}_sf{sf0_init}_{synapse_config}.png"))

    plt.show()


if __name__ == "__main__":
    # run(synapse_config='yuval')
    run(synapse_config='boyle')
