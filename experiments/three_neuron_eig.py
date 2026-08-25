import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import settings
from functions import *

"""
Testing three neurons
NOT WORKING
"""

settings.reset_defaults()

# Intentional legacy three-cell configuration.
settings.G_syne = 0.5
settings.G_syni = 0.25
settings.k_syn = 0.125
settings.beta = 0.1
settings.G_gap = 0.0
settings.numCells = 3

# Test synaptic gating variables
'''
V = np.linspace(-100, 20, 100)
s = 1.0 / (1.0 + np.exp(-settings.k_syn * (V - settings.V_th)))
plt.plot(V, s)
plt.xlabel("Voltage")
plt.ylabel("Synaptic Gating Variable")
plt.show()
'''

MEDIA_DIR = os.path.join(os.path.dirname(__file__), '..', 'media')

def run(Ion=3, V0=settings.L, save=False):
    # Synapse parameters 
    k_exc  = settings.k_syn
    k_inh  = settings.k_syn
    V_th   = settings.V_th
    G_syne = settings.G_syne
    G_syni = settings.G_syni
    E_syne = settings.E_syne
    E_syni = settings.E_syni

    def ode(t, y):
        # y[0]: Neuron 0 (excitatory — drives inh onto neuron 1 via exc synapse)
        # y[1]: Neuron 1 (inhibitory — drives inh onto neuron 0)
        # y[2]: Neuron 2 (gap-junction follower of neuron 1)
        # y[3]: SF 0
        # y[4]: SF 1
        # y[5]: w0 (recovery variable)
        # y[6]: w1 (recovery variable)
        # y[7]: w2 (recovery variable)

        fv = f_vec(y[0:3].reshape(3, 1))
        sf = sf_vec(y[0:2].reshape(2,), y[3:5].reshape(2,))
        w  = y[5:8]

        # Presynaptic sigmoid gating — neuron 0 is excitatory, neuron 1 is inhibitory
        s0 = 1.0 / (1.0 + np.exp(-k_exc * (y[0] - V_th)))
        s1 = 1.0 / (1.0 + np.exp(-k_inh * (y[1] - V_th)))

        I_inh0 = G_syni * s1 * (y[0] - E_syni) * y[4]
        I_exc1 = G_syne * s0 * (y[1] - E_syne) * y[3]

        # Gap junction connection from Neuron 1 to Neuron 2
        I_gap2 = settings.G_gap * (y[2] - y[1])
        I_gap3 = settings.G_gap * (y[1] - y[2])

        Iapp = 1.5

        w_inf = w_inf_vec(y[0:3].reshape(3,))

        z = np.empty(8,)
        z[0] = (settings.g*fv[0] - w[0] - I_inh0 + Iapp) / settings.C
        z[1] = (settings.g*fv[1] - w[1] - I_exc1 - I_gap2) / settings.C
        z[2] = (settings.g*fv[2] - w[2] - I_gap3) / settings.C
        z[3] = sf[0]
        z[4] = sf[1]
        z[5] = (w_inf[0] - w[0]) / settings.tau_w
        z[6] = (w_inf[1] - w[1]) / settings.tau_w
        z[7] = (w_inf[2] - w[2]) / settings.tau_w

        return z

    # Set time
    t0 = 0
    tf = 5000
    dt = 1
    t = np.linspace(t0, tf, int((tf-t0)/dt))

    # Initial Voltages
    V_init = np.array([V0, V0, V0]).reshape(settings.numCells, 1)

    # Initial Fatigue and recovery variables (w starts at 0)
    F_init = np.array([1, 1]).reshape(settings.numCells-1, 1)
    W_init = np.zeros(settings.numCells)
    inits = np.append(np.append(V_init, F_init), W_init)
    sol = solve_ivp(ode, [0.0, tf], inits, method='BDF', t_eval=t, max_step=5.0, rtol=1e-3, atol=1e-5)

    fig, axes = plt.subplots(4, 1, figsize=(10, 10))
    fig.suptitle(
        f"Three Neurons \n"
        f"beta={settings.beta} nS/mV,  tau_w={settings.tau_w} ms",
        fontsize=13
    )

    axes[0].set_title(f"exc: 0 -> 1, inh: 1 -> 0, gj: 1 -> 2")
    axes[0].plot(sol.t, sol.y[0, :], color="blue",   label="N0 (exc)")
    axes[0].plot(sol.t, sol.y[1, :], color="red",    label="N1 (inh)")
    axes[0].plot(sol.t, sol.y[2, :], color="orange", linestyle=":", label="N2 (gj)")
    axes[0].axhline(settings.L, color="steelblue",   linestyle=":", linewidth=1.0, label="L (rest)")
    axes[0].axhline(settings.T, color="forestgreen", linestyle=":", linewidth=1.0, label="T (threshold)")
    axes[0].axhline(settings.H, color="darkorange",  linestyle=":", linewidth=1.0, label="H (plateau)")
    axes[0].legend(fontsize=7, loc="upper right")
    axes[0].set_ylabel("Voltage (mV)")
    axes[1].plot(sol.t, -G_syne * (sol.y[1,:] - E_syne) * sol.y[3, :] / (1.0 + np.exp(-k_exc * (sol.y[0,:] - V_th))), color="red")
    axes[1].plot(sol.t, -G_syni * (sol.y[0,:] - E_syni) * sol.y[4, :] / (1.0 + np.exp(-k_inh * (sol.y[1,:] - V_th))), color='blue')
    axes[1].set_ylabel("Synapse Activity")
    axes[2].plot(sol.t, sol.y[3, :], color="blue", label="SF0")
    axes[2].plot(sol.t, sol.y[4, :], color="red",  label="SF1")
    axes[2].set_ylabel("Synaptic Efficacy")
    axes[2].legend(fontsize=7)
    axes[3].plot(sol.t, sol.y[5, :], color="blue",   label="w0")
    axes[3].plot(sol.t, sol.y[6, :], color="red",    label="w1")
    axes[3].plot(sol.t, sol.y[7, :], color="orange", label="w2", linestyle=":")
    axes[3].set_ylabel("Recovery w (nS)")
    axes[3].set_xlabel("Time (ms)")
    axes[3].legend(fontsize=7)

    fig.tight_layout()

    if save:
        plt.savefig(os.path.join(MEDIA_DIR, f"test_three_EIG_voltage_fatigue.png"))

    plt.show()


if __name__ == "__main__":
    run(V0=-75.0)
