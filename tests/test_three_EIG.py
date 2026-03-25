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

# For Yuval Model
settings.C = 7.0    # pF Membrane Capacitance
settings.g = 1.0    # nS Membrane Conductance
settings.L = -70.0  # mV Resting Potential
settings.T = -45.0  # mV Depolarization Threshold
settings.H = -35.0  # mV Plateau Potential
settings.m1 = 0.7
settings.m2 = 1/81.0
settings.m3 = -1/30.0
settings.m4 = 0.17

# Synapses
settings.G_syne = 0.5
settings.E_syne = 0
settings.G_syni = 0.25
settings.E_syni = -100.0

# Synaptic gating parameters (sigmoid threshold)
settings.k_syn = 0.125 # sigmoid steepness
settings.V_th  = -52.0  # mV half-activation voltage

# Synaptic Fatigue parameters
settings.a = 0.000035
settings.b = 0.005

# Gap Junction parameters 
settings.G_gap = 0.0

# Number of Cells
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

def run(Ion=3, V0=settings.L, save=True, synapse_config='yuval'):
    # Synapse parameters — Boyle: steep sigmoids activating at rest
    #   excitatory (ACh):  k=500, V_th=-70, G_syne=0.02 nS
    #   inhibitory (GABA): k=100, V_th=-70, G_syni=0.01 nS
    # Yuval: single shallow sigmoid (k=0.125, V_th=-52) for both
    if synapse_config == 'boyle':
        k_exc  = 500.0
        k_inh  = 100.0
        V_th   = -70.0   # half-activation at resting potential
        G_syne = 0.02    # nS (20 pS)
        G_syni = 0.01    # nS (10 pS)
    else:
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

        fv = f_vec(y[0:3].reshape(3, 1))
        sf = sf_vec(y[0:2].reshape(2,), y[3:].reshape(2,))

        # Presynaptic sigmoid gating — neuron 0 is excitatory, neuron 1 is inhibitory
        s0 = 1.0 / (1.0 + np.exp(-k_exc * (y[0] - V_th)))
        s1 = 1.0 / (1.0 + np.exp(-k_inh * (y[1] - V_th)))

        I_inh0 = G_syni * s1 * (y[0] - E_syni) * y[4]
        I_exc1 = G_syne * s0 * (y[1] - E_syne) * y[3]

        # Gap junction connection from Neuron 1 to Neuron 2
        I_gap3 = settings.G_gap * (y[1] - y[2])

        Iapp = 0

        z = np.empty(5,)
        z[0] = (settings.g*fv[0] - I_inh0 + Iapp) / settings.C
        z[1] = (settings.g*fv[1] - I_exc1 - 0.5*Iapp) / settings.C
        z[2] = (settings.g*fv[2] - I_gap3) / settings.C
        z[3] = sf[0]
        z[4] = sf[1]

        return z

    # Set time
    t0 = 0
    tf = 5000
    dt = 1
    t = np.linspace(t0, tf, int((tf-t0)/dt))

    # Initial Voltages
    V_init = np.array([V0, V0, V0]).reshape(settings.numCells, 1)

    # Initial Fatigue
    F_init = np.array([1, 1]).reshape(settings.numCells-1, 1)
    inits = np.append(V_init, F_init)
    sol = solve_ivp(ode, [0.0, tf], inits, method='BDF', t_eval=t, max_step=5.0, rtol=1e-3, atol=1e-5)

    fig, axes = plt.subplots(3, 1)
    fig.suptitle(f"Three Neurons  [{synapse_config} synapses]", fontsize=14)

    axes[0].set_title(f"exc: 0 -> 1, inh: 1 -> 0, gj: 1 -> 2")
    axes[0].plot(sol.t, sol.y[0, :], color="blue")
    axes[0].plot(sol.t, sol.y[1, :], color="red")
    axes[0].plot(sol.t, sol.y[2, :], color="orange", linestyle=":")
    axes[0].set_ylabel("Voltage (mV)")
    axes[1].plot(sol.t, -G_syne * (sol.y[1,:] - E_syne) * sol.y[3, :] / (1.0 + np.exp(-k_exc * (sol.y[0,:] - V_th))), color="red")
    axes[1].plot(sol.t, -G_syni * (sol.y[0,:] - E_syni) * sol.y[4, :] / (1.0 + np.exp(-k_inh * (sol.y[1,:] - V_th))), color='blue')
    axes[1].set_ylabel("Synapse Activity")
    axes[2].plot(sol.t, sol.y[3, :], color="blue")
    axes[2].plot(sol.t, sol.y[4, :], color="red")
    axes[2].set_ylabel("Synaptic Efficacy")
    axes[2].set_xlabel("Time (ms)")

    fig.tight_layout()

    if save:
        plt.savefig(os.path.join(MEDIA_DIR, f"test_three_EIG_voltage_fatigue_{synapse_config}.png"))

    plt.show()


if __name__ == "__main__":
    for cfg in ('yuval', 'boyle'):
        run(V0=-75.0, synapse_config=cfg)