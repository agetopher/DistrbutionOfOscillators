import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import settings
from functions import *

"""
Testing two neurons for starting point of DE
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
settings.G_syni = 0.5
settings.E_syni = -100.0

# Synaptic gating parameters (sigmoid threshold)
settings.k_syn = 0.5 # sigmoid steepness
settings.V_th  = -52.0  # mV half-activation voltage

# Synaptic Fatigue parameters
settings.a = 0.000035
settings.b = 0.005

# Number of Cells
settings.numCells = 2

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

def run(Iapp=3, V0=-45.0, first_start=0.3, second_start=0.4, save=True, synapse_config='yuval'):
    # Synapse parameters — Boyle: steep step-like sigmoid activating at rest (k=100, V_th=-70)
    #                     Yuval: shallow sigmoid activating near threshold (k=0.5, V_th=-52)
    if synapse_config == 'boyle':
        k_syn  = 100.0
        V_th   = -70.0   # half-activation at resting potential
        G_syni = 0.3    # nS (10 pS)
    else:
        k_syn  = settings.k_syn
        V_th   = settings.V_th
        G_syni = settings.G_syni
    E_syni = settings.E_syni

    def ode(t, y):
        # y[0]: Neuron 0
        # y[1]: Neuron 1
        # y[2]: SF 0
        # y[3]: SF 1

        fv = f_vec(y[0:2].reshape(2, 1))
        sf = sf_vec(y[0:2].reshape(2,), y[2:].reshape(2,))

        # Presynaptic sigmoid gating variable
        s0 = 1.0 / (1.0 + np.exp(-k_syn * (y[0] - V_th)))
        s1 = 1.0 / (1.0 + np.exp(-k_syn * (y[1] - V_th)))

        I_inh0 = G_syni * s1 * (y[0] - E_syni) * y[3]
        I_inh1 = G_syni * s0 * (y[1] - E_syni) * y[2]

        z = np.empty(4,)
        z[0] = (settings.g*fv[0] - I_inh0 + Iapp) / settings.C
        z[1] = (settings.g*fv[1] - I_inh1 + Iapp) / settings.C
        z[2] = sf[0]
        z[3] = sf[1]

        return z

    # Set time
    t0 = 0
    tf = 5000
    dt = 1
    t = np.linspace(t0, tf, int((tf-t0)/dt))

    # Initial Voltages
    V_init = np.array([V0, settings.L-0.5]).reshape(settings.numCells, 1)

    # Initial Fatigue
    F_init_first = np.array([first_start, 1]).reshape(settings.numCells, 1)
    inits = np.append(V_init, F_init_first)
    sol = solve_ivp(ode, [0.0, tf], inits, method='BDF', t_eval=t)

    F_init_second = np.array([second_start, 1]).reshape(settings.numCells, 1)
    inits1 = np.append(V_init, F_init_second)
    sol1 = solve_ivp(ode, [0.0, tf], inits1, method='BDF', t_eval=t)

    fig, axes = plt.subplots(2, 2)
    fig.suptitle(f"Mutual Inhibition  Iapp={Iapp}  [{synapse_config} synapses]", fontsize=14)

    # "bad" plot
    axes[0, 0].set_title(f"sf0={first_start}, sf1={F_init_first[1,0]}")
    axes[0, 0].plot(sol.t, sol.y[0, :], color="blue")
    axes[0, 0].plot(sol.t, sol.y[1, :], color="red")
    axes[0, 0].set_ylabel("Voltage (mV)")
    axes[1, 0].plot(sol.t, sol.y[2, :], color="blue")
    axes[1, 0].plot(sol.t, sol.y[3, :], color="red")
    axes[1, 0].set_ylabel("Synaptic Fatigue")
    axes[1, 0].set_xlabel("Time (ms)")

    # "good" plot
    axes[0, 1].set_title(f"sf0={second_start}, sf1={F_init_second[1,0]}")
    axes[0, 1].plot(sol1.t, sol1.y[0, :], color="blue")
    axes[0, 1].plot(sol1.t, sol1.y[1, :], color="red")
    axes[1, 1].plot(sol1.t, sol1.y[2, :], color="blue")
    axes[1, 1].plot(sol1.t, sol1.y[3, :], color="red")
    axes[1, 1].set_xlabel("Time (ms)")

    fig.tight_layout()

    if save:
        plt.savefig(os.path.join(MEDIA_DIR, f"test_two_II_Iapp{Iapp}_sf0{first_start*100}_sf1{second_start*100}_{synapse_config}.png"))

    plt.show()


if __name__ == "__main__":
    for cfg in ('yuval', ):
        run(Iapp=2, synapse_config=cfg)
        run(Iapp=3, synapse_config=cfg)
        run(Iapp=4, synapse_config=cfg)
