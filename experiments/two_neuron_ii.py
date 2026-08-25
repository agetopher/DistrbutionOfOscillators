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

settings.reset_defaults()

# Intentional legacy mutual-inhibition configuration.
settings.G_syni = 0.5
settings.k_syn = 0.5
settings.beta = 0.2
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

def run(Iapp=3, V0=-45.0, first_start=0.3, second_start=0.4, save=False):
    # Synapse parameters
    k_syn  = settings.k_syn
    V_th   = settings.V_th
    G_syni = settings.G_syni
    E_syni = settings.E_syni

    def ode(t, y):
        # y[0]: Neuron 0
        # y[1]: Neuron 1
        # y[2]: SF 0
        # y[3]: SF 1
        # y[4]: w0 (recovery variable)
        # y[5]: w1 (recovery variable)

        fv = f_vec(y[0:2].reshape(2, 1))
        sf = sf_vec(y[0:2].reshape(2,), y[2:4].reshape(2,))
        w  = y[4:6]

        # Presynaptic sigmoid gating variable
        s0 = 1.0 / (1.0 + np.exp(-k_syn * (y[0] - V_th)))
        s1 = 1.0 / (1.0 + np.exp(-k_syn * (y[1] - V_th)))

        I_inh0 = G_syni * s1 * (y[0] - E_syni) * y[3]
        I_inh1 = G_syni * s0 * (y[1] - E_syni) * y[2]

        w_inf = w_inf_vec(y[0:2].reshape(2,))

        z = np.empty(6,)
        z[0] = (settings.g*fv[0] - w[0] - I_inh0 + Iapp) / settings.C
        z[1] = (settings.g*fv[1] - w[1] - I_inh1 + Iapp) / settings.C
        z[2] = sf[0]
        z[3] = sf[1]
        z[4] = (w_inf[0] - w[0]) / settings.tau_w
        z[5] = (w_inf[1] - w[1]) / settings.tau_w

        return z

    # Set time
    t0 = 0
    tf = 5000
    dt = 1
    t = np.linspace(t0, tf, int((tf-t0)/dt))

    # Initial Voltages
    V_init = np.array([V0, settings.L-0.5]).reshape(settings.numCells, 1)

    # Initial Fatigue and recovery variables (w starts at 0)
    W_init = np.zeros(settings.numCells)

    F_init_first = np.array([first_start, 1]).reshape(settings.numCells, 1)
    inits = np.append(np.append(V_init, F_init_first), W_init)
    sol = solve_ivp(ode, [0.0, tf], inits, method='BDF', t_eval=t)

    F_init_second = np.array([second_start, 1]).reshape(settings.numCells, 1)
    inits1 = np.append(np.append(V_init, F_init_second), W_init)
    sol1 = solve_ivp(ode, [0.0, tf], inits1, method='BDF', t_eval=t)

    fig, axes = plt.subplots(3, 2, figsize=(10, 8))
    fig.suptitle(
        f"Mutual Inhibition  Iapp={Iapp} ]\n"
        f"beta={settings.beta} nS/mV,  tau_w={settings.tau_w} ms",
        fontsize=13
    )

    # "bad" plot
    axes[0, 0].set_title(f"sf0={first_start}, sf1={F_init_first[1,0]}")
    axes[0, 0].plot(sol.t, sol.y[0, :], color="blue")
    axes[0, 0].plot(sol.t, sol.y[1, :], color="red")
    axes[0, 0].set_ylabel("Voltage (mV)")
    axes[1, 0].plot(sol.t, sol.y[2, :], color="blue")
    axes[1, 0].plot(sol.t, sol.y[3, :], color="red")
    axes[1, 0].set_ylabel("Synaptic Fatigue")
    axes[2, 0].plot(sol.t, sol.y[4, :], color="blue", label="w0")
    axes[2, 0].plot(sol.t, sol.y[5, :], color="red",  label="w1")
    axes[2, 0].set_ylabel("Recovery w (nS)")
    axes[2, 0].set_xlabel("Time (ms)")
    axes[2, 0].legend(fontsize=8)

    # "good" plot
    axes[0, 1].set_title(f"sf0={second_start}, sf1={F_init_second[1,0]}")
    axes[0, 1].plot(sol1.t, sol1.y[0, :], color="blue")
    axes[0, 1].plot(sol1.t, sol1.y[1, :], color="red")
    axes[1, 1].plot(sol1.t, sol1.y[2, :], color="blue")
    axes[1, 1].plot(sol1.t, sol1.y[3, :], color="red")
    axes[2, 1].plot(sol1.t, sol1.y[4, :], color="blue", label="w0")
    axes[2, 1].plot(sol1.t, sol1.y[5, :], color="red",  label="w1")
    axes[2, 1].set_xlabel("Time (ms)")
    axes[2, 1].legend(fontsize=8)

    fig.tight_layout()

    if save:
        plt.savefig(os.path.join(MEDIA_DIR, f"recovery_test_two_II_Iapp{Iapp}_sf0{first_start*100}_sf1{second_start*100}.png"))

    plt.show()


if __name__ == "__main__":
    run(Iapp=2)
    run(Iapp=3)
    run(Iapp=4) 
