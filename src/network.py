import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import settings
from functions import *

"""
Full network model
"""

"""
Parameters
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
settings.G_syne = 0.05
settings.E_syne = 0
settings.G_syni = 0.05
settings.E_syni = -100.0

# Synaptic gating parameters (sigmoid threshold)
settings.k_syn = 0.125 # sigmoid steepness
settings.V_th  = -52.0  # mV half-activation voltage

# Synaptic Fatigue parameters
settings.a = 0.000035
settings.b = 0.005

# Gap Junction parameters
settings.G_gap = 0.001

# Load connectivity matrices and initial voltages
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

# Raw files: row=post, col=pre. Used directly so that (E_conn @ pre)[i] = input to cell i.
settings.E_conn  = np.loadtxt(os.path.join(DATA_DIR, 'ConnectivityMatrix_SixSegments_ExcitatorySynapses.txt'), delimiter=',')
settings.I_conn  = np.loadtxt(os.path.join(DATA_DIR, 'ConnectivityMatrix_SixSegments_InhibitorySynapses.txt'), delimiter=',')
settings.GJ_conn = np.loadtxt(os.path.join(DATA_DIR, 'ConnectivityMatrix_SixSegments_GapJunctions.txt'),       delimiter=',')
V_init           = np.loadtxt(os.path.join(DATA_DIR, 'InitialVoltages.dat'))

settings.numCells = V_init.size

def ode(t, y):
    N  = settings.numCells
    V  = y[:N]
    SF = y[N:]

    fv = f_vec(V)  # shape (N,)

    # Presynaptic sigmoid gating variable for each cell
    s = 1.0 / (1.0 + np.exp(-settings.k_syn * (V - settings.V_th)))  # shape (N,)

    # Presynaptic signal weighted by synaptic fatigue
    pre = s * SF  # shape (N,)

    # Synaptic currents: sum over all presynaptic neurons j via connectivity matrices
    I_exc = settings.G_syne * (settings.E_conn  @ pre) * (V - settings.E_syne)
    I_inh = settings.G_syni * (settings.I_conn  @ pre) * (V - settings.E_syni)

    # Gap junction currents: G_gap * sum_j( GJ_conn[i,j] * (V[i] - V[j]) )
    I_gap = settings.G_gap  * (settings.GJ_conn.sum(axis=1) * V - settings.GJ_conn @ V)

    # Synaptic fatigue dynamics
    sf_dot = sf_vec(V, SF)

    dV = (settings.g * fv - I_exc - I_inh - I_gap) / settings.C

    z = np.empty(2 * N)
    z[:N]  = dV
    z[N:]  = sf_dot

    return z


MEDIA_DIR = os.path.join(os.path.dirname(__file__), '..', 'media')

def run(Iapp=0, tf=5000, save=True):
    N  = settings.numCells
    t  = np.linspace(0, tf, int(tf))

    SF_init = np.zeros(N)
    inits   = np.append(V_init, SF_init)

    print(f"Running {N}-cell network  (Iapp={Iapp}, tf={tf})...")
    sol = solve_ivp(
        lambda t, y: ode(t, y) + np.append(np.full(N, Iapp / settings.C), np.zeros(N)),
        [0.0, tf], inits, method='BDF', t_eval=t
    )
    print("Done.")

    V = sol.y[:N, :]   # shape (N, T)

    fig, axes = plt.subplots(
        2, 1, figsize=(12, 7),
        gridspec_kw={'height_ratios': [3, 1]}
    )
    fig.suptitle(
        f"Full Network  (N={N}, Iapp={Iapp}, G_syne={settings.G_syne}, "
        f"G_syni={settings.G_syni}, G_gap={settings.G_gap})",
        fontsize=12
    )

    # Voltage heatmap — all neurons
    im = axes[0].imshow(
        V, aspect='auto', origin='lower',
        extent=[sol.t[0], sol.t[-1], 0, N],
        cmap='RdBu_r', vmin=settings.L, vmax=settings.H
    )
    axes[0].set_title("Voltage (all neurons)")
    axes[0].set_ylabel("Neuron index")
    fig.colorbar(im, ax=axes[0], label="mV")

    # Mean voltage across all neurons
    axes[1].plot(sol.t, V.mean(axis=0), color='black', linewidth=0.8)
    axes[1].set_title("Mean voltage")
    axes[1].set_ylabel("mV")
    axes[1].set_xlabel("Time (ms)")

    fig.tight_layout()

    if save:
        plt.savefig(os.path.join(
            MEDIA_DIR,
            f"network_Iapp{Iapp}_Gsyne{settings.G_syne}_Gsyni{settings.G_syni}_Ggap{settings.G_gap}.png"
        ))

    plt.show()


if __name__ == "__main__":
    run()
