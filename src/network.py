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

# Start from the learned full-network baseline in settings.py.
settings.reset_defaults()

# Load connectivity matrices and initial voltages
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

# Raw files: row=post, col=pre. Used directly so that (E_conn @ pre)[i] = input to cell i.
E_CONN  = np.loadtxt(os.path.join(DATA_DIR, 'ConnectivityMatrix_SixSegments_ExcitatorySynapses.txt'), delimiter=',')
I_CONN  = np.loadtxt(os.path.join(DATA_DIR, 'ConnectivityMatrix_SixSegments_InhibitorySynapses.txt'), delimiter=',')
GJ_CONN = np.loadtxt(os.path.join(DATA_DIR, 'ConnectivityMatrix_SixSegments_GapJunctions.txt'),       delimiter=',')
V_init           = np.loadtxt(os.path.join(DATA_DIR, 'InitialVoltages.dat'))
# V_init           = np.ones(V_init.size)*-70.0
classes          = np.loadtxt(os.path.join(DATA_DIR, 'CellsClassification.dat')).astype(int)

N_CELLS = V_init.size

# Command-interneuron injection targets (by cell class)
AVA_CLASSES = (1, 2, 7)   # AS, DA, VA
AVB_CLASSES = (1, 3, 6)   # AS, DB, VB  (AS receives both IAVA and IAVB)

def configure():
    """Attach the fixed 102-cell circuit to the shared scalar settings."""
    settings.numCells = N_CELLS
    settings.E_conn = E_CONN
    settings.I_conn = I_CONN
    settings.GJ_conn = GJ_CONN
    settings.I_inj = np.zeros(N_CELLS)


configure()


def drive_vector(IAVA=0.0, IAVB=0.0):
    """Return the class-targeted command-current vector in pA."""
    current = np.zeros(settings.numCells)
    current[np.isin(classes, AVA_CLASSES)] += IAVA
    current[np.isin(classes, AVB_CLASSES)] += IAVB
    return current


def reduced_rest_state():
    """Return the reduced ``[V, w]`` state used by long network experiments."""
    return np.concatenate([V_init.copy(), np.zeros(settings.numCells)])


def rhs_vw(state, current):
    """Reduced full-network dynamics with inert synaptic fatigue omitted."""
    N = settings.numCells
    V = state[:N]
    w = state[N:]
    pre = 1.0 / (1.0 + np.exp(-settings.k_syn * (V - settings.V_th)))
    I_exc = settings.G_syne * (settings.E_conn @ pre) * (V - settings.E_syne)
    I_inh = settings.G_syni * (settings.I_conn @ pre) * (V - settings.E_syni)
    Vdiff = V[:, None] - V[None, :]
    I_gap = settings.G_gap * (settings.GJ_conn * Vdiff).sum(axis=1)
    dV = (
        settings.g * f_vec(V) - w - I_exc - I_inh - I_gap + current
    ) / settings.C
    dw = (w_inf_vec(V) - w) / settings.tau_w
    return np.concatenate([dV, dw])


def ode(t, y):
    N  = settings.numCells
    V  = y[:N]
    SF = y[N:2*N]
    w  = y[2*N:]

    fv   = f_vec(V)        # shape (N,)
    winf = w_inf_vec(V)    # steady-state recovery, shape (N,)

    # Presynaptic sigmoid gating variable for each cell
    s = 1.0 / (1.0 + np.exp(-settings.k_syn * (V - settings.V_th)))  # shape (N,)

    # Presynaptic signal weighted by synaptic fatigue
    pre = s # * SF  # shape (N,)

    # Synaptic currents: sum over all presynaptic neurons j via connectivity matrices
    I_exc = settings.G_syne * (settings.E_conn  @ pre) * (V - settings.E_syne)
    I_inh = settings.G_syni * (settings.I_conn  @ pre) * (V - settings.E_syni)

    # Gap junction currents: G_gap * sum_j( GJ_conn[i,j] * (V[i] - V[j]) )
    # Vdiff[i,j] = V_i - V_j; elementwise with GJ_conn, then sum over j.
    Vdiff = V[:, None] - V[None, :]
    I_gap = settings.G_gap  * (settings.GJ_conn * Vdiff).sum(axis=1)

    # Synaptic fatigue dynamics
    sf_dot = sf_vec(V, SF)

    dV = (settings.g * fv - w - I_exc - I_inh - I_gap + settings.I_inj) / settings.C
    dw = (winf - w) / settings.tau_w

    z = np.empty(3 * N)
    z[:N]     = dV
    z[N:2*N]  = sf_dot
    z[2*N:]   = dw

    return z


MEDIA_DIR = os.path.join(os.path.dirname(__file__), '..', 'media')

def run(IAVA=0.0, IAVB=0.0, tf=5000, save=False):
    N  = settings.numCells
    t  = np.linspace(0, tf, int(tf))

    # Class-targeted command-interneuron drives. AS (class 1) gets both.
    settings.I_inj = drive_vector(IAVA, IAVB)   # read inside ode() -> dV

    SF_init = np.zeros(N)
    w_init  = np.zeros(N)
    inits   = np.concatenate([V_init, SF_init, w_init])

    print(f"Running {N}-cell network  (IAVA={IAVA}, IAVB={IAVB}, tf={tf})...")
    sol = solve_ivp(ode, [0.0, tf], inits, method='BDF', t_eval=t)
    print("Done.")

    V = sol.y[:N, :]   # shape (N, T)

    # Muscle cells, already ordered anterior -> posterior (segment by segment)
    dorsal_idx  = np.where(classes == 8)[0]
    ventral_idx = np.where(classes == 9)[0]
    V_dorsal    = V[dorsal_idx, :]
    V_ventral   = V[ventral_idx, :]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle(
        f"Muscle kymogram  (N={N}, IAVA={IAVA}, IAVB={IAVB}, G_syne={settings.G_syne}, "
        f"G_syni={settings.G_syni}, G_gap={settings.G_gap})",
        fontsize=12
    )

    # Kymograms: body position (anterior at top) on y, time on x, voltage in colour
    for ax, data, name in ((axes[0], V_dorsal, 'Dorsal'),
                           (axes[1], V_ventral, 'Ventral')):
        im = ax.imshow(
            data, aspect='auto', origin='upper',
            extent=[sol.t[0], sol.t[-1], data.shape[0], 0],
            cmap='viridis', vmin=settings.L, vmax=settings.H
        )
        ax.set_title(f"{name} muscle activation")
        ax.set_ylabel("body position\n(anterior → posterior)")
        fig.colorbar(im, ax=ax, label="mV")
    axes[1].set_xlabel("Time (ms)")

    fig.tight_layout()

    if save:
        plt.savefig(os.path.join(
            MEDIA_DIR,
            f"network_IAVA{IAVA}_IAVB{IAVB}_Gsyne{settings.G_syne}_Gsyni{settings.G_syni}_Ggap{settings.G_gap}.png"
        ))

    plt.show()


if __name__ == "__main__":
    run(IAVA=2.5, IAVB=0.0, save=False)
