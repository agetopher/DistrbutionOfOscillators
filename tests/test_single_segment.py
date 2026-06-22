import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import settings
from functions import f_vec, sf_vec, w_inf_vec

"""
Single-segment test: first 17 cells of the 102-cell network, driven by two
constant current injections that stand in for the command interneurons
AVA and AVB.

Segment layout (17 cells, classes 1-9; cell IDs and functional groups):
  Class 1 : cells 0-1   (2)  AS  — Group A, dorsal excitatory
  Class 2 : cell  2     (1)  DA  — Group A, dorsal excitatory
  Class 3 : cell  3     (1)  DB  — Group A, dorsal excitatory
  Class 4 : cell  4     (1)  DD  — inhibitory (GABA)
  Class 5 : cells 5-6   (2)  VD  — inhibitory (GABA)
  Class 6 : cells 7-8   (2)  VB  — Group B, ventral excitatory
  Class 7 : cells 9-10  (2)  VA  — Group B, ventral excitatory
  Class 8 : cells 11-13 (3)  dorsal muscle   (passive)
  Class 9 : cells 14-16 (3)  ventral muscle  (passive)

Functional grouping (see docs/circuit_analysis):
  Group A = AS, DA, DB (classes 1,2,3): dorsal excitatory; NO direct inhibitory
            input — return to rest only via gap junctions.
  Group B = VB, VA (classes 6,7): ventral excitatory; receive direct inhibition
            from VD (class 5).
  Inhibitory: DD (class 4) inhibits VD and dorsal muscle; VD (class 5) inhibits
            Group B and ventral muscle.

Command-interneuron current injections (constant):
  IAVA → AS (class 1), DA (class 2), VA (class 7)
  IAVB → AS (class 1), DB (class 3), VB (class 6)
  AS receives both IAVA and IAVB.

State: [V (17), SF (17), w (17)].
"""

# ── Data ─────────────────────────────────────────────────────────────────────
DATA_DIR  = os.path.join(os.path.dirname(__file__), '..', 'data')
MEDIA_DIR = os.path.join(os.path.dirname(__file__), '..', 'media')
SEG       = 17

E_conn  = np.loadtxt(os.path.join(DATA_DIR, 'ConnectivityMatrix_SixSegments_ExcitatorySynapses.txt'), delimiter=',')[:SEG, :SEG]
I_conn  = np.loadtxt(os.path.join(DATA_DIR, 'ConnectivityMatrix_SixSegments_InhibitorySynapses.txt'), delimiter=',')[:SEG, :SEG]
GJ_conn = np.loadtxt(os.path.join(DATA_DIR, 'ConnectivityMatrix_SixSegments_GapJunctions.txt'),       delimiter=',')[:SEG, :SEG]
V_init  = np.loadtxt(os.path.join(DATA_DIR, 'InitialVoltages.dat'))[:SEG]
V_init  = np.ones(V_init.shape)*-70.0
classes = np.loadtxt(os.path.join(DATA_DIR, 'CellsClassification.dat')).astype(int)[:SEG]

# ── Parameters ────────────────────────────────────────────────────────────────
settings.C   = 7.0
settings.g   = 1.0
settings.L   = -70.0
settings.T   = -45.0
settings.H   = -35.0
settings.m1  = 0.7
settings.m2  = 1 / 81.0
settings.m3  = -1 / 30.0
settings.m4  = 0.17

# Segment synaptic parameters
settings.G_syne = 0.07
settings.E_syne = 0.0
settings.G_syni = 0.05
settings.E_syni = -100.0
settings.k_syn  = 0.25
settings.V_th   = -52.0

settings.a = 0.000035
settings.b = 0.005

settings.G_gap = 0.03

# Recovery variable
settings.beta = 1.0
tau_w         = 200.0

# ── Circuit indices ───────────────────────────────────────────────────────────
N = SEG
settings.numCells = N

# Command-interneuron injection targets (by cell class)
AVA_CLASSES = (1, 2, 7)   # AS, DA, VA
AVB_CLASSES = (1, 3, 6)   # AS, DB, VB

# ── Class colours / names for plotting ────────────────────────────────────────
CLASS_COLORS = {
    1: 'royalblue', 2: 'darkorange', 3: 'forestgreen',
    4: 'crimson',   5: 'purple',     6: 'saddlebrown',
    7: 'teal',      8: 'gray',       9: 'lightcoral',
}

CLASS_NAMES = {
    1: 'AS', 2: 'DA', 3: 'DB', 4: 'DD', 5: 'VD',
    6: 'VB', 7: 'VA', 8: 'dorsal muscle', 9: 'ventral muscle',
}


def run(IAVA=0.0, IAVB=0.0, tf=5000, save=False):

    # Constant current injection vector. AS (class 1) is targeted by both,
    # so it receives IAVA + IAVB.
    I_inj = np.zeros(N)
    I_inj[np.isin(classes, AVA_CLASSES)] += IAVA
    I_inj[np.isin(classes, AVB_CLASSES)] += IAVB

    def ode(t, y):
        V  = y[:N]
        SF = y[N:2*N]
        w  = y[2*N:]

        fv   = f_vec(V)
        winf = w_inf_vec(V)

        s = 1.0 / (1.0 + np.exp(-settings.k_syn * (V - settings.V_th)))

        # ── Segment internal dynamics (SF-gated) ──────────────────────────────
        pre = s # * SF

        # Connectivity files are row=post, col=pre, so (conn @ pre)[i] = input to cell i
        # (matches src/network.py). No transpose.
        I_exc = settings.G_syne * (E_conn  @ pre) * (V - settings.E_syne)
        I_inh = settings.G_syni * (I_conn  @ pre) * (V - settings.E_syni)
        # Gap junctions: ohmic sum over neighbours, sum_j GJ[i,j]*(V_i - V_j).
        # Vdiff[i,j] = V_i - V_j; elementwise with GJ_conn, then sum over j.
        Vdiff = V[:, None] - V[None, :]
        I_gap = settings.G_gap  * (GJ_conn * Vdiff).sum(axis=1)

        # ── dV ────────────────────────────────────────────────────────────────
        dV = (settings.g * fv - w
              - I_exc - I_inh - I_gap
              + I_inj) / settings.C

        sf_dot = sf_vec(V, SF)
        dw     = (winf - w) / tau_w

        z = np.empty(3 * N)
        z[:N]      = dV
        z[N:2*N]   = sf_dot
        z[2*N:]    = dw
        return z

    t = np.linspace(0, tf, int(tf))

    V_init_clipped = np.where(V_init > settings.H, settings.T, V_init)
    SF_init = np.full(N, 0.5)
    w_init  = np.zeros(N)

    inits = np.concatenate([V_init_clipped, SF_init, w_init])

    print(f"Running single segment ({N} cells)  IAVA={IAVA} pA, IAVB={IAVB} pA...")
    sol = solve_ivp(ode, [0.0, tf], inits, method='BDF', t_eval=t)
    print("Done.")

    V_sol = sol.y[:N, :]
    w_sol = sol.y[2*N:, :]

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    fig.suptitle(
        f"Single Segment  IAVA={IAVA} pA (AS,DA,VA), IAVB={IAVB} pA (AS,DB,VB)\n"
        f"G_syne={settings.G_syne}, G_syni={settings.G_syni}, G_gap={settings.G_gap} nS  |  "
        f"beta={settings.beta}, tau_w={tau_w} ms",
        fontsize=11
    )

    hlines = [(settings.L, 'steelblue', 'L'),
              (settings.T, 'forestgreen', 'T'),
              (settings.H, 'darkorange', 'H')]

    ax = axes[0]
    for i in range(N):
        cls = classes[i]
        lw  = 0.8 if cls in (8, 9) else 1.1
        ls  = '--' if cls in (8, 9) else '-'
        ax.plot(sol.t, V_sol[i], color=CLASS_COLORS[cls], linewidth=lw,
                linestyle=ls, alpha=0.85,
                label=CLASS_NAMES[cls] if i == np.where(classes == cls)[0][0] else '')
    for v, color, lbl in hlines:
        ax.axhline(v, color=color, linestyle=':', linewidth=0.7, alpha=0.5)
    ax.set_ylabel('V (mV)')
    ax.legend(fontsize=7, loc='upper right', ncol=4)

    # w for one representative cell of each non-muscle class (1-7)
    ax = axes[1]
    for cls in range(1, 8):
        i = np.where(classes == cls)[0][0]
        ax.plot(sol.t, w_sol[i], color=CLASS_COLORS[cls], linewidth=1.0,
                label=CLASS_NAMES[cls])
    ax.set_ylabel('w (pA)')
    ax.legend(fontsize=7, ncol=4)

    # Muscle output: V of the dorsal (class 8) and ventral (class 9) muscle cells
    ax = axes[2]
    for cls in (8, 9):
        cells = np.where(classes == cls)[0]
        for j, i in enumerate(cells):
            ax.plot(sol.t, V_sol[i], color=CLASS_COLORS[cls], linewidth=1.0,
                    alpha=0.85, label=CLASS_NAMES[cls] if j == 0 else '')
    for v, color, lbl in hlines:
        ax.axhline(v, color=color, linestyle=':', linewidth=0.7, alpha=0.5)
    ax.set_ylabel('muscle V (mV)')
    ax.set_xlabel('Time (ms)')
    ax.set_title('Muscle output')
    ax.legend(fontsize=7, loc='upper right')

    fig.tight_layout()

    if save:
        fname = os.path.join(
            MEDIA_DIR,
            f'test_single_segment_IAVA{IAVA}_IAVB{IAVB}_Gsyne{settings.G_syne}_Gsyni{settings.G_syni}_yuval.png'
        )
        plt.savefig(fname, dpi=150)
        print(f'Saved → {fname}')

    plt.show()


if __name__ == '__main__':
    run(IAVA=0.0, IAVB=2.5, save=False)
