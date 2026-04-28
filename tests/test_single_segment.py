import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import settings
from functions import f_vec, sf_vec, w_inf_vec

"""
Single-segment test: first 17 cells of the 102-cell network,
driven by 2 external 'head' inhibitory neurons forming EI loops
with the 2 class-1 oscillator cells.

Segment layout (17 cells, classes 1-9):
  Class 1 : cells 0-1   (2)  interneuron  ← oscillators, receive Iapp
  Class 2 : cell  2     (1)  interneuron
  Class 3 : cell  3     (1)  interneuron
  Class 4 : cell  4     (1)  interneuron
  Class 5 : cells 5-6   (2)  motor neuron
  Class 6 : cells 7-8   (2)  motor neuron
  Class 7 : cells 9-10  (2)  motor neuron
  Class 8 : cells 11-13 (3)  dorsal muscle   (passive)
  Class 9 : cells 14-16 (3)  ventral muscle  (passive)

Head neurons (cells 17-18): external inhibitory, no direct input.
  Cell 17 ↔ Cell 0 (class 1)  EI loop
  Cell 18 ↔ Cell 1 (class 1)  EI loop

Head neurons have no SF. State: [V (19), SF (19), w (19)]   SF[17:19] = 0 always.
"""

# ── Data ─────────────────────────────────────────────────────────────────────
DATA_DIR  = os.path.join(os.path.dirname(__file__), '..', 'data')
MEDIA_DIR = os.path.join(os.path.dirname(__file__), '..', 'media')
SEG       = 17

E_conn  = np.loadtxt(os.path.join(DATA_DIR, 'ConnectivityMatrix_SixSegments_ExcitatorySynapses.txt'), delimiter=',')[:SEG, :SEG]
I_conn  = np.loadtxt(os.path.join(DATA_DIR, 'ConnectivityMatrix_SixSegments_InhibitorySynapses.txt'), delimiter=',')[:SEG, :SEG]
GJ_conn = np.loadtxt(os.path.join(DATA_DIR, 'ConnectivityMatrix_SixSegments_GapJunctions.txt'),       delimiter=',')[:SEG, :SEG]
V_init  = np.loadtxt(os.path.join(DATA_DIR, 'InitialVoltages.dat'))[:SEG]
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
settings.G_syne = 0.065
settings.E_syne = 0.0
settings.G_syni = 0.05
settings.E_syni = -100.0
settings.k_syn  = 0.25
settings.V_th   = -52.0

settings.a = 0.000035
settings.b = 0.005

settings.G_gap = 0.01

# Recovery variable
settings.beta = 0.2
tau_w         = 200.0

# ── Circuit indices ───────────────────────────────────────────────────────────
N_HEAD   = 2
N        = SEG + N_HEAD          # 19 total cells
osc_idx  = np.where(classes == 1)[0]   # [0, 1] — class-1 oscillators
head_idx = np.arange(SEG, N)           # [17, 18] — head neurons

settings.numCells = N

# ── Class colours for plotting ────────────────────────────────────────────────
CLASS_COLORS = {
    1: 'royalblue', 2: 'darkorange', 3: 'forestgreen',
    4: 'crimson',   5: 'purple',     6: 'saddlebrown',
    7: 'teal',      8: 'gray',       9: 'lightcoral',
    'head': 'black',
}


def run(Iapp=3.0, tf=5000, save=True):

    # Iapp vector: drive only the class-1 oscillator cells
    Iapp_vec        = np.zeros(N)
    Iapp_vec[osc_idx] = Iapp

    def ode(t, y):
        V  = y[:N]
        SF = y[N:2*N]
        w  = y[2*N:]

        fv   = f_vec(V)
        winf = w_inf_vec(V)

        s = 1.0 / (1.0 + np.exp(-settings.k_syn * (V - settings.V_th)))

        # ── Segment internal dynamics (SF-gated) ──────────────────────────────
        pre_seg = s[:SEG] * SF[:SEG]

        I_exc_seg = settings.G_syne * (E_conn.T  @ pre_seg) * (V[:SEG] - settings.E_syne)
        I_inh_seg = settings.G_syni * (I_conn.T  @ pre_seg) * (V[:SEG] - settings.E_syni)
        I_gap_seg = settings.G_gap  * (GJ_conn.sum(axis=1) * V[:SEG] - GJ_conn @ V[:SEG])

        # ── EI loop: head → oscillator (no SF) ───────────────────────────────
        I_inh_head = np.zeros(SEG)
        for k in range(N_HEAD):
            I_inh_head[osc_idx[k]] = (settings.G_syni
                                      * s[head_idx[k]]
                                      * (V[osc_idx[k]] - settings.E_syni))

        # ── EI loop: oscillator → head (no SF) ───────────────────────────────
        I_exc_head = np.zeros(N_HEAD)
        for k in range(N_HEAD):
            I_exc_head[k] = (settings.G_syne
                             * s[osc_idx[k]]
                             * (V[head_idx[k]] - settings.E_syne))

        # ── dV ────────────────────────────────────────────────────────────────
        dV_seg  = (settings.g * fv[:SEG] - w[:SEG]
                   - I_exc_seg - I_inh_seg - I_gap_seg
                   - I_inh_head
                   + Iapp_vec[:SEG]) / settings.C

        dV_head = (settings.g * fv[SEG:] - w[SEG:] - I_exc_head) / settings.C

        # ── SF (segment only; head SF stays 0) ───────────────────────────────
        sf_dot = np.zeros(N)
        sf_dot[:SEG] = sf_vec(V[:SEG], SF[:SEG])

        dw = (winf - w) / tau_w

        z = np.empty(3 * N)
        z[:SEG]      = dV_seg
        z[SEG:N]     = dV_head
        z[N:2*N]     = sf_dot
        z[2*N:]      = dw
        return z

    t = np.linspace(0, tf, int(tf))

    V_init_clipped = np.where(V_init > settings.H, settings.T, V_init)
    V_init_head    = np.full(N_HEAD, settings.L - 0.5)

    SF_init = np.zeros(N)
    SF_init[:SEG] = 0.5
    w_init = np.zeros(N)

    inits = np.concatenate([V_init_clipped, V_init_head, SF_init, w_init])

    print(f"Running single-segment + {N_HEAD} head neurons  ({N} cells total)  Iapp={Iapp} pA...")
    sol = solve_ivp(ode, [0.0, tf], inits, method='BDF', t_eval=t)
    print("Done.")

    V_sol = sol.y[:N, :]
    w_sol = sol.y[2*N:, :]

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(
        f"Single Segment + Head EI Loop  Iapp={Iapp} pA on class 1\n"
        f"G_syne={settings.G_syne}, G_syni={settings.G_syni}, G_gap={settings.G_gap} nS  |  "
        f"beta={settings.beta}, tau_w={tau_w} ms",
        fontsize=11
    )

    hlines = [(settings.L, 'steelblue', 'L'),
              (settings.T, 'forestgreen', 'T'),
              (settings.H, 'darkorange', 'H')]

    ax = axes[0]
    for i in range(SEG):
        cls = classes[i]
        lw  = 0.8 if cls in (8, 9) else 1.1
        ls  = '--' if cls in (8, 9) else '-'
        ax.plot(sol.t, V_sol[i], color=CLASS_COLORS[cls], linewidth=lw,
                linestyle=ls, alpha=0.85,
                label=f'class {cls}' if i == np.where(classes == cls)[0][0] else '')
    for k in range(N_HEAD):
        ax.plot(sol.t, V_sol[head_idx[k]], color=CLASS_COLORS['head'],
                linewidth=1.1, linestyle=':', alpha=0.9,
                label='head' if k == 0 else '')
    for v, color, lbl in hlines:
        ax.axhline(v, color=color, linestyle=':', linewidth=0.7, alpha=0.5)
    ax.set_ylabel('V (mV)')
    ax.legend(fontsize=7, loc='upper right', ncol=4)

    ax = axes[1]
    for i in osc_idx:
        cls = classes[i]
        ax.plot(sol.t, w_sol[i], color=CLASS_COLORS[cls], linewidth=1.0,
                label=f'osc cell {i}')
    for k in range(N_HEAD):
        ax.plot(sol.t, w_sol[head_idx[k]], color=CLASS_COLORS['head'],
                linewidth=1.0, linestyle=':', label=f'head {k}')
    ax.set_ylabel('w (pA)')
    ax.set_xlabel('Time (ms)')
    ax.legend(fontsize=7)

    fig.tight_layout()

    if save:
        fname = os.path.join(
            MEDIA_DIR,
            f'test_single_segment_EI_Iapp{Iapp}_Gsyne{settings.G_syne}_Gsyni{settings.G_syni}_yuval.png'
        )
        plt.savefig(fname, dpi=150)
        print(f'Saved → {fname}')

    plt.show()


if __name__ == '__main__':
    run(Iapp=3.0, save=True)
