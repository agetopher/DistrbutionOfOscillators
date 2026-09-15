import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import settings
import segment

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

MEDIA_DIR = os.path.join(os.path.dirname(__file__), '..', 'media')

settings.reset_defaults()
segment.configure()

N = segment.N_CELLS
classes = segment.CLASSES

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

    I_inj = segment.drive_vector(IAVA, IAVB)
    t = np.linspace(0, tf, int(tf))
    inits = segment.full_rest_state()

    print(f"Running single segment ({N} cells)  IAVA={IAVA} pA, IAVB={IAVB} pA...")
    sol = solve_ivp(
        lambda time, state: segment.rhs_full(time, state, I_inj),
        [0.0, tf],
        inits,
        method='BDF',
        t_eval=t,
    )
    print("Done.")

    V_sol = sol.y[:N, :]
    w_sol = sol.y[2*N:, :]

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    fig.suptitle(
        f"Single Segment  IAVA={IAVA} pA (AS,DA,VA), IAVB={IAVB} pA (AS,DB,VB)\n"
        f"G_syne={settings.G_syne}, G_syni={settings.G_syni}, G_gap={settings.G_gap} nS  |  "
        f"beta={settings.beta}, k_syn={settings.k_syn} ms",
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
    run(IAVA=0.0, IAVB=0.0, save=False)
