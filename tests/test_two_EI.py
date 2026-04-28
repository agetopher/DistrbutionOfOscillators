import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import settings
from functions import f_vec, w_inf_vec

"""
Two-neuron excitatory-inhibitory circuit — no synaptic fatigue.
  - Excitatory synapse : N0 --> N1  (G_syne)
  - Inhibitory synapse : N1 --> N0  (G_syni)

Constant Iapp injected into N0. N1 receives no direct input.

State vector: [V0, V1, w0, w1]
"""

# Membrane parameters (Yuval model)
settings.C   = 7.0
settings.g   = 1.0
settings.L   = -70.0
settings.T   = -45.0
settings.H   = -35.0
settings.m1  = 0.7
settings.m2  = 1 / 81.0
settings.m3  = -1 / 30.0
settings.m4  = 0.17
settings.numCells = 2

# Synaptic parameters (Yuval)
settings.G_syne = 0.063
settings.E_syne = 0.0
settings.G_syni = 0.05
settings.E_syni = -100.0
settings.k_syn  = 0.25
settings.V_th   = -52.0

# Recovery variable
settings.beta = 0.2
tau_w         = 200.0

MEDIA_DIR = os.path.join(os.path.dirname(__file__), '..', 'media')


def run(Iapp=3.5, save=True):
    G_syne = settings.G_syne
    G_syni = settings.G_syni
    E_syne = settings.E_syne
    E_syni = settings.E_syni
    k_syn  = settings.k_syn
    V_th   = settings.V_th

    def ode(t, y):
        V0, V1, w0, w1 = y

        fv   = f_vec(np.array([V0, V1]))
        winf = w_inf_vec(np.array([V0, V1]))

        s0 = 1.0 / (1.0 + np.exp(-k_syn * (V0 - V_th)))
        s1 = 1.0 / (1.0 + np.exp(-k_syn * (V1 - V_th)))

        I_inh0 = G_syni * s1 * (V0 - E_syni)
        I_exc1 = G_syne * s0 * (V1 - E_syne)

        dV0 = (settings.g * fv[0] - w0 - I_inh0 + Iapp) / settings.C
        dV1 = (settings.g * fv[1] - w1 - I_exc1) / settings.C
        dw0 = (winf[0] - w0) / tau_w
        dw1 = (winf[1] - w1) / tau_w

        return [dV0, dV1, dw0, dw1]

    tf = 5000
    t  = np.linspace(0, tf, int(tf))

    inits = [settings.T, settings.L - 0.5, 0.0, 0.0]
    sol = solve_ivp(ode, [0.0, tf], inits, method='BDF', t_eval=t)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle(
        f"Exc-Inh Circuit — No Synaptic Fatigue  [Yuval]\n"
        f"Iapp={Iapp} pA  |  G_syne={G_syne}, G_syni={G_syni} nS  |  "
        f"beta={settings.beta} nS/mV, tau_w={tau_w} ms",
        fontsize=11
    )

    hlines = [
        (settings.L, 'steelblue',   'L'),
        (settings.T, 'forestgreen', 'T'),
        (settings.H, 'darkorange',  'H'),
    ]

    ax = axes[0]
    ax.plot(sol.t, sol.y[0], color='blue', linewidth=1.0, label='N0 (exc)')
    ax.plot(sol.t, sol.y[1], color='red',  linewidth=1.0, label='N1 (inh)')
    for v, color, lbl in hlines:
        ax.axhline(v, color=color, linestyle=':', linewidth=0.8, label=lbl)
    ax.set_ylabel('V (mV)')
    ax.legend(fontsize=7, loc='upper right')

    ax = axes[1]
    ax.plot(sol.t, sol.y[2], color='blue', linewidth=1.0, label='w0')
    ax.plot(sol.t, sol.y[3], color='red',  linewidth=1.0, label='w1')
    ax.set_ylabel('w (pA)')
    ax.set_xlabel('Time (ms)')
    ax.legend(fontsize=7)

    fig.tight_layout()

    if save:
        fname = os.path.join(
            MEDIA_DIR,
            f'test_two_EI_no_sf_Iapp{Iapp}_Gsyne{G_syne}_Gsyni{G_syni}_yuval.png'
        )
        plt.savefig(fname, dpi=150)
        print(f'Saved → {fname}')

    plt.show()


if __name__ == '__main__':
    for Iapp in [2, 3, 4, 5]:
        run(Iapp=Iapp)
