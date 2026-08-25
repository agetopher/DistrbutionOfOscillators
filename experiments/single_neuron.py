import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import settings
from functions import f

"""
Single-neuron test with slow recovery variable w.

Protocol
--------
  [0,       t_on)  : I_app = 0  — neuron at rest
  [t_on,  t_off]   : I_app > 0  — drives neuron above threshold to plateau
  (t_off,    tf]   : I_app = 0  — neuron should return to rest via w

Without w (left column): neuron locks onto the plateau permanently.
With    w (right column): w accumulates during the plateau and pulls
the neuron back to rest once I_app is removed.
"""

settings.reset_defaults()

settings.tau_w = 500.0

MEDIA_DIR = os.path.join(os.path.dirname(__file__), '..', 'media')


def w_inf(V):
    """Scalar piecewise-linear recovery target."""
    return settings.beta * max(V - settings.T, 0.0)


# ── ODEs ──────────────────────────────────────────────────────────────────────
offset = 0.0
def ode_no_recovery(t, y, Iapp, t_on, t_off):
    """Original single-neuron ODE — no recovery variable."""
    V = y[0]
    Iapp_now = Iapp if t_on <= t <= t_off else 0.0
    dV = (settings.g * (f(V) - offset) + Iapp_now) / settings.C
    return [dV]


def ode_with_recovery(t, y, Iapp, t_on, t_off):
    """Extended ODE with slow recovery variable w."""
    V, w = y
    Iapp_now = Iapp if t_on <= t <= t_off else 0.0
    dV = (settings.g * f(V) - w + Iapp_now) / settings.C
    dw = (w_inf(V) - w) / settings.tau_w
    return [dV, dw]


# ── Main ──────────────────────────────────────────────────────────────────────

def run(Iapp=3.5, t_on=200.0, t_off=300.0, tf=2000.0, save=False):
    t_eval = np.linspace(0, tf, int(tf))

    # Without recovery: state = [V]
    sol_base = solve_ivp(
        lambda t, y: ode_no_recovery(t, y, Iapp, t_on, t_off),
        [0.0, tf], [settings.L],
        method='BDF', t_eval=t_eval, max_step=5.0
    )

    # With recovery: state = [V, w]
    sol_rec = solve_ivp(
        lambda t, y: ode_with_recovery(t, y, Iapp, t_on, t_off),
        [0.0, tf], [settings.L, 0.0],
        method='BDF', t_eval=t_eval, max_step=5.0
    )

    Iapp_trace = np.where(
        (sol_rec.t >= t_on) & (sol_rec.t <= t_off), Iapp, 0.0
    )

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
    fig.suptitle(
        f"Single Neuron: Base with offset {offset} vs Recovery Variable\n"
        f"I_app = {Iapp} pA  applied [{t_on}–{t_off} ms],  "
        f"tau_w = {settings.tau_w} ms,  beta = {settings.beta} nS/mV",
        fontsize=12
    )

    stim_kw = dict(alpha=0.12, color='red')
    hline_kw = [
        (settings.L, 'steelblue',   'L (rest)'),
        (settings.T, 'forestgreen', 'T (threshold)'),
        (settings.H, 'darkorange',  'H (plateau)'),
    ]

    for col, (sol, label) in enumerate([
        (sol_base, 'Without recovery variable'),
        (sol_rec,  f'With recovery variable  (w)'),
    ]):
        V = sol.y[0]

        # Row 0: voltage
        ax = axes[0, col]
        ax.plot(sol.t, V, color='black', linewidth=1.2)
        for vline, color, lbl in hline_kw:
            ax.axhline(vline, color=color, linestyle=':', linewidth=1.0, label=lbl)
        ax.axvspan(t_on, t_off, **stim_kw)
        ax.set_ylabel('V (mV)')
        ax.set_title(label)
        ax.set_ylim(settings.L - 5, settings.H + 20)
        ax.legend(fontsize=7, loc='upper right')

        # Row 1: recovery variable (zero for baseline)
        ax = axes[1, col]
        if col == 1:
            ax.plot(sol.t, sol.y[1], color='tomato', linewidth=1.2, label='w')
            # overlay w_inf(V) for reference
            w_inf_trace = np.array([w_inf(v) for v in V])
            ax.plot(sol.t, w_inf_trace, color='tomato', linewidth=0.8,
                    linestyle='--', alpha=0.6, label='w_inf(V)')
            ax.legend(fontsize=7)
        else:
            ax.axhline(0, color='tomato', linewidth=1.2, linestyle='--', label='w = 0 (absent)')
            ax.legend(fontsize=7)
        ax.axvspan(t_on, t_off, **stim_kw)
        ax.set_ylabel('w  (nS)')

        # Row 2: applied current
        ax = axes[2, col]
        ax.plot(sol.t, Iapp_trace, color='steelblue', linewidth=1.2)
        ax.axvspan(t_on, t_off, **stim_kw)
        ax.set_ylabel('I_app (pA)')
        ax.set_xlabel('Time (ms)')
        ax.set_ylim(-0.2, Iapp + 0.5)

    fig.tight_layout()

    if save:
        fname = os.path.join(MEDIA_DIR, f'test_single_neuron_offset{offset}_beta{settings.beta}_Iapp{Iapp}_tr{t_off-t_on}.png')
        plt.savefig(fname, dpi=150)
        print(f'Saved → {fname}')

    plt.show()


if __name__ == '__main__':
    run(save=True)
