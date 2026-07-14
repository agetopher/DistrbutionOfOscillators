import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import settings
import bifurcations as B

'''
Command-interneuron alternation on a single segment.

Drives the segment with a time-varying command sequence — AVA and AVB activated
in alternating blocks, plus a co-activation ('BOTH') block — and shows:

  * while exactly one command is on the segment oscillates continuously, and it
    re-routes which ventral motor neuron carries the rhythm (VA under AVA,
    VB under AVB);
  * simultaneous AVA + AVB co-activation SILENCES the segment (it locks at a
    depolarised fixed point) — consistent with AVA/AVB being mutually exclusive.

Reuses the reduced (V, w) model from bifurcations.py (synaptic fatigue is inert,
so V and w are the only dynamical variables). Drive switches are handled by
integrating block-by-block and carrying the state across each boundary.
'''

N       = B.N
classes = B.classes
CLS     = B.CLASS_NAMES
MEDIA_DIR = os.path.join(os.path.dirname(__file__), '..', 'media')

# (phase, duration_ms). 'AVA'/'AVB' drive one command; 'BOTH' drives both.
DEFAULT_PROTOCOL = [('AVA', 6000.0), ('AVB', 6000.0), ('AVA', 6000.0),
                    ('BOTH', 6000.0), ('AVB', 6000.0), ('AVA', 6000.0)]


def drive_of(phase, D):
    """Per-cell injection vector for a protocol phase."""
    return {'AVA':  B._drive_vector(D, 0.0),
            'AVB':  B._drive_vector(0.0, D),
            'BOTH': B._drive_vector(D, D)}[phase]


def simulate_protocol(protocol=DEFAULT_PROTOCOL, D=2.5, dt=8.0, x0=None):
    """Integrate the segment through a sequence of drive blocks, carrying the
    (V, w) state across each switch. Returns (t, V, IAVA, IAVB) with V of shape
    (N, len(t)) and the two drive-amplitude time series."""
    x = B.rest_state() if x0 is None else np.array(x0, float)
    t0 = 0.0
    T, Y, IA, IB = [], [], [], []
    for phase, dur in protocol:
        I = drive_of(phase, D)
        t_eval = np.arange(t0, t0 + dur, dt)
        sol = solve_ivp(lambda s, xx: B.rhs_Vw(xx, I), [t0, t0 + dur], x,
                        method='BDF', t_eval=t_eval)
        x = sol.y[:, -1]
        t0 += dur
        T.append(sol.t)
        Y.append(sol.y[:N])
        IA.append(np.full(sol.t.size, D if phase in ('AVA', 'BOTH') else 0.0))
        IB.append(np.full(sol.t.size, D if phase in ('AVB', 'BOTH') else 0.0))
    return (np.concatenate(T), np.concatenate(Y, axis=1),
            np.concatenate(IA), np.concatenate(IB))


def plot(t, V, IA, IB, protocol=DEFAULT_PROTOCOL, D=2.5, save=True):
    ts = t / 1000.0
    bounds = np.cumsum([d for _, d in protocol]) / 1000.0
    starts = np.concatenate([[0.0], bounds[:-1]])
    names  = [p for p, _ in protocol]

    dors = np.where(classes == 8)[0]
    vent = np.where(classes == 9)[0]
    VA   = np.where(classes == 7)[0]   # ventral A motor neurons (AVA-driven)
    VB   = np.where(classes == 6)[0]   # ventral B motor neurons (AVB-driven)

    fig, ax = plt.subplots(4, 1, figsize=(13, 11), sharex=True,
                           height_ratios=[0.45, 1.5, 0.8, 0.8])

    ax[0].plot(ts, IA, color='crimson',   lw=1.7, label='IAVA (→AS,DA,VA)')
    ax[0].plot(ts, IB, color='steelblue', lw=1.7, label='IAVB (→AS,DB,VB)')
    ax[0].set_ylabel('drive (pA)')
    ax[0].legend(fontsize=8, ncol=2, loc='center left')
    ax[0].set_ylim(0, D * 1.25)
    ax[0].set_title(f"Alternating AVA/AVB with a co-activation block "
                    f"(single segment, D={D} pA, β={settings.beta}, "
                    f"τ_w={settings.tau_w} ms)")
    for nm, s, e in zip(names, starts, bounds):
        ax[0].text((s + e) / 2, D * 1.05, nm, ha='center', fontsize=8,
                   color=('purple' if nm == 'BOTH' else
                          'crimson' if nm == 'AVA' else 'steelblue'))

    im = ax[1].imshow(V, aspect='auto', origin='upper',
                      extent=[ts[0], ts[-1], N, 0], cmap='viridis',
                      vmin=settings.L, vmax=settings.H)
    ax[1].set_yticks(np.arange(N) + 0.5)
    ax[1].set_yticklabels([f"{i} {CLS[classes[i]]}" for i in range(N)], fontsize=6)
    ax[1].set_ylabel('cell')
    fig.colorbar(im, ax=ax[1], label='V (mV)', pad=0.01)

    ax[2].plot(ts, V[VA].mean(0), color='crimson',   lw=1.1,
               label='VA (ventral-A, AVA-driven)')
    ax[2].plot(ts, V[VB].mean(0), color='steelblue', lw=1.1,
               label='VB (ventral-B, AVB-driven)')
    ax[2].set_ylabel('MN V (mV)')
    ax[2].legend(fontsize=8, loc='center left')

    ax[3].plot(ts, V[dors].mean(0), color='saddlebrown', lw=1.1, label='dorsal muscle')
    ax[3].plot(ts, V[vent].mean(0), color='teal',        lw=1.1, label='ventral muscle')
    ax[3].set_ylabel('muscle V (mV)')
    ax[3].set_xlabel('time (s)')
    ax[3].legend(fontsize=8, loc='center left')

    for a in ax:
        for b in bounds[:-1]:
            a.axvline(b, color='0.5', ls=':', lw=0.7)

    fig.tight_layout()
    if save:
        fname = os.path.join(MEDIA_DIR, 'segment_alternating_IAVA_IAVB.png')
        plt.savefig(fname, dpi=140)
        print(f"Saved → {fname}")
    plt.show()


def run(protocol=DEFAULT_PROTOCOL, D=2.5, dt=8.0, save=True):
    t, V, IA, IB = simulate_protocol(protocol, D=D, dt=dt)

    # Report the swing in each block (a silenced block has ~0 swing).
    bounds = np.cumsum([d for _, d in protocol])
    starts = np.concatenate([[0.0], bounds[:-1]])
    print(f"Single-segment command alternation  (D={D} pA, β={settings.beta}):")
    for (phase, _), s, e in zip(protocol, starts, bounds):
        seg = V[:, (t >= s + 2000.0) & (t <= e)]   # skip 2 s switch transient
        swing = float((seg.max(1) - seg.min(1)).max()) if seg.size else np.nan
        print(f"  {s/1000:5.0f}-{e/1000:2.0f}s  {phase:>4}  "
              f"max cell swing = {swing:5.1f} mV  "
              f"{'(silenced)' if swing < 1.0 else '(oscillating)'}")

    plot(t, V, IA, IB, protocol=protocol, D=D, save=save)


if __name__ == '__main__':
    run()
