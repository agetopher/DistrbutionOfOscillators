import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import settings
import network as NW                      # sets 102-cell params + loads full matrices (run() is guarded)
from functions import f_vec, w_inf_vec

'''
Command-interneuron alternation on the FULL 102-cell network.

Drives the network with alternating AVA / AVB command blocks and reads out the
dorsal and ventral muscle kymograms (body position anterior→posterior on y,
time on x). The question is whether AVA vs AVB reverses the direction of the
anterior↔posterior travelling wave (backward vs forward locomotion).

Integrates the reduced (V, w) system: synaptic fatigue is inert in the current
model (network.ode uses pre = s(V), the fatigue variable feeds back nowhere), so
V and w are the only dynamical variables. This matches network.ode exactly while
being ~40x faster and avoiding the stiff-solver trouble the full 3N state hits on
long runs. (If synaptic fatigue is ever re-enabled, revisit this.)

Drive switches are handled by integrating block-by-block on a fixed global time
grid, carrying the (V, w) state across each boundary.
'''

settings.tau_w = 200.0                     # network.py keeps tau_w module-local; set it here
N       = settings.numCells
classes = NW.classes
Ec, Ic, Gj = settings.E_conn, settings.I_conn, settings.GJ_conn
MEDIA_DIR = os.path.join(os.path.dirname(__file__), '..', 'media')

DEFAULT_PROTOCOL = [('AVA', 8000.0), ('AVB', 8000.0), ('AVA', 8000.0), ('AVB', 8000.0)]


def rhs(x, I):
    """Reduced (V, w) right-hand side for the full network (mirrors network.ode
    with the inert synaptic-fatigue variable dropped)."""
    V = x[:N]
    w = x[N:]
    s = 1.0 / (1.0 + np.exp(-settings.k_syn * (V - settings.V_th)))
    I_exc = settings.G_syne * (Ec @ s) * (V - settings.E_syne)
    I_inh = settings.G_syni * (Ic @ s) * (V - settings.E_syni)
    Vdiff = V[:, None] - V[None, :]
    I_gap = settings.G_gap * (Gj * Vdiff).sum(axis=1)
    dV = (settings.g * f_vec(V) - w - I_exc - I_inh - I_gap + I) / settings.C
    dw = (w_inf_vec(V) - w) / settings.tau_w
    return np.concatenate([dV, dw])


def drive_of(phase, D):
    I = np.zeros(N)
    if phase in ('AVA', 'BOTH'):
        I[np.isin(classes, NW.AVA_CLASSES)] += D
    if phase in ('AVB', 'BOTH'):
        I[np.isin(classes, NW.AVB_CLASSES)] += D
    return I


def simulate_protocol(protocol=DEFAULT_PROTOCOL, D=2.5, dt=20.0, x0=None):
    """Integrate the network through the drive sequence on a fixed time grid,
    carrying (V, w) across switches. Returns (t, V, IAVA, IBVB)."""
    total = sum(d for _, d in protocol)
    t = np.arange(0.0, total, dt)
    V = np.zeros((N, t.size))
    IA = np.zeros(t.size)
    IB = np.zeros(t.size)
    x = np.concatenate([np.full(N, -70.0), np.zeros(N)]) if x0 is None else np.array(x0, float)
    t0 = 0.0
    for phase, dur in protocol:
        s, e = t0, t0 + dur
        mask = (t >= s) & (t < e)
        sol = solve_ivp(lambda tt, xx: rhs(xx, drive_of(phase, D)), [s, e], x,
                        method='BDF', t_eval=t[mask])
        assert sol.success and sol.t.size == mask.sum(), (phase, sol.message)
        V[:, mask] = sol.y[:N]
        x = sol.y[:, -1]
        IA[mask] = D if phase in ('AVA', 'BOTH') else 0.0
        IB[mask] = D if phase in ('AVB', 'BOTH') else 0.0
        t0 = e
    return t, V, IA, IB


def _period_ms(trace, dt, prominence=8.0):
    pk, _ = find_peaks(trace, prominence=prominence)
    return float(np.median(np.diff(pk)) * dt) if len(pk) >= 3 else np.nan


def wave_gradient(t, V, s_ms, e_ms, dt, skip_ms=2000.0):
    """Anterior→posterior phase lag of the dorsal-muscle rows over one block.

    For each body row, the lag (ms) vs the anterior-most row is the peak of the
    cross-correlation, searched only within ±half a period so it can't alias.
    Returns (lags, slope) where slope>0 ⇒ posterior lags anterior (A→P wave),
    slope<0 ⇒ P→A wave. slope is nan if the block does not oscillate cleanly."""
    dor = np.where(classes == 8)[0]
    m = (t >= s_ms + skip_ms) & (t < e_ms)
    seg = V[dor][:, m]
    seg = seg - seg.mean(axis=1, keepdims=True)
    P = _period_ms(seg[0], dt)
    if not np.isfinite(P):
        return np.full(len(dor), np.nan), np.nan
    maxlag = int(P / 2 / dt)
    ref = seg[0]
    center = len(ref) - 1
    lags = []
    for r in seg:
        c = np.correlate(r, ref, mode='full')
        win = c[center - maxlag: center + maxlag + 1]
        lags.append((int(np.argmax(win)) - maxlag) * dt)
    lags = np.array(lags, float)
    slope = np.polyfit(np.arange(len(lags)), lags, 1)[0]   # ms per body row
    return lags, slope


def plot(t, V, IA, IB, protocol=DEFAULT_PROTOCOL, D=2.5, save=True):
    ts = t / 1000.0
    bounds = np.cumsum([d for _, d in protocol]) / 1000.0
    dor = np.where(classes == 8)[0]
    ven = np.where(classes == 9)[0]

    fig, ax = plt.subplots(3, 1, figsize=(13, 9), sharex=True,
                           height_ratios=[0.5, 1, 1])
    ax[0].plot(ts, IA, color='crimson',   lw=1.6, label='IAVA')
    ax[0].plot(ts, IB, color='steelblue', lw=1.6, label='IAVB')
    ax[0].legend(fontsize=8, ncol=2, loc='center left')
    ax[0].set_ylabel('drive (pA)')
    ax[0].set_ylim(0, D * 1.25)
    ax[0].set_title(f"Full network alternating AVA/AVB  (N={N}, D={D} pA, "
                    f"β={settings.beta}, τ_w={settings.tau_w} ms)")

    for a, idx, nm in ((ax[1], dor, 'Dorsal'), (ax[2], ven, 'Ventral')):
        im = a.imshow(V[idx], aspect='auto', origin='upper',
                      extent=[ts[0], ts[-1], len(idx), 0], cmap='viridis',
                      vmin=settings.L, vmax=settings.H)
        a.set_ylabel(f'{nm} muscle\n(ant→post)')
        fig.colorbar(im, ax=a, label='mV', pad=0.01)
    ax[2].set_xlabel('time (s)')
    for a in ax:
        for b in bounds[:-1]:
            a.axvline(b, color='w', ls=':', lw=0.9)

    fig.tight_layout()
    if save:
        fname = os.path.join(MEDIA_DIR, 'network_alternating_IAVA_IAVB.png')
        plt.savefig(fname, dpi=140)
        print(f"Saved → {fname}")
    plt.show()


def run(protocol=DEFAULT_PROTOCOL, D=2.5, dt=20.0, save=True):
    t, V, IA, IB = simulate_protocol(protocol, D=D, dt=dt)

    bounds = np.cumsum([d for _, d in protocol])
    starts = np.concatenate([[0.0], bounds[:-1]])
    print(f"Full-network command alternation  (D={D} pA, β={settings.beta}):")
    for (phase, _), s, e in zip(protocol, starts, bounds):
        _, slope = wave_gradient(t, V, s, e, dt)
        if np.isnan(slope):
            direction = "no clean oscillation"
        elif abs(slope) < 1.0:
            direction = "≈synchronous (no clear wave)"
        else:
            direction = "A→P wave" if slope > 0 else "P→A wave"
        print(f"  {s/1000:5.0f}-{e/1000:2.0f}s  {phase:>4}  "
              f"dorsal phase gradient = {slope:+6.1f} ms/row  → {direction}")

    plot(t, V, IA, IB, protocol=protocol, D=D, save=save)


if __name__ == '__main__':
    run()
