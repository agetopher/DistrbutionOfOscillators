import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import settings
import network as NW
from functions import f_vec, w_inf_vec

'''
Command-dependent travelling-wave direction on the full 102-cell network.

Under a STEADY command, the network settles into a robust anterior↔posterior
travelling wave whose DIRECTION depends on the command:
    AVA  →  A→P wave (posterior lags,  seg0→5 ≈ +180 ms)
    AVB  →  P→A wave (posterior leads, seg0→5 ≈ -140 ms)
Both profiles are essentially identical across random-IC seeds, so the direction
is set by the circuit, not the seed. The wave is real but weak (~0.2-0.25 of a
body wavelength), which is why it is hard to read by eye in a raw voltage
kymogram — this figure overlays the extracted wavefronts to make it legible.

Mechanism (see the exploration notes): the wave is carried by the inter-segment
GAP junctions, which are heterotypic (AS↔VA, DB↔VB couple different cell types in
adjacent segments) and therefore impose a directional phase offset despite being
electrically symmetric. The directional excitatory synapses onto DD do NOT carry
it (removing them barely changes the wave; alone they give only a weak, incoherent
backward drift). Uses the reduced (V, w) model (synaptic fatigue is inert).
'''

N       = settings.numCells
classes = NW.classes
seg     = np.arange(N) // 17
E0, I0, G0 = settings.E_conn.copy(), settings.I_conn.copy(), settings.GJ_conn.copy()
dm      = np.where(classes == 8)[0]
DT      = 20.0
MEDIA_DIR = os.path.join(os.path.dirname(__file__), '..', 'media')


def rhs(x, I):
    V = x[:N]
    w = x[N:]
    s = 1.0 / (1.0 + np.exp(-settings.k_syn * (V - settings.V_th)))
    I_exc = settings.G_syne * (E0 @ s) * (V - settings.E_syne)
    I_inh = settings.G_syni * (I0 @ s) * (V - settings.E_syni)
    Vdiff = V[:, None] - V[None, :]
    I_gap = settings.G_gap * (G0 * Vdiff).sum(axis=1)
    dV = (settings.g * f_vec(V) - w - I_exc - I_inh - I_gap + I) / settings.C
    dw = (w_inf_vec(V) - w) / settings.tau_w
    return np.concatenate([dV, dw])


def _drive(cmd, D):
    I = np.zeros(N)
    I[np.isin(classes, NW.AVA_CLASSES if cmd == 'AVA' else NW.AVB_CLASSES)] += D
    return I


def simulate(cmd, seed, D=2.5, tf=18000.0):
    rng = np.random.default_rng(seed)
    x0 = np.concatenate([np.full(N, -70.0) + rng.standard_normal(N) * 2.0, np.zeros(N)])
    t = np.arange(0.0, tf, DT)
    sol = solve_ivp(lambda tt, xx: rhs(xx, _drive(cmd, D)), [0.0, tf], x0,
                    method='BDF', t_eval=t)
    return t, sol.y[:N]


def segment_traces(t, V):
    """Per-segment mean dorsal-muscle trace over the steady-state window."""
    keep = t >= 0.5 * t[-1]
    st = np.array([V[dm[seg[dm] == k]].mean(0)[keep] for k in range(6)])
    return t[keep], st - st.mean(axis=1, keepdims=True)


def phase_profile(st):
    """Per-segment phase lag vs segment 0 (ms), period-aware; also returns period."""
    pk, _ = find_peaks(st[0], prominence=8.0)
    P = np.median(np.diff(pk)) * DT
    maxlag = int(P / 2 / DT)
    ref = st[0]
    c0 = len(ref) - 1
    lags = [(int(np.argmax(np.correlate(r, ref, 'full')[c0 - maxlag: c0 + maxlag + 1]))
             - maxlag) * DT for r in st]
    return np.array(lags, float), P


def wavefronts(tt, st, P):
    """Trace successive wavefronts: for each seg-0 peak, follow the nearest peak
    (within ±0.6 period) across segments. Returns a list of 6-element time lists."""
    out = []
    for p in find_peaks(st[0], prominence=8.0)[0]:
        front = [tt[p]]
        ok = True
        for k in range(1, 6):
            cand = tt[find_peaks(st[k], prominence=8.0)[0]]
            if cand.size == 0:
                ok = False
                break
            j = int(np.argmin(np.abs(cand - front[-1])))
            if abs(cand[j] - front[-1]) > 0.6 * P:
                ok = False
                break
            front.append(cand[j])
        if ok:
            out.append(front)
    return out


def run(n_seed=3, save=True):
    prof, first = {}, {}
    for cmd in ('AVA', 'AVB'):
        profs = []
        for sd in range(n_seed):
            t, V = simulate(cmd, sd)
            tt, st = segment_traces(t, V)
            p, P = phase_profile(st)
            profs.append(p)
            if sd == 0:
                first[cmd] = (tt, st, P)
        prof[cmd] = np.array(profs)
        m = prof[cmd].mean(0)
        print(f"  {cmd}: seg0->5 = {m[-1]:+.0f} ms  "
              f"({'A→P' if m[-1] > 0 else 'P→A'} wave)  profile={np.round(m).astype(int)}")

    fig = plt.figure(figsize=(15, 7))
    gs = GridSpec(2, 2, width_ratios=[1.0, 1.15], hspace=0.32, wspace=0.2)
    cc = {'AVA': 'crimson', 'AVB': 'steelblue'}

    axp = fig.add_subplot(gs[:, 0])
    for cmd in ('AVA', 'AVB'):
        m = prof[cmd].mean(0)
        axp.fill_between(range(6), prof[cmd].min(0), prof[cmd].max(0),
                         color=cc[cmd], alpha=0.15)
        axp.plot(range(6), m, 'o-', color=cc[cmd], lw=2.4, ms=7,
                 label=f"{cmd}: {'A→P' if m[-1] > 0 else 'P→A'} wave "
                       f"(seg0→5 {m[-1]:+.0f} ms)")
    axp.axhline(0, color='0.5', lw=0.8)
    axp.set_xlabel("segment  (0 = anterior → 5 = posterior)")
    axp.set_ylabel("phase lag vs segment 0  (ms)")
    axp.set_title("The command reverses the travelling-wave direction\n"
                  "(steady drive; profiles ≈ identical across seeds)", fontsize=11)
    axp.legend(fontsize=10, loc='center left')

    for row, cmd in enumerate(('AVA', 'AVB')):
        tt, st, P = first[cmd]
        ts = tt / 1000.0
        w0 = ts[0] + 6.0
        w1 = w0 + 3.5
        m = (ts >= w0) & (ts <= w1)
        ax = fig.add_subplot(gs[row, 1])
        ax.imshow(st[:, m], aspect='auto', origin='upper', extent=[w0, w1, 6, 0],
                  cmap='viridis')
        for f in wavefronts(tt, st, P):
            f = np.array(f) / 1000.0
            if f.min() >= w0 and f.max() <= w1:
                ax.plot(f, np.arange(6) + 0.5, '-', color='w', lw=1.4, alpha=0.9)
        ax.set_yticks(np.arange(6) + 0.5)
        ax.set_yticklabels([f"seg{k}" for k in range(6)], fontsize=8)
        ax.set_ylabel(cmd, fontsize=11)
        ax.set_title(f"{cmd}: dorsal-muscle wavefronts (white-line tilt = wave direction)",
                     fontsize=9)
        ax.set_xlabel("time (s)") if row == 1 else ax.set_xticklabels([])

    fig.suptitle("Command-dependent wave-direction reversal — full 102-cell network, "
                 f"steady 2.5 pA, β={settings.beta}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    if save:
        fname = os.path.join(MEDIA_DIR, 'wave_direction_reversal.png')
        plt.savefig(fname, dpi=140)
        print(f"Saved → {fname}")
    plt.show()


if __name__ == '__main__':
    print("Command-dependent wave direction (steady AVA vs AVB, per-segment phase lag):")
    run()
