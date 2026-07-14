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
Does the anterior→posterior travelling wave depend on inter-segment coupling
strength?  Scales the inter-segment gap-junction weights and, under a steady
AVA command, measures the per-segment phase lag of the dorsal muscle.

Result: the full network already produces a robust, reproducible A→P wave at
baseline (the direction is intrinsic — identical across random initial-condition
seeds). WEAKENING the inter-segment coupling DEGRADES the wave (the phase profile
loses monotonicity and becomes seed-dependent), it does not create one. So the
coupling is what phase-locks the segments into a coherent wave; overall coupling
strength is the wrong knob for lengthening it (coupling asymmetry is the lever —
see the exploration notes). The baseline wave is real but weak: ~0.24 of a body
wavelength vs the worm's ~1 full wavelength.

The inter-segment wiring is symmetric (equal forward/backward edge weight in both
excitation and gap junctions) and only connects adjacent segments; inhibition is
purely intra-segment. Uses the reduced (V, w) model (synaptic fatigue is inert).
'''

settings.tau_w = 200.0
N       = settings.numCells
classes = NW.classes
SEG     = 17
seg     = np.arange(N) // SEG                       # segment index of each cell
E0, I0, G0 = settings.E_conn.copy(), settings.I_conn.copy(), settings.GJ_conn.copy()
inter   = seg[:, None] != seg[None, :]              # [post, pre] inter-segment edges
dm      = np.where(classes == 8)[0]                 # dorsal muscle cells (3 per segment)
MEDIA_DIR = os.path.join(os.path.dirname(__file__), '..', 'media')

SCALES = [1.0, 0.5, 0.25, 0.1]                      # inter-segment gap-junction multipliers
N_SEED = 5
DT     = 20.0                                       # ms, output sampling


def rhs(x, I, G):
    """Reduced (V, w) RHS with a supplied gap-junction matrix G."""
    V = x[:N]
    w = x[N:]
    s = 1.0 / (1.0 + np.exp(-settings.k_syn * (V - settings.V_th)))
    I_exc = settings.G_syne * (E0 @ s) * (V - settings.E_syne)
    I_inh = settings.G_syni * (I0 @ s) * (V - settings.E_syni)
    Vdiff = V[:, None] - V[None, :]
    I_gap = settings.G_gap * (G * Vdiff).sum(axis=1)
    dV = (settings.g * f_vec(V) - w - I_exc - I_inh - I_gap + I) / settings.C
    dw = (w_inf_vec(V) - w) / settings.tau_w
    return np.concatenate([dV, dw])


def _ava_drive(D):
    I = np.zeros(N)
    I[np.isin(classes, NW.AVA_CLASSES)] += D
    return I


def simulate(scale, seed, D=2.5, tf=20000.0):
    """Steady AVA drive with inter-segment gap junctions scaled by `scale`, from
    a small random (symmetry-breaking) perturbation of rest. Returns (t, V)."""
    G = G0.copy()
    G[inter] *= scale
    rng = np.random.default_rng(seed)
    x0 = np.concatenate([np.full(N, -70.0) + rng.standard_normal(N) * 2.0, np.zeros(N)])
    t = np.arange(0.0, tf, DT)
    sol = solve_ivp(lambda tt, xx: rhs(xx, _ava_drive(D), G), [0.0, tf], x0,
                    method='BDF', t_eval=t)
    return t, sol.y[:N]


def segment_phase(t, V):
    """Per-segment dorsal-muscle phase lag vs segment 0 (ms), period-aware
    (cross-correlation restricted to ±half a period). Returns (lags[6], period)."""
    keep = t >= 0.5 * t[-1]
    st = np.array([V[dm[seg[dm] == k]].mean(0)[keep] for k in range(6)])
    st = st - st.mean(axis=1, keepdims=True)
    pk, _ = find_peaks(st[0], prominence=8.0)
    if len(pk) < 3:
        return np.full(6, np.nan), np.nan
    P = np.median(np.diff(pk)) * DT
    maxlag = int(P / 2 / DT)
    ref = st[0]
    c0 = len(ref) - 1
    lags = [(int(np.argmax(np.correlate(r, ref, 'full')[c0 - maxlag: c0 + maxlag + 1]))
             - maxlag) * DT for r in st]
    return np.array(lags, float), P


def run(save=True):
    kym, lag, period = {}, {s: [] for s in SCALES}, None
    t = None
    for s in SCALES:
        for sd in range(N_SEED):
            t, V = simulate(s, sd)
            lags, P = segment_phase(t, V)
            lag[s].append(lags)
            if sd == 0:
                kym[s] = V[dm]
            if s == 1.0 and sd == 0:
                period = P
        prof = np.nanmean(np.array(lag[s]), 0)
        print(f"  gap x{s:<4}  seg0->5 lag = {prof[-1]:5.0f} ms "
              f"({prof[-1]/period:+.2f} cycle)  profile={np.round(prof).astype(int)}")
    lag = {s: np.array(v) for s, v in lag.items()}

    ts = t / 1000.0
    fig = plt.figure(figsize=(15.5, 9))
    gs = GridSpec(4, 3, width_ratios=[1.1, 0.045, 1.0], hspace=0.32, wspace=0.06)
    w0, w1 = 11.0, 15.0
    m = (ts >= w0) & (ts <= w1)
    im = None
    for i, s in enumerate(SCALES):
        ax = fig.add_subplot(gs[i, 0])
        im = ax.imshow(kym[s][:, m], aspect='auto', origin='upper',
                       extent=[w0, w1, 18, 0], cmap='viridis',
                       vmin=settings.L, vmax=settings.H)
        tot = np.nanmean(lag[s], 0)[-1]
        ax.set_ylabel(f"×{s}", rotation=0, ha='right', va='center', fontsize=13)
        ax.text(0.01, 0.05, f"seg0→5 lag ≈ {tot:.0f} ms ({tot/period:.2f} cycle)",
                transform=ax.transAxes, color='w', fontsize=8, va='bottom')
        if i == 0:
            ax.set_title("Dorsal-muscle kymogram  (anterior→posterior ↓, steady state)",
                         fontsize=10)
        if i < len(SCALES) - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("time (s)")
    cax = fig.add_subplot(gs[:, 1])
    fig.colorbar(im, cax=cax).set_label("dorsal muscle V (mV)", fontsize=9)
    fig.text(0.045, 0.5, "inter-segment gap-junction scale", rotation=90,
             va='center', fontsize=11)

    axp = fig.add_subplot(gs[:, 2])
    cmap = plt.cm.viridis
    cols = {s: cmap(i / (len(SCALES) - 1)) for i, s in enumerate(SCALES)}
    for s in SCALES:
        L = lag[s]
        axp.fill_between(np.arange(6), np.nanmin(L, 0), np.nanmax(L, 0),
                         color=cols[s], alpha=0.18)
        axp.plot(np.arange(6), np.nanmean(L, 0), 'o-', color=cols[s], lw=2,
                 label=f"gap ×{s}")
    axp.axhline(0, color='0.5', lw=0.8)
    axp.set_xlabel("segment  (0 = anterior → 5 = posterior)")
    axp.set_ylabel("phase lag vs segment 0  (ms)")
    axp.set_title("Per-segment phase lag (mean; band = min–max over 5 seeds)\n"
                  "monotone rise = coherent A→P wave; flat/scattered = incoherent",
                  fontsize=10)
    axp.legend(title="inter-seg gap", fontsize=9, loc='upper left')
    axp.text(0.02, 0.02,
             f"period ≈ {period:.0f} ms;  baseline lag ≈ "
             f"{np.nanmean(lag[1.0],0)[-1]/period:.2f} of a full body wavelength\n"
             f"(real worm ≈ 1 full wavelength)",
             transform=axp.transAxes, fontsize=8, color='0.3')

    fig.suptitle("Inter-segment coupling SETS the travelling wave — weakening it "
                 "degrades the wave, it does not create one\n"
                 f"Full 102-cell network, steady IAVA = 2.5 pA, β={settings.beta}, "
                 f"τ_w={settings.tau_w} ms   (wave present at baseline, but weak)",
                 fontsize=12)
    fig.tight_layout(rect=[0.05, 0, 1, 0.94])
    if save:
        fname = os.path.join(MEDIA_DIR, 'wave_intersegment_coupling.png')
        plt.savefig(fname, dpi=140)
        print(f"Saved → {fname}")
    plt.show()


if __name__ == '__main__':
    print("Inter-segment coupling vs travelling wave (steady AVA, per-segment phase lag):")
    run()
