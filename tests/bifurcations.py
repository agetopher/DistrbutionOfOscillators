import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import settings
from functions import f_vec, sf_vec, w_inf_vec

'''
bifurcations test: finding and classifying bifurcations in the
single-segment. 
'''

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
settings.C      = 7.0
settings.g      = 1.0
settings.L      = -70.0
settings.T      = -45.0
settings.H      = -35.0
settings.m1     = 0.7
settings.m2     = 1 / 81.0
settings.m3     = -1 / 30.0
settings.m4     = 0.17

# Segment synaptic parameters
settings.G_syne = 0.07
settings.E_syne = 0.0
settings.G_syni = 0.05
settings.E_syni = -100.0
settings.k_syn  = 0.25
settings.V_th   = -52.0

settings.a      = 0.000035
settings.b      = 0.005

settings.G_gap  = 0.03

# Recovery variable
settings.beta   = 1.0
settings.tau_w  = 200.0

# ── Circuit indices ───────────────────────────────────────────────────────────
N = SEG
settings.numCells = N

from scipy.signal import find_peaks

# Command-interneuron injection targets (by cell class)
AVA_CLASSES = (1, 2, 7)   # AS, DA, VA
AVB_CLASSES = (1, 3, 6)   # AS, DB, VB

CLASS_NAMES = {
    1: 'AS', 2: 'DA', 3: 'DB', 4: 'DD', 5: 'VD',
    6: 'VB', 7: 'VA', 8: 'dorsal muscle', 9: 'ventral muscle',
}

# Non-muscle (oscillator-capable) cells
NONMUSCLE = np.where(classes < 8)[0]

# Analysis settings
TRANSIENT_FRAC = 0.4     # fraction of trace discarded before measuring
PROMINENCE     = 8.0     # mV; minimum peak prominence to count as an oscillation peak


# ── Model ──────────────────────────────────────────────────────────────────────
def ode(t, y, I_inj):
    V  = y[:N]
    SF = y[N:2*N]
    w  = y[2*N:]

    fv   = f_vec(V)
    winf = w_inf_vec(V)
    s    = 1.0 / (1.0 + np.exp(-settings.k_syn * (V - settings.V_th)))
    pre  = s  # synaptic-fatigue gating currently disabled (matches src/network.py)

    I_exc = settings.G_syne * (E_conn @ pre) * (V - settings.E_syne)
    I_inh = settings.G_syni * (I_conn @ pre) * (V - settings.E_syni)
    Vdiff = V[:, None] - V[None, :]
    I_gap = settings.G_gap * (GJ_conn * Vdiff).sum(axis=1)

    dV     = (settings.g * fv - w - I_exc - I_inh - I_gap + I_inj) / settings.C
    sf_dot = sf_vec(V, SF)
    dw     = (winf - w) / settings.tau_w

    z = np.empty(3 * N)
    z[:N]    = dV
    z[N:2*N] = sf_dot
    z[2*N:]  = dw
    return z


def default_init():
    """Fresh initial state: all cells at rest, SF=0.5, w=0."""
    V_init_clipped = np.where(V_init > settings.H, settings.T, V_init)
    return np.concatenate([V_init_clipped, np.full(N, 0.5), np.zeros(N)])


def simulate(IAVA=0.0, IAVB=0.0, tf=8000.0, n_eval=3000, y0=None):
    """Integrate the segment under constant command-interneuron drive.

    y0 : optional full initial state (3N,). If None, starts from rest.
         Passing the previous run's final state implements parameter
         continuation (needed for hysteresis sweeps).

    Returns (t, V, y_end) where V has shape (N, n_eval) and y_end is the
    full final state (3N,) for continuing the next point.
    """
    I_inj = np.zeros(N)
    I_inj[np.isin(classes, AVA_CLASSES)] += IAVA
    I_inj[np.isin(classes, AVB_CLASSES)] += IAVB

    inits = default_init() if y0 is None else y0

    t   = np.linspace(0.0, tf, n_eval)
    sol = solve_ivp(lambda tt, yy: ode(tt, yy, I_inj), [0.0, tf], inits,
                    method='BDF', t_eval=t)
    return sol.t, sol.y[:N, :], sol.y[:, -1]


def cell_metrics(t_ss, trace):
    """Envelope (min, max), oscillation flag, and frequency [Hz] of one
    post-transient trace."""
    vmin, vmax = float(trace.min()), float(trace.max())
    peaks, _ = find_peaks(trace, prominence=PROMINENCE)
    oscillates = len(peaks) >= 2
    freq = 1000.0 / np.diff(t_ss[peaks]).mean() if oscillates else np.nan  # t[ms]->Hz
    return vmin, vmax, oscillates, freq


def measure(t, V):
    """Post-transient analysis of a whole-segment simulation.
    Returns per-cell envelope, segment-wide oscillation flag, and the
    dominant (largest-swing) non-muscle cell."""
    cutoff = int(TRANSIENT_FRAC * len(t))
    t_ss   = t[cutoff:]
    V_ss   = V[:, cutoff:]

    vmin = V_ss.min(axis=1)
    vmax = V_ss.max(axis=1)
    amp  = vmax - vmin

    # Segment oscillates if any non-muscle cell has a clear repeated peak.
    osc = [i for i in NONMUSCLE
           if len(find_peaks(V_ss[i], prominence=PROMINENCE)[0]) >= 2]
    dom = int(NONMUSCLE[np.argmax(amp[NONMUSCLE])])

    return dict(t_ss=t_ss, V_ss=V_ss, vmin=vmin, vmax=vmax, amp=amp,
                oscillates=len(osc) > 0, n_osc=len(osc), dom=dom)


# ── Bifurcation sweep ───────────────────────────────────────────────────────────
def sweep(branch, values, tf=8000.0, continued=False, y0_start=None):
    """Sweep one command-interneuron drive while holding the other at 0.

    branch    : 'AVA' or 'AVB'
    values     : sequence of drive amplitudes (pA), traversed in the given order
    continued  : if True, each point starts from the previous point's final
                 state (parameter continuation). If False, every point starts
                 from rest. Continuation is what reveals hysteresis.
    y0_start   : optional initial state for the first point (e.g. the up-sweep's
                 final state, to start a down-sweep from the oscillating branch).

    Readout is the first dorsal muscle cell (class 8) — the functional
    locomotion output. The oscillation band reflects the muscle's own swing;
    n_osc additionally reports how many non-muscle cells oscillate.
    """
    readout_cls = 8                                     # dorsal muscle
    readout     = int(np.where(classes == readout_cls)[0][0])

    rec = dict(branch=branch, values=np.asarray(values, float), readout=readout,
               readout_name=CLASS_NAMES[readout_cls], continued=continued,
               vmin=[], vmax=[], freq=[], oscillates=[], n_osc=[], y_end=None)

    y0 = y0_start
    for x in values:
        IAVA, IAVB = (x, 0.0) if branch == 'AVA' else (0.0, x)
        t, V, y_end = simulate(IAVA, IAVB, tf=tf, y0=y0)
        m    = measure(t, V)
        vmn, vmx, osc, frq = cell_metrics(m['t_ss'], m['V_ss'][readout])

        rec['vmin'].append(vmn)
        rec['vmax'].append(vmx)
        rec['freq'].append(frq)
        rec['oscillates'].append(osc)
        rec['n_osc'].append(m['n_osc'])
        print(f"  {branch}={x:5.2f} pA  osc={osc!s:5}  "
              f"n_osc={m['n_osc']:2d}  {rec['readout_name']} amp={vmx-vmn:5.1f} mV  "
              f"f={frq:.2f} Hz")

        if continued:
            y0 = y_end

    rec['y_end'] = y_end
    for k in ('vmin', 'vmax', 'freq', 'oscillates', 'n_osc'):
        rec[k] = np.asarray(rec[k])
    return rec


def sweep_beta(branch, betas, drive=2.5, tf=8000.0):
    """Fix the command-interneuron drive and sweep the recovery slope beta.

    branch : 'AVA' (IAVA=drive, IAVB=0) or 'AVB' (IAVB=drive, IAVA=0)
    betas  : sequence of beta values (nS/mV)
    drive  : the active command-interneuron current (pA)

    Each point starts from rest. Dorsal-muscle readout, as in sweep().
    """
    readout_cls = 8                                     # dorsal muscle
    readout     = int(np.where(classes == readout_cls)[0][0])
    IAVA, IAVB  = (drive, 0.0) if branch == 'AVA' else (0.0, drive)

    rec = dict(branch=branch, drive=drive, betas=np.asarray(betas, float),
               readout=readout, readout_name=CLASS_NAMES[readout_cls],
               vmin=[], vmax=[], freq=[], oscillates=[], n_osc=[])

    beta_saved = settings.beta
    try:
        for be in betas:
            settings.beta = float(be)                   # read inside ode() via w_inf_vec
            t, V, _ = simulate(IAVA, IAVB, tf=tf)
            m = measure(t, V)
            vmn, vmx, osc, frq = cell_metrics(m['t_ss'], m['V_ss'][readout])

            rec['vmin'].append(vmn)
            rec['vmax'].append(vmx)
            rec['freq'].append(frq)
            rec['oscillates'].append(osc)
            rec['n_osc'].append(m['n_osc'])
            print(f"  {branch} (I={drive} pA)  beta={be:4.2f}  osc={osc!s:5}  "
                  f"n_osc={m['n_osc']:2d}  {rec['readout_name']} amp={vmx-vmn:5.1f} mV  "
                  f"f={frq:.2f} Hz")
    finally:
        settings.beta = beta_saved                      # restore global

    for k in ('vmin', 'vmax', 'freq', 'oscillates', 'n_osc'):
        rec[k] = np.asarray(rec[k])
    return rec


def osc_window(rec, key='values'):
    """(lo, hi) sweep-parameter values bracketing oscillation, independent of
    sweep order. `key` selects the swept axis ('values' for drive, 'betas' for
    the recovery-slope sweep). Returns (nan, nan) if it never oscillates."""
    osc_vals = rec[key][rec['oscillates']]
    if osc_vals.size == 0:
        return np.nan, np.nan
    return float(osc_vals.min()), float(osc_vals.max())


def onset(rec):
    """First drive value at which the segment oscillates (nan if never)."""
    idx = np.where(rec['oscillates'])[0]
    return rec['values'][idx[0]] if len(idx) else np.nan


# ── Plotting ────────────────────────────────────────────────────────────────────
def plot_branches(pairs, save=False):
    """pairs : list of (branch, up_rec, down_rec). Overlays the up- and
    down-sweeps so any hysteresis (oscillation band that persists to lower
    drive on the way down) is visible as a mismatch between the two."""
    fig, axes = plt.subplots(2, len(pairs), figsize=(6 * len(pairs), 8),
                             sharex='col', squeeze=False)

    UP   = dict(color='steelblue', label='up-sweep')
    DOWN = dict(color='crimson',   label='down-sweep')

    for col, (branch, up, down) in enumerate(pairs):
        lo_up, hi_up = osc_window(up)    # oscillation interval going up
        lo_dn, hi_dn = osc_window(down)  # oscillation interval coming down

        # Row 0: voltage envelope (bifurcation diagram), up vs down
        ax = axes[0][col]
        for rec, sty, fa in ((up, UP, 0.22), (down, DOWN, 0.18)):
            x, osc = rec['values'], rec['oscillates']
            ax.fill_between(x, rec['vmin'], rec['vmax'], where=osc,
                            color=sty['color'], alpha=fa)
            ax.plot(x, rec['vmax'], color=sty['color'], lw=1.2, label=sty['label'])
            ax.plot(x, rec['vmin'], color=sty['color'], lw=1.2)
        # mark the four bifurcation edges
        for v, c in ((lo_up, 'steelblue'), (hi_up, 'steelblue'),
                     (lo_dn, 'crimson'),   (hi_dn, 'crimson')):
            if not np.isnan(v):
                ax.axvline(v, color=c, ls='--', lw=0.9)
        # shade bistable (hysteretic) gaps at each edge where up/down disagree
        first_gap = True
        for a, b in ((lo_dn, lo_up), (hi_dn, hi_up)):
            if not np.isnan(a) and not np.isnan(b) and abs(a - b) > 1e-9:
                ax.axvspan(min(a, b), max(a, b), color='gold', alpha=0.25,
                           label='hysteretic gap' if first_gap else None)
                first_gap = False
        ax.axhline(settings.T, color='forestgreen', ls=':', lw=0.7, alpha=0.6)
        ax.set_ylabel(f"{up['readout_name']} V envelope (mV)")
        ax.set_title(f"{branch} branch  "
                     f"(I{branch} swept, I{'AVB' if branch=='AVA' else 'AVA'}=0)")
        ax.legend(fontsize=7)

        # Row 1: frequency, up vs down
        ax = axes[1][col]
        ax.plot(up['values'],   up['freq'],   'o-', ms=3, lw=1.0,
                color=UP['color'],   label='up-sweep')
        ax.plot(down['values'], down['freq'], 's--', ms=3, lw=1.0,
                color=DOWN['color'], label='down-sweep')
        for v, c in ((lo_up, 'steelblue'), (hi_up, 'steelblue'),
                     (lo_dn, 'crimson'),   (hi_dn, 'crimson')):
            if not np.isnan(v):
                ax.axvline(v, color=c, ls='--', lw=0.9)
        ax.set_ylabel(f"{up['readout_name']} frequency (Hz)")
        ax.set_xlabel(f"I{branch} drive (pA)")
        ax.legend(fontsize=7)

    fig.suptitle(
        "Single-segment bifurcation with hysteresis (up vs down sweep)\n"
        f"G_syne={settings.G_syne}, G_syni={settings.G_syni}, G_gap={settings.G_gap} nS  |  "
        f"beta={settings.beta}, tau_w={settings.tau_w} ms",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save:
        fname = os.path.join(MEDIA_DIR, 'bifurcation_IAVA_IAVB_hysteresis.png')
        plt.savefig(fname, dpi=150)
        print(f"Saved → {fname}")
    plt.show()


def sweep_hysteresis(branch, values, tf=8000.0):
    """Up-sweep (continued from rest) then down-sweep (continued from the
    up-sweep's top state). Returns (up_rec, down_rec)."""
    vals = np.asarray(values, float)
    print(f"{branch} up-sweep (ascending, continued from rest):")
    up = sweep(branch, vals, tf=tf, continued=True)
    print(f"{branch} down-sweep (descending, continued from top state):")
    down = sweep(branch, vals[::-1], tf=tf, continued=True, y0_start=up['y_end'])
    return up, down


def plot_bifurcation(recs, save=False):
    """Plain bifurcation diagram (no hysteresis): each point sweeps from rest.
    One column per branch: voltage envelope (top) and frequency (bottom), with
    the oscillation onset marked."""
    fig, axes = plt.subplots(2, len(recs), figsize=(6 * len(recs), 8),
                             sharex='col', squeeze=False)

    for col, rec in enumerate(recs):
        x   = rec['values']
        osc = rec['oscillates']
        lo, hi = osc_window(rec)

        # Row 0: voltage envelope (bifurcation diagram)
        ax = axes[0][col]
        ax.fill_between(x, rec['vmin'], rec['vmax'], where=osc,
                        color='steelblue', alpha=0.25, label='oscillation band')
        ax.plot(x, rec['vmax'], color='steelblue', lw=1.2)
        ax.plot(x, rec['vmin'], color='steelblue', lw=1.2)
        ax.plot(x[~osc], rec['vmax'][~osc], '.', color='0.4', ms=4,
                label='fixed point')
        if not np.isnan(lo):
            ax.axvline(lo, color='crimson', ls='--', lw=1.0,
                       label=f'onset ≈ {lo:.2f} pA')
        if not np.isnan(hi):
            ax.axvline(hi, color='darkorange', ls='--', lw=1.0,
                       label=f'offset ≈ {hi:.2f} pA')
        ax.axhline(settings.T, color='forestgreen', ls=':', lw=0.7, alpha=0.6)
        ax.set_ylabel(f"{rec['readout_name']} V envelope (mV)")
        ax.set_title(f"{rec['branch']} branch  "
                     f"(I{rec['branch']} swept, "
                     f"I{'AVB' if rec['branch']=='AVA' else 'AVA'}=0)")
        ax.legend(fontsize=8)

        # Row 1: frequency
        ax = axes[1][col]
        ax.plot(x, rec['freq'], 'o-', color='darkorange', ms=3, lw=1.0)
        if not np.isnan(lo):
            ax.axvline(lo, color='crimson', ls='--', lw=1.0)
        if not np.isnan(hi):
            ax.axvline(hi, color='darkorange', ls='--', lw=1.0)
        ax.set_ylabel(f"{rec['readout_name']} frequency (Hz)")
        ax.set_xlabel(f"I{rec['branch']} drive (pA)")

    fig.suptitle(
        "Single-segment bifurcation diagram (each point from rest, no hysteresis)\n"
        f"G_syne={settings.G_syne}, G_syni={settings.G_syni}, G_gap={settings.G_gap} nS  |  "
        f"beta={settings.beta}, tau_w={settings.tau_w} ms",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save:
        fname = os.path.join(MEDIA_DIR, 'bifurcation_IAVA_IAVB.png')
        plt.savefig(fname, dpi=150)
        print(f"Saved → {fname}")
    plt.show()


def plot_beta_sweep(recs, save=False):
    """Bifurcation diagram with the recovery slope beta on the x-axis, command
    drive fixed. One column per branch: envelope (top) and frequency (bottom)."""
    fig, axes = plt.subplots(2, len(recs), figsize=(6 * len(recs), 8),
                             sharex='col', squeeze=False)

    for col, rec in enumerate(recs):
        x   = rec['betas']
        osc = rec['oscillates']
        lo, hi = osc_window(rec, key='betas')

        # Row 0: voltage envelope
        ax = axes[0][col]
        ax.fill_between(x, rec['vmin'], rec['vmax'], where=osc,
                        color='steelblue', alpha=0.25, label='oscillation band')
        ax.plot(x, rec['vmax'], color='steelblue', lw=1.2)
        ax.plot(x, rec['vmin'], color='steelblue', lw=1.2)
        ax.plot(x[~osc], rec['vmax'][~osc], '.', color='0.4', ms=4,
                label='fixed point')
        if not np.isnan(lo):
            ax.axvline(lo, color='crimson', ls='--', lw=1.0,
                       label=f'onset β ≈ {lo:.2f}')
        ax.axhline(settings.T, color='forestgreen', ls=':', lw=0.7, alpha=0.6)
        ax.set_ylabel(f"{rec['readout_name']} V envelope (mV)")
        ax.set_title(f"{rec['branch']} branch  "
                     f"(I{rec['branch']}={rec['drive']} pA, other=0)")
        ax.legend(fontsize=8)

        # Row 1: frequency
        ax = axes[1][col]
        ax.plot(x, rec['freq'], 'o-', color='darkorange', ms=3, lw=1.0)
        if not np.isnan(lo):
            ax.axvline(lo, color='crimson', ls='--', lw=1.0)
        ax.set_ylabel(f"{rec['readout_name']} frequency (Hz)")
        ax.set_xlabel("recovery slope β (nS/mV)")

    fig.suptitle(
        "Single-segment oscillation vs recovery slope β (command drive fixed)\n"
        f"G_syne={settings.G_syne}, G_syni={settings.G_syni}, G_gap={settings.G_gap} nS  |  "
        f"tau_w={settings.tau_w} ms",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save:
        fname = os.path.join(MEDIA_DIR, 'bifurcation_beta.png')
        plt.savefig(fname, dpi=150)
        print(f"Saved → {fname}")
    plt.show()


if __name__ == '__main__':
    # MODE: 'beta'        -> fix command drive, sweep recovery slope beta
    #       'drive'       -> fix beta, sweep command drive with up/down hysteresis
    #       'bifurcation' -> fix beta, sweep command drive from rest (no hysteresis)
    MODE  = 'drive'
    DRIVE = 2.5          # pA, the active command-interneuron current (try 2.0 or 2.5)

    if MODE == 'beta':
        betas = np.arange(0.2, 1.001, 0.02)
        print(f"AVA branch (IAVA={DRIVE} pA, IAVB=0), sweeping beta:")
        rec_ava = sweep_beta('AVA', betas, drive=DRIVE)
        print(f"AVB branch (IAVB={DRIVE} pA, IAVA=0), sweeping beta:")
        rec_avb = sweep_beta('AVB', betas, drive=DRIVE)

        for name, rec in (('AVA', rec_ava), ('AVB', rec_avb)):
            lo, hi = osc_window(rec, key='betas')
            print(f"\n{name} (I={DRIVE} pA): oscillates for beta in "
                  f"{lo:.2f}–{hi:.2f} nS/mV" if not np.isnan(lo)
                  else f"\n{name} (I={DRIVE} pA): no oscillation over the beta range")

        plot_beta_sweep([rec_ava, rec_avb], save=False)

    elif MODE == 'bifurcation':
        drive = np.arange(0.0, 6.01, 0.01)
        print(f"AVA branch (IAVA swept 0–6 pA, IAVB=0), each point from rest:")
        rec_ava = sweep('AVA', drive, continued=False)
        print(f"AVB branch (IAVB swept 0–6 pA, IAVA=0), each point from rest:")
        rec_avb = sweep('AVB', drive, continued=False)

        for name, rec in (('AVA', rec_ava), ('AVB', rec_avb)):
            lo, hi = osc_window(rec)
            print(f"\n{name}: oscillates for drive in {lo:.2f}–{hi:.2f} pA"
                  if not np.isnan(lo)
                  else f"\n{name}: no oscillation over the drive range")

        plot_bifurcation([rec_ava, rec_avb], save=True)

    elif MODE == 'drive':
        drive = np.arange(0.0, 6.01, 0.01)
        ava_up, ava_dn = sweep_hysteresis('AVA', drive)
        avb_up, avb_dn = sweep_hysteresis('AVB', drive)

        for name, up, dn in (('AVA', ava_up, ava_dn), ('AVB', avb_up, avb_dn)):
            lo_up, hi_up = osc_window(up)
            lo_dn, hi_dn = osc_window(dn)
            lower_gap = abs(lo_up - lo_dn)   # bistability at the oscillation-birth edge
            upper_gap = abs(hi_up - hi_dn)   # bistability at the oscillation-death edge
            bistable  = (lower_gap > 1e-9) or (upper_gap > 1e-9)
            print(f"\n{name}: up oscillates {lo_up:.2f}–{hi_up:.2f} pA | "
                  f"down oscillates {lo_dn:.2f}–{hi_dn:.2f} pA")
            print(f"      lower-edge gap = {lower_gap:.2f} pA | "
                  f"upper-edge gap = {upper_gap:.2f} pA  "
                  + ("→ HYSTERESIS / bistable region" if bistable else "→ no hysteresis"))

        plot_branches([('AVA', ava_up, ava_dn), ('AVB', avb_up, avb_dn)], save=False)