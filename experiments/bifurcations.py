import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import settings
import segment
from analysis import dorsoventral_activity

'''
bifurcations test: finding and classifying bifurcations in the
single-segment. 
'''

MEDIA_DIR = os.path.join(os.path.dirname(__file__), '..', 'media')

settings.reset_defaults()
segment.configure()

N = segment.N_CELLS
classes = segment.CLASSES

from scipy.signal import find_peaks

# Command-interneuron injection targets (by cell class)
AVA_CLASSES = segment.AVA_CLASSES
AVB_CLASSES = segment.AVB_CLASSES

CLASS_NAMES = {
    1: 'AS', 2: 'DA', 3: 'DB', 4: 'DD', 5: 'VD',
    6: 'VB', 7: 'VA', 8: 'dorsal muscle', 9: 'ventral muscle',
}

# Non-muscle (oscillator-capable) cells
NONMUSCLE = np.where(classes < 8)[0]

# Analysis settings
TRANSIENT_FRAC = 0.4     # fraction of trace discarded before measuring
PROMINENCE     = 8.0     # mV; minimum peak prominence to count as an oscillation peak


def dv_activity(voltage):
    """Instantaneous mean dorsal-minus-ventral muscle voltage."""
    return dorsoventral_activity(voltage, classes)


def simulate(IAVA=0.0, IAVB=0.0, tf=8000.0, n_eval=3000, y0=None):
    """Integrate the segment under constant command-interneuron drive.

    y0 : optional full initial state (3N,). If None, starts from rest.
         Passing the previous run's final state implements parameter
         continuation (needed for hysteresis sweeps).

    Returns (t, V, y_end) where V has shape (N, n_eval) and y_end is the
    full final state (3N,) for continuing the next point.
    """
    I_inj = segment.drive_vector(IAVA, IAVB)
    inits = segment.full_rest_state() if y0 is None else y0

    t   = np.linspace(0.0, tf, n_eval)
    sol = solve_ivp(lambda tt, yy: segment.rhs_full(tt, yy, I_inj), [0.0, tf], inits,
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

    Readout is the instantaneous difference between the side-averaged dorsal
    and ventral muscle voltages. The oscillation band therefore measures the
    functional dorsoventral output; n_osc separately reports how many
    non-muscle cells oscillate.
    """
    rec = dict(branch=branch, values=np.asarray(values, float),
               readout_name='mean dorsal - mean ventral', continued=continued,
               vmin=[], vmax=[], freq=[], oscillates=[], n_osc=[], y_end=None)

    y0 = y0_start
    for x in values:
        IAVA, IAVB = (x, 0.0) if branch == 'AVA' else (0.0, x)
        t, V, y_end = simulate(IAVA, IAVB, tf=tf, y0=y0)
        m    = measure(t, V)
        vmn, vmx, osc, frq = cell_metrics(m['t_ss'], dv_activity(m['V_ss']))

        rec['vmin'].append(vmn)
        rec['vmax'].append(vmx)
        rec['freq'].append(frq)
        rec['oscillates'].append(osc)
        rec['n_osc'].append(m['n_osc'])
        print(f"  {branch}={x:5.2f} pA  osc={osc!s:5}  "
              f"n_osc={m['n_osc']:2d}  DV amp={vmx-vmn:5.1f} mV  "
              f"f={frq:.2f} Hz")

        if continued:
            y0 = y_end

    rec['y_end'] = y_end
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


# ── Equilibrium & eigenvalue analysis (reduced (V, w) system) ────────────────────
# Synaptic fatigue is inert here (pre = s(V) only, matching src/network.py), so
# the dynamically relevant state is (V, w), a 2N-dimensional system. Working in
# that subspace lets us find the segment's equilibria and classify their
# stability from the Jacobian, so the bifurcations are located and typed
# directly (saddle-node vs Hopf) instead of inferred from the waveform.

def _drive_along(branch, d):
    return segment.drive_along(branch, d)


rhs_Vw = segment.rhs_vw


def jac_Vw(x, I_inj, eps=1e-7):
    """Forward-difference Jacobian of rhs_Vw (2N x 2N)."""
    n = len(x)
    J = np.empty((n, n))
    f0 = rhs_Vw(x, I_inj)
    for k in range(n):
        xp = x.copy()
        xp[k] += eps
        J[:, k] = (rhs_Vw(xp, I_inj) - f0) / eps
    return J


def equilibrium(I_inj, x0, tol=1e-6):
    """Solve rhs_Vw = 0 from seed x0; return the root, or None if not converged."""
    sol, info, ier, msg = fsolve(lambda z: rhs_Vw(z, I_inj), x0, full_output=True)
    if ier == 1 and np.abs(rhs_Vw(sol, I_inj)).max() < tol:
        return sol
    return None


rest_state = segment.reduced_rest_state


def simulate_Vw(branch, drive, tf, n_eval, x0=None):
    """Fast reduced-system integration of (V, w) under single-branch drive.
    Returns (t, V) with V of shape (N, n_eval)."""
    I = _drive_along(branch, drive)
    x0 = rest_state() if x0 is None else x0
    t = np.linspace(0.0, tf, n_eval)
    sol = solve_ivp(lambda tt, xx: rhs_Vw(xx, I), [0.0, tf], x0,
                    method='BDF', t_eval=t)
    return sol.t, sol.y[:N, :]


def continue_equilibrium(branch, drives, x0):
    """Natural-parameter continuation of one equilibrium branch.

    Follows the equilibrium seeded at x0 as the active command drive steps
    through `drives`, recording the driven-cell mean voltage and, from the
    Jacobian, the leading eigenvalue (largest real part) and the count of
    unstable modes. Where no nearby equilibrium exists (a fold), the point is
    flagged exists=False and the seed is held so continuation can resume if the
    branch reappears. Returns a dict of per-point arrays."""
    rec = dict(branch=branch, drives=np.asarray(drives, float),
               activity=[], max_re=[], lead_im=[], n_unstable=[], exists=[])
    x = np.array(x0, float)
    for d in drives:
        I = _drive_along(branch, d)
        sol = equilibrium(I, x)
        if sol is None:
            rec['activity'].append(np.nan); rec['max_re'].append(np.nan)
            rec['lead_im'].append(np.nan); rec['n_unstable'].append(-1)
            rec['exists'].append(False)
            continue
        x = sol
        ev = np.linalg.eigvals(jac_Vw(sol, I))
        lead = ev[np.argmax(ev.real)]
        rec['activity'].append(float(dv_activity(sol[:N])))
        rec['max_re'].append(float(lead.real))
        rec['lead_im'].append(float(abs(lead.imag)))
        rec['n_unstable'].append(int((ev.real > 1e-9).sum()))
        rec['exists'].append(True)
    for k in ('activity', 'max_re', 'lead_im', 'n_unstable', 'exists'):
        rec[k] = np.asarray(rec[k])
    return rec


def locate_fold(branch, hi=3.0, step=0.001):
    """Highest drive at which the resting equilibrium persists (continuation
    from rest). Just above it the stable node and a saddle annihilate — the
    saddle-node / SNIC point. Returns (I_fold, x_fold) or (nan, None)."""
    x = rest_state()
    last, last_x = np.nan, None
    for d in np.arange(0.0, hi, step):
        sol = equilibrium(_drive_along(branch, d), x)
        if sol is None:
            break
        x, last, last_x = sol, d, sol
    return last, last_x


def locate_upper_hopf(branch, seed_drive=5.5, drives=None, tf=15000.0):
    """Continue the depolarised equilibrium down from high drive (seeded from a
    settled simulation) and return the drive where its leading complex pair
    crosses Re = 0 (a Hopf bifurcation), together with the branch record.
    Returns (I_hopf, rec); I_hopf is nan if no stable depolarised equilibrium /
    Hopf is found in range."""
    if drives is None:
        drives = np.arange(seed_drive, 1.5, -0.05)
    a, b = (seed_drive, 0.0) if branch == 'AVA' else (0.0, seed_drive)
    _, _, y_end = simulate(a, b, tf=tf, n_eval=200)
    x0 = np.concatenate([y_end[:N], y_end[2 * N:]])   # (V, w) from (V, SF, w)
    rec = continue_equilibrium(branch, drives, x0)
    re, im, ex = rec['max_re'], rec['lead_im'], rec['exists']
    I_hopf = np.nan
    for i in range(1, len(drives)):
        if ex[i] and ex[i - 1] and re[i - 1] < 0 <= re[i] and im[i] > 1e-4:
            I_hopf = float(drives[i])
            break
    return I_hopf, rec


def onset_scaling(branch, I_SN,
                  offsets=(0.003, 0.005, 0.008, 0.013, 0.02, 0.035, 0.06, 0.1),
                  prominence=PROMINENCE):
    """Limit-cycle period just above the fold, to expose the onset scaling law.

    Near a SNIC the period diverges as T ~ (I - I_SN)^(-1/2): frequency-squared
    is linear in dI = I - I_SN with a zero intercept, and T*sqrt(dI) -> a
    nonzero constant. A saddle-homoclinic instead gives a logarithmic
    divergence T ~ -ln(dI), for which T*sqrt(dI) -> 0. Integration time is
    lengthened as onset is approached to resolve the long periods. Returns
    dict(dI, freq, period, Tsqrt, snic_slope, snic_intercept, snic_r2)."""
    driven = np.where(np.isin(classes,
                              AVA_CLASSES if branch == 'AVA' else AVB_CLASSES))[0]
    dI, freq, period = [], [], []
    for off in offsets:
        # ~20 cycles + transient; period ~ 235/sqrt(off) ms empirically.
        tf = float(np.clip(20.0 * 235.0 / np.sqrt(off), 40000.0, 200000.0))
        t, V = simulate_Vw(branch, I_SN + off, tf, int(tf / 12))
        keep = t >= 0.5 * tf
        tt = t[keep]
        sub = V[driven][:, keep]
        tr = sub[np.argmax(sub.max(1) - sub.min(1))]
        pk, _ = find_peaks(tr, prominence=prominence)
        if len(pk) < 3:
            continue
        T = float(np.median(np.diff(tt[pk])))
        dI.append(off); period.append(T); freq.append(1000.0 / T)
    dI = np.asarray(dI); freq = np.asarray(freq); period = np.asarray(period)

    slope = intercept = r2 = np.nan
    if dI.size >= 2:
        A = np.vstack([dI, np.ones_like(dI)]).T
        (slope, intercept), *_ = np.linalg.lstsq(A, freq ** 2, rcond=None)
        ss = np.sum((freq ** 2 - (slope * dI + intercept)) ** 2)
        tot = np.sum((freq ** 2 - np.mean(freq ** 2)) ** 2)
        r2 = 1.0 - ss / tot if tot > 0 else np.nan
    return dict(branch=branch, I_SN=I_SN, dI=dI, freq=freq, period=period,
                Tsqrt=period * np.sqrt(dI), snic_slope=slope,
                snic_intercept=intercept, snic_r2=r2)


def oscillation_branch(branch, drives, tf=25000.0, n_eval=2500,
                       prominence=PROMINENCE):
    """Reduced-system up-sweep from rest at each drive. Returns, per drive: the
    driven-cell mean-voltage envelope (for the stability diagram), the
    oscillation count, and the frequency of the largest-swing non-muscle cell
    (the segment's rhythm, which need not be a driven cell)."""
    rec = dict(branch=branch, drives=np.asarray(drives, float),
               vmin=[], vmax=[], oscillates=[], n_osc=[], freq=[])
    for d in drives:
        t, V = simulate_Vw(branch, d, tf, n_eval)
        keep = t >= 0.4 * tf
        tt = t[keep]
        Vss = V[:, keep]
        mtrace = dv_activity(Vss)
        n_osc = sum(len(find_peaks(Vss[i], prominence=prominence)[0]) >= 2
                    for i in NONMUSCLE)
        pk, _ = find_peaks(mtrace, prominence=prominence)
        output_oscillates = len(pk) >= 2
        freq = 1000.0 / np.median(np.diff(tt[pk])) if len(pk) >= 3 else np.nan
        rec['vmin'].append(float(mtrace.min()))
        rec['vmax'].append(float(mtrace.max()))
        rec['oscillates'].append(output_oscillates)
        rec['n_osc'].append(n_osc)
        rec['freq'].append(freq)
    for k in ('vmin', 'vmax', 'oscillates', 'n_osc', 'freq'):
        rec[k] = np.asarray(rec[k])
    return rec


def analyze_stability(branch, drive_max=6.0, verbose=True):
    """Locate and type the segment's bifurcations along one command branch.

    Returns a results dict bundling: the lower fold (SNIC) point, the rest and
    upper equilibrium branches (with eigenvalues), the SNIC onset-scaling test,
    the upper Hopf point, and the simulated oscillation envelope."""
    I_SN, _ = locate_fold(branch, hi=min(drive_max, 3.0))

    lo_drives = np.arange(0.0, (I_SN if not np.isnan(I_SN) else drive_max) + 0.02,
                          0.02)
    rest_rec = continue_equilibrium(branch, lo_drives, rest_state())

    I_hopf, upper_rec = locate_upper_hopf(
        branch, seed_drive=min(drive_max, 5.5),
        drives=np.arange(min(drive_max, 5.5), 1.5, -0.05))

    scaling = onset_scaling(branch, I_SN) if not np.isnan(I_SN) else None
    env = oscillation_branch(branch, np.arange(0.0, drive_max + 0.05, 0.1))

    res = dict(branch=branch, I_SN=I_SN, I_hopf=I_hopf, rest_rec=rest_rec,
               upper_rec=upper_rec, scaling=scaling, env=env)

    if verbose:
        print(f"\n=== {branch} branch  (beta={settings.beta}, tau_w={settings.tau_w}) ===")
        print(f"  lower edge : SNIC / saddle-node at I_SN = {I_SN:.3f} pA")
        if scaling is not None and not np.isnan(scaling['snic_r2']):
            ts = scaling['Tsqrt']
            print(f"    onset scaling: f^2 = {scaling['snic_slope']:.2f}*dI "
                  f"{scaling['snic_intercept']:+.4f}  (R^2={scaling['snic_r2']:.3f}); "
                  f"T*sqrt(dI) {ts[-1]:.0f}->{ts[0]:.0f} ms as dI->0 "
                  f"(-> nonzero const = SNIC)")
        if not np.isnan(I_hopf):
            print(f"  upper edge : Hopf at I_hopf = {I_hopf:.3f} pA "
                  f"(depolarised equilibrium regains stability above)")
        else:
            print("  upper edge : none — no stable upper equilibrium was located "
                  "in the searched range; oscillations persist in simulation")
    return res


# ── β-sweep analysis (recovery slope at fixed command drive) ─────────────────────
# Sweeping beta at fixed drive crosses a *different* onset than sweeping drive:
# the fixed point does not annihilate, it loses stability through a Hopf. The
# same reduced-(V,w) machinery applies, now stepping settings.beta.

def fft_fundamental(trace, dt_ms, fmin=0.2, fmax=8.0):
    """Fundamental frequency (Hz) of a trace via the windowed FFT peak in
    [fmin, fmax]. Robust to the multi-phasic waveforms that confuse peak
    counting. dt_ms is the sample spacing in ms."""
    x = np.asarray(trace, float)
    x = x - x.mean()
    n = x.size
    if n < 8 or np.allclose(x, 0.0):
        return np.nan
    sp = np.abs(np.fft.rfft(x * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, d=dt_ms / 1000.0)
    band = (freqs >= fmin) & (freqs <= fmax)
    if not band.any() or sp[band].max() == 0:
        return np.nan
    return float(freqs[band][np.argmax(sp[band])])


def beta_fixed_point_branch(branch, betas, drive):
    """Continue the depolarised fixed point across `betas` at fixed drive,
    recording its leading eigenvalue. The Hopf is where a complex pair crosses
    Re = 0. Seeds from a settled simulation at betas[0] (assumed below onset, so
    the trajectory rests at the fixed point)."""
    rec = dict(branch=branch, betas=np.asarray(betas, float), drive=drive,
               activity=[], max_re=[], lead_im=[], n_unstable=[], exists=[])
    saved = settings.beta
    try:
        settings.beta = float(betas[0])
        t, V = simulate_Vw(branch, drive, 20000.0, 1000)
        x = np.concatenate([V[:, -1],
                            settings.beta * np.maximum(V[:, -1] - settings.T, 0.0)])
        for be in betas:
            settings.beta = float(be)
            I = _drive_along(branch, drive)
            sol = equilibrium(I, x)
            if sol is None:
                rec['activity'].append(np.nan); rec['max_re'].append(np.nan)
                rec['lead_im'].append(np.nan); rec['n_unstable'].append(-1)
                rec['exists'].append(False)
                continue
            x = sol
            ev = np.linalg.eigvals(jac_Vw(sol, I))
            lead = ev[np.argmax(ev.real)]
            rec['activity'].append(float(dv_activity(sol[:N])))
            rec['max_re'].append(float(lead.real))
            rec['lead_im'].append(float(abs(lead.imag)))
            rec['n_unstable'].append(int((ev.real > 1e-9).sum()))
            rec['exists'].append(True)
    finally:
        settings.beta = saved
    for k in ('activity', 'max_re', 'lead_im', 'n_unstable', 'exists'):
        rec[k] = np.asarray(rec[k])
    return rec


def beta_oscillation_branch(branch, betas, drive, tf=25000.0, n_eval=4000,
                            prominence=PROMINENCE):
    """Per-beta up-sweep from rest at fixed drive: driven-cell mean-voltage
    envelope, oscillation count, and the dominant cell's FFT frequency."""
    rec = dict(branch=branch, betas=np.asarray(betas, float), drive=drive,
               vmin=[], vmax=[], oscillates=[], n_osc=[], freq=[])
    saved = settings.beta
    try:
        for be in betas:
            settings.beta = float(be)
            t, V = simulate_Vw(branch, drive, tf, n_eval)
            keep = t >= 0.4 * tf
            tt = t[keep]
            Vss = V[:, keep]
            mtrace = dv_activity(Vss)
            n_osc = sum(len(find_peaks(Vss[i], prominence=prominence)[0]) >= 2
                        for i in NONMUSCLE)
            peaks, _ = find_peaks(mtrace, prominence=prominence)
            output_oscillates = len(peaks) >= 2
            freq = (fft_fundamental(mtrace, tt[1] - tt[0])
                    if output_oscillates else np.nan)
            rec['vmin'].append(float(mtrace.min()))
            rec['vmax'].append(float(mtrace.max()))
            rec['oscillates'].append(output_oscillates)
            rec['n_osc'].append(n_osc)
            rec['freq'].append(freq)
    finally:
        settings.beta = saved
    for k in ('vmin', 'vmax', 'oscillates', 'n_osc', 'freq'):
        rec[k] = np.asarray(rec[k])
    return rec


def analyze_beta(branch, drive=2.5, betas=None, verbose=True):
    """β-sweep at fixed command drive: fixed-point stability (Hopf), oscillation
    envelope/frequency, and the fold-of-cycles onset. Returns a results dict.

    `betas` should start below the oscillation onset so the fixed-point seed is
    valid (default 0.8, below the canonical ~0.90 cycle onset)."""
    if betas is None:
        betas = np.arange(0.8, 2.001, 0.02)
    betas = np.asarray(betas, float)
    fp = beta_fixed_point_branch(branch, betas, drive)
    env = beta_oscillation_branch(branch, betas, drive)

    # Hopf: leading complex pair's real part crosses 0 (linear-interpolated)
    re, im, ex = fp['max_re'], fp['lead_im'], fp['exists']
    beta_hopf = np.nan
    for i in range(1, len(betas)):
        if ex[i] and ex[i - 1] and re[i - 1] < 0 <= re[i] and im[i] > 1e-4:
            beta_hopf = float(betas[i - 1] + (betas[i] - betas[i - 1])
                              * (-re[i - 1]) / (re[i] - re[i - 1]))
            break
    # fold of cycles: first beta (from rest) that oscillates
    osc_idx = np.where(env['n_osc'] > 0)[0]
    beta_onset = float(betas[osc_idx[0]]) if osc_idx.size else np.nan

    res = dict(branch=branch, drive=drive, betas=betas, fp=fp, env=env,
               beta_hopf=beta_hopf, beta_onset=beta_onset)
    if verbose:
        print(f"\n=== {branch} β-sweep  (I{branch}={drive} pA fixed) ===")
        print(f"  fixed-point Hopf (complex pair Re→0) : β = {beta_hopf:.3f}")
        print(f"  oscillation onset from rest          : β = {beta_onset:.3f}")
        if (not np.isnan(beta_hopf) and not np.isnan(beta_onset)
                and beta_onset < beta_hopf - 1e-6):
            print(f"  => SUBCRITICAL Hopf: bistable window β ∈ "
                  f"[{beta_onset:.3f}, {beta_hopf:.3f}] — a stable fixed point "
                  f"coexists with the limit cycle")
    return res


# ── k_syn analysis (zero-command spontaneous oscillation) ─────────────────────

def k_syn_equilibrium_branch(k_values):
    """Continue the zero-command equilibrium from steep to shallow gating.

    ``k_values`` must be descending and begin where the equilibrium is stable.
    The branch ends in a saddle-node as ``k_syn`` decreases. The critical value
    is estimated by the saddle-node scaling lambda^2 ∝ k_syn - k_critical for
    the leading real eigenvalue.
    """
    k_values = np.asarray(k_values, float)
    if np.any(np.diff(k_values) >= 0):
        raise ValueError("k_values must be strictly descending")

    current = segment.drive_vector()
    saved = settings.k_syn
    rec = dict(k_values=k_values, V_DD=[], max_re=[], lead_im=[],
               n_unstable=[], exists=[])
    try:
        settings.k_syn = float(k_values[0])
        state = equilibrium(current, rest_state())
        for k_syn in k_values:
            settings.k_syn = float(k_syn)
            state = equilibrium(current, state) if state is not None else None
            if state is None:
                rec['V_DD'].append(np.nan)
                rec['max_re'].append(np.nan)
                rec['lead_im'].append(np.nan)
                rec['n_unstable'].append(-1)
                rec['exists'].append(False)
                continue
            eigenvalues = np.linalg.eigvals(jac_Vw(state, current))
            lead = eigenvalues[np.argmax(eigenvalues.real)]
            rec['V_DD'].append(float(state[np.where(classes == 4)[0][0]]))
            rec['max_re'].append(float(lead.real))
            rec['lead_im'].append(float(abs(lead.imag)))
            rec['n_unstable'].append(int((eigenvalues.real > 1e-9).sum()))
            rec['exists'].append(True)
    finally:
        settings.k_syn = saved

    for key in ('V_DD', 'max_re', 'lead_im', 'n_unstable', 'exists'):
        rec[key] = np.asarray(rec[key])

    # Use the points where the critical eigenvalue has separated from the
    # recovery eigenvalue (-1/tau_w) and is approaching zero.
    fit = (rec['exists']
           & (rec['max_re'] > -0.0049)
           & (rec['max_re'] < 0.0)
           & (rec['lead_im'] < 1e-8))
    rec['k_critical'] = np.nan
    rec['fold_fit_r2'] = np.nan
    if fit.sum() >= 3:
        x = k_values[fit]
        y = rec['max_re'][fit] ** 2
        slope, intercept = np.polyfit(x, y, 1)
        predicted = slope * x + intercept
        total = np.sum((y - y.mean()) ** 2)
        rec['fold_fit_r2'] = (
            1.0 - np.sum((y - predicted) ** 2) / total
            if total > 0 else np.nan
        )
        rec['k_critical'] = float(-intercept / slope)
    return rec


def k_syn_cycle_branch(k_values=None, tf=30000.0, dt=20.0):
    """Continue the zero-command DD limit cycle upward in ``k_syn``.

    The final state at one value seeds the next. This directly tests whether
    the periodic orbit terminates at the equilibrium saddle-node or survives
    above it in a bistable window.
    """
    if k_values is None:
        k_values = np.unique(np.concatenate([
            np.arange(0.125, 0.136, 0.001),
            np.arange(0.136, 0.13901, 0.0001),
        ]))
    k_values = np.asarray(k_values, float)
    if np.any(np.diff(k_values) <= 0):
        raise ValueError("k_values must be strictly ascending")

    current = segment.drive_vector()
    dd = int(np.where(classes == 4)[0][0])
    saved = settings.k_syn
    rec = dict(k_values=k_values, amplitude=[], frequency=[], period=[],
               cv_period=[], oscillates=[])
    try:
        settings.k_syn = float(k_values[0])
        seed = segment.reduced_rest_state()
        for k_syn in k_values:
            settings.k_syn = float(k_syn)
            time = np.arange(0.0, tf, dt)
            sol = solve_ivp(
                lambda t, state: segment.rhs_vw(state, current),
                [0.0, tf],
                seed,
                method='BDF',
                t_eval=time,
            )
            seed = sol.y[:, -1]
            keep = time >= 0.5 * tf
            trace = sol.y[dd, keep]
            peaks, _ = find_peaks(trace, prominence=PROMINENCE)
            intervals = (
                np.diff(time[keep][peaks])
                if len(peaks) >= 3 else np.array([])
            )
            period = float(np.median(intervals)) if intervals.size else np.nan
            amplitude = float(np.ptp(trace))
            oscillates = bool(amplitude >= PROMINENCE and intervals.size >= 2)
            rec['amplitude'].append(amplitude)
            rec['period'].append(period)
            rec['frequency'].append(
                1000.0 / period if np.isfinite(period) else np.nan
            )
            rec['cv_period'].append(
                float(np.std(intervals) / np.mean(intervals))
                if intervals.size and np.mean(intervals) > 0 else np.nan
            )
            rec['oscillates'].append(oscillates)
    finally:
        settings.k_syn = saved

    for key in ('amplitude', 'frequency', 'period', 'cv_period', 'oscillates'):
        rec[key] = np.asarray(rec[key])

    active = np.where(rec['oscillates'])[0]
    inactive = np.where(~rec['oscillates'])[0]
    rec['k_cycle_end'] = np.nan
    if active.size and inactive.size:
        first_inactive = inactive[inactive > active[-1]]
        if first_inactive.size:
            rec['k_cycle_end'] = float(
                0.5 * (
                    k_values[active[-1]]
                    + k_values[first_inactive[0]]
                )
            )
    return rec


def k_syn_trace(k_syn, tf=30000.0, dt=20.0):
    """Return the steady zero-command DD trace for one synaptic steepness."""
    saved = settings.k_syn
    try:
        settings.k_syn = float(k_syn)
        time = np.arange(0.0, tf, dt)
        sol = solve_ivp(
            lambda t, state: segment.rhs_vw(
                state, segment.drive_vector()
            ),
            [0.0, tf],
            segment.reduced_rest_state(),
            method='BDF',
            t_eval=time,
        )
        keep = time >= tf - 5000.0
        dd = int(np.where(classes == 4)[0][0])
        return time[keep], sol.y[dd, keep]
    finally:
        settings.k_syn = saved


def analyze_k_syn(verbose=True):
    """Classify the zero-command transition caused by shallower synapses."""
    branch = k_syn_equilibrium_branch(
        np.arange(0.30, 0.12 - 1e-12, -0.00005)
    )
    k_critical = branch['k_critical']
    cycle = k_syn_cycle_branch()
    traces = {
        0.25: k_syn_trace(0.25),
        0.125: k_syn_trace(0.125),
    }
    result = dict(branch=branch, cycle=cycle, traces=traces)
    if verbose:
        print("\n=== k_syn sweep at zero command drive ===")
        print(
            f"  saddle-node: k_syn = {k_critical:.5f}  "
            f"(real eigenvalue -> 0; fit R^2={branch['fold_fit_r2']:.3f})"
        )
        active = cycle['oscillates']
        print(
            f"  periodic orbit ends near k_syn = "
            f"{cycle['k_cycle_end']:.5f}; at its last resolved point: "
            f"amplitude={cycle['amplitude'][active][-1]:.1f} mV, "
            f"period={cycle['period'][active][-1]:.0f} ms"
        )
        print(
            "  => SNIC in k_syn: the finite-amplitude periodic orbit "
            "terminates where the equilibrium has a saddle-node, with its "
            "period increasing toward the collision"
        )
        print(
            "  k_syn=0.125 is below the SNIC and therefore produces "
            "spontaneous DD oscillation without AVA/AVB drive"
        )
    return result


def plot_k_syn_analysis(result, save=False):
    """Collaborator-ready summary of the zero-command k_syn SNIC."""
    branch = result['branch']
    cycle = result['cycle']
    k_critical = branch['k_critical']

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    for k_syn, (time, trace) in result['traces'].items():
        ax.plot((time - time[0]) / 1000.0, trace, lw=1.0,
                label=fr"$k_{{syn}}={k_syn}$")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("DD voltage (mV)")
    ax.set_title("Undriven segment: quiescence vs spontaneous rhythm")
    ax.legend()

    ax = axes[0, 1]
    exists = branch['exists']
    ax.plot(branch['k_values'][exists], branch['V_DD'][exists],
            color='0.2', lw=1.2)
    ax.axvline(k_critical, color='crimson', ls='--',
               label=fr"saddle-node $k_c={k_critical:.5f}$")
    ax.invert_xaxis()
    ax.set_xlabel(r"$k_{syn}$ (steep $\rightarrow$ shallow)")
    ax.set_ylabel("equilibrium DD voltage (mV)")
    ax.set_title("The zero-drive equilibrium terminates")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(branch['k_values'][exists], branch['max_re'][exists],
            color='0.2', lw=1.2)
    ax.axhline(0.0, color='crimson', lw=0.8)
    ax.axvline(k_critical, color='crimson', ls='--')
    ax.invert_xaxis()
    ax.set_xlabel(r"$k_{syn}$")
    ax.set_ylabel(r"max Re$(\lambda)$ (1/ms)")
    ax.set_title("A real eigenvalue approaches zero")

    ax = axes[1, 1]
    active = cycle['oscillates']
    ax.plot(cycle['k_values'][active], cycle['period'][active],
            'o-', color='steelblue', ms=3, label='period')
    ax.axvline(k_critical, color='crimson', ls='--',
               label=fr"equilibrium fold $k_c={k_critical:.5f}$")
    ax.axvline(cycle['k_cycle_end'], color='darkorange', ls=':',
               label=fr"cycle end $\approx{cycle['k_cycle_end']:.5f}$")
    ax.set_xlabel(r"$k_{syn}$")
    ax.set_ylabel("DD period (ms)")
    ax.set_title("The periodic orbit slows and terminates at the fold")
    ax.legend(fontsize=8)

    fig.suptitle(
        "Full 17-cell segment: shallower synaptic gating creates a "
        "zero-command SNIC",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if save:
        filename = os.path.join(
            MEDIA_DIR, 'bifurcation_k_syn_zero_drive.png'
        )
        fig.savefig(filename, dpi=180)
        print(f"Saved → {filename}")
    plt.show()


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
        ax.axhline(0.0, color='forestgreen', ls=':', lw=0.7, alpha=0.6)
        ax.set_ylabel(r"$\Delta_{DV}$ envelope (mV)")
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
        ax.set_ylabel(r"$\Delta_{DV}$ frequency (Hz)")
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
        ax.axhline(0.0, color='forestgreen', ls=':', lw=0.7, alpha=0.6)
        ax.set_ylabel(r"$\Delta_{DV}$ envelope (mV)")
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
        ax.set_ylabel(r"$\Delta_{DV}$ frequency (Hz)")
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


def plot_beta_stability(results, save=False):
    """β-sweep stability diagram at fixed command drive (one column per branch),
    all rows sharing the β axis.

    Row 0 — the fixed point's driven-cell mean V (solid = stable, dashed =
      unstable) with the simulated oscillation envelope; the Hopf, the
      fold-of-cycles onset, and the bistable window between them are marked.
    Row 1 — leading-eigenvalue real part vs β: a complex pair crossing 0 is the
      (subcritical) Hopf.
    Row 2 — oscillation frequency (FFT fundamental) vs β."""
    fig, axes = plt.subplots(3, len(results), figsize=(6.2 * len(results), 11),
                             squeeze=False)

    for col, res in enumerate(results):
        brn = res['branch']
        b = res['betas']
        fp, env = res['fp'], res['env']
        bH, bO = res['beta_hopf'], res['beta_onset']
        bistable = (not np.isnan(bH) and not np.isnan(bO) and bO < bH)

        def mark(ax):
            if bistable:
                ax.axvspan(bO, bH, color='gold', alpha=0.2,
                           label='bistable window')
            if not np.isnan(bO):
                ax.axvline(bO, color='crimson', ls='--', lw=1.1,
                           label=f'cycle onset  β={bO:.2f}')
            if not np.isnan(bH):
                ax.axvline(bH, color='darkorange', ls='--', lw=1.1,
                           label=f'Hopf  β={bH:.2f}')

        # Row 0 — fixed-point branch + oscillation envelope
        ax = axes[0][col]
        osc = env['oscillates']
        ax.fill_between(b, env['vmin'], env['vmax'], where=osc,
                        color='steelblue', alpha=0.22, label='oscillation (sim)')
        ax.plot(b, np.where(osc, env['vmax'], np.nan), color='steelblue', lw=1.0)
        ax.plot(b, np.where(osc, env['vmin'], np.nan), color='steelblue', lw=1.0)
        _plot_equilibrium_branch(ax, fp, color='k',
                                 label_stable='fixed point (stable)',
                                 label_unstable='fixed point (unstable)')
        mark(ax)
        ax.axhline(0.0, color='forestgreen', ls=':', lw=0.7, alpha=0.6)
        ax.set_ylabel(r"$\Delta_{DV}$ (mV)")
        ax.set_title(f"{brn} branch  (I{brn} = {res['drive']} pA fixed)")
        ax.legend(fontsize=7, loc='best')

        # Row 1 — fixed-point leading eigenvalue real part
        ax = axes[1][col]
        m = fp['exists']
        ax.plot(b[m], fp['max_re'][m], '.-', ms=3, lw=0.9, color='0.25')
        ax.axhline(0.0, color='crimson', lw=0.8)
        if not np.isnan(bH):
            ax.axvline(bH, color='darkorange', ls='--', lw=1.1)
        ax.set_ylabel("max Re(eigenvalue)  (1/ms)")
        ax.set_xlabel("recovery slope β (nS/mV)")
        ax.set_title("fixed-point stability (Hopf where a complex pair Re→0)")

        # Row 2 — oscillation frequency vs β
        ax = axes[2][col]
        good = np.isfinite(env['freq'])
        ax.plot(b[good], env['freq'][good], 'o-', color='steelblue', ms=3, lw=0.9)
        mark(ax)
        ax.set_ylim(bottom=0)
        ax.set_ylabel("oscillation frequency f (Hz, FFT)")
        ax.set_xlabel("recovery slope β (nS/mV)")
        ax.set_title("frequency vs β")

    fig.suptitle(
        "Single-segment β-sweep at fixed command drive (reduced (V,w) analysis)\n"
        f"G_syne={settings.G_syne}, G_syni={settings.G_syni}, G_gap={settings.G_gap} nS  |  "
        f"tau_w={settings.tau_w} ms",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save:
        fname = os.path.join(MEDIA_DIR, 'bifurcation_beta_stability.png')
        plt.savefig(fname, dpi=150)
        print(f"Saved → {fname}")
    plt.show()


def _plot_equilibrium_branch(ax, rec, **kw):
    """Plot an equilibrium branch: solid where stable, dashed where unstable.
    The swept axis is rec['drives'] (drive sweep) or rec['betas'] (β sweep)."""
    x = rec['betas'] if 'betas' in rec else rec['drives']
    V, nu = rec['activity'], rec['n_unstable']
    ex = rec['exists']
    stable = ex & (nu == 0)
    unstab = ex & (nu > 0)
    # break the line wherever stability flips or the branch has a gap
    for mask, ls, lbl in ((stable, '-', kw.get('label_stable')),
                          (unstab, '--', kw.get('label_unstable'))):
        seg = np.where(mask, V, np.nan)
        ax.plot(x, seg, ls=ls, color=kw.get('color', 'k'), lw=1.6, label=lbl)


def plot_stability(results, save=False):
    """Stability / bifurcation diagram for one or more branches (one column
    each), all three rows sharing the drive axis.

    Row 0 — bifurcation diagram: the equilibrium's driven-cell mean voltage
      (solid = stable, dashed = unstable) with the simulated oscillation
      envelope overlaid; SNIC and Hopf marked.
    Row 1 — equilibrium stability: leading-eigenvalue real part vs drive. A real
      eigenvalue rising to 0 where the branch ends is the saddle-node (SNIC); a
      complex pair crossing 0 is the Hopf.
    Row 2 — onset frequency vs drive: near a SNIC the frequency falls to 0 as
      the drive approaches I_SN (the period diverges), following f ∝ √(I−I_SN).
      A Hopf would instead switch on at a finite frequency."""
    fig, axes = plt.subplots(3, len(results), figsize=(6.2 * len(results), 11),
                             squeeze=False)

    for col, res in enumerate(results):
        brn = res['branch']
        I_SN, I_hopf = res['I_SN'], res['I_hopf']
        env, rest_rec, up_rec = res['env'], res['rest_rec'], res['upper_rec']
        sc = res['scaling']
        xmax = float(env['drives'].max())
        has_upper = bool(up_rec['exists'].any())

        def mark_edges(ax):
            if not np.isnan(I_SN):
                ax.axvline(I_SN, color='crimson', ls='--', lw=1.1)
            if not np.isnan(I_hopf):
                ax.axvline(I_hopf, color='darkorange', ls='--', lw=1.1)

        # Row 0 — equilibrium branches + oscillation envelope
        ax = axes[0][col]
        osc = env['oscillates']
        ax.fill_between(env['drives'], env['vmin'], env['vmax'], where=osc,
                        color='steelblue', alpha=0.22, label='oscillation (sim)')
        ax.plot(env['drives'], np.where(osc, env['vmax'], np.nan),
                color='steelblue', lw=1.0)
        ax.plot(env['drives'], np.where(osc, env['vmin'], np.nan),
                color='steelblue', lw=1.0)
        _plot_equilibrium_branch(ax, rest_rec, color='k',
                                 label_stable='equilibrium (stable)',
                                 label_unstable='equilibrium (unstable)')
        _plot_equilibrium_branch(ax, up_rec, color='k')
        if not np.isnan(I_SN):
            ax.axvline(I_SN, color='crimson', ls='--', lw=1.1,
                       label=f'SNIC  {I_SN:.2f} pA')
        if not np.isnan(I_hopf):
            ax.axvline(I_hopf, color='darkorange', ls='--', lw=1.1,
                       label=f'Hopf  {I_hopf:.2f} pA')
        if not has_upper and not np.isnan(I_SN):
            ax.axvspan(I_SN, xmax, color='0.92', zorder=0,
                       label='no stable upper\nequilibrium located')
        ax.axhline(0.0, color='forestgreen', ls=':', lw=0.7, alpha=0.6)
        ax.set_xlim(0, xmax)
        ax.set_ylabel(r"$\Delta_{DV}$ (mV)")
        ax.set_title(f"{brn} branch  (I{brn} swept, other = 0)")
        ax.legend(fontsize=7, loc='best')

        # Row 1 — leading eigenvalue real part
        ax = axes[1][col]
        for rec, lbl in ((rest_rec, 'rest branch'), (up_rec, 'upper branch')):
            m = rec['exists']
            if m.any():
                ax.plot(rec['drives'][m], rec['max_re'][m], '.-', ms=3, lw=0.9,
                        color='0.25', label=lbl)
        ax.axhline(0.0, color='crimson', lw=0.8)
        mark_edges(ax)
        ax.set_xlim(0, xmax)
        if not has_upper and not np.isnan(I_SN):
            ax.axvspan(I_SN, xmax, color='0.92', zorder=0)
            ymid = np.mean(ax.get_ylim())
            ax.text(0.5 * (I_SN + xmax), ymid,
                    "no stable upper equilibrium located\n"
                    "(oscillation persists in simulation)",
                    ha='center', va='center', fontsize=8, color='0.35')
        ax.set_ylabel("max Re(eigenvalue)  (1/ms)")
        ax.set_xlabel(f"I{brn} drive (pA)")
        ax.set_title("equilibrium stability")
        ax.legend(fontsize=7, loc='best')

        # Row 2 — onset frequency vs drive (period divergence at the SNIC)
        ax = axes[2][col]
        good = np.isfinite(env['freq'])
        ax.plot(env['drives'][good], env['freq'][good], 'o-', color='steelblue',
                ms=3, lw=0.9, label='frequency (sim)')
        if sc is not None and sc['dI'].size >= 2:
            # accurate near-onset points from the long integrations
            ax.plot(sc['I_SN'] + sc['dI'], sc['freq'], 's', color='crimson',
                    ms=4, label='near-onset (long run)')
            # sqrt-law guide anchored to the closest-to-onset measured point,
            # drawn only across the fitted near-onset window (the sqrt law is an
            # onset asymptote; higher-order terms bend f below it further out).
            k = sc['freq'][0] / np.sqrt(sc['dI'][0])
            xs = np.linspace(0, sc['dI'].max(), 200)
            ax.plot(sc['I_SN'] + xs, k * np.sqrt(xs), '--', color='crimson',
                    lw=1.0, label='f ∝ √(I − I_SN)  (onset)')
        mark_edges(ax)
        ax.set_xlim(0, xmax)
        ax.set_ylim(bottom=0)
        ax.set_ylabel("oscillation frequency f (Hz)")
        ax.set_xlabel(f"I{brn} drive (pA)")
        ax.set_title("onset: f → 0 at SNIC (period diverges)")
        ax.legend(fontsize=7, loc='best')

    fig.suptitle(
        "Single-segment bifurcation type & location (reduced (V,w) equilibrium analysis)\n"
        f"G_syne={settings.G_syne}, G_syni={settings.G_syni}, G_gap={settings.G_gap} nS  |  "
        f"beta={settings.beta}, tau_w={settings.tau_w} ms",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save:
        fname = os.path.join(MEDIA_DIR, 'bifurcation_stability_eigen.png')
        plt.savefig(fname, dpi=150)
        print(f"Saved → {fname}")
    plt.show()


if __name__ == '__main__':
    # MODE: 'beta'        -> fix command drive, sweep recovery slope beta with the
    #                        reduced (V,w) analysis (fixed-point Hopf + bistable window)
    #       'drive'       -> fix beta, sweep command drive with up/down hysteresis
    #       'bifurcation' -> fix beta, sweep command drive from rest (no hysteresis)
    #       'eigen'       -> locate & type the bifurcations via the reduced (V,w)
    #                        equilibrium/eigenvalue analysis (SNIC + upper Hopf)
    #       'k_syn'       -> zero-command continuation in synaptic steepness
    MODE  = 'bifurcation'
    DRIVE = 2.5          # pA, the active command-interneuron current (try 2.0 or 2.5)

    if MODE == 'k_syn':
        result = analyze_k_syn()
        plot_k_syn_analysis(result, save=True)

    elif MODE == 'eigen':
        res_ava = analyze_stability('AVA')
        res_avb = analyze_stability('AVB')
        plot_stability([res_ava, res_avb], save=True)

    elif MODE == 'beta':
        betas = np.arange(0.8, 2.001, 0.02)
        res_ava = analyze_beta('AVA', drive=DRIVE, betas=betas)
        res_avb = analyze_beta('AVB', drive=DRIVE, betas=betas)
        plot_beta_stability([res_ava, res_avb], save=False)

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

        plot_bifurcation([rec_ava, rec_avb], save=False)

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
