import numpy as np
from scipy.signal import find_peaks


def dorsoventral_activity(voltage, classes):
    """Return mean dorsal-muscle voltage minus mean ventral-muscle voltage.

    Voltage may be a single state with shape (cells,) or a trajectory with
    shape (cells, time). Taking the instantaneous difference preserves
    antiphase activity that would cancel in a difference of time averages.
    """
    voltage = np.asarray(voltage, dtype=float)
    classes = np.asarray(classes)
    if voltage.ndim not in (1, 2):
        raise ValueError("voltage must have shape (cells,) or (cells, time)")
    if voltage.shape[0] != classes.size:
        raise ValueError("voltage cell axis and classes must have equal length")

    dorsal = classes == 8
    ventral = classes == 9
    if not dorsal.any() or not ventral.any():
        raise ValueError("classes must contain dorsal (8) and ventral (9) muscles")
    return voltage[dorsal].mean(axis=0) - voltage[ventral].mean(axis=0)


def oscillation_metric(t, V, transient_frac=0.3, prominence=10.0):
    """
    Assess oscillation quality from a single voltage trace.

    Parameters
    ----------
    t             : 1D array of time points
    V             : 1D array of voltage values
    transient_frac: fraction of the trace to discard as transient
    prominence    : minimum peak prominence (mV) to count as a spike

    Returns
    -------
    dict with:
      oscillates : bool   — True if >= 2 peaks detected after transient
      mean_isi   : float  — mean inter-spike interval (nan if no oscillation)
      cv_isi     : float  — coefficient of variation of ISI; lower = more regular
    """
    cutoff = int(transient_frac * len(t))
    t_ss = t[cutoff:]
    V_ss = V[cutoff:]

    peaks, _ = find_peaks(V_ss, prominence=prominence)
    if len(peaks) < 2:
        return dict(oscillates=False, mean_isi=np.nan, cv_isi=np.nan)

    isis = np.diff(t_ss[peaks])
    mean_isi = np.mean(isis)
    cv_isi = np.std(isis) / mean_isi if mean_isi > 0 else np.nan

    return dict(oscillates=True, mean_isi=mean_isi, cv_isi=cv_isi)


def phase_difference(t, V0, V1, transient_frac=0.3, prominence=10.0):
    """
    Estimate the normalised phase difference between two oscillating neurons.

    Returns a value in [0, 0.5]: 0 = in-phase, 0.5 = antiphase.
    Returns nan if either trace does not oscillate.
    """
    cutoff = int(transient_frac * len(t))
    t_ss = t[cutoff:]

    peaks0, _ = find_peaks(V0[cutoff:], prominence=prominence)
    peaks1, _ = find_peaks(V1[cutoff:], prominence=prominence)

    if len(peaks0) < 2 or len(peaks1) < 2:
        return np.nan

    period = np.mean(np.diff(t_ss[peaks0]))

    t0 = t_ss[peaks0]
    t1 = t_ss[peaks1]
    diffs = []
    for tp in t0:
        nearest = t1[np.argmin(np.abs(t1 - tp))]
        diffs.append(abs(tp - nearest) / period)

    phase = np.mean(diffs) % 1.0
    return min(phase, 1.0 - phase)   # fold to [0, 0.5]
