"""Cycle-resolved frequency-locking diagnostics for segment voltage traces."""

import numpy as np
from scipy.signal import find_peaks


def reference_peak_times(time, reference, prominence=10.0):
    """Return reference-cycle peak times and their mean period in milliseconds."""
    time = np.asarray(time, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if time.ndim != 1 or time.size < 3:
        raise ValueError("time must be a one-dimensional array with >= 3 samples")
    if reference.shape != time.shape:
        raise ValueError("reference must have the same shape as time")
    if np.any(np.diff(time) <= 0):
        raise ValueError("time must be strictly increasing")

    peaks, _ = find_peaks(reference, prominence=prominence)
    if peaks.size < 4:
        raise ValueError("reference must contain at least four prominent cycles")
    peak_times = time[peaks]
    return peak_times, float(np.mean(np.diff(peak_times)))


def _cycle_windows(reference_peaks):
    """Return complete peak-centered cycle windows and their anchors."""
    reference_peaks = np.asarray(reference_peaks, dtype=float)
    boundaries = 0.5 * (reference_peaks[:-1] + reference_peaks[1:])
    return boundaries[:-1], boundaries[1:], reference_peaks[1:-1]


def _lag_correlation(voltage, lag):
    left = voltage[:-lag]
    right = voltage[lag:]
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return np.nan
    return float(np.corrcoef(left, right)[0, 1])


def cell_locking_metrics(
    time,
    voltage,
    reference_peaks,
    min_amplitude_mv=5.0,
    peak_prominence_mv=5.0,
    min_lag_correlation=0.98,
    max_relative_frequency_error=0.01,
):
    """Measure whether one voltage trace is 1:1 locked to reference cycles.

    One representative voltage maximum is selected in every reference-cycle
    window. This estimates the fundamental phase without mistaking the two
    within-cycle peaks of DD/VD waveforms for a doubled oscillator frequency.
    Raw peak multiplicity is reported separately as ``peaks_per_cycle``.
    """
    time = np.asarray(time, dtype=float)
    voltage = np.asarray(voltage, dtype=float)
    reference_peaks = np.asarray(reference_peaks, dtype=float)
    if voltage.shape != time.shape:
        raise ValueError("voltage must have the same shape as time")

    starts, stops, anchors = _cycle_windows(reference_peaks)
    if anchors.size < 2:
        raise ValueError("at least four reference peaks are required")

    representative_peaks = []
    cycle_amplitudes = []
    for start, stop in zip(starts, stops):
        mask = (time >= start) & (time < stop)
        if not np.any(mask):
            raise ValueError("reference cycle contains no time samples")
        cycle_time = time[mask]
        cycle_voltage = voltage[mask]
        representative_peaks.append(cycle_time[np.argmax(cycle_voltage)])
        cycle_amplitudes.append(float(np.ptp(cycle_voltage)))

    representative_peaks = np.asarray(representative_peaks)
    local_periods = 0.5 * (
        np.diff(reference_peaks[:-1]) + np.diff(reference_peaks[1:])
    )
    phase_offsets = (representative_peaks - anchors) / local_periods
    phase_angles = 2.0 * np.pi * phase_offsets
    phase_concentration = float(np.abs(np.mean(np.exp(1j * phase_angles))))
    unwrapped_phase = np.unwrap(phase_angles) / (2.0 * np.pi)
    phase_drift = float(np.polyfit(np.arange(anchors.size), unwrapped_phase, 1)[0])

    reference_period = float(np.mean(np.diff(reference_peaks)))
    reference_frequency = 1000.0 / reference_period

    dt = float(np.median(np.diff(time)))
    lag_samples = int(round(reference_period / dt))
    if lag_samples <= 0 or lag_samples >= time.size:
        raise ValueError("reference period is outside the sampled time range")
    lag_correlation = _lag_correlation(voltage, lag_samples)
    candidate_lags = np.arange(
        max(1, int(round(0.8 * reference_period / dt))),
        min(time.size - 1, int(round(1.2 * reference_period / dt))) + 1,
    )
    candidate_correlations = np.array(
        [
            _lag_correlation(voltage, lag)
            for lag in candidate_lags
        ]
    )
    if np.any(np.isfinite(candidate_correlations)):
        best_lag = int(candidate_lags[np.nanargmax(candidate_correlations)])
        cell_period = best_lag * dt
        cell_frequency = 1000.0 / cell_period
        relative_frequency_error = (
            abs(cell_frequency - reference_frequency) / reference_frequency
        )
    else:
        cell_frequency = np.nan
        relative_frequency_error = np.nan

    raw_peaks, _ = find_peaks(voltage, prominence=peak_prominence_mv)
    complete_start, complete_stop = starts[0], stops[-1]
    raw_peaks = raw_peaks[
        (time[raw_peaks] >= complete_start) & (time[raw_peaks] < complete_stop)
    ]
    peaks_per_cycle = float(raw_peaks.size / anchors.size)

    amplitude = float(np.ptp(voltage))
    active = amplitude >= min_amplitude_mv
    locked = bool(
        active
        and lag_correlation >= min_lag_correlation
        and relative_frequency_error <= max_relative_frequency_error
    )
    return {
        "active": bool(active),
        "locked_1_to_1": locked,
        "amplitude_mv": amplitude,
        "median_cycle_amplitude_mv": float(np.median(cycle_amplitudes)),
        "frequency_hz": cell_frequency if active else np.nan,
        "reference_frequency_hz": reference_frequency,
        "relative_frequency_error": relative_frequency_error if active else np.nan,
        "phase_concentration": phase_concentration if active else np.nan,
        "phase_drift_cycles_per_cycle": phase_drift if active else np.nan,
        "lag_correlation": lag_correlation if active else np.nan,
        "peaks_per_cycle": peaks_per_cycle if active else 0.0,
        "cycles_analyzed": int(anchors.size),
    }
