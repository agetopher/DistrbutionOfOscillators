import numpy as np

from frequency_locking import cell_locking_metrics, reference_peak_times


def test_phase_shifted_trace_is_frequency_locked():
    time = np.arange(0.0, 20000.0, 2.0)
    omega = 2.0 * np.pi / 1000.0
    reference = 20.0 * np.cos(omega * time)
    voltage = 18.0 * np.cos(omega * time - 0.7)

    peaks, _ = reference_peak_times(time, reference)
    metrics = cell_locking_metrics(time, voltage, peaks)

    assert metrics["active"]
    assert metrics["locked_1_to_1"]
    assert abs(metrics["frequency_hz"] - 1.0) < 1e-3
    assert metrics["phase_concentration"] > 0.999


def test_two_peaks_per_cycle_do_not_imply_double_fundamental_frequency():
    time = np.arange(0.0, 20000.0, 2.0)
    omega = 2.0 * np.pi / 1000.0
    reference = 20.0 * np.cos(omega * time)
    voltage = 14.0 * np.cos(omega * time - 0.4) + 10.0 * np.cos(2.0 * omega * time)

    peaks, _ = reference_peak_times(time, reference)
    metrics = cell_locking_metrics(time, voltage, peaks, peak_prominence_mv=3.0)

    assert metrics["locked_1_to_1"]
    assert abs(metrics["frequency_hz"] - 1.0) < 1e-3
    assert metrics["peaks_per_cycle"] > 1.5


def test_silent_trace_is_not_classified_as_locked():
    time = np.arange(0.0, 20000.0, 2.0)
    omega = 2.0 * np.pi / 1000.0
    reference = 20.0 * np.cos(omega * time)
    voltage = -60.0 + 0.5 * np.cos(omega * time)

    peaks, _ = reference_peak_times(time, reference)
    metrics = cell_locking_metrics(time, voltage, peaks)

    assert not metrics["active"]
    assert not metrics["locked_1_to_1"]
    assert np.isnan(metrics["frequency_hz"])


def test_detuned_trace_fails_locking_test():
    time = np.arange(0.0, 30000.0, 2.0)
    omega = 2.0 * np.pi / 1000.0
    reference = 20.0 * np.cos(omega * time)
    voltage = 20.0 * np.cos(1.03 * omega * time)

    peaks, _ = reference_peak_times(time, reference)
    metrics = cell_locking_metrics(time, voltage, peaks)

    assert metrics["active"]
    assert not metrics["locked_1_to_1"]
