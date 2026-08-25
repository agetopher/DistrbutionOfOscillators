import numpy as np

from analysis import dorsoventral_activity, oscillation_metric, phase_difference


def test_dorsoventral_activity_averages_each_side_before_differencing():
    classes = np.array([8, 8, 9, 9])
    voltage = np.array([
        [2.0, 4.0],
        [4.0, 6.0],
        [1.0, 2.0],
        [1.0, 4.0],
    ])

    np.testing.assert_allclose(
        dorsoventral_activity(voltage, classes),
        np.array([2.0, 2.0]),
    )
    assert dorsoventral_activity(voltage[:, 0], classes) == 2.0


def test_dorsoventral_activity_requires_both_muscle_sides():
    with np.testing.assert_raises(ValueError):
        dorsoventral_activity(np.array([1.0, 2.0]), np.array([8, 8]))


def test_oscillation_metric_detects_regular_oscillation():
    t = np.linspace(0.0, 10.0, 1001)
    voltage = 20.0 * np.sin(2.0 * np.pi * t)

    result = oscillation_metric(t, voltage, transient_frac=0.0, prominence=10.0)

    assert result["oscillates"]
    assert np.isclose(result["mean_isi"], 1.0, atol=0.02)
    assert result["cv_isi"] < 0.02


def test_oscillation_metric_rejects_constant_trace():
    t = np.linspace(0.0, 10.0, 1001)
    result = oscillation_metric(t, np.zeros_like(t), transient_frac=0.0)

    assert not result["oscillates"]
    assert np.isnan(result["mean_isi"])
    assert np.isnan(result["cv_isi"])


def test_phase_difference_detects_antiphase():
    t = np.linspace(0.0, 10.0, 1001)
    first = 20.0 * np.sin(2.0 * np.pi * t)
    second = -first

    phase = phase_difference(
        t,
        first,
        second,
        transient_frac=0.0,
        prominence=10.0,
    )

    assert np.isclose(phase, 0.5, atol=0.02)
