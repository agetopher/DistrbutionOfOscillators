import numpy as np

import settings
from functions import f, f_vec, sf_vec, w_inf_vec


def setup_module():
    settings.reset_defaults()


def test_vectorized_membrane_equation_matches_scalar_equation():
    voltage = np.array([-80.0, -60.0, -40.0, -30.0])
    settings.numCells = voltage.size

    expected = np.array([f(value) for value in voltage])

    np.testing.assert_allclose(f_vec(voltage), expected)


def test_recovery_target_is_zero_below_threshold():
    voltage = np.array([-70.0, -45.0, -40.0])
    settings.numCells = voltage.size

    np.testing.assert_allclose(
        w_inf_vec(voltage),
        np.array([0.0, 0.0, settings.beta * 5.0]),
    )


def test_synaptic_fatigue_depletes_and_recovers():
    voltage = np.array([-60.0, -70.0, -70.0])
    fatigue = np.array([0.5, 0.5, 1.0])

    derivative = sf_vec(voltage, fatigue)

    assert derivative[0] < 0.0
    assert derivative[1] == settings.b
    assert derivative[2] == 0.0


def test_reset_defaults_restores_learned_baseline_and_clears_circuit_state():
    settings.beta = 0.2
    settings.G_gap = 0.0
    settings.numCells = 17
    settings.E_conn = np.eye(17)

    settings.reset_defaults()

    assert settings.beta == 1.03
    assert settings.G_syne == 0.07
    assert settings.G_syni == 0.05
    assert settings.G_gap == 0.03
    assert settings.k_syn == 0.25
    assert settings.tau_w == 400.0
    assert settings.numCells == 1
    assert settings.E_conn is None
