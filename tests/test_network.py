import numpy as np

import network
import settings


def setup_function():
    settings.reset_defaults()
    network.configure()


def test_command_drives_expected_full_network_classes():
    current = network.drive_vector(IAVA=2.0, IAVB=3.0)
    expected = np.zeros(network.N_CELLS)
    expected[np.isin(network.classes, network.AVA_CLASSES)] += 2.0
    expected[np.isin(network.classes, network.AVB_CLASSES)] += 3.0

    np.testing.assert_array_equal(current, expected)


def test_reduced_full_network_rhs_is_finite():
    derivative = network.rhs_vw(
        network.reduced_rest_state(),
        network.drive_vector(IAVB=3.0),
    )

    assert derivative.shape == (2 * network.N_CELLS,)
    assert np.all(np.isfinite(derivative))
