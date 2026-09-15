import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import whole_network_va10_vb8_ablation as experiment
import network
import settings


def setup_function():
    settings.reset_defaults()
    network.configure()


def test_repeated_ablation_indices_cover_all_six_segments():
    np.testing.assert_array_equal(
        experiment.ablation_indices("VA10"),
        [10, 27, 44, 61, 78, 95],
    )
    np.testing.assert_array_equal(
        experiment.ablation_indices("VB8"),
        [8, 25, 42, 59, 76, 93],
    )


def test_exact_rest_state_is_uniform_with_zero_recovery():
    state = experiment.exact_rest_state()
    np.testing.assert_array_equal(state[: network.N_CELLS], settings.L)
    np.testing.assert_array_equal(state[network.N_CELLS :], 0.0)


def test_ablated_cells_are_clamped_and_do_not_affect_rhs_shape():
    ablated = experiment.ablation_indices("VA10")
    state = experiment.exact_rest_state()
    derivative = experiment.rhs(state, network.drive_vector(IAVA=2.5), ablated)
    assert derivative.shape == state.shape
    np.testing.assert_array_equal(derivative[ablated], 0.0)
    np.testing.assert_array_equal(derivative[network.N_CELLS + ablated], 0.0)
