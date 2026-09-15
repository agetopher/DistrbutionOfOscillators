import numpy as np

import segment
import settings


def setup_module():
    settings.reset_defaults()
    segment.configure()


def test_ablated_cell_state_is_clamped():
    state = segment.reduced_rest_state()
    current = segment.drive_along("AVA", 3.0)

    derivative = segment.rhs_vw(state, current, ablated_indices=[10])

    assert derivative[10] == 0.0
    assert derivative[segment.N_CELLS + 10] == 0.0


def test_ablated_cell_cannot_influence_remaining_cells():
    state_a = segment.reduced_rest_state()
    state_b = state_a.copy()
    state_b[10] = 20.0
    state_b[segment.N_CELLS + 10] = 5.0
    current = segment.drive_along("AVA", 3.0)

    derivative_a = segment.rhs_vw(state_a, current, ablated_indices=[10])
    derivative_b = segment.rhs_vw(state_b, current, ablated_indices=[10])
    remaining = np.arange(2 * segment.N_CELLS)
    remaining = remaining[(remaining != 10) & (remaining != segment.N_CELLS + 10)]

    np.testing.assert_allclose(derivative_a[remaining], derivative_b[remaining])


def test_empty_ablation_matches_baseline_rhs():
    state = segment.reduced_rest_state()
    current = segment.drive_along("AVB", 2.5)

    baseline = segment.rhs_vw(state, current)
    explicit_empty = segment.rhs_vw(state, current, ablated_indices=[])

    np.testing.assert_allclose(baseline, explicit_empty)
