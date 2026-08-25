import numpy as np

import segment
import settings


def setup_function():
    settings.reset_defaults()
    segment.configure()


def test_command_drives_target_expected_cell_classes():
    current = segment.drive_vector(IAVA=2.0, IAVB=3.0)

    expected = np.zeros(segment.N_CELLS)
    expected[np.isin(segment.CLASSES, segment.AVA_CLASSES)] += 2.0
    expected[np.isin(segment.CLASSES, segment.AVB_CLASSES)] += 3.0

    np.testing.assert_array_equal(current, expected)


def test_unknown_command_branch_is_rejected():
    try:
        segment.drive_along("UNKNOWN", 2.5)
    except ValueError as error:
        assert "UNKNOWN" in str(error)
    else:
        raise AssertionError("Unknown command branch should raise ValueError")


def test_full_and_reduced_segment_dynamics_agree_when_fatigue_is_inert():
    current = segment.drive_vector(IAVA=2.5)
    full_state = segment.full_rest_state()
    reduced_state = segment.reduced_rest_state()

    full_derivative = segment.rhs_full(0.0, full_state, current)
    reduced_derivative = segment.rhs_vw(reduced_state, current)

    np.testing.assert_allclose(
        full_derivative[: segment.N_CELLS],
        reduced_derivative[: segment.N_CELLS],
    )
    np.testing.assert_allclose(
        full_derivative[2 * segment.N_CELLS :],
        reduced_derivative[segment.N_CELLS :],
    )
