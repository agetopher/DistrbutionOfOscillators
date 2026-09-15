import numpy as np
import pytest

import segment
import settings


def setup_module():
    settings.reset_defaults()
    segment.configure()


def test_unity_voltage_multipliers_match_baseline_rhs():
    state = segment.reduced_rest_state()
    state[: segment.N_CELLS] += np.linspace(0.0, 30.0, segment.N_CELLS)
    current = segment.drive_along("AVA", 3.0)

    baseline = segment.rhs_vw(state, current)
    scaled = segment.rhs_vw(
        state,
        current,
        voltage_multipliers=np.ones(segment.N_CELLS),
    )

    np.testing.assert_allclose(baseline, scaled)


def test_voltage_scaling_is_centered_on_rest():
    voltage = np.array([settings.L, settings.L + 10.0] + [settings.L] * 15)
    multipliers = np.ones(segment.N_CELLS)
    multipliers[1] = 1.5

    effective = segment._scaled_coupling_voltage(voltage, multipliers)

    assert effective[0] == settings.L
    assert effective[1] == settings.L + 15.0


def test_negative_voltage_multiplier_is_rejected():
    multipliers = np.ones(segment.N_CELLS)
    multipliers[0] = -0.5

    with pytest.raises(ValueError, match="nonnegative"):
        segment._scaled_coupling_voltage(
            np.full(segment.N_CELLS, settings.L),
            multipliers,
        )


def test_unity_presynaptic_multipliers_match_baseline_rhs():
    state = segment.reduced_rest_state()
    state[: segment.N_CELLS] += np.linspace(0.0, 30.0, segment.N_CELLS)
    current = segment.drive_along("AVA", 3.0)

    baseline = segment.rhs_vw(state, current)
    scaled = segment.rhs_vw(
        state,
        current,
        presynaptic_multipliers=np.ones(segment.N_CELLS),
    )

    np.testing.assert_allclose(baseline, scaled)


def test_presynaptic_multiplier_scales_only_chemical_output():
    voltage = np.full(segment.N_CELLS, settings.L + 10.0)
    gating = np.linspace(0.1, 0.9, segment.N_CELLS)
    multipliers = np.ones(segment.N_CELLS)
    multipliers[0] = 1.5

    baseline_exc, baseline_inh, baseline_gap = segment._currents(voltage, gating)
    scaled_exc, scaled_inh, scaled_gap = segment._currents(
        voltage,
        gating,
        presynaptic_multipliers=multipliers,
    )
    expected_exc = (
        settings.G_syne
        * (segment.E_CONN @ (gating * multipliers))
        * (voltage - settings.E_syne)
    )

    np.testing.assert_allclose(scaled_exc, expected_exc)
    np.testing.assert_allclose(scaled_inh, baseline_inh)
    np.testing.assert_allclose(scaled_gap, baseline_gap)
    assert not np.allclose(scaled_exc, baseline_exc)


def test_negative_presynaptic_multiplier_is_rejected():
    multipliers = np.ones(segment.N_CELLS)
    multipliers[0] = -0.5

    with pytest.raises(ValueError, match="nonnegative"):
        segment._currents(
            np.full(segment.N_CELLS, settings.L),
            np.ones(segment.N_CELLS),
            presynaptic_multipliers=multipliers,
        )


def test_incoming_disconnection_removes_target_currents_but_keeps_output():
    voltage = np.linspace(settings.L, settings.H, segment.N_CELLS)
    gating = np.linspace(0.1, 0.9, segment.N_CELLS)
    disconnected = [0, 1]

    baseline_exc, baseline_inh, _ = segment._currents(voltage, gating)
    exc, inh, gap = segment._currents(
        voltage,
        gating,
        incoming_disconnected_indices=disconnected,
    )

    np.testing.assert_allclose(exc[disconnected], 0.0)
    np.testing.assert_allclose(inh[disconnected], 0.0)
    np.testing.assert_allclose(gap[disconnected], 0.0)
    np.testing.assert_allclose(exc[2:], baseline_exc[2:])
    np.testing.assert_allclose(inh[2:], baseline_inh[2:])
    assert baseline_exc[2] != 0.0
