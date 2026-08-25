import numpy as np

from meng_command import MengCommand


def test_tonic_ava_maps_to_forward_command_baseline():
    command = MengCommand()
    iava, iavb = command.currents(command.ava_tonic, command.slow_tonic)

    assert iava == 0.0
    assert iavb == command.iavb_tonic_pa
    assert command.slow_derivative_per_ms(
        command.ava_tonic, command.slow_tonic
    ) == 0.0


def test_phasic_ava_immediately_suppresses_avb():
    command = MengCommand()
    iava, iavb = command.currents(command.ava_phasic, command.slow_tonic)

    assert iava == command.iava_peak_pa
    assert np.isclose(iavb, 0.0)


def test_slow_ava_history_excites_avb_after_event():
    command = MengCommand()
    elevated_slow_state = command.slow_tonic + 1.0
    _, iavb = command.currents(command.ava_tonic, elevated_slow_state)

    assert iavb > command.iavb_tonic_pa
