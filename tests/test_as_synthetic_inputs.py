import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from as_synthetic_inputs import normalized_waveform


@pytest.mark.parametrize("kind", ("sine", "square", "slow_pulse"))
def test_synthetic_waveforms_are_bounded_and_zero_mean(kind):
    time = np.linspace(0.0, 1000.0, 10000, endpoint=False)
    waveform = normalized_waveform(kind, time, 1.0)

    assert waveform.min() >= -1.0
    assert waveform.max() <= 1.0
    assert np.mean(waveform) == pytest.approx(0.0, abs=2e-4)


def test_unknown_waveform_is_rejected():
    with pytest.raises(ValueError, match="Unknown waveform"):
        normalized_waveform("triangle", np.array([0.0]), 1.0)
