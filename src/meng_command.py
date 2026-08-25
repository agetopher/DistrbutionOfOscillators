"""AVA-master command dynamics adapted from Meng et al. (2024).

The published calcium-activity model is

    x_AVB = a*x_AVA + b*y
    dy/dt = -y/tau + x_AVA,

with fast inhibition (a < 0) and slow excitation (b > 0). Calcium activity is
not a current, so this module keeps the paper coefficients in activity units
and makes the separate activity-to-pA calibration explicit.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MengCommand:
    """Published wild-type dynamics plus an explicit motor-current calibration."""

    a: float = -0.51
    b: float = 0.02
    tau_s: float = 3.5
    ava_tonic: float = 1.0
    ava_phasic: float = 3.0
    iava_peak_pa: float = 2.5
    iavb_tonic_pa: float = 2.5

    @property
    def slow_tonic(self):
        """Steady slow state at tonic AVA activity."""
        return self.tau_s * self.ava_tonic

    @property
    def avb_scale_pa(self):
        """Scale making the instantaneous phasic AVA jump cancel tonic AVB."""
        fast_drop = -self.a * (self.ava_phasic - self.ava_tonic)
        return self.iavb_tonic_pa / fast_drop

    def currents(self, ava_activity, slow_state):
        """Map centered activity deviations to non-negative command currents."""
        span = self.ava_phasic - self.ava_tonic
        ava_fraction = np.clip((ava_activity - self.ava_tonic) / span, 0.0, 1.0)
        avb_modulation = (
            self.a * (ava_activity - self.ava_tonic)
            + self.b * (slow_state - self.slow_tonic)
        )
        iava = self.iava_peak_pa * ava_fraction
        iavb = np.maximum(
            0.0,
            self.iavb_tonic_pa + self.avb_scale_pa * avb_modulation,
        )
        return iava, iavb

    def slow_derivative_per_ms(self, ava_activity, slow_state):
        """Paper's slow-state derivative converted from seconds to milliseconds."""
        return (-slow_state / self.tau_s + ava_activity) / 1000.0
