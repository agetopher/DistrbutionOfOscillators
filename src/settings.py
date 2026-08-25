"""Shared baseline parameters for Distribution of Oscillators experiments.

The defaults represent the current robust single-segment/full-network baseline:

* the Yuval piecewise membrane model;
* the command-driven conductances used for the travelling-wave result;
* ``beta=1.03``, just beyond the beta-bistable window identified by the
  reduced-system bifurcation analysis.

Experiments may override individual module attributes after importing this
module. Call :func:`reset_defaults` before configuring a new experiment in the
same Python process so it does not inherit changes made by an earlier one.

Connectivity arrays and applied-current vectors are circuit-specific and
therefore start as ``None``. Set ``numCells`` and those arrays when constructing
a multi-cell circuit.
"""

# Membrane model ---------------------------------------------------------------

MEMBRANE_DEFAULTS = {
    "C": 7.0,          # pF, membrane capacitance
    "g": 1.0,          # nS, membrane conductance scale
    "L": -70.0,        # mV, resting potential
    "T": -45.0,        # mV, depolarization threshold
    "H": -35.0,        # mV, plateau potential
    "m1": 0.7,
    "m2": 1.0 / 81.0,
    "m3": -1.0 / 30.0,
    "m4": 0.17,
}

# Chemical and electrical coupling --------------------------------------------

SYNAPSE_DEFAULTS = {
    "G_syne": 0.07,    # nS, excitatory synaptic conductance
    "E_syne": 0.0,     # mV, excitatory reversal potential
    "G_syni": 0.05,    # nS, inhibitory synaptic conductance
    "E_syni": -100.0,  # mV, inhibitory reversal potential
    "k_syn": 0.25,     # 1/mV, sigmoid steepness
    "V_th": -52.0,     # mV, sigmoid half-activation voltage
    "G_gap": 0.03,     # nS, gap-junction conductance
}

# Slow variables ---------------------------------------------------------------

RECOVERY_DEFAULTS = {
    "beta": 1.03,      # nS/mV, recovery-target slope
    "tau_w": 400.0,    # ms, recovery time scale
    "a": 0.000035,     # synaptic-fatigue depletion rate
    "b": 0.005,        # synaptic-fatigue recovery rate
}

DEFAULTS = {
    **MEMBRANE_DEFAULTS,
    **SYNAPSE_DEFAULTS,
    **RECOVERY_DEFAULTS,
}


def reset_defaults():
    """Restore all scalar parameters and clear circuit-specific state."""
    globals().update(DEFAULTS)

    global numCells, E_conn, I_conn, GJ_conn, I_inj
    numCells = 1
    E_conn = None
    I_conn = None
    GJ_conn = None
    I_inj = None


def init():
    """Backward-compatible alias for :func:`reset_defaults`."""
    reset_defaults()


reset_defaults()
