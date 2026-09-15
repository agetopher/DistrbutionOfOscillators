"""Reusable one-segment model for experiments and analyses.

This module owns the first 17-cell circuit, command-drive construction, and
both state representations used in the project:

* ``rhs_full`` for ``[V, SF, w]``;
* ``rhs_vw`` for the reduced ``[V, w]`` system while synaptic fatigue is inert.

Analysis-specific operations such as continuation and bifurcation
classification belong in ``experiments/`` rather than here.
"""

from pathlib import Path

import numpy as np

import settings
from functions import f_vec, sf_vec, w_inf_vec


N_CELLS = 17
AVA_CLASSES = (1, 2, 7)  # AS, DA, VA
AVB_CLASSES = (1, 3, 6)  # AS, DB, VB

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Connectivity files use row=post, col=pre.
E_CONN = np.loadtxt(
    DATA_DIR / "ConnectivityMatrix_SixSegments_ExcitatorySynapses.txt",
    delimiter=",",
)[:N_CELLS, :N_CELLS]
I_CONN = np.loadtxt(
    DATA_DIR / "ConnectivityMatrix_SixSegments_InhibitorySynapses.txt",
    delimiter=",",
)[:N_CELLS, :N_CELLS]
GJ_CONN = np.loadtxt(
    DATA_DIR / "ConnectivityMatrix_SixSegments_GapJunctions.txt",
    delimiter=",",
)[:N_CELLS, :N_CELLS]
CLASSES = np.loadtxt(DATA_DIR / "CellsClassification.dat").astype(int)[:N_CELLS]


def configure():
    """Configure shared settings for the one-segment circuit."""
    settings.numCells = N_CELLS


def drive_vector(IAVA=0.0, IAVB=0.0):
    """Return per-cell command-interneuron current injection in pA."""
    current = np.zeros(N_CELLS)
    current[np.isin(CLASSES, AVA_CLASSES)] += IAVA
    current[np.isin(CLASSES, AVB_CLASSES)] += IAVB
    return current


def drive_along(branch, amplitude):
    """Return a drive vector for one active command branch."""
    if branch == "AVA":
        return drive_vector(IAVA=amplitude)
    if branch == "AVB":
        return drive_vector(IAVB=amplitude)
    raise ValueError(f"Unknown command branch: {branch!r}")


def full_rest_state(fatigue=0.5):
    """Return the baseline ``[V, SF, w]`` state."""
    voltage = np.full(N_CELLS, settings.L)
    return np.concatenate(
        [voltage, np.full(N_CELLS, fatigue), np.zeros(N_CELLS)]
    )


def reduced_rest_state():
    """Return the baseline ``[V, w]`` state."""
    return np.concatenate(
        [np.full(N_CELLS, settings.L), np.zeros(N_CELLS)]
    )


def _active_mask(ablated_indices=None):
    """Return a Boolean cell mask after validating optional ablations."""
    active = np.ones(N_CELLS, dtype=bool)
    if ablated_indices is None:
        return active
    ablated = np.asarray(tuple(ablated_indices), dtype=int)
    if ablated.size and (ablated.min() < 0 or ablated.max() >= N_CELLS):
        raise ValueError("ablated cell index is outside the segment")
    active[ablated] = False
    return active


def _scaled_coupling_voltage(voltage, voltage_multipliers=None):
    """Scale voltage excursions around rest for synaptic/electrical coupling."""
    if voltage_multipliers is None:
        return voltage
    multipliers = np.asarray(voltage_multipliers, dtype=float)
    if multipliers.shape != (N_CELLS,):
        raise ValueError("voltage_multipliers must have one value per segment cell")
    if np.any(multipliers < 0.0):
        raise ValueError("voltage multipliers must be nonnegative")
    return settings.L + multipliers * (voltage - settings.L)


def _currents(
    voltage,
    presynaptic_signal,
    active=None,
    coupling_voltage=None,
    presynaptic_multipliers=None,
    incoming_disconnected_indices=None,
):
    if active is None:
        active = np.ones(N_CELLS, dtype=bool)
    presynaptic_signal = presynaptic_signal * active
    if presynaptic_multipliers is not None:
        multipliers = np.asarray(presynaptic_multipliers, dtype=float)
        if multipliers.shape != (N_CELLS,):
            raise ValueError(
                "presynaptic_multipliers must have one value per segment cell"
            )
        if np.any(multipliers < 0.0):
            raise ValueError("presynaptic multipliers must be nonnegative")
        presynaptic_signal = presynaptic_signal * multipliers
    excitatory = (
        settings.G_syne
        * (E_CONN @ presynaptic_signal)
        * (voltage - settings.E_syne)
    )
    inhibitory = (
        settings.G_syni
        * (I_CONN @ presynaptic_signal)
        * (voltage - settings.E_syni)
    )
    connected_targets = _active_mask(incoming_disconnected_indices)
    excitatory = excitatory * connected_targets
    inhibitory = inhibitory * connected_targets
    if coupling_voltage is None:
        coupling_voltage = voltage
    voltage_difference = coupling_voltage[:, None] - coupling_voltage[None, :]
    electrically_connected = active & connected_targets
    active_pairs = electrically_connected[:, None] & electrically_connected[None, :]
    gap = settings.G_gap * (
        GJ_CONN * active_pairs * voltage_difference
    ).sum(axis=1)
    return excitatory, inhibitory, gap


def rhs_full(
    t,
    state,
    current,
    ablated_indices=None,
    voltage_multipliers=None,
    presynaptic_multipliers=None,
    incoming_disconnected_indices=None,
):
    """Right-hand side for ``state=[V, SF, w]``.

    Synaptic fatigue evolves but is currently not multiplied into the
    presynaptic signal, matching the established network model.
    """
    del t
    voltage = state[:N_CELLS]
    fatigue = state[N_CELLS : 2 * N_CELLS]
    recovery = state[2 * N_CELLS :]

    coupling_voltage = _scaled_coupling_voltage(voltage, voltage_multipliers)
    gating = 1.0 / (
        1.0 + np.exp(-settings.k_syn * (coupling_voltage - settings.V_th))
    )
    active = _active_mask(ablated_indices)
    excitatory, inhibitory, gap = _currents(
        voltage,
        gating,
        active,
        coupling_voltage,
        presynaptic_multipliers,
        incoming_disconnected_indices,
    )

    voltage_dot = (
        settings.g * f_vec(voltage)
        - recovery
        - excitatory
        - inhibitory
        - gap
        + current
    ) / settings.C
    fatigue_dot = sf_vec(voltage, fatigue)
    recovery_dot = (w_inf_vec(voltage) - recovery) / settings.tau_w
    voltage_dot[~active] = 0.0
    fatigue_dot[~active] = 0.0
    recovery_dot[~active] = 0.0
    return np.concatenate([voltage_dot, fatigue_dot, recovery_dot])


def rhs_vw(
    state,
    current,
    ablated_indices=None,
    voltage_multipliers=None,
    presynaptic_multipliers=None,
    incoming_disconnected_indices=None,
):
    """Right-hand side for the reduced ``state=[V, w]`` system."""
    voltage = state[:N_CELLS]
    recovery = state[N_CELLS:]

    coupling_voltage = _scaled_coupling_voltage(voltage, voltage_multipliers)
    gating = 1.0 / (
        1.0 + np.exp(-settings.k_syn * (coupling_voltage - settings.V_th))
    )
    active = _active_mask(ablated_indices)
    excitatory, inhibitory, gap = _currents(
        voltage,
        gating,
        active,
        coupling_voltage,
        presynaptic_multipliers,
        incoming_disconnected_indices,
    )

    voltage_dot = (
        settings.g * f_vec(voltage)
        - recovery
        - excitatory
        - inhibitory
        - gap
        + current
    ) / settings.C
    recovery_dot = (w_inf_vec(voltage) - recovery) / settings.tau_w
    voltage_dot[~active] = 0.0
    recovery_dot[~active] = 0.0
    return np.concatenate([voltage_dot, recovery_dot])
