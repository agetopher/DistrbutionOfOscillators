"""Activation-event extraction for segment-level raster diagnostics."""

import numpy as np


CLASS_NAMES = {
    1: "AS",
    2: "DA",
    3: "DB",
    4: "DD",
    5: "VD",
    6: "VB",
    7: "VA",
    8: "D-muscle",
    9: "V-muscle",
}


def ordered_neuron_indices(classes, include_muscles=False):
    """Return cells in fixed biological class/index order.

    By default this preserves the original neuron-only behavior. Set
    ``include_muscles=True`` to append dorsal and then ventral muscle cells.
    """
    classes = np.asarray(classes, dtype=int)
    indices = np.where(classes <= (9 if include_muscles else 7))[0]
    return indices[np.lexsort((indices, classes[indices]))]


def neuron_labels(classes, indices=None):
    """Return stable labels such as ``AS-0`` and ``VD-6`` for raster rows."""
    classes = np.asarray(classes, dtype=int)
    if indices is None:
        indices = ordered_neuron_indices(classes)
    return [f"{CLASS_NAMES[int(classes[i])]}-{int(i)}" for i in indices]


def threshold_crossings(
    time,
    voltage,
    threshold=-45.0,
    cell_indices=None,
    refractory_ms=0.0,
):
    """Extract linearly interpolated upward voltage-threshold crossings.

    Returns a list of dictionaries so the exact events can be written directly
    to CSV. A refractory interval may be used to suppress numerical chatter,
    but defaults to zero because the plateau model normally crosses once per
    activation.
    """
    time = np.asarray(time, dtype=float)
    voltage = np.asarray(voltage, dtype=float)
    if time.ndim != 1 or time.size < 2:
        raise ValueError("time must be a one-dimensional array with >= 2 samples")
    if np.any(np.diff(time) <= 0):
        raise ValueError("time must be strictly increasing")
    if voltage.ndim != 2 or voltage.shape[1] != time.size:
        raise ValueError("voltage must have shape (cells, time)")
    if refractory_ms < 0:
        raise ValueError("refractory_ms must be nonnegative")

    if cell_indices is None:
        cell_indices = np.arange(voltage.shape[0])

    events = []
    for row, cell in enumerate(np.asarray(cell_indices, dtype=int)):
        trace = voltage[cell]
        crossings = np.where((trace[:-1] < threshold) & (trace[1:] >= threshold))[0]
        last_time = -np.inf
        event_number = 0
        for sample in crossings:
            v0, v1 = trace[sample], trace[sample + 1]
            fraction = (threshold - v0) / (v1 - v0)
            crossing_time = time[sample] + fraction * (time[sample + 1] - time[sample])
            if crossing_time - last_time < refractory_ms:
                continue
            event_number += 1
            events.append(
                {
                    "cell_index": int(cell),
                    "raster_row": int(row),
                    "event_number": event_number,
                    "crossing_time_ms": float(crossing_time),
                }
            )
            last_time = crossing_time
    return events
