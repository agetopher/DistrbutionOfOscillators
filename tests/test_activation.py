import numpy as np

from activation import neuron_labels, ordered_neuron_indices, threshold_crossings


def test_threshold_crossings_interpolate_upward_events():
    time = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    voltage = np.array([[-2.0, -1.0, 1.0, -1.0, 1.0]])

    events = threshold_crossings(time, voltage, threshold=0.0)

    assert [event["event_number"] for event in events] == [1, 2]
    np.testing.assert_allclose(
        [event["crossing_time_ms"] for event in events],
        [1.5, 3.5],
    )


def test_threshold_crossings_apply_refractory_interval():
    time = np.arange(7.0)
    voltage = np.array([[-1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0]])

    events = threshold_crossings(time, voltage, threshold=0.0, refractory_ms=3.0)

    np.testing.assert_allclose(
        [event["crossing_time_ms"] for event in events],
        [0.5, 5.5],
    )


def test_neuron_order_is_biological_then_original_index():
    classes = np.array([5, 1, 8, 3, 1, 7, 9, 2])

    indices = ordered_neuron_indices(classes)

    np.testing.assert_array_equal(indices, np.array([1, 4, 7, 3, 0, 5]))
    assert neuron_labels(classes, indices) == [
        "AS-1",
        "AS-4",
        "DA-7",
        "DB-3",
        "VD-0",
        "VA-5",
    ]


def test_muscles_can_be_appended_in_dorsal_then_ventral_order():
    classes = np.array([9, 1, 8, 7, 8, 9])

    indices = ordered_neuron_indices(classes, include_muscles=True)

    np.testing.assert_array_equal(indices, np.array([1, 3, 2, 4, 0, 5]))
    assert neuron_labels(classes, indices) == [
        "AS-1",
        "VA-3",
        "D-muscle-2",
        "D-muscle-4",
        "V-muscle-0",
        "V-muscle-5",
    ]
