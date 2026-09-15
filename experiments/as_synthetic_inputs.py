"""Replace incoming AS connections with calibrated periodic current inputs."""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import activation
import analysis
import frequency_locking
import segment
import settings


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "as_synthetic_inputs"
OPERATING_POINTS = {"AVA": 3.0, "AVB": 2.5}
WAVEFORMS = ("sine", "square", "slow_pulse")
FREQUENCY_RATIOS = (0.5, 1.0, 2.0)
AS_INDICES = np.flatnonzero(segment.CLASSES == 1)
CLASS_COLORS = {
    1: "#4c78a8",
    2: "#72b7b2",
    3: "#54a24b",
    4: "#eeca3b",
    5: "#f58518",
    6: "#e45756",
    7: "#b279a2",
    8: "#1f77b4",
    9: "#d62728",
}


def normalized_waveform(kind, time_ms, frequency_hz):
    """Return a zero-mean waveform bounded by -1 and 1."""
    phase = np.mod(np.asarray(time_ms) * frequency_hz / 1000.0, 1.0)
    if kind == "sine":
        return -np.cos(2.0 * np.pi * phase)
    if kind == "square":
        return np.where(phase < 0.5, -1.0, 1.0)
    if kind == "slow_pulse":
        result = np.empty_like(phase, dtype=float)
        low = phase < 0.25
        rise = (phase >= 0.25) & (phase < 0.5)
        high = (phase >= 0.5) & (phase < 0.75)
        fall = phase >= 0.75
        result[low] = -1.0
        result[rise] = -np.cos(4.0 * np.pi * (phase[rise] - 0.25))
        result[high] = 1.0
        result[fall] = np.cos(4.0 * np.pi * (phase[fall] - 0.75))
        return result
    raise ValueError(f"Unknown waveform: {kind!r}")


def integrate(current_at, args, disconnect_as=False):
    time = np.arange(0.0, args.duration_ms + 0.5 * args.sample_ms, args.sample_ms)
    solution = solve_ivp(
        lambda t, state: segment.rhs_vw(
            state,
            current_at(t),
            incoming_disconnected_indices=AS_INDICES if disconnect_as else None,
        ),
        (0.0, args.duration_ms),
        segment.reduced_rest_state(),
        method="BDF",
        t_eval=time,
        max_step=args.max_step_ms,
        rtol=args.rtol,
        atol=args.atol,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.t, solution.y[: segment.N_CELLS]


def canonical_calibration(branch, drive, args):
    current = segment.drive_along(branch, drive)
    time, voltage = integrate(lambda _: current, args)
    cutoff = int(args.transient_fraction * time.size)
    time = time[cutoff:] - time[cutoff]
    voltage = voltage[:, cutoff:]
    dorsal = voltage[segment.CLASSES == 8].mean(axis=0)
    peaks, period = frequency_locking.reference_peak_times(time, dorsal)

    incoming = np.empty((AS_INDICES.size, time.size))
    for column in range(time.size):
        gating = 1.0 / (
            1.0
            + np.exp(-settings.k_syn * (voltage[:, column] - settings.V_th))
        )
        excitatory, inhibitory, gap = segment._currents(
            voltage[:, column],
            gating,
        )
        net = current - excitatory - inhibitory - gap
        incoming[:, column] = net[AS_INDICES]

    return {
        "frequency_hz": 1000.0 / period,
        "current_mean_pA": incoming.mean(axis=1),
        "current_half_range_pA": 0.5 * np.ptp(incoming, axis=1),
        "current_min_pA": incoming.min(axis=1),
        "current_max_pA": incoming.max(axis=1),
        "reference_peak_count": int(peaks.size),
    }


def synthetic_current(branch, drive, calibration, kind, frequency_hz, time_ms):
    current = segment.drive_along(branch, drive)
    current[AS_INDICES] = 0.0
    wave = float(normalized_waveform(kind, time_ms, frequency_hz))
    current[AS_INDICES] = (
        calibration["current_mean_pA"]
        + calibration["current_half_range_pA"] * wave
    )
    return current


def lag_correlation_at_period(time, signal, period_ms):
    lag = int(round(period_ms / np.median(np.diff(time))))
    if lag <= 0 or lag >= signal.size:
        return np.nan
    left = signal[:-lag]
    right = signal[lag:]
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return np.nan
    return float(np.corrcoef(left, right)[0, 1])


def analyze_run(branch, drive, calibration, kind, ratio, args):
    input_frequency = calibration["frequency_hz"] * ratio
    current_at = lambda t: synthetic_current(
        branch,
        drive,
        calibration,
        kind,
        input_frequency,
        t,
    )
    time, voltage = integrate(current_at, args, disconnect_as=True)
    cutoff = int(args.transient_fraction * time.size)
    time = time[cutoff:] - time[cutoff]
    voltage = voltage[:, cutoff:]
    input_current = np.vstack(
        [current_at(t)[AS_INDICES] for t in time]
    ).T

    dorsal = voltage[segment.CLASSES == 8].mean(axis=0)
    ventral = voltage[segment.CLASSES == 9].mean(axis=0)
    dv_activity = dorsal - ventral
    try:
        reference_peaks, reference_period = frequency_locking.reference_peak_times(
            time,
            dorsal,
        )
    except ValueError:
        reference_peaks = None
        reference_period = np.nan

    output_frequency = 1000.0 / reference_period
    input_lag_correlation = lag_correlation_at_period(
        time,
        dorsal,
        1000.0 / input_frequency,
    )
    entrained = bool(
        np.isfinite(output_frequency)
        and abs(output_frequency / input_frequency - 1.0) <= 0.02
        and input_lag_correlation >= 0.95
    )

    active_cells = int(np.sum(np.ptp(voltage, axis=1) >= 5.0))
    cell_rows = []
    if reference_peaks is not None:
        for cell in range(segment.N_CELLS):
            metrics = frequency_locking.cell_locking_metrics(
                time,
                voltage[cell],
                reference_peaks,
            )
            cell_rows.append(
                {
                    "branch": branch,
                    "waveform": kind,
                    "frequency_ratio": ratio,
                    "cell_index": cell,
                    "cell_label": activation.neuron_labels(
                        segment.CLASSES,
                        [cell],
                    )[0],
                    "active": metrics["active"],
                    "locked_1_to_1": metrics["locked_1_to_1"],
                    "amplitude_mv": metrics["amplitude_mv"],
                    "frequency_hz": metrics["frequency_hz"],
                    "relative_frequency_error": metrics["relative_frequency_error"],
                    "lag_correlation": metrics["lag_correlation"],
                    "peaks_per_cycle": metrics["peaks_per_cycle"],
                }
            )
    else:
        for cell in range(segment.N_CELLS):
            amplitude = float(np.ptp(voltage[cell]))
            cell_rows.append(
                {
                    "branch": branch,
                    "waveform": kind,
                    "frequency_ratio": ratio,
                    "cell_index": cell,
                    "cell_label": activation.neuron_labels(
                        segment.CLASSES,
                        [cell],
                    )[0],
                    "active": amplitude >= 5.0,
                    "locked_1_to_1": False,
                    "amplitude_mv": amplitude,
                    "frequency_hz": np.nan,
                    "relative_frequency_error": np.nan,
                    "lag_correlation": np.nan,
                    "peaks_per_cycle": np.nan,
                }
            )
    locked_cells = sum(row["locked_1_to_1"] for row in cell_rows)

    summary = {
        "branch": branch,
        "command_drive_pA": drive,
        "waveform": kind,
        "frequency_ratio": ratio,
        "input_frequency_hz": input_frequency,
        "output_frequency_hz": output_frequency,
        "output_to_input_frequency_ratio": output_frequency / input_frequency,
        "entrained_1_to_1": entrained,
        "input_period_lag_correlation": input_lag_correlation,
        "dv_amplitude_mv": float(np.ptp(dv_activity)),
        "dv_phase_cycles": float(
            analysis.phase_difference(
                time,
                dorsal,
                ventral,
                transient_frac=0.0,
                prominence=10.0,
            )
        ),
        "active_cells": active_cells,
        "locked_active_cells": locked_cells,
    }
    return {
        "summary": summary,
        "time": time,
        "voltage": voltage,
        "input_current": input_current,
        "cells": cell_rows,
    }


def write_results(runs, calibrations, output, args):
    summaries = [run["summary"] for run in runs]
    with (output / "run_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    cell_rows = [row for run in runs for row in run["cells"]]
    with (output / "cell_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cell_rows[0]))
        writer.writeheader()
        writer.writerows(cell_rows)

    calibration_rows = []
    for branch, calibration in calibrations.items():
        for local_index, cell in enumerate(AS_INDICES):
            calibration_rows.append(
                {
                    "branch": branch,
                    "cell_index": int(cell),
                    "canonical_frequency_hz": calibration["frequency_hz"],
                    "current_mean_pA": calibration["current_mean_pA"][local_index],
                    "current_half_range_pA": calibration["current_half_range_pA"][local_index],
                    "current_min_pA": calibration["current_min_pA"][local_index],
                    "current_max_pA": calibration["current_max_pA"][local_index],
                }
            )
    with (output / "calibration.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(calibration_rows[0]))
        writer.writeheader()
        writer.writerows(calibration_rows)

    payload = {
        "manipulation": (
            "All incoming chemical and electrical connections to AS are removed, "
            "as is direct command current to AS. Each AS cell instead receives a "
            "synthetic current with the mean and half-range of its canonical total "
            "incoming current. Other command targets and all AS outgoing chemical "
            "connections remain intact."
        ),
        "slow_pulse_definition": (
            "A zero-mean smooth trapezoid: quarter-cycle low plateau, quarter-cycle "
            "cosine rise, quarter-cycle high plateau, quarter-cycle cosine fall."
        ),
        "parameters": {
            **vars(args),
            "output": str(args.output),
        },
        "calibrations": {
            branch: {
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in calibration.items()
            }
            for branch, calibration in calibrations.items()
        },
        "runs": summaries,
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")

    for run in runs:
        row = run["summary"]
        slug = f"{row['branch'].lower()}_{row['waveform']}_{row['frequency_ratio']:g}x"
        np.savez_compressed(
            output / f"{slug}_trajectory.npz",
            time_ms=run["time"],
            voltage_mv=run["voltage"],
            as_input_current_pA=run["input_current"],
            cell_classes=segment.CLASSES,
        )


def draw_voltage_stack(axis, run, window_ms=8000.0):
    keep = run["time"] <= window_ms
    time = run["time"][keep] / 1000.0
    voltage = run["voltage"][:, keep]
    indices = activation.ordered_neuron_indices(
        segment.CLASSES,
        include_muscles=True,
    )
    labels = activation.neuron_labels(segment.CLASSES, indices)
    bases = (len(indices) - 1 - np.arange(len(indices))) * 70.0
    for cell, base in zip(indices, bases):
        axis.plot(
            time,
            base + voltage[cell] - settings.L,
            color=CLASS_COLORS[int(segment.CLASSES[cell])],
            lw=0.7,
        )
    axis.set_yticks(bases, labels, fontsize=6)
    axis.set_xlim(time[0], time[-1])
    axis.set_ylim(-25.0, bases[0] + 65.0)
    row = run["summary"]
    status = "1:1" if row["entrained_1_to_1"] else "not 1:1"
    axis.set_title(
        f"{row['waveform'].replace('_', ' ')} · {row['frequency_ratio']:g}× expected\n"
        f"input={row['input_frequency_hz']:.3f} Hz · output={row['output_frequency_hz']:.3f} Hz · {status}",
        fontsize=8,
    )
    axis.grid(axis="x", alpha=0.18)


def save_voltage_grids(runs, output):
    for branch in OPERATING_POINTS:
        figure, axes = plt.subplots(3, 3, figsize=(17, 14), sharex=True, sharey=True)
        branch_runs = [run for run in runs if run["summary"]["branch"] == branch]
        for row_index, kind in enumerate(WAVEFORMS):
            for column, ratio in enumerate(FREQUENCY_RATIOS):
                run = next(
                    run
                    for run in branch_runs
                    if run["summary"]["waveform"] == kind
                    and run["summary"]["frequency_ratio"] == ratio
                )
                axis = axes[row_index, column]
                draw_voltage_stack(axis, run)
                if row_index == 2:
                    axis.set_xlabel("Post-transient time (s)")
                if column == 0:
                    axis.set_ylabel("Cell")
                else:
                    axis.tick_params(axis="y", labelleft=False)
        figure.suptitle(
            f"{branch}: AS incoming connections replaced by calibrated synthetic current",
            fontsize=15,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.97))
        figure.savefig(
            output / f"{branch.lower()}_synthetic_input_voltage_grid.png",
            dpi=180,
        )
        plt.close(figure)


def save_response_figure(runs, output):
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    metrics = (
        ("output_to_input_frequency_ratio", "Output/input frequency"),
        ("dv_amplitude_mv", "DV amplitude (mV)"),
        ("dv_phase_cycles", "DV phase (cycles)"),
    )
    for row_index, branch in enumerate(OPERATING_POINTS):
        for kind in WAVEFORMS:
            rows = [
                run["summary"]
                for run in runs
                if run["summary"]["branch"] == branch
                and run["summary"]["waveform"] == kind
            ]
            for column, (key, ylabel) in enumerate(metrics):
                axes[row_index, column].plot(
                    [row["frequency_ratio"] for row in rows],
                    [row[key] for row in rows],
                    "o-",
                    label=kind.replace("_", " "),
                )
                axes[row_index, column].set_ylabel(f"{branch}: {ylabel}")
        axes[row_index, 0].axhline(1.0, color="0.35", ls=":", lw=1.0)
        axes[row_index, 2].axhline(0.5, color="0.35", ls=":", lw=1.0)
        for axis in axes[row_index]:
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel("Input frequency / canonical frequency")
    figure.suptitle("Response to synthetic AS current waveforms")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(output / "synthetic_input_response.png", dpi=180)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-ms", type=float, default=60000.0)
    parser.add_argument("--sample-ms", type=float, default=5.0)
    parser.add_argument("--max-step-ms", type=float, default=5.0)
    parser.add_argument("--transient-fraction", type=float, default=0.5)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main():
    args = parse_args()
    settings.reset_defaults()
    segment.configure()
    args.output.mkdir(parents=True, exist_ok=True)

    calibrations = {}
    for branch, drive in OPERATING_POINTS.items():
        print(f"Calibrating {branch} canonical AS input...", flush=True)
        calibrations[branch] = canonical_calibration(branch, drive, args)

    runs = []
    for branch, drive in OPERATING_POINTS.items():
        for kind in WAVEFORMS:
            for ratio in FREQUENCY_RATIOS:
                print(f"Running {branch} · {kind} · {ratio:g}×...", flush=True)
                runs.append(
                    analyze_run(
                        branch,
                        drive,
                        calibrations[branch],
                        kind,
                        ratio,
                        args,
                    )
                )

    write_results(runs, calibrations, args.output, args)
    save_voltage_grids(runs, args.output)
    save_response_figure(runs, args.output)
    for run in runs:
        row = run["summary"]
        print(
            f"{row['branch']} {row['waveform']} {row['frequency_ratio']:g}×: "
            f"input={row['input_frequency_hz']:.4f} Hz, "
            f"output={row['output_frequency_hz']:.4f} Hz, "
            f"entrained={row['entrained_1_to_1']}, "
            f"DV amp={row['dv_amplitude_mv']:.2f} mV, "
            f"phase={row['dv_phase_cycles']:.3f}"
        )
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
