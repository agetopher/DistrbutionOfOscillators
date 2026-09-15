"""Scale AS presynaptic efficacy and measure the canonical segment response."""

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
DEFAULT_OUTPUT = ROOT / "results" / "as_contribution_scaling"
OPERATING_POINTS = {"AVA": 3.0, "AVB": 2.5}
DEFAULT_MULTIPLIERS = (0.5, 1.0, 1.5)
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


def simulate(branch, drive, multiplier, args):
    presynaptic_multipliers = np.ones(segment.N_CELLS)
    presynaptic_multipliers[segment.CLASSES == 1] = multiplier
    current = segment.drive_along(branch, drive)
    initial = segment.reduced_rest_state()
    time = np.arange(0.0, args.duration_ms + 0.5 * args.sample_ms, args.sample_ms)
    solution = solve_ivp(
        lambda _, state: segment.rhs_vw(
            state,
            current,
            presynaptic_multipliers=presynaptic_multipliers,
        ),
        (0.0, args.duration_ms),
        initial,
        method="BDF",
        t_eval=time,
        max_step=args.max_step_ms,
        rtol=args.rtol,
        atol=args.atol,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.t, solution.y[: segment.N_CELLS], presynaptic_multipliers


def analyze_run(branch, drive, multiplier, args):
    time, voltage, presynaptic_multipliers = simulate(branch, drive, multiplier, args)
    cutoff = int(args.transient_fraction * time.size)
    time = time[cutoff:] - time[cutoff]
    voltage = voltage[:, cutoff:]

    dorsal = voltage[segment.CLASSES == 8].mean(axis=0)
    ventral = voltage[segment.CLASSES == 9].mean(axis=0)
    dv_activity = dorsal - ventral
    try:
        reference_peaks, reference_period = frequency_locking.reference_peak_times(
            time,
            dorsal,
            prominence=10.0,
        )
    except ValueError:
        reference_peaks = None
        reference_period = np.nan

    cell_rows = []
    for cell in activation.ordered_neuron_indices(
        segment.CLASSES,
        include_muscles=True,
    ):
        if reference_peaks is None:
            amplitude = float(np.ptp(voltage[cell]))
            metrics = {
                "active": amplitude >= 5.0,
                "locked_1_to_1": False,
                "amplitude_mv": amplitude,
                "frequency_hz": np.nan,
                "peaks_per_cycle": np.nan,
            }
        else:
            metrics = frequency_locking.cell_locking_metrics(
                time,
                voltage[cell],
                reference_peaks,
            )
        cell_rows.append(
            {
                "branch": branch,
                "drive_pA": drive,
                "as_contribution_multiplier": multiplier,
                "cell_index": int(cell),
                "cell_label": activation.neuron_labels(segment.CLASSES, [cell])[0],
                "active": metrics["active"],
                "locked_1_to_1": metrics["locked_1_to_1"],
                "amplitude_mv": metrics["amplitude_mv"],
                "frequency_hz": metrics["frequency_hz"],
                "peaks_per_cycle": metrics["peaks_per_cycle"],
            }
        )
    active = [row for row in cell_rows if row["active"]]
    summary = {
        "branch": branch,
        "drive_pA": drive,
        "as_contribution_multiplier": multiplier,
        "oscillatory": reference_peaks is not None,
        "reference_frequency_hz": 1000.0 / reference_period,
        "dv_amplitude_mv": float(np.ptp(dv_activity)),
        "dorsal_mean_amplitude_mv": float(np.ptp(dorsal)),
        "ventral_mean_amplitude_mv": float(np.ptp(ventral)),
        "as_mean_voltage_amplitude_mv": float(
            np.mean(np.ptp(voltage[segment.CLASSES == 1], axis=1))
        ),
        "as_effective_signal_amplitude": float(
            multiplier
            * np.mean(
                np.ptp(
                    1.0
                    / (
                        1.0
                        + np.exp(
                            -settings.k_syn
                            * (voltage[segment.CLASSES == 1] - settings.V_th)
                        )
                    ),
                    axis=1,
                )
            )
        ),
        "as_effective_signal_mean": float(
            multiplier
            * np.mean(
                1.0
                / (
                    1.0
                    + np.exp(
                        -settings.k_syn
                        * (voltage[segment.CLASSES == 1] - settings.V_th)
                    )
                )
            )
        ),
        "dv_phase_cycles": float(
            analysis.phase_difference(
                time,
                dorsal,
                ventral,
                transient_frac=0.0,
                prominence=10.0,
            )
        ),
        "active_cells": len(active),
        "locked_active_cells": sum(row["locked_1_to_1"] for row in active),
        "silent_cells": [row["cell_label"] for row in cell_rows if not row["active"]],
    }
    return {
        "summary": summary,
        "cells": cell_rows,
        "time": time,
        "voltage": voltage,
        "presynaptic_multipliers": presynaptic_multipliers,
    }


def add_control_changes(runs):
    controls = {
        run["summary"]["branch"]: run["summary"]
        for run in runs
        if run["summary"]["as_contribution_multiplier"] == 1.0
    }
    for run in runs:
        row = run["summary"]
        control = controls[row["branch"]]
        row["frequency_change_percent"] = 100.0 * (
            row["reference_frequency_hz"] / control["reference_frequency_hz"] - 1.0
        )
        row["dv_amplitude_change_percent"] = 100.0 * (
            row["dv_amplitude_mv"] / control["dv_amplitude_mv"] - 1.0
        )
        row["dv_phase_change_cycles"] = row["dv_phase_cycles"] - control["dv_phase_cycles"]


def write_results(runs, output, args):
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

    payload = {
        "manipulation": (
            "For AS cells only, the presynaptic gating signal entering the "
            "excitatory connectivity matrix is multiplied by m. Intrinsic AS "
            "dynamics, incoming connections, and gap-junction conductance are unchanged."
        ),
        "parameters": {
            "duration_ms": args.duration_ms,
            "sample_interval_ms": args.sample_ms,
            "maximum_solver_step_ms": args.max_step_ms,
            "transient_fraction": args.transient_fraction,
            "solver_relative_tolerance": args.rtol,
            "solver_absolute_tolerance": args.atol,
            "G_syne": settings.G_syne,
            "G_syni": settings.G_syni,
            "G_gap": settings.G_gap,
            "beta": settings.beta,
            "tau_w_ms": settings.tau_w,
            "k_syn": settings.k_syn,
        },
        "runs": summaries,
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")

    for run in runs:
        summary = run["summary"]
        multiplier_slug = str(summary["as_contribution_multiplier"]).replace(".", "p")
        np.savez_compressed(
            output / f"{summary['branch'].lower()}_as_{multiplier_slug}x_trajectory.npz",
            time_ms=run["time"],
            voltage_mv=run["voltage"],
            cell_classes=segment.CLASSES,
            presynaptic_multipliers=run["presynaptic_multipliers"],
        )


def draw_voltage_stack(axis, run, window_ms=6000.0):
    keep = run["time"] <= window_ms
    time_seconds = run["time"][keep] / 1000.0
    voltage = run["voltage"][:, keep]
    indices = activation.ordered_neuron_indices(
        segment.CLASSES,
        include_muscles=True,
    )
    labels = activation.neuron_labels(segment.CLASSES, indices)
    bases = (len(indices) - 1 - np.arange(len(indices))) * 70.0
    for cell, base in zip(indices, bases):
        axis.plot(
            time_seconds,
            base + voltage[cell] - settings.L,
            color=CLASS_COLORS[int(segment.CLASSES[cell])],
            lw=0.8,
        )
        axis.axhline(
            base + settings.T - settings.L,
            color="0.78",
            ls=":",
            lw=0.4,
            zorder=0,
        )
    axis.set_yticks(bases, labels, fontsize=7)
    axis.set_ylim(-25.0, bases[0] + 65.0)
    axis.set_xlim(time_seconds[0], time_seconds[-1])
    summary = run["summary"]
    axis.set_title(
        f"AS contribution {summary['as_contribution_multiplier']:g}x\n"
        f"f={summary['reference_frequency_hz']:.3f} Hz | "
        f"phase={summary['dv_phase_cycles']:.3f} | "
        f"lock={summary['locked_active_cells']}/{summary['active_cells']}",
        fontsize=9,
    )
    axis.grid(axis="x", alpha=0.18)


def save_voltage_grid(runs, output):
    figure, axes = plt.subplots(2, 3, figsize=(17, 12), sharex=True, sharey=True)
    for row_index, branch in enumerate(("AVA", "AVB")):
        branch_runs = [run for run in runs if run["summary"]["branch"] == branch]
        for column, run in enumerate(branch_runs):
            draw_voltage_stack(axes[row_index, column], run)
            if row_index == 1:
                axes[row_index, column].set_xlabel("Post-transient time (s)")
            if column == 0:
                axes[row_index, column].set_ylabel(f"{branch} · cell")
            else:
                axes[row_index, column].tick_params(
                    axis="y",
                    which="both",
                    labelleft=False,
                )
    figure.suptitle(
        "AS contribution scaling at canonical operating points\n"
        "AS outgoing excitatory gating = multiplier × baseline gating",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output / "as_contribution_scaling_traces.png", dpi=180)
    plt.close(figure)


def save_response_figure(runs, output):
    figure, axes = plt.subplots(1, 4, figsize=(18, 4))
    for branch, color in (("AVA", "#4c78a8"), ("AVB", "#e45756")):
        rows = [run["summary"] for run in runs if run["summary"]["branch"] == branch]
        multiplier = [row["as_contribution_multiplier"] for row in rows]
        axes[0].plot(multiplier, [row["reference_frequency_hz"] for row in rows], "o-", label=branch, color=color)
        axes[1].plot(multiplier, [row["dv_amplitude_mv"] for row in rows], "o-", label=branch, color=color)
        axes[2].plot(multiplier, [row["dv_phase_cycles"] for row in rows], "o-", label=branch, color=color)
        axes[3].plot(multiplier, [row["as_effective_signal_mean"] for row in rows], "o-", label=branch, color=color)
    axes[0].set_ylabel("Frequency (Hz)")
    axes[1].set_ylabel("DV amplitude (mV)")
    axes[2].set_ylabel("DV phase (cycles)")
    axes[3].set_ylabel("Mean effective AS output")
    axes[2].axhline(0.5, color="0.35", ls=":", lw=1.0)
    for axis in axes:
        axis.set_xlabel("AS contribution multiplier")
        axis.grid(alpha=0.2)
        axis.legend()
    figure.suptitle("Response to AS contribution scaling")
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(output / "as_contribution_scaling_response.png", dpi=180)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multipliers", type=float, nargs="+", default=DEFAULT_MULTIPLIERS)
    parser.add_argument("--duration-ms", type=float, default=30000.0)
    parser.add_argument("--sample-ms", type=float, default=5.0)
    parser.add_argument("--max-step-ms", type=float, default=5.0)
    parser.add_argument("--transient-fraction", type=float, default=0.5)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main():
    args = parse_args()
    if 1.0 not in args.multipliers:
        raise ValueError("multipliers must include the 1.0 control")
    settings.reset_defaults()
    segment.configure()
    args.output.mkdir(parents=True, exist_ok=True)

    runs = []
    for branch, drive in OPERATING_POINTS.items():
        for multiplier in args.multipliers:
            print(f"Running {branch} · AS contribution {multiplier:g}x...", flush=True)
            runs.append(analyze_run(branch, drive, multiplier, args))
    add_control_changes(runs)
    write_results(runs, args.output, args)
    save_voltage_grid(runs, args.output)
    save_response_figure(runs, args.output)

    for run in runs:
        row = run["summary"]
        print(
            f"{row['branch']} {row['as_contribution_multiplier']:g}x: "
            f"f={row['reference_frequency_hz']:.6f} Hz, "
            f"DV amplitude={row['dv_amplitude_mv']:.3f} mV, "
            f"phase={row['dv_phase_cycles']:.4f}, "
            f"locked={row['locked_active_cells']}/{row['active_cells']}, "
            f"silent={row['silent_cells']}"
        )
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
