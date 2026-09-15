"""Run VA10/VB8 ablations at the canonical AVA/AVB operating points."""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import activation
import analysis
import frequency_locking
import segment
import settings


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "targeted_ablation"
OPERATING_POINTS = {"AVA": 3.0, "AVB": 2.5}
ABLATIONS = {"control": (), "VA-10": (10,), "VB-8": (8,)}


def simulate(branch, drive, ablated, duration_ms, sample_ms, max_step_ms, rtol, atol):
    current = segment.drive_along(branch, drive)
    initial = segment.reduced_rest_state()
    time = np.arange(0.0, duration_ms + 0.5 * sample_ms, sample_ms)
    solution = solve_ivp(
        lambda _, state: segment.rhs_vw(
            state,
            current,
            ablated_indices=ablated,
        ),
        (0.0, duration_ms),
        initial,
        method="BDF",
        t_eval=time,
        max_step=max_step_ms,
        rtol=rtol,
        atol=atol,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.t, solution.y[: segment.N_CELLS]


def peak_frequency(time, trace, prominence=10.0):
    peaks, _ = find_peaks(trace, prominence=prominence)
    if peaks.size < 2:
        return np.nan
    return float(1000.0 / np.mean(np.diff(time[peaks])))


def analyze_run(branch, drive, ablation_label, ablated, args):
    time, voltage = simulate(
        branch,
        drive,
        ablated,
        args.duration_ms,
        args.sample_ms,
        args.max_step_ms,
        args.rtol,
        args.atol,
    )
    cutoff = int(args.transient_fraction * time.size)
    time = time[cutoff:] - time[cutoff]
    voltage = voltage[:, cutoff:]

    dorsal = voltage[segment.CLASSES == 8].mean(axis=0)
    ventral = voltage[segment.CLASSES == 9].mean(axis=0)
    reference = dorsal
    dv_activity = dorsal - ventral
    try:
        reference_peaks, reference_period = frequency_locking.reference_peak_times(
            time,
            reference,
            prominence=10.0,
        )
    except ValueError:
        reference_peaks = None
        reference_period = np.nan
    ablated_set = set(ablated)
    cell_rows = []
    for cell in activation.ordered_neuron_indices(
        segment.CLASSES,
        include_muscles=True,
    ):
        label = activation.neuron_labels(segment.CLASSES, [cell])[0]
        if cell in ablated_set:
            metrics = {
                "active": False,
                "locked_1_to_1": False,
                "amplitude_mv": 0.0,
                "frequency_hz": np.nan,
                "peaks_per_cycle": 0.0,
            }
            status = "ablated"
        elif reference_peaks is not None:
            full_metrics = frequency_locking.cell_locking_metrics(
                time,
                voltage[cell],
                reference_peaks,
            )
            metrics = {
                key: full_metrics[key]
                for key in (
                    "active",
                    "locked_1_to_1",
                    "amplitude_mv",
                    "frequency_hz",
                    "peaks_per_cycle",
                )
            }
            status = "active" if metrics["active"] else "silent"
        else:
            amplitude = float(np.ptp(voltage[cell]))
            active = amplitude >= 5.0
            metrics = {
                "active": active,
                "locked_1_to_1": False,
                "amplitude_mv": amplitude,
                "frequency_hz": peak_frequency(time, voltage[cell]),
                "peaks_per_cycle": np.nan,
            }
            status = "active_unlocked" if active else "silent"
        cell_rows.append(
            {
                "branch": branch,
                "drive_pA": drive,
                "ablation": ablation_label,
                "cell_index": int(cell),
                "cell_label": label,
                "status": status,
                **metrics,
            }
        )

    active_rows = [row for row in cell_rows if row["status"] == "active"]
    dv_phase = analysis.phase_difference(
        time,
        dorsal,
        ventral,
        transient_frac=0.0,
        prominence=10.0,
    )
    summary = {
        "branch": branch,
        "drive_pA": drive,
        "ablation": ablation_label,
        "ablated_indices": list(ablated),
        "reference_frequency_hz": 1000.0 / reference_period,
        "frequency_reference": "mean dorsal muscle voltage",
        "dv_amplitude_mv": float(np.ptp(dv_activity)),
        "dorsal_mean_amplitude_mv": float(np.ptp(dorsal)),
        "ventral_mean_amplitude_mv": float(np.ptp(ventral)),
        "dv_phase_cycles": float(dv_phase),
        "active_nonablated_cells": sum(
            row["status"] in ("active", "active_unlocked") for row in cell_rows
        ),
        "locked_active_cells": sum(
            row["locked_1_to_1"]
            for row in cell_rows
            if row["status"] in ("active", "active_unlocked")
        ),
        "silent_nonablated_cells": [
            row["cell_label"] for row in cell_rows if row["status"] == "silent"
        ],
    }
    return {
        "summary": summary,
        "cells": cell_rows,
        "time": time,
        "dorsal": dorsal,
        "ventral": ventral,
        "reference": reference,
        "dv_activity": dv_activity,
        "voltage": voltage,
    }


def write_results(runs, output, args):
    summaries = [run["summary"] for run in runs]
    baseline = {
        row["branch"]: row
        for row in summaries
        if row["ablation"] == "control"
    }
    for row in summaries:
        control = baseline[row["branch"]]
        row["frequency_change_percent"] = 100.0 * (
            row["reference_frequency_hz"] / control["reference_frequency_hz"] - 1.0
        )
        row["dv_amplitude_change_percent"] = 100.0 * (
            row["dv_amplitude_mv"] / control["dv_amplitude_mv"] - 1.0
        )

    with (output / "run_summary.csv").open("w", newline="") as handle:
        fields = list(summaries[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)

    cell_rows = [row for run in runs for row in run["cells"]]
    with (output / "cell_summary.csv").open("w", newline="") as handle:
        fields = list(cell_rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(cell_rows)

    payload = {
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
        "ablation_definition": (
            "Ablated cells are clamped at their rest initial state and removed "
            "from chemical-synapse and gap-junction coupling."
        ),
        "runs": summaries,
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")

    for run in runs:
        summary = run["summary"]
        ablation_slug = summary["ablation"].lower().replace("-", "")
        np.savez_compressed(
            output / f"{summary['branch'].lower()}_{ablation_slug}_trajectory.npz",
            time_ms=run["time"],
            voltage_mv=run["voltage"],
            cell_classes=segment.CLASSES,
            ablated_indices=np.asarray(summary["ablated_indices"], dtype=int),
        )


def save_figure(runs, output):
    figure, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True, sharey=True)
    for row_index, branch in enumerate(("AVA", "AVB")):
        branch_runs = [run for run in runs if run["summary"]["branch"] == branch]
        for column, run in enumerate(branch_runs):
            axis = axes[row_index, column]
            time_seconds = run["time"] / 1000.0
            axis.plot(time_seconds, run["dorsal"], label="mean dorsal", lw=1.0)
            axis.plot(time_seconds, run["ventral"], label="mean ventral", lw=1.0)
            summary = run["summary"]
            axis.set_title(
                f"{branch} · {summary['ablation']}\n"
                f"f={summary['reference_frequency_hz']:.3f} Hz, "
                f"DV amp={summary['dv_amplitude_mv']:.1f} mV, "
                f"phase={summary['dv_phase_cycles']:.3f}"
            )
            axis.grid(alpha=0.2)
            if row_index == 1:
                axis.set_xlabel("Post-transient time (s)")
            if column == 0:
                axis.set_ylabel("Mean muscle voltage (mV)")
    axes[0, 0].legend(fontsize=8, loc="lower right")
    figure.suptitle(
        "Targeted single-segment ablations at canonical operating points\n"
        "VA10 and VB8 are fully removed from chemical and electrical coupling"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(output / "targeted_ablation.png", dpi=180)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-ms", type=float, default=30000.0)
    parser.add_argument("--sample-ms", type=float, default=5.0)
    parser.add_argument("--max-step-ms", type=float, default=5.0)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--transient-fraction", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0.0 <= args.transient_fraction < 1.0:
        raise ValueError("transient-fraction must lie in [0, 1)")
    settings.reset_defaults()
    segment.configure()
    args.output.mkdir(parents=True, exist_ok=True)

    runs = []
    for branch, drive in OPERATING_POINTS.items():
        for ablation_label, ablated in ABLATIONS.items():
            print(f"Running {branch} · {ablation_label}...")
            runs.append(
                analyze_run(branch, drive, ablation_label, ablated, args)
            )
    write_results(runs, args.output, args)
    save_figure(runs, args.output)

    for run in runs:
        row = run["summary"]
        print(
            f"{row['branch']:3s} {row['ablation']:7s}: "
            f"f={row['reference_frequency_hz']:.6f} Hz, "
            f"DV amplitude={row['dv_amplitude_mv']:.3f} mV, "
            f"phase={row['dv_phase_cycles']:.4f}, "
            f"locked={row['locked_active_cells']}/{row['active_nonablated_cells']}"
        )
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
