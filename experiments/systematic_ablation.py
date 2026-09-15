"""Ablate each single-segment neuron under canonical AVA and AVB drive."""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import activation
import segment
import settings
from targeted_ablation import analyze_run


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "systematic_ablation"
OPERATING_POINTS = {"AVA": 3.0, "AVB": 2.5}


def neuron_targets():
    indices = activation.ordered_neuron_indices(segment.CLASSES)
    return [
        (activation.neuron_labels(segment.CLASSES, [cell])[0], (int(cell),))
        for cell in indices
    ]


def add_control_changes(runs):
    controls = {
        run["summary"]["branch"]: run["summary"]
        for run in runs
        if run["summary"]["ablation"] == "control"
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

    branch_summaries = []
    for branch in OPERATING_POINTS:
        rows = [
            row for row in summaries
            if row["branch"] == branch and row["ablation"] != "control"
        ]
        strongest_phase = max(
            rows,
            key=lambda row: abs(row["dv_phase_change_cycles"])
            if np.isfinite(row["dv_phase_change_cycles"])
            else np.inf,
        )
        strongest_amplitude = min(rows, key=lambda row: row["dv_amplitude_change_percent"])
        branch_summaries.append(
            {
                "branch": branch,
                "largest_phase_disruption": strongest_phase["ablation"],
                "largest_phase_disruption_cycles": strongest_phase["dv_phase_change_cycles"],
                "largest_amplitude_loss": strongest_amplitude["ablation"],
                "largest_amplitude_loss_percent": strongest_amplitude["dv_amplitude_change_percent"],
                "incomplete_locking": [
                    {
                        "ablation": row["ablation"],
                        "locked_active_cells": row["locked_active_cells"],
                        "active_nonablated_cells": row["active_nonablated_cells"],
                    }
                    for row in rows
                    if row["locked_active_cells"] < row["active_nonablated_cells"]
                ],
            }
        )

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
            "Each neuron is individually clamped at its rest initial state and "
            "removed from chemical-synapse and gap-junction coupling."
        ),
        "branch_summary": branch_summaries,
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
    labels = [label for label, _ in neuron_targets()]
    figure, axes = plt.subplots(4, 2, figsize=(15, 12), sharex="col")
    for column, branch in enumerate(OPERATING_POINTS):
        rows = [
            run["summary"] for run in runs
            if run["summary"]["branch"] == branch
            and run["summary"]["ablation"] != "control"
        ]
        x = np.arange(len(rows))
        frequency = np.array([row["frequency_change_percent"] for row in rows])
        amplitude = np.array([row["dv_amplitude_change_percent"] for row in rows])
        phase = np.array([row["dv_phase_cycles"] for row in rows])
        locked_percent = np.array(
            [
                100.0 * row["locked_active_cells"] / row["active_nonablated_cells"]
                if row["active_nonablated_cells"]
                else np.nan
                for row in rows
            ]
        )

        axes[0, column].bar(x, frequency, color="#4c78a8")
        axes[0, column].axhline(0.0, color="0.25", lw=0.8)
        axes[0, column].set_title(f"{branch} command")
        axes[0, column].set_ylabel("Frequency change (%)")

        axes[1, column].bar(x, amplitude, color="#f58518")
        axes[1, column].axhline(0.0, color="0.25", lw=0.8)
        axes[1, column].set_ylabel("DV amplitude change (%)")

        axes[2, column].bar(x, phase, color="#54a24b")
        axes[2, column].axhline(0.5, color="0.25", ls=":", lw=1.0)
        axes[2, column].set_ylabel("Dorsoventral phase (cycles)")
        axes[3, column].bar(x, locked_percent, color="#b279a2")
        axes[3, column].axhline(100.0, color="0.25", ls=":", lw=1.0)
        axes[3, column].set_ylabel("Active cells locked (%)")
        axes[3, column].set_ylim(0.0, 105.0)
        axes[3, column].set_xticks(x, labels, rotation=60, ha="right")

        for axis in axes[:, column]:
            axis.grid(axis="y", alpha=0.2)

    figure.suptitle(
        "Systematic single-neuron ablations at canonical operating points\n"
        "each cell removed from chemical and electrical coupling"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output / "systematic_ablation.png", dpi=180)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
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
    settings.reset_defaults()
    segment.configure()
    args.output.mkdir(parents=True, exist_ok=True)

    runs = []
    targets = [("control", ())] + neuron_targets()
    for branch, drive in OPERATING_POINTS.items():
        for label, ablated in targets:
            print(f"Running {branch} · {label}...", flush=True)
            runs.append(analyze_run(branch, drive, label, ablated, args))
    add_control_changes(runs)
    write_results(runs, args.output, args)
    save_figure(runs, args.output)

    for run in runs:
        row = run["summary"]
        if row["ablation"] == "control":
            continue
        print(
            f"{row['branch']:3s} {row['ablation']:5s}: "
            f"df={row['frequency_change_percent']:+7.2f}%, "
            f"dA={row['dv_amplitude_change_percent']:+7.2f}%, "
            f"phase={row['dv_phase_cycles']:.3f}, "
            f"active={row['active_nonablated_cells']}, "
            f"locked={row['locked_active_cells']}"
        )
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
