"""Test cell-by-cell frequency locking at the canonical AVA/AVB operating points."""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import activation
import analysis
import frequency_locking
import segment


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results" / "activation_rasters"
DEFAULT_OUTPUT = ROOT / "results" / "frequency_locking"
OPERATING_POINTS = {"AVA": 3.0, "AVB": 2.5}


def trajectory_path(input_root, branch, drive):
    return input_root / branch.lower() / f"drive_{drive:.5f}_pA" / "trajectory.npz"


def analyze_branch(input_root, branch, drive):
    source = trajectory_path(input_root, branch, drive)
    with np.load(source) as trajectory:
        time = trajectory["time_ms"]
        voltage = trajectory["voltage_mv"]
        classes = trajectory["cell_classes"].astype(int)

    reference = analysis.dorsoventral_activity(voltage, classes)
    reference_peaks, reference_period = frequency_locking.reference_peak_times(
        time,
        reference,
        prominence=10.0,
    )
    rows = []
    for cell in activation.ordered_neuron_indices(classes, include_muscles=True):
        metrics = frequency_locking.cell_locking_metrics(
            time,
            voltage[cell],
            reference_peaks,
        )
        rows.append(
            {
                "branch": branch,
                "drive_pA": drive,
                "cell_index": int(cell),
                "cell_class": int(classes[cell]),
                "cell_label": activation.neuron_labels(classes, [cell])[0],
                **metrics,
            }
        )
    return {
        "branch": branch,
        "drive_pA": drive,
        "source": str(source.relative_to(ROOT)),
        "reference_period_ms": reference_period,
        "reference_frequency_hz": 1000.0 / reference_period,
        "reference_cycles": int(reference_peaks.size),
        "rows": rows,
    }


def write_csv(results, output):
    rows = [row for result in results for row in result["rows"]]
    fields = list(rows[0])
    with (output / "frequency_locking.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(results, output):
    branches = []
    for result in results:
        active = [row for row in result["rows"] if row["active"]]
        silent = [row["cell_label"] for row in result["rows"] if not row["active"]]
        unlocked = [
            row["cell_label"]
            for row in active
            if not row["locked_1_to_1"]
        ]
        branches.append(
            {
                "branch": result["branch"],
                "drive_pA": result["drive_pA"],
                "source": result["source"],
                "reference_frequency_hz": result["reference_frequency_hz"],
                "reference_cycles": result["reference_cycles"],
                "active_cells": len(active),
                "locked_active_cells": sum(row["locked_1_to_1"] for row in active),
                "silent_cells": silent,
                "unlocked_active_cells": unlocked,
                "interpretation": (
                    "All recruited cells are 1:1 frequency locked."
                    if not unlocked
                    else "At least one recruited cell fails the 1:1 locking criteria."
                ),
            }
        )
    payload = {
        "method": {
            "reference": "peaks of mean dorsal minus mean ventral muscle voltage",
            "active_threshold_mv_peak_to_peak": 5.0,
            "locking_criteria": {
                "minimum_one_period_lag_correlation": 0.98,
                "maximum_relative_frequency_error": 0.01,
            },
            "note": (
                "The fundamental uses one representative maximum per reference cycle. "
                "Raw peak multiplicity is reported separately because DD/VD can have "
                "two voltage peaks in one network cycle."
            ),
        },
        "branches": branches,
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")


def save_figure(results, output):
    labels = [row["cell_label"] for row in results[0]["rows"]]
    figure, axes = plt.subplots(2, 2, figsize=(14, 8), sharex="col")
    for column, result in enumerate(results):
        rows = result["rows"]
        frequencies = np.array([row["frequency_hz"] for row in rows])
        multiplicities = np.array([row["peaks_per_cycle"] for row in rows])
        active = np.array([row["active"] for row in rows])
        colors = ["#4c78a8" if value else "#bab0ac" for value in active]

        axes[0, column].bar(np.arange(len(rows)), np.nan_to_num(frequencies), color=colors)
        silent_indices = np.where(~active)[0]
        axes[0, column].scatter(
            silent_indices,
            np.full(silent_indices.size, 0.04 * result["reference_frequency_hz"]),
            marker="x",
            s=45,
            linewidths=1.5,
            color="#e45756",
            zorder=3,
        )
        for cell in silent_indices:
            axes[0, column].text(
                cell,
                0.08 * result["reference_frequency_hz"],
                "silent",
                rotation=90,
                ha="center",
                va="bottom",
                fontsize=7,
                color="#e45756",
            )
        axes[0, column].axhline(
            result["reference_frequency_hz"],
            color="#e45756",
            linestyle="--",
            linewidth=1.3,
            label="network fundamental",
        )
        axes[0, column].set_title(
            f"{result['branch']} drive = {result['drive_pA']:g} pA"
        )
        axes[0, column].set_ylabel("Fundamental frequency (Hz)")
        axes[0, column].legend(fontsize=8)

        axes[1, column].bar(np.arange(len(rows)), multiplicities, color=colors)
        axes[1, column].axhline(1.0, color="0.35", linestyle=":", linewidth=1.0)
        axes[1, column].set_ylabel("Raw voltage peaks / cycle")
        axes[1, column].set_xticks(np.arange(len(labels)), labels, rotation=70, ha="right")
        axes[1, column].set_ylim(0.0, max(2.25, multiplicities.max() + 0.2))

    figure.suptitle(
        "Single-segment cell-by-cell frequency locking\n"
        "fundamental locking separated from within-cycle waveform multiplicity"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output / "frequency_locking.png", dpi=180)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = [
        analyze_branch(args.input, branch, drive)
        for branch, drive in OPERATING_POINTS.items()
    ]
    write_csv(results, args.output)
    write_summary(results, args.output)
    save_figure(results, args.output)
    for result in results:
        active = [row for row in result["rows"] if row["active"]]
        silent = [row["cell_label"] for row in result["rows"] if not row["active"]]
        print(
            f"{result['branch']}: {sum(row['locked_1_to_1'] for row in active)}/"
            f"{len(active)} active cells locked at "
            f"{result['reference_frequency_hz']:.6f} Hz; silent={silent}"
        )
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
