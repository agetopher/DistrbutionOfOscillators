"""Render all-cell voltage traces for the systematic single-neuron ablations."""

import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import activation
import settings


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results" / "systematic_ablation"
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
ABLATION_ORDER = (
    "control",
    "AS-0",
    "AS-1",
    "DA-2",
    "DB-3",
    "DD-4",
    "VD-5",
    "VD-6",
    "VB-7",
    "VB-8",
    "VA-9",
    "VA-10",
)


def slug(label):
    return label.lower().replace("-", "")


def load_summaries(input_root):
    with (input_root / "run_summary.csv").open() as handle:
        return {
            (row["branch"], row["ablation"]): row
            for row in csv.DictReader(handle)
        }


def load_run(input_root, branch, ablation, summaries):
    source = input_root / f"{branch.lower()}_{slug(ablation)}_trajectory.npz"
    with np.load(source) as trajectory:
        time = trajectory["time_ms"]
        voltage = trajectory["voltage_mv"]
        classes = trajectory["cell_classes"].astype(int)
        ablated = set(trajectory["ablated_indices"].astype(int))
    indices = activation.ordered_neuron_indices(classes, include_muscles=True)
    return {
        "branch": branch,
        "ablation": ablation,
        "summary": summaries[(branch, ablation)],
        "time": time,
        "voltage": voltage,
        "classes": classes,
        "ablated": ablated,
        "indices": indices,
        "labels": activation.neuron_labels(classes, indices),
    }


def draw_voltage_stack(axis, run, window_ms=None, show_labels=True):
    time = run["time"]
    voltage = run["voltage"]
    if window_ms is not None:
        keep = time <= time[0] + window_ms
        time = time[keep]
        voltage = voltage[:, keep]

    time_seconds = (time - time[0]) / 1000.0
    row_spacing = 70.0
    reference_voltage = settings.L
    bases = (len(run["indices"]) - 1 - np.arange(len(run["indices"]))) * row_spacing
    for cell, base in zip(run["indices"], bases):
        is_ablated = cell in run["ablated"]
        axis.plot(
            time_seconds,
            base + voltage[cell] - reference_voltage,
            color="#777777" if is_ablated else CLASS_COLORS[int(run["classes"][cell])],
            lw=1.4 if is_ablated else 0.8,
            ls="--" if is_ablated else "-",
            alpha=0.9,
        )
        axis.axhline(
            base + settings.T - reference_voltage,
            color="0.78",
            ls=":",
            lw=0.4,
            zorder=0,
        )

    if show_labels:
        labels = [
            f"{label} (ablated)" if cell in run["ablated"] else label
            for cell, label in zip(run["indices"], run["labels"])
        ]
        axis.set_yticks(bases, labels, fontsize=7)
    else:
        axis.set_yticks([])
    axis.set_ylim(-25.0, bases[0] + 65.0)
    axis.set_xlim(time_seconds[0], time_seconds[-1])
    axis.grid(axis="x", alpha=0.18)

    summary = run["summary"]
    axis.set_title(
        f"{run['ablation']}\n"
        f"f={float(summary['reference_frequency_hz']):.3f} Hz | "
        f"phase={float(summary['dv_phase_cycles']):.3f} | "
        f"lock={summary['locked_active_cells']}/{summary['active_nonablated_cells']}",
        fontsize=8,
    )


def save_individual(run, output):
    figure, axis = plt.subplots(figsize=(11, 8.5))
    draw_voltage_stack(axis, run)
    axis.set_xlabel("Post-transient time (s)")
    axis.set_ylabel("Cell (stacked voltage traces)")
    figure.suptitle(
        f"{run['branch']} command · {run['ablation']} ablation",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    destination = output / "individual" / run["branch"].lower()
    destination.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination / f"{slug(run['ablation'])}_voltage_traces.png", dpi=180)
    plt.close(figure)


def save_branch_grid(runs, output, window_ms):
    branch = runs[0]["branch"]
    figure, axes = plt.subplots(3, 4, figsize=(20, 20), squeeze=False)
    for panel, (axis, run) in enumerate(zip(axes.flat, runs)):
        draw_voltage_stack(
            axis,
            run,
            window_ms=window_ms,
            show_labels=panel % 4 == 0,
        )
        if panel >= 8:
            axis.set_xlabel("Time (s)")
        if panel % 4 == 0:
            axis.set_ylabel("Cell")
    figure.suptitle(
        f"{branch} command: all-cell voltage traces after systematic ablation\n"
        f"first {window_ms / 1000.0:g} s of each post-transient trajectory; "
        f"dotted lines = {settings.T:g} mV threshold",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(output / f"{branch.lower()}_ablation_voltage_grid.png", dpi=180)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--grid-window-ms", type=float, default=6000.0)
    return parser.parse_args()


def main():
    args = parse_args()
    settings.reset_defaults()
    output = args.output or args.input / "voltage_traces"
    output.mkdir(parents=True, exist_ok=True)
    summaries = load_summaries(args.input)

    for branch in ("AVA", "AVB"):
        runs = [
            load_run(args.input, branch, ablation, summaries)
            for ablation in ABLATION_ORDER
        ]
        for run in runs:
            save_individual(run, output)
        save_branch_grid(runs, output, args.grid_window_ms)
    print(f"Saved 24 individual traces and 2 comparison grids to {output}")


if __name__ == "__main__":
    main()
