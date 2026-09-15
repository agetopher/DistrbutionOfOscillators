"""Generate matched AVA/AVB neuronal activation rasters at explicit drives.

This first raster layer deliberately performs no automatic key-point
detection. It samples fixed offsets from the established drive-SNIC locations
plus a shared absolute drive, and saves both the exact crossing events and the
post-transient trajectories used to render each panel.
"""

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
import segment
import settings


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "activation_rasters"
SNIC_DRIVES = {"AVA": 1.68143, "AVB": 1.59643}
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


def simulate(branch, drive, duration_ms, sample_ms):
    current = segment.drive_along(branch, drive)
    initial = segment.reduced_rest_state()
    time = np.arange(0.0, duration_ms + 0.5 * sample_ms, sample_ms)
    solution = solve_ivp(
        lambda _, state: segment.rhs_vw(state, current),
        (0.0, duration_ms),
        initial,
        method="BDF",
        t_eval=time,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.t, solution.y[: segment.N_CELLS], solution.y[:, -1]


def point_directory(output, branch, drive):
    return output / branch.lower() / f"drive_{drive:.5f}_pA"


def write_run(output, branch, drive, label, duration_ms, sample_ms, transient_fraction):
    time, voltage, final_state = simulate(branch, drive, duration_ms, sample_ms)
    cutoff = int(transient_fraction * time.size)
    analysis_time = time[cutoff:] - time[cutoff]
    analysis_voltage = voltage[:, cutoff:]
    indices = activation.ordered_neuron_indices(
        segment.CLASSES,
        include_muscles=True,
    )
    labels = activation.neuron_labels(segment.CLASSES, indices)
    events = activation.threshold_crossings(
        analysis_time,
        analysis_voltage,
        threshold=settings.T,
        cell_indices=indices,
    )

    destination = point_directory(output, branch, drive)
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination / "trajectory.npz",
        time_ms=analysis_time,
        voltage_mv=analysis_voltage,
        cell_classes=segment.CLASSES,
        final_state=final_state,
    )

    with (destination / "events.csv").open("w", newline="") as handle:
        fields = [
            "branch",
            "drive_pA",
            "cell_index",
            "cell_class",
            "cell_label",
            "raster_row",
            "event_number",
            "crossing_time_ms",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for event in events:
            cell = event["cell_index"]
            writer.writerow(
                {
                    "branch": branch,
                    "drive_pA": drive,
                    "cell_class": int(segment.CLASSES[cell]),
                    "cell_label": activation.neuron_labels(segment.CLASSES, [cell])[0],
                    **event,
                }
            )

    manifest = {
        "branch": branch,
        "drive_pA": drive,
        "point_label": label,
        "distance_from_snic_pA": drive - SNIC_DRIVES[branch],
        "known_snic_drive_pA": SNIC_DRIVES[branch],
        "duration_ms": duration_ms,
        "sample_interval_ms": sample_ms,
        "transient_fraction": transient_fraction,
        "analysis_start_ms": float(time[cutoff]),
        "event_definition": "upward voltage crossing",
        "event_threshold_mv": settings.T,
        "cell_order": labels,
        "parameters": {
            "G_syne": settings.G_syne,
            "G_syni": settings.G_syni,
            "G_gap": settings.G_gap,
            "beta": settings.beta,
            "tau_w_ms": settings.tau_w,
            "k_syn": settings.k_syn,
        },
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        "branch": branch,
        "drive": drive,
        "label": label,
        "time": analysis_time,
        "voltage": analysis_voltage,
        "indices": indices,
        "labels": labels,
        "events": events,
        "destination": destination,
    }


def draw_raster(axis, run):
    classes = segment.CLASSES
    for event in run["events"]:
        cell = event["cell_index"]
        axis.scatter(
            event["crossing_time_ms"] / 1000.0,
            event["raster_row"],
            marker="|",
            s=85,
            linewidths=1.4,
            color=CLASS_COLORS[int(classes[cell])],
        )
    axis.set_yticks(np.arange(len(run["labels"])))
    axis.set_yticklabels(run["labels"], fontsize=7)
    axis.set_ylim(len(run["labels"]) - 0.5, -0.5)
    time_seconds = run["time"] / 1000.0
    axis.set_xlim(time_seconds[0], time_seconds[-1])
    axis.set_title(f"{run['branch']} · {run['label']} · {run['drive']:.5f} pA", fontsize=9)
    axis.grid(axis="x", alpha=0.2)


def save_individual_raster(run):
    figure, axis = plt.subplots(figsize=(9, 4.5))
    draw_raster(axis, run)
    axis.set_xlabel("Post-transient time (s)")
    axis.set_ylabel("Cell (fixed biological order)")
    figure.tight_layout()
    figure.savefig(run["destination"] / "raster.png", dpi=180)
    plt.close(figure)


def draw_voltage_stack(axis, run):
    """Draw every saved cell voltage with fixed vertical offsets.

    A shared 70 mV row spacing prevents the 17 traces from obscuring one
    another.  Voltages are shifted only for display; the dashed line in each
    row is the same activation threshold used by the raster, and a scale bar
    reports the unshifted voltage magnitude.
    """
    time_seconds = run["time"] / 1000.0
    row_spacing = 70.0
    reference_voltage = settings.L
    n_rows = len(run["indices"])
    bases = (n_rows - 1 - np.arange(n_rows)) * row_spacing

    for row, (cell, base) in enumerate(zip(run["indices"], bases)):
        trace = run["voltage"][cell]
        axis.plot(
            time_seconds,
            base + trace - reference_voltage,
            color=CLASS_COLORS[int(segment.CLASSES[cell])],
            lw=0.9,
        )
        axis.axhline(
            base + settings.T - reference_voltage,
            color="0.72",
            ls=":",
            lw=0.45,
            zorder=0,
        )

    axis.set_yticks(bases, run["labels"], fontsize=7)
    axis.set_ylim(-25.0, bases[0] + 65.0)
    axis.set_title(
        f"{run['branch']} · {run['label']} · {run['drive']:.5f} pA",
        fontsize=9,
    )
    axis.grid(axis="x", alpha=0.18)

    # An explicit scale bar makes clear that row offsets are not voltage.
    x_span = time_seconds[-1] - time_seconds[0]
    bar_x = time_seconds[-1] - 0.025 * x_span
    bar_base = -15.0
    axis.plot([bar_x, bar_x], [bar_base, bar_base + 20.0], color="0.15", lw=1.4)
    axis.text(
        bar_x - 0.01 * x_span,
        bar_base + 10.0,
        "20 mV",
        ha="right",
        va="center",
        fontsize=7,
    )


def save_individual_voltage_plot(run):
    figure, axis = plt.subplots(figsize=(10, 8.5))
    draw_voltage_stack(axis, run)
    axis.set_xlabel("Post-transient time (s)")
    axis.set_ylabel("Cell (stacked voltage traces)")
    figure.tight_layout()
    figure.savefig(run["destination"] / "voltage_traces.png", dpi=180)
    plt.close(figure)


def save_comparison(runs, output):
    branch_runs = {
        branch: [run for run in runs if run["branch"] == branch]
        for branch in ("AVA", "AVB")
    }
    n_columns = len(branch_runs["AVA"])
    if len(branch_runs["AVB"]) != n_columns:
        raise ValueError("AVA and AVB comparisons require equal point counts")
    figure, axes = plt.subplots(2, n_columns, figsize=(5 * n_columns, 8), squeeze=False)
    for row, branch in enumerate(("AVA", "AVB")):
        for column, run in enumerate(branch_runs[branch]):
            draw_raster(axes[row, column], run)
            if row == 1:
                axes[row, column].set_xlabel("Post-transient time (s)")
            if column == 0:
                axes[row, column].set_ylabel("Cell")
    figure.suptitle(
        "Single-segment activation order: matched SNIC offsets and branch operating points\n"
        f"threshold={settings.T:g} mV, beta={settings.beta:g}, tau_w={settings.tau_w:g} ms",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output / "ava_avb_activation_rasters.png", dpi=180)
    plt.close(figure)


def save_voltage_comparison(runs, output):
    branch_runs = {
        branch: [run for run in runs if run["branch"] == branch]
        for branch in ("AVA", "AVB")
    }
    n_columns = len(branch_runs["AVA"])
    if len(branch_runs["AVB"]) != n_columns:
        raise ValueError("AVA and AVB comparisons require equal point counts")
    figure, axes = plt.subplots(
        2,
        n_columns,
        figsize=(5.5 * n_columns, 13),
        squeeze=False,
    )
    for row, branch in enumerate(("AVA", "AVB")):
        for column, run in enumerate(branch_runs[branch]):
            draw_voltage_stack(axes[row, column], run)
            if row == 1:
                axes[row, column].set_xlabel("Post-transient time (s)")
            if column == 0:
                axes[row, column].set_ylabel("Cell (stacked voltage traces)")
    figure.suptitle(
        "Single-segment voltage trajectories: matched SNIC offsets and branch operating points\n"
        f"threshold={settings.T:g} mV (dotted), beta={settings.beta:g}, "
        f"tau_w={settings.tau_w:g} ms",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(output / "ava_avb_voltage_traces.png", dpi=180)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offsets", type=float, nargs="+", default=(0.02, 0.10))
    parser.add_argument("--ava-drive", type=float, default=3.0)
    parser.add_argument("--avb-drive", type=float, default=2.5)
    parser.add_argument("--duration-ms", type=float, default=30000.0)
    parser.add_argument("--sample-ms", type=float, default=5.0)
    parser.add_argument("--tau-w", type=float, default=400.0)
    parser.add_argument("--transient-fraction", type=float, default=0.5)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0.0 <= args.transient_fraction < 1.0:
        raise ValueError("transient-fraction must lie in [0, 1)")

    settings.reset_defaults()
    settings.tau_w = float(args.tau_w)
    segment.configure()
    output = args.output
    if output is None:
        output = (
            DEFAULT_OUTPUT
            if settings.tau_w == 400.0
            else ROOT / "results" / f"activation_rasters_tau_w_{settings.tau_w:g}"
        )
    output.mkdir(parents=True, exist_ok=True)

    runs = []
    for branch in ("AVA", "AVB"):
        for offset in args.offsets:
            runs.append(
                write_run(
                    output,
                    branch,
                    SNIC_DRIVES[branch] + offset,
                    f"SNIC + {offset:g}",
                    args.duration_ms,
                    args.sample_ms,
                    args.transient_fraction,
                )
            )
        operating_drive = args.ava_drive if branch == "AVA" else args.avb_drive
        runs.append(
            write_run(
                output,
                branch,
                operating_drive,
                f"operating I = {operating_drive:g}",
                args.duration_ms,
                args.sample_ms,
                args.transient_fraction,
            )
        )

    for run in runs:
        save_individual_raster(run)
        save_individual_voltage_plot(run)
        print(f"Saved {run['branch']} {run['drive']:.5f} pA -> {run['destination']}")
    save_comparison(runs, output)
    print(f"Saved comparison -> {output / 'ava_avb_activation_rasters.png'}")
    save_voltage_comparison(runs, output)
    print(f"Saved voltage comparison -> {output / 'ava_avb_voltage_traces.png'}")


if __name__ == "__main__":
    main()
