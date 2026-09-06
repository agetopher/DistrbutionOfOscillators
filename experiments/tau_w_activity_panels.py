"""Generate activation rasters and all-cell voltage traces across tau_w.

The default points span quiescence, recruitment, the AVA/AVB phase optima,
the shared operating optimum, and slow recovery. Every point retains the raw
post-transient trajectory, crossing-event table, manifest, and individual
figures in addition to branch-level comparison grids.
"""

from argparse import ArgumentParser
import json
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import activation_rasters as activity  # noqa: E402
import segment  # noqa: E402
import settings  # noqa: E402


DEFAULT_TAU_VALUES = (100.0, 140.0, 160.0, 170.0, 200.0,
                      370.0, 410.0, 530.0, 1000.0)
DEFAULT_OUTPUT = ROOT / "results" / "tau_w_activity_operating_drives"


def save_branch_grid(runs, branch, output, kind, columns=3):
    branch_runs = sorted(
        (run for run in runs if run["branch"] == branch),
        key=lambda run: run["tau_w_ms"],
    )
    rows = math.ceil(len(branch_runs) / columns)
    if kind == "raster":
        figsize = (5.5 * columns, 4.5 * rows)
        draw = activity.draw_raster
        filename = output / f"{branch.lower()}_tau_w_rasters.png"
    elif kind == "voltage":
        figsize = (6.0 * columns, 6.3 * rows)
        draw = activity.draw_voltage_stack
        filename = output / f"{branch.lower()}_tau_w_voltage_traces.png"
    else:
        raise ValueError(f"Unknown grid kind: {kind!r}")

    figure, axes = plt.subplots(rows, columns, figsize=figsize, squeeze=False)
    for index, axis in enumerate(axes.flat):
        if index >= len(branch_runs):
            axis.axis("off")
            continue
        run = branch_runs[index]
        draw(axis, run)
        row, column = divmod(index, columns)
        if row == rows - 1:
            axis.set_xlabel("Post-transient time (s)")
        if column == 0:
            axis.set_ylabel(
                "Cell" if kind == "raster" else "Cell (stacked voltage traces)"
            )

    drive = branch_runs[0]["drive"]
    figure.suptitle(
        f"{branch} recovery-timescale sensitivity at I{branch}={drive:g} pA\n"
        f"threshold={settings.T:g} mV"
        + (" (dotted in each voltage row)" if kind == "voltage" else ""),
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(filename, dpi=180)
    plt.close(figure)
    print(f"Saved -> {filename}")


def run(tau_values=DEFAULT_TAU_VALUES, ava_drive=3.0, avb_drive=2.5,
        duration_ms=30000.0, sample_ms=5.0, transient_fraction=0.5,
        output=DEFAULT_OUTPUT):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    drives = {"AVA": float(ava_drive), "AVB": float(avb_drive)}
    runs = []

    for tau_w in tau_values:
        settings.reset_defaults()
        settings.beta = 1.03
        settings.tau_w = float(tau_w)
        segment.configure()
        tau_output = output / f"tau_w_{tau_w:g}_ms"

        for branch in ("AVA", "AVB"):
            drive = drives[branch]
            result = activity.write_run(
                tau_output,
                branch,
                drive,
                f"tau_w = {tau_w:g} ms",
                duration_ms,
                sample_ms,
                transient_fraction,
            )
            result["tau_w_ms"] = float(tau_w)
            activity.save_individual_raster(result)
            activity.save_individual_voltage_plot(result)
            runs.append(result)
            print(
                f"Saved {branch} tau_w={tau_w:g} ms at {drive:g} pA "
                f"({len(result['events'])} crossing events)"
            )

    # Restore the shared settings before labeling the comparison figures.
    settings.reset_defaults()
    settings.beta = 1.03
    segment.configure()
    for branch in ("AVA", "AVB"):
        save_branch_grid(runs, branch, output, "raster")
        save_branch_grid(runs, branch, output, "voltage")

    index = {
        "tau_w_values_ms": [float(value) for value in tau_values],
        "drives_pA": drives,
        "duration_ms": float(duration_ms),
        "sample_interval_ms": float(sample_ms),
        "transient_fraction": float(transient_fraction),
        "event_threshold_mV": float(settings.T),
        "runs": [
            {
                "branch": result["branch"],
                "tau_w_ms": result["tau_w_ms"],
                "drive_pA": result["drive"],
                "event_count": len(result["events"]),
                "directory": str(result["destination"].relative_to(output)),
            }
            for result in runs
        ],
    }
    (output / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(f"Saved -> {output / 'index.json'}")
    return runs


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--tau-values", type=float, nargs="+",
                        default=DEFAULT_TAU_VALUES)
    parser.add_argument("--ava-drive", type=float, default=3.0)
    parser.add_argument("--avb-drive", type=float, default=2.5)
    parser.add_argument("--duration-ms", type=float, default=30000.0)
    parser.add_argument("--sample-ms", type=float, default=5.0)
    parser.add_argument("--transient-fraction", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(
        tau_values=arguments.tau_values,
        ava_drive=arguments.ava_drive,
        avb_drive=arguments.avb_drive,
        duration_ms=arguments.duration_ms,
        sample_ms=arguments.sample_ms,
        transient_fraction=arguments.transient_fraction,
        output=arguments.output,
    )
