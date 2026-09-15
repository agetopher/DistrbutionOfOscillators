"""Whole-animal all-rest baselines and VA10/VB8 ablation kymograms."""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from functions import f_vec, w_inf_vec
import network
import settings


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "whole_network_va10_vb8_ablation"
SEGMENT_SIZE = 17
N_SEGMENTS = 6
LOCAL_INDICES = {"VA10": 10, "VB8": 8}
DRIVES = {"AVA": 3.0, "AVB": 2.5}


def ablation_indices(label):
    """Return the six global indices for a repeated local cell label."""
    if label == "control":
        return np.array([], dtype=int)
    if label not in LOCAL_INDICES:
        raise ValueError(f"unknown ablation {label!r}")
    return np.arange(N_SEGMENTS) * SEGMENT_SIZE + LOCAL_INDICES[label]


def exact_rest_state():
    """Return [V,w] with every cell exactly at the scalar rest state."""
    return np.concatenate(
        [np.full(network.N_CELLS, settings.L), np.zeros(network.N_CELLS)]
    )


def rhs(state, current, ablated):
    """Reduced whole-network RHS with selected cells fully removed and clamped."""
    n_cells = network.N_CELLS
    voltage = state[:n_cells]
    recovery = state[n_cells:]
    active = np.ones(n_cells, dtype=bool)
    active[ablated] = False

    gating = 1.0 / (
        1.0 + np.exp(-settings.k_syn * (voltage - settings.V_th))
    )
    gating = gating * active
    excitatory = (
        settings.G_syne
        * (network.E_CONN @ gating)
        * (voltage - settings.E_syne)
    )
    inhibitory = (
        settings.G_syni
        * (network.I_CONN @ gating)
        * (voltage - settings.E_syni)
    )
    voltage_difference = voltage[:, None] - voltage[None, :]
    active_pairs = active[:, None] & active[None, :]
    gap = settings.G_gap * (
        network.GJ_CONN * active_pairs * voltage_difference
    ).sum(axis=1)

    voltage_dot = (
        settings.g * f_vec(voltage)
        - recovery
        - excitatory
        - inhibitory
        - gap
        + current
    ) / settings.C
    recovery_dot = (w_inf_vec(voltage) - recovery) / settings.tau_w
    voltage_dot[~active] = 0.0
    recovery_dot[~active] = 0.0
    return np.concatenate([voltage_dot, recovery_dot])


def simulate(branch, ablation, args):
    drive = DRIVES[branch]
    current = network.drive_vector(
        IAVA=drive if branch == "AVA" else 0.0,
        IAVB=drive if branch == "AVB" else 0.0,
    )
    ablated = ablation_indices(ablation)
    current[ablated] = 0.0
    time = np.arange(
        0.0,
        args.duration_ms + 0.5 * args.sample_ms,
        args.sample_ms,
    )
    solution = solve_ivp(
        lambda _, state: rhs(state, current, ablated),
        (0.0, args.duration_ms),
        exact_rest_state(),
        method="BDF",
        t_eval=time,
        max_step=args.max_step_ms,
        rtol=args.rtol,
        atol=args.atol,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.t, solution.y[: network.N_CELLS], ablated


def muscle_traces(voltage):
    segment = np.arange(network.N_CELLS) // SEGMENT_SIZE
    dorsal_indices = np.where(network.classes == 8)[0]
    ventral_indices = np.where(network.classes == 9)[0]
    dorsal = np.array(
        [voltage[dorsal_indices[segment[dorsal_indices] == k]].mean(axis=0)
         for k in range(N_SEGMENTS)]
    )
    ventral = np.array(
        [voltage[ventral_indices[segment[ventral_indices] == k]].mean(axis=0)
         for k in range(N_SEGMENTS)]
    )
    return dorsal_indices, ventral_indices, dorsal, ventral


def phase_profile(time, traces, sample_ms):
    centered = traces - traces.mean(axis=1, keepdims=True)
    peaks, _ = find_peaks(centered[0], prominence=8.0)
    if peaks.size < 3:
        return np.full(N_SEGMENTS, np.nan), np.nan
    period_ms = float(np.median(np.diff(time[peaks])))
    max_lag_samples = max(1, int(round(0.5 * period_ms / sample_ms)))
    reference = centered[0]
    center = reference.size - 1
    lags = []
    for trace in centered:
        correlation = np.correlate(trace, reference, mode="full")
        window = correlation[
            center - max_lag_samples : center + max_lag_samples + 1
        ]
        lags.append((int(np.argmax(window)) - max_lag_samples) * sample_ms)
    return np.asarray(lags, dtype=float), period_ms


def summarize(branch, ablation, time, voltage, ablated, args):
    keep = time >= args.transient_fraction * args.duration_ms
    steady_time = time[keep]
    steady_voltage = voltage[:, keep]
    dorsal_indices, ventral_indices, dorsal, ventral = muscle_traces(steady_voltage)
    lags, period_ms = phase_profile(steady_time, dorsal, args.sample_ms)
    valid = np.isfinite(lags)
    if valid.sum() >= 2:
        fit = np.polyfit(np.arange(N_SEGMENTS)[valid], lags[valid], 1)
        fitted = np.polyval(fit, np.arange(N_SEGMENTS)[valid])
        residual = np.sum((lags[valid] - fitted) ** 2)
        total = np.sum((lags[valid] - np.mean(lags[valid])) ** 2)
        phase_linearity_r2 = 1.0 - residual / total if total > 0 else 1.0
        lag_per_segment_ms = float(fit[0])
    else:
        phase_linearity_r2 = np.nan
        lag_per_segment_ms = np.nan
    dv = dorsal - ventral
    summary = {
        "branch": branch,
        "drive_pA": DRIVES[branch],
        "ablation": ablation,
        "ablated_global_indices": ablated.tolist(),
        "frequency_hz": float(1000.0 / period_ms) if np.isfinite(period_ms) else np.nan,
        "period_ms": period_ms,
        "segment_0_to_5_lag_ms": float(lags[-1]),
        "body_phase_span_cycles": float(lags[-1] / period_ms),
        "lag_per_segment_ms": lag_per_segment_ms,
        "phase_linearity_r2": float(phase_linearity_r2),
        "mean_dv_amplitude_mv": float(np.mean(np.ptp(dv, axis=1))),
        "mean_dorsal_amplitude_mv": float(np.mean(np.ptp(dorsal, axis=1))),
        "mean_ventral_amplitude_mv": float(np.mean(np.ptp(ventral, axis=1))),
        "wave_direction": (
            "anterior-to-posterior" if lags[-1] > 0
            else "posterior-to-anterior" if lags[-1] < 0
            else "synchronous"
        ),
    }
    return {
        "summary": summary,
        "time": steady_time,
        "voltage": steady_voltage,
        "dorsal_indices": dorsal_indices,
        "ventral_indices": ventral_indices,
        "dorsal": dorsal,
        "ventral": ventral,
        "dv": dv,
        "lags": lags,
    }


def add_changes(runs):
    baselines = {
        run["summary"]["branch"]: run["summary"]
        for run in runs if run["summary"]["ablation"] == "control"
    }
    for run in runs:
        row = run["summary"]
        baseline = baselines[row["branch"]]
        for metric in (
            "frequency_hz",
            "mean_dv_amplitude_mv",
            "mean_dorsal_amplitude_mv",
            "mean_ventral_amplitude_mv",
        ):
            row[f"{metric}_change_percent"] = 100.0 * (
                row[metric] / baseline[metric] - 1.0
            )
        row["body_phase_span_change_cycles"] = (
            row["body_phase_span_cycles"] - baseline["body_phase_span_cycles"]
        )


def save_results(runs, output, args):
    rows = [run["summary"] for run in runs]
    with (output / "run_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "parameters": {
            "initial_voltage_mv": settings.L,
            "initial_recovery": 0.0,
            "duration_ms": args.duration_ms,
            "transient_fraction": args.transient_fraction,
            "sample_ms": args.sample_ms,
            "max_step_ms": args.max_step_ms,
            "rtol": args.rtol,
            "atol": args.atol,
            "G_syne": settings.G_syne,
            "G_syni": settings.G_syni,
            "G_gap": settings.G_gap,
            "beta": settings.beta,
            "tau_w_ms": settings.tau_w,
            "k_syn": settings.k_syn,
        },
        "ablation_definition": (
            "All six homologues are clamped at rest, their chemical output is "
            "zeroed, and all incident electrical coupling is removed."
        ),
        "runs": rows,
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    for run in runs:
        row = run["summary"]
        np.savez_compressed(
            output / f"{row['branch'].lower()}_{row['ablation'].lower()}_steady.npz",
            time_ms=run["time"],
            voltage_mv=run["voltage"],
            dorsal_segment_mean_mv=run["dorsal"],
            ventral_segment_mean_mv=run["ventral"],
            ablated_global_indices=np.asarray(row["ablated_global_indices"], dtype=int),
        )


def _display_window(run, seconds=5.0):
    end = run["time"][-1]
    return run["time"] >= max(run["time"][0], end - 1000.0 * seconds)


def save_baseline_kymograms(runs, output):
    controls = [run for run in runs if run["summary"]["ablation"] == "control"]
    figure, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    for row_index, run in enumerate(controls):
        mask = _display_window(run)
        times = run["time"][mask] / 1000.0
        for column, (indices, title) in enumerate(
            ((run["dorsal_indices"], "dorsal muscles"),
             (run["ventral_indices"], "ventral muscles"))
        ):
            axis = axes[row_index, column]
            image = axis.imshow(
                run["voltage"][indices][:, mask],
                aspect="auto",
                origin="upper",
                extent=[times[0], times[-1], N_SEGMENTS, 0],
                cmap="viridis",
                vmin=settings.L,
                vmax=settings.H,
            )
            axis.set_title(f"{run['summary']['branch']} · {title}")
            axis.set_ylabel("anterior → posterior segment")
            if row_index == 1:
                axis.set_xlabel("time (s)")
    figure.suptitle(
        "Whole-animal controls from exact rest · "
        f"AVA {DRIVES['AVA']:.1f} pA · AVB {DRIVES['AVB']:.1f} pA"
    )
    figure.subplots_adjust(top=0.9, right=0.87, hspace=0.28, wspace=0.18)
    color_axis = figure.add_axes([0.9, 0.15, 0.015, 0.7])
    figure.colorbar(image, cax=color_axis, label="membrane voltage (mV)")
    figure.savefig(output / "baseline_muscle_kymograms.png", dpi=180)
    plt.close(figure)


def save_ablation_kymograms(runs, output):
    order = ["control", "VA10", "VB8"]
    lookup = {(r["summary"]["branch"], r["summary"]["ablation"]): r for r in runs}
    limit = max(float(np.max(np.abs(run["dv"]))) for run in runs)
    figure, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True, sharey=True)
    for row_index, branch in enumerate(("AVA", "AVB")):
        for column, ablation in enumerate(order):
            run = lookup[(branch, ablation)]
            mask = _display_window(run)
            times = run["time"][mask] / 1000.0
            axis = axes[row_index, column]
            image = axis.imshow(
                run["dv"][:, mask],
                aspect="auto",
                origin="upper",
                extent=[times[0], times[-1], N_SEGMENTS, 0],
                cmap="coolwarm",
                vmin=-limit,
                vmax=limit,
            )
            row = run["summary"]
            axis.set_title(
                f"{branch} · {ablation}\n"
                f"span={row['body_phase_span_cycles']:+.3f} cyc, "
                f"DV={row['mean_dv_amplitude_mv']:.1f} mV"
            )
            if column == 0:
                axis.set_ylabel("anterior → posterior segment")
            if row_index == 1:
                axis.set_xlabel("time (s)")
    figure.suptitle("Whole-animal VA10/VB8 ablations from exact rest")
    figure.subplots_adjust(top=0.88, right=0.87, hspace=0.3, wspace=0.17)
    color_axis = figure.add_axes([0.9, 0.15, 0.015, 0.7])
    figure.colorbar(
        image,
        cax=color_axis,
        label="mean dorsal − ventral voltage (mV)",
    )
    figure.savefig(output / "ablation_dv_kymograms.png", dpi=180)
    plt.close(figure)


def save_phase_profiles(runs, output):
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    colors = {"control": "black", "VA10": "#b279a2", "VB8": "#e45756"}
    for axis, branch in zip(axes, ("AVA", "AVB")):
        for run in runs:
            row = run["summary"]
            if row["branch"] != branch:
                continue
            axis.plot(
                np.arange(N_SEGMENTS),
                run["lags"],
                "o-",
                color=colors[row["ablation"]],
                label=row["ablation"],
            )
        axis.axhline(0.0, color="0.7", lw=0.8)
        axis.set_title(branch)
        axis.set_xlabel("segment (0 anterior → 5 posterior)")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("lag from segment 0 (ms)")
    axes[0].legend()
    figure.suptitle("Whole-animal dorsal-muscle phase profiles")
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(output / "phase_profiles.png", dpi=180)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-ms", type=float, default=30000.0)
    parser.add_argument("--sample-ms", type=float, default=10.0)
    parser.add_argument("--max-step-ms", type=float, default=2.5)
    parser.add_argument("--transient-fraction", type=float, default=0.5)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--atol", type=float, default=1e-8)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main():
    args = parse_args()
    settings.reset_defaults()
    network.configure()
    args.output.mkdir(parents=True, exist_ok=True)
    runs = []
    for branch in ("AVA", "AVB"):
        for ablation in ("control", "VA10", "VB8"):
            print(f"Running {branch} · {ablation} from exact rest...", flush=True)
            time, voltage, ablated = simulate(branch, ablation, args)
            runs.append(summarize(branch, ablation, time, voltage, ablated, args))
    add_changes(runs)
    save_results(runs, args.output, args)
    save_baseline_kymograms(runs, args.output)
    save_ablation_kymograms(runs, args.output)
    save_phase_profiles(runs, args.output)
    for run in runs:
        row = run["summary"]
        print(
            f"{row['branch']} {row['ablation']:7s}: "
            f"f={row['frequency_hz']:.4f} Hz, "
            f"span={row['body_phase_span_cycles']:+.4f} cycles, "
            f"DV={row['mean_dv_amplitude_mv']:.2f} mV, "
            f"R2={row['phase_linearity_r2']:.3f}",
            flush=True,
        )
    print(f"Saved results to {args.output}", flush=True)


if __name__ == "__main__":
    main()
