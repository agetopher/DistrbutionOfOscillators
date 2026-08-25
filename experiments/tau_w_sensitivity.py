"""Measure how the recovery time constant changes neuron and segment activity.

The command drive and every parameter except tau_w are held fixed. Results
separate internal neural activity from the functional dorsoventral muscle
output so that a neural limit cycle is not confused with usable segment output.
"""

from argparse import ArgumentParser
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import bifurcations as bif  # noqa: E402
from analysis import dorsoventral_activity, phase_difference  # noqa: E402
import segment  # noqa: E402
import settings  # noqa: E402


BRANCHES = ("AVA", "AVB")
CLASS_NAMES = ("AS", "DA", "DB", "DD", "VD", "VB", "VA", "D muscle", "V muscle")
DEFAULT_TAU_VALUES = (100.0, 200.0, 300.0, 400.0, 500.0, 750.0, 1000.0)


def _frequency(time, trace, prominence=8.0):
    peaks, _ = find_peaks(trace, prominence=prominence)
    if len(peaks) < 3:
        return np.nan
    return float(1000.0 / np.median(np.diff(time[peaks])))


def measure(branch, tau_w, drive=2.5, tf=30000.0, n_eval=6001):
    settings.reset_defaults()
    settings.beta = 1.03
    settings.tau_w = float(tau_w)
    segment.configure()

    time, voltage = bif.simulate_Vw(branch, drive, tf, n_eval)
    keep = time >= 0.5 * tf
    steady_time = time[keep]
    steady_voltage = voltage[:, keep]

    class_traces = np.array([
        steady_voltage[segment.CLASSES == cell_class].mean(axis=0)
        for cell_class in range(1, 10)
    ])
    class_amplitudes = np.ptp(class_traces, axis=1)
    neuron_amplitudes = np.ptp(steady_voltage[segment.CLASSES < 8], axis=1)
    neuron_frequencies = np.array([
        _frequency(steady_time, trace)
        for trace in steady_voltage[segment.CLASSES < 8]
    ])
    oscillating_neurons = int(sum(
        len(find_peaks(trace, prominence=bif.PROMINENCE)[0]) >= 2
        for trace in steady_voltage[segment.CLASSES < 8]
    ))

    dorsal = class_traces[7]
    ventral = class_traces[8]
    dv_trace = dorsoventral_activity(steady_voltage, segment.CLASSES)
    dv_amplitude = float(np.ptp(dv_trace))
    dv_frequency = _frequency(steady_time, dv_trace)
    dv_phase = phase_difference(
        steady_time,
        dorsal,
        ventral,
        transient_frac=0.0,
        prominence=5.0,
    )

    return {
        "branch": branch,
        "tau_w_ms": float(tau_w),
        "drive_pA": float(drive),
        "class_amplitudes_mV": class_amplitudes.tolist(),
        "mean_neuron_amplitude_mV": float(neuron_amplitudes.mean()),
        "oscillating_neurons": oscillating_neurons,
        "median_neuron_frequency_Hz": (
            float(np.nanmedian(neuron_frequencies))
            if np.any(np.isfinite(neuron_frequencies)) else None
        ),
        "dorsal_amplitude_mV": float(np.ptp(dorsal)),
        "ventral_amplitude_mV": float(np.ptp(ventral)),
        "dv_amplitude_mV": dv_amplitude,
        "dv_frequency_Hz": float(dv_frequency) if np.isfinite(dv_frequency) else None,
        "dv_phase_cycles": float(dv_phase) if np.isfinite(dv_phase) else None,
    }


def run(tau_values=DEFAULT_TAU_VALUES, drive=2.5):
    records = [
        measure(branch, tau_w, drive=drive)
        for branch in BRANCHES
        for tau_w in tau_values
    ]
    for record in records:
        print(
            f"{record['branch']} tau_w={record['tau_w_ms']:4.0f} ms  "
            f"neurons={record['oscillating_neurons']:2d}/11  "
            f"neuron amp={record['mean_neuron_amplitude_mV']:5.1f} mV  "
            f"DV amp={record['dv_amplitude_mV']:5.1f} mV  "
            f"DV f={record['dv_frequency_Hz']} Hz  "
            f"DV phase={record['dv_phase_cycles']}"
        )
    return records


def plot(records, output):
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), sharex="col")
    for column, branch in enumerate(BRANCHES):
        branch_records = [r for r in records if r["branch"] == branch]
        tau = np.array([r["tau_w_ms"] for r in branch_records])
        class_amplitude = np.array([
            r["class_amplitudes_mV"] for r in branch_records
        ]).T

        ax = axes[0, column]
        image = ax.imshow(
            class_amplitude,
            aspect="auto",
            origin="lower",
            extent=[tau.min(), tau.max(), 0.5, 9.5],
            cmap="viridis",
        )
        ax.set_yticks(range(1, 10), CLASS_NAMES)
        ax.set_ylabel("cell class")
        ax.set_title(f"{branch}: class-mean voltage amplitude")
        fig.colorbar(image, ax=ax, label="amplitude (mV)")

        ax = axes[1, column]
        ax.plot(tau, [r["dv_amplitude_mV"] for r in branch_records],
                "o-", color="#3569a8", label=r"$\Delta_{DV}$ amplitude")
        ax.plot(tau, [r["mean_neuron_amplitude_mV"] for r in branch_records],
                "s--", color="#555555", label="mean neuron amplitude")
        ax.set_ylabel("amplitude (mV)")
        ax.set_title("internal rhythm versus functional output")
        ax.legend(fontsize=8)

        ax = axes[2, column]
        frequency = [
            np.nan if r["dv_frequency_Hz"] is None else r["dv_frequency_Hz"]
            for r in branch_records
        ]
        phase = [
            np.nan if r["dv_phase_cycles"] is None else r["dv_phase_cycles"]
            for r in branch_records
        ]
        ax.plot(tau, frequency, "o-", color="#d06b32",
                label=r"$\Delta_{DV}$ frequency")
        ax.set_ylabel("frequency (Hz)")
        ax.set_xlabel(r"$\tau_w$ (ms)")
        twin = ax.twinx()
        twin.plot(tau, phase, "s--", color="#4daf4a", label="DV phase")
        twin.axhline(0.5, color="#4daf4a", ls=":", lw=0.8)
        twin.set_ylim(0.0, 0.55)
        twin.set_ylabel("DV phase (cycles)")
        lines = ax.lines + twin.lines[:1]
        ax.legend(lines, [line.get_label() for line in lines], fontsize=8)
        ax.set_title("segment rhythm and antiphase")

    fig.suptitle(
        fr"Single-segment recovery-timescale sensitivity "
        fr"($\beta=1.03$, command drive = {records[0]['drive_pA']:g} pA)",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--drive", type=float, default=2.5)
    args = parser.parse_args()

    records = run(drive=args.drive)
    media = ROOT / "media"
    media.mkdir(exist_ok=True)
    stem = f"tau_w_sensitivity_drive_{args.drive:g}"
    figure = media / f"{stem}.png"
    data = media / f"{stem}.json"
    plot(records, figure)
    data.write_text(json.dumps(records, indent=2) + "\n")
    print(f"Saved -> {figure}")
    print(f"Saved -> {data}")


if __name__ == "__main__":
    main()
