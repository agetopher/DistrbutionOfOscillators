"""Meeting-ready comparison of the two candidate synaptic steepness values.

All circuit and membrane parameters are held fixed while ``k_syn`` is changed
between 0.25 and 0.125.  The reduced ``(V, w)`` segment is used because
synaptic fatigue is inert in the current network model.

Generated figures:

* ``media/k_syn_activation_comparison.png``
* ``media/k_syn_zero_drive_comparison.png``
* ``media/k_syn_command_comparison.png``
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import segment  # noqa: E402
import settings  # noqa: E402


MEDIA_DIR = ROOT / "media"
K_VALUES = (0.25, 0.125)
K_COLORS = {0.25: "#31688e", 0.125: "#b63679"}
CLASS_NAMES = {
    1: "AS",
    2: "DA",
    3: "DB",
    4: "DD",
    5: "VD",
    6: "VB",
    7: "VA",
    8: "dorsal muscle",
    9: "ventral muscle",
}


def simulate(k_syn, IAVA=0.0, IAVB=0.0, tf=15000.0, dt=10.0):
    """Simulate one parameter condition from the common resting state."""
    settings.reset_defaults()
    settings.k_syn = float(k_syn)
    segment.configure()
    time = np.arange(0.0, tf + 0.5 * dt, dt)
    current = segment.drive_vector(IAVA=IAVA, IAVB=IAVB)
    solution = solve_ivp(
        lambda _, state: segment.rhs_vw(state, current),
        (0.0, tf),
        segment.reduced_rest_state(),
        method="BDF",
        t_eval=time,
        rtol=1e-7,
        atol=1e-9,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.t, solution.y[: segment.N_CELLS]


def steady_window(time, voltage, duration=5000.0):
    keep = time >= time[-1] - duration
    return (time[keep] - time[keep][0]) / 1000.0, voltage[:, keep]


def representative_by_class(voltage):
    """Return the first trace for each of the nine cell classes."""
    return {
        cls: voltage[np.where(segment.CLASSES == cls)[0][0]]
        for cls in range(1, 10)
    }


def class_amplitudes(voltage):
    """Maximum steady-state peak-to-peak amplitude within each class."""
    return np.asarray(
        [
            np.max(np.ptp(voltage[segment.CLASSES == cls], axis=1))
            for cls in range(1, 10)
        ]
    )


def rhythm_metrics(time, trace):
    """Return peak-to-peak amplitude and frequency for one steady trace."""
    amplitude = float(np.ptp(trace))
    peaks, _ = find_peaks(trace, prominence=5.0)
    if len(peaks) < 3:
        return amplitude, np.nan
    period = float(np.median(np.diff(time[peaks])))
    return amplitude, 1.0 / period if period > 0 else np.nan


def plot_activation(save=True):
    settings.reset_defaults()
    voltage = np.linspace(-80.0, -25.0, 1000)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    ax = axes[0]
    for k_syn in K_VALUES:
        gate = 1.0 / (
            1.0 + np.exp(-k_syn * (voltage - settings.V_th))
        )
        rest_gate = 1.0 / (
            1.0 + np.exp(-k_syn * (settings.L - settings.V_th))
        )
        ax.plot(
            voltage,
            gate,
            color=K_COLORS[k_syn],
            lw=2,
            label=fr"$k_{{syn}}={k_syn}$; $s(L)={rest_gate:.3f}$",
        )
        ax.scatter(
            [settings.L],
            [rest_gate],
            color=K_COLORS[k_syn],
            s=35,
            zorder=3,
        )
    for level, label in (
        (settings.L, "rest L"),
        (settings.T, "membrane threshold T"),
        (settings.H, "plateau H"),
    ):
        ax.axvline(level, color="0.55", lw=0.8, ls=":")
        ax.text(level + 0.5, 0.97, label, rotation=90, va="top", fontsize=8)
    ax.set(xlabel="presynaptic voltage (mV)", ylabel="synaptic gate $s(V)$")
    ax.set_title("The shallow gate is substantially open at rest")
    ax.legend(loc="lower right", fontsize=9)

    ax = axes[1]
    labels = ("excitatory", "inhibitory")
    conductances = (settings.G_syne, settings.G_syni)
    reversals = (settings.E_syne, settings.E_syni)
    x = np.arange(len(labels))
    width = 0.34
    for offset, k_syn in zip((-width / 2, width / 2), K_VALUES):
        rest_gate = 1.0 / (
            1.0 + np.exp(-k_syn * (settings.L - settings.V_th))
        )
        magnitudes = [
            abs(g_syn * rest_gate * (settings.L - reversal))
            for g_syn, reversal in zip(conductances, reversals)
        ]
        bars = ax.bar(
            x + offset,
            magnitudes,
            width,
            color=K_COLORS[k_syn],
            label=fr"$k_{{syn}}={k_syn}$",
        )
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("resting current per connection (pA)")
    ax.set_title("Basal coupling with unchanged conductances")
    ax.legend(fontsize=9)

    fig.suptitle(
        "Why changing only $k_{syn}$ changes the resting circuit",
        fontsize=13,
    )
    fig.tight_layout()
    if save:
        filename = MEDIA_DIR / "k_syn_activation_comparison.png"
        fig.savefig(filename, dpi=200, bbox_inches="tight")
        print(f"Saved -> {filename}")
    return fig


def plot_zero_drive(results, save=True):
    fig = plt.figure(figsize=(12, 8.5))
    grid = fig.add_gridspec(2, 2, height_ratios=(1.15, 1.0))
    heat_axes = [fig.add_subplot(grid[0, i]) for i in range(2)]
    trace_ax = fig.add_subplot(grid[1, 0])
    amp_ax = fig.add_subplot(grid[1, 1])
    image = None
    steady = {}

    for column, (ax, k_syn) in enumerate(zip(heat_axes, K_VALUES)):
        time, voltage = steady_window(*results[k_syn])
        steady[k_syn] = (time, voltage)
        image = ax.imshow(
            voltage,
            aspect="auto",
            origin="lower",
            extent=(time[0], time[-1], -0.5, segment.N_CELLS - 0.5),
            vmin=-72,
            vmax=-30,
            cmap="viridis",
            interpolation="nearest",
        )
        ax.set_title(fr"$k_{{syn}}={k_syn}$")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("cell" if column == 0 else "")
        ax.set_yticks(
            np.arange(segment.N_CELLS),
            [
                f"{idx}: {CLASS_NAMES[cls]}"
                for idx, cls in enumerate(segment.CLASSES)
            ],
            fontsize=7,
        )

    color_axis = fig.add_axes((0.945, 0.57, 0.012, 0.27))
    colorbar = fig.colorbar(image, cax=color_axis)
    colorbar.set_label("voltage (mV)")

    dd = int(np.where(segment.CLASSES == 4)[0][0])
    for k_syn in K_VALUES:
        time, voltage = steady[k_syn]
        trace_ax.plot(
            time,
            voltage[dd],
            color=K_COLORS[k_syn],
            lw=1.3,
            label=fr"$k_{{syn}}={k_syn}$",
        )
    trace_ax.set(
        xlabel="time (s)",
        ylabel="DD voltage (mV)",
        title="Representative inhibitory motor neuron",
    )
    trace_ax.legend()

    x = np.arange(1, 10)
    width = 0.34
    for offset, k_syn in zip((-width / 2, width / 2), K_VALUES):
        amplitudes = class_amplitudes(steady[k_syn][1])
        amp_ax.bar(
            x + offset,
            amplitudes,
            width,
            color=K_COLORS[k_syn],
            label=fr"$k_{{syn}}={k_syn}$",
        )
        print(
            f"k_syn={k_syn}: zero-drive class amplitudes (mV) "
            + ", ".join(
                f"{CLASS_NAMES[cls]}={amplitudes[cls - 1]:.2f}"
                for cls in range(1, 10)
            )
        )
    amp_ax.set_xticks(x, [CLASS_NAMES[cls] for cls in range(1, 10)])
    amp_ax.tick_params(axis="x", rotation=35)
    amp_ax.set(
        ylabel="maximum peak-to-peak amplitude (mV)",
        title="Steady-state activity by cell class",
    )
    amp_ax.legend()

    fig.suptitle(
        "Undriven 17-cell segment: the shallow sigmoid creates activity",
        fontsize=13,
    )
    fig.subplots_adjust(
        left=0.10, right=0.92, bottom=0.10, top=0.90, hspace=0.35, wspace=0.25
    )
    if save:
        filename = MEDIA_DIR / "k_syn_zero_drive_comparison.png"
        fig.savefig(filename, dpi=200, bbox_inches="tight")
        print(f"Saved -> {filename}")
    return fig


def plot_command_response(results, drive=3.0, save=True):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    command_rows = (("AVA", drive, 0.0), ("AVB", 0.0, drive))

    for row, (name, _, _) in enumerate(command_rows):
        for col, k_syn in enumerate(K_VALUES):
            ax = axes[row, col]
            time, voltage = steady_window(*results[(name, k_syn)])
            dorsal = voltage[segment.CLASSES == 8].mean(axis=0)
            ventral = voltage[segment.CLASSES == 9].mean(axis=0)
            ax.plot(time, dorsal, color="#2c7fb8", lw=1.2, label="dorsal muscle")
            ax.plot(
                time,
                ventral,
                color="#d95f0e",
                lw=1.2,
                label="ventral muscle",
            )
            ax.set_title(fr"{name}={drive:g} pA, $k_{{syn}}={k_syn}$")
            ax.set_xlabel("time (s)")
            ax.set_ylabel("mean muscle voltage (mV)")
            dorsal_amplitude, frequency = rhythm_metrics(time, dorsal)
            metric_text = fr"$f={frequency:.2f}$ Hz" if np.isfinite(frequency) else "no rhythm"
            metric_text += f"\n" + fr"dorsal $\Delta V={dorsal_amplitude:.1f}$ mV"
            ax.text(
                0.02,
                0.96,
                metric_text,
                transform=ax.transAxes,
                va="top",
                fontsize=8,
                bbox=dict(facecolor="white", edgecolor="0.8", alpha=0.8),
            )

            neural_amp = np.max(
                np.ptp(voltage[segment.CLASSES <= 7], axis=1)
            )
            muscle_amp = np.max(
                np.ptp(voltage[segment.CLASSES >= 8], axis=1)
            )
            print(
                f"{name}, k_syn={k_syn}: max neural amplitude="
                f"{neural_amp:.2f} mV, max muscle amplitude="
                f"{muscle_amp:.2f} mV, dorsal frequency="
                f"{frequency:.3f} Hz"
            )

    fig.suptitle(
        "Same command drive and conductances: segment muscle output",
        fontsize=13,
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=2,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    if save:
        filename = MEDIA_DIR / "k_syn_command_comparison.png"
        fig.savefig(filename, dpi=200, bbox_inches="tight")
        print(f"Saved -> {filename}")
    return fig


def main():
    MEDIA_DIR.mkdir(exist_ok=True)
    plot_activation()

    zero_drive = {
        k_syn: simulate(k_syn)
        for k_syn in K_VALUES
    }
    plot_zero_drive(zero_drive)

    drive = 3.0
    command_results = {}
    for command, IAVA, IAVB in (
        ("AVA", drive, 0.0),
        ("AVB", 0.0, drive),
    ):
        for k_syn in K_VALUES:
            command_results[(command, k_syn)] = simulate(
                k_syn, IAVA=IAVA, IAVB=IAVB
            )
    plot_command_response(command_results, drive=drive)
    plt.close("all")


if __name__ == "__main__":
    main()
