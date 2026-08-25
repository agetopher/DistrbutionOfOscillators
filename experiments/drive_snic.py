"""Focused evidence for SNIC onset under AVA and AVB command drive.

This script complements ``experiments/bifurcations.py`` with two compact
figures aimed specifically at classifying the lower command-drive transition.
For each command branch it combines:

* continuation of the resting equilibrium to its saddle-node;
* the critical *real* eigenvalue approaching zero at the fold;
* finite-amplitude dorsoventral muscle output immediately above the fold;
* period divergence and the SNIC scaling ``f^2 ~ I - I_SN``.

The canonical recovery timescale is ``tau_w=400 ms``, selected from a
single-segment sensitivity sweep balancing AVA and AVB dorsoventral phase.
"""

from argparse import ArgumentParser
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import bifurcations as bif  # noqa: E402
import segment  # noqa: E402
import settings  # noqa: E402


BRANCHES = ("AVA", "AVB")
COLORS = {"AVA": "#3569a8", "AVB": "#d06b32"}
DEFAULT_OFFSETS = np.array(
    [0.0001, 0.0002, 0.0004, 0.0008, 0.0015, 0.003, 0.006, 0.012, 0.025]
)


def _linear_fit(x, y):
    """Return slope, intercept, and coefficient of determination."""
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    total = np.sum((y - y.mean()) ** 2)
    r_squared = (
        1.0 - np.sum((y - predicted) ** 2) / total
        if total > 0.0
        else np.nan
    )
    return float(slope), float(intercept), float(r_squared)


def equilibrium_diagnostics(branch, fold_step=0.001, refine_step=0.00001):
    """Continue the rest branch and extrapolate its saddle-node location.

    Near a generic saddle-node, the critical real eigenvalue obeys
    ``lambda^2 ~ I_SN - I``.  Fitting that relation gives a more precise fold
    estimate than reporting the last successful natural-continuation step.
    """
    grid_fold, _ = bif.locate_fold(branch, hi=3.0, step=fold_step)
    coarse_stop = max(0.0, grid_fold - 0.15)
    drives = np.unique(
        np.concatenate(
            [
                np.arange(0.0, coarse_stop, 0.02),
                np.linspace(coarse_stop, grid_fold, 151),
            ]
        )
    )
    record = bif.continue_equilibrium(branch, drives, bif.rest_state())

    # Natural continuation at a 0.001-pA step only brackets the fold. Refine
    # from a safe nearby equilibrium at 1e-5 pA, retaining the actual states so
    # the root solver remains on the same branch.
    refine_start = grid_fold - 0.01
    state = bif.equilibrium(
        bif._drive_along(branch, refine_start), bif.rest_state()
    )
    refined_drives = []
    refined_max_re = []
    refined_lead_im = []
    for current in np.arange(
        refine_start, grid_fold + 0.002 + 0.5 * refine_step, refine_step
    ):
        state = bif.equilibrium(
            bif._drive_along(branch, current), state, tol=1e-7
        )
        if state is None:
            break
        eigenvalues = np.linalg.eigvals(
            bif.jac_Vw(state, bif._drive_along(branch, current))
        )
        lead = eigenvalues[np.argmax(eigenvalues.real)]
        refined_drives.append(float(current))
        refined_max_re.append(float(lead.real))
        refined_lead_im.append(float(abs(lead.imag)))

    refined_drives = np.asarray(refined_drives)
    refined_max_re = np.asarray(refined_max_re)
    refined_lead_im = np.asarray(refined_lead_im)
    fit = (
        (refined_lead_im < 1e-7)
        & (refined_max_re < -1e-7)
    )
    if fit.sum() < 20:
        raise RuntimeError(
            f"Too few near-fold real eigenvalues to classify {branch}: "
            f"{fit.sum()}"
        )
    fit_indices = np.where(fit)[0][-30:]
    slope, intercept, r_squared = _linear_fit(
        refined_drives[fit_indices], refined_max_re[fit_indices] ** 2
    )
    critical_drive = -intercept / slope
    refined_last = float(refined_drives[-1])
    if slope >= 0.0 or not (
        refined_last <= critical_drive <= refined_last + 5 * refine_step
    ):
        raise RuntimeError(
            f"Implausible saddle-node fit for {branch}: "
            f"last root={refined_last}, fitted={critical_drive}, slope={slope}"
        )

    return {
        "branch": branch,
        "grid_fold": refined_last,
        "critical_drive": float(critical_drive),
        "fold_fit_slope": slope,
        "fold_fit_r2": r_squared,
        "fit_indices": fit_indices,
        "record": record,
        "refined_drives": refined_drives,
        "refined_max_re": refined_max_re,
        "refined_lead_im": refined_lead_im,
    }


def cycle_diagnostics(branch, critical_drive, offsets=DEFAULT_OFFSETS):
    """Measure dorsoventral output cycles just above the saddle-node."""
    rows = []
    for offset in np.asarray(offsets, float):
        # Resolve about ten cycles. The additive term approximates the regular
        # (non-bottleneck) traversal time around the finite-amplitude orbit.
        expected_period = 140.0 / np.sqrt(offset) + 1400.0
        final_time = float(np.clip(28.0 * expected_period, 60000.0, 350000.0))
        n_eval = int(final_time / 10.0) + 1
        time, voltage = bif.simulate_Vw(
            branch, critical_drive + offset, final_time, n_eval
        )
        keep = time >= 0.60 * final_time
        steady_time = time[keep]
        steady_voltage = voltage[:, keep]
        dv_trace = bif.dv_activity(steady_voltage)
        amplitude = float(np.ptp(dv_trace))
        crossing_level = 0.5 * (float(dv_trace.min()) + float(dv_trace.max()))
        crossings = np.where(
            (dv_trace[:-1] < crossing_level)
            & (dv_trace[1:] >= crossing_level)
        )[0]
        if len(crossings) < 4:
            raise RuntimeError(
                f"Could not resolve a {branch} cycle at dI={offset:g} pA; "
                f"found {len(crossings)} upward midpoint crossings"
            )
        fractions = (
            (crossing_level - dv_trace[crossings])
            / (dv_trace[crossings + 1] - dv_trace[crossings])
        )
        crossing_times = (
            steady_time[crossings]
            + fractions * (steady_time[crossings + 1] - steady_time[crossings])
        )
        intervals = np.diff(crossing_times)
        period = float(np.median(intervals))
        rows.append(
            {
                "offset": float(offset),
                "drive": float(critical_drive + offset),
                "period": period,
                "frequency": 1000.0 / period,
                "amplitude": amplitude,
                "observable": "mean dorsal muscle V - mean ventral muscle V",
                "period_method": "upward midpoint crossings of DV activity",
                "period_cv": float(np.std(intervals) / np.mean(intervals)),
            }
        )
        print(
            f"{branch} dI={offset:7.4f} pA  T={period:8.1f} ms  "
            f"f={1000.0 / period:6.3f} Hz  "
            f"DV amplitude={amplitude:5.1f} mV"
        )

    offset = np.array([row["offset"] for row in rows])
    period = np.array([row["period"] for row in rows])
    frequency = np.array([row["frequency"] for row in rows])

    # Restrict asymptotic fits to the closest five points.
    near = np.arange(min(5, len(rows)))
    log_slope, log_intercept, log_r2 = _linear_fit(
        np.log(offset[near]), np.log(period[near])
    )
    f2_slope, f2_intercept, f2_r2 = _linear_fit(
        offset[near], frequency[near] ** 2
    )
    inverse_sqrt_slope, inverse_sqrt_intercept, inverse_sqrt_r2 = _linear_fit(
        1.0 / np.sqrt(offset[near]), period[near]
    )
    return {
        "branch": branch,
        "rows": rows,
        "log_period_slope": log_slope,
        "log_period_intercept": log_intercept,
        "log_period_r2": log_r2,
        "f2_slope": f2_slope,
        "f2_intercept": f2_intercept,
        "f2_r2": f2_r2,
        "inverse_sqrt_slope": inverse_sqrt_slope,
        "inverse_sqrt_intercept": inverse_sqrt_intercept,
        "inverse_sqrt_r2": inverse_sqrt_r2,
    }


def plot_equilibria(results, output):
    """Plot equilibrium termination and the critical eigenvalue."""
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5))
    for column, branch in enumerate(BRANCHES):
        result = results[branch]["equilibrium"]
        record = result["record"]
        current = record["drives"]
        exists = record["exists"]
        critical = result["critical_drive"]
        color = COLORS[branch]

        ax = axes[0, column]
        ax.plot(current[exists], record["activity"][exists], color=color, lw=2)
        ax.axvline(critical, color="crimson", ls="--", lw=1.3)
        ax.set_ylabel(r"$\Delta_{DV}$ equilibrium (mV)")
        ax.set_title(f"{branch}: resting equilibrium ends at a fold")
        ax.annotate(
            fr"$I_{{SN}}={critical:.4f}$ pA",
            xy=(critical, record["activity"][exists][-1]),
            xytext=(-92, -35),
            textcoords="offset points",
            arrowprops={"arrowstyle": "->", "color": "0.3"},
            fontsize=9,
        )

        ax = axes[1, column]
        refined_current = result["refined_drives"]
        refined_eigenvalue = result["refined_max_re"]
        ax.plot(refined_current, refined_eigenvalue, color=color, lw=2)
        ax.axhline(0.0, color="0.2", lw=0.8)
        ax.axvline(critical, color="crimson", ls="--", lw=1.3)
        ax.set_xlabel(f"I{branch} (pA)")
        ax.set_ylabel(r"critical Re$(\lambda)$ (ms$^{-1}$)")
        ax.set_xlim(refined_current[0], critical + 0.0002)
        ax.set_title(
            "A real eigenvalue tends to zero "
            + fr"($R^2={result['fold_fit_r2']:.4f}$)"
        )

    fig.suptitle(
        "AVA and AVB command onset: saddle-node evidence",
        fontsize=14,
    )
    fig.text(
        0.5,
        0.01,
        fr"$\beta={settings.beta:g}$, $\tau_w={settings.tau_w:g}$ ms, "
        fr"$G_E={settings.G_syne:g}$, $G_I={settings.G_syni:g}$, "
        fr"$G_{{gap}}={settings.G_gap:g}$ nS",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.035, 1, 0.95])
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_cycles(results, output):
    """Plot the finite-amplitude, infinite-period SNIC signatures."""
    fig, axes = plt.subplots(3, 2, figsize=(11.5, 10.5))
    for column, branch in enumerate(BRANCHES):
        result = results[branch]["cycle"]
        rows = result["rows"]
        offset = np.array([row["offset"] for row in rows])
        amplitude = np.array([row["amplitude"] for row in rows])
        period = np.array([row["period"] for row in rows])
        frequency = np.array([row["frequency"] for row in rows])
        color = COLORS[branch]
        near = np.arange(min(5, len(rows)))

        ax = axes[0, column]
        ax.semilogx(offset, amplitude, "o-", color=color)
        ax.set_ylabel(r"$\Delta_{DV}$ amplitude (mV)")
        ax.set_title(f"{branch}: dorsoventral output remains finite at onset")

        ax = axes[1, column]
        ax.loglog(offset, period, "o-", color=color, label="simulation")
        reference = (
            result["inverse_sqrt_slope"] / np.sqrt(offset)
            + result["inverse_sqrt_intercept"]
        )
        ax.loglog(
            offset,
            reference,
            "--",
            color="0.25",
            label=r"$T=A/\sqrt{\Delta I}+B$",
        )
        ax.set_ylabel("period T (ms)")
        ax.set_title(
            "inverse-square-root period divergence "
            + fr"($R^2={result['inverse_sqrt_r2']:.4f}$)"
        )
        ax.legend(fontsize=8)

        ax = axes[2, column]
        inverse_sqrt = 1.0 / np.sqrt(offset)
        ax.plot(inverse_sqrt, period, "o", color=color, label="simulation")
        fit_x = np.linspace(inverse_sqrt[near][-1], inverse_sqrt[near][0], 150)
        fit_y = result["inverse_sqrt_slope"] * fit_x + result["inverse_sqrt_intercept"]
        ax.plot(fit_x, fit_y, "-", color="0.2", label="near-onset fit")
        ax.set_xlabel(r"$1/\sqrt{\Delta I}$ (pA$^{-1/2}$)")
        ax.set_ylabel("period T (ms)")
        ax.set_title(
            fr"$T=A/\sqrt{{\Delta I}}+B$ "
            fr"($R^2={result['inverse_sqrt_r2']:.4f}$)"
        )
        ax.legend(fontsize=8)

    fig.suptitle(
        "AVA and AVB command onset: finite dorsoventral output, infinite period",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output, dpi=180)
    plt.close(fig)


def serializable_results(results):
    """Remove dense plotting arrays while retaining numerical diagnostics."""
    output = {
        "parameters": {
            "beta": settings.beta,
            "tau_w_ms": settings.tau_w,
            "G_syne_nS": settings.G_syne,
            "G_syni_nS": settings.G_syni,
            "G_gap_nS": settings.G_gap,
            "k_syn_per_mV": settings.k_syn,
            "V_th_mV": settings.V_th,
        }
    }
    for branch in BRANCHES:
        equilibrium = results[branch]["equilibrium"]
        cycle = results[branch]["cycle"]
        output[branch] = {
            "grid_fold_pA": equilibrium["grid_fold"],
            "fitted_I_SN_pA": equilibrium["critical_drive"],
            "fold_eigenvalue_fit_r2": equilibrium["fold_fit_r2"],
            "log_period_slope": cycle["log_period_slope"],
            "log_period_fit_r2": cycle["log_period_r2"],
            "f2_slope_Hz2_per_pA": cycle["f2_slope"],
            "f2_intercept_Hz2": cycle["f2_intercept"],
            "f2_fit_r2": cycle["f2_r2"],
            "inverse_sqrt_slope_ms_sqrt_pA": cycle["inverse_sqrt_slope"],
            "inverse_sqrt_intercept_ms": cycle["inverse_sqrt_intercept"],
            "inverse_sqrt_fit_r2": cycle["inverse_sqrt_r2"],
            "cycles": cycle["rows"],
        }
    return output


def run(beta=1.03, tau_w=400.0):
    settings.reset_defaults()
    settings.beta = float(beta)
    settings.tau_w = float(tau_w)
    segment.configure()

    results = {}
    for branch in BRANCHES:
        equilibrium = equilibrium_diagnostics(branch)
        print(
            f"{branch} saddle-node: I_SN={equilibrium['critical_drive']:.6f} pA "
            f"(eigenvalue fit R^2={equilibrium['fold_fit_r2']:.6f})"
        )
        cycle = cycle_diagnostics(branch, equilibrium["critical_drive"])
        results[branch] = {"equilibrium": equilibrium, "cycle": cycle}

    media = ROOT / "media"
    media.mkdir(exist_ok=True)
    equilibrium_figure = media / "bifurcation_drive_snic_equilibrium.png"
    cycle_figure = media / "bifurcation_drive_snic_cycles.png"
    results_file = media / "bifurcation_drive_snic_results.json"
    plot_equilibria(results, equilibrium_figure)
    plot_cycles(results, cycle_figure)
    results_file.write_text(
        json.dumps(serializable_results(results), indent=2) + "\n"
    )
    print(f"Saved -> {equilibrium_figure}")
    print(f"Saved -> {cycle_figure}")
    print(f"Saved -> {results_file}")
    return results


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--beta", type=float, default=1.03)
    parser.add_argument(
        "--tau-w",
        type=float,
        default=400.0,
        help="Recovery time constant in ms (default: canonical value 400)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(beta=arguments.beta, tau_w=arguments.tau_w)
