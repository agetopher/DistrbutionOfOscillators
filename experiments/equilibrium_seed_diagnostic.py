"""Show how single-seed equilibrium continuation misses valid branches.

This is a solver diagnostic, not a replacement for pseudo-arclength
continuation. It compares the legacy 5.5-pA trajectory seed used by
``bifurcations.locate_upper_hopf`` with equilibria discovered independently by
many seeds at 3 pA and then continued in both drive directions.
"""

from argparse import ArgumentParser
import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import bifurcations as bif  # noqa: E402
import segment  # noqa: E402
import settings  # noqa: E402


OUTPUT = ROOT / "media" / "equilibrium_continuation_seed_diagnostic.png"
BRANCH_COLORS = ("#7b3294", "#008837", "#e66101", "#5e3c99")


def root_metrics(branch, drive, state):
    current = segment.drive_along(branch, drive)
    eigenvalues = np.linalg.eigvals(bif.jac_Vw(state, current))
    leading = eigenvalues[np.argmax(eigenvalues.real)]
    return {
        "drive_pA": float(drive),
        "dv_equilibrium_mV": float(bif.dv_activity(state[: segment.N_CELLS])),
        "mean_voltage_mV": float(state[: segment.N_CELLS].mean()),
        "max_real_eigenvalue_per_ms": float(leading.real),
        "leading_imaginary_magnitude_per_ms": float(abs(leading.imag)),
        "unstable_modes": int((eigenvalues.real > 1e-9).sum()),
        "max_rhs_residual": float(np.abs(bif.rhs_Vw(state, current)).max()),
    }


def discover_roots(branch, drive=3.0):
    """Find distinct roots from limit-cycle snapshots and uniform states."""
    current = segment.drive_along(branch, drive)
    trajectory = solve_ivp(
        lambda _, state: segment.rhs_vw(state, current),
        (0.0, 15000.0),
        segment.reduced_rest_state(),
        method="BDF",
        t_eval=np.linspace(9000.0, 15000.0, 121),
    )
    if not trajectory.success:
        raise RuntimeError(trajectory.message)

    seeds = [trajectory.y[:, index]
             for index in range(trajectory.y.shape[1])]
    for voltage in np.linspace(-80.0, 0.0, 33):
        uniform_voltage = np.full(segment.N_CELLS, voltage)
        recovery = np.maximum(
            settings.beta * (uniform_voltage - settings.T), 0.0
        )
        seeds.append(np.concatenate([uniform_voltage, recovery]))

    roots = []
    for seed in seeds:
        root = bif.equilibrium(current, seed, tol=1e-7)
        if root is None:
            continue
        if all(np.linalg.norm(root - known) > 1e-2 for known in roots):
            roots.append(root)
    return roots


def continue_from_anchor(branch, anchor, anchor_drive=3.0, step=0.025,
                         low=1.5, high=6.0):
    """Natural continuation away from one independently discovered root."""
    records = []
    for direction, drives in (
        ("down", np.arange(anchor_drive, low - 0.5 * step, -step)),
        ("up", np.arange(anchor_drive + step, high + 0.5 * step, step)),
    ):
        state = anchor.copy()
        for drive in drives:
            root = bif.equilibrium(
                segment.drive_along(branch, float(drive)), state, tol=1e-7
            )
            if root is None:
                break
            state = root
            row = root_metrics(branch, drive, state)
            row["direction"] = direction
            records.append(row)
    records.sort(key=lambda row: row["drive_pA"])
    return records


def rest_branch(branch, step=0.025):
    fold, _ = bif.locate_fold(branch, hi=3.0, step=0.001)
    drives = np.arange(0.0, fold + 0.5 * step, step)
    state = segment.reduced_rest_state()
    records = []
    for drive in drives:
        root = bif.equilibrium(segment.drive_along(branch, drive), state,
                               tol=1e-7)
        if root is None:
            break
        state = root
        records.append(root_metrics(branch, drive, state))
    return fold, records


def legacy_branch(branch):
    """Reproduce the single trajectory-seeded continuation used in the plot."""
    _, record = bif.locate_upper_hopf(
        branch,
        seed_drive=5.5,
        drives=np.arange(5.5, 1.5, -0.05),
        tf=15000.0,
    )
    rows = []
    for index, exists in enumerate(record["exists"]):
        if not exists:
            continue
        rows.append({
            "drive_pA": float(record["drives"][index]),
            "dv_equilibrium_mV": float(record["activity"][index]),
            "max_real_eigenvalue_per_ms": float(record["max_re"][index]),
            "unstable_modes": int(record["n_unstable"][index]),
        })
    rows.sort(key=lambda row: row["drive_pA"])
    return rows


def write_rows(path, results):
    fields = (
        "command_branch", "discovery_method", "root_id", "drive_pA",
        "dv_equilibrium_mV", "mean_voltage_mV",
        "max_real_eigenvalue_per_ms", "leading_imaginary_magnitude_per_ms",
        "unstable_modes", "max_rhs_residual",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for branch, result in results.items():
            for row in result["rest"]:
                writer.writerow({
                    "command_branch": branch,
                    "discovery_method": "rest continuation",
                    "root_id": "rest",
                    **{field: row.get(field, "") for field in fields[3:]},
                })
            for root_id, records in enumerate(result["multistart"]):
                for row in records:
                    writer.writerow({
                        "command_branch": branch,
                        "discovery_method": "multistart at 3 pA",
                        "root_id": root_id,
                        **{field: row.get(field, "") for field in fields[3:]},
                    })
            for row in result["legacy"]:
                writer.writerow({
                    "command_branch": branch,
                    "discovery_method": "legacy 5.5 pA trajectory seed",
                    "root_id": "legacy",
                    **{field: row.get(field, "") for field in fields[3:]},
                })


def plot(results, output):
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), sharex="col")
    for column, branch in enumerate(("AVA", "AVB")):
        result = results[branch]
        fold = result["fold"]

        top = axes[0, column]
        bottom = axes[1, column]
        rest = result["rest"]
        top.plot(
            [row["drive_pA"] for row in rest],
            [row["dv_equilibrium_mV"] for row in rest],
            color="0.15", lw=2.0, label="rest branch",
        )
        bottom.plot(
            [row["drive_pA"] for row in rest],
            [row["max_real_eigenvalue_per_ms"] for row in rest],
            color="0.15", lw=2.0, label="rest branch",
        )

        legacy = result["legacy"]
        if legacy:
            top.plot(
                [row["drive_pA"] for row in legacy],
                [row["dv_equilibrium_mV"] for row in legacy],
                "x", color="0.55", ms=4.5,
                label="legacy 5.5 pA seed",
            )
            bottom.plot(
                [row["drive_pA"] for row in legacy],
                [row["max_real_eigenvalue_per_ms"] for row in legacy],
                "x", color="0.55", ms=4.5,
                label="legacy 5.5 pA seed",
            )
        else:
            top.text(
                4.55, 0.70, "legacy seed found\nno AVA roots",
                transform=top.get_xaxis_transform(), ha="center", va="top",
                color="0.45", fontsize=9,
            )

        for root_id, records in enumerate(result["multistart"]):
            color = BRANCH_COLORS[root_id % len(BRANCH_COLORS)]
            label = f"multistart root {root_id + 1}"
            drives = [row["drive_pA"] for row in records]
            top.plot(
                drives,
                [row["dv_equilibrium_mV"] for row in records],
                "o--", color=color, ms=2.6, lw=1.2, label=label,
            )
            bottom.plot(
                drives,
                [row["max_real_eigenvalue_per_ms"] for row in records],
                "o--", color=color, ms=2.6, lw=1.2, label=label,
            )
            anchor = min(records, key=lambda row: abs(row["drive_pA"] - 3.0))
            all_dv = [
                row["dv_equilibrium_mV"]
                for branch_records in result["multistart"]
                for row in branch_records
            ]
            high_branch = anchor["dv_equilibrium_mV"] > (
                min(all_dv) + 0.65 * (max(all_dv) - min(all_dv))
            )
            text_offset = (12, -34) if high_branch else (12, 14)
            top.annotate(
                f"{anchor['unstable_modes']} unstable modes\n"
                f"residual ≤ {anchor['max_rhs_residual']:.1e}",
                xy=(anchor["drive_pA"], anchor["dv_equilibrium_mV"]),
                xytext=text_offset, textcoords="offset points",
                fontsize=7.5, color=color,
                arrowprops={"arrowstyle": "->", "color": color, "lw": 0.8},
            )

        for axis in (top, bottom):
            axis.axvline(fold, color="crimson", ls=":", lw=1.2)
            axis.grid(alpha=0.18)
            axis.set_xlim(0.0, 6.0)
        bottom.axhline(0.0, color="crimson", lw=0.9)
        top.set_title(f"{branch}: equilibrium branches found by different seeds")
        top.set_ylabel(r"equilibrium $\Delta_{DV}$ (mV)")
        bottom.set_ylabel(r"max Re$(\lambda)$ (ms$^{-1}$)")
        bottom.set_xlabel(f"I{branch} drive (pA)")
        bottom.set_title("Positive eigenvalue confirms instability")
        top.legend(fontsize=7.5, loc="best")

    figure.suptitle(
        "Why the single-seed bifurcation plot misses equilibria\n"
        r"multistart roots have small $\|F(x^*)\|_\infty$ and positive "
        r"Re$(\lambda)$; absence from one continuation is not nonexistence"
        fr"   |   $\beta={settings.beta:g}$, $\tau_w={settings.tau_w:g}$ ms",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(output, dpi=190)
    figure.savefig(output.with_suffix(".svg"))
    plt.close(figure)


def run(output=OUTPUT):
    settings.reset_defaults()
    settings.beta = 1.03
    settings.tau_w = 400.0
    segment.configure()

    results = {}
    for branch in ("AVA", "AVB"):
        roots = discover_roots(branch, drive=3.0)
        fold, rest = rest_branch(branch)
        results[branch] = {
            "fold": fold,
            "rest": rest,
            "legacy": legacy_branch(branch),
            "multistart": [continue_from_anchor(branch, root) for root in roots],
        }
        print(
            f"{branch}: multistart found {len(roots)} root(s) at 3 pA; "
            f"legacy continuation retained {len(results[branch]['legacy'])} point(s)"
        )

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plot(results, output)
    table = output.with_name(f"{output.stem}_points.csv")
    write_rows(table, results)
    print(f"Saved -> {output}")
    print(f"Saved -> {output.with_suffix('.svg')}")
    print(f"Saved -> {table}")


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.output)
