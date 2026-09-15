"""Analyze VA-10 and VB-8 connections across adjacent body segments."""

import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUT = ROOT / "results" / "va10_vb8_adjacent_connectivity"
SEGMENT_SIZE = 17
N_SEGMENTS = 6
FOCUS_LOCAL_INDICES = {"VB-8": 8, "VA-10": 10}
LOCAL_LABELS = (
    "AS-0", "AS-1", "DA-2", "DB-3", "DD-4", "VD-5", "VD-6",
    "VB-7", "VB-8", "VA-9", "VA-10", "D-muscle-11", "D-muscle-12",
    "D-muscle-13", "V-muscle-14", "V-muscle-15", "V-muscle-16",
)


def load_matrices():
    return {
        "excitatory": np.loadtxt(
            DATA / "ConnectivityMatrix_SixSegments_ExcitatorySynapses.txt",
            delimiter=",",
        ),
        "inhibitory": np.loadtxt(
            DATA / "ConnectivityMatrix_SixSegments_InhibitorySynapses.txt",
            delimiter=",",
        ),
        "gap junction": np.loadtxt(
            DATA / "ConnectivityMatrix_SixSegments_GapJunctions.txt",
            delimiter=",",
        ),
    }


def cell_label(index):
    segment = index // SEGMENT_SIZE + 1
    local = index % SEGMENT_SIZE
    return f"S{segment}:{LOCAL_LABELS[local]}"


def adjacent_edge_rows(matrices=None):
    """Return every intersegment edge incident to VA-10 or VB-8."""
    if matrices is None:
        matrices = load_matrices()
    rows = []
    for focus_name, local_index in FOCUS_LOCAL_INDICES.items():
        for focus_segment in range(N_SEGMENTS):
            focus = focus_segment * SEGMENT_SIZE + local_index

            for connection_type in ("excitatory", "inhibitory"):
                matrix = matrices[connection_type]
                for partner in np.flatnonzero(matrix[:, focus]):
                    partner_segment = partner // SEGMENT_SIZE
                    if partner_segment == focus_segment:
                        continue
                    rows.append(
                        edge_row(
                            focus_name,
                            focus,
                            partner,
                            connection_type,
                            "outgoing",
                            matrix[partner, focus],
                        )
                    )
                for partner in np.flatnonzero(matrix[focus]):
                    partner_segment = partner // SEGMENT_SIZE
                    if partner_segment == focus_segment:
                        continue
                    rows.append(
                        edge_row(
                            focus_name,
                            focus,
                            partner,
                            connection_type,
                            "incoming",
                            matrix[focus, partner],
                        )
                    )

            gap = matrices["gap junction"]
            for partner in np.flatnonzero(gap[focus]):
                partner_segment = partner // SEGMENT_SIZE
                if partner_segment == focus_segment:
                    continue
                rows.append(
                    edge_row(
                        focus_name,
                        focus,
                        partner,
                        "gap junction",
                        "bidirectional",
                        gap[focus, partner],
                    )
                )

    rows.sort(
        key=lambda row: (
            row["focus_cell"],
            row["focus_segment"],
            row["connection_type"],
            row["partner_cell"],
        )
    )
    return rows


def edge_row(focus_name, focus, partner, connection_type, role, weight):
    focus_segment = focus // SEGMENT_SIZE
    partner_segment = partner // SEGMENT_SIZE
    offset = partner_segment - focus_segment
    if abs(offset) != 1:
        raise ValueError("focus cell has a non-adjacent intersegment connection")
    return {
        "focus_cell": focus_name,
        "focus_segment": focus_segment + 1,
        "focus_global_index": focus,
        "role": role,
        "connection_type": connection_type,
        "partner_direction": "posterior" if offset == 1 else "anterior",
        "partner_segment": partner_segment + 1,
        "partner_cell": LOCAL_LABELS[partner % SEGMENT_SIZE],
        "partner_global_index": partner,
        "matrix_weight": float(weight),
    }


def pattern_rows(rows):
    counts = Counter(
        (
            row["focus_cell"],
            row["role"],
            row["connection_type"],
            row["partner_direction"],
            row["partner_cell"],
            row["matrix_weight"],
        )
        for row in rows
    )
    return [
        {
            "focus_cell": key[0],
            "role": key[1],
            "connection_type": key[2],
            "partner_direction": key[3],
            "partner_cell": key[4],
            "matrix_weight": key[5],
            "boundary_instances": count,
        }
        for key, count in sorted(counts.items())
    ]


def save_csv(rows, patterns, output):
    with (output / "edge_instances.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "edge_patterns.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(patterns[0]))
        writer.writeheader()
        writer.writerows(patterns)


def save_summary(rows, patterns, output):
    payload = {
        "matrix_convention": "row = postsynaptic/target; column = presynaptic/source",
        "segment_convention": "S1 is anterior and S6 is posterior",
        "edge_instances": len(rows),
        "patterns": patterns,
        "interpretation": {
            "VB-8": (
                "VB-8 couples only toward the next posterior segment: it excites "
                "DD-4 and forms gap junctions with DB-3 and VB-7. This places VB-8 "
                "directly on the anterior-to-posterior propagation pathway."
            ),
            "VA-10": (
                "VA-10 has no intersegment chemical synapse. It gap-couples only "
                "to AS-1 in the immediately anterior segment, placing it on a "
                "posterior-to-anterior electrical pathway."
            ),
        },
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")


def draw_node(axis, x, y, label, color):
    axis.scatter([x], [y], s=1700, color=color, edgecolor="0.25", zorder=3)
    axis.text(x, y, label, ha="center", va="center", fontsize=10, zorder=4)


def save_figure(patterns, output):
    figure, axes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.1, 1.5]})
    schematic, table_axis = axes
    for x, label in ((0, "segment n−1\n(anterior)"), (1, "segment n"), (2, "segment n+1\n(posterior)")):
        schematic.axvspan(x - 0.35, x + 0.35, color="0.96", zorder=0)
        schematic.text(x, 2.75, label, ha="center", va="bottom", fontsize=10)

    draw_node(schematic, 0, 1.6, "AS-1", "#4c78a8")
    draw_node(schematic, 1, 1.6, "VA-10", "#b279a2")
    schematic.plot([0.18, 0.82], [1.65, 1.65], color="#7f7f7f", lw=3)
    schematic.plot([0.18, 0.82], [1.55, 1.55], color="#7f7f7f", lw=3)

    draw_node(schematic, 1, -0.6, "VB-8", "#e45756")
    draw_node(schematic, 2, 0.5, "DD-4", "#eeca3b")
    draw_node(schematic, 2, -0.6, "DB-3", "#54a24b")
    draw_node(schematic, 2, -1.7, "VB-7", "#e45756")
    schematic.annotate(
        "",
        xy=(1.82, 0.36),
        xytext=(1.18, -0.46),
        arrowprops={"arrowstyle": "-|>", "lw": 2.5, "color": "#2f6f9f"},
    )
    for y_target in (-0.6, -1.7):
        schematic.plot([1.18, 1.82], [-0.6 + 0.04 * np.sign(y_target + 0.6), y_target + 0.04], color="#7f7f7f", lw=3)
        schematic.plot([1.18, 1.82], [-0.6 - 0.04 * np.sign(y_target + 0.6), y_target - 0.04], color="#7f7f7f", lw=3)

    schematic.text(0.5, 1.88, "gap junction", ha="center", color="0.35")
    schematic.text(1.55, 0.2, "excitatory", rotation=58, ha="center", color="#2f6f9f")
    schematic.text(1.5, -1.25, "gap junctions", ha="center", color="0.35")
    schematic.set_xlim(-0.45, 2.45)
    schematic.set_ylim(-2.35, 3.1)
    schematic.axis("off")
    schematic.set_title("Representative adjacent-segment motifs", fontsize=13)

    table_axis.axis("off")
    cells = [
        [
            row["focus_cell"],
            row["role"],
            row["connection_type"],
            row["partner_direction"],
            row["partner_cell"],
            row["boundary_instances"],
        ]
        for row in patterns
    ]
    table = table_axis.table(
        cellText=cells,
        colLabels=("focus", "role", "type", "partner side", "partner", "n"),
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.7)
    table_axis.set_title("Pattern repeated at the five internal boundaries", fontsize=13, pad=18)
    figure.suptitle("VA-10 and VB-8 connectivity to adjacent segments", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output / "adjacent_connectivity.png", dpi=200)
    plt.close(figure)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = adjacent_edge_rows()
    patterns = pattern_rows(rows)
    save_csv(rows, patterns, OUTPUT)
    save_summary(rows, patterns, OUTPUT)
    save_figure(patterns, OUTPUT)
    print(f"Found {len(rows)} adjacent-segment edge instances in {len(patterns)} patterns")
    for pattern in patterns:
        print(pattern)
    print(f"Saved results to {OUTPUT}")


if __name__ == "__main__":
    main()
