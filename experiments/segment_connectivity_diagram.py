"""Draw the active 17-cell single-segment circuit from the model matrices.

The left panel is a pathway-oriented network diagram. The right panel is an
exact source-to-target matrix so dense or overlapping graph edges remain
auditable. Chemical matrices use row=post, column=pre; symmetric electrical
entries are collapsed to one undirected edge in the network panel.
"""

from argparse import ArgumentParser
import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import segment  # noqa: E402
import settings  # noqa: E402


OUTPUT = ROOT / "media" / "single_segment_connectivity.png"
EDGE_COLORS = {
    "excitatory": "#2878b5",
    "inhibitory": "#d64b3c",
    "gap junction": "#238b45",
    "command drive": "#7b3294",
}
NODE_COLORS = {
    "AS": "#4c78a8",
    "DA/DB": "#72b7b2",
    "DD/VD": "#f2b134",
    "VA/VB": "#b279a2",
    "dorsal muscle": "#8ecae6",
    "ventral muscle": "#fb6a6a",
}


def cell_labels():
    base = {
        1: "AS",
        2: "DA",
        3: "DB",
        4: "DD",
        5: "VD",
        6: "VB",
        7: "VA",
        8: "D muscle",
        9: "V muscle",
    }
    return {index: f"{base[int(cell_class)]}-{index}"
            for index, cell_class in enumerate(segment.CLASSES)}


def node_group(cell_class):
    cell_class = int(cell_class)
    if cell_class == 1:
        return "AS"
    if cell_class in (2, 3):
        return "DA/DB"
    if cell_class in (4, 5):
        return "DD/VD"
    if cell_class in (6, 7):
        return "VA/VB"
    if cell_class == 8:
        return "dorsal muscle"
    return "ventral muscle"


def model_edges():
    excitatory = [
        (int(pre), int(post), float(segment.E_CONN[post, pre]))
        for post, pre in np.argwhere(segment.E_CONN != 0.0)
    ]
    inhibitory = [
        (int(pre), int(post), float(segment.I_CONN[post, pre]))
        for post, pre in np.argwhere(segment.I_CONN != 0.0)
    ]

    if not np.allclose(segment.GJ_CONN, segment.GJ_CONN.T):
        raise ValueError("Single-segment gap-junction matrix is not symmetric")
    gap = [
        (pre, post, float(segment.GJ_CONN[post, pre]))
        for post in range(segment.N_CELLS)
        for pre in range(post)
        if segment.GJ_CONN[post, pre] != 0.0
    ]
    return excitatory, inhibitory, gap


def pathway_layout():
    return {
        11: (0.5, 3.5), 12: (2.0, 3.5), 13: (3.5, 3.5),
        0: (-1.0, 1.7), 1: (0.0, 1.7), 2: (1.1, 1.7),
        3: (2.25, 1.7), 4: (3.5, 1.7),
        5: (1.7, 0.15), 6: (3.0, 0.15),
        7: (0.2, -1.45), 8: (1.35, -1.45), 9: (2.5, -1.45),
        10: (3.65, -1.45),
        14: (0.5, -3.25), 15: (2.0, -3.25), 16: (3.5, -3.25),
        "AVA": (-3.1, 0.65), "AVB": (-3.1, -0.65),
    }


def draw_network(axis, excitatory, inhibitory, gap):
    labels = cell_labels()
    positions = pathway_layout()
    graph = nx.DiGraph()
    graph.add_nodes_from(range(segment.N_CELLS))

    # Electrical edges are drawn first because they carry no direction arrow.
    nx.draw_networkx_edges(
        graph,
        positions,
        ax=axis,
        edgelist=[(pre, post) for pre, post, _ in gap],
        edge_color=EDGE_COLORS["gap junction"],
        width=1.5,
        style=(0, (4, 2)),
        alpha=0.75,
        arrows=False,
    )
    nx.draw_networkx_edges(
        graph,
        positions,
        ax=axis,
        edgelist=[(pre, post) for pre, post, _ in excitatory],
        edge_color=EDGE_COLORS["excitatory"],
        width=0.9,
        alpha=0.48,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=10,
        min_source_margin=12,
        min_target_margin=12,
        connectionstyle="arc3,rad=0.055",
    )
    nx.draw_networkx_edges(
        graph,
        positions,
        ax=axis,
        edgelist=[(pre, post) for pre, post, _ in inhibitory],
        edge_color=EDGE_COLORS["inhibitory"],
        width=1.15,
        alpha=0.75,
        arrows=True,
        arrowstyle="-[",
        arrowsize=8,
        min_source_margin=12,
        min_target_margin=12,
        connectionstyle="arc3,rad=-0.09",
    )

    for group, color in NODE_COLORS.items():
        nodes = [index for index, cell_class in enumerate(segment.CLASSES)
                 if node_group(cell_class) == group]
        muscles = group.endswith("muscle")
        nx.draw_networkx_nodes(
            graph,
            positions,
            ax=axis,
            nodelist=nodes,
            node_color=color,
            node_shape="s" if muscles else "o",
            node_size=760 if muscles else 680,
            edgecolors="0.18",
            linewidths=0.8,
        )
    nx.draw_networkx_labels(
        graph,
        positions,
        labels={index: label.replace("-", "\n", 1)
                for index, label in labels.items()},
        ax=axis,
        font_size=7.2,
    )

    command_graph = nx.DiGraph()
    command_graph.add_nodes_from(("AVA", "AVB"))
    nx.draw_networkx_nodes(
        command_graph,
        positions,
        ax=axis,
        node_color="#efe6f2",
        node_shape="D",
        node_size=820,
        edgecolors=EDGE_COLORS["command drive"],
        linewidths=1.3,
    )
    nx.draw_networkx_labels(
        command_graph,
        positions,
        labels={"AVA": "IAVA", "AVB": "IAVB"},
        ax=axis,
        font_size=8,
        font_weight="bold",
    )
    command_edges = [
        ("AVA", cell) for cell in np.where(
            np.isin(segment.CLASSES, segment.AVA_CLASSES)
        )[0]
    ] + [
        ("AVB", cell) for cell in np.where(
            np.isin(segment.CLASSES, segment.AVB_CLASSES)
        )[0]
    ]
    nx.draw_networkx_edges(
        nx.DiGraph(command_edges),
        positions,
        ax=axis,
        edgelist=command_edges,
        edge_color=EDGE_COLORS["command drive"],
        width=1.0,
        style=(0, (2, 2)),
        alpha=0.55,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=9,
        min_source_margin=15,
        min_target_margin=12,
        connectionstyle="arc3,rad=0.08",
    )

    axis.text(2.0, 4.15, "dorsal muscle", ha="center", fontsize=9,
              color="#2878b5", fontweight="bold")
    axis.text(2.0, -3.95, "ventral muscle", ha="center", fontsize=9,
              color="#c92f2f", fontweight="bold")
    axis.set_xlim(-3.8, 4.35)
    axis.set_ylim(-4.25, 4.45)
    axis.set_aspect("equal")
    axis.axis("off")
    axis.set_title("Pathway-oriented circuit graph", fontsize=12)


def draw_exact_matrix(axis):
    labels = cell_labels()
    n = segment.N_CELLS
    axis.set_xlim(-0.5, n - 0.5)
    axis.set_ylim(n - 0.5, -0.5)
    axis.set_aspect("equal")
    axis.set_xticks(range(n), [labels[i] for i in range(n)], rotation=90,
                    fontsize=6.5)
    axis.set_yticks(range(n), [labels[i] for i in range(n)], fontsize=6.5)
    axis.set_xlabel("presynaptic/source cell")
    axis.set_ylabel("postsynaptic/target cell")
    axis.set_title("Exact model matrices (row = target, column = source)",
                   fontsize=12)
    axis.set_xticks(np.arange(-0.5, n, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, n, 1), minor=True)
    axis.grid(which="minor", color="0.88", linewidth=0.45)
    axis.tick_params(which="minor", bottom=False, left=False)

    for post, pre in np.argwhere(segment.E_CONN != 0.0):
        axis.scatter(pre, post, marker=">", s=24,
                     color=EDGE_COLORS["excitatory"], zorder=3)
    for post, pre in np.argwhere(segment.I_CONN != 0.0):
        axis.scatter(pre, post, marker="s", s=18,
                     color=EDGE_COLORS["inhibitory"], zorder=4)
    for post, pre in np.argwhere(segment.GJ_CONN != 0.0):
        axis.scatter(pre, post, marker="o", s=28, facecolors="none",
                     edgecolors=EDGE_COLORS["gap junction"], linewidths=1.0,
                     zorder=5)


def save_edge_list(path, excitatory, inhibitory, gap):
    labels = cell_labels()
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("source_index", "source", "target_index", "target",
                        "connection_type", "matrix_weight"),
        )
        writer.writeheader()
        for connection_type, edges in (
            ("excitatory", excitatory),
            ("inhibitory", inhibitory),
            ("gap junction (undirected)", gap),
        ):
            for source, target, weight in edges:
                writer.writerow({
                    "source_index": source,
                    "source": labels[source],
                    "target_index": target,
                    "target": labels[target],
                    "connection_type": connection_type,
                    "matrix_weight": weight,
                })


def run(output=OUTPUT):
    settings.reset_defaults()
    segment.configure()
    excitatory, inhibitory, gap = model_edges()

    figure, axes = plt.subplots(1, 2, figsize=(20, 10),
                                gridspec_kw={"width_ratios": (1.08, 1.0)})
    draw_network(axes[0], excitatory, inhibitory, gap)
    draw_exact_matrix(axes[1])

    edge_legend = [
        Line2D([0], [0], color=EDGE_COLORS["excitatory"], lw=1.6,
               marker=">", markevery=[1], label="excitatory chemical"),
        Line2D([0], [0], color=EDGE_COLORS["inhibitory"], lw=1.6,
               marker="s", markevery=[1], label="inhibitory chemical"),
        Line2D([0], [0], color=EDGE_COLORS["gap junction"], lw=1.6,
               ls="--", marker="o", markerfacecolor="none",
               label="gap junction (electrical)"),
        Line2D([0], [0], color=EDGE_COLORS["command drive"], lw=1.4,
               ls=":", label="applied command drive"),
    ]
    figure.legend(handles=edge_legend, loc="lower center", ncol=4,
                  fontsize=9, frameon=False)
    figure.suptitle(
        "Single-segment C. elegans motor-circuit connectivity\n"
        f"17 cells · {len(excitatory)} excitatory · {len(inhibitory)} inhibitory · "
        f"{len(gap)} electrical pairs   |   "
        f"$G_E={settings.G_syne:g}$, $G_I={settings.G_syni:g}$, "
        f"$G_{{gap}}={settings.G_gap:g}$ nS",
        fontsize=15,
    )
    figure.tight_layout(rect=(0, 0.055, 1, 0.93))

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200)
    figure.savefig(output.with_suffix(".svg"))
    plt.close(figure)
    save_edge_list(output.with_name(f"{output.stem}_edges.csv"),
                   excitatory, inhibitory, gap)
    print(f"Saved -> {output}")
    print(f"Saved -> {output.with_suffix('.svg')}")
    print(f"Saved -> {output.with_name(f'{output.stem}_edges.csv')}")


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.output)
