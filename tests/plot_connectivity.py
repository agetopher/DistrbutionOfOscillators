import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

MEDIA_DIR = os.path.join(os.path.dirname(__file__), '..', 'media')
DATA_DIR  = os.path.join(os.path.dirname(__file__), '..', 'data')

# Load and transpose matrices (rows = presynaptic, cols = postsynaptic)
E   = np.loadtxt(os.path.join(DATA_DIR, 'ConnectivityMatrix_SixSegments_ExcitatorySynapses.txt'), delimiter=',').T
I   = np.loadtxt(os.path.join(DATA_DIR, 'ConnectivityMatrix_SixSegments_InhibitorySynapses.txt'), delimiter=',').T
G   = np.loadtxt(os.path.join(DATA_DIR, 'ConnectivityMatrix_SixSegments_GapJunctions.txt'),       delimiter=',').T
cls = np.loadtxt(os.path.join(DATA_DIR, 'CellsClassification.dat')).astype(int)

classes = np.unique(cls)
n_cls   = len(classes)
counts  = [int((cls == c).sum()) for c in classes]

# Build stats table: rows = metrics, cols = classes
metrics      = ['E out', 'E in', 'I out', 'I in', 'Gap junctions']
metric_colors = ['#2196F3', '#90CAF9', '#F44336', '#EF9A9A', '#4CAF50']
data = np.zeros((len(metrics), n_cls))

for j, c in enumerate(classes):
    idx = np.where(cls == c)[0]
    data[0, j] = E[idx, :].sum(axis=1).mean()   # E out
    data[1, j] = E[:, idx].sum(axis=0).mean()   # E in
    data[2, j] = I[idx, :].sum(axis=1).mean()   # I out
    data[3, j] = I[:, idx].sum(axis=0).mean()   # I in
    data[4, j] = G[idx, :].sum(axis=1).mean()   # GJ (symmetric)


def run(save=True):
    fig = plt.figure(figsize=(13, 8))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[2.5, 1], hspace=0.45)

    # ── Top: grouped bar chart ───────────────────────────────────────────────
    ax_bar = fig.add_subplot(gs[0])

    x       = np.arange(n_cls)
    n_met   = len(metrics)
    width   = 0.15
    offsets = np.linspace(-(n_met - 1) / 2, (n_met - 1) / 2, n_met) * width

    for i, (label, color, offset) in enumerate(zip(metrics, metric_colors, offsets)):
        ax_bar.bar(x + offset, data[i], width=width, label=label, color=color, edgecolor='white', linewidth=0.5)

    # Annotate cell count below each class label
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([f"Class {c}\n(N={n})" for c, n in zip(classes, counts)], fontsize=9)
    ax_bar.set_ylabel("Mean connections per cell")
    ax_bar.set_title("Connectivity by Cell Class  —  Six-Segment C. elegans HPC Network", fontsize=12, fontweight='bold')
    ax_bar.legend(loc='upper left', framealpha=0.9, fontsize=9)
    ax_bar.set_xlim(-0.6, n_cls - 0.4)
    ax_bar.spines[['top', 'right']].set_visible(False)
    ax_bar.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax_bar.set_axisbelow(True)

    # ── Bottom: heatmap of same data ─────────────────────────────────────────
    ax_hm = fig.add_subplot(gs[1])

    cmap = LinearSegmentedColormap.from_list('wblue', ['#FFFFFF', '#1565C0'])
    im   = ax_hm.imshow(data, aspect='auto', cmap=cmap, vmin=0, vmax=data.max())

    ax_hm.set_xticks(x)
    ax_hm.set_xticklabels([f"Class {c}" for c in classes], fontsize=9)
    ax_hm.set_yticks(range(len(metrics)))
    ax_hm.set_yticklabels(metrics, fontsize=9)
    ax_hm.set_title("Heatmap view", fontsize=10)

    # Annotate each cell with its value
    for i in range(len(metrics)):
        for j in range(n_cls):
            val = data[i, j]
            if val > 0:
                text_color = 'white' if val > data.max() * 0.6 else 'black'
                ax_hm.text(j, i, f"{val:.1f}", ha='center', va='center',
                           fontsize=8, color=text_color, fontweight='bold')

    fig.colorbar(im, ax=ax_hm, orientation='vertical', label='Mean connections', pad=0.02)

    if save:
        plt.savefig(os.path.join(MEDIA_DIR, 'connectivity_by_class.png'), dpi=150, bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    run()
