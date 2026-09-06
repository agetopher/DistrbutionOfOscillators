import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import segment_connectivity_diagram as diagram  # noqa: E402


def test_diagram_edges_match_single_segment_matrices():
    excitatory, inhibitory, gap = diagram.model_edges()

    assert len(excitatory) == 39
    assert len(inhibitory) == 12
    assert len(gap) == 8
    assert all(source < target for source, target, _ in gap)
    assert set(diagram.cell_labels()) == set(range(17))
