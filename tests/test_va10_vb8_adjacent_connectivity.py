import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from va10_vb8_adjacent_connectivity import adjacent_edge_rows, pattern_rows


def test_va10_vb8_adjacent_edge_patterns_match_full_matrices():
    patterns = pattern_rows(adjacent_edge_rows())
    compact = {
        (
            row["focus_cell"],
            row["role"],
            row["connection_type"],
            row["partner_direction"],
            row["partner_cell"],
        ): row["boundary_instances"]
        for row in patterns
    }

    assert compact == {
        ("VA-10", "bidirectional", "gap junction", "anterior", "AS-1"): 5,
        ("VB-8", "bidirectional", "gap junction", "posterior", "DB-3"): 5,
        ("VB-8", "bidirectional", "gap junction", "posterior", "VB-7"): 5,
        ("VB-8", "outgoing", "excitatory", "posterior", "DD-4"): 5,
    }


def test_all_adjacent_weights_are_one():
    assert {row["matrix_weight"] for row in adjacent_edge_rows()} == {1.0}
