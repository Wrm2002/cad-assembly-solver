import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from direct_assembly_graph import (
    build_pair_candidates,
    enumerate_connection_topologies,
    select_direct_connections,
)


class DirectAssemblyGraphTests(unittest.TestCase):
    def _match(self, a, b, relation, score):
        return {
            "type": relation,
            "parts": (a, b),
            "feat_a_idx": 0,
            "feat_b_idx": 0,
            "score": score,
            "confidence": "high" if score >= 0.75 else "medium",
        }

    def test_global_skeleton_connects_all_known_parts(self):
        matches = [
            self._match("a", "b", "clearance", 0.95),
            self._match("b", "c", "planar_mate", 0.85),
            self._match("a", "c", "planar_align", 0.60),
        ]
        pairs = build_pair_candidates(matches)
        result = select_direct_connections(["a", "b", "c"], pairs)
        self.assertTrue(result["connected"])
        skeleton = {
            tuple(row["parts"])
            for row in result["selected"]
            if row["selection_role"] == "connected_skeleton"
        }
        self.assertEqual(skeleton, {("a", "b"), ("b", "c")})

    def test_multiple_constraints_support_one_direct_connection(self):
        matches = [
            self._match("flange_a", "flange_b", "coaxial", 0.90),
            self._match("flange_a", "flange_b", "planar_mate", 0.88),
        ]
        pairs = build_pair_candidates(matches)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(
            set(pairs[0]["relation_types"]),
            {"coaxial", "planar_mate"},
        )

    def test_disconnected_candidates_are_reported_not_fabricated(self):
        pairs = build_pair_candidates([
            self._match("a", "b", "coaxial", 0.9),
        ])
        result = select_direct_connections(["a", "b", "c"], pairs)
        self.assertFalse(result["connected"])
        self.assertEqual(result["unresolved_parts"], ["c"])

    def test_topology_frontier_keeps_lower_scoring_correct_star(self):
        # Mirrors the case4 failure mode: a strong accidental satellite edge
        # must not delete the main-board star before pose closure is checked.
        pairs = build_pair_candidates([
            self._match("pcba", "dimm", "planar_mate", 0.856),
            self._match("dimm", "cpu", "planar_mate", 0.872),
            self._match("pcba", "cpu", "pocket_mate", 0.783),
        ])
        frontier = enumerate_connection_topologies(
            ["pcba", "dimm", "cpu"], pairs, maximum=3
        )
        self.assertEqual(len(frontier), 3)
        pair_sets = [
            {tuple(pair) for pair in topology["pairs"]}
            for topology in frontier
        ]
        self.assertIn(
            {("dimm", "pcba"), ("cpu", "pcba")},
            pair_sets,
        )
        self.assertTrue(all(row["pose_validation_required"] for row in frontier))

    def test_selector_exposes_bounded_topology_frontier(self):
        pairs = build_pair_candidates([
            self._match("a", "b", "coaxial", 0.9),
            self._match("a", "c", "planar_mate", 0.8),
            self._match("b", "c", "clearance", 0.85),
        ])
        result = select_direct_connections(
            ["a", "b", "c"], pairs, topology_limit=2
        )
        self.assertEqual(result["topology_frontier_count"], 2)
        self.assertEqual(result["topology_frontier"][0]["rank"], 1)


if __name__ == "__main__":
    unittest.main()
