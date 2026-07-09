import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from direct_assembly_graph import build_pair_candidates, select_direct_connections


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


if __name__ == "__main__":
    unittest.main()
