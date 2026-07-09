import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from match_pruning import prune_match_graph


def mate(kind, a, b, score, ia=0, ib=0):
    return {
        "type": kind,
        "parts": (a, b),
        "feat_a_idx": ia,
        "feat_b_idx": ib,
        "score": score,
        "confidence": "high" if score >= 0.75 else "medium",
        "reason": {},
    }


class MatchPruningTests(unittest.TestCase):
    def test_low_score_and_duplicate_are_audited(self):
        matches = [
            mate("coaxial", "a", "b", 0.9),
            mate("coaxial", "a", "b", 0.8),
            mate("clearance", "a", "b", 0.2, 1, 1),
        ]
        kept, removed = prune_match_graph(matches)
        self.assertEqual(len(kept), 1)
        self.assertEqual(
            {item["removal_reason"] for item in removed},
            {"duplicate", "low_score"},
        )

    def test_strong_edge_dominates_planar_align(self):
        matches = [
            mate("clearance", "a", "b", 0.8),
            mate("planar_align", "a", "b", 0.9, 1, 1),
        ]
        kept, removed = prune_match_graph(matches)
        self.assertEqual([item["type"] for item in kept], ["clearance"])
        self.assertEqual(removed[0]["removal_reason"], "weak_planar_only")

    def test_planar_only_pair_keeps_one_edge(self):
        matches = [
            mate("planar_mate", "a", "b", 0.8),
            mate("planar_align", "a", "b", 0.7, 1, 1),
        ]
        kept, removed = prune_match_graph(matches)
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(removed), 1)
        self.assertEqual(removed[0]["removal_reason"], "weak_planar_only")

    def test_reliable_pose_mode_can_keep_multiple_planar_hypotheses(self):
        matches = [
            {
                "type": "planar_mate",
                "parts": ["a", "b"],
                "feat_a_idx": index,
                "feat_b_idx": 0,
                "score": 0.9 - index * 0.01,
            }
            for index in range(4)
        ]
        kept, removed = prune_match_graph(
            matches,
            top_k_pair=4,
            planar_hypotheses_per_pair=3,
        )
        self.assertEqual(len(kept), 3)
        self.assertEqual(len(removed), 1)
        self.assertEqual(removed[0]["removal_reason"], "weak_planar_only")
        self.assertEqual(removed[0]["removal_reason"], "weak_planar_only")

    def test_top_k_pair(self):
        matches = [
            mate("coaxial", "a", "b", 0.9, 0, 0),
            mate("coaxial", "a", "b", 0.8, 1, 1),
            mate("clearance", "a", "b", 0.7, 2, 2),
        ]
        kept, removed = prune_match_graph(matches, top_k_pair=2)
        self.assertEqual(len(kept), 2)

    def test_strong_pair_reserves_planar_pose_hypotheses(self):
        matches = [
            mate("coaxial", "a", "b", 0.99, 0, 0),
            mate("coaxial", "a", "b", 0.98, 1, 1),
            mate("coaxial", "a", "b", 0.97, 2, 2),
            mate("planar_mate", "a", "b", 0.96, 3, 3),
            mate("planar_mate", "a", "b", 0.95, 4, 4),
        ]
        kept, _ = prune_match_graph(
            matches,
            min_score=0.0,
            top_k_pair=4,
            max_neighbors=4,
            planar_hypotheses_per_pair=2,
        )
        self.assertEqual(
            sum(match["type"] == "planar_mate" for match in kept),
            2,
        )
        self.assertEqual(
            sum(match["type"] == "coaxial" for match in kept),
            2,
        )

    def test_max_neighbors(self):
        matches = [
            mate("coaxial", "a", "b", 0.9),
            mate("coaxial", "a", "c", 0.8),
        ]
        kept, removed = prune_match_graph(matches, max_neighbors=1)
        self.assertEqual(len(kept), 1)
        self.assertEqual(removed[0]["removal_reason"], "exceeded_max_neighbors")


if __name__ == "__main__":
    unittest.main()
