import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from joinable_candidate_provider import select_joinable_pairs


def _row(part_a, part_b, lift, *, status="success"):
    return {
        "pair_id": f"pool:{part_a}:{part_b}",
        "pool_id": "pool",
        "part_a": part_a,
        "part_b": part_b,
        "status": status,
        "pair_features": {
            "top_1_uniform_lift": lift,
            "top_2_logit_margin": lift / 10,
            "normalized_entropy": 0.5,
        },
        "candidates": [
            {
                "rank": 1,
                "part_a_entity": {"entity_type": "face", "topology_index": 1},
                "part_b_entity": {"entity_type": "face", "topology_index": 2},
                "softmax_probability": 0.8,
            }
        ],
    }


class JoinableCandidateProviderTests(unittest.TestCase):
    def test_selects_bounded_recall_frontier_without_auto_accept(self):
        report = {
            "pairs": [
                _row("a.step", "b.step", 5.0),
                _row("a.step", "c.step", 4.0),
                _row("a.step", "d.step", 0.5),
            ]
        }
        selected, audit = select_joinable_pairs(
            report,
            pool_id="pool",
            part_ids=["a.step", "b.step", "c.step", "d.step"],
            maximum_neighbors_per_part=1,
            minimum_uniform_lift=1.0,
        )
        self.assertIn(("a.step", "b.step"), selected)
        self.assertIn(("a.step", "c.step"), selected)
        self.assertNotIn(("a.step", "d.step"), selected)
        self.assertFalse(
            audit["model_boundary"]["pair_connection_decision_available"]
        )
        self.assertFalse(audit["model_boundary"]["provider_can_auto_accept"])
        self.assertTrue(
            all(
                not row["creates_physical_evidence"]
                and not row["can_auto_accept"]
                for row in audit["selected_pairs"]
            )
        )

    def test_ignores_failed_and_foreign_pool_rows(self):
        report = {
            "pairs": [
                _row("a", "b", 9.0, status="inference_failed"),
                {**_row("a", "c", 9.0), "pool_id": "other"},
            ]
        }
        selected, audit = select_joinable_pairs(
            report,
            pool_id="pool",
            part_ids=["a", "b", "c"],
            maximum_neighbors_per_part=2,
        )
        self.assertEqual(selected, set())
        self.assertEqual(audit["selected_pair_count"], 0)
        self.assertEqual(len(audit["ignored_rows"]), 1)


if __name__ == "__main__":
    unittest.main()
