import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contracts import GroupProposal
from global_grouping import assign_groups, evaluate_groups


CONFIG = {"minimum_group_score": 0.5}


def proposal(group_id, parts, score):
    return GroupProposal(
        group_id=group_id,
        parts=parts,
        candidate_edges=[],
        geometry_score=score,
        connected=True,
        status="candidate",
        reasons=[],
    )


class GlobalGroupingTests(unittest.TestCase):
    def test_exact_partition_prefers_higher_total_utility(self):
        proposals = [
            proposal("ab", ["a", "b"], 0.9),
            proposal("cd", ["c", "d"], 0.8),
            proposal("abc", ["a", "b", "c"], 0.7),
        ]
        result = assign_groups(["a", "b", "c", "d"], proposals, CONFIG)
        selected = {
            frozenset(item["parts"])
            for item in result["selected_groups"]
            if len(item["parts"]) > 1
        }
        self.assertEqual(
            selected, {frozenset(("a", "b")), frozenset(("c", "d"))}
        )

    def test_pair_metrics_give_partial_credit_without_exact_match(self):
        selected = [{"parts": ["a", "b"]}, {"parts": ["c"]}]
        gt = {"true_groups": [{"parts": ["a", "b", "c"]}]}
        metrics = evaluate_groups(selected, gt)
        self.assertEqual(metrics["exact_group_true_positive"], 0)
        self.assertEqual(metrics["copart_pair_true_positive"], 1)
        self.assertAlmostEqual(metrics["copart_pair_precision"], 1.0)
        self.assertAlmostEqual(metrics["copart_pair_recall"], 1 / 3)

    def test_rejected_audit_conflicts_do_not_depend_on_proposal_order(self):
        proposals = [
            proposal("loser", ["a", "c"], 0.6),
            proposal("winner", ["a", "b"], 0.9),
        ]
        result = assign_groups(["a", "b", "c"], proposals, CONFIG)
        loser = next(
            item for item in result["proposal_audit"]
            if item["group_id"] == "loser"
        )
        self.assertIn("conflicting_parts=a", loser["reasons"])

    def test_bounded_external_utility_can_break_a_tie(self):
        proposals = [
            proposal("ab", ["a", "b"], 0.8),
            proposal("ac", ["a", "c"], 0.8),
        ]
        result = assign_groups(
            ["a", "b", "c"],
            proposals,
            CONFIG,
            {"ab": 0.1, "ac": 0.2},
        )
        selected = {
            item["group_id"]
            for item in result["selected_groups"]
            if len(item["parts"]) > 1
        }
        self.assertEqual(selected, {"ac"})


if __name__ == "__main__":
    unittest.main()
