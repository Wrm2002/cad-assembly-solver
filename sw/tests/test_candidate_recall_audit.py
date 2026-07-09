import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from candidate_recall_audit import audit_pool


class CandidateRecallAuditTests(unittest.TestCase):
    def test_distinguishes_missing_from_pruned_truth_interface(self):
        with tempfile.TemporaryDirectory() as temp:
            pool = Path(temp)
            (pool / "index").mkdir()
            (pool / "pool_gt.json").write_text(
                json.dumps(
                    {
                        "true_groups": [
                            {
                                "group_id": "T1",
                                "parts": ["a.step", "b.step", "c.step"],
                                "true_mates": [
                                    {
                                        "part_a": "a.step",
                                        "part_b": "b.step",
                                        "type": "clearance",
                                    },
                                    {
                                        "part_a": "b.step",
                                        "part_b": "c.step",
                                        "type": "planar_mate",
                                    },
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            candidate = {
                "candidate_id": "C1",
                "parts": ["a.step", "b.step"],
                "candidate_type": "clearance",
                "audit_reason": {"removal_reason": "low_score"},
            }
            (pool / "index" / "geometry_candidates.json").write_text(
                json.dumps([candidate]), encoding="utf-8"
            )
            (pool / "index" / "pruned_candidates.json").write_text(
                "[]", encoding="utf-8"
            )
            (pool / "index" / "removed_candidates.json").write_text(
                json.dumps([candidate]), encoding="utf-8"
            )
            features = [
                {
                    "part_id": name,
                    "cylindrical_faces": [{}],
                    "planar_faces": [{}],
                }
                for name in ("a.step", "b.step", "c.step")
            ]
            (pool / "index" / "part_features.json").write_text(
                json.dumps(features), encoding="utf-8"
            )
            rows = audit_pool(pool)
            clearance = next(
                row for row in rows if row["candidate_type"] == "clearance"
            )
            planar = next(
                row for row in rows if row["candidate_type"] == "planar_mate"
            )
            self.assertTrue(clearance["generated_or_not"])
            self.assertTrue(clearance["pruned_or_not"])
            self.assertFalse(planar["generated_or_not"])
            self.assertFalse(planar["pruned_or_not"])


if __name__ == "__main__":
    unittest.main()
