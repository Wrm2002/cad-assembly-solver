import sys
import unittest
from pathlib import Path

from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contracts import CandidateEdge, CandidateStatus, CONTRACT_MODELS


class ContractTests(unittest.TestCase):
    def test_every_contract_generates_json_schema(self):
        for model in CONTRACT_MODELS.values():
            schema = model.model_json_schema()
            self.assertEqual(schema["type"], "object")
            self.assertFalse(schema.get("additionalProperties", True))

    def test_candidate_requires_distinct_parts(self):
        with self.assertRaises(ValidationError):
            CandidateEdge(
                candidate_id="C1",
                parts=["same", "same"],
                candidate_type="coaxial",
                geometry_score=0.5,
                confidence="medium",
                geometric_evidence=[],
                status=CandidateStatus.generated,
                audit_reason={},
            )


if __name__ == "__main__":
    unittest.main()
