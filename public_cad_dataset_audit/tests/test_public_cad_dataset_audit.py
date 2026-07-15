"""Regression tests for the public-dataset-only conversion path."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from fusion360_common import convert_assembly  # noqa: E402


EXAMPLE = (
    PROJECT
    / "vendor"
    / "Fusion360GalleryDataset"
    / "tools"
    / "testdata"
    / "assembly_examples"
    / "belt_clamp"
    / "assembly.json"
)


class PublicCadDatasetAuditTests(unittest.TestCase):
    def test_official_fusion_example_maps_relations_and_pairs(
        self,
    ) -> None:
        graph = convert_assembly(EXAMPLE)

        self.assertEqual(
            graph["source_dataset"],
            "fusion360_gallery_assembly",
        )
        self.assertEqual(graph["quality"]["status"], "usable")
        self.assertEqual(graph["quality"]["part_count"], 5)
        self.assertEqual(graph["quality"]["positive_pair_count"], 4)
        self.assertEqual(graph["quality"]["negative_pair_count"], 6)
        self.assertFalse(graph["failure_reasons"])

        part_ids = {part["part_id"] for part in graph["parts"]}
        for edge in (
            graph["positive_part_pair_edges"]
            + graph["negative_part_pair_edges"]
        ):
            self.assertEqual(len(edge["part_pair"]), 2)
            self.assertLessEqual(set(edge["part_pair"]), part_ids)
            self.assertIn("failure_reasons", edge)
            self.assertIn("unavailable_fields", edge)

    def test_graph_output_is_json_serializable_and_auditable(
        self,
    ) -> None:
        graph = convert_assembly(EXAMPLE)
        round_trip = json.loads(json.dumps(graph))

        self.assertIn("failure_reasons", round_trip)
        self.assertIn("unavailable_fields", round_trip)
        self.assertTrue(all(
            part["geometry"]["available"]
            for part in round_trip["parts"]
        ))


if __name__ == "__main__":
    unittest.main()
