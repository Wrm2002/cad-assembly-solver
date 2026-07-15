"""Regression checks for the measured JoinABLe migration artifacts."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from joinable_common import build_common_sample  # noqa: E402


JOINT_ROOT = Path(
    r"D:\Model_match_public_data\fusion360_joint"
    r"\sample20\j1.0.0\joint"
)


class JoinableMigrationAuditTests(unittest.TestCase):
    @unittest.skipUnless(
        JOINT_ROOT.is_dir(), "official Fusion joint sample unavailable"
    )
    def test_real_joint_round_trip_fields(self) -> None:
        sample = build_common_sample(
            JOINT_ROOT / "joint_set_00000.json", 0
        )
        self.assertTrue(sample["relation"]["has_joint"])
        self.assertEqual(
            sample["relation"]["label_semantics"],
            "designer_selected_joint",
        )
        self.assertTrue(sample["interface_a"]["entity_ids"])
        self.assertTrue(sample["interface_b"]["entity_ids"])
        self.assertIsNotNone(sample["relation"]["axis_origin"])
        self.assertIsNotNone(sample["relation"]["axis_direction"])
        self.assertIsNotNone(sample["relation"]["transform_a_to_b"])
        self.assertIn(
            "source_explicit_assembly_id",
            sample["unavailable_fields"],
        )
        self.assertEqual(
            sample["metadata"]["assembly_id_semantics"],
            "inferred_from_common_body_name_prefix",
        )

    def test_twenty_conversions_and_three_step_graphs(self) -> None:
        summary = json.loads(
            (ROOT / "conversion_summary.json").read_text(
                encoding="utf-8"
            )
        )
        step = json.loads(
            (ROOT / "step_brep_graph_probe_report.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["success_count"], 20)
        self.assertEqual(summary["failure_count"], 0)
        self.assertTrue(summary["acceptance_met"])
        self.assertEqual(step["success_count"], 3)
        self.assertEqual(step["failure_count"], 0)
        self.assertTrue(step["acceptance_met"])

    def test_step_graph_entity_ids_and_adjacency_are_traceable(
        self,
    ) -> None:
        graph_paths = sorted(
            (ROOT / "step_brep_graph_samples").glob("*.json")
        )
        self.assertEqual(len(graph_paths), 3)
        for path in graph_paths:
            graph = json.loads(path.read_text(encoding="utf-8"))
            node_ids = {node["node_id"] for node in graph["nodes"]}
            self.assertTrue(node_ids)
            self.assertTrue(graph["edges"])
            self.assertEqual(
                graph["metadata"]["extraction_status"], "success"
            )
            for edge in graph["edges"]:
                self.assertIn(edge["src"], node_ids)
                self.assertIn(edge["dst"], node_ids)
                self.assertEqual(
                    edge["relation"], "face_edge_adjacency"
                )

    def test_machine_outputs_expose_failures_and_unavailable(self) -> None:
        paths = [
            ROOT / "fusion_joint_schema_sample.json",
            ROOT / "conversion_failures.json",
            ROOT / "conversion_summary.json",
            ROOT / "step_brep_graph_probe_report.json",
            ROOT / "schema_gap_report.json",
            ROOT / "joint_interface_schema.example.json",
        ]
        paths.extend(
            sorted((ROOT / "converted_joint_samples").glob("*.json"))
        )
        paths.extend(
            sorted((ROOT / "step_brep_graph_samples").glob("*.json"))
        )
        for path in paths:
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertIn("failure_reasons", data, str(path))
            self.assertIn("unavailable_fields", data, str(path))


if __name__ == "__main__":
    unittest.main()
