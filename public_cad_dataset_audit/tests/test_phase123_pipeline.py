"""Regression tests for the frozen-data and real mixed-pool foundations."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from audit_assembly_graph_quality import audit_graph  # noqa: E402
from build_real_mixed_pools import (  # noqa: E402
    build_split_assignment,
    choose_connected_subset,
    edge_pair,
    signature_distance,
)
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


class Phase123PipelineTests(unittest.TestCase):
    def test_strict_graph_audit_rejects_example_without_step(self) -> None:
        graph = convert_assembly(EXAMPLE)
        with tempfile.TemporaryDirectory() as temporary:
            graph_path = Path(temporary) / f"{graph['assembly_id']}.json"
            graph_path.write_text(
                json.dumps(graph, ensure_ascii=False),
                encoding="utf-8",
            )
            record, _ = audit_graph(graph_path)

        self.assertEqual(record["status"], "rejected")
        self.assertTrue(record["pair_partition_complete"])
        self.assertEqual(
            record["step_geometry_missing_count"],
            record["part_count"],
        )
        self.assertEqual(
            record["relation_count"], record["mapped_relation_count"]
        )

    def test_connected_subset_is_bounded_and_connected(self) -> None:
        graph = convert_assembly(EXAMPLE)
        selected = choose_connected_subset(graph, 4, "unit-test")

        self.assertGreaterEqual(len(selected), 2)
        self.assertLessEqual(len(selected), 4)
        allowed = set(selected)
        adjacency = {part_id: set() for part_id in selected}
        for edge in graph["positive_part_pair_edges"]:
            pair = edge_pair(edge)
            if pair and pair[0] in allowed and pair[1] in allowed:
                adjacency[pair[0]].add(pair[1])
                adjacency[pair[1]].add(pair[0])
        seen = {selected[0]}
        frontier = [selected[0]]
        while frontier:
            current = frontier.pop()
            for neighbor in adjacency[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    frontier.append(neighbor)
        self.assertEqual(seen, allowed)

    def test_source_assembly_split_has_no_overlap(self) -> None:
        assignment = build_split_assignment(
            [f"assembly_{index:02d}" for index in range(10)],
            seed=20260705,
        )
        self.assertEqual(len(assignment["train"]), 6)
        self.assertEqual(len(assignment["validation"]), 2)
        self.assertEqual(len(assignment["test"]), 2)
        self.assertFalse(
            set(assignment["train"]) & set(assignment["validation"])
        )
        self.assertFalse(
            set(assignment["train"]) & set(assignment["test"])
        )
        self.assertFalse(
            set(assignment["validation"]) & set(assignment["test"])
        )

    def test_signature_distance_is_symmetric(self) -> None:
        first = {
            "file_bytes": 1000,
            "entity_counts": {"advanced_face": 4, "plane": 2},
        }
        second = {
            "file_bytes": 2000,
            "entity_counts": {"advanced_face": 8, "plane": 3},
        }
        self.assertAlmostEqual(
            signature_distance(first, second),
            signature_distance(second, first),
        )
        self.assertEqual(signature_distance(first, first), 0.0)


if __name__ == "__main__":
    unittest.main()
