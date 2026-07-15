from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SW_DIR = Path(__file__).resolve().parents[1]
if str(SW_DIR) not in sys.path:
    sys.path.insert(0, str(SW_DIR))

from known_group_assembly import (  # noqa: E402
    _attach_brep_graph_sidecars,
    _axial_group_centering_candidates_for_candidate,
    _axial_group_centering_diagnostics,
    _pose_closure,
)
from pose_search import (  # noqa: E402
    matrix_to_placement,
    placement_to_matrix,
    recall_axial_compound_candidates,
)


def _support(
    z_low: float,
    z_high: float,
    contact_normal: float,
    *,
    scale: float = 1.0,
) -> dict:
    z_low *= scale
    z_high *= scale
    outer_normal = -contact_normal
    outer_z = z_low if contact_normal > 0 else z_high
    return {
        "cylinders": [
            {
                "origin": [0.0, 0.0, 0.0],
                "axis": [0.0, 0.0, 1.0],
                "radius": 20.0 * scale,
                "area": 1000.0 * scale * scale,
                "surface_polarity": "convex",
            },
            {
                "origin": [0.0, 0.0, 0.0],
                "axis": [0.0, 0.0, 1.0],
                "radius": 5.2 * scale,
                "area": 500.0 * scale * scale,
                "surface_polarity": "concave",
            },
        ],
        "planes": [
            {
                "position": [0.0, 0.0, 0.0],
                "normal": [0.0, 0.0, contact_normal],
                "area": 1000.0 * scale * scale,
            },
            {
                "position": [0.0, 0.0, outer_z],
                "normal": [0.0, 0.0, outer_normal],
                "area": 800.0 * scale * scale,
            },
        ],
        "bbox": {
            "min": [-20.0 * scale, -20.0 * scale, z_low],
            "max": [20.0 * scale, 20.0 * scale, z_high],
        },
    }


def _shaft(*, scale: float = 1.0) -> dict:
    return {
        "cylinders": [{
            "origin": [0.0, 0.0, 0.0],
            "axis": [0.0, 0.0, 1.0],
            "radius": 5.0 * scale,
            "area": 500.0 * scale * scale,
            "surface_polarity": "convex",
        }],
        "planes": [
            {
                "position": [0.0, 0.0, -20.0 * scale],
                "normal": [0.0, 0.0, -1.0],
                "area": 75.0 * scale * scale,
            },
            {
                "position": [0.0, 0.0, 20.0 * scale],
                "normal": [0.0, 0.0, 1.0],
                "area": 75.0 * scale * scale,
            },
        ],
        "bbox": {
            "min": [-5.0 * scale, -5.0 * scale, -20.0 * scale],
            "max": [5.0 * scale, 5.0 * scale, 20.0 * scale],
        },
    }


def _select_compound(
    fixed: dict,
    moving: dict,
    fixed_z: float,
    moving_z: float,
) -> tuple[dict, dict]:
    recall = recall_axial_compound_candidates(
        fixed,
        moving,
        minimum_face_area_ratio=0.05,
    )
    for candidate in recall["candidates"]:
        if (
            abs(candidate["fixed_end_face"]["position"][2] - fixed_z) > 1e-6
            or abs(candidate["moving_end_face"]["position"][2] - moving_z)
            > 1e-6
        ):
            continue
        for proposal in candidate["proposals"]:
            if (
                proposal.get("axis_polarity") == 1
                and proposal.get("end_face_orientation_compatible") is True
            ):
                return candidate, proposal
    raise AssertionError("requested compound proposal was not recalled")


def _fixture(*, scale: float = 1.0) -> tuple[dict, dict, dict, dict, dict]:
    features = {
        "left": _support(-50.0, 0.0, 1.0, scale=scale),
        "right": _support(0.0, 50.0, -1.0, scale=scale),
        "shaft": _shaft(scale=scale),
        "key": {
            "cylinders": [],
            "planes": [],
            "bbox": {
                "min": [5.0 * scale, -1.0 * scale, -5.0 * scale],
                "max": [7.0 * scale, 1.0 * scale, 5.0 * scale],
            },
        },
    }
    for part_features in features.values():
        part_features["brep_graph_sidecar"] = {"hash_verified": True}
    support_candidate, support_proposal = _select_compound(
        features["left"], features["right"], 0.0, 0.0
    )
    shaft_candidate, shaft_proposal = _select_compound(
        features["right"],
        features["shaft"],
        50.0 * scale,
        -20.0 * scale,
    )
    left_world = np.eye(4)
    right_world = np.asarray(support_proposal["transform"], dtype=float)
    shaft_world = right_world @ np.asarray(
        shaft_proposal["transform"], dtype=float
    )
    key_relative = np.eye(4)
    key_relative[0, 3] = 6.0 * scale
    placements = {
        "left": matrix_to_placement(left_world),
        "right": matrix_to_placement(right_world),
        "shaft": matrix_to_placement(shaft_world),
        "key": matrix_to_placement(shaft_world @ key_relative),
    }
    support_row = {
        "connection_id": "support_edge",
        "fixed_part": "left",
        "moving_part": "right",
        "candidate_id": support_candidate["candidate_id"],
        "proposal_id": support_proposal["proposal_id"],
        "compound_candidate": support_candidate,
        "compound_proposal": support_proposal,
        "phase_witness": [{
            "fixed_witness_id": "slot:left",
            "moving_witness_id": "slot:right",
            "fixed_kind": "topological_key_slot",
            "moving_kind": "topological_key_slot",
        }],
    }
    shaft_row = {
        "connection_id": "shaft_edge",
        "fixed_part": "right",
        "moving_part": "shaft",
        "candidate_id": shaft_candidate["candidate_id"],
        "proposal_id": shaft_proposal["proposal_id"],
        "compound_candidate": shaft_candidate,
        "compound_proposal": shaft_proposal,
        "phase_witness": [{
            "fixed_witness_id": "slot:right",
            "moving_witness_id": "slot:shaft",
            "fixed_kind": "topological_key_slot",
            "moving_kind": "topological_key_slot",
        }],
    }
    graph = {
        "selected": [
            {
                "connection_id": "support_edge",
                "parts": ["left", "right"],
                "relation_types": ["coaxial"],
                "matches": [],
            },
            {
                "connection_id": "shaft_edge",
                "parts": ["right", "shaft"],
                "relation_types": ["clearance"],
                "matches": [],
            },
            {
                "connection_id": "key_edge",
                "parts": ["shaft", "key"],
                "relation_types": ["planar_mate"],
                "matches": [],
            },
        ]
    }
    candidate = {
        "placements": placements,
        "axial_compound_history": [support_row, shaft_row],
        "selected_mates": [],
        "score": 0.0,
        "penalty": 0.0,
        "total_score": 0.0,
        "penalty_details": {},
    }
    return features, graph, candidate, support_row, shaft_row


class AxialGroupCenteringTests(unittest.TestCase):
    def test_one_end_stop_is_replaced_by_centered_two_sided_insertion(self):
        features, graph, candidate, support_row, shaft_row = _fixture()
        raw = _axial_group_centering_diagnostics(
            support_row,
            shaft_row,
            candidate["placements"],
            features,
            graph,
        )
        self.assertTrue(raw["pattern_detected"])
        self.assertFalse(raw["supported"])
        self.assertAlmostEqual(raw["centering_offset_mm"], -70.0, places=6)
        self.assertEqual(raw["two_sided_overlap_mm"], [0.0, 0.0])

        old_relative = (
            np.linalg.inv(placement_to_matrix(candidate["placements"]["shaft"]))
            @ placement_to_matrix(candidate["placements"]["key"])
        )
        generated, required = _axial_group_centering_candidates_for_candidate(
            candidate, graph, features
        )
        self.assertEqual(required, ["shaft_edge"])
        self.assertEqual(len(generated), 1)
        centered = generated[0]
        detail = centered["axial_group_centering"]
        self.assertTrue(detail["supported"])
        self.assertAlmostEqual(detail["centering_residual_mm"], 0.0, places=6)
        self.assertEqual(detail["two_sided_overlap_mm"], [20.0, 20.0])
        self.assertTrue(centered["proposal_only"])
        self.assertTrue(centered["review_required"])
        self.assertFalse(centered["can_auto_accept"])
        new_relative = (
            np.linalg.inv(placement_to_matrix(centered["placements"]["shaft"]))
            @ placement_to_matrix(centered["placements"]["key"])
        )
        self.assertTrue(np.allclose(new_relative, old_relative, atol=1e-9))

    def test_closure_recomputes_exact_bound_evidence_and_rejects_stale_id(self):
        features, graph, candidate, _, _ = _fixture()
        generated, required = _axial_group_centering_candidates_for_candidate(
            candidate, graph, features
        )
        raw = copy.deepcopy(candidate)
        raw["axial_group_centering_required_connection_ids"] = required
        raw_row = next(
            row for row in _pose_closure(raw, graph, features)["connections"]
            if row["connection_id"] == "shaft_edge"
        )
        self.assertFalse(raw_row["closed"])
        self.assertEqual(
            raw_row["closure_evidence"],
            "axial_group_centering_residual_failed",
        )

        centered = generated[0]
        centered_row = next(
            row for row in _pose_closure(centered, graph, features)["connections"]
            if row["connection_id"] == "shaft_edge"
        )
        self.assertTrue(centered_row["closed"])
        self.assertEqual(
            centered_row["closure_evidence"],
            "axial_group_centered_two_sided_insertion",
        )

        stale = copy.deepcopy(centered)
        stale["axial_group_centering_history"][0]["shaft_proposal_id"] = (
            "another_proposal"
        )
        stale["axial_group_centering"] = stale[
            "axial_group_centering_history"
        ][0]
        stale_row = next(
            row for row in _pose_closure(stale, graph, features)["connections"]
            if row["connection_id"] == "shaft_edge"
        )
        self.assertFalse(stale_row["closed"])

    def test_clearance_compound_end_stop_without_insertion_cannot_close(self):
        features, graph, candidate, _, _ = _fixture()
        row = next(
            connection
            for connection in _pose_closure(candidate, graph, features)[
                "connections"
            ]
            if connection["connection_id"] == "shaft_edge"
        )
        self.assertFalse(row["closed"])
        self.assertEqual(
            row["closure_evidence"],
            "axial_compound_clearance_overlap_failed",
        )

    def test_abstains_without_clearance_or_second_support_bore(self):
        features, graph, candidate, _, _ = _fixture()
        no_clearance = copy.deepcopy(graph)
        no_clearance["selected"][1]["relation_types"] = ["coaxial"]
        generated, required = _axial_group_centering_candidates_for_candidate(
            candidate, no_clearance, features
        )
        self.assertEqual(generated, [])
        self.assertEqual(required, [])

        no_bore = copy.deepcopy(features)
        no_bore["left"]["cylinders"] = no_bore["left"]["cylinders"][:1]
        generated, required = _axial_group_centering_candidates_for_candidate(
            candidate, graph, no_bore
        )
        self.assertEqual(generated, [])
        self.assertEqual(required, [])

        external_boss = copy.deepcopy(features)
        external_boss["left"]["cylinders"][1]["surface_polarity"] = "convex"
        generated, required = _axial_group_centering_candidates_for_candidate(
            candidate, graph, external_boss
        )
        self.assertEqual(generated, [])
        self.assertEqual(required, [])

    def test_abstains_without_paired_topological_key_slot_witness(self):
        for history_index, edge_name in ((0, "support"), (1, "shaft")):
            with self.subTest(edge=edge_name):
                features, graph, candidate, _, _ = _fixture()
                candidate["axial_compound_history"][history_index][
                    "phase_witness"
                ] = [{
                    "fixed_witness_id": "slot:fixed",
                    "moving_witness_id": "slot:moving",
                    "fixed_kind": "topological_key_slot",
                    "moving_kind": "generic_planar_witness",
                }]

                generated, required = (
                    _axial_group_centering_candidates_for_candidate(
                        candidate, graph, features
                    )
                )

                self.assertEqual(generated, [])
                self.assertEqual(required, [])

    def test_abstains_for_unverified_or_idless_topology_evidence(self):
        features, graph, candidate, _, _ = _fixture()
        idless = copy.deepcopy(candidate)
        idless["axial_compound_history"][0]["phase_witness"][0].pop(
            "fixed_witness_id"
        )
        generated, required = _axial_group_centering_candidates_for_candidate(
            idless, graph, features
        )
        self.assertEqual(generated, [])
        self.assertEqual(required, [])

        unverified = copy.deepcopy(features)
        unverified["right"]["brep_graph_sidecar"]["hash_verified"] = False
        generated, required = _axial_group_centering_candidates_for_candidate(
            candidate, graph, unverified
        )
        self.assertEqual(generated, [])
        self.assertEqual(required, [])

    def test_sidecar_requires_matching_declared_geometry_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            step = root / "P17.step"
            step.write_bytes(b"anonymous-step-geometry")
            graph_path = root / "P17.brep_graph.json"
            payload = {
                "schema_version": "1.0.0",
                "nodes": [{"node_id": "face_000001"}],
                "metadata": {"edge_topology_features": {"available": True}},
            }
            features = {step.name: {}}

            graph_path.write_text(json.dumps(payload), encoding="utf-8")
            audit = _attach_brep_graph_sidecars(features, [step], root)
            self.assertEqual(audit["status"], "partial")
            self.assertEqual(
                audit["rejected_parts"][0]["reason"],
                "source_geometry_sha256_missing",
            )

            payload["source_geometry_sha256"] = "0" * 64
            graph_path.write_text(json.dumps(payload), encoding="utf-8")
            audit = _attach_brep_graph_sidecars(features, [step], root)
            self.assertEqual(
                audit["rejected_parts"][0]["reason"],
                "source_geometry_sha256_mismatch",
            )

            payload["source_geometry_sha256"] = hashlib.sha256(
                step.read_bytes()
            ).hexdigest()
            graph_path.write_text(json.dumps(payload), encoding="utf-8")
            audit = _attach_brep_graph_sidecars(features, [step], root)
            self.assertEqual(audit["status"], "loaded")
            self.assertTrue(
                features[step.name]["brep_graph_sidecar"]["hash_verified"]
            )

    def test_abstains_when_key_like_dependent_does_not_cross_contact_plane(self):
        features, graph, candidate, support_row, shaft_row = _fixture()
        features["key"]["bbox"] = {
            "min": [5.0, -1.0, 2.0],
            "max": [7.0, 1.0, 10.0],
        }

        generated, required = _axial_group_centering_candidates_for_candidate(
            candidate, graph, features
        )

        self.assertEqual(generated, [])
        self.assertEqual(required, ["shaft_edge"])

        centered_placements = copy.deepcopy(candidate["placements"])
        for part in ("shaft", "key"):
            world = placement_to_matrix(centered_placements[part])
            world[2, 3] -= 70.0
            centered_placements[part] = matrix_to_placement(world)
        diagnostics = _axial_group_centering_diagnostics(
            support_row,
            shaft_row,
            centered_placements,
            features,
            graph,
        )
        self.assertTrue(diagnostics["pattern_detected"])
        self.assertFalse(diagnostics["cross_interface_dependent_supported"])
        self.assertFalse(diagnostics["supported"])
        self.assertEqual(
            diagnostics["reason"],
            "symmetric_support_requires_centered_two_sided_shaft_insertion",
        )

    def test_scale_consistency_at_ten_x(self):
        features, graph, candidate, _, _ = _fixture(scale=10.0)

        generated, required = _axial_group_centering_candidates_for_candidate(
            candidate, graph, features
        )

        self.assertEqual(required, ["shaft_edge"])
        self.assertEqual(len(generated), 1)
        self.assertTrue(generated[0]["axial_group_centering"]["supported"])

    def test_scale_consistency_at_one_tenth_x(self):
        """Scale-normalised recall/group gates also support smaller geometry."""

        recall = recall_axial_compound_candidates(
            _support(0.0, 50.0, -1.0, scale=0.1),
            _shaft(scale=0.1),
            minimum_face_area_ratio=0.05,
        )
        self.assertEqual(
            recall.get("status"),
            "recalled",
            recall.get("reason"),
        )

        features, graph, candidate, _, _ = _fixture(scale=0.1)

        generated, required = _axial_group_centering_candidates_for_candidate(
            candidate, graph, features
        )

        self.assertEqual(required, ["shaft_edge"])
        self.assertEqual(len(generated), 1)
        self.assertTrue(generated[0]["axial_group_centering"]["supported"])

if __name__ == "__main__":
    unittest.main()
