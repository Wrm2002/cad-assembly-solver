import sys
import tempfile
import unittest
import json
import math
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contracts import DirectAssemblyConnection, KnownGroupAssemblyResult
from placement_validation import transform_point
from known_group_assembly import (
    _apply_joinable_support,
    _axial_compound_pose_candidates,
    _candidate_from_placements,
    _conservative_pose_output,
    _enclosure_bay_candidates_for_connection,
    _edge_slot_candidates_for_connection,
    _exact_collision_risk,
    _joinable_pose_parameter_candidates,
    _localized_interference_review,
    _obb_insertion_pose_candidates,
    _planar_footprint_pose_candidates,
    _pose_closure,
    _portable_components,
    _relative_transform,
    _select_exact_rank_budget,
)


class KnownGroupContractTests(unittest.TestCase):
    def test_edge_slot_candidate_closes_only_at_recalled_pose_review_only(self):
        features = {
            "carrier.step": {
                "bbox": {"min": [-100, -50, -2], "max": [100, 50, 2]},
                "obb": {
                    "center": [0, 0, 0],
                    "axes": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "dimensions": [200, 100, 4],
                },
                "planes": [],
                "cylinders": [],
            },
            "module.step": {
                "bbox": {"min": [-50, -15, -1], "max": [50, 15, 1]},
                "obb": {
                    "center": [0, 0, 0],
                    "axes": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "dimensions": [100, 30, 2],
                },
                "planes": [],
                "cylinders": [],
            },
        }
        source = {
            "placements": {
                part: {"translate": [0.0, 0.0, 0.0]} for part in features
            },
            "components": [],
            "selected_mates": [],
            "total_score": 0.0,
        }
        connection = {
            "connection_id": "edge_slot",
            "parts": ["carrier.step", "module.step"],
            "relation_types": ["planar_mate"],
            "matches": [],
        }
        transform = [
            [1.0, 0.0, 0.0, 10.0],
            [0.0, 1.0, 0.0, 20.0],
            [0.0, 0.0, 1.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        recall = {
            "status": "success",
            "reason": "synthetic integration fixture",
            "audit": {"bounded_channel_floor_count": 4},
            "proposals": [{
                "slot_family_id": 0,
                "slot_family_size": 4,
                "slot_rank": 1,
                "floor_plane_index": 10,
                "wall_plane_indices": [11, 12],
                "channel_gap": 1.7,
                "slot_pitch": 7.5,
                "long_axis_sign": 1,
                "transform_matrix": transform,
                "length_relative_error": 0.01,
                "floor_evidence": {"mirror_score": 0.02},
                "evidence_families": ["length", "walls", "family", "axis"],
                "independent_evidence_count": 4,
                "has_multi_evidence_support": True,
                "proposal_only": True,
                "review_required": True,
                "can_auto_accept": False,
            }],
        }
        with patch(
            "known_group_assembly._cached_edge_slot_recall",
            return_value=recall,
        ):
            candidates = _edge_slot_candidates_for_connection(
                [source],
                connection,
                features,
                max_candidates=2,
                refinement_phase="independent_edge_quota",
            )

        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertEqual(
            candidate["placements"]["module.step"]["translate"],
            [10.0, 20.0, 30.0],
        )
        self.assertTrue(candidate["proposal_only"])
        self.assertTrue(candidate["review_required"])
        self.assertFalse(candidate["can_auto_accept"])
        closure = _pose_closure(
            candidate, {"selected": [connection]}, features
        )
        self.assertTrue(closure["fully_closed"])
        self.assertTrue(closure["review_required"])
        self.assertEqual(
            closure["connections"][0]["closure_evidence"],
            "repeated_bounded_edge_slot_multi_evidence",
        )
        shifted = json.loads(json.dumps(candidate))
        shifted["placements"]["module.step"]["translate"][0] += 1.0
        shifted_closure = _pose_closure(
            shifted, {"selected": [connection]}, features
        )
        self.assertFalse(shifted_closure["fully_closed"])

    def test_enclosure_bay_candidate_closes_only_at_recalled_pose_review_only(self):
        features = {
            "carrier.step": {
                "bbox": {"min": [-100, -50, -100], "max": [100, 50, 100]},
                "obb": {
                    "center": [0, 0, 0],
                    "axes": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "dimensions": [200, 100, 200],
                },
                "planes": [],
                "cylinders": [],
            },
            "module.step": {
                "bbox": {"min": [-20, -10, -30], "max": [20, 10, 30]},
                "obb": {
                    "center": [0, 0, 0],
                    "axes": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "dimensions": [40, 20, 60],
                },
                "planes": [],
                "cylinders": [],
            },
        }
        source = {
            "placements": {
                part: {"translate": [0.0, 0.0, 0.0]} for part in features
            },
            "components": [],
            "selected_mates": [],
            "total_score": 0.0,
        }
        connection = {
            "connection_id": "bay_edge",
            "parts": ["carrier.step", "module.step"],
            "relation_types": ["planar_mate", "pocket_mate"],
            "matches": [],
        }
        transform = [
            [1.0, 0.0, 0.0, 10.0],
            [0.0, 1.0, 0.0, 20.0],
            [0.0, 0.0, 1.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        recall = {
            "status": "proposed",
            "reason": "synthetic integration fixture",
            "functional_body_envelope_audit": {"status": "proposed"},
            "proposals": [{
                "candidate_id": "enclosure_slot_0_polarity_+1",
                "slot_index": 0,
                "depth_polarity": 1,
                "opening_polarity": "preferred",
                "transform_4x4": transform,
                "evidence": [
                    "opposing_wall_pairs",
                    "repeated_equivalent_slots",
                    "paired_repeated_supports",
                    "functional_body_envelope_fit",
                ],
                "independent_evidence_count": 4,
                "proposal_score": 0.9,
                "functional_body_derivation_status": "proposed",
                "functional_body_excluded_protrusion_risk": False,
                "proposal_only": True,
                "review_required": True,
                "can_auto_accept": False,
            }],
        }
        with patch(
            "known_group_assembly._cached_enclosure_bay_recall",
            return_value=recall,
        ):
            candidates = _enclosure_bay_candidates_for_connection(
                [source],
                connection,
                features,
                max_candidates=2,
                refinement_phase="independent_edge_quota",
            )

        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertEqual(
            candidate["placements"]["module.step"]["translate"],
            [10.0, 20.0, 30.0],
        )
        self.assertTrue(candidate["proposal_only"])
        self.assertTrue(candidate["review_required"])
        self.assertFalse(candidate["can_auto_accept"])
        closure = _pose_closure(
            candidate, {"selected": [connection]}, features
        )
        self.assertTrue(closure["fully_closed"])
        self.assertTrue(closure["review_required"])
        self.assertEqual(
            closure["connections"][0]["closure_evidence"],
            "repeated_enclosure_bay_multi_evidence",
        )
        shifted = json.loads(json.dumps(candidate))
        shifted["placements"]["module.step"]["translate"][0] += 1.0
        shifted_closure = _pose_closure(
            shifted, {"selected": [connection]}, features
        )
        self.assertFalse(shifted_closure["fully_closed"])

    @staticmethod
    def _two_leaf_axial_fixture():
        features = {
            "carrier.step": {
                "bbox": {"min": [-5, -5, -20], "max": [5, 5, 20]},
                "cylinders": [{
                    "axis": [0, 0, 1],
                    "origin": [0, 0, 0],
                    "radius": 2.0,
                }],
            },
            "leaf_a.step": {
                "bbox": {"min": [-2, -2, -2], "max": [2, 2, 2]},
                "cylinders": [{
                    "axis": [1, 0, 0],
                    "origin": [1, 0, 0],
                    "radius": 3.0,
                }],
            },
            "leaf_b.step": {
                "bbox": {"min": [-3, -3, -3], "max": [3, 3, 3]},
                "cylinders": [{
                    "axis": [0, 1, 0],
                    "origin": [0, 1, 0],
                    "radius": 4.0,
                }],
            },
        }

        def connection(connection_id, leaf):
            return {
                "connection_id": connection_id,
                "parts": ["carrier.step", leaf],
                "relation_types": ["coaxial"],
                "matches": [{
                    "type": "coaxial",
                    "parts": ["carrier.step", leaf],
                    "feat_a_idx": 0,
                    "feat_b_idx": 0,
                }],
            }

        graph = {"selected": [
            connection("edge_a", "leaf_a.step"),
            connection("edge_b", "leaf_b.step"),
        ]}
        source = {
            "placements": {
                part: {"translate": [0.0, 0.0, 0.0]} for part in features
            },
            "components": [],
            "selected_mates": [],
            "score": 0.0,
            "penalty": 0.0,
            "total_score": 0.0,
            "penalty_details": {},
        }
        return features, graph, source

    @staticmethod
    def _two_leaf_obb_fixture():
        def obb(dimensions):
            return {
                "center": [0.0, 0.0, 0.0],
                "axes": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                "dimensions": list(dimensions),
            }

        carrier_plane = {
            "normal": [0.0, 1.0, 0.0],
            "centroid": [0.0, 0.0, 0.0],
            "position": [0.0, 0.0, 0.0],
            "area": 100.0,
        }
        leaf_plane = {
            "normal": [1.0, 0.0, 0.0],
            "centroid": [4.0, 2.0, -3.0],
            "position": [4.0, 2.0, -3.0],
            "area": 100.0,
        }
        features = {
            "carrier.step": {
                "bbox": {
                    "min": [-193.935, -14.68, -213.025],
                    "max": [193.935, 14.68, 213.025],
                },
                "obb": obb((387.87, 29.36, 426.05)),
                "planes": [carrier_plane],
            },
            "dimm.step": {
                "bbox": {
                    "min": [-1.385, -66.675, -15.625],
                    "max": [1.385, 66.675, 15.625],
                },
                "obb": obb((2.77, 133.35, 31.25)),
                "planes": [leaf_plane],
            },
            "cpu.step": {
                "bbox": {
                    "min": [-36.01, -3.085, -37.71],
                    "max": [36.01, 3.085, 37.71],
                },
                "obb": obb((72.02, 6.17, 75.42)),
                "planes": [leaf_plane],
            },
        }

        def connection(connection_id, leaf):
            return {
                "connection_id": connection_id,
                "parts": ["carrier.step", leaf],
                "relation_types": ["planar_mate"],
                "matches": [{
                    "type": "planar_mate",
                    "parts": ["carrier.step", leaf],
                    "feat_a_idx": 0,
                    "feat_b_idx": 0,
                }],
            }

        graph = {"selected": [
            connection("dimm_edge", "dimm.step"),
            connection("cpu_edge", "cpu.step"),
        ]}
        source = {
            "placements": {
                part: {"translate": [0.0, 0.0, 0.0]} for part in features
            },
            "components": [],
            "selected_mates": [],
            "score": 0.0,
            "penalty": 0.0,
            "total_score": 0.0,
            "penalty_details": {},
        }
        return features, graph, source

    @staticmethod
    def _two_leaf_footprint_fixture():
        features, graph, source = KnownGroupContractTests._two_leaf_obb_fixture()

        def plane(center, dimensions):
            return {
                "normal": [0.0, 1.0, 0.0],
                "centroid": list(center),
                "position": list(center),
                "footprint_axes": [
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                "footprint_dimensions": list(dimensions),
                "area": float(dimensions[0] * dimensions[1]),
            }

        features["carrier.step"]["planes"] = [
            plane([0.0, 0.0, 0.0], [31.25, 133.35]),
            plane([0.0, 1.0, 0.0], [31.25, 133.35]),
            plane([100.0, 0.0, 200.0], [72.02, 75.42]),
            plane([100.0, 1.5, 200.0], [72.02, 75.42]),
        ]
        return features, graph, source

    @staticmethod
    def _axial_compound_fixture(*, with_slot_witness=False):
        def part(end_normal, end_z):
            cylinders = [{
                "axis": [0.0, 0.0, 1.0],
                "origin": [0.0, 0.0, 0.0],
                "radius": 20.0,
                "area": 500.0,
            }]
            for angle in (0, 60, 120, 180, 240, 300):
                radians = math.radians(angle)
                cylinders.append({
                    "axis": [0.0, 0.0, 1.0],
                    "origin": [
                        30.0 * math.cos(radians),
                        30.0 * math.sin(radians),
                        0.0,
                    ],
                    "radius": 2.0,
                    "area": 25.0,
                })
            result = {
                "bbox": {"min": [-40, -40, -5], "max": [40, 40, 5]},
                "cylinders": cylinders,
                "planes": [{
                    "normal": list(end_normal),
                    "position": [0.0, 0.0, float(end_z)],
                    "centroid": [0.0, 0.0, float(end_z)],
                    "area": 1000.0,
                }],
            }
            if with_slot_witness:
                result["axial_orientation_witnesses"] = [{
                    "witness_id": "audited_slot",
                    "kind": "topological_key_slot",
                    "angle_degrees": 0.0,
                    "asymmetric": True,
                    "topology_supported": True,
                }]
            return result

        features = {
            "fixed.step": part([0.0, 0.0, 1.0], 5.0),
            "moving.step": part([0.0, 0.0, -1.0], -5.0),
        }
        graph = {"selected": [{
            "connection_id": "axial_edge",
            "parts": ["fixed.step", "moving.step"],
            # The legacy matcher may miss the planar relation entirely.  The
            # compound provider must recover it from own-axis end faces.
            "relation_types": ["coaxial"],
            "matches": [{
                "type": "coaxial",
                "parts": ["fixed.step", "moving.step"],
                "feat_a_idx": 0,
                "feat_b_idx": 0,
            }],
        }]}
        source = {
            "placements": {
                part_id: {"translate": [0.0, 0.0, 0.0]}
                for part_id in features
            },
            "components": [],
            "selected_mates": [],
            "score": 0.0,
            "penalty": 0.0,
            "total_score": 0.0,
            "penalty_details": {},
        }
        return features, graph, source

    def test_portable_components_support_independent_output_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            case_dir = root / "inputs" / "case"
            output_dir = root / "independent" / "exam" / "case3"
            case_dir.mkdir(parents=True)
            output_dir.mkdir(parents=True)
            source = case_dir / "part.step"
            source.touch()

            components = _portable_components(
                [{"source": str(source), "label": "part"}],
                case_dir,
                output_dir,
            )

            resolved = (output_dir / components[0]["source"]).resolve()
            self.assertEqual(resolved, source.resolve())

    def test_relative_transform_uses_part_b_frame(self):
        placements = {
            "a": {"translate": [5.0, 0.0, 0.0]},
            "b": {"translate": [2.0, 0.0, 0.0]},
        }
        matrix = _relative_transform("a", "b", placements)
        self.assertAlmostEqual(matrix[0][3], 3.0)
        self.assertAlmostEqual(matrix[1][3], 0.0)

    def test_minimal_single_part_document(self):
        document = KnownGroupAssemblyResult.model_validate({
            "assembly_id": "single",
            "parts": ["only.step"],
            "reference_part": "only.step",
            "assembly_connected": True,
            "pose_status": "valid",
            "direct_connections": [],
            "assembly_relations": [],
            "components": [{"source": "../only.step"}],
            "collision_validation": {"status": "success"},
            "candidate_summary": {},
        })
        self.assertEqual(document.schema_version, "2.0.0")

    def test_direct_connection_contract_accepts_pose_closure_evidence(self):
        connection = DirectAssemblyConnection.model_validate({
            "connection_id": "edge-1",
            "parts": ["shaft.step", "hub.step"],
            "primary_relation_type": "coaxial",
            "supporting_relation_types": ["coaxial", "planar_mate"],
            "constraint_ids": ["constraint-1"],
            "score": 0.9,
            "confidence": "high",
            "selection_role": "connected_skeleton",
            "constraint_closed_in_selected_pose": True,
            "review_required": True,
            "closure_evidence": "axial_compound_interface",
            "axial_compound_evidence": [{
                "candidate_id": "compound-1",
                "compound_constraints_satisfied": True,
            }],
            "enclosure_bay_evidence": [{
                "candidate_id": "bay-1",
                "supported": False,
            }],
            "edge_slot_evidence": [{
                "slot_family_id": 0,
                "slot_rank": 2,
                "supported": True,
            }],
            "relative_transform_a_to_b": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        })

        self.assertEqual(connection.closure_evidence, "axial_compound_interface")
        self.assertEqual(
            connection.axial_compound_evidence[0]["candidate_id"],
            "compound-1",
        )
        self.assertEqual(
            connection.enclosure_bay_evidence[0]["candidate_id"],
            "bay-1",
        )
        self.assertTrue(connection.edge_slot_evidence[0]["supported"])

    def test_direct_connection_contract_still_forbids_unknown_evidence(self):
        payload = {
            "connection_id": "edge-1",
            "parts": ["shaft.step", "hub.step"],
            "primary_relation_type": "coaxial",
            "supporting_relation_types": ["coaxial"],
            "constraint_ids": ["constraint-1"],
            "score": 0.9,
            "confidence": "high",
            "selection_role": "connected_skeleton",
            "constraint_closed_in_selected_pose": True,
            "review_required": False,
            "relative_transform_a_to_b": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "unexpected_pose_evidence": [],
        }

        with self.assertRaises(ValidationError) as raised:
            DirectAssemblyConnection.model_validate(payload)

        self.assertTrue(any(
            error["type"] == "extra_forbidden"
            and tuple(error["loc"]) == ("unexpected_pose_evidence",)
            for error in raised.exception.errors()
        ))

    def test_joinable_axial_candidate_boundedly_supports_clearance(self):
        match = {
            "type": "clearance",
            "parts": ("shaft.step", "hub.step"),
            "score": 0.7,
            "confidence": "medium",
        }
        learned = {
            ("hub.step", "shaft.step"): {
                "pair_id": "p1",
                "top_interface_candidates": [{
                    "rank": 1,
                    "family_hint": "coaxial_or_cylindrical",
                    "softmax_probability": 0.5,
                }],
            }
        }
        result = _apply_joinable_support([match], learned)[0]
        self.assertGreater(result["score"], 0.7)
        self.assertLessEqual(result["score"], 0.78)
        self.assertEqual(result["joinable_support"]["rank"], 1)

    def test_pose_valid_without_precision_gate_is_review_not_accepted(self):
        result = _conservative_pose_output(
            {
                "assembly_id": "anonymous",
                "parts": ["a", "b"],
                "assembly_connected": True,
                "pose_status": "valid",
                "collision_validation": {
                    "status": "success",
                    "selected_pose_rank": 1,
                },
                "direct_connections": [],
                "unresolved_parts": [],
            },
            [{"rank": 1, "constraint_closure": {"fully_closed": True}}],
        )
        self.assertEqual(result["accepted_groups"], [])
        self.assertEqual(len(result["review_groups"]), 1)
        self.assertEqual(
            result["review_groups"][0]["decision_reasons"],
            ["precision_pose_validation_missing"],
        )

    def test_pose_and_precision_valid_can_be_accepted(self):
        result = _conservative_pose_output(
            {
                "assembly_id": "anonymous",
                "parts": ["a", "b"],
                "assembly_connected": True,
                "pose_status": "valid",
                "precision_pose_validation": {
                    "precision_status": "valid",
                    "independent_evidence_count": 2,
                },
                "collision_validation": {
                    "status": "success",
                    "selected_pose_rank": 1,
                },
                "direct_connections": [],
                "unresolved_parts": [],
            },
            [{"rank": 1, "constraint_closure": {"fully_closed": True}}],
        )
        self.assertEqual(len(result["accepted_groups"]), 1)
        self.assertEqual(result["review_groups"], [])

    def test_obb_proposal_cannot_auto_accept_even_with_precision_valid(self):
        result = _conservative_pose_output(
            {
                "assembly_id": "anonymous",
                "parts": ["a", "b"],
                "assembly_connected": True,
                "pose_status": "valid",
                "precision_pose_validation": {
                    "precision_status": "valid",
                    "independent_evidence_count": 2,
                },
                "collision_validation": {
                    "status": "success",
                    "selected_pose_rank": 1,
                },
                "direct_connections": [],
                "unresolved_parts": [],
            },
            [{
                "rank": 1,
                "candidate_origin": "obb_insertion_axis_role_search",
                "proposal_only": True,
                "review_required": True,
                "can_auto_accept": False,
                "constraint_closure": {"fully_closed": True},
                "obb_insertion": {"review_required": True},
            }],
        )
        self.assertEqual(result["accepted_groups"], [])
        self.assertEqual(len(result["review_groups"]), 1)
        self.assertEqual(
            result["review_groups"][0]["decision_reasons"],
            ["proposal_only_pose_requires_human_review"],
        )

    def test_derived_candidate_inherits_obb_proposal_guard(self):
        source = {
            "placements": {"a": {}, "b": {}},
            "proposal_only": True,
            "review_required": True,
            "can_auto_accept": False,
            "obb_insertion_history": [{
                "connection_id": "edge",
                "moving_axis_role": "middle",
            }],
        }

        derived = _candidate_from_placements(
            {},
            source["placements"],
            source,
            origin="joinable_pose_parameter_search",
            score_penalty=0.1,
            extra={},
        )

        self.assertTrue(derived["proposal_only"])
        self.assertTrue(derived["review_required"])
        self.assertFalse(derived["can_auto_accept"])
        self.assertEqual(
            derived["obb_insertion_history"],
            source["obb_insertion_history"],
        )

    def test_derived_candidate_inherits_planar_footprint_guard(self):
        source = {
            "placements": {"a": {}, "b": {}},
            "proposal_only": True,
            "review_required": True,
            "can_auto_accept": False,
            "planar_footprint_history": [{
                "connection_id": "edge",
                "independent_evidence_count": 2,
                "has_multi_evidence_support": True,
            }],
        }

        derived = _candidate_from_placements(
            {},
            source["placements"],
            source,
            origin="joinable_pose_parameter_search",
            score_penalty=0.1,
            extra={},
        )

        self.assertTrue(derived["proposal_only"])
        self.assertTrue(derived["review_required"])
        self.assertFalse(derived["can_auto_accept"])
        self.assertEqual(
            derived["planar_footprint_history"],
            source["planar_footprint_history"],
        )

    def test_multi_planar_containment_closes_pose_for_review_only(self):
        features = {
            "host": {
                "bbox": {"min": [0, 0, 0], "max": [10, 10, 10]},
                "planes": [
                    {"position": [2, 0, 0], "normal": [1, 0, 0]},
                    {"position": [0, 2, 0], "normal": [0, 1, 0]},
                ],
            },
            "insert": {
                "bbox": {"min": [2, 2, 2], "max": [4, 4, 4]},
                "planes": [
                    {"position": [2, 0, 0], "normal": [-1, 0, 0]},
                    {"position": [0, 2, 0], "normal": [0, 1, 0]},
                ],
            },
        }
        graph = {"selected": [{
            "connection_id": "c1",
            "parts": ["host", "insert"],
            "relation_types": ["planar_mate", "planar_align", "pocket_mate"],
            "matches": [
                {
                    "type": "planar_mate",
                    "parts": ["host", "insert"],
                    "feat_a_idx": 0,
                    "feat_b_idx": 0,
                },
                {
                    "type": "planar_align",
                    "parts": ["host", "insert"],
                    "feat_a_idx": 1,
                    "feat_b_idx": 1,
                },
                {
                    "type": "pocket_mate",
                    "parts": ["host", "insert"],
                    "pocket_a": {"center": [0, 0, 0]},
                    "pocket_b": {"center": [9, 9, 9]},
                },
            ],
        }]}
        candidate = {"placements": {"host": {}, "insert": {}}}

        closure = _pose_closure(candidate, graph, features)

        self.assertTrue(closure["fully_closed"])
        self.assertTrue(closure["review_required"])
        self.assertEqual(
            closure["connections"][0]["closure_evidence"],
            "multi_planar_plus_strict_containment",
        )

    def test_large_exact_budget_reserves_containment_candidate(self):
        def row(rank, score, kind):
            return (
                rank,
                {"total_score": score},
                {
                    "constraint_closure": {
                        "fully_closed": True,
                        "closure_ratio": 1.0,
                    },
                    "group_pose_precheck_score": score,
                    "overlap_objective": {
                        "bbox_overlap_items": [{"pair_kind": kind}],
                    },
                },
            )

        selected = _select_exact_rank_budget(
            [
                row(1, 1.0, "selected_pair_containment"),
                row(2, 9.0, "selected_pair_overlap"),
                row(3, 8.0, "selected_pair_overlap"),
            ],
            budget=2,
            large_step_case=True,
        )

        self.assertEqual(selected, {1, 2})

    def test_large_exact_budget_zero_selects_nothing(self):
        selected = _select_exact_rank_budget(
            [(1, {"total_score": 1.0}, {
                "constraint_closure": {
                    "fully_closed": True,
                    "closure_ratio": 1.0,
                },
                "group_pose_precheck_score": 1.0,
                "overlap_objective": {"bbox_overlap_items": []},
            })],
            budget=0,
            large_step_case=True,
        )
        self.assertEqual(selected, set())

    def test_large_exact_budget_does_not_spend_slot_on_open_pose(self):
        def row(rank, closed, score):
            return (
                rank,
                {"total_score": score},
                {
                    "constraint_closure": {
                        "fully_closed": closed,
                        "closure_ratio": float(closed),
                    },
                    "group_pose_precheck_score": score,
                    "overlap_objective": {"bbox_overlap_items": []},
                },
            )

        selected = _select_exact_rank_budget(
            [row(1, False, 100.0), row(2, True, 1.0)],
            budget=1,
            large_step_case=True,
        )
        self.assertEqual(selected, {2})

    def test_large_exact_budget_covers_distinct_topologies_first(self):
        def row(rank, topology_rank, score):
            return (
                rank,
                {
                    "topology_id": f"t{topology_rank}",
                    "topology_rank": topology_rank,
                    "total_score": score,
                },
                {
                    "constraint_closure": {
                        "fully_closed": True,
                        "closure_ratio": 1.0,
                    },
                    "group_pose_precheck_score": score,
                    "overlap_objective": {"bbox_overlap_items": []},
                },
            )

        selected = _select_exact_rank_budget(
            [
                row(1, 1, 10.0),
                row(2, 1, 9.0),
                row(3, 2, 3.0),
                row(4, 3, 2.0),
            ],
            budget=3,
            large_step_case=True,
        )
        self.assertEqual(selected, {1, 3, 4})

    def test_large_exact_budget_covers_distinct_obb_roles(self):
        def row(rank, role, score):
            return (
                rank,
                {
                    "topology_id": "t1",
                    "topology_rank": 1,
                    "total_score": score,
                    "obb_insertion_history": [{
                        "connection_id": "edge",
                        "moving_axis_role": role,
                    }],
                },
                {
                    "constraint_closure": {
                        "fully_closed": True,
                        "closure_ratio": 1.0,
                    },
                    "group_pose_precheck_score": score,
                    "overlap_objective": {"bbox_overlap_items": []},
                },
            )

        selected = _select_exact_rank_budget(
            [
                row(1, "shortest", 10.0),
                row(2, "shortest", 9.0),
                row(3, "middle", 8.0),
                row(4, "longest", 7.0),
            ],
            budget=3,
            large_step_case=True,
        )
        self.assertEqual(selected, {1, 3, 4})

    def test_localized_interference_is_review_only(self):
        exact = {
            "status": "success",
            "collisions": [{
                "minimum_part_volume_ratio": 0.0011,
                "solid_intersection_count": 2,
                "intersection_volume_mm3": 192.0,
            }],
        }
        closure = {"fully_closed": True, "review_required": True}

        audit = _localized_interference_review(exact, closure)

        self.assertTrue(audit["eligible_for_review"])
        self.assertFalse(audit["can_auto_accept"])
        self.assertEqual(_exact_collision_risk(exact), (0.0011, 2, 192.0))

    def test_large_or_unreviewed_collision_is_not_localized_review(self):
        exact = {
            "status": "success",
            "collisions": [{
                "minimum_part_volume_ratio": 0.02,
                "solid_intersection_count": 1,
                "intersection_volume_mm3": 4000.0,
            }],
        }
        self.assertFalse(_localized_interference_review(
            exact,
            {"fully_closed": True, "review_required": True},
        )["eligible_for_review"])
        self.assertFalse(_localized_interference_review(
            {**exact, "collisions": [{
                "minimum_part_volume_ratio": 0.001,
                "solid_intersection_count": 1,
                "intersection_volume_mm3": 10.0,
            }]},
            {"fully_closed": True, "review_required": False},
        )["eligible_for_review"])

    def test_joinable_parameter_budget_covers_every_selected_edge(self):
        features, graph, source = self._two_leaf_axial_fixture()

        candidates = _joinable_pose_parameter_candidates(
            [source], graph, features, max_candidates=8
        )

        self.assertLessEqual(len(candidates), 8)
        independently_refined = {
            row["joinable_pose_search"]["connection_id"]
            for row in candidates
            if row["joinable_pose_search"]["refinement_phase"]
            == "independent_edge_quota"
        }
        self.assertEqual(independently_refined, {"edge_a", "edge_b"})

    def test_joinable_parameter_budget_composes_two_leaf_edges(self):
        features, graph, source = self._two_leaf_axial_fixture()

        candidates = _joinable_pose_parameter_candidates(
            [source], graph, features, max_candidates=8
        )

        composed = [
            row for row in candidates
            if row.get("candidate_origin") == "joinable_two_edge_pose_composition"
        ]
        self.assertTrue(composed)
        history_ids = {
            item["connection_id"]
            for item in composed[0]["joinable_pose_refinement_history"]
        }
        self.assertEqual(history_ids, {"edge_a", "edge_b"})
        self.assertTrue(
            composed[0]["placements"]["leaf_a.step"].get("rotate_sequence")
        )
        self.assertTrue(
            composed[0]["placements"]["leaf_b.step"].get("rotate_sequence")
        )

    def test_joinable_parameter_budget_preserves_diverse_obb_sources(self):
        features, graph, source = self._two_leaf_axial_fixture()
        sources = []
        for role in ("shortest", "middle"):
            sources.append({
                **source,
                "proposal_only": True,
                "review_required": True,
                "can_auto_accept": False,
                "obb_insertion_history": [{
                    "connection_id": "seed_edge",
                    "moving_axis_role": role,
                }],
            })

        candidates = _joinable_pose_parameter_candidates(
            sources,
            {"selected": [graph["selected"][0]]},
            features,
            max_candidates=2,
        )

        self.assertEqual(len(candidates), 2)
        self.assertEqual(
            {
                row["obb_insertion_history"][0]["moving_axis_role"]
                for row in candidates
            },
            {"shortest", "middle"},
        )
        self.assertTrue(all(row["proposal_only"] for row in candidates))

    def test_obb_budget_covers_axis_roles_on_every_selected_edge(self):
        features, graph, source = self._two_leaf_obb_fixture()

        candidates = _obb_insertion_pose_candidates(
            [source], graph, features, max_candidates=16
        )

        self.assertLessEqual(len(candidates), 16)
        for connection_id in ("dimm_edge", "cpu_edge"):
            rows = [
                row for row in candidates
                if row["obb_insertion"]["connection_id"] == connection_id
                and row["obb_insertion"]["refinement_phase"]
                == "independent_edge_quota"
            ]
            roles = {
                row["obb_insertion"]["moving_axis_role"] for row in rows
            }
            self.assertEqual(roles, {"shortest", "middle", "longest"})
            self.assertTrue(all(
                row["obb_insertion"]["axis_compatibility"] >= 0.90
                for row in rows
            ))
            representative_placements = {
                role: next(
                    row["placements"]
                    for row in rows
                    if row["obb_insertion"]["moving_axis_role"] == role
                )
                for role in roles
            }
            self.assertEqual(
                len({
                    json.dumps(value, sort_keys=True)
                    for value in representative_placements.values()
                }),
                3,
            )

    def test_planar_footprint_budget_composes_two_leaf_edges_review_only(self):
        features, graph, source = self._two_leaf_footprint_fixture()

        candidates = _planar_footprint_pose_candidates(
            [source], graph, features, max_candidates=16
        )

        self.assertLessEqual(len(candidates), 16)
        composed = [
            row for row in candidates
            if row.get("candidate_origin")
            == "planar_footprint_star_composition"
        ]
        self.assertTrue(composed)
        candidate = next(
            row for row in composed
            if set(row.get(
                "planar_footprint_refined_connection_ids"
            ) or []) == {"dimm_edge", "cpu_edge"}
        )
        closure = _pose_closure(candidate, graph, features)
        self.assertTrue(closure["fully_closed"])
        self.assertTrue(closure["review_required"])
        self.assertTrue(candidate["proposal_only"])
        self.assertFalse(candidate["can_auto_accept"])
        self.assertTrue(all(
            row["closure_evidence"]
            == "planar_footprint_multi_evidence_proposal"
            for row in closure["connections"]
        ))

    def test_obb_budget_composes_two_leaf_edges_review_only(self):
        features, graph, source = self._two_leaf_obb_fixture()

        candidates = _obb_insertion_pose_candidates(
            [source], graph, features, max_candidates=16
        )

        composed = [
            row for row in candidates
            if row.get("candidate_origin")
            == "obb_two_edge_insertion_composition"
        ]
        self.assertTrue(composed)
        self.assertTrue(any(
            set(row.get("obb_refined_connection_ids") or [])
            == {"dimm_edge", "cpu_edge"}
            for row in composed
        ))
        for row in composed:
            self.assertTrue(row["obb_insertion"]["review_required"])
            self.assertFalse(row["obb_insertion"]["can_auto_accept"])

    def test_obb_matched_feature_anchor_handles_eccentric_interface(self):
        features, graph, source = self._two_leaf_obb_fixture()

        candidates = _obb_insertion_pose_candidates(
            [source], graph, features, max_candidates=16
        )
        candidate = next(
            row for row in candidates
            if row["obb_insertion"]["connection_id"] == "dimm_edge"
            and row["obb_insertion"]["moving_axis_role"] == "middle"
            and row["obb_insertion"]["anchor_strategy"]
            == "matched_feature_centroid"
            and row["obb_insertion"]["sampled_depth_mm"] == 0.0
        )

        placed_anchor = transform_point(
            features["dimm.step"]["planes"][0]["centroid"],
            candidate["placements"]["dimm.step"],
        )
        self.assertTrue(all(
            abs(value) <= 1e-8 for value in placed_anchor
        ))
        self.assertTrue(candidate["proposal_only"])
        self.assertTrue(candidate["review_required"])

    def test_axial_compound_budget_closes_physical_edge_but_c6_stays_review(self):
        features, graph, source = self._axial_compound_fixture()

        candidates = _axial_compound_pose_candidates(
            [source], graph, features, max_candidates=6
        )

        self.assertEqual(len(candidates), 6)
        self.assertTrue(all(row["proposal_only"] for row in candidates))
        self.assertTrue(all(row["review_required"] for row in candidates))
        self.assertEqual({
            row["axial_compound_interface"]["axis_polarity"]
            for row in candidates
        }, {1})
        self.assertTrue(all(
            len(row["axial_compound_interface"]["phase_orbit_degrees"]) == 6
            for row in candidates
        ))
        closure = _pose_closure(candidates[0], graph, features)
        self.assertTrue(closure["fully_closed"])
        self.assertTrue(closure["review_required"])
        edge = closure["connections"][0]
        self.assertEqual(edge["closure_evidence"], "axial_compound_interface")
        self.assertIn(
            "compound_coaxial_radial_end_face_phase",
            edge["satisfied_relation_types"],
        )
        self.assertTrue(
            edge["axial_compound_evidence"][0][
                "compound_constraints_satisfied"
            ]
        )
        self.assertEqual(
            edge["axial_compound_evidence"][0]["current_pose_validation"][
                "collision_scope"
            ],
            "deferred_to_exact_occt",
        )

    def test_axial_compound_slot_witness_propagates_c1_through_derivation(self):
        features, graph, source = self._axial_compound_fixture(
            with_slot_witness=True
        )
        candidates = _axial_compound_pose_candidates(
            [source], graph, features, max_candidates=4
        )

        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        detail = candidate["axial_compound_interface"]
        self.assertFalse(candidate["proposal_only"])
        self.assertFalse(candidate["review_required"])
        self.assertEqual(detail["whole_part_symmetry_order"], 1)
        self.assertEqual(detail["phase_status"], "resolved_by_asymmetric_witness")
        self.assertEqual(len(detail["phase_orbit_degrees"]), 1)
        self.assertTrue(detail["phase_witness"])
        self.assertEqual(
            detail["phase_witness"][0]["fixed_kind"],
            "topological_key_slot",
        )

        derived = _candidate_from_placements(
            features,
            candidate["placements"],
            candidate,
            origin="bounded_residual_refinement",
            score_penalty=0.01,
            extra={},
        )
        self.assertEqual(
            derived["axial_compound_history"],
            candidate["axial_compound_history"],
        )
        closure = _pose_closure(derived, graph, features)
        self.assertTrue(closure["fully_closed"])
        self.assertFalse(closure["review_required"])

    def test_axial_compound_budget_composes_two_central_axis_leaves(self):
        features, one_edge_graph, source = self._axial_compound_fixture(
            with_slot_witness=True
        )
        features["moving_b.step"] = json.loads(json.dumps(
            features["moving.step"]
        ))
        source["placements"]["moving_b.step"] = {
            "translate": [0.0, 0.0, 0.0]
        }
        first = one_edge_graph["selected"][0]
        graph = {"selected": [
            first,
            {
                **json.loads(json.dumps(first)),
                "connection_id": "axial_edge_b",
                "parts": ["fixed.step", "moving_b.step"],
                "matches": [{
                    **json.loads(json.dumps(first["matches"][0])),
                    "parts": ["fixed.step", "moving_b.step"],
                }],
            },
        ]}

        candidates = _axial_compound_pose_candidates(
            [source], graph, features, max_candidates=8
        )
        composed = [
            row for row in candidates
            if row.get("candidate_origin")
            == "axial_compound_star_composition"
        ]

        self.assertTrue(composed)
        candidate = composed[0]
        self.assertEqual(
            set(candidate["axial_compound_refined_connection_ids"]),
            {"axial_edge", "axial_edge_b"},
        )
        self.assertTrue(all(
            row["fixed_part"] == "fixed.step"
            for row in candidate["axial_compound_history"]
        ))
        closure = _pose_closure(candidate, graph, features)
        self.assertTrue(closure["fully_closed"])
        self.assertEqual(closure["closed_connection_count"], 2)


if __name__ == "__main__":
    unittest.main()
