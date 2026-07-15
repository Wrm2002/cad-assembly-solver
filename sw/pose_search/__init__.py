"""Reusable pose-search primitives for known-pair and known-group assembly.

The package is deliberately independent from case names and relation labels.
JoinABLe supplies ranked B-Rep entity pairs and joint-axis seeds; this package
turns those seeds into rigid transforms and evaluates them.  Exact OCCT
collision validation remains a separate final gate.
"""

from .joinable_search import (
    JointAxisSeed,
    JoinablePoseSearch,
    PoseEvaluation,
    PoseSearchResult,
)
from .group_pose import (
    PairPoseSeed,
    compose_group_pose,
    compose_group_pose_hypotheses,
    load_joinable_pair_pose,
    load_joinable_pair_pose_candidates,
    load_joinable_pair_pose_candidate_directory,
    load_joinable_pair_pose_directory,
)
from .transforms import matrix_to_placement, placement_to_matrix
from .axial_features import (
    AxialCircularFeature,
    AxialPlanarWitness,
    CircularPattern,
    RotationHypothesis,
    build_circular_patterns,
    extract_axial_circular_features,
    extract_axial_planar_witnesses,
    generate_axial_rotation_hypotheses,
)
from .axial_compound_interface import (
    PHASE_CONVENTION,
    construct_compound_transform,
    correspondence_phase_degrees,
    phase_residual_degrees,
    recall_axial_compound_candidates,
    validate_axial_compound_pose,
)
from .key_slot_features import KeySlotCandidate, extract_key_slot_evidence
from .prismatic_key_features import (
    PrismaticKeyFeature,
    extract_prismatic_key_feature,
    match_prismatic_key_to_slots,
)
from .interface_roi import build_roi_subgraph, match_roi_pairs, rank_interface_rois
from .obb_insertion import enumerate_axis_role_frames
from .planar_footprint import recall_planar_footprint_proposals
from .dominant_planar_envelope import (
    derive_dominant_planar_envelope,
    infer_functional_body_obb,
)
from .enclosure_bay import (
    propose_enclosure_bay_placements,
    propose_enclosure_bays,
)
from .edge_slot_interface import (
    propose_edge_slot_interface_placements,
    recall_edge_slot_interface_proposals,
)
from .collision_clearance_refinement import (
    propose_collision_clearance_refinement,
)

__all__ = [
    "JointAxisSeed",
    "JoinablePoseSearch",
    "PoseEvaluation",
    "PoseSearchResult",
    "matrix_to_placement",
    "placement_to_matrix",
    "PairPoseSeed",
    "compose_group_pose",
    "compose_group_pose_hypotheses",
    "load_joinable_pair_pose",
    "load_joinable_pair_pose_candidates",
    "load_joinable_pair_pose_candidate_directory",
    "load_joinable_pair_pose_directory",
    "AxialCircularFeature",
    "AxialPlanarWitness",
    "CircularPattern",
    "RotationHypothesis",
    "extract_axial_circular_features",
    "extract_axial_planar_witnesses",
    "build_circular_patterns",
    "generate_axial_rotation_hypotheses",
    "PHASE_CONVENTION",
    "construct_compound_transform",
    "correspondence_phase_degrees",
    "phase_residual_degrees",
    "recall_axial_compound_candidates",
    "validate_axial_compound_pose",
    "KeySlotCandidate",
    "extract_key_slot_evidence",
    "PrismaticKeyFeature",
    "extract_prismatic_key_feature",
    "match_prismatic_key_to_slots",
    "rank_interface_rois",
    "enumerate_axis_role_frames",
    "build_roi_subgraph",
    "match_roi_pairs",
    "recall_planar_footprint_proposals",
    "derive_dominant_planar_envelope",
    "infer_functional_body_obb",
    "propose_enclosure_bay_placements",
    "propose_enclosure_bays",
    "propose_edge_slot_interface_placements",
    "recall_edge_slot_interface_proposals",
    "propose_collision_clearance_refinement",
]
