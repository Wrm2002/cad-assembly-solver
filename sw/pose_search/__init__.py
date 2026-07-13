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
from .key_slot_features import KeySlotCandidate, extract_key_slot_evidence
from .prismatic_key_features import (
    PrismaticKeyFeature,
    extract_prismatic_key_feature,
    match_prismatic_key_to_slots,
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
    "KeySlotCandidate",
    "extract_key_slot_evidence",
    "PrismaticKeyFeature",
    "extract_prismatic_key_feature",
    "match_prismatic_key_to_slots",
]
