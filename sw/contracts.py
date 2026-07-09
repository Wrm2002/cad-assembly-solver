"""Versioned public data contracts for the pool-matching pipeline."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


SCHEMA_VERSION = "1.0.0"
LENGTH_UNIT = "mm"
ANGLE_UNIT = "degree"
COORDINATE_FRAME = "local_part"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Units(StrictModel):
    length: Literal["mm"] = LENGTH_UNIT
    angle: Literal["degree"] = ANGLE_UNIT


class BoundingBox(StrictModel):
    minimum: list[float] = Field(min_length=3, max_length=3)
    maximum: list[float] = Field(min_length=3, max_length=3)
    size: list[float] = Field(min_length=3, max_length=3)


class DetectionStatus(str, Enum):
    measured = "measured"
    heuristic = "heuristic"
    unavailable = "unavailable"


class FeatureSummary(StrictModel):
    feature_id: str
    kind: str
    parameters: dict[str, Any]
    detection_status: DetectionStatus = DetectionStatus.measured
    reason: str | None = None


class PartFeature(StrictModel):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    part_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    source_file: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    units: Units = Field(default_factory=Units)
    coordinate_frame: Literal["local_part"] = COORDINATE_FRAME
    bbox: BoundingBox
    volume: float | None = Field(default=None, ge=0)
    center_of_mass: list[float] | None = None
    principal_axes: list[list[float]] = Field(default_factory=list)
    principal_axes_method: str
    planar_faces: list[FeatureSummary] = Field(default_factory=list)
    cylindrical_faces: list[FeatureSummary] = Field(default_factory=list)
    holes: list[FeatureSummary] = Field(default_factory=list)
    hole_patterns: list[FeatureSummary] = Field(default_factory=list)
    geometric_class: str
    functional_semantics: dict[str, Any] = Field(default_factory=dict)
    extraction: dict[str, Any]


class CandidateStatus(str, Enum):
    generated = "generated"
    kept = "kept"
    removed = "removed"
    rejected_by_prescreen = "rejected_by_prescreen"


class CandidateEdge(StrictModel):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    candidate_id: str
    parts: list[str] = Field(min_length=2, max_length=2)
    candidate_type: str
    feature_refs: list[str] = Field(default_factory=list)
    geometry_score: float = Field(ge=0, le=1)
    confidence: str
    geometric_evidence: list[str]
    collision_free: bool | None = None
    status: CandidateStatus
    audit_reason: dict[str, Any]

    @model_validator(mode="after")
    def distinct_parts(self):
        if self.parts[0] == self.parts[1]:
            raise ValueError("candidate parts must be distinct")
        return self


class PairEdge(StrictModel):
    """Provider-aware pair relation used by functional proposal generation."""

    schema_version: Literal["2.0.0"] = "2.0.0"
    pair_edge_id: str
    parts: list[str] = Field(min_length=2, max_length=2)
    providers: list[str] = Field(default_factory=list)
    provider_count: int = Field(default=0, ge=0)
    analytic_candidate_ids: list[str] = Field(default_factory=list)
    learned_candidate_ids: list[str] = Field(default_factory=list)
    candidate_ids: list[str] = Field(default_factory=list)
    relation_types: list[str] = Field(default_factory=list)
    best_analytic_geometry_score: float = Field(default=0.0, ge=0, le=1)
    best_joinable_probability: float = Field(default=0.0, ge=0, le=1)
    best_joinable_rank: int | None = Field(default=None, ge=1)
    physical_evidence: list[str] = Field(default_factory=list)
    independent_physical_evidence_count: int = Field(default=0, ge=0)
    provider_agreement_present: bool = False
    provider_agreement_counts_as_independent_evidence: Literal[False] = False
    learned_only: bool = False
    critical_learned_only: bool = False
    audit_trace: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def distinct_parts(self):
        if self.parts[0] == self.parts[1]:
            raise ValueError("pair edge parts must be distinct")
        return self


class GroupProposal(StrictModel):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    group_id: str
    parts: list[str] = Field(min_length=1, max_length=6)
    candidate_edges: list[str] = Field(default_factory=list)
    geometry_score: float = Field(ge=0, le=1)
    connected: bool
    status: str
    reasons: list[str]
    pair_edge_ids: list[str] = Field(default_factory=list)
    assembly_family: str = "unknown"
    center_part_ids: list[str] = Field(default_factory=list)
    role_assignment: dict[str, Any] = Field(default_factory=dict)
    slot_coverage: dict[str, Any] = Field(default_factory=dict)
    completeness_status: str = "unknown"
    completeness_score: float = Field(default=0.0, ge=0, le=1)
    relation_coverage: float = Field(default=0.0, ge=0, le=1)
    independent_evidence_count: int = Field(default=0, ge=0)
    missing_required_slots: list[str] = Field(default_factory=list)
    missing_required_relations: list[str] = Field(default_factory=list)
    subset_of: list[str] = Field(default_factory=list)
    supersets: list[str] = Field(default_factory=list)
    proposal_cluster_id: str | None = None
    status_modifiers: list[str] = Field(default_factory=list)
    ranking_features: dict[str, float] = Field(default_factory=dict)
    audit_trace: list[str] = Field(default_factory=list)


class ValidationStatus(str, Enum):
    success = "success"
    partial_success = "partial_success"
    failed = "failed"


class ValidationResult(StrictModel):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    subject_id: str
    status: ValidationStatus
    num_parts: int = Field(ge=1)
    solved_parts: list[str]
    unsolved_parts: list[str]
    max_constraint_residual: float | None = Field(default=None, ge=0)
    collision_count: int = Field(ge=0)
    severe_penetration_count: int = Field(ge=0)
    warnings: list[str]
    metrics: dict[str, Any]


class AgentEvent(StrictModel):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    event_id: str
    timestamp: datetime
    run_id: str
    sequence: int = Field(ge=0)
    state: str
    action: str
    tool: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    outcome: str
    evidence_refs: list[str] = Field(default_factory=list)
    retry_count: int = Field(default=0, ge=0)
    message: str


class SemanticDecision(StrictModel):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    proposal_id: str
    verdict: Literal["accept", "reject", "abstain"]
    plausibility_score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    reason_codes: list[str] = Field(max_length=8)
    explanation: str = Field(max_length=800)
    risk_flags: list[str] = Field(default_factory=list, max_length=8)


class AssemblyInterface(StrictModel):
    """One localized geometric interface on a part."""

    feature_kind: Literal["cylinder", "plane", "pocket", "brep_entity", "unknown"]
    feature_index: int | None = Field(default=None, ge=0)
    feature_id: str | None = None
    geometry: dict[str, Any] = Field(default_factory=dict)


class AssemblyConstraint(StrictModel):
    """One labelled geometric constraint supporting a direct connection."""

    constraint_id: str
    connection_id: str
    parts: list[str] = Field(min_length=2, max_length=2)
    relation_type: Literal[
        "coaxial",
        "clearance",
        "planar_mate",
        "planar_align",
        "pocket_mate",
    ]
    interface_a: AssemblyInterface
    interface_b: AssemblyInterface
    score: float = Field(ge=0.0, le=1.0)
    confidence: Literal["high", "medium", "low"]
    providers: list[str] = Field(default_factory=list)
    used_for_pose: bool = False
    constraint_residual: float | None = Field(default=None, ge=0.0)
    evidence: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def distinct_parts(self):
        if self.parts[0] == self.parts[1]:
            raise ValueError("assembly constraint parts must be distinct")
        return self


class DirectAssemblyConnection(StrictModel):
    """A selected direct part-to-part edge, possibly supported by many constraints."""

    connection_id: str
    parts: list[str] = Field(min_length=2, max_length=2)
    primary_relation_type: Literal[
        "coaxial",
        "clearance",
        "planar_mate",
        "planar_align",
        "pocket_mate",
    ]
    supporting_relation_types: list[str]
    assembly_method_relation_types: list[str] = Field(default_factory=list)
    assembly_method_reason: list[str] = Field(default_factory=list)
    constraint_ids: list[str]
    score: float = Field(ge=0.0, le=1.0)
    confidence: Literal["high", "medium", "low"]
    selection_role: Literal["connected_skeleton", "additional_supported_edge"]
    constraint_closed_in_selected_pose: bool
    review_required: bool
    providers: list[str] = Field(default_factory=list)
    relative_transform_a_to_b: list[list[float]]
    joinable_interface_candidates: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_connection(self):
        if self.parts[0] == self.parts[1]:
            raise ValueError("direct connection parts must be distinct")
        if len(self.relative_transform_a_to_b) != 4 or any(
            len(row) != 4 for row in self.relative_transform_a_to_b
        ):
            raise ValueError("relative transform must be a 4x4 matrix")
        return self


class KnownGroupAssemblyResult(StrictModel):
    """Primary delivery contract for a known-related set of STEP parts."""

    schema_version: Literal["2.0.0"] = "2.0.0"
    task: Literal["known_group_assembly_relation_recognition"] = (
        "known_group_assembly_relation_recognition"
    )
    assembly_id: str
    input_assumption: Literal["all_parts_belong_to_one_assembly"] = (
        "all_parts_belong_to_one_assembly"
    )
    parts: list[str] = Field(min_length=1, max_length=5)
    reference_part: str
    assembly_connected: bool
    pose_status: Literal["valid", "failed", "uncertain"]
    direct_connections: list[DirectAssemblyConnection]
    assembly_relations: list[AssemblyConstraint]
    components: list[dict[str, Any]]
    unresolved_parts: list[str] = Field(default_factory=list)
    collision_validation: dict[str, Any]
    candidate_summary: dict[str, Any]
    limitations: list[str] = Field(default_factory=list)


CONTRACT_MODELS = {
    "part_feature": PartFeature,
    "candidate_edge": CandidateEdge,
    "pair_edge": PairEdge,
    "group_proposal": GroupProposal,
    "validation_result": ValidationResult,
    "agent_event": AgentEvent,
    "semantic_decision": SemanticDecision,
    "known_group_assembly_result": KnownGroupAssemblyResult,
}


def write_json_schemas(output_dir: str | Path) -> list[Path]:
    import json

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, model in CONTRACT_MODELS.items():
        path = output / f"{name}.schema.json"
        path.write_text(
            json.dumps(model.model_json_schema(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        paths.append(path)
    return paths
