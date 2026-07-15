"""Three-stage visual-semantic guidance for CAD assembly candidate recall.

The module deliberately stops before metric pose estimation.  It recognizes a
part role, assesses numbered carrier regions, and synthesizes typed constraints
that may *schedule* geometry candidates.  It cannot accept an assembly, delete
all protected geometry candidates, or bypass B-Rep/OCCT validation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field

from multimodal_reviewer import QwenVLReviewer


PROMPT1_VERSION = "part_role_functional_faces.v2"
PROMPT2_VERSION = "carrier_region_semantics.v2"
PROMPT3_VERSION = "assembly_hypothesis_synthesis.v2"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FunctionalFace(StrictModel):
    face_id: str
    role: Literal[
        "service_face",
        "mounting_face",
        "internal_connector",
        "insertion_end",
        "guide_face",
        "functional_body",
        "unknown",
    ]
    evidence: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class SymmetryAssessment(StrictModel):
    has_symmetry: bool
    symmetry_type: str
    orientation_ambiguity: bool


class PartRoleAnalysis(StrictModel):
    part_id: str
    part_role: str
    possible_names: list[str]
    assembly_family_candidates: list[str]
    functional_description: str
    functional_faces: list[FunctionalFace]
    likely_assembly_actions: list[
        Literal[
            "insert",
            "slide",
            "mate",
            "nest",
            "fasten",
            "coaxial_insert",
            "unknown",
        ]
    ]
    principal_axis_semantics: str
    symmetry: SymmetryAssessment
    risks: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    review_required: bool


class CarrierAssessment(StrictModel):
    role: str
    evidence: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


class RegionAssessment(StrictModel):
    region_id: str
    region_type: str
    compatible_part_roles: list[str]
    opening_direction: str
    internal_direction: str
    possible_insertion_axis: str
    visible_interfaces: list[str]
    semantic_score: float = Field(ge=0.0, le=1.0)
    reasons: list[str]
    possible_equivalent_slot: bool
    forbidden_for_current_part: bool


class ForbiddenRegion(StrictModel):
    region_id: str
    reason: str


class CarrierRegionAnalysis(StrictModel):
    carrier: CarrierAssessment
    region_assessments: list[RegionAssessment]
    preferred_region_ids: list[str] = Field(max_length=3)
    forbidden_region_ids: list[ForbiddenRegion]
    equivalent_region_groups: list[list[str]]
    confidence: float = Field(ge=0.0, le=1.0)
    review_required: bool


class PartAssessment(StrictModel):
    role: str
    possible_names: list[str]
    functional_description: str
    evidence: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


class PreferredRegion(StrictModel):
    region_id: str
    semantic_score: float = Field(ge=0.0, le=1.0)
    reasons: list[str]
    possible_equivalent_slot: bool


class AssemblyHypothesisBody(StrictModel):
    assembly_family: str
    relation: str
    assembly_action: Literal["insert", "mate", "slide", "nest", "fasten", "unknown"]
    target_region_type: str
    preferred_region_ids: list[PreferredRegion] = Field(max_length=3)
    forbidden_region_ids: list[ForbiddenRegion]


class OrientationConstraints(StrictModel):
    external_face_ids: list[str]
    internal_face_ids: list[str]
    mounting_face_ids: list[str]
    insertion_axis_relative_to_part: str
    service_face_must_remain_visible: bool
    mirror_transform_allowed: bool
    reasons: list[str]


class RequiredGeometryEvidence(StrictModel):
    interface_type: str
    part_feature_ids: list[str]
    carrier_region_ids: list[str]
    importance: Literal["required", "supporting", "locking_only"]
    reason: str


class AmbiguityAssessment(StrictModel):
    has_multiple_valid_regions: bool
    equivalent_region_ids: list[str]
    cannot_be_resolved_from_images: bool
    reason: str


class RiskAssessment(StrictModel):
    possible_visual_misclassification: bool
    possible_hidden_interface: bool
    possible_scale_ambiguity: bool
    possible_symmetry: bool
    notes: list[str]


class AssemblyHypothesis(StrictModel):
    carrier: CarrierAssessment
    part: PartAssessment
    assembly_hypothesis: AssemblyHypothesisBody
    orientation_constraints: OrientationConstraints
    required_geometry_evidence: list[RequiredGeometryEvidence]
    ambiguity: AmbiguityAssessment
    risk: RiskAssessment
    semantic_confidence: float = Field(ge=0.0, le=1.0)
    review_required: bool
    suggested_action: Literal["prioritize_regions", "review", "unresolved"]


PART_ROLE_SYSTEM_PROMPT = r"""
You are a visual semantic analyzer for mechanical CAD parts.  Do NOT calculate
a pose, rotation, translation, or 4x4 matrix.  Infer what the numbered part
looks like, its likely functional role, its service/mounting/connector faces,
and its likely assembly action from standardized multiview renders and the
provided B-Rep summary.

Prioritize visible engineering evidence: buttons, USB or power sockets, fans,
connectors, flanges, repeated holes, rails, latches, keyways, openings and
maintenance faces.  A filename is weak evidence only.  Distinguish service
face, insertion end, mounting flange and functional body.  If evidence is
insufficient, use "unknown" and set review_required=true.  Never invent face
IDs that are absent from the input.

Return ONLY strict JSON with exactly these keys:
part_id, part_role, possible_names, assembly_family_candidates,
functional_description, functional_faces, likely_assembly_actions,
principal_axis_semantics, symmetry, risks, confidence, review_required.
Functional face roles must be one of service_face, mounting_face,
internal_connector, insertion_end, guide_face, functional_body, unknown.
Assembly actions must be one of insert, slide, mate, nest, fasten,
coaxial_insert, unknown.  Every functional_faces item must include face_id,
role, evidence (array), and confidence (0..1).  Do not replace an object with
a string or a list with an object.
""".strip()


CARRIER_REGION_SYSTEM_PROMPT = r"""
You are a visual semantic analyzer for receiving regions on a mechanical CAD
carrier.  Do NOT calculate a pose or matrix.  Use the carrier multiviews,
numbered candidate-region previews, region geometry summaries, and the prior
part-role JSON to assess which regions can functionally receive the part.

Do not select a region merely because it is close, large, collision-free, or
bounding-box compatible.  Examine openings, guide surfaces, stops, service
access, connector direction, carrier edges and whether the service face can
remain outside.  Preserve all genuinely equivalent slots.  Select at most
three preferred region IDs.  If no region is reliable, return an empty list
and review_required=true.  Never invent a region ID not present in the input.

Return ONLY strict JSON with exactly these keys: carrier,
region_assessments, preferred_region_ids, forbidden_region_ids,
equivalent_region_groups, confidence, review_required.
carrier must be an object containing role, evidence and confidence.  Every
region assessment must include all fields in the supplied JSON schema.
""".strip()


ASSEMBLY_SYNTHESIS_SYSTEM_PROMPT = r"""
You are a mechanical CAD assembly-hypothesis synthesizer.  Merge the previous
part-role and carrier-region analyses into typed guidance for a geometry pose
solver.  Do NOT output a pose, metric translation, rotation or 4x4 matrix.
Semantic plausibility is not geometric feasibility.

Classify evidence as required, supporting, or locking_only.  For an inserted
module, the passable opening, guide faces and stop may be required while holes
may be locking_only.  Do not automatically use a hole pattern as the first
locator.  Preserve equivalent regions and uncertainty.  Never invent face or
region IDs absent from the two prior stages.

Return ONLY strict JSON with exactly these keys: carrier, part,
assembly_hypothesis, orientation_constraints, required_geometry_evidence,
ambiguity, risk, semantic_confidence, review_required, suggested_action.
The suggested action must be prioritize_regions, review, or unresolved.
""".strip()


def _bounded_confidence(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if item is not None]
    return []


def _normalize_carrier(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        return {"role": value, "evidence": [], "confidence": 0.0}
    row = value if isinstance(value, dict) else {}
    return {
        "role": str(row.get("role", "unknown")),
        "evidence": _string_list(row.get("evidence")),
        "confidence": _bounded_confidence(row.get("confidence")),
    }


def _normalize_part_output(value: Any) -> dict[str, Any]:
    row = value if isinstance(value, dict) else {}
    confidence = _bounded_confidence(row.get("confidence"))
    raw_faces = row.get("functional_faces") or []
    if isinstance(raw_faces, dict):
        raw_faces = [
            {"face_id": face_id, "role": role}
            for face_id, role in raw_faces.items()
        ]
    allowed_roles = {
        "service_face", "mounting_face", "internal_connector",
        "insertion_end", "guide_face", "functional_body", "unknown",
    }
    faces = []
    for face in raw_faces if isinstance(raw_faces, list) else []:
        if not isinstance(face, dict) or not face.get("face_id"):
            continue
        role = str(face.get("role", "unknown"))
        faces.append(
            {
                "face_id": str(face["face_id"]),
                "role": role if role in allowed_roles else "unknown",
                "evidence": _string_list(face.get("evidence")),
                "confidence": _bounded_confidence(
                    face.get("confidence"), confidence
                ),
            }
        )
    raw_symmetry = row.get("symmetry")
    symmetry = raw_symmetry if isinstance(raw_symmetry, dict) else {}
    allowed_actions = {
        "insert", "slide", "mate", "nest", "fasten",
        "coaxial_insert", "unknown",
    }
    actions = [
        action if action in allowed_actions else "unknown"
        for action in _string_list(row.get("likely_assembly_actions"))
    ] or ["unknown"]
    return {
        "part_id": str(row.get("part_id", "unknown")),
        "part_role": str(row.get("part_role", "unknown")),
        "possible_names": _string_list(row.get("possible_names")),
        "assembly_family_candidates": _string_list(
            row.get("assembly_family_candidates")
        ),
        "functional_description": str(
            row.get("functional_description", "")
        ),
        "functional_faces": faces,
        "likely_assembly_actions": actions,
        "principal_axis_semantics": str(
            row.get("principal_axis_semantics", "unknown")
        ),
        "symmetry": {
            "has_symmetry": bool(symmetry.get("has_symmetry", False)),
            "symmetry_type": str(symmetry.get("symmetry_type", "unknown")),
            "orientation_ambiguity": bool(
                symmetry.get("orientation_ambiguity", True)
            ),
        },
        "risks": _string_list(row.get("risks")),
        "confidence": confidence,
        "review_required": bool(row.get("review_required", confidence < 0.8)),
    }


def _normalize_regions_output(value: Any) -> dict[str, Any]:
    row = value if isinstance(value, dict) else {}
    raw_regions = row.get("region_assessments") or []
    if isinstance(raw_regions, dict):
        raw_regions = [
            {"region_id": region_id, **(entry if isinstance(entry, dict) else {})}
            for region_id, entry in raw_regions.items()
        ]
    regions = []
    for region in raw_regions if isinstance(raw_regions, list) else []:
        if not isinstance(region, dict) or not region.get("region_id"):
            continue
        regions.append(
            {
                "region_id": str(region["region_id"]),
                "region_type": str(region.get("region_type", "unknown")),
                "compatible_part_roles": _string_list(
                    region.get("compatible_part_roles")
                ),
                "opening_direction": str(region.get("opening_direction", "unknown")),
                "internal_direction": str(region.get("internal_direction", "unknown")),
                "possible_insertion_axis": str(
                    region.get("possible_insertion_axis", "unknown")
                ),
                "visible_interfaces": _string_list(region.get("visible_interfaces")),
                "semantic_score": _bounded_confidence(region.get("semantic_score")),
                "reasons": _string_list(region.get("reasons")),
                "possible_equivalent_slot": bool(
                    region.get("possible_equivalent_slot", False)
                ),
                "forbidden_for_current_part": bool(
                    region.get("forbidden_for_current_part", False)
                ),
            }
        )
    preferred = []
    for item in row.get("preferred_region_ids") or []:
        region_id = item.get("region_id") if isinstance(item, dict) else item
        if region_id is not None:
            preferred.append(str(region_id))
    forbidden = []
    for item in row.get("forbidden_region_ids") or []:
        if isinstance(item, dict):
            region_id = item.get("region_id")
            reason = item.get("reason", "model_marked_forbidden")
        else:
            region_id, reason = item, "model_marked_forbidden"
        if region_id is not None:
            forbidden.append({"region_id": str(region_id), "reason": str(reason)})
    confidence = _bounded_confidence(row.get("confidence"))
    return {
        "carrier": _normalize_carrier(row.get("carrier")),
        "region_assessments": regions,
        "preferred_region_ids": preferred[:3],
        "forbidden_region_ids": forbidden,
        "equivalent_region_groups": [
            _string_list(group)
            for group in (row.get("equivalent_region_groups") or [])
            if isinstance(group, (list, tuple))
        ],
        "confidence": confidence,
        "review_required": bool(row.get("review_required", confidence < 0.8)),
    }


def _normalize_hypothesis_output(value: Any) -> dict[str, Any]:
    row = value if isinstance(value, dict) else {}
    raw_part = row.get("part")
    part = raw_part if isinstance(raw_part, dict) else {
        "role": raw_part if isinstance(raw_part, str) else "unknown"
    }
    raw_body = row.get("assembly_hypothesis")
    body = raw_body if isinstance(raw_body, dict) else {}
    preferred = []
    for item in body.get("preferred_region_ids") or []:
        if isinstance(item, dict):
            region_id = item.get("region_id")
            semantic_score = item.get("semantic_score")
            reasons = item.get("reasons")
            equivalent = item.get("possible_equivalent_slot", False)
        else:
            region_id = item
            semantic_score = row.get("semantic_confidence", 0.0)
            reasons = []
            equivalent = False
        if region_id is not None:
            preferred.append(
                {
                    "region_id": str(region_id),
                    "semantic_score": _bounded_confidence(semantic_score),
                    "reasons": _string_list(reasons),
                    "possible_equivalent_slot": bool(equivalent),
                }
            )
    forbidden = []
    for item in body.get("forbidden_region_ids") or []:
        if isinstance(item, dict):
            region_id = item.get("region_id")
            reason = item.get("reason", "model_marked_forbidden")
        else:
            region_id, reason = item, "model_marked_forbidden"
        if region_id is not None:
            forbidden.append({"region_id": str(region_id), "reason": str(reason)})
    raw_orientation = row.get("orientation_constraints")
    orientation = raw_orientation if isinstance(raw_orientation, dict) else {}
    evidence_rows = []
    for item in row.get("required_geometry_evidence") or []:
        if not isinstance(item, dict):
            continue
        importance = str(item.get("importance", "supporting"))
        if importance not in {"required", "supporting", "locking_only"}:
            importance = "supporting"
        evidence_rows.append(
            {
                "interface_type": str(item.get("interface_type", "unknown")),
                "part_feature_ids": _string_list(item.get("part_feature_ids")),
                "carrier_region_ids": _string_list(item.get("carrier_region_ids")),
                "importance": importance,
                "reason": str(item.get("reason", "")),
            }
        )
    raw_ambiguity = row.get("ambiguity")
    ambiguity = raw_ambiguity if isinstance(raw_ambiguity, dict) else {}
    raw_risk = row.get("risk")
    risk = raw_risk if isinstance(raw_risk, dict) else {}
    semantic_confidence = _bounded_confidence(row.get("semantic_confidence"))
    action = str(body.get("assembly_action", "unknown"))
    if action not in {"insert", "mate", "slide", "nest", "fasten", "unknown"}:
        action = "unknown"
    suggested = str(row.get("suggested_action", "review"))
    if suggested not in {"prioritize_regions", "review", "unresolved"}:
        suggested = "review"
    return {
        "carrier": _normalize_carrier(row.get("carrier")),
        "part": {
            "role": str(part.get("role", "unknown")),
            "possible_names": _string_list(part.get("possible_names")),
            "functional_description": str(part.get("functional_description", "")),
            "evidence": _string_list(part.get("evidence")),
            "confidence": _bounded_confidence(part.get("confidence")),
        },
        "assembly_hypothesis": {
            "assembly_family": str(body.get("assembly_family", "unknown")),
            "relation": str(body.get("relation", "unknown")),
            "assembly_action": action,
            "target_region_type": str(body.get("target_region_type", "unknown")),
            "preferred_region_ids": preferred[:3],
            "forbidden_region_ids": forbidden,
        },
        "orientation_constraints": {
            "external_face_ids": _string_list(orientation.get("external_face_ids")),
            "internal_face_ids": _string_list(orientation.get("internal_face_ids")),
            "mounting_face_ids": _string_list(orientation.get("mounting_face_ids")),
            "insertion_axis_relative_to_part": str(
                orientation.get("insertion_axis_relative_to_part", "unknown")
            ),
            "service_face_must_remain_visible": bool(
                orientation.get("service_face_must_remain_visible", True)
            ),
            "mirror_transform_allowed": bool(
                orientation.get("mirror_transform_allowed", False)
            ),
            "reasons": _string_list(orientation.get("reasons")),
        },
        "required_geometry_evidence": evidence_rows,
        "ambiguity": {
            "has_multiple_valid_regions": bool(
                ambiguity.get("has_multiple_valid_regions", False)
            ),
            "equivalent_region_ids": _string_list(
                ambiguity.get("equivalent_region_ids")
            ),
            "cannot_be_resolved_from_images": bool(
                ambiguity.get("cannot_be_resolved_from_images", True)
            ),
            "reason": str(ambiguity.get("reason", "")),
        },
        "risk": {
            "possible_visual_misclassification": bool(
                risk.get("possible_visual_misclassification", True)
            ),
            "possible_hidden_interface": bool(
                risk.get("possible_hidden_interface", True)
            ),
            "possible_scale_ambiguity": bool(
                risk.get("possible_scale_ambiguity", True)
            ),
            "possible_symmetry": bool(risk.get("possible_symmetry", True)),
            "notes": _string_list(risk.get("notes")),
        },
        "semantic_confidence": semantic_confidence,
        "review_required": bool(
            row.get("review_required", semantic_confidence < 0.8)
        ),
        "suggested_action": suggested,
    }


def _model_validator(
    model_type: type[BaseModel],
    normalizer: Callable[[Any], Any] | None = None,
):
    def validate(value: Any) -> dict[str, Any]:
        if normalizer is not None:
            value = normalizer(value)
        return model_type.model_validate(value).model_dump(mode="json")

    return validate


def _fallback_part(part_id: str) -> dict[str, Any]:
    return PartRoleAnalysis(
        part_id=part_id,
        part_role="unknown",
        possible_names=[],
        assembly_family_candidates=[],
        functional_description="Insufficient visual-semantic evidence.",
        functional_faces=[],
        likely_assembly_actions=["unknown"],
        principal_axis_semantics="unknown",
        symmetry=SymmetryAssessment(
            has_symmetry=False,
            symmetry_type="unknown",
            orientation_ambiguity=True,
        ),
        risks=["visual_semantic_abstention"],
        confidence=0.0,
        review_required=True,
    ).model_dump(mode="json")


def _fallback_regions() -> dict[str, Any]:
    return CarrierRegionAnalysis(
        carrier=CarrierAssessment(role="unknown", evidence=[], confidence=0.0),
        region_assessments=[],
        preferred_region_ids=[],
        forbidden_region_ids=[],
        equivalent_region_groups=[],
        confidence=0.0,
        review_required=True,
    ).model_dump(mode="json")


def _fallback_hypothesis(part: dict[str, Any]) -> dict[str, Any]:
    return AssemblyHypothesis(
        carrier=CarrierAssessment(role="unknown", evidence=[], confidence=0.0),
        part=PartAssessment(
            role=str(part.get("part_role", "unknown")),
            possible_names=list(part.get("possible_names") or []),
            functional_description=str(part.get("functional_description", "")),
            evidence=[],
            confidence=float(part.get("confidence", 0.0)),
        ),
        assembly_hypothesis=AssemblyHypothesisBody(
            assembly_family="unknown",
            relation="unknown",
            assembly_action="unknown",
            target_region_type="unknown",
            preferred_region_ids=[],
            forbidden_region_ids=[],
        ),
        orientation_constraints=OrientationConstraints(
            external_face_ids=[],
            internal_face_ids=[],
            mounting_face_ids=[],
            insertion_axis_relative_to_part="unknown",
            service_face_must_remain_visible=True,
            mirror_transform_allowed=False,
            reasons=["visual_semantic_abstention"],
        ),
        required_geometry_evidence=[],
        ambiguity=AmbiguityAssessment(
            has_multiple_valid_regions=False,
            equivalent_region_ids=[],
            cannot_be_resolved_from_images=True,
            reason="Visual-semantic stage abstained.",
        ),
        risk=RiskAssessment(
            possible_visual_misclassification=True,
            possible_hidden_interface=True,
            possible_scale_ambiguity=True,
            possible_symmetry=True,
            notes=["No semantic output may change acceptance."],
        ),
        semantic_confidence=0.0,
        review_required=True,
        suggested_action="unresolved",
    ).model_dump(mode="json")


class VisualSemanticPipeline:
    """Orchestrate three isolated Qwen-VL stages."""

    def __init__(self, reviewer: QwenVLReviewer):
        self.reviewer = reviewer

    def analyze_part(
        self,
        part_id: str,
        image_paths: list[str | Path],
        brep_summary: dict[str, Any],
        *,
        source_filename: str = "",
        mode: Literal["live", "cache_only", "off"] = "live",
    ) -> dict[str, Any]:
        context = {
            "task": "part_role_and_functional_face_analysis",
            "part_id": part_id,
            "source_filename_weak_evidence_only": source_filename,
            "available_face_ids": list(brep_summary.get("functional_face_ids") or []),
            "brep_summary": brep_summary,
            "required_output_json_schema": PartRoleAnalysis.model_json_schema(),
        }
        return self.reviewer.structured_review(
            f"prompt1:{part_id}",
            image_paths,
            json.dumps(context, ensure_ascii=False, indent=2),
            system_prompt=PART_ROLE_SYSTEM_PROMPT,
            prompt_version=PROMPT1_VERSION,
            validate_output=_model_validator(
                PartRoleAnalysis, _normalize_part_output
            ),
            fallback_output=_fallback_part(part_id),
            mode=mode,
        )

    def analyze_regions(
        self,
        part_id: str,
        image_paths: list[str | Path],
        part_analysis: dict[str, Any],
        region_summaries: list[dict[str, Any]],
        carrier_summary: dict[str, Any],
        *,
        mode: Literal["live", "cache_only", "off"] = "live",
    ) -> dict[str, Any]:
        context = {
            "task": "carrier_region_semantic_analysis",
            "part_id": part_id,
            "part_role_analysis": part_analysis,
            "carrier_brep_summary": carrier_summary,
            "numbered_regions": region_summaries,
            "instruction": "Assess only the listed region IDs; choose at most three.",
            "required_output_json_schema": CarrierRegionAnalysis.model_json_schema(),
        }
        return self.reviewer.structured_review(
            f"prompt2:{part_id}",
            image_paths,
            json.dumps(context, ensure_ascii=False, indent=2),
            system_prompt=CARRIER_REGION_SYSTEM_PROMPT,
            prompt_version=PROMPT2_VERSION,
            validate_output=_model_validator(
                CarrierRegionAnalysis, _normalize_regions_output
            ),
            fallback_output=_fallback_regions(),
            mode=mode,
        )

    def synthesize(
        self,
        part_id: str,
        image_paths: list[str | Path],
        part_analysis: dict[str, Any],
        region_analysis: dict[str, Any],
        geometry_summary: dict[str, Any],
        *,
        mode: Literal["live", "cache_only", "off"] = "live",
    ) -> dict[str, Any]:
        context = {
            "task": "assembly_hypothesis_synthesis",
            "part_id": part_id,
            "prompt1_part_analysis": part_analysis,
            "prompt2_region_analysis": region_analysis,
            "geometry_summary_without_pose_matrix": geometry_summary,
            "safety": {
                "may_compute_pose": False,
                "may_auto_accept": False,
                "geometry_and_occt_still_required": True,
            },
            "required_output_json_schema": AssemblyHypothesis.model_json_schema(),
        }
        return self.reviewer.structured_review(
            f"prompt3:{part_id}",
            image_paths,
            json.dumps(context, ensure_ascii=False, indent=2),
            system_prompt=ASSEMBLY_SYNTHESIS_SYSTEM_PROMPT,
            prompt_version=PROMPT3_VERSION,
            validate_output=_model_validator(
                AssemblyHypothesis, _normalize_hypothesis_output
            ),
            fallback_output=_fallback_hypothesis(part_analysis),
            mode=mode,
        )


def fuse_candidate_quotas(
    candidates: list[dict[str, Any]],
    region_analysis: dict[str, Any],
    *,
    total_k: int = 20,
    geometry_quota: int = 8,
    semantic_quota: int = 8,
    protected_quota: int = 4,
) -> list[dict[str, Any]]:
    """Protected-union scheduling; semantic evidence cannot erase geometry."""
    if total_k <= 0:
        return []
    assessments = {
        row["region_id"]: row
        for row in region_analysis.get("region_assessments", [])
        if row.get("region_id")
    }
    preferred = list(region_analysis.get("preferred_region_ids") or [])
    preferred_rank = {region_id: index for index, region_id in enumerate(preferred)}
    forbidden = {
        row.get("region_id")
        for row in region_analysis.get("forbidden_region_ids", [])
        if row.get("region_id")
    }

    annotated: list[dict[str, Any]] = []
    for original in candidates:
        row = dict(original)
        rid = str(row.get("region_id", ""))
        semantic = float(assessments.get(rid, {}).get("semantic_score", 0.0))
        sources = list(dict.fromkeys(row.get("candidate_sources") or ["analytic"]))
        if rid in preferred and "vision_semantic" not in sources:
            sources.append("vision_semantic")
        if row.get("protected") and "protected" not in sources:
            sources.append("protected")
        row.update(
            {
                "candidate_sources": sources,
                "semantic_region_score": semantic,
                "semantic_preferred_rank": preferred_rank.get(rid),
                "semantic_forbidden": rid in forbidden,
                "can_auto_accept_from_semantics": False,
            }
        )
        annotated.append(row)

    geometry = sorted(
        annotated,
        key=lambda row: float(row.get("geometry_score", row.get("score", 0.0))),
        reverse=True,
    )
    semantic = sorted(
        [row for row in annotated if row.get("region_id") in preferred],
        key=lambda row: (
            row.get("semantic_preferred_rank", 999),
            -float(row.get("semantic_region_score", 0.0)),
        ),
    )
    protected = [
        row for row in geometry
        if row.get("protected") or "protected" in row.get("candidate_sources", [])
    ]
    exploration = [row for row in geometry if row not in semantic and row not in protected]

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(rows: list[dict[str, Any]], quota: int) -> None:
        count = 0
        for row in rows:
            key = str(row.get("candidate_id") or row.get("region_id"))
            if not key or key in seen:
                continue
            selected.append(row)
            seen.add(key)
            count += 1
            if count >= quota or len(selected) >= total_k:
                break

    add(geometry, geometry_quota)
    add(semantic, semantic_quota)
    add(protected + exploration, protected_quota)
    add(geometry + semantic + protected + exploration, total_k)
    return selected[:total_k]


def write_stage_output(path: str | Path, record: dict[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return destination
