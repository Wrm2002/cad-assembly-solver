"""Prepare semantic review inputs without allowing an LLM to alter grouping.

The current public mixed-pool data is intentionally anonymized and has no
functional role labels.  This script records that absence, produces structured
inputs, and enforces the existing failed calibration gate.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = Path(
    r"D:\Model_match_public_data\fusion360_mixed_pools_real_v1_20260705"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "public_cad_dataset_audit"
    / "outputs"
    / "phase8_semantic_gate"
)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _descriptor_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "available": False,
            "failure_reason": "geometry_descriptor_missing",
        }
    descriptor = _load(path)
    entities = descriptor.get("entities", [])
    types = Counter(
        f"{row.get('entity_type')}:{row.get('geometry_type')}"
        for row in entities
    )
    radii = sorted(
        {
            round(float(row["radius"]), 6)
            for row in entities
            if row.get("radius") is not None
        }
    )
    return {
        "available": True,
        "topology_counts": descriptor.get("topology_counts"),
        "retained_entity_count": descriptor.get("retained_entity_count"),
        "analytic_entity_types": dict(types),
        "observed_radii_mm": radii[:24],
        "summary_limitations": [
            "analytic geometry does not identify functional part role",
            "entity reservoir is bounded and is not complete B-Rep semantics",
        ],
    }


def _calibration_gate() -> dict[str, Any]:
    prior = _load(
        PROJECT_ROOT
        / "sw"
        / "data"
        / "results"
        / "semantic_calibration_report.json"
    )
    reasons = list(prior.get("gate_failure_reasons", []))
    reasons.extend(
        [
            "current Fusion mixed pools lack human functional semantic labels",
            "part_role, assembly_family, and functional_relation are unavailable",
            "no real holdout demonstrates unchanged auto-accept precision",
        ]
    )
    return {
        "schema_version": "1.0.0",
        "semantic_reranking_enabled": False,
        "semantic_application_mode": "explanation_only",
        "prior_calibration": {
            "study": prior.get("study"),
            "review_count": prior.get("review_count"),
            "semantic_auc": prior.get(
                "semantic_auc_against_source_truth"
            ),
            "semantic_brier_score": prior.get("semantic_brier_score"),
            "geometry_brier_score": prior.get("geometry_brier_score"),
            "human_labels_available": prior.get("human_labels_available"),
        },
        "current_holdout": {
            "human_functional_labels_available": False,
            "auto_accept_precision_not_decreased": False,
            "false_positive_count_not_increased": False,
        },
        "gate_rules": prior.get("gate_rules"),
        "gate_failure_reasons": sorted(set(reasons)),
        "provider_configuration": {
            "deepseek_api_configured": bool(
                os.environ.get("DEEPSEEK_API_KEY", "").strip()
            ),
            "qwen_api_configured": bool(
                os.environ.get("QWEN_API_KEY", "").strip()
                or os.environ.get("DASHSCOPE_API_KEY", "").strip()
            ),
            "provider_called": False,
            "reason_not_called": (
                "Missing functional semantic fields and failed calibration; "
                "additional prompt calls cannot establish validity."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT
    )
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT
    )
    args = parser.parse_args()

    queue = _load(
        PROJECT_ROOT
        / "public_cad_dataset_audit"
        / "outputs"
        / "phase7_pose_validation"
        / "pose_validation_queue.json"
    )["jobs"]
    pose_audit_path = (
        PROJECT_ROOT
        / "public_cad_dataset_audit"
        / "outputs"
        / "phase7_pose_validation"
        / "pose_validation_full_audit.json"
    )
    pose_records = (
        _load(pose_audit_path).get("records", [])
        if pose_audit_path.is_file()
        else []
    )
    pose_by_job = {
        row["pose_job_id"]: row
        for row in pose_records
        if row.get("pose_job_id")
    }
    inputs = []
    for job in queue:
        if "blind_production" not in job["queue_modes"]:
            continue
        pool = args.dataset_root / job["pool_id"]
        pool_input = _load(pool / "pool_input.json")
        part_meta = {
            row["part_id"]: row for row in pool_input["parts"]
        }
        candidate = job.get("candidate") or {}
        geometry_evidence = candidate.get("geometry_evidence") or {}
        parts = []
        for part_id in job["parts"]:
            source = part_meta[part_id]
            parts.append(
                {
                    "part_id": part_id,
                    "file_name": f"{part_id}.step",
                    "part_name": None,
                    "part_role": None,
                    "material": None,
                    "color": None,
                    "bom_hint": None,
                    "cad_metadata": {
                        "geometry_format": source["geometry_format"],
                        "geometry_bytes": source["geometry_bytes"],
                        "geometry_signature": source[
                            "geometry_signature"
                        ],
                    },
                    "geometry_summary": _descriptor_summary(
                        pool
                        / "geometry_descriptors"
                        / f"{part_id}.descriptors.json"
                    ),
                    "unavailable_fields": [
                        "source_part_name",
                        "part_role",
                        "material",
                        "color",
                        "BOM_hint",
                    ],
                }
            )
        pose = pose_by_job.get(job["job_id"], {})
        inputs.append(
            {
                "schema_version": "1.0.0",
                "proposal_id": job["source_candidate_id"],
                "pool_id": job["pool_id"],
                "parts": parts,
                "assembly_family": None,
                "functional_relation": None,
                "interface_types": sorted(
                    geometry_evidence.get(
                        "interface_family_counts", {}
                    ).keys()
                ),
                "rendered_part_images": [],
                "rendered_assembly_image": None,
                "geometry_evidence": {
                    "candidate_priority_score": candidate.get(
                        "candidate_priority_score"
                    ),
                    "geometry_score": candidate.get("geometry_score"),
                    "independent_evidence_count": geometry_evidence.get(
                        "independent_evidence_count", 0
                    ),
                    "weak_single_interface_match": (
                        candidate.get("consistency") or {}
                    ).get("weak_single_interface_match"),
                    "group_consistency_score": (
                        candidate.get("consistency") or {}
                    ).get("group_consistency_score"),
                },
                "physical_validation": {
                    "pose_status": pose.get(
                        "final_pose_status", "not_completed"
                    ),
                    "collision_result": pose.get(
                        "collision_result", "not_completed"
                    ),
                    "occt_common_volume": pose.get("occt_common_volume"),
                },
                "constraints": {
                    "source_identity_hidden": True,
                    "ground_truth_not_in_input": True,
                    "semantic_output_explanation_only": True,
                    "semantic_output_cannot_change_score": True,
                    "semantic_output_cannot_change_tier": True,
                },
                "failure_reasons": [
                    "functional semantic fields unavailable"
                ],
                "unavailable_fields": [
                    "assembly_family",
                    "functional_relation",
                    "part_roles",
                    "BOM_hint",
                    "rendered_part_images",
                    "rendered_assembly_image",
                    "human_functional_label",
                ],
            }
        )

    gate = _calibration_gate()
    reviews = [
        {
            "schema_version": "1.0.0",
            "proposal_id": row["proposal_id"],
            "semantic_validity": "unknown",
            "semantic_score": 0.5,
            "functional_reason": (
                "Functional roles, assembly family, and human semantic labels "
                "are unavailable; abstention is mandatory."
            ),
            "possible_system": "unknown",
            "risk": (
                "Geometry-valid but provenance- or functionally-wrong grouping "
                "cannot be excluded."
            ),
            "is_geometrically_feasible_but_semantically_invalid": None,
            "review_required": True,
            "suggested_action": "abstain",
            "provider": "deterministic_calibration_gate",
            "provider_called": False,
            "application_mode": "explanation_only",
            "affects_final_score": False,
            "affects_grouping": False,
            "failure_reasons": gate["gate_failure_reasons"],
            "unavailable_fields": [
                "calibrated_semantic_probability",
                "functional_validity_verdict",
            ],
        }
        for row in inputs
    ]
    _write(args.output_root / "semantic_inputs.json", inputs)
    _write(args.output_root / "semantic_reviews.json", reviews)
    _write(
        args.output_root / "semantic_calibration_report.json", gate
    )
    decision = f"""# Semantic gate decision

**Decision: disabled for grouping and reranking.**

- Application mode: explanation-only
- Inputs prepared: {len(inputs)}
- LLM/VLM calls made: 0
- Human functional labels: unavailable
- Part roles / assembly family / functional relation: unavailable
- Prior semantic AUC: {gate['prior_calibration']['semantic_auc']}

The configured DeepSeek/Qwen credentials do not repair missing evidence.
Calling either provider now would generate prose, not calibrated truth.
Every candidate therefore receives an explicit abstention and remains review.
"""
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "semantic_gate_decision.md").write_text(
        decision, encoding="utf-8"
    )
    system_review = f"""# Step 8 independent systemic review

1. The module only prepares evidence and enforces calibration.
2. No source IDs or truth labels enter a production semantic input.
3. No API call was made because role/family/function labels and a real holdout
   are absent.
4. DeepSeek and Qwen cannot change scores, tiers, or final grouping.
5. The missing fields are recorded per input instead of hallucinated.
6. The smallest next improvement is human role/family annotation, not prompt
   expansion or another model.
"""
    (args.output_root / "semantic_gate_system_review.md").write_text(
        system_review, encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "semantic_input_count": len(inputs),
                "semantic_review_count": len(reviews),
                "semantic_reranking_enabled": False,
                "provider_called": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
