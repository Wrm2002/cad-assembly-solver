"""Review only ambiguous group proposals and produce bounded utility overrides."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from contracts import GroupProposal
from global_grouping import assign_groups, evaluate_groups
from multimodal_reviewer import QwenVLReviewer
from render_parts_tray import render_parts_tray
from semantic_review import DeepSeekReviewer


EVALUATION_ONLY_SEMANTIC_SOURCES = {
    "function_grounded_metadata",
    "locked_holdout_metadata_evaluation_only",
}


def _resolve_step_path(
    pool: Path,
    part_id: str,
    info: dict[str, Any],
) -> Path | None:
    """Resolve the actual STEP source without creating `.step.step` paths."""
    candidates = [
        info.get("filepath"),
        info.get("source_file"),
        pool / "parts" / part_id,
    ]
    if not str(part_id).lower().endswith((".step", ".stp")):
        candidates.append(pool / "parts" / f"{part_id}.step")
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.is_file():
            return path.resolve()
    return None


def _production_semantic_hint(
    info: dict[str, Any],
    *,
    allow_evaluation_only_semantics: bool,
) -> dict[str, Any]:
    """Return only production-available semantic fields by default."""
    functional = info.get("functional_semantics", {}) or {}
    source = str(functional.get("source", "unknown"))
    evaluation_only = (
        source in EVALUATION_ONLY_SEMANTIC_SOURCES
        or functional.get(
            "source_template_disclosed_for_evaluation_only"
        )
        is False
    )
    if evaluation_only and not allow_evaluation_only_semantics:
        return {
            "available": False,
            "source": source,
            "excluded_reason": "evaluation_truth_not_available_in_production",
        }
    fields = {
        key: functional.get(key)
        for key in (
            "part_name",
            "part_role",
            "interface_types",
            "functions",
        )
        if functional.get(key)
    }
    return {
        "available": bool(fields),
        "source": source,
        "fields": fields,
        "excluded_reason": None,
    }


def _safe_part_summary(part: dict[str, Any]) -> dict[str, Any]:
    cylinders = part.get("cylindrical_faces", [])
    planes = part.get("planar_faces", [])
    functional = part.get("functional_semantics", {}) or {}
    summary: dict[str, Any] = {
        "part_id": part["part_id"],
        "bbox_size_mm": [
            round(float(value), 4) for value in part["bbox"]["size"]
        ],
        "volume_mm3": round(float(part["volume"]), 4),
        "geometric_class": part["geometric_class"],
        "cylinder_count": len(cylinders),
        "cylinder_radii_mm": sorted(
            {
                round(float(item["parameters"].get("radius", 0.0)), 4)
                for item in cylinders
            }
        )[:12],
        "planar_face_count": len(planes),
        "largest_planar_areas_mm2": sorted(
            (
                round(float(item["parameters"].get("area", 0.0)), 3)
                for item in planes
            ),
            reverse=True,
        )[:6],
        "hole_candidate_count": len(part.get("holes", [])),
        "hole_pattern_candidate_count": len(part.get("hole_patterns", [])),
    }
    # Enrich with functional semantics when available (D0 functional dataset)
    if functional.get("part_role"):
        summary["part_role"] = functional["part_role"]
    if functional.get("interface_types"):
        summary["interface_types"] = functional["interface_types"]
    if functional.get("part_name"):
        summary["part_name"] = functional["part_name"]
    if functional.get("functions"):
        summary["functions"] = functional["functions"]
    return summary


def build_summary(
    proposal: GroupProposal,
    part_map: dict[str, dict[str, Any]],
    edge_map: dict[str, dict[str, Any]],
    pool_part_count: int,
    *,
    pool_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    edges = []
    functional_relations: list[str] = []
    for edge_id in proposal.candidate_edges:
        edge = edge_map.get(edge_id)
        if not edge:
            continue
        edges.append(
            {
                "candidate_id": edge_id,
                "parts": edge["parts"],
                "type": edge["candidate_type"],
                "geometry_score": edge["geometry_score"],
                "confidence": edge["confidence"],
                "numeric_evidence": {
                    key: value
                    for key, value in edge.get("audit_reason", {}).items()
                    if isinstance(value, (int, float, bool))
                },
            }
        )
        # Collect functional relations when available from enriched edges
        func_rel = edge.get("functional_relation")
        if func_rel and isinstance(func_rel, str):
            functional_relations.append(func_rel)
    summary: dict[str, Any] = {
        "schema_version": "2.0.0",
        "proposal_id": proposal.group_id,
        "task": "judge whether these parts plausibly form one assembly group",
        "pool_part_count": pool_part_count,
        "proposal_part_count": len(proposal.parts),
        "geometry_group_score": proposal.geometry_score,
        "parts": [
            _safe_part_summary(part_map[part])
            for part in proposal.parts
        ],
        "candidate_edges": edges,
        "hard_geometry_status": "candidate_only_not_pose_validated",
        "constraints": {
            "semantic_review_cannot_override_geometry_failure": True,
            "filenames_and_provenance_are_anonymous": True,
        },
    }
    # Include assembly-family context when pool metadata is available
    if pool_metadata:
        if pool_metadata.get("assembly_family"):
            summary["assembly_family"] = pool_metadata["assembly_family"]
    # Include functional relations collected from edges
    if functional_relations:
        summary["functional_relations"] = functional_relations
    return summary


def ambiguous_proposals(
    part_ids: list[str],
    proposals: list[GroupProposal],
    grouping_config: dict[str, Any],
    semantic_config: dict[str, Any],
) -> list[GroupProposal]:
    baseline = assign_groups(part_ids, proposals, grouping_config)
    selected_ids = {
        item["group_id"]
        for item in baseline["selected_groups"]
        if len(item["parts"]) > 1
    }
    selected = [
        proposal for proposal in proposals
        if proposal.group_id in selected_ids
    ]
    alternatives = {}
    for chosen in selected:
        chosen_parts = set(chosen.parts)
        for alternative in proposals:
            if alternative.group_id == chosen.group_id:
                continue
            if not chosen_parts.intersection(alternative.parts):
                continue
            alternatives[alternative.group_id] = alternative
    ranked_selected = sorted(
        selected,
        key=lambda item: (
            item.geometry_score,
            len(item.parts),
            item.group_id,
        ),
        reverse=True,
    )
    ranked_alternatives = sorted(
        (
            proposal for group_id, proposal in alternatives.items()
            if group_id not in selected_ids
        ),
        key=lambda item: (
            item.geometry_score,
            len(item.parts),
            item.group_id,
        ),
        reverse=True,
    )
    maximum = int(semantic_config["maximum_reviews_per_pool"])
    return (ranked_selected + ranked_alternatives)[:maximum]


def review_pool(
    pool_dir: str | Path,
    config_path: str | Path,
    *,
    mode: str = "live",
) -> dict[str, Any]:
    pool = Path(pool_dir).resolve()
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    part_features = json.loads(
        (pool / "index" / "part_features.json").read_text(encoding="utf-8")
    )
    part_map = {item["part_id"]: item for item in part_features}
    edge_items = json.loads(
        (pool / "index" / "pruned_candidates.json").read_text(
            encoding="utf-8"
        )
    )
    edge_map = {item["candidate_id"]: item for item in edge_items}
    proposals = [
        GroupProposal.model_validate(item)
        for item in json.loads(
            (pool / "grouping" / "group_proposals.json").read_text(
                encoding="utf-8"
            )
        )
    ]
    review_candidates = ambiguous_proposals(
        sorted(part_map),
        proposals,
        config["global_grouping"],
        config["semantic_review"],
    )
    semantic_dir = pool / "semantic"
    reviewer = DeepSeekReviewer(
        config["semantic_review"], semantic_dir / "cache"
    )
    reviews = []
    hypothetical_overrides = {}
    margin = float(config["semantic_review"]["ambiguity_score_margin"])
    minimum = float(config["global_grouping"]["minimum_group_score"])
    for proposal in review_candidates:
        summary = build_summary(
            proposal, part_map, edge_map, len(part_map)
        )
        record = reviewer.review(summary, mode=mode)
        decision = record["decision"]
        effective = 0.5 + (
            float(decision["plausibility_score"]) - 0.5
        ) * float(decision["confidence"])
        base_utility = (
            proposal.geometry_score - minimum
        ) * len(proposal.parts)
        # A semantic decision can perturb utility by at most half the
        # configured near-tie band. It cannot overturn a clear geometry gap.
        delta = margin * (effective - 0.5) * len(proposal.parts)
        hypothetical_overrides[proposal.group_id] = base_utility + delta
        reviews.append(
            {
                **record,
                "base_utility": base_utility,
                "bounded_semantic_delta": delta,
                "effective_plausibility": effective,
            }
        )
    application_mode = config["semantic_review"].get(
        "application_mode", "explanation_only"
    )
    overrides = (
        hypothetical_overrides if application_mode == "rerank" else {}
    )
    semantic_assignment = assign_groups(
        sorted(part_map),
        proposals,
        config["global_grouping"],
        overrides,
    )
    gt_path = pool / "pool_gt.json"
    metrics = None
    if gt_path.is_file():
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        metrics = evaluate_groups(
            semantic_assignment["selected_groups"], gt
        )
    report = {
        "schema_version": "1.0.0",
        "pool_id": pool.name,
        "mode": mode,
        "review_count": len(reviews),
        "reviews": reviews,
        "utility_overrides": overrides,
        "hypothetical_utility_overrides": hypothetical_overrides,
        "application_mode": application_mode,
        "explanation_only": application_mode != "rerank",
        "semantic_assignment": semantic_assignment,
        "metrics_before_pose_validation": metrics,
    }
    semantic_dir.mkdir(parents=True, exist_ok=True)
    (semantic_dir / "semantic_review_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def review_pool_multimodal(
    pool_dir: str | Path,
    config_path: str | Path,
    *,
    mode: str = "live",
    render_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Multimodal review using Qwen-VL with rendered parts-tray images.

    For each review candidate:
    1. Loads STEP files and renders a parts-tray PNG.
    2. Builds structured text context with part_role, interface_types if available.
    3. Sends image + text to Qwen-VL for functional validity judgment.
    """
    pool = Path(pool_dir).resolve()
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    part_features = json.loads(
        (pool / "index" / "part_features.json").read_text(encoding="utf-8")
    )
    part_map = {item["part_id"]: item for item in part_features}
    edge_items = json.loads(
        (pool / "index" / "pruned_candidates.json").read_text(encoding="utf-8")
    )
    edge_map = {item["candidate_id"]: item for item in edge_items}
    proposals = [
        GroupProposal.model_validate(item)
        for item in json.loads(
            (pool / "grouping" / "group_proposals.json").read_text(encoding="utf-8")
        )
    ]
    review_candidates = ambiguous_proposals(
        sorted(part_map),
        proposals,
        config["global_grouping"],
        config["semantic_review"],
    )

    semantic_dir = pool / "semantic_multimodal"
    render_root = Path(render_dir) if render_dir else (semantic_dir / "renders")
    render_root.mkdir(parents=True, exist_ok=True)

    reviewer = QwenVLReviewer(
        config.get("multimodal_review", {}), semantic_dir / "cache"
    )
    multimodal_config = config.get("multimodal_review", {})
    allow_evaluation_semantics = bool(
        multimodal_config.get(
            "allow_evaluation_only_semantics", False
        )
    )

    reviews = []
    for proposal in review_candidates:
        # Collect STEP paths and labels
        step_paths = []
        labels = []
        semantic_hints: list[str] = []
        excluded_semantic_sources: list[str] = []
        evaluation_semantic_sources_used: list[str] = []
        missing_part_ids: list[str] = []
        for part_id in proposal.parts:
            info = part_map.get(part_id, {})
            step_path = _resolve_step_path(pool, part_id, info)
            if step_path is None:
                missing_part_ids.append(part_id)
                continue
            step_paths.append(str(step_path))
            semantic = _production_semantic_hint(
                info,
                allow_evaluation_only_semantics=allow_evaluation_semantics,
            )
            labels.append(part_id[:8])
            if semantic["available"]:
                fields = semantic["fields"]
                if semantic.get("source") in EVALUATION_ONLY_SEMANTIC_SOURCES:
                    evaluation_semantic_sources_used.append(
                        str(semantic["source"])
                    )
                semantic_hints.append(
                    f"  {part_id[:8]}: "
                    f"role={fields.get('part_role', 'unknown')}, "
                    f"interfaces={fields.get('interface_types', [])}, "
                    f"name={fields.get('part_name', 'unknown')}"
                )
            elif semantic.get("source"):
                excluded_semantic_sources.append(
                    str(semantic["source"])
                )

        if missing_part_ids or len(step_paths) != len(proposal.parts):
            reviews.append({
                "proposal_id": proposal.group_id,
                "decision": {
                    "verdict": "abstain",
                    "reason_codes": ["missing_step_files"],
                    "explanation": (
                        "Cannot render the complete candidate; STEP files "
                        f"not located for {sorted(missing_part_ids)}."
                    ),
                },
            })
            continue

        # Render parts tray
        tray_path = render_root / f"{proposal.group_id}_tray.png"
        try:
            render_parts_tray(step_paths, labels, tray_path)
        except Exception as exc:
            reviews.append({
                "proposal_id": proposal.group_id,
                "decision": {
                    "verdict": "abstain",
                    "reason_codes": ["render_failed"],
                    "explanation": f"Parts tray rendering failed: {exc}",
                },
            })
            continue

        # Build text context with structured semantic data
        text_lines = [
            f"Proposal: {proposal.group_id}",
            f"Part count: {len(proposal.parts)}",
            f"Geometry score: {proposal.geometry_score:.4f}",
            "",
            "Parts:",
        ]
        if semantic_hints:
            text_lines.extend(semantic_hints)
        else:
            for label in labels:
                text_lines.append(f"  {label}")

        # Include edge evidence summary
        text_lines.append("")
        text_lines.append("Evidence:")
        for edge_id in proposal.candidate_edges:
            edge = edge_map.get(edge_id, {})
            if edge:
                etype = edge.get("candidate_type", "unknown")
                escore = edge.get("geometry_score", 0.0)
                text_lines.append(f"  {edge_id[:12]}: type={etype}, score={escore:.4f}")

        text_context = "\n".join(text_lines)

        record = reviewer.review(
            proposal.group_id,
            [str(tray_path)],
            text_context,
            mode=mode,
        )
        record["tray_image"] = str(tray_path)
        record["semantic_input_policy"] = {
            "allow_evaluation_only_semantics": (
                allow_evaluation_semantics
            ),
            "excluded_semantic_sources": sorted(
                set(excluded_semantic_sources)
            ),
            "synthetic_truth_fields_used": (
                bool(evaluation_semantic_sources_used)
            ),
            "evaluation_semantic_sources_used": sorted(
                set(evaluation_semantic_sources_used)
            ),
        }
        reviews.append(record)

    report = {
        "schema_version": "2.0.0",
        "pool_id": pool.name,
        "mode": mode,
        "reviewer": "qwen-vl",
        "review_count": len(reviews),
        "reviews": reviews,
        "application_mode": "explanation_only",
        "calibration_gate_passed": False,
        "allow_evaluation_only_semantics": allow_evaluation_semantics,
        "semantic_outputs_affect_grouping": False,
    }
    semantic_dir.mkdir(parents=True, exist_ok=True)
    (semantic_dir / "multimodal_review_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pool_dir")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "configs" / "pool_pipeline.json"),
    )
    parser.add_argument(
        "--mode",
        choices=("live", "cache_only", "off"),
        default="live",
    )
    parser.add_argument(
        "--reviewer",
        choices=("text", "multimodal"),
        default="text",
    )
    args = parser.parse_args()
    report = (
        review_pool_multimodal(
            args.pool_dir, args.config, mode=args.mode
        )
        if args.reviewer == "multimodal"
        else review_pool(args.pool_dir, args.config, mode=args.mode)
    )
    print(
        json.dumps(
            {
                "pool_id": report["pool_id"],
                "review_count": report["review_count"],
                "metrics": report.get("metrics_before_pose_validation"),
                "reviewer": report.get("reviewer", "deepseek-text"),
                "application_mode": report.get("application_mode"),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
