"""Index a STEP pool, prescreen pairs, and run auditable detailed matching."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from itertools import combinations
from pathlib import Path

from constraints import match_features
from contracts import CandidateEdge, CandidateStatus
from joinable_candidate_provider import load_pair_report, select_joinable_pairs
from match_pruning import prune_match_graph
from match_scoring import score_matches
from part_index import index_part


def _load_config(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _diagonal(part):
    return math.sqrt(sum(value * value for value in part.bbox.size))


def _prescreen(a, b, config):
    reasons = []
    coarse_score = 0.0
    diag_a, diag_b = _diagonal(a), _diagonal(b)
    ratio = max(diag_a, diag_b) / max(min(diag_a, diag_b), 1e-9)
    if ratio <= config["maximum_bbox_diagonal_ratio"]:
        reasons.append({"kind": "bbox_scale", "ratio": ratio})
        coarse_score += 0.05 / max(ratio, 1.0)

    relative = config["cylinder_radius_relative_tolerance"]
    absolute = config["cylinder_radius_absolute_tolerance_mm"]
    cylinder_pairs = []
    for first in a.cylindrical_faces:
        for second in b.cylindrical_faces:
            ra = float(first.parameters.get("radius", 0.0))
            rb = float(second.parameters.get("radius", 0.0))
            difference = abs(ra - rb)
            tolerance = max(absolute, relative * max(ra, rb))
            if ra > 0 and rb > 0 and difference <= tolerance:
                cylinder_pairs.append({
                    "feature_a": first.feature_id,
                    "feature_b": second.feature_id,
                    "radius_difference": difference,
                    "tolerance": tolerance,
                })
    if cylinder_pairs:
        reasons.append({"kind": "compatible_cylinders", "matches": cylinder_pairs})
        best = max(
            1.0 - item["radius_difference"] / max(item["tolerance"], 1e-9)
            for item in cylinder_pairs
        )
        coarse_score += 0.60 * best

    planar_pairs = []
    minimum_ratio = config["minimum_planar_area_ratio"]
    for first in a.planar_faces:
        for second in b.planar_faces:
            area_a = float(first.parameters.get("area", 0.0))
            area_b = float(second.parameters.get("area", 0.0))
            area_ratio = min(area_a, area_b) / max(area_a, area_b, 1e-9)
            if area_a > 0 and area_b > 0 and area_ratio >= minimum_ratio:
                planar_pairs.append({
                    "feature_a": first.feature_id,
                    "feature_b": second.feature_id,
                    "area_ratio": area_ratio,
                })
    if planar_pairs:
        reasons.append({
            "kind": "compatible_planar_areas",
            "count": len(planar_pairs),
            "best_area_ratio": max(item["area_ratio"] for item in planar_pairs),
        })
        coarse_score += 0.25 * max(item["area_ratio"] for item in planar_pairs)

    pattern_pairs = []
    for first in a.hole_patterns:
        for second in b.hole_patterns:
            if first.parameters["count"] == second.parameters["count"]:
                pattern_pairs.append({
                    "feature_a": first.feature_id,
                    "feature_b": second.feature_id,
                    "count": first.parameters["count"],
                })
    if pattern_pairs:
        reasons.append({"kind": "compatible_hole_pattern_candidates", "matches": pattern_pairs})
        coarse_score += 0.10

    # Bbox scale is context, not sufficient mate evidence by itself.
    independent_evidence_types = sorted(
        {
            {
                "compatible_cylinders": "radius_compatibility",
                "compatible_planar_areas": "planar_extent_compatibility",
                "compatible_hole_pattern_candidates": "hole_pattern_compatibility",
            }[reason["kind"]]
            for reason in reasons
            if reason["kind"] != "bbox_scale"
        }
    )
    accepted = bool(independent_evidence_types)
    return accepted, {
        "parts": [a.part_id, b.part_id],
        "accepted": accepted,
        "evidence": reasons,
        "rejection_reason": None if accepted else "no configured coarse geometric evidence",
        "bbox_diagonal_ratio": ratio,
        "coarse_score": min(1.0, coarse_score),
        "independent_evidence_types": independent_evidence_types,
        "independent_evidence_count": len(independent_evidence_types),
        "weak_single_interface_match": len(independent_evidence_types) < 2,
        "ranking_policy": (
            "independent_evidence_count_then_coarse_score"
        ),
    }


def _legacy_features(part):
    return {
        "filepath": part.source_file,
        "bbox": {
            "min": part.bbox.minimum,
            "max": part.bbox.maximum,
        },
        "cylinders": [item.parameters for item in part.cylindrical_faces],
        "planes": [item.parameters for item in part.planar_faces],
        "cones": [],
        "torii": [],
        "spheres": [],
    }


def _candidate_id(match):
    payload = json.dumps(
        {
            "parts": sorted(match["parts"]),
            "type": match["type"],
            "a": match.get("feat_a_idx"),
            "b": match.get("feat_b_idx"),
        },
        sort_keys=True,
    )
    return "C_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _contract(match, status, extra_reason=None):
    kind = match["type"]
    collection = "plane" if kind.startswith("planar") else "cylinder"
    refs = [
        f"{match['parts'][0]}:{collection}:{match.get('feat_a_idx', -1)}",
        f"{match['parts'][1]}:{collection}:{match.get('feat_b_idx', -1)}",
    ]
    reason = dict(match.get("reason", {}))
    if extra_reason:
        reason.update(extra_reason)
    evidence = [
        f"candidate_type={kind}",
        f"geometry_score={float(match.get('score', 0.0)):.6f}",
    ]
    return CandidateEdge(
        candidate_id=_candidate_id(match),
        parts=list(match["parts"]),
        candidate_type=kind,
        feature_refs=refs,
        geometry_score=float(match.get("score", 0.0)),
        confidence=str(match.get("confidence", "unknown")),
        geometric_evidence=evidence,
        status=status,
        audit_reason=reason,
    )


def index_pool(
    input_dir,
    output_dir,
    config_path,
    joinable_report_path=None,
):
    input_dir, output_dir = Path(input_dir).resolve(), Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _load_config(config_path)
    prescreen_config = config["prescreen"]
    files = sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".step", ".stp"}
    )
    # The pool-local ID is the complete filename and matches pool_gt.json
    # exactly; downstream modules must never guess or strip extensions.
    semantic_path = input_dir.parent / "part_semantics.json"
    semantic_metadata = (
        json.loads(semantic_path.read_text(encoding="utf-8"))
        if semantic_path.is_file()
        else {}
    )
    parts = {
        path.name: index_part(
            path,
            path.name,
            semantic_metadata.get(path.name, {}),
        )
        for path in files
    }
    audit = []
    for first_id, second_id in combinations(parts, 2):
        accepted, record = _prescreen(
            parts[first_id], parts[second_id], prescreen_config
        )
        audit.append(record)
    maximum_neighbors = int(prescreen_config["maximum_coarse_neighbors_per_part"])
    ranked_by_part = {part_id: [] for part_id in parts}
    for record in audit:
        if not record["accepted"]:
            continue
        for part_id in record["parts"]:
            ranked_by_part[part_id].append(record)
    allowed_pairs = set()
    for records in ranked_by_part.values():
        ranked = sorted(
            records,
            key=lambda item: (
                int(item.get("independent_evidence_count", 0)),
                float(item["coarse_score"]),
                tuple(item["parts"]),
            ),
            reverse=True,
        )[:maximum_neighbors]
        allowed_pairs.update(tuple(sorted(item["parts"])) for item in ranked)

    joinable_config = config.get("joinable_candidate_provider", {})
    joinable_audit = {
        "schema_version": "1.0.0",
        "provider": "pretrained_joinable",
        "pool_id": input_dir.parent.name,
        "enabled": bool(joinable_config.get("enabled", True)),
        "status": "not_configured",
        "selected_pair_count": 0,
        "selected_pairs": [],
    }
    joinable_pairs = set()
    configured_report = joinable_report_path or joinable_config.get(
        "cached_report_path"
    )
    if joinable_audit["enabled"] and configured_report:
        report = load_pair_report(configured_report)
        joinable_pairs, provider_audit = select_joinable_pairs(
            report,
            pool_id=input_dir.parent.name,
            part_ids=sorted(parts),
            maximum_neighbors_per_part=int(
                joinable_config.get(
                    "maximum_neighbors_per_part",
                    maximum_neighbors,
                )
            ),
            minimum_uniform_lift=float(
                joinable_config.get("minimum_uniform_lift", 1.0)
            ),
            audit_top_candidates=int(
                joinable_config.get("audit_top_candidates", 3)
            ),
        )
        joinable_audit = {
            **provider_audit,
            "enabled": True,
            "status": "loaded",
            "source_report": str(Path(configured_report).resolve()),
        }
        # Keep the analytic frontier intact and add a separate bounded learned
        # frontier.  JoinABLe does not add geometric evidence or score here.
        allowed_pairs.update(joinable_pairs)

    accepted_pairs = []
    for record in audit:
        pair = tuple(sorted(record["parts"]))
        record["candidate_providers"] = ["analytic_geometry"]
        if pair in joinable_pairs:
            record["candidate_providers"].append("pretrained_joinable")
            record["joinable_candidate_recall_nomination"] = True
            if not record["accepted"]:
                record["accepted"] = True
                record["rejection_reason"] = None
                record["acceptance_reason"] = (
                    "rescued_by_joinable_for_detailed_analytic_matching"
                )
        if record["accepted"] and pair not in allowed_pairs:
            record["accepted"] = False
            record["rejection_reason"] = (
                f"outside top {maximum_neighbors} coarse neighbors for both parts"
            )
        if record["accepted"]:
            accepted_pairs.append(tuple(record["parts"]))

    raw_matches = []
    legacy = {part_id: _legacy_features(part) for part_id, part in parts.items()}
    generation = config["candidate_generation"]
    for first_id, second_id in accepted_pairs:
        pair_features = {
            first_id: legacy[first_id],
            second_id: legacy[second_id],
        }
        raw_matches.extend(
            match_features(
                pair_features,
                generation.get("detailed_thresholds", {}),
            )
        )
    scored = score_matches(raw_matches, legacy)
    kept, removed = prune_match_graph(
        scored,
        min_score=float(generation["minimum_score"]),
        top_k_pair=int(generation["top_k_per_pair"]),
        max_neighbors=int(generation["maximum_neighbors"]),
    )
    outputs = {
        "part_features.json": [
            part.model_dump(mode="json") for part in parts.values()
        ],
        "screening_audit.json": {
            "config": prescreen_config,
            "total_pairs": len(audit),
            "accepted_pairs": len(accepted_pairs),
            "rejected_pairs": len(audit) - len(accepted_pairs),
            "pairs": audit,
        },
        "joinable_candidate_provider_audit.json": joinable_audit,
        "geometry_candidates.json": [
            _contract(match, CandidateStatus.generated).model_dump(mode="json")
            for match in scored
        ],
        "pruned_candidates.json": [
            _contract(match, CandidateStatus.kept).model_dump(mode="json")
            for match in kept
        ],
        "removed_candidates.json": [
            _contract(
                match,
                CandidateStatus.removed,
                {
                    "removal_reason": match.get("removal_reason"),
                    "removal_detail": match.get("removal_detail"),
                },
            ).model_dump(mode="json")
            for match in removed
        ],
    }
    gt_path = input_dir.parent / "pool_gt.json"
    if gt_path.is_file():
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        truth = set()
        for group in gt.get("true_groups", []):
            for mate in group.get("true_mates", []):
                mate_parts = mate.get("parts") or [
                    mate.get("part_a"), mate.get("part_b")
                ]
                truth.add((tuple(sorted(mate_parts)), mate.get("type")))
        accepted_pair_keys = {
            tuple(sorted(record["parts"]))
            for record in audit if record["accepted"]
        }
        generated_keys = {
            (tuple(sorted(item.parts)), item.candidate_type)
            for item in [_contract(match, CandidateStatus.generated) for match in scored]
        }
        kept_keys = {
            (tuple(sorted(item.parts)), item.candidate_type)
            for item in [_contract(match, CandidateStatus.kept) for match in kept]
        }
        true_pairs = {pair for pair, _ in truth}
        outputs["index_quality.json"] = {
            "ground_truth_edges": len(truth),
            "ground_truth_pairs": len(true_pairs),
            "prescreen_pair_recall": (
                len(true_pairs & accepted_pair_keys) / len(true_pairs)
                if true_pairs else None
            ),
            "generated_typed_edge_recall": (
                len(truth & generated_keys) / len(truth) if truth else None
            ),
            "pruned_typed_edge_recall": (
                len(truth & kept_keys) / len(truth) if truth else None
            ),
            "missing_after_prescreen": [
                list(pair) for pair in sorted(true_pairs - accepted_pair_keys)
            ],
            "missing_after_generation": [
                {"parts": list(pair), "type": kind}
                for pair, kind in sorted(truth - generated_keys)
            ],
            "missing_after_pruning": [
                {"parts": list(pair), "type": kind}
                for pair, kind in sorted(truth - kept_keys)
            ],
        }
    for name, data in outputs.items():
        (output_dir / name).write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "configs" / "pool_pipeline.json"),
    )
    parser.add_argument(
        "--joinable-report",
        help=(
            "Cached output from batch_mixed_pool_inference.py. JoinABLe "
            "only expands the detailed analytic candidate frontier."
        ),
    )
    args = parser.parse_args()
    outputs = index_pool(
        args.input_dir,
        args.output_dir,
        args.config,
        joinable_report_path=args.joinable_report,
    )
    print(
        f"parts={len(outputs['part_features.json'])} "
        f"candidates={len(outputs['geometry_candidates.json'])} "
        f"kept={len(outputs['pruned_candidates.json'])}"
    )
    print(Path(args.output_dir).resolve())


if __name__ == "__main__":
    main()
