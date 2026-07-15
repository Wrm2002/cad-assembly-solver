"""Build a conservative auditable assembly candidate graph.

Rule-only pair predictions are never promoted to automatic acceptance.  This
preserves the project's false-positive-first safety policy until a learned
interface scorer is independently calibrated.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path


def dump(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical_pair(parts: list[str]) -> tuple[str, str]:
    return tuple(sorted(parts))  # type: ignore[return-value]


def components(parts: list[str], edges: list[dict]) -> list[list[str]]:
    adjacency = {part: set() for part in parts}
    for edge in edges:
        a, b = edge["parts"]
        adjacency[a].add(b)
        adjacency[b].add(a)
    seen = set()
    result = []
    for start in parts:
        if start in seen:
            continue
        queue = deque([start])
        group = []
        seen.add(start)
        while queue:
            current = queue.popleft()
            group.append(current)
            for neighbor in sorted(adjacency[current]):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        result.append(sorted(group))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-pair-report", required=True)
    parser.add_argument("--manual-interfaces", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    source = load(args.all_pair_report)
    out = Path(args.out_dir)
    truth = {}
    for path in Path(args.manual_interfaces).glob("case_*.json"):
        template = load(path)
        truth[str(template["case_id"])] = {
            canonical_pair(row["parts"]) for row in template["positive_part_pairs"]
        }

    accepted = []
    review = []
    rejected = []
    unresolved = []
    assembly_outputs = []
    graph_cases = []
    events = []
    event_counter = 0

    def event(case_id: str, action: str, status: str, details: dict) -> None:
        nonlocal event_counter
        event_counter += 1
        events.append(
            {
                "event_id": f"AE{event_counter:05d}",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "case_id": case_id,
                "actor": "deterministic_agent_controller",
                "action": action,
                "status": status,
                "details": details,
            }
        )

    for case in source["cases"]:
        case_id = str(case["case_id"])
        case_accepted = []
        case_review = []
        case_rejected = []
        event(
            case_id,
            "enumerate_part_pairs",
            "success",
            {"part_count": len(case["parts"]), "pair_count": case["pair_count"]},
        )
        for pair in case["pairs"]:
            prediction = load(pair["prediction_path"]) if Path(pair["prediction_path"]).exists() else {}
            best_prediction = (
                prediction.get("candidates", [None])[0]
                if prediction.get("candidates")
                else None
            )
            valid_result = None
            uncertain_count = 0
            checked_count = 0
            for validation in pair["validations"]:
                checked_count += 1
                result = load(validation["output"])
                if result.get("final_pose_status") == "valid" and valid_result is None:
                    valid_result = result
                if result.get("final_pose_status") == "uncertain":
                    uncertain_count += 1
            chosen_score = (
                float(valid_result["candidate_score"])
                if valid_result and valid_result.get("candidate_score") is not None
                else (
                    float(best_prediction["score"])
                    if best_prediction is not None
                    else None
                )
            )
            evidence = []
            if best_prediction:
                score_evidence = best_prediction.get("score_evidence", {})
                if float(score_evidence.get("type_compatibility", 0.0)) >= 0.9:
                    evidence.append("analytic_entity_type_compatibility")
                if max(
                    float(score_evidence.get("characteristic_size_compatibility", 0.0)),
                    float(score_evidence.get("radius_compatibility", 0.0)),
                ) >= 0.8:
                    evidence.append("dimension_or_radius_compatibility")
            if valid_result:
                evidence.extend(["occt_contact", "occt_collision_free"])
            evidence = sorted(set(evidence))
            record = {
                "case_id": case_id,
                "candidate_id": pair["pair_id"],
                "parts": pair["parts"],
                "interface_score": chosen_score,
                "evidence": evidence,
                "independent_evidence_count": len(evidence),
                "checked_candidate_count": checked_count,
                "pose_status": (
                    "valid"
                    if valid_result
                    else ("uncertain" if uncertain_count else "failed")
                ),
                "selected_pose_path": (
                    next(
                        (
                            row["output"]
                            for row in pair["validations"]
                            if load(row["output"]).get("final_pose_status") == "valid"
                        ),
                        None,
                    )
                    if valid_result
                    else None
                ),
                "predictor_kind": source["predictor_kind"],
                "semantic_reranking_enabled": False,
                "is_manual_positive_for_evaluation_only": canonical_pair(pair["parts"])
                in truth.get(case_id, set()),
                "failure_reasons": [],
                "unavailable_fields": ["calibrated_learned_interface_probability"],
            }
            if valid_result:
                record.update(
                    {
                        "tier": "review",
                        "review_required": True,
                        "decision_reason": (
                            "OCCT found a collision-free contact pose, but the "
                            "interface ranker is an uncalibrated deterministic "
                            "baseline; physical feasibility alone is insufficient."
                        ),
                    }
                )
                case_review.append(record)
                review.append(record)
                event(
                    case_id,
                    "gate_pair_edge",
                    "review",
                    {
                        "candidate_id": pair["pair_id"],
                        "reason": "valid_pose_but_uncalibrated_rule_ranker",
                    },
                )
            elif uncertain_count:
                record.update(
                    {
                        "tier": "review",
                        "review_required": True,
                        "decision_reason": (
                            "Bounded OCCT validation was numerically or "
                            "resource uncertain."
                        ),
                    }
                )
                case_review.append(record)
                review.append(record)
                event(
                    case_id,
                    "gate_pair_edge",
                    "review",
                    {
                        "candidate_id": pair["pair_id"],
                        "reason": "pose_validation_uncertain",
                    },
                )
            else:
                record.update(
                    {
                        "tier": "rejected",
                        "review_required": False,
                        "decision_reason": (
                            "No collision-free contact pose was found within "
                            "the bounded ranked candidate search."
                        ),
                        "failure_reasons": ["bounded_pose_search_no_valid_pose"],
                    }
                )
                case_rejected.append(record)
                rejected.append(record)
                event(
                    case_id,
                    "gate_pair_edge",
                    "rejected",
                    {
                        "candidate_id": pair["pair_id"],
                        "reason": "bounded_pose_search_no_valid_pose",
                    },
                )

        accepted_components = components(case["parts"], case_accepted)
        possible_components = components(
            case["parts"], case_accepted + case_review
        )
        accepted_part_ids = {
            part for edge in case_accepted for part in edge["parts"]
        }
        case_unresolved = [
            {
                "case_id": case_id,
                "part": part,
                "reason": "not_connected_by_any_accepted_edge",
            }
            for part in case["parts"]
            if part not in accepted_part_ids
        ]
        unresolved.extend(case_unresolved)
        graph_cases.append(
            {
                "case_id": case_id,
                "parts": case["parts"],
                "accepted_edges": case_accepted,
                "review_edges": case_review,
                "rejected_edges": case_rejected,
                "accepted_connected_components": accepted_components,
                "possible_components_including_review": possible_components,
                "unresolved_parts": case_unresolved,
            }
        )
        assembly_outputs.append(
            {
                "case_id": case_id,
                "status": "not_generated",
                "output_path": None,
                "reason": "no_calibrated_auto_accepted_connected_component",
                "unavailable_fields": ["generated_assembly_step"],
            }
        )
        event(
            case_id,
            "build_conservative_graph",
            "success",
            {
                "accepted_edges": len(case_accepted),
                "review_edges": len(case_review),
                "rejected_edges": len(case_rejected),
            },
        )

    manual_positive_count = sum(len(rows) for rows in truth.values())
    reviewed_manual_positives = sum(
        row["is_manual_positive_for_evaluation_only"] for row in review
    )
    rejected_manual_positives = sum(
        row["is_manual_positive_for_evaluation_only"] for row in rejected
    )
    metrics = {
        "schema_version": "1.0.0",
        "accepted_edge_count": len(accepted),
        "review_edge_count": len(review),
        "rejected_edge_count": len(rejected),
        "unresolved_parts_count": len(unresolved),
        "auto_accept_precision": None,
        "auto_accept_precision_reason": "undefined_because_no_rule_only_edge_is_auto_accepted",
        "false_positive_count": 0,
        "review_rate": len(review) / max(len(review) + len(rejected), 1),
        "manual_positive_edge_count": manual_positive_count,
        "manual_positive_edges_in_review": reviewed_manual_positives,
        "manual_positive_edges_rejected_within_bound": rejected_manual_positives,
        "rejected_reason_coverage": (
            sum(bool(row["failure_reasons"]) for row in rejected) / len(rejected)
            if rejected
            else 1.0
        ),
        "safety_interpretation": (
            "False positives are structurally prevented at this stage by "
            "requiring calibrated learned evidence before auto-accept."
        ),
    }
    dump(out / "assembly_candidate_graph.json", {"schema_version": "1.0.0", "cases": graph_cases})
    dump(out / "accepted_edges.json", accepted)
    dump(out / "review_edges.json", review)
    dump(out / "rejected_edges.json", rejected)
    dump(out / "unresolved_parts.json", unresolved)
    dump(out / "assembly_outputs.json", assembly_outputs)
    dump(out / "conservative_metrics.json", metrics)
    dump(out / "agent_events.json", events)

    lines = [
        "# Conservative Assembly Candidate Graph",
        "",
        "This graph is a conservative engineering result, not a claim of complete automatic assembly.",
        "",
        f"- Accepted edges: {len(accepted)}",
        f"- Review edges: {len(review)}",
        f"- Rejected edges: {len(rejected)}",
        f"- Unresolved parts: {len(unresolved)}",
        f"- False-positive count among auto-accepted edges: {metrics['false_positive_count']}",
        "- Semantic reranking: disabled",
        "- Learned JoinABLe scorer: unavailable in the isolated OCCT environment",
        "",
        "Rule-only physically feasible poses remain in review because collision-free contact is not proof of functional assembly.",
        "",
        "## Per case",
        "",
    ]
    for case in graph_cases:
        lines.extend(
            [
                f"### Case {case['case_id']}",
                "",
                f"- Parts: {len(case['parts'])}",
                f"- Accepted/review/rejected edges: "
                f"{len(case['accepted_edges'])}/{len(case['review_edges'])}/{len(case['rejected_edges'])}",
                f"- Possible components including review: "
                f"{case['possible_components_including_review']}",
                "",
            ]
        )
    (out / "assembly_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"conservative graph accepted/review/rejected="
        f"{len(accepted)}/{len(review)}/{len(rejected)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
