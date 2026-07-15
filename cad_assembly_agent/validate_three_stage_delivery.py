"""Machine-check the three-stage delivery before writing a completion marker."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_check(path: Path, checks: list[dict]) -> dict:
    if not path.exists():
        checks.append({"check": str(path), "passed": False, "reason": "missing"})
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        checks.append({"check": str(path), "passed": True, "reason": "valid_json"})
        return data
    except Exception as exc:
        checks.append(
            {
                "check": str(path),
                "passed": False,
                "reason": f"{type(exc).__name__}:{exc}",
            }
        )
        return {}


def expect(checks: list[dict], name: str, condition: bool, reason: str) -> None:
    checks.append({"check": name, "passed": bool(condition), "reason": reason})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    checks: list[dict] = []
    stage1 = load_check(
        root / "datasets" / "step_real_benchmark" / "frozen_pair_benchmark.json",
        checks,
    )
    edge = load_check(
        root / "datasets" / "step_real_benchmark" / "edge_local_feature_report.json",
        checks,
    )
    pair = load_check(root / "reports" / "pair_benchmark_report.json", checks)
    all_pair = load_check(
        root / "reports" / "all_pair_validation_summary.json", checks
    )
    graph_dir = root / "reports" / "conservative_graph"
    graph = load_check(graph_dir / "assembly_candidate_graph.json", checks)
    metrics = load_check(graph_dir / "conservative_metrics.json", checks)
    reviews = load_check(root / "reports" / "qwen_semantic_reviews.json", checks)
    for required in (
        "accepted_edges.json",
        "review_edges.json",
        "rejected_edges.json",
        "unresolved_parts.json",
        "assembly_outputs.json",
        "agent_events.json",
    ):
        load_check(graph_dir / required, checks)
    expect(
        checks,
        "stage1_has_eight_semantic_pairs",
        stage1.get("semantic_positive_pair_count") == 8,
        f"actual={stage1.get('semantic_positive_pair_count')}",
    )
    expect(
        checks,
        "stage1_records_unavailable_pose_truth",
        int(stage1.get("unusable_expected_pose_pair_count", 0)) > 0,
        "missing labels must be explicit rather than fabricated",
    )
    expect(
        checks,
        "edge_features_cover_fourteen_parts",
        edge.get("part_count") == 14,
        f"actual={edge.get('part_count')}",
    )
    pcba_rows = [
        row
        for row in edge.get("results", [])
        if row.get("part") == "01-62DC24-MLB-PCBA.stp"
    ]
    expect(
        checks,
        "pcba_uses_candidate_local_edge_scope",
        len(pcba_rows) == 1
        and pcba_rows[0].get("scope") == "candidate_local"
        and (
            int(pcba_rows[0].get("candidate_face_count", 0)) > 0
            or int(pcba_rows[0].get("candidate_edge_count", 0)) > 0
        ),
        f"rows={pcba_rows}",
    )
    expect(
        checks,
        "stage2_has_eight_positive_pair_runs",
        pair.get("pair_count") == 8,
        f"actual={pair.get('pair_count')}",
    )
    expect(
        checks,
        "stage2_predictions_completed",
        bool(pair.get("pairs"))
        and all(
            row.get("prediction_status") == "success"
            for row in pair.get("pairs", [])
        ),
        "all eight pairs need ranked candidates",
    )
    expect(
        checks,
        "stage2_has_at_least_one_exact_valid_pose",
        int(pair.get("pair_pose_success_count", 0)) > 0,
        f"actual={pair.get('pair_pose_success_count')}",
    )
    expect(
        checks,
        "stage3_enumerates_all_within_case_pairs",
        all_pair.get("pair_count") == 14,
        f"actual={all_pair.get('pair_count')}",
    )
    expect(
        checks,
        "stage3_graph_has_five_cases",
        len(graph.get("cases", [])) == 5,
        f"actual={len(graph.get('cases', []))}",
    )
    expect(
        checks,
        "semantic_reranking_remains_disabled",
        reviews.get("semantic_reranking_enabled") is False,
        f"actual={reviews.get('semantic_reranking_enabled')}",
    )
    expect(
        checks,
        "false_positive_safe_gate",
        metrics.get("false_positive_count") == 0
        and metrics.get("accepted_edge_count") == 0,
        (
            f"false_positive_count={metrics.get('false_positive_count')}, "
            f"accepted={metrics.get('accepted_edge_count')}"
        ),
    )
    passed = all(check["passed"] for check in checks)
    output = {
        "schema_version": "1.0.0",
        "delivery_complete": passed,
        "check_count": len(checks),
        "passed_count": sum(check["passed"] for check in checks),
        "failed_count": sum(not check["passed"] for check in checks),
        "checks": checks,
        "failure_reasons": [
            f"{check['check']}:{check['reason']}"
            for check in checks
            if not check["passed"]
        ],
        "unavailable_fields": [],
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"delivery validation: {output['passed_count']}/{output['check_count']} "
        f"passed; complete={passed}"
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
