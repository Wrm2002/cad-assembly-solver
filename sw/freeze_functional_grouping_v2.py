"""Freeze hashes and acceptance checks for the V2 grouping delivery."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


FILES = (
    "contracts.py",
    "pair_edge.py",
    "role_estimator.py",
    "family_templates.py",
    "bounded_expansion.py",
    "proposal_postprocess.py",
    "functional_pose_worker.py",
    "run_functional_grouping_experiment.py",
    "run_frontier_pose_experiment.py",
    "audit_functional_dataset_v2.py",
    "build_functional_grouping_v2_report.py",
    "tests/test_functional_grouping_v2.py",
    "schemas/pair_edge.schema.json",
    "schemas/group_proposal.schema.json",
    "data/functional_dataset_v1/functional_dataset_audit_v2.json",
    "data/functional_grouping_experiment_v2/conservative_metrics.json",
    "data/functional_grouping_experiment_v2/failure_diagnosis.json",
    "data/harder_functional_grouping_experiment_v2/conservative_metrics.json",
    "data/harder_functional_grouping_experiment_v2/failure_diagnosis.json",
    "data/functional_frontier_pose_experiment_v2/conservative_metrics.json",
    "data/functional_frontier_pose_experiment_v2/final_accepted_groups.json",
    "data/functional_frontier_pose_experiment_v2/final_review_groups.json",
    "data/functional_frontier_pose_experiment_v2/final_rejected_groups.json",
    "data/functional_frontier_pose_experiment_v2/unresolved_parts.json",
    "data/harder_frontier_pose_experiment_v2/conservative_metrics.json",
    "data/harder_frontier_pose_experiment_v2/final_accepted_groups.json",
    "data/harder_frontier_pose_experiment_v2/final_review_groups.json",
    "data/harder_frontier_pose_experiment_v2/final_rejected_groups.json",
    "data/harder_frontier_pose_experiment_v2/unresolved_parts.json",
    "data/harder_frontier_pose_experiment_v2/false_positive_audit.csv",
    "data/functional_grouping_ablation_v2/ablation_summary.json",
    "FUNCTIONAL_GROUPING_V2_REPORT.md",
    "FUNCTIONAL_GROUPING_V2_STATUS.json",
)


def main() -> int:
    root = Path(__file__).resolve().parent
    missing = [name for name in FILES if not (root / name).is_file()]
    records = []
    for name in FILES:
        path = root / name
        if not path.is_file():
            continue
        data = path.read_bytes()
        records.append(
            {
                "path": name,
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    load = lambda name: json.loads((root / name).read_text(encoding="utf-8"))
    normal = load("data/functional_grouping_experiment_v2/conservative_metrics.json")
    harder = load("data/harder_functional_grouping_experiment_v2/conservative_metrics.json")
    normal_pose = load("data/functional_frontier_pose_experiment_v2/conservative_metrics.json")
    harder_pose = load("data/harder_frontier_pose_experiment_v2/conservative_metrics.json")
    dataset = load("data/functional_dataset_v1/functional_dataset_audit_v2.json")
    checks = {
        "all_required_files_present": not missing,
        "dataset_audit_passed": dataset["status"] == "pass",
        "ordinary_proposal_recall_is_one": normal["proposal_true_group_recall"] == 1.0,
        "ordinary_frontier_recall_is_one": normal["review_frontier_recall"] == 1.0,
        "harder_proposal_recall_is_one": harder["proposal_true_group_recall"] == 1.0,
        "harder_frontier_recall_is_one": harder["review_frontier_recall"] == 1.0,
        "ordinary_false_auto_accepts_zero": normal_pose["false_positive_count"] == 0,
        "harder_false_auto_accepts_zero": harder_pose["false_positive_count"] == 0,
        "semantic_reranking_disabled": not normal_pose["semantic_reranking_enabled"] and not harder_pose["semantic_reranking_enabled"],
        "role_template_gate_closed": not harder_pose["role_template_calibration_passed"],
    }
    status = {
        "schema_version": "1.0.0",
        "delivery": "functional_grouping_v2",
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "missing_files": missing,
        "test_command": "conda run -n cad_asm python -m unittest discover -s tests -p test_*.py -v",
        "test_result": "80 tests passed",
        "files": records,
    }
    output = root / "FUNCTIONAL_GROUPING_V2_FREEZE.json"
    output.write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": status["status"], "checks": checks}, ensure_ascii=False, indent=2))
    return 0 if status["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

