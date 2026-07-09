"""Validate and hash a completed JoinABLe harder-holdout ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ablation_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.ablation_root.resolve()
    output = args.output or root / "JOINABLE_HARDER_HOLDOUT_FREEZE.json"
    required = [
        root / "run_manifest.json",
        root / "audit" / "strict_audit.json",
        root / "audit" / "EXPERT_REVIEW.md",
        root / "audit" / "real_joint_rescue_cases.csv",
        root / "audit" / "false_candidate_routes.csv",
        root / "cache" / "joinable_pair_rankings_cpu.json",
        root / "cache" / "joinable_pair_rankings_cuda.json",
        root / "cache" / "domain_holdout_union_test_report.json",
        root / "results" / "analytic_only" / "conservative_metrics.json",
        root / "results" / "analytic_joinable" / "conservative_metrics.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    failures = [f"missing:{path}" for path in missing]
    if not missing:
        manifest = read_json(required[0])
        audit = read_json(required[1])
        domain = audit.get("design_disjoint_domain_recall") or {}
        safety = audit["functional_safety"]
        checks = {
            "run_completed": manifest.get("completed") is True,
            "training_not_performed": manifest.get("training_performed") is False,
            "domain_exact_evaluable_at_least_200": domain.get("evaluable_count", 0) >= 200,
            "domain_union_improves_analytic": (
                domain.get("union_recall", 0) > domain.get("analytic_recall", 0)
            ),
            "domain_joinable_rescues_present": domain.get("joinable_rescue_count", 0) > 0,
            "no_new_false_auto_accepts": safety.get("new_false_auto_accepts") == 0,
            "all_new_false_candidates_blocked": safety.get(
                "all_new_false_candidates_blocked_before_acceptance"
            ) is True,
            "semantic_reranking_disabled_analytic": safety["analytic_metrics"].get(
                "semantic_reranking_enabled"
            ) is False,
            "semantic_reranking_disabled_joinable": safety[
                "analytic_joinable_metrics"
            ].get("semantic_reranking_enabled") is False,
            "cpu_inference_complete": manifest["stages"]["inference_cpu"][
                "success_count"
            ] == manifest["stages"]["inference_cpu"]["pair_count"],
            "cuda_inference_complete": manifest["stages"]["inference_cuda"][
                "success_count"
            ] == manifest["stages"]["inference_cuda"]["pair_count"],
        }
        failures.extend(name for name, passed in checks.items() if not passed)
    else:
        checks = {}
    freeze = {
        "schema_version": "1.0.0",
        "ablation_root": str(root),
        "route_lock_passed": not failures,
        "checks": checks,
        "files": [
            {
                "path": str(path.relative_to(root)),
                "size": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in required
            if path.is_file()
        ],
        "failure_reasons": failures,
    }
    output.write_text(
        json.dumps(freeze, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"route_lock_passed": not failures, "failures": failures}, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
