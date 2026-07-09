"""Prepare structured D6 inputs for the functional benchmark without an LLM call."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def prepare(
    pools_root: str | Path,
    results_root: str | Path,
) -> dict[str, Any]:
    pools = Path(pools_root).resolve()
    results = Path(results_root).resolve()
    review = _load(results / "final_review_groups.json")
    selected = [
        row
        for row in review
        if row.get("review_queue_state") == "selected"
    ]
    semantic_by_pool = {
        pool.name: _load(pool / "part_semantics.json")
        for pool in pools.iterdir()
        if pool.is_dir() and (pool / "part_semantics.json").is_file()
    }
    inputs = []
    reviews = []
    for candidate in selected:
        semantics = semantic_by_pool[candidate["pool_id"]]
        parts = []
        for part_id in candidate["parts"]:
            semantic = semantics.get(part_id, {})
            parts.append(
                {
                    "part_id": part_id,
                    "part_name": semantic.get("part_name"),
                    "file_name": semantic.get("file_name", part_id),
                    "part_role": semantic.get("part_role"),
                    "interface_type": semantic.get("interface_type", []),
                    "assembly_family": semantic.get(
                        "assembly_family", "unknown"
                    ),
                    "functional_relation": semantic.get(
                        "functional_relation", []
                    ),
                    "functions": semantic.get("functions", []),
                    "BOM_hint": None,
                    "CAD_metadata": {
                        "source": semantic.get("source"),
                    },
                }
            )
        payload = {
            "schema_version": "1.0.0",
            "candidate_id": candidate["group_id"],
            "pool_id": candidate["pool_id"],
            "task": "functional assembly review with abstention",
            "parts": parts,
            "assembly_family_hypotheses": sorted(
                {
                    part["assembly_family"]
                    for part in parts
                    if part["assembly_family"] != "unknown"
                }
            ),
            "functional_relations": sorted(
                {
                    relation
                    for part in parts
                    for relation in part["functional_relation"]
                }
            ),
            "geometry_summary": {
                "geometry_score": candidate.get("geometry_score"),
                "group_consistency": candidate.get("consistency"),
                "pose_status": (
                    candidate.get("pose") or {}
                ).get("final_pose_status"),
            },
            "constraints": {
                "source_identity_hidden": True,
                "source_template_for_production_decision": False,
                "semantic_output_may_change_grouping": False,
                "abstention_allowed": True,
            },
        }
        inputs.append(payload)
        reviews.append(
            {
                "candidate_id": candidate["group_id"],
                "semantic_validity": "unknown",
                "semantic_score": None,
                "functional_reason": (
                    "Structured functional evidence prepared; no calibrated "
                    "semantic provider was called."
                ),
                "possible_system": payload[
                    "assembly_family_hypotheses"
                ],
                "risk": "semantic calibration gate remains closed",
                "is_geometrically_feasible_but_semantically_invalid": None,
                "review_required": True,
                "suggested_action": "abstain",
                "provider_called": False,
                "affects_final_decision": False,
            }
        )
    calibration = {
        "schema_version": "1.0.0",
        "semantic_reranking_enabled": False,
        "semantic_application_mode": "explanation_only",
        "structured_functional_fields_available": True,
        "input_count": len(inputs),
        "provider_called": False,
        "gate_rules": {
            "minimum_semantic_auc": 0.70,
            "brier_must_improve_over_geometry": True,
            "required_verdicts": ["accept", "reject", "abstain"],
            "holdout_auto_accept_precision_not_decreased": True,
            "holdout_false_positive_count_not_increased": True,
        },
        "gate_failure_reasons": [
            "no provider calibration was run on the functional holdout",
            "holdout acceptance safety has not been demonstrated",
        ],
    }
    _write(results / "semantic_inputs.json", inputs)
    _write(results / "semantic_reviews.json", reviews)
    _write(results / "semantic_calibration_report.json", calibration)
    (results / "semantic_gate_decision.md").write_text(
        "# Functional semantic gate\n\n"
        "**Disabled for grouping and reranking.**\n\n"
        f"- Structured inputs: {len(inputs)}\n"
        "- Required fields: part name, role, interface type, assembly family, "
        "and functional relation are present.\n"
        "- Provider calls: 0\n"
        "- Suggested action: abstain/review\n"
        "- Semantic outputs do not change scores or tiers.\n",
        encoding="utf-8",
    )
    return calibration


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pools-root",
        default=str(
            Path(__file__).resolve().parent
            / "data"
            / "functional_mixed_pools_v1"
        ),
    )
    parser.add_argument(
        "--results-root",
        default=str(
            Path(__file__).resolve().parent
            / "data"
            / "functional_results"
        ),
    )
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(args.pools_root, args.results_root),
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
