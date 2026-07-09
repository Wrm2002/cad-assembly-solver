"""Force all D0 hard negatives through the exact D4/D5 decision path."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from conservative_pipeline import (
    _passes_final_acceptance,
    _pose_record,
    geometry_tiers,
)
from contracts import GroupProposal
from geometry_pipeline import isolated_solve_and_validate_group
from pool_index import index_pool


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _candidate_id(case_id: str, negative_id: str) -> str:
    digest = hashlib.sha256(
        f"{case_id}:{negative_id}".encode("utf-8")
    ).hexdigest()[:12]
    return f"HN_{digest}"


def _copy_case_parts(case_dir: Path, pool: Path) -> None:
    parts_dir = pool / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    for source in sorted((case_dir / "parts").glob("*.step")):
        shutil.copy2(source, parts_dir / source.name)
    for source in sorted((case_dir / "negatives").glob("*.step")):
        shutil.copy2(source, parts_dir / source.name)


def _proposal(
    case_id: str,
    negative: dict[str, Any],
    edges: list[dict[str, Any]],
) -> dict[str, Any]:
    part_files = [
        part
        if str(part).lower().endswith((".step", ".stp"))
        else f"{part}.step"
        for part in negative["parts"]
    ]
    parts = set(part_files)
    matching = [
        edge for edge in edges if set(edge["parts"]) == parts
    ]
    matching.sort(
        key=lambda edge: (
            float(edge["geometry_score"]),
            edge["candidate_id"],
        ),
        reverse=True,
    )
    selected = matching[:1]
    score = (
        float(selected[0]["geometry_score"]) if selected else 0.0
    )
    return {
        "schema_version": "1.0.0",
        "group_id": _candidate_id(case_id, negative["negative_id"]),
        "parts": part_files,
        "candidate_edges": [
            edge["candidate_id"] for edge in selected
        ],
        "geometry_score": score,
        "connected": bool(selected),
        "status": "forced_evaluation_candidate",
        "reasons": [
            "evaluation_only_forced_exact_hard_negative",
            (
                "production_edge_available"
                if selected
                else "no_production_edge_available"
            ),
        ],
    }


def run(
    dataset_root: str | Path,
    workspace_root: str | Path,
    results_root: str | Path,
    pipeline_config_path: str | Path,
    conservative_config_path: str | Path,
) -> dict[str, Any]:
    dataset_root = Path(dataset_root).resolve()
    workspace_root = Path(workspace_root).resolve()
    results_root = Path(results_root).resolve()
    pipeline_config_path = Path(pipeline_config_path).resolve()
    conservative_config_path = Path(conservative_config_path).resolve()
    pipeline_config = _load(pipeline_config_path)
    conservative_config = _load(conservative_config_path)
    workspace_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for metadata_path in sorted(dataset_root.glob("*/metadata.json")):
        case_dir = metadata_path.parent
        metadata = _load(metadata_path)
        pool = workspace_root / metadata["case_id"]
        if pool.exists():
            shutil.rmtree(pool)
        _copy_case_parts(case_dir, pool)
        index_pool(
            pool / "parts",
            pool / "index",
            pipeline_config_path,
        )
        edges = _load(pool / "index" / "pruned_candidates.json")
        proposals = [
            _proposal(metadata["case_id"], negative, edges)
            for negative in metadata["negative_groups"]
        ]
        _write(pool / "grouping" / "group_proposals.json", proposals)
        accepted, review, rejected = geometry_tiers(
            pool, proposals, edges, conservative_config
        )
        tiers = {
            item["group_id"]: item
            for item in accepted + review + rejected
        }

        for negative, proposal_data in zip(
            metadata["negative_groups"], proposals
        ):
            item = tiers[proposal_data["group_id"]]
            pose: dict[str, Any] | None = None
            if item["geometry_tier"] != "rejected":
                validation = isolated_solve_and_validate_group(
                    pool,
                    GroupProposal.model_validate(proposal_data),
                    pipeline_config_path,
                    pipeline_config,
                )
                pose = _pose_record(
                    pool, item, validation, checked=True
                )

            if item["geometry_tier"] == "rejected":
                final_decision = "rejected"
                final_reason = "rejected_by_d4_geometry_gate"
            elif pose is None or pose["final_pose_status"] == "uncertain":
                final_decision = "review"
                final_reason = "pose_uncertain"
            elif pose["final_pose_status"] == "failed":
                final_decision = "rejected"
                final_reason = "pose_failed"
            elif _passes_final_acceptance(
                item, pose, conservative_config
            ):
                final_decision = "accepted"
                final_reason = "passed_production_d4_d5_gate"
            else:
                final_decision = "review"
                final_reason = "valid_pose_but_conservative_gate_requires_review"

            records.append(
                {
                    "case_id": metadata["case_id"],
                    "assembly_family": metadata["assembly_family"],
                    "negative_id": negative["negative_id"],
                    "negative_type": negative["negative_type"],
                    "parts": proposal_data["parts"],
                    "geometry_feasible_label": negative[
                        "geometry_feasible"
                    ],
                    "functional_validity": negative[
                        "functional_validity"
                    ],
                    "candidate_id": proposal_data["group_id"],
                    "production_edge_available": bool(
                        proposal_data["candidate_edges"]
                    ),
                    "candidate_edge_count": len(
                        proposal_data["candidate_edges"]
                    ),
                    "geometry_score": item["geometry_score"],
                    "group_consistency_score": item["consistency"][
                        "group_consistency_score"
                    ],
                    "independent_evidence_count": item["consistency"][
                        "independent_evidence_count"
                    ],
                    "group_completeness_score": item["consistency"][
                        "group_completeness_score"
                    ],
                    "central_part_coverage": item["consistency"][
                        "central_part_coverage"
                    ],
                    "interface_diversity_score": item["consistency"][
                        "interface_diversity_score"
                    ],
                    "d4_geometry_tier": item["geometry_tier"],
                    "d4_reasons": item["decision_reasons"],
                    "d5_executed": pose is not None,
                    "pose_status": (
                        pose["final_pose_status"]
                        if pose
                        else "not_applicable_d4_reject"
                    ),
                    "checked_pose_count": (
                        pose["checked_pose_count"] if pose else 0
                    ),
                    "collision_result": (
                        pose["collision_result"] if pose else "not_run"
                    ),
                    "occt_common_volume": (
                        pose["occt_common_volume"] if pose else None
                    ),
                    "final_decision": final_decision,
                    "final_reason": final_reason,
                    "auto_accepted_functional_false_positive": (
                        final_decision == "accepted"
                    ),
                    "evaluation_only": True,
                }
            )

    decision_counts = Counter(
        row["final_decision"] for row in records
    )
    type_decisions: dict[str, Counter[str]] = {}
    for row in records:
        type_decisions.setdefault(
            row["negative_type"], Counter()
        )[row["final_decision"]] += 1
    summary = {
        "schema_version": "1.0.0",
        "artifact_role": "evaluation_only",
        "truth_basis": "functional_validity",
        "hard_negative_count": len(records),
        "exact_candidate_count": len(records),
        "production_edge_available_count": sum(
            row["production_edge_available"] for row in records
        ),
        "d5_executed_count": sum(
            row["d5_executed"] for row in records
        ),
        "auto_accepted_functional_false_positive_count": sum(
            row["auto_accepted_functional_false_positive"]
            for row in records
        ),
        "decision_counts": dict(sorted(decision_counts.items())),
        "decisions_by_negative_type": {
            key: dict(sorted(value.items()))
            for key, value in sorted(type_decisions.items())
        },
        "records": records,
    }
    _write(results_root / "forced_hard_negative_audit.json", summary)

    fields = [
        "case_id",
        "assembly_family",
        "negative_id",
        "negative_type",
        "parts",
        "geometry_feasible_label",
        "functional_validity",
        "production_edge_available",
        "geometry_score",
        "group_consistency_score",
        "independent_evidence_count",
        "group_completeness_score",
        "central_part_coverage",
        "interface_diversity_score",
        "d4_geometry_tier",
        "d4_reasons",
        "d5_executed",
        "pose_status",
        "checked_pose_count",
        "collision_result",
        "occt_common_volume",
        "final_decision",
        "final_reason",
        "auto_accepted_functional_false_positive",
    ]
    with (results_root / "forced_hard_negative_audit.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            output = {key: row.get(key) for key in fields}
            output["parts"] = "|".join(row["parts"])
            output["d4_reasons"] = "|".join(row["d4_reasons"])
            writer.writerow(output)

    lines = [
        "# Forced Exact Hard-Negative D4/D5 Audit",
        "",
        f"- Exact hard-negative subjects: {len(records)}",
        (
            "- Production edge available: "
            f"{summary['production_edge_available_count']}"
        ),
        f"- D5 executed: {summary['d5_executed_count']}",
        (
            "- Auto-accepted functional false positives: "
            f"{summary['auto_accepted_functional_false_positive_count']}"
        ),
        f"- Final decisions: {summary['decision_counts']}",
        "",
        "| Case | Type | D4 | D5 pose | Final |",
        "|---|---|---|---|---|",
    ]
    for row in records:
        lines.append(
            f"| {row['case_id']} | {row['negative_type']} | "
            f"{row['d4_geometry_tier']} | {row['pose_status']} | "
            f"{row['final_decision']} |"
        )
    (results_root / "forced_hard_negative_audit.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        default=str(here / "data" / "functional_dataset_v1"),
    )
    parser.add_argument(
        "--workspace-root",
        default=str(
            here / "data" / "functional_hard_negative_benchmark_v1"
        ),
    )
    parser.add_argument(
        "--results-root",
        default=str(here / "data" / "functional_results"),
    )
    parser.add_argument(
        "--pipeline-config",
        default=str(here / "configs" / "pool_pipeline.json"),
    )
    parser.add_argument(
        "--conservative-config",
        default=str(here / "configs" / "conservative_pipeline.json"),
    )
    args = parser.parse_args()
    summary = run(
        args.dataset_root,
        args.workspace_root,
        args.results_root,
        args.pipeline_config,
        args.conservative_config,
    )
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "hard_negative_count",
                    "production_edge_available_count",
                    "d5_executed_count",
                    "auto_accepted_functional_false_positive_count",
                    "decision_counts",
                    "decisions_by_negative_type",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
