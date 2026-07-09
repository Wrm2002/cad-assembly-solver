"""Build the frozen audit report for the mixed-pool functional route."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from run_frontier_pose_experiment import _decide


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    root = Path(__file__).resolve().parent
    normal = load(root / "data/functional_grouping_experiment_v2/conservative_metrics.json")
    harder = load(root / "data/harder_functional_grouping_experiment_v2/conservative_metrics.json")
    normal_pose = load(root / "data/functional_frontier_pose_experiment_v2/conservative_metrics.json")
    harder_pose = load(root / "data/harder_frontier_pose_experiment_v2/conservative_metrics.json")
    dataset_audit = load(
        root / "data/functional_dataset_v1/functional_dataset_audit_v2.json"
    )
    pose_rows = load(root / "data/harder_frontier_pose_experiment_v2/pose_validation_records.json")
    provisional_accepted, _, _ = _decide(
        pose_rows, role_template_calibration_passed=True
    )
    truth = {
        (pool.name, frozenset(group["parts"]))
        for pool in (root / "data/functional_cad_holdout_pools_v1").iterdir()
        if pool.is_dir() and (pool / "pool_gt.json").is_file()
        for group in load(pool / "pool_gt.json").get("true_groups", [])
    }
    unsafe = [
        {
            "pool_id": row["pool_id"],
            "group_id": row["group_id"],
            "parts": row["parts"],
            "assembly_family": row["assembly_family"],
            "is_true_group": (
                row["pool_id"], frozenset(row["parts"])
            ) in truth,
            "initial_decision": "accepted_before_role_template_calibration_gate",
            "final_decision": "review",
            "transfer_reason": "locked holdout showed every final provisional accept was false",
        }
        for row in provisional_accepted
    ]
    ablation_dirs = {
        "analytic_only": "analytic_only",
        "analytic_joinable": "analytic_joinable",
        "analytic_joinable_union": "analytic_joinable_union",
        "queue_5": "queue_5",
        "queue_10": "queue_10",
    }
    ablations = {}
    for name, directory in ablation_dirs.items():
        metric = load(
            root
            / "data/functional_grouping_ablation_v2"
            / directory
            / "conservative_metrics.json"
        )
        ablations[name] = {
            key: metric[key]
            for key in (
                "proposal_count",
                "proposal_true_group_recall",
                "review_frontier_count",
                "review_frontier_recall",
                "review_frontier_precision",
                "false_positive_count",
            )
        }
    summary = {
        "schema_version": "1.0.0",
        "route": "conservative_functional_mixed_pool_grouping_v2",
        "implementation_status": "complete_with_auto_accept_gate_closed",
        "normal_functional_benchmark": normal,
        "locked_harder_holdout": harder,
        "normal_pose_experiment": normal_pose,
        "harder_pose_experiment": harder_pose,
        "functional_dataset_audit": dataset_audit,
        "ablations": ablations,
        "unsafe_provisional_accepts_transferred_to_review": unsafe,
        "external_method_alignment": {
            "joinable_paper": "https://arxiv.org/abs/2111.12772",
            "joinable_code": "https://github.com/AutodeskAILab/JoinABLe",
            "fusion360_dataset": "https://github.com/AutodeskAILab/Fusion360GalleryDataset",
            "conclusion": (
                "JoinABLe predicts pair-level B-Rep joint entities/axes and pose; "
                "it is a core PairEdge provider, not proof of complete functional grouping."
            ),
        },
        "training_decision": {
            "gpu_training_started": False,
            "reason": (
                "Locked holdout proposal recall is already 4/4; remaining failure is "
                "functional ambiguity and pose-valid false groups, not learned pair recall."
            ),
        },
        "known_limits": [
            "Locked harder holdout still lacks mechanical-engineer signoff.",
            "Role/template calibration gate failed because all 3 provisional accepts were false.",
            "Auto-accept precision is not estimable while accepted_group_count is zero.",
            "Two provisional true groups failed exact pose and were rejected under D5 rules.",
            "JoinABLe union did not improve locked-holdout frontier recall over analytic-only in this four-group sample.",
        ],
    }
    write(root / "FUNCTIONAL_GROUPING_V2_STATUS.json", summary)
    write(
        root / "data/functional_grouping_ablation_v2/ablation_summary.json",
        ablations,
    )
    with (root / "data/harder_frontier_pose_experiment_v2/false_positive_audit.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        fields = [
            "pool_id", "group_id", "parts", "assembly_family",
            "is_true_group", "initial_decision", "final_decision",
            "transfer_reason",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in unsafe:
            item = dict(row)
            item["parts"] = "|".join(item["parts"])
            writer.writerow(item)

    report = f"""# Mixed-pool Functional Assembly Grouping V2

## Outcome

The eight requested components are implemented and tested. The route is now a conservative, family-aware proposal and review system. It does not force a full partition, does not use semantic models for hard decisions, and does not allow JoinABLe confidence to become physical evidence.

## Implemented route

1. Unified `PairEdge` records aggregate analytic and JoinABLe providers while separating provider agreement from physical evidence.
2. Geometry-only multi-label center/role estimation ignores embedded synthetic semantic truth.
3. Family slot templates cover `cover_base`, `shaft_hub_key`, and `bearing_housing`, including optional axial/bearing retainers.
4. Bounded expansion enumerates center + slots rather than arbitrary subsets.
5. Subset/superset links are recorded before pose; demotion occurs only when the superset is also pose-valid.
6. Proposal clustering is family-isolated and only compresses review presentation.
7. Failure diagnosis separates candidate recall, proposal generation, ranking, pose, and semantic/calibration failures.
8. The locked topology-varied functional holdout is evaluated separately from the ordinary functional benchmark.

## Proposal and review results

| Benchmark | Proposals | Proposal recall | Review frontier | Frontier recall | Frontier precision |
|---|---:|---:|---:|---:|---:|
| Previous ordinary baseline | 9,668 | 100% | 72 | 22.22% | 2.78% |
| V2 ordinary functional | {normal['proposal_count']} | {normal['proposal_true_group_recall']:.2%} | {normal['review_frontier_count']} | {normal['review_frontier_recall']:.2%} | {normal['review_frontier_precision']:.2%} |
| V2 locked harder holdout | {harder['proposal_count']} | {harder['proposal_true_group_recall']:.2%} | {harder['review_frontier_count']} | {harder['review_frontier_recall']:.2%} | {harder['review_frontier_precision']:.2%} |

The ordinary proposal count fell by {(1-normal['proposal_count']/9668):.2%}, while review-frontier true-group recall rose from 2/9 to 9/9. The locked harder holdout generated and surfaced 4/4 provisional true groups.

The D0 functional dataset audit passed: {dataset_audit['case_count']} cases, three cases per family, zero invalid cases, zero generic cone/ring/block/plate stack positives, all three required hard-negative types present, and no production decision treats source ID as truth.

## Exact pose and false-positive audit

| Benchmark | Checked | Pose valid | Failed | Uncertain | Final accepted | Review | Rejected | False auto accepts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Ordinary functional | 60 | {normal_pose['pose_valid_count']} | {normal_pose['pose_failed_count']} | {normal_pose['pose_uncertain_count']} | {normal_pose['accepted_group_count']} | {normal_pose['review_group_count']} | {normal_pose['rejected_group_count']} | {normal_pose['false_positive_count']} |
| Locked harder holdout | 40 | {harder_pose['pose_valid_count']} | {harder_pose['pose_failed_count']} | {harder_pose['pose_uncertain_count']} | {harder_pose['accepted_group_count']} | {harder_pose['review_group_count']} | {harder_pose['rejected_group_count']} | {harder_pose['false_positive_count']} |

On the ordinary benchmark, 49 false groups were pose-valid; on the harder holdout, 21 false groups were pose-valid. This empirically confirms that pose success and collision freedom are not evidence of functional membership.

An exploratory gate would have auto-accepted {len(unsafe)} harder-holdout groups; all {len(unsafe)} were false. Those groups are now transferred to review and the role/template calibration gate is closed. Final false auto accepts are zero. Accepted precision is therefore not estimable rather than being reported as 100%.

## Ablation conclusions

- Queue size 5 recalled {ablations['queue_5']['review_frontier_recall']:.2%}; queue size 10 recalled {ablations['queue_10']['review_frontier_recall']:.2%}. Ten per pool is the smallest tested budget with 4/4 harder-holdout recall.
- Analytic-only and analytic+JoinABLe union both generated 4/4 true proposals, but each surfaced only 3/4 at its current ablation frontier. JoinABLe remains a core recall provider, but this holdout shows it does not solve group ranking by itself.
- Pre-pose subset demotion was tested and rejected: it reduced ordinary frontier recall from 9/9 to 6/9. The implementation now waits for pose confirmation.
- Qwen/DeepSeek reranking remained disabled in every experiment.

## External-method review

The [JoinABLe paper](https://arxiv.org/abs/2111.12772) and [official implementation](https://github.com/AutodeskAILab/JoinABLe) define a pair-of-parts B-Rep entity/link-prediction and joint-pose task. That supports using JoinABLe as a first-class `PairEdge` provider but does not support treating it as a complete mixed-pool grouping judge. The [Fusion 360 Gallery Assembly Dataset](https://github.com/AutodeskAILab/Fusion360GalleryDataset) provides joint and assembly graph supervision, but functional validity still requires the project-specific signed holdout.

## Gate status and remaining blocker

- Semantic reranking: disabled.
- Role/template auto-accept calibration: failed and closed.
- Mechanical-engineer signoff for the locked holdout: pending.
- GPU training: not started because proposal recall is already saturated on both tested benchmarks; current errors are semantic/functional ambiguity and pose-valid false combinations.

The next minimal step is not larger search or model training. It is signed review of the locked holdout plus discriminative functional-interface features that separate the 49/21 pose-valid false groups from true groups. Until that calibration passes, the correct engineering output is high-recall review with zero automatic false acceptance.
"""
    (root / "FUNCTIONAL_GROUPING_V2_REPORT.md").write_text(
        report, encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "report": str(root / "FUNCTIONAL_GROUPING_V2_REPORT.md"),
                "status": str(root / "FUNCTIONAL_GROUPING_V2_STATUS.json"),
                "unsafe_transferred_to_review": len(unsafe),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
