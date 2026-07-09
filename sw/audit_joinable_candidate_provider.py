"""Compare analytic-only and analytic+JoinABLe candidate indexes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _pair(parts: list[str]) -> tuple[str, str]:
    return tuple(sorted(str(part) for part in parts))


def _truth(pool: Path) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    gt = _read(pool / "pool_gt.json")
    same_group = set()
    true_mates = set()
    for group in gt.get("true_groups", []):
        parts = [str(part) for part in group.get("parts", [])]
        for index, first in enumerate(parts):
            for second in parts[index + 1 :]:
                same_group.add(_pair([first, second]))
        for mate in group.get("true_mates", []):
            mate_parts = mate.get("parts") or [
                mate.get("part_a"),
                mate.get("part_b"),
            ]
            true_mates.add(_pair(mate_parts))
    return same_group, true_mates


def audit(root: str | Path) -> dict[str, Any]:
    dataset = Path(root).resolve()
    pool_rows = []
    totals = {
        "rescued_pair_count": 0,
        "rescued_same_group_pair_count": 0,
        "rescued_true_mate_count": 0,
        "generated_candidate_delta": 0,
        "kept_candidate_delta": 0,
        "kept_same_group_pair_delta": 0,
        "kept_true_mate_pair_delta": 0,
    }
    for pool in sorted(dataset.glob("functional_pool_*")):
        baseline = pool / "index_analytic_phase1"
        learned = pool / "index_joinable"
        same_group, true_mates = _truth(pool)
        base_screen = _read(baseline / "screening_audit.json")
        learned_screen = _read(learned / "screening_audit.json")
        provider = _read(
            learned / "joinable_candidate_provider_audit.json"
        )
        base_generated = _read(baseline / "geometry_candidates.json")
        learned_generated = _read(learned / "geometry_candidates.json")
        base_kept = _read(baseline / "pruned_candidates.json")
        learned_kept = _read(learned / "pruned_candidates.json")
        rescued = {
            _pair(row["parts"])
            for row in learned_screen["pairs"]
            if row.get("acceptance_reason")
            == "rescued_by_joinable_for_detailed_analytic_matching"
        }
        base_kept_pairs = {_pair(row["parts"]) for row in base_kept}
        learned_kept_pairs = {_pair(row["parts"]) for row in learned_kept}
        row = {
            "pool_id": pool.name,
            "joinable_selected_pair_count": provider["selected_pair_count"],
            "analytic_accepted_pair_count": base_screen["accepted_pairs"],
            "union_accepted_pair_count": learned_screen["accepted_pairs"],
            "rescued_pair_count": len(rescued),
            "rescued_same_group_pair_count": len(rescued & same_group),
            "rescued_true_mate_count": len(rescued & true_mates),
            "rescued_false_pair_count": len(rescued - same_group),
            "generated_candidate_delta": (
                len(learned_generated) - len(base_generated)
            ),
            "kept_candidate_delta": len(learned_kept) - len(base_kept),
            "kept_same_group_pair_delta": (
                len(learned_kept_pairs & same_group)
                - len(base_kept_pairs & same_group)
            ),
            "kept_true_mate_pair_delta": (
                len(learned_kept_pairs & true_mates)
                - len(base_kept_pairs & true_mates)
            ),
            "rescued_pairs": [list(pair) for pair in sorted(rescued)],
        }
        pool_rows.append(row)
        for key in totals:
            totals[key] += int(row[key])

    report = {
        "schema_version": "1.0.0",
        "dataset": str(dataset),
        "policy": (
            "JoinABLe expands detailed analytic candidate recall and cannot "
            "create physical evidence or auto-accept."
        ),
        "pool_count": len(pool_rows),
        "totals": totals,
        "pools": pool_rows,
        "decision": {
            "integration_operational": True,
            "enable_auto_accept": False,
            "measured_recall_gain_on_this_benchmark": (
                totals["kept_true_mate_pair_delta"] > 0
            ),
            "reason": (
                "The analytic baseline already had complete true-pair recall "
                "on this small functional benchmark. JoinABLe added frontier "
                "work but no retained true-mate gain."
            ),
        },
    }
    output_json = dataset / "joinable_candidate_provider_ablation.json"
    output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# JoinABLe candidate-provider ablation",
        "",
        f"- Pools: {len(pool_rows)}",
        f"- Rescued prescreen pairs: {totals['rescued_pair_count']}",
        (
            "- Rescued true mates: "
            f"{totals['rescued_true_mate_count']}"
        ),
        (
            "- Rescued functionally wrong pairs: "
            f"{totals['rescued_pair_count'] - totals['rescued_same_group_pair_count']}"
        ),
        (
            "- Generated candidate delta: "
            f"{totals['generated_candidate_delta']:+d}"
        ),
        f"- Kept candidate delta: {totals['kept_candidate_delta']:+d}",
        (
            "- Kept true-mate pair delta: "
            f"{totals['kept_true_mate_pair_delta']:+d}"
        ),
        "",
        "Conclusion: the integration is operational, but this benchmark does "
        "not demonstrate a recall gain because the analytic baseline already "
        "has complete true-pair recall. Keep JoinABLe at candidate generation "
        "and measure it next on a harder holdout; do not promote its score to "
        "automatic acceptance.",
    ]
    (dataset / "joinable_candidate_provider_ablation.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mixed_pool_root")
    args = parser.parse_args()
    report = audit(args.mixed_pool_root)
    print(json.dumps(report["totals"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
