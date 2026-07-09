"""Audit truth-interface recall before and after candidate pruning.

This module is evaluation-only. Ground truth is read solely to explain where a
known interface was lost; it is never exposed to inference or ranking code.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


FIELDS = [
    "pool_id",
    "true_group_id",
    "parts",
    "group_size",
    "candidate_type",
    "generated_or_not",
    "pruned_or_not",
    "pruning_reason",
    "missing_reason_guess",
    "geometry_features_available",
    "required_interface_type",
]


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _pair(parts: Iterable[str]) -> tuple[str, str]:
    values = sorted(str(value) for value in parts)
    if len(values) != 2:
        raise ValueError(f"expected two parts, got {values}")
    return values[0], values[1]


def _candidate_key(item: dict[str, Any]) -> tuple[tuple[str, str], str]:
    return _pair(item["parts"]), str(item["candidate_type"])


def _features_available(
    interface_type: str,
    parts: tuple[str, str],
    feature_map: dict[str, dict[str, Any]],
) -> str:
    summaries = []
    for part in parts:
        feature = feature_map.get(part, {})
        if interface_type in {"clearance", "coaxial", "pocket_mate"}:
            count = len(feature.get("cylindrical_faces", []))
            summaries.append(f"{part}:cylinders={count}")
        elif interface_type in {"planar_mate", "planar_align"}:
            count = len(feature.get("planar_faces", []))
            summaries.append(f"{part}:planes={count}")
        else:
            summaries.append(f"{part}:feature_type_unknown")
    return ";".join(summaries)


def _missing_guess(
    interface_type: str,
    generated: bool,
    available: str,
    removed: list[dict[str, Any]],
) -> str:
    if generated and removed:
        return "candidate_generated_then_pruned"
    if generated:
        return ""
    counts = [
        int(token.rsplit("=", 1)[1])
        for token in available.split(";")
        if "=" in token and token.rsplit("=", 1)[1].isdigit()
    ]
    if counts and min(counts) == 0:
        return f"required_{interface_type}_feature_not_extracted"
    return "detailed_matcher_or_gt_interface_semantics_mismatch"


def audit_pool(pool_dir: str | Path) -> list[dict[str, Any]]:
    pool = Path(pool_dir).resolve()
    gt = _load(pool / "pool_gt.json")
    generated = _load(pool / "index" / "geometry_candidates.json")
    kept = _load(pool / "index" / "pruned_candidates.json")
    removed = _load(pool / "index" / "removed_candidates.json")
    features = _load(pool / "index" / "part_features.json")
    feature_map = {item["part_id"]: item for item in features}

    generated_map: dict[
        tuple[tuple[str, str], str], list[dict[str, Any]]
    ] = defaultdict(list)
    kept_map: dict[
        tuple[tuple[str, str], str], list[dict[str, Any]]
    ] = defaultdict(list)
    removed_map: dict[
        tuple[tuple[str, str], str], list[dict[str, Any]]
    ] = defaultdict(list)
    for item in generated:
        generated_map[_candidate_key(item)].append(item)
    for item in kept:
        kept_map[_candidate_key(item)].append(item)
    for item in removed:
        removed_map[_candidate_key(item)].append(item)

    rows = []
    for group in gt.get("true_groups", []):
        group_size = len(group["parts"])
        for mate in group.get("true_mates", []):
            parts = _pair((mate["part_a"], mate["part_b"]))
            interface_type = str(mate["type"])
            key = (parts, interface_type)
            generated_items = generated_map.get(key, [])
            kept_items = kept_map.get(key, [])
            removed_items = removed_map.get(key, [])
            available = _features_available(
                interface_type, parts, feature_map
            )
            reasons = sorted(
                {
                    str(
                        item.get("audit_reason", {}).get(
                            "removal_reason", "unknown"
                        )
                    )
                    for item in removed_items
                }
            )
            rows.append(
                {
                    "pool_id": pool.name,
                    "true_group_id": group["group_id"],
                    "parts": "|".join(parts),
                    "group_size": group_size,
                    "candidate_type": interface_type,
                    "generated_or_not": bool(generated_items),
                    "pruned_or_not": bool(
                        generated_items and not kept_items
                    ),
                    "pruning_reason": "|".join(reasons),
                    "missing_reason_guess": _missing_guess(
                        interface_type,
                        bool(generated_items),
                        available,
                        removed_items,
                    ),
                    "geometry_features_available": available,
                    "required_interface_type": interface_type,
                }
            )
    return rows


def _summary(
    rows: list[dict[str, Any]], key: str
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    result = {}
    for value, items in sorted(groups.items()):
        generated = sum(bool(item["generated_or_not"]) for item in items)
        kept = sum(
            bool(item["generated_or_not"]) and not item["pruned_or_not"]
            for item in items
        )
        total = len(items)
        result[value] = {
            "truth_interfaces": total,
            "generated": generated,
            "kept_after_pruning": kept,
            "generated_recall": generated / total if total else None,
            "post_pruning_recall": kept / total if total else None,
        }
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def run_audit(
    root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    root = Path(root).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    pools = sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "pool_gt.json").is_file()
    )
    rows = [row for pool in pools for row in audit_pool(pool)]
    missed = [row for row in rows if not row["generated_or_not"]]
    pruned = [
        row
        for row in rows
        if row["generated_or_not"] and row["pruned_or_not"]
    ]
    by_type = _summary(rows, "candidate_type")
    by_size = _summary(rows, "group_size")
    _write_csv(output / "missed_true_candidates.csv", missed)
    _write_csv(output / "pruned_true_candidates.csv", pruned)
    (output / "candidate_recall_by_type.json").write_text(
        json.dumps(by_type, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output / "candidate_recall_by_group_size.json").write_text(
        json.dumps(by_size, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    total = len(rows)
    generated = total - len(missed)
    kept = total - len(missed) - len(pruned)
    report = "\n".join(
        [
            "# Candidate Recall Audit",
            "",
            "Ground truth is used only by this evaluation report.",
            "",
            f"- Pools: {len(pools)}",
            f"- Truth interfaces: {total}",
            f"- Generated: {generated}/{total} "
            f"({generated / total:.2%})" if total else "- Generated: n/a",
            f"- Kept after pruning: {kept}/{total} "
            f"({kept / total:.2%})" if total else "- Kept: n/a",
            f"- Missing before pruning: {len(missed)}",
            f"- Removed by pruning: {len(pruned)}",
            "",
            "Candidate generation should favor recall. Strictness belongs "
            "in the downstream acceptance gate.",
            "",
        ]
    )
    (output / "candidate_recall_audit.md").write_text(
        report, encoding="utf-8"
    )
    return {
        "pools": len(pools),
        "truth_interfaces": total,
        "generated": generated,
        "kept_after_pruning": kept,
        "generated_recall": generated / total if total else None,
        "post_pruning_recall": kept / total if total else None,
        "missed": len(missed),
        "pruned": len(pruned),
        "by_type": by_type,
        "by_group_size": by_size,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default="mixed_pools_v1")
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parent / "data" / "results"),
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run_audit(args.root, args.output),
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
