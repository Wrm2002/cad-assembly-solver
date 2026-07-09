"""Build deterministic mixed STEP pools with group-membership ground truth."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path


GROUP_LAYOUTS = [
    (2, 3),
    (2, 2, 3),
    (3, 4),
    (2, 4, 5),
    (3, 5),
    (2, 3, 6),
]
DIVERSE_456_LAYOUTS = [
    (4, 4),
    (4, 5),
    (5, 5),
    (4, 6),
    (5, 6),
    (6, 6),
]


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _complete_cases(dataset_root, allowed_case_ids=None):
    by_size = {size: [] for size in range(1, 7)}
    for case in sorted(path for path in Path(dataset_root).iterdir() if path.is_dir()):
        if allowed_case_ids is not None and case.name not in allowed_case_ids:
            continue
        gt_path = case / "gt.json"
        if not gt_path.is_file():
            continue
        gt = _read_json(gt_path)
        size = int(gt.get("group_size", 0))
        part_paths = [case / "step" / name for name in gt.get("parts", [])]
        if size in by_size and len(part_paths) == size and all(
            path.is_file() and path.stat().st_size > 0 for path in part_paths
        ):
            by_size[size].append((case, gt))
    return by_size


def _anonymous_name(pool_id, source_case, source_name):
    payload = f"{pool_id}|{source_case}|{source_name}".encode("utf-8")
    return "part_" + hashlib.sha256(payload).hexdigest()[:12] + ".step"


def _remap_mate(mate, mapping):
    item = dict(mate)
    if "parts" in item:
        item["parts"] = [mapping[name] for name in item["parts"]]
    else:
        item["part_a"] = mapping[item["part_a"]]
        item["part_b"] = mapping[item["part_b"]]
    return item


def build_pools(
    dataset_root,
    output_root,
    *,
    num_pools=12,
    seed=20260702,
    min_parts=5,
    max_parts=12,
    layouts=None,
    allowed_case_ids=None,
    global_unique=False,
    distractor_sizes=(1,),
    add_distractor=True,
):
    dataset_root, output_root = Path(dataset_root).resolve(), Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    by_size = _complete_cases(dataset_root, allowed_case_ids)
    layouts = layouts or GROUP_LAYOUTS
    rng = random.Random(seed)
    results = []
    globally_used_source_cases = set()
    for pool_index in range(1, num_pools + 1):
        pool_id = f"pool_{pool_index:03d}"
        layout = layouts[(pool_index - 1) % len(layouts)]
        if sum(layout) > max_parts:
            raise ValueError(f"layout {layout} exceeds max_parts={max_parts}")
        pool_dir = output_root / pool_id
        parts_dir = pool_dir / "parts"
        parts_dir.mkdir(parents=True, exist_ok=True)
        groups, all_parts, provenance, part_semantics = [], [], {}, {}
        used_source_cases = set()
        for group_index, size in enumerate(layout, start=1):
            if not by_size[size]:
                raise ValueError(f"no complete group_size={size} cases")
            available = [
                item
                for item in by_size[size]
                if item[0].name not in used_source_cases
                and (
                    not global_unique
                    or item[0].name not in globally_used_source_cases
                )
            ]
            if not available:
                raise ValueError(f"not enough distinct group_size={size} cases")
            source_case, gt = rng.choice(available)
            used_source_cases.add(source_case.name)
            globally_used_source_cases.add(source_case.name)
            mapping = {}
            for source_name in gt["parts"]:
                target_name = _anonymous_name(pool_id, source_case.name, source_name)
                mapping[source_name] = target_name
                shutil.copy2(source_case / "step" / source_name, parts_dir / target_name)
                all_parts.append(target_name)
                provenance[target_name] = {
                    "source_case": source_case.name,
                    "source_part": source_name,
                    "role": "true_group_member",
                }
                source_semantics = gt.get("part_semantics", {}).get(
                    source_name, {}
                )
                if source_semantics:
                    part_semantics[target_name] = {
                        **source_semantics,
                        "part_id": target_name,
                    }
            groups.append({
                "group_id": f"G{group_index:03d}",
                "parts": sorted(mapping.values()),
                "source_case": source_case.name,
                "template": gt.get("template"),
                "system_class": gt.get("system_class"),
                "true_mates": [
                    _remap_mate(mate, mapping) for mate in gt.get("true_mates", [])
                ],
                "placements": {
                    mapping[name]: placement
                    for name, placement in gt.get("placements", {}).items()
                },
            })

        distractors = []
        target_count = (
            max(min_parts, min(max_parts, sum(layout) + 1))
            if add_distractor
            else sum(layout)
        )
        while len(all_parts) < target_count:
            available = [
                item
                for size in distractor_sizes
                for item in by_size[size]
                if item[0].name not in used_source_cases
                and (
                    not global_unique
                    or item[0].name not in globally_used_source_cases
                )
            ]
            if not available:
                raise ValueError("no unused source case available for distractor")
            source_case, gt = rng.choice(available)
            used_source_cases.add(source_case.name)
            globally_used_source_cases.add(source_case.name)
            source_name = gt["parts"][0]
            target_name = _anonymous_name(pool_id, source_case.name, source_name)
            if target_name in provenance:
                continue
            shutil.copy2(source_case / "step" / source_name, parts_dir / target_name)
            all_parts.append(target_name)
            distractors.append(target_name)
            provenance[target_name] = {
                "source_case": source_case.name,
                "source_part": source_name,
                "role": "distractor",
            }
            source_semantics = gt.get("part_semantics", {}).get(
                source_name, {}
            )
            if source_semantics:
                part_semantics[target_name] = {
                    **source_semantics,
                    "part_id": target_name,
                }
        rng.shuffle(all_parts)
        pool_gt = {
            "schema_version": "1.0.0",
            "pool_id": pool_id,
            "seed": seed,
            "units": {"length": "mm", "angle": "degree"},
            "parts": all_parts,
            "true_groups": groups,
            "distractors": sorted(distractors),
            "provenance": provenance,
            "naming_policy": "content-independent deterministic hash; no group label in filename",
        }
        (pool_dir / "pool_gt.json").write_text(
            json.dumps(pool_gt, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (pool_dir / "part_semantics.json").write_text(
            json.dumps(
                part_semantics, indent=2, ensure_ascii=False
            )
            + "\n",
            encoding="utf-8",
        )
        results.append({
            "pool_id": pool_id,
            "num_parts": len(all_parts),
            "num_true_groups": len(groups),
            "num_distractors": len(distractors),
            "layout": list(layout),
        })
    report = {
        "schema_version": "1.0.0",
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "seed": seed,
        "num_pools": len(results),
        "pools": results,
    }
    (output_root / "pool_build_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--num-pools", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--min-parts", type=int, default=5)
    parser.add_argument("--max-parts", type=int, default=12)
    parser.add_argument(
        "--profile",
        choices=("legacy", "diverse_456"),
        default="legacy",
    )
    parser.add_argument("--split-manifest")
    parser.add_argument(
        "--split",
        choices=("train", "calibration", "test"),
    )
    parser.add_argument("--global-unique", action="store_true")
    parser.add_argument("--no-distractor", action="store_true")
    args = parser.parse_args()
    allowed_case_ids = None
    if args.split_manifest:
        if not args.split:
            parser.error("--split is required with --split-manifest")
        split_manifest = _read_json(Path(args.split_manifest))
        allowed_case_ids = {
            case_id
            for case_id, item in split_manifest["assignments"].items()
            if item["split"] == args.split
        }
    report = build_pools(
        args.dataset_root,
        args.output_root,
        num_pools=args.num_pools,
        seed=args.seed,
        min_parts=args.min_parts,
        max_parts=args.max_parts,
        layouts=(
            DIVERSE_456_LAYOUTS
            if args.profile == "diverse_456"
            else GROUP_LAYOUTS
        ),
        allowed_case_ids=allowed_case_ids,
        global_unique=args.global_unique,
        distractor_sizes=(
            (4, 5, 6) if args.profile == "diverse_456" else (1,)
        ),
        add_distractor=not args.no_distractor,
    )
    print(f"pools={report['num_pools']}")
    print(Path(args.output_root).resolve())


if __name__ == "__main__":
    main()
