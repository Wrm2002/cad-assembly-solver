"""Build compressed interface proxies for the 14 anonymized real parts."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Any

from feature_proxy import build_proxy, load_json, sha256_file, write_json


HERE = Path(__file__).resolve().parent


def projected_comparisons(rows: list[dict[str, Any]]) -> int:
    return sum(
        first["planes"] * second["planes"]
        + first["cylinders"] * second["cylinders"]
        for first, second in itertools.combinations(rows, 2)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validation-root",
        default=str(HERE / "real_validation_20260703"),
    )
    parser.add_argument(
        "--output-root",
        default=str(HERE / "real_feature_proxies_20260703"),
    )
    args = parser.parse_args()
    validation_root = Path(args.validation_root).resolve()
    output_root = Path(args.output_root).resolve()
    feature_dir = (
        validation_root / "mixed_real_pool" / "index" / "parts"
    )
    private_map = load_json(
        validation_root / "private_source_map.json"
    )
    full_rows = []
    proxy_rows = []
    audit_rows = []
    source_hashes_unchanged = True
    for feature_path in sorted(feature_dir.glob("part_*.json")):
        full = load_json(feature_path)
        part_id = full["part_id"]
        source = Path(private_map[part_id]["source_path"])
        source_unchanged = (
            sha256_file(source) == private_map[part_id]["sha256"]
        )
        source_hashes_unchanged &= source_unchanged
        proxy = build_proxy(full, source_path=feature_path)
        proxy_path = output_root / "parts" / f"{part_id}.proxy.json"
        write_json(proxy_path, proxy)
        coarse_path = output_root / "coarse" / f"{part_id}.coarse.json"
        write_json(coarse_path, {
            "schema_version": "1.0.0",
            "part_id": part_id,
            "representation": "coarse interface-family index",
            "exact_proxy_file": str(proxy_path.resolve()),
            "exact_proxy_sha256": sha256_file(proxy_path),
            "bbox": proxy.get("bbox"),
            "volume": proxy.get("volume"),
            "geometric_class": proxy.get("geometric_class"),
            "plane_families": proxy["plane_families"],
            "cylinder_families": proxy["cylinder_families"],
            "compression": proxy["compression"],
            "refinement_policy": (
                "A coarse family hit must be expanded through the exact "
                "proxy and then checked against the original STEP."
            ),
        })
        compression = proxy["compression"]
        full_rows.append({
            "part_id": part_id,
            "case_id": private_map[part_id]["case_id"],
            "planes": compression["full_plane_count"],
            "cylinders": compression["full_cylinder_count"],
        })
        proxy_rows.append({
            "part_id": part_id,
            "case_id": private_map[part_id]["case_id"],
            "planes": compression["proxy_plane_count"],
            "cylinders": compression["proxy_cylinder_count"],
        })
        coarse_plane_count = compression[
            "coarse_plane_family_count"
        ]
        coarse_cylinder_count = compression[
            "coarse_cylinder_family_count"
        ]
        audit_rows.append({
            "case_id": private_map[part_id]["case_id"],
            "part_id": part_id,
            "full_planes": compression["full_plane_count"],
            "proxy_planes": compression["proxy_plane_count"],
            "full_cylinders": compression["full_cylinder_count"],
            "proxy_cylinders": compression[
                "proxy_cylinder_count"
            ],
            "full_holes": compression["full_hole_count"],
            "proxy_holes": compression["proxy_hole_count"],
            "coarse_plane_families": coarse_plane_count,
            "coarse_cylinder_families": coarse_cylinder_count,
            "matching_feature_reduction_fraction": compression[
                "matching_feature_reduction_fraction"
            ],
            "all_members_accounted_for": compression[
                "all_members_accounted_for"
            ],
            "source_hash_unchanged": source_unchanged,
            "exact_proxy_bytes": proxy_path.stat().st_size,
            "coarse_index_bytes": coarse_path.stat().st_size,
        })
        print(
            f"{part_id}: "
            f"{compression['full_plane_count'] + compression['full_cylinder_count']:,}"
            " -> "
            f"{compression['proxy_plane_count'] + compression['proxy_cylinder_count']:,}",
            flush=True,
        )
    coarse_rows = []
    for audit in audit_rows:
        coarse_rows.append({
            "part_id": audit["part_id"],
            "case_id": audit["case_id"],
            "planes": audit["coarse_plane_families"],
            "cylinders": audit["coarse_cylinder_families"],
        })
    full_mixed = projected_comparisons(full_rows)
    proxy_mixed = projected_comparisons(proxy_rows)
    coarse_mixed = projected_comparisons(coarse_rows)
    by_case = {}
    for case_id in sorted({row["case_id"] for row in full_rows}):
        full_case = [
            row for row in full_rows if row["case_id"] == case_id
        ]
        proxy_case = [
            row for row in proxy_rows if row["case_id"] == case_id
        ]
        coarse_case = [
            row for row in coarse_rows if row["case_id"] == case_id
        ]
        full_value = projected_comparisons(full_case)
        proxy_value = projected_comparisons(proxy_case)
        coarse_value = projected_comparisons(coarse_case)
        by_case[case_id] = {
            "full_projected_comparisons": full_value,
            "proxy_projected_comparisons": proxy_value,
            "coarse_family_projected_comparisons": coarse_value,
            "reduction_fraction": (
                1.0 - proxy_value / max(full_value, 1)
            ),
        }
    summary = {
        "schema_version": "1.0.0",
        "representation": "reversible coarse interface proxy",
        "part_count": len(audit_rows),
        "all_source_hashes_unchanged": source_hashes_unchanged,
        "all_feature_members_accounted_for": all(
            row["all_members_accounted_for"] for row in audit_rows
        ),
        "full_mixed_pool_projected_comparisons": full_mixed,
        "proxy_mixed_pool_projected_comparisons": proxy_mixed,
        "coarse_family_mixed_pool_projected_comparisons": (
            coarse_mixed
        ),
        "mixed_pool_reduction_fraction": (
            1.0 - proxy_mixed / max(full_mixed, 1)
        ),
        "coarse_family_mixed_pool_reduction_fraction": (
            1.0 - coarse_mixed / max(full_mixed, 1)
        ),
        "by_case": by_case,
        "geometry_policy": (
            "Original STEP/STP files remain unchanged and are retained for "
            "full-resolution refinement and exact collision validation."
        ),
        "results": audit_rows,
    }
    write_json(output_root / "proxy_summary.json", summary)
    with (output_root / "proxy_results.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(audit_rows[0])
        )
        writer.writeheader()
        writer.writerows(audit_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if (
        not summary["all_source_hashes_unchanged"]
        or not summary["all_feature_members_accounted_for"]
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
