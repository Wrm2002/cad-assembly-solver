"""Convert a minimum of 10 usable Fusion assemblies to the unified schema."""

from __future__ import annotations

import argparse
from pathlib import Path

from fusion360_common import (
    convert_assembly,
    discover_assembly_files,
    write_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument(
        "--output-dir", default="outputs/fusion360_assembly_graphs"
    )
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--maximum-parts", type=int, default=200)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    converted = []
    rejected = []
    for assembly_file in discover_assembly_files(Path(args.input)):
        if len(converted) >= args.limit:
            break
        try:
            graph = convert_assembly(assembly_file)
        except Exception as exc:
            rejected.append({
                "source_path": str(assembly_file.resolve()),
                "failure_reasons": [
                    f"conversion_exception:{type(exc).__name__}:{exc}"
                ],
                "unavailable_fields": ["assembly_graph"],
            })
            continue
        quality = graph["quality"]
        if quality["part_count"] > args.maximum_parts:
            rejected.append({
                "source_path": str(assembly_file.resolve()),
                "failure_reasons": [
                    f"part_count_exceeds_limit:{quality['part_count']}"
                ],
                "unavailable_fields": ["negative_part_pair_edges"],
            })
            continue
        if quality["status"] != "usable":
            rejected.append({
                "source_path": str(assembly_file.resolve()),
                "failure_reasons": (
                    quality["failure_reasons"]
                    + ["fewer_than_two_parts_or_no_positive_pair"]
                ),
                "unavailable_fields": quality["unavailable_fields"],
            })
            continue
        destination = output / f"{graph['assembly_id']}.json"
        write_json(destination, graph)
        converted.append({
            "assembly_id": graph["assembly_id"],
            "output_path": str(destination),
            "part_count": quality["part_count"],
            "positive_pair_count": quality["positive_pair_count"],
            "negative_pair_count": quality["negative_pair_count"],
            "failure_reasons": quality["failure_reasons"],
            "unavailable_fields": quality["unavailable_fields"],
        })
    manifest = {
        "schema_version": "1.0.0",
        "source_dataset": "fusion360_gallery_assembly",
        "requested_count": args.limit,
        "converted_count": len(converted),
        "acceptance_met": len(converted) >= args.limit,
        "converted": converted,
        "rejected": rejected,
        "failure_reasons": (
            [] if len(converted) >= args.limit
            else [f"only_{len(converted)}_usable_assemblies_converted"]
        ),
        "unavailable_fields": sorted({
            field for row in converted
            for field in row["unavailable_fields"]
        }),
    }
    write_json(output / "conversion_manifest.json", manifest)
    print(
        f"Converted {len(converted)}/{args.limit} usable assemblies"
    )
    return 0 if manifest["acceptance_met"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
