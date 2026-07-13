"""Build positive/negative part-pair classification samples from graph JSON."""

from __future__ import annotations

import argparse
from pathlib import Path

from fusion360_common import load_json, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graph_dir")
    parser.add_argument(
        "--output", default="outputs/pair_dataset_manifest.json"
    )
    args = parser.parse_args()
    graph_dir = Path(args.graph_dir).resolve()
    samples = []
    failures = []
    assembly_summaries = []
    graph_files = sorted(
        path for path in graph_dir.glob("*.json")
        if path.name != "conversion_manifest.json"
    )
    for graph_file in graph_files:
        try:
            graph = load_json(graph_file)
        except Exception as exc:
            failures.append(
                f"{graph_file}:load_failed:{type(exc).__name__}:{exc}"
            )
            continue
        assembly_id = graph.get("assembly_id")
        parts = {
            part["part_id"]: part for part in graph.get("parts", [])
        }
        positive_count = negative_count = 0
        for label, key in (
            (1, "positive_part_pair_edges"),
            (0, "negative_part_pair_edges"),
        ):
            for edge in graph.get(key, []):
                pair = edge.get("part_pair") or []
                if len(pair) != 2 or any(
                    part_id not in parts for part_id in pair
                ):
                    failures.append(
                        f"{assembly_id}:{edge.get('edge_id')}:"
                        "invalid_part_pair_reference"
                    )
                    continue
                sample = {
                    "sample_id": (
                        f"{assembly_id}:{edge.get('edge_id')}"
                    ),
                    "assembly_id": assembly_id,
                    "part_pair": pair,
                    "part_geometry_paths": [
                        parts[part_id]["geometry"].get("path")
                        for part_id in pair
                    ],
                    "label": label,
                    "relation_types": (
                        edge.get("relation_types", [])
                        if label else ["none_observed"]
                    ),
                    "source_dataset": graph.get("source_dataset"),
                    "source_graph_path": str(graph_file),
                    "failure_reasons": edge.get(
                        "failure_reasons", []
                    ),
                    "unavailable_fields": edge.get(
                        "unavailable_fields", []
                    ),
                }
                samples.append(sample)
                if label:
                    positive_count += 1
                else:
                    negative_count += 1
        assembly_summaries.append({
            "assembly_id": assembly_id,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "failure_reasons": [],
            "unavailable_fields": [],
        })
    positives = sum(sample["label"] == 1 for sample in samples)
    negatives = len(samples) - positives
    manifest = {
        "schema_version": "1.0.0",
        "task": "part_pair_binary_classification",
        "positive_definition": (
            "The pair has at least one joint, as-built joint, or contact."
        ),
        "negative_definition": (
            "The two parts occur in the same assembly but have no recorded "
            "joint, as-built joint, or contact."
        ),
        "negative_label_caveat": (
            "This is a closed-world annotation negative, not proof that "
            "the parts are mechanically incompatible."
        ),
        "assembly_count": len(assembly_summaries),
        "sample_count": len(samples),
        "positive_count": positives,
        "negative_count": negatives,
        "can_build_classification_dataset": (
            positives > 0 and negatives > 0
        ),
        "split_requirement": (
            "Split by assembly/source document before sampling to prevent "
            "part and author leakage."
        ),
        "assemblies": assembly_summaries,
        "samples": samples,
        "failure_reasons": failures,
        "unavailable_fields": [
            "proof_that_negative_pairs_cannot_mate",
            "training_split_not_created",
        ],
    }
    write_json(Path(args.output), manifest)
    print(
        f"Pair samples: {len(samples)} "
        f"(positive={positives}, negative={negatives})"
    )
    return (
        0 if manifest["can_build_classification_dataset"]
        and not failures else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
