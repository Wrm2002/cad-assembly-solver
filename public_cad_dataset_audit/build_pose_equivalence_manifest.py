"""Extract audit-only Pose-equivalence seeds from leakage-safe Fusion records.

The Joint Dataset already records equivalent B-Rep entities and parametric
free DOFs.  Earlier training flattened each supervision into one rigid matrix,
which incorrectly makes some valid symmetric/sliding states look negative.
This script preserves those seeds in a separate manifest.  The manifest is
for the next geometry builder only; paths and auxiliary joint types are never
model features.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Iterator


def _rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _entity(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "entity_kind": value.get("entity_kind"),
        "topology_index": int(value.get("topology_index", -1)),
        "geometry_type": value.get("geometry_type"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pure_brep_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    counters: Counter[str] = Counter()
    dof_counts: Counter[str] = Counter()
    manifest_path = args.output_dir / "pose_equivalence_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as output:
        for split in ("train", "dev", "test"):
            source = args.pure_brep_dir / f"fusion360_pure_brep_{split}.jsonl"
            for record in _rows(source):
                counters[f"records_{split}"] += 1
                storage = record.get("storage") or {}
                for index, item in enumerate(record.get("supervision") or []):
                    counters["supervision_total"] += 1
                    left = [_entity(row) for row in item.get("equivalent_entities_a") or []]
                    right = [_entity(row) for row in item.get("equivalent_entities_b") or []]
                    dof = [int(value) for value in item.get("free_dof_mask") or []]
                    dof_counts[str(dof)] += 1
                    if left: counters["has_equivalent_a"] += 1
                    if right: counters["has_equivalent_b"] += 1
                    if left and right: counters["has_equivalent_both"] += 1
                    if abs(float(item.get("offset") or 0.0)) > 1e-9: counters["nonzero_offset"] += 1
                    if abs(float(item.get("angle") or 0.0)) > 1e-9: counters["nonzero_angle"] += 1
                    if bool(item.get("is_flipped")): counters["flipped"] += 1
                    # Storage is deliberately separated from model_input.  It
                    # gives a later geometry worker access to B-Rep patches
                    # required to instantiate equivalent frames.
                    row = {
                        "schema_version": "fusion_pose_equivalence_seed.v1",
                        "split": split,
                        "record_id": record.get("record_id"),
                        "supervision_index": index,
                        "storage": {
                            "body_graph_a": storage.get("body_graph_a"),
                            "body_graph_b": storage.get("body_graph_b"),
                            "split_group_hash": storage.get("split_group_hash"),
                        },
                        "primary_entities": [_entity(item.get("entity_a") or {}), _entity(item.get("entity_b") or {})],
                        "equivalent_entities": {"a": left, "b": right},
                        "relative_pose": item.get("relative_pose"),
                        "free_dof_mask": dof,
                        "parametric_seed": {
                            "offset": float(item.get("offset") or 0.0),
                            "angle": float(item.get("angle") or 0.0),
                            "is_flipped": bool(item.get("is_flipped")),
                        },
                        "model_input_policy": "No storage, ID, auxiliary type or text field may enter tensors.",
                    }
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
    total = max(1, counters["supervision_total"])
    report = {
        "schema_version": "fusion_pose_equivalence_manifest_audit.v1",
        "manifest": str(manifest_path.resolve()),
        "counts": dict(counters),
        "rates": {
            "equivalent_a": counters["has_equivalent_a"] / total,
            "equivalent_b": counters["has_equivalent_b"] / total,
            "equivalent_both": counters["has_equivalent_both"] / total,
            "nonzero_offset": counters["nonzero_offset"] / total,
            "nonzero_angle": counters["nonzero_angle"] / total,
            "flipped": counters["flipped"] / total,
        },
        "free_dof_counts": dict(dof_counts),
        "next_builder_contract": [
            "Instantiate alternate local frames only from primary/equivalent B-Rep entities.",
            "Treat states differing only in declared free DOFs as equivalence candidates, not automatic negatives.",
            "Measure every generated candidate with local B-Rep contact and, where available, OCCT before assigning a hard-negative label.",
            "Never use auxiliary_joint_type, record ID, graph path or split hash as a model tensor.",
        ],
    }
    (args.output_dir / "pose_equivalence_manifest_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
