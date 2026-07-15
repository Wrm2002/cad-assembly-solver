"""Build compact mapped labels for Fusion-to-STEP JoinABLe adaptation."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audit_official_step_transfer import (
    all_geometry_entities,
    map_entity_to_occt,
)
from cad_assembly_agent.tools.joinable_interface_predictor.pretrained_joinable_predictor import (
    validate_graph,
)


DEFAULT_SUBSET = Path(
    r"D:\Model_match_public_data\fusion360_joint\domain_adapt_300"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def mapped_indices(mapping: dict[str, Any]) -> set[int]:
    if mapping.get("status") not in {"mapped", "ambiguous_equivalent"}:
        return set()
    return {
        int(match["joinable_node_index"])
        for match in mapping.get("matches", [])
    }


def cartesian_pairs(a: set[int], b: set[int]) -> set[tuple[int, int]]:
    return {(left, right) for left in a for right in b}


def source_design_id(body_id: str) -> str:
    tokens = body_id.split("_")
    return "_".join(tokens[:2]) if len(tokens) >= 2 else body_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-root", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--max-node-count", type=int, default=950)
    parser.add_argument("--max-candidate-pairs", type=int, default=100000)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output path. Defaults to "
            "<subset-root>/domain_adaptation_manifest.json."
        ),
    )
    args = parser.parse_args()
    output_path = (
        args.output
        if args.output is not None
        else args.subset_root / "domain_adaptation_manifest.json"
    )

    subset = read_json(args.subset_root / "subset_split.json")
    raw_root = args.subset_root / "j1.0.0" / "joint"
    graph_root = args.subset_root / "occt_checkpoint_graphs"
    joint_designs: dict[str, set[str]] = {}
    for joint_ids in subset["splits"].values():
        for joint_id in joint_ids:
            joint = read_json(raw_root / f"{joint_id}.json")
            joint_designs[joint_id] = {
                source_design_id(str(joint["body_one"])),
                source_design_id(str(joint["body_two"])),
            }
    allowed_ids: dict[str, set[str]] = {
        "train": set(),
        "validation": set(),
        "test": set(subset["splits"]["test"]),
    }
    test_designs = set().union(
        *(joint_designs[joint_id] for joint_id in allowed_ids["test"])
    )
    for joint_id in subset["splits"]["validation"]:
        if joint_designs[joint_id].isdisjoint(test_designs):
            allowed_ids["validation"].add(joint_id)
    validation_designs = set().union(
        *(
            joint_designs[joint_id]
            for joint_id in allowed_ids["validation"]
        )
    )
    heldout_designs = test_designs | validation_designs
    for joint_id in subset["splits"]["train"]:
        if joint_designs[joint_id].isdisjoint(heldout_designs):
            allowed_ids["train"].add(joint_id)
    split_rows: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    failures = []
    exact_side_attempts = 0
    exact_side_mapped = 0
    all_mapping_statuses: Counter[str] = Counter()

    for split_name, joint_ids in subset["splits"].items():
        for joint_id in joint_ids:
            joint_path = raw_root / f"{joint_id}.json"
            if not joint_path.is_file():
                failures.append(
                    {
                        "joint_set": joint_id,
                        "reason": "joint_json_missing",
                    }
                )
                continue
            joint_set = read_json(joint_path)
            body_a = str(joint_set["body_one"])
            body_b = str(joint_set["body_two"])
            design_ids = sorted(joint_designs[joint_id])
            if joint_id not in allowed_ids[split_name]:
                split_rows[split_name].append(
                    {
                        "sample_id": joint_id,
                        "body_a": body_a,
                        "body_b": body_b,
                        "source_design_ids": design_ids,
                        "status": "excluded_source_design_leakage",
                        "exact_positive_pairs": [],
                        "equivalent_positive_pairs": [],
                        "failure_reasons": [
                            "source_design_present_in_higher_priority_heldout_split"
                        ],
                        "unavailable_fields": [
                            "training_or_evaluation_eligibility"
                        ],
                    }
                )
                continue
            source_graphs = {
                body_a: read_json(raw_root / f"{body_a}.json"),
                body_b: read_json(raw_root / f"{body_b}.json"),
            }
            step_graph_paths = {
                body_a: graph_root / f"{body_a}.brep_graph.json",
                body_b: graph_root / f"{body_b}.brep_graph.json",
            }
            if not all(path.is_file() for path in step_graph_paths.values()):
                failures.append(
                    {
                        "joint_set": joint_id,
                        "reason": "occt_graph_missing",
                    }
                )
                continue
            step_graphs = {
                body: read_json(path)
                for body, path in step_graph_paths.items()
            }
            try:
                for body, path in step_graph_paths.items():
                    validate_graph(step_graphs[body], path)
            except Exception as exc:
                failures.append(
                    {
                        "joint_set": joint_id,
                        "reason": f"graph_invalid:{type(exc).__name__}:{exc}",
                    }
                )
                continue

            node_count_a = len(step_graphs[body_a]["nodes"])
            node_count_b = len(step_graphs[body_b]["nodes"])
            combined_nodes = node_count_a + node_count_b
            candidate_pair_count = node_count_a * node_count_b
            exact_pairs: set[tuple[int, int]] = set()
            equivalent_pairs: set[tuple[int, int]] = set()
            mapping_records = []

            for joint_index, joint in enumerate(joint_set["joints"]):
                exact_by_body = {body_a: set(), body_b: set()}
                equivalent_by_body = {body_a: set(), body_b: set()}
                for geometry_key in (
                    "geometry_or_origin_one",
                    "geometry_or_origin_two",
                ):
                    geometry = joint[geometry_key]
                    for entity_index, entity in enumerate(
                        all_geometry_entities(geometry)
                    ):
                        body = str(entity["body"])
                        if body not in source_graphs:
                            continue
                        mapping = map_entity_to_occt(
                            entity,
                            source_graphs[body],
                            step_graphs[body],
                        )
                        all_mapping_statuses[mapping["status"]] += 1
                        mapping_records.append(
                            {
                                "joint_index": joint_index,
                                "body": body,
                                "is_exact": entity_index == 0,
                                "source_type": entity["type"],
                                "source_index": int(entity["index"]),
                                "status": mapping["status"],
                                "reason": mapping.get("reason"),
                                "mapped_indices": sorted(
                                    mapped_indices(mapping)
                                ),
                            }
                        )
                        indices = mapped_indices(mapping)
                        if entity_index == 0:
                            exact_side_attempts += 1
                            if indices:
                                exact_side_mapped += 1
                            exact_by_body[body].update(indices)
                        equivalent_by_body[body].update(indices)

                exact_pairs.update(
                    cartesian_pairs(
                        exact_by_body[body_a], exact_by_body[body_b]
                    )
                )
                equivalent_pairs.update(
                    cartesian_pairs(
                        equivalent_by_body[body_a],
                        equivalent_by_body[body_b],
                    )
                )

            if combined_nodes > args.max_node_count:
                status = "unsupported_official_node_limit"
            elif candidate_pair_count > args.max_candidate_pairs:
                status = "unsupported_local_candidate_pair_limit"
            elif exact_pairs:
                status = "training_and_evaluation"
            elif equivalent_pairs:
                status = "evaluation_only_no_exact_mapping"
            else:
                status = "unusable_no_mapped_truth"

            split_rows[split_name].append(
                {
                    "sample_id": joint_id,
                    "source_joint_json": str(joint_path),
                    "body_a": body_a,
                    "body_b": body_b,
                    "source_design_ids": design_ids,
                    "step_graph_a": str(step_graph_paths[body_a]),
                    "step_graph_b": str(step_graph_paths[body_b]),
                    "node_count_a": node_count_a,
                    "node_count_b": node_count_b,
                    "combined_node_count": combined_nodes,
                    "candidate_pair_count": candidate_pair_count,
                    "joint_count": len(joint_set["joints"]),
                    "exact_positive_pairs": [
                        list(pair) for pair in sorted(exact_pairs)
                    ],
                    "equivalent_positive_pairs": [
                        list(pair) for pair in sorted(equivalent_pairs)
                    ],
                    "status": status,
                    "mapping_records": mapping_records,
                    "failure_reasons": (
                        []
                        if status in {
                            "training_and_evaluation",
                            "evaluation_only_no_exact_mapping",
                        }
                        else [status]
                    ),
                    "unavailable_fields": [
                        "functional_assembly_validity",
                    ],
                }
            )

    status_distribution = {
        split_name: dict(Counter(row["status"] for row in rows))
        for split_name, rows in split_rows.items()
    }
    usable_counts = {
        split_name: sum(
            row["status"] == "training_and_evaluation" for row in rows
        )
        for split_name, rows in split_rows.items()
    }
    included_designs = {
        split_name: {
            design
            for row in rows
            if row["status"] != "excluded_source_design_leakage"
            for design in row.get("source_design_ids", [])
        }
        for split_name, rows in split_rows.items()
    }
    overlap_after_filter = {
        "train_validation": len(
            included_designs["train"] & included_designs["validation"]
        ),
        "train_test": len(
            included_designs["train"] & included_designs["test"]
        ),
        "validation_test": len(
            included_designs["validation"] & included_designs["test"]
        ),
    }
    report = {
        "schema_version": "1.0.0",
        "purpose": (
            "Compact STEP graph and mapped designer-entity labels for "
            "JoinABLe domain adaptation"
        ),
        "subset_root": str(args.subset_root),
        "max_node_count": args.max_node_count,
        "max_candidate_pairs": args.max_candidate_pairs,
        "splits": split_rows,
        "summary": {
            "requested_counts": {
                key: len(value) for key, value in subset["splits"].items()
            },
            "manifest_counts": {
                key: len(value) for key, value in split_rows.items()
            },
            "training_usable_counts": usable_counts,
            "status_distribution": status_distribution,
            "split_policy": (
                "test priority, then validation; train excludes any source "
                "design present in heldout splits"
            ),
            "source_design_overlap_after_filter": overlap_after_filter,
            "exact_entity_side_mapping_rate": (
                exact_side_mapped / exact_side_attempts
                if exact_side_attempts
                else None
            ),
            "exact_entity_side_attempt_count": exact_side_attempts,
            "exact_entity_side_mapped_count": exact_side_mapped,
            "mapping_status_distribution": dict(all_mapping_statuses),
            "external_failure_count": len(failures),
        },
        "external_failures": failures,
        "failure_reasons": [
            row["reason"] for row in failures
        ],
        "unavailable_fields": [
            "functional_assembly_validity",
            "permanent_topology_ids_across_reexport",
        ],
    }
    write_json(output_path, report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
