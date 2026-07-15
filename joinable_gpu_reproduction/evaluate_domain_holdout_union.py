"""Evaluate analytic, pretrained, and union ranking on a design-disjoint STEP test split."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Batch

from joinable_compat import (
    DEFAULT_CHECKPOINT,
    batch_to_device,
    build_model,
    load_checkpoint,
    write_json,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from cad_assembly_agent.tools.joinable_interface_predictor.pretrained_joinable_predictor import (  # noqa: E402
    body_to_data,
    make_joint_graph,
    validate_graph,
)
from cad_assembly_agent.tools.joinable_interface_predictor.rule_interface_predictor import (  # noqa: E402
    score_pair,
)


DEFAULT_MANIFEST = Path(
    r"D:\Model_match_public_data\fusion360_joint\domain_adapt_2600"
    r"\domain_adaptation_manifest.json"
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def descriptor(node: dict[str, Any]) -> dict[str, Any]:
    geometry_type = (
        node.get("surface_type")
        if node["entity_type"] == "face"
        else node.get("curve_type")
    ) or "unknown"
    measure = float(
        node.get("area", 0.0)
        if node["entity_type"] == "face"
        else node.get("length", 0.0)
    )
    radius = node.get("radius")
    radius = float(radius) if radius not in (None, 0) else None
    characteristic_size = (
        radius
        if radius and radius > 0
        else (
            math.sqrt(max(measure, 1e-12))
            if node["entity_type"] == "face"
            else max(measure, 1e-12)
        )
    )
    return {
        "entity_id": node["node_id"],
        "entity_type": node["entity_type"],
        "topology_index": int(node["occt_topology_index"]),
        "geometry_type": geometry_type,
        "measure": measure,
        "characteristic_size": characteristic_size,
        "radius": radius,
        "salience": math.log1p(max(measure, 0.0)),
    }


def first_rank(order: list[int], truth: set[tuple[int, int]], n2: int) -> int | None:
    for rank, flat_index in enumerate(order, 1):
        if (flat_index // n2, flat_index % n2) in truth:
            return rank
    return None


def summarize(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    result: dict[str, Any] = {"evaluable_count": len(rows)}
    for k in (1, 5, 10, 20):
        analytic = sum(
            row[f"analytic_{prefix}_rank"] is not None
            and row[f"analytic_{prefix}_rank"] <= k
            for row in rows
        )
        joinable = sum(
            row[f"joinable_{prefix}_rank"] is not None
            and row[f"joinable_{prefix}_rank"] <= k
            for row in rows
        )
        union = sum(
            (
                row[f"analytic_{prefix}_rank"] is not None
                and row[f"analytic_{prefix}_rank"] <= k
            )
            or (
                row[f"joinable_{prefix}_rank"] is not None
                and row[f"joinable_{prefix}_rank"] <= k
            )
            for row in rows
        )
        result[f"analytic_top_{k}_recall"] = analytic / len(rows) if rows else None
        result[f"joinable_top_{k}_recall"] = joinable / len(rows) if rows else None
        result[f"union_top_{k}_plus_{k}_recall"] = union / len(rows) if rows else None
        result[f"union_rescue_count_at_{k}"] = union - analytic
        result[f"analytic_only_hit_count_at_{k}"] = sum(
            row[f"analytic_{prefix}_rank"] is not None
            and row[f"analytic_{prefix}_rank"] <= k
            and not (
                row[f"joinable_{prefix}_rank"] is not None
                and row[f"joinable_{prefix}_rank"] <= k
            )
            for row in rows
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--split", default="test", choices=("validation", "test"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    manifest = read_json(args.manifest)
    eligible = [
        row
        for row in manifest["splits"][args.split]
        if row["status"] in {
            "training_and_evaluation",
            "evaluation_only_no_exact_mapping",
        }
        and row.get("equivalent_positive_pairs")
    ]
    if args.limit > 0:
        eligible = eligible[: args.limit]
    checkpoint, official_args = load_checkpoint(args.checkpoint)
    model = build_model(checkpoint, official_args).to(device).eval()
    rows = []
    failures = []
    with torch.inference_mode():
        for index, source in enumerate(eligible, 1):
            try:
                graph_a_json = read_json(Path(source["step_graph_a"]))
                graph_b_json = read_json(Path(source["step_graph_b"]))
                validate_graph(graph_a_json, Path(source["step_graph_a"]))
                validate_graph(graph_b_json, Path(source["step_graph_b"]))
                extent = max(
                    float(graph_a_json["metadata"]["checkpoint_pair_normalization_extent"]),
                    float(graph_b_json["metadata"]["checkpoint_pair_normalization_extent"]),
                )
                scale = 0.999999 / extent
                graph_a, nodes_a = body_to_data(graph_a_json, scale)
                graph_b, nodes_b = body_to_data(graph_b_json, scale)
                joint = make_joint_graph(graph_a.num_nodes, graph_b.num_nodes)
                batch = batch_to_device(
                    (
                        Batch.from_data_list([graph_a]),
                        Batch.from_data_list([graph_b]),
                        Batch.from_data_list([joint]),
                    ),
                    device,
                )
                logits = model(*batch).detach().cpu()
                joinable_order = torch.argsort(
                    logits, descending=True, stable=True
                ).tolist()
                analytic_scores = []
                for left, node_a in enumerate(nodes_a):
                    a = descriptor(node_a)
                    for right, node_b in enumerate(nodes_b):
                        score, _ = score_pair(a, descriptor(node_b))
                        if score > 0:
                            analytic_scores.append(
                                (float(score), left * len(nodes_b) + right)
                            )
                analytic_order = [
                    flat_index
                    for _, flat_index in sorted(
                        analytic_scores,
                        key=lambda item: (item[0], item[1]),
                        reverse=True,
                    )
                ]
                exact = {
                    (int(left), int(right))
                    for left, right in source.get("exact_positive_pairs", [])
                }
                equivalent = {
                    (int(left), int(right))
                    for left, right in source["equivalent_positive_pairs"]
                }
                n2 = len(nodes_b)
                rows.append(
                    {
                        "sample_id": source["sample_id"],
                        "source_design_ids": source["source_design_ids"],
                        "candidate_pair_count": source["candidate_pair_count"],
                        "analytic_ranked_candidate_count": len(analytic_order),
                        "exact_evaluable": bool(exact),
                        "analytic_exact_rank": first_rank(analytic_order, exact, n2) if exact else None,
                        "joinable_exact_rank": first_rank(joinable_order, exact, n2) if exact else None,
                        "analytic_equivalent_rank": first_rank(analytic_order, equivalent, n2),
                        "joinable_equivalent_rank": first_rank(joinable_order, equivalent, n2),
                    }
                )
            except Exception as exc:
                failures.append(
                    {"sample_id": source.get("sample_id"), "reason": f"{type(exc).__name__}:{exc}"}
                )
            if index % 50 == 0 or index == len(eligible):
                print(f"evaluated {index}/{len(eligible)}", flush=True)
    exact_rows = [row for row in rows if row["exact_evaluable"]]
    report = {
        "schema_version": "1.0.0",
        "purpose": "Design-disjoint STEP test comparison of graph-derived analytic and official pretrained JoinABLe ranking",
        "training_performed": False,
        "manifest": str(args.manifest.resolve()),
        "manifest_summary": manifest["summary"],
        "split": args.split,
        "device": str(device),
        "eligible_count": len(eligible),
        "evaluated_count": len(rows),
        "exact": summarize(exact_rows, "exact"),
        "equivalent": summarize(rows, "equivalent"),
        "rows": rows,
        "failures": failures,
        "limitations": [
            "This split was previously used for evaluation in domain-adaptation experiments; it is design-disjoint but not untouched.",
            "The analytic comparator is graph-derived and deterministic; it is not the mixed-pool D4 geometry score.",
            "Interface ranking does not establish functional assembly validity.",
        ],
    }
    write_json(args.output.resolve(), report)
    print(json.dumps({"exact": report["exact"], "equivalent": report["equivalent"]}, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
