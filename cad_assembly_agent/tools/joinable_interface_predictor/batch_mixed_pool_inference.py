"""Run the frozen JoinABLe model over every pair in anonymized mixed pools.

This command produces interface rankings and pair-level diagnostic features.
It deliberately does not decide whether a pair is connected: the released
JoinABLe model has no trained "no joint" class.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Batch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from joinable_gpu_reproduction.joinable_compat import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    batch_to_device,
    build_model,
    environment_summary,
    load_checkpoint,
)

from cad_assembly_agent.tools.joinable_interface_predictor.pretrained_joinable_predictor import (  # noqa: E402
    body_to_data,
    family_hint,
    make_joint_graph,
    public_entity,
    validate_graph,
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def device_from_name(name: str) -> torch.device:
    if name == "auto":
        return torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device("cuda:0" if name == "cuda" else "cpu")


def infer_pair(
    model: torch.nn.Module,
    graph_a: dict[str, Any],
    graph_b: dict[str, Any],
    device: torch.device,
    top_k: int,
    max_node_count: int,
) -> dict[str, Any]:
    combined_nodes = len(graph_a["nodes"]) + len(graph_b["nodes"])
    if max_node_count > 0 and combined_nodes > max_node_count:
        raise ValueError(
            f"combined_node_count_exceeds_official_limit:"
            f"{combined_nodes}>{max_node_count}"
        )
    extent_a = float(
        graph_a["metadata"]["checkpoint_pair_normalization_extent"]
    )
    extent_b = float(
        graph_b["metadata"]["checkpoint_pair_normalization_extent"]
    )
    pair_scale = 0.999999 / max(extent_a, extent_b)
    data_a, nodes_a = body_to_data(graph_a, pair_scale)
    data_b, nodes_b = body_to_data(graph_b, pair_scale)
    joint_graph = make_joint_graph(data_a.num_nodes, data_b.num_nodes)
    batch = (
        Batch.from_data_list([data_a]),
        Batch.from_data_list([data_b]),
        Batch.from_data_list([joint_graph]),
    )
    batch = batch_to_device(batch, device)
    with torch.inference_mode():
        logits = model(*batch)
        probabilities = torch.softmax(logits, dim=0)
    candidate_count = int(logits.numel())
    retained = min(max(1, top_k), candidate_count)
    order = torch.argsort(
        logits, descending=True, stable=True
    )[:retained].detach().cpu()
    logits_cpu = logits.detach().cpu()
    probabilities_cpu = probabilities.detach().cpu()
    sorted_logits = torch.sort(logits_cpu, descending=True).values
    margin = (
        float(sorted_logits[0] - sorted_logits[1])
        if candidate_count > 1
        else None
    )
    entropy = float(
        -(
            probabilities_cpu
            * torch.log(probabilities_cpu.clamp_min(1e-12))
        ).sum()
    )
    normalized_entropy = (
        entropy / math.log(candidate_count)
        if candidate_count > 1
        else 0.0
    )
    joint_edges = joint_graph.edge_index
    candidates = []
    for rank, flat_index in enumerate(order.tolist(), start=1):
        node_a_index = int(joint_edges[0, flat_index])
        node_b_index = int(joint_edges[1, flat_index]) - data_a.num_nodes
        entity_a = public_entity(nodes_a[node_a_index])
        entity_b = public_entity(nodes_b[node_b_index])
        entity_a["scaled_length"] = float(
            data_a.length[node_a_index]
        )
        entity_b["scaled_length"] = float(
            data_b.length[node_b_index]
        )
        candidates.append(
            {
                "rank": rank,
                "flat_candidate_index": int(flat_index),
                "part_a_entity": entity_a,
                "part_b_entity": entity_b,
                "joint_family_candidate": family_hint(
                    entity_a, entity_b
                ),
                "score": float(logits_cpu[flat_index]),
                "softmax_probability": float(
                    probabilities_cpu[flat_index]
                ),
            }
        )
    top_probability = float(probabilities_cpu[order[0]])
    return {
        "status": "success",
        "part_a_node_count": int(data_a.num_nodes),
        "part_b_node_count": int(data_b.num_nodes),
        "combined_node_count": combined_nodes,
        "candidate_entity_pair_count": candidate_count,
        "pair_scale": pair_scale,
        "pair_features": {
            "top_1_logit": float(logits_cpu[order[0]]),
            "top_1_probability": top_probability,
            "top_1_uniform_lift": top_probability * candidate_count,
            "top_2_logit_margin": margin,
            "entropy": entropy,
            "normalized_entropy": normalized_entropy,
        },
        "candidates": candidates,
        "failure_reasons": [],
        "unavailable_fields": [
            "trained_no_joint_probability",
            "functional_assembly_validity",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mixed-pool-root", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_CHECKPOINT
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-node-count", type=int, default=950)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    args = parser.parse_args()
    root = args.mixed_pool_root.resolve()
    manifest = read_json(root / "mixed_pool_manifest.json")
    checkpoint, official_args = load_checkpoint(args.checkpoint)
    model = build_model(checkpoint, official_args).eval()
    device = device_from_name(args.device)
    model.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    rows = []
    failures = []
    started = time.perf_counter()
    for pool in manifest.get("pools", []):
        pool_id = str(pool["pool_id"])
        pool_dir = root / pool_id
        pool_input = read_json(pool_dir / "pool_input.json")
        part_ids = sorted(
            str(part["part_id"])
            for part in pool_input.get("parts", [])
        )
        graphs = {}
        for part_id in part_ids:
            graph_path = (
                pool_dir
                / "joinable_graphs"
                / f"{part_id}.brep_graph.json"
            )
            try:
                graph = read_json(graph_path)
                validate_graph(graph, graph_path)
                graphs[part_id] = graph
            except Exception as exc:
                failures.append(
                    {
                        "pool_id": pool_id,
                        "part_id": part_id,
                        "reason": (
                            f"graph_unusable:{type(exc).__name__}:{exc}"
                        ),
                    }
                )
        for part_a, part_b in combinations(part_ids, 2):
            pair_id = f"{pool_id}:{part_a}:{part_b}"
            row = {
                "pair_id": pair_id,
                "pool_id": pool_id,
                "split": pool.get("split"),
                "part_a": part_a,
                "part_b": part_b,
            }
            if part_a not in graphs or part_b not in graphs:
                row.update(
                    {
                        "status": "graph_unusable",
                        "candidates": [],
                        "failure_reasons": [
                            "one_or_both_part_graphs_unusable"
                        ],
                        "unavailable_fields": [
                            "ranked_interface_candidates"
                        ],
                    }
                )
            else:
                try:
                    row.update(
                        infer_pair(
                            model,
                            graphs[part_a],
                            graphs[part_b],
                            device,
                            args.top_k,
                            args.max_node_count,
                        )
                    )
                except Exception as exc:
                    reason = (
                        f"{type(exc).__name__}:{exc}"
                    )
                    row.update(
                        {
                            "status": "inference_failed",
                            "candidates": [],
                            "failure_reasons": [reason],
                            "unavailable_fields": [
                                "ranked_interface_candidates"
                            ],
                        }
                    )
                    failures.append(
                        {
                            "pair_id": pair_id,
                            "reason": reason,
                        }
                    )
            rows.append(row)
        print(
            f"completed {pool_id}: "
            f"{sum(row['pool_id'] == pool_id for row in rows)} pairs",
            flush=True,
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    success_count = sum(row["status"] == "success" for row in rows)
    output = {
        "schema_version": "1.0.0",
        "purpose": (
            "Frozen JoinABLe interface rankings for every anonymized "
            "mixed-pool part pair."
        ),
        "dataset_id": manifest.get("dataset_id"),
        "mixed_pool_root": str(root),
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "epoch": checkpoint.get("epoch"),
            "global_step": checkpoint.get("global_step"),
            "strict_weight_load": True,
        },
        "runtime": environment_summary(),
        "device": str(device),
        "model_boundary": {
            "trained_no_joint_class": False,
            "pair_connection_decision_available": False,
            "interface_ranking_available": True,
            "why": (
                "Released JoinABLe is trained to rank interface entities on "
                "known positive part pairs; every input pair receives a "
                "ranking."
            ),
        },
        "top_k": args.top_k,
        "max_node_count": args.max_node_count,
        "pair_count": len(rows),
        "success_count": success_count,
        "failure_count": len(rows) - success_count,
        "elapsed_seconds": elapsed,
        "pairs_per_second": (
            success_count / elapsed if elapsed > 0 else None
        ),
        "gpu_peak_allocated_mib": (
            torch.cuda.max_memory_allocated() / 1024**2
            if device.type == "cuda"
            else None
        ),
        "pairs": rows,
        "failures": failures,
        "failure_reasons": [
            str(row.get("reason")) for row in failures
        ],
        "unavailable_fields": [
            "trained_no_joint_probability",
            "functional_assembly_validity",
            "final_group_decision",
        ],
    }
    write_json(args.output.resolve(), output)
    print(
        f"mixed-pool inference {success_count}/{len(rows)} pairs "
        f"in {elapsed:.2f}s"
    )
    return 0 if success_count == len(rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
