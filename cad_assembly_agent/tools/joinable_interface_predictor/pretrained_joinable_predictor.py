"""Run the released JoinABLe checkpoint on two audited STEP graph JSON files.

This process is intentionally isolated from OCCT.  STEP extraction happens in
the CAD environment; this predictor consumes only structured JSON and emits
ranked interface candidates.  Its score never implies final assembly acceptance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data


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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_graph(graph: dict[str, Any], source: Path) -> None:
    metadata = graph.get("metadata", {})
    if metadata.get("extraction_status") != "success":
        raise ValueError(f"graph_extraction_not_success:{source}")
    if not metadata.get("released_checkpoint_minimal_features_available"):
        raise ValueError(f"checkpoint_features_unavailable:{source}")
    nodes = graph.get("nodes", [])
    indices = [node.get("joinable_node_index") for node in nodes]
    if indices != list(range(len(nodes))):
        raise ValueError(f"non_sequential_joinable_node_indices:{source}")
    for node in nodes:
        required = (
            "joinable_entity_type_index",
            "is_face",
            "length",
            "face_reversed",
            "edge_reversed",
            "geometry_signature",
            "occt_topology_index",
        )
        missing = [field for field in required if field not in node]
        if missing:
            raise ValueError(
                f"node_missing_checkpoint_fields:{node.get('node_id')}:{missing}"
            )
    extent = metadata.get("checkpoint_pair_normalization_extent")
    if extent is None or not math.isfinite(float(extent)) or float(extent) <= 0:
        raise ValueError(f"invalid_normalization_extent:{source}:{extent}")


def body_to_data(
    graph: dict[str, Any], pair_scale: float
) -> tuple[Data, list[dict[str, Any]]]:
    nodes = sorted(
        graph["nodes"], key=lambda row: row["joinable_node_index"]
    )
    node_count = len(nodes)
    type_indices = torch.tensor(
        [int(node["joinable_entity_type_index"]) for node in nodes],
        dtype=torch.long,
    )
    entity_types = F.one_hot(type_indices, num_classes=16)
    is_face = torch.tensor(
        [int(node["is_face"]) for node in nodes], dtype=torch.long
    )
    lengths = torch.tensor(
        [float(node["length"]) * pair_scale for node in nodes],
        dtype=torch.float,
    )
    face_reversed = torch.tensor(
        [int(node["face_reversed"]) for node in nodes], dtype=torch.long
    )
    edge_reversed = torch.tensor(
        [int(node["edge_reversed"]) for node in nodes], dtype=torch.long
    )

    id_to_index = {
        str(node["node_id"]): int(node["joinable_node_index"])
        for node in nodes
    }
    directed_edges: list[tuple[int, int]] = []
    for edge in graph.get("edges", []):
        source = id_to_index[str(edge["src"])]
        target = id_to_index[str(edge["dst"])]
        directed_edges.append((source, target))
        directed_edges.append((target, source))
    if not directed_edges:
        raise ValueError("body_graph_has_no_adjacency")
    edge_index = torch.tensor(directed_edges, dtype=torch.long).t().contiguous()
    data = Data(
        num_nodes=node_count,
        edge_index=edge_index,
        entity_types=entity_types,
        is_face=is_face,
        length=lengths,
        face_reversed=face_reversed,
        edge_reversed=edge_reversed,
    )
    return data, nodes


def make_joint_graph(n1: int, n2: int) -> Data:
    first = torch.arange(n1)
    second = torch.arange(n2) + n1
    edge_index = torch.cartesian_prod(first, second).t().contiguous()
    return Data(
        num_nodes=n1 + n2,
        edge_index=edge_index,
        num_nodes_graph1=n1,
        num_nodes_graph2=n2,
    )


def public_entity(node: dict[str, Any]) -> dict[str, Any]:
    geometry_type = (
        node.get("surface_type")
        if node["entity_type"] == "face"
        else node.get("curve_type")
    )
    return {
        "entity_id": node["node_id"],
        "entity_type": node["entity_type"],
        "topology_index": int(node["occt_topology_index"]),
        "joinable_node_index": int(node["joinable_node_index"]),
        "joinable_entity_type": node["joinable_entity_type"],
        "joinable_entity_type_index": int(
            node["joinable_entity_type_index"]
        ),
        "joinable_entity_type_mapping_quality": node[
            "joinable_entity_type_mapping_quality"
        ],
        "geometry_type": geometry_type,
        "geometry_signature": node["geometry_signature"],
        "orientation": node.get("orientation"),
        "scaled_length": None,
    }


def family_hint(a: dict[str, Any], b: dict[str, Any]) -> str:
    types = {a["joinable_entity_type"], b["joinable_entity_type"]}
    if any("Cylinder" in value or "Circle" in value for value in types):
        return "coaxial_or_cylindrical"
    if types == {"PlaneSurfaceType"}:
        return "planar"
    return "axis_or_interface_alignment"


def predict(
    part_a_graph: Path,
    part_b_graph: Path,
    checkpoint_path: Path,
    output_path: Path,
    top_k: int,
    device_name: str,
    max_node_count: int,
) -> int:
    try:
        graph_a = read_json(part_a_graph)
        graph_b = read_json(part_b_graph)
        validate_graph(graph_a, part_a_graph)
        validate_graph(graph_b, part_b_graph)
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

        checkpoint, official_args = load_checkpoint(checkpoint_path)
        model = build_model(checkpoint, official_args).eval()
        if device_name == "auto":
            device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )
        else:
            if device_name == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable")
            device = torch.device(
                "cuda:0" if device_name == "cuda" else "cpu"
            )
        model.to(device)
        batch = batch_to_device(batch, device)
        with torch.inference_mode():
            logits = model(*batch)
            probabilities = torch.softmax(logits, dim=0)

        candidate_count = int(logits.numel())
        retained = min(max(1, top_k), candidate_count)
        top_indices = torch.argsort(
            logits, descending=True, stable=True
        )[:retained].detach().cpu()
        logits_cpu = logits.detach().cpu()
        probabilities_cpu = probabilities.detach().cpu()
        joint_edges = joint_graph.edge_index
        candidates = []
        approximate_mapping_used = False
        for rank, flat_index in enumerate(top_indices.tolist(), 1):
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
            approximate_mapping_used |= any(
                entity["joinable_entity_type_mapping_quality"] != "exact"
                for entity in (entity_a, entity_b)
            )
            candidates.append(
                {
                    "candidate_id": (
                        f"{entity_a['entity_id']}__{entity_b['entity_id']}"
                    ),
                    "rank": rank,
                    "part_a_entity": entity_a,
                    "part_b_entity": entity_b,
                    "joint_family_candidate": family_hint(
                        entity_a, entity_b
                    ),
                    "score": float(logits_cpu[flat_index]),
                    "softmax_probability": float(
                        probabilities_cpu[flat_index]
                    ),
                    "model_kind": "official_pretrained_joinable",
                    "review_required": True,
                    "review_reason": (
                        "Learned interface evidence alone cannot establish "
                        "assembly correctness."
                    ),
                }
            )

        output = {
            "schema_version": "1.0.0",
            "part_a": graph_a["source_step_path"],
            "part_b": graph_b["source_step_path"],
            "source_graph_a": str(part_a_graph.resolve()),
            "source_graph_b": str(part_b_graph.resolve()),
            "predictor": "official_pretrained_joinable_modern_runtime",
            "is_pretrained_joinable": True,
            "checkpoint": {
                "path": str(checkpoint_path.resolve()),
                "sha256": sha256_file(checkpoint_path),
                "epoch": checkpoint.get("epoch"),
                "global_step": checkpoint.get("global_step"),
                "strict_weight_load": True,
                "input_features": official_args.input_features,
            },
            "runtime": environment_summary(),
            "device": str(device),
            "normalization": {
                "method": (
                    "official pair common scale approximated from exact STEP "
                    "bounding-box half extents"
                ),
                "part_a_extent": extent_a,
                "part_b_extent": extent_b,
                "pair_scale": pair_scale,
            },
            "top_k": retained,
            "candidate_count": candidate_count,
            "candidates": candidates,
            "gate_policy": {
                "shadow_mode": True,
                "can_change_accepted_groups": False,
                "requires_pose_and_collision_validation": True,
                "requires_multi_evidence_gate": True,
            },
            "mapping_audit": {
                "approximate_entity_mapping_used_in_top_k": (
                    approximate_mapping_used
                ),
                "part_a_approximated_entities": graph_a["metadata"].get(
                    "released_checkpoint_approximated_entity_mappings", []
                ),
                "part_b_approximated_entities": graph_b["metadata"].get(
                    "released_checkpoint_approximated_entity_mappings", []
                ),
                "topology_id_stability": graph_a["metadata"].get(
                    "id_stability_scope"
                ),
            },
            "failure_reasons": [],
            "unavailable_fields": [
                "designer_selected_interface_truth_for_arbitrary_step",
                "functional_assembly_validity",
                "final_pose_without_external_solver",
                "permanent_cross_export_topology_ids",
            ],
        }
        write_json(output_path, output)
        print(
            f"pretrained JoinABLe: {retained}/{candidate_count} candidates "
            f"on {device}"
        )
        return 0
    except Exception as exc:
        write_json(
            output_path,
            {
                "schema_version": "1.0.0",
                "predictor": "official_pretrained_joinable_modern_runtime",
                "is_pretrained_joinable": True,
                "candidates": [],
                "failure_reasons": [f"{type(exc).__name__}:{exc}"],
                "unavailable_fields": ["ranked_interface_candidates"],
            },
        )
        return 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part-a-graph", type=Path, required=True)
    parser.add_argument("--part-b-graph", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-node-count", type=int, default=950)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    args = parser.parse_args()
    return predict(
        args.part_a_graph,
        args.part_b_graph,
        args.checkpoint,
        args.output,
        args.top_k,
        args.device,
        args.max_node_count,
    )


if __name__ == "__main__":
    raise SystemExit(main())
