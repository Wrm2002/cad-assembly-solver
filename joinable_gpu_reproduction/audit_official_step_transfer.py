"""Audit pretrained JoinABLe transfer from Fusion graphs to their exported STEP.

The audit uses the official sample20 data where each body has both the source
Fusion B-Rep graph and a STEP export.  It measures cross-kernel topology parity,
maps designer-selected entities by geometry, and evaluates pretrained Top-K
ranking without assuming that Fusion and OCCT edge indices are identical.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.data import Data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cad_assembly_agent.tools.joinable_interface_predictor.pretrained_joinable_predictor import (  # noqa: E402
    body_to_data,
    make_joint_graph,
    public_entity,
    validate_graph,
)
from joinable_gpu_reproduction.joinable_compat import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    batch_to_device,
    build_model,
    environment_summary,
    load_checkpoint,
    write_json,
)


DEFAULT_SAMPLE_ROOT = Path(
    r"D:\Model_match_public_data\fusion360_joint\sample20\j1.0.0\joint"
)
DEFAULT_GRAPH_ROOT = (
    PROJECT_ROOT / "joinable_migration_audit" / "phase1_sample20_graphs"
)
ENTITY_TYPE_MAP = {
    "PlaneSurfaceType": 0,
    "CylinderSurfaceType": 1,
    "ConeSurfaceType": 2,
    "SphereSurfaceType": 3,
    "TorusSurfaceType": 4,
    "EllipticalCylinderSurfaceType": 5,
    "EllipticalConeSurfaceType": 6,
    "NurbsSurfaceType": 7,
    "Line3DCurveType": 8,
    "Arc3DCurveType": 9,
    "Circle3DCurveType": 10,
    "Ellipse3DCurveType": 11,
    "EllipticalArc3DCurveType": 12,
    "InfiniteLine3DCurveType": 13,
    "NurbsCurve3DCurveType": 14,
    "Degenerate3DCurveType": 15,
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def point_from_source_node(node: dict[str, Any], entity_type: str) -> list[float]:
    if entity_type == "face":
        return [
            float(node[f"centroid_{axis}"]) for axis in ("x", "y", "z")
        ]
    curve_type = str(node.get("curve_type", ""))
    if curve_type in {"Circle3DCurveType", "Ellipse3DCurveType"} and all(
        f"center_{axis}" in node for axis in ("x", "y", "z")
    ):
        return [float(node[f"center_{axis}"]) for axis in ("x", "y", "z")]
    points = np.asarray(node.get("points", []), dtype=float).reshape(-1, 3)
    if points.size == 0:
        raise ValueError("source_curve_has_no_points")
    return points.mean(axis=0).tolist()


def source_bbox_extent(source_graph: dict[str, Any]) -> float:
    bbox = source_graph["properties"]["bounding_box"]
    low = bbox["min_point"]
    high = bbox["max_point"]
    return max(
        (float(high[axis]) - float(low[axis])) * 0.5
        for axis in ("x", "y", "z")
    )


def source_entity_node(
    source_graph: dict[str, Any], entity: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    entity_type = "face" if entity["type"] == "BRepFace" else "edge"
    index = int(entity["index"])
    if entity_type == "edge":
        index += int(source_graph["properties"]["face_count"])
    nodes = source_graph["nodes"]
    if index < 0 or index >= len(nodes):
        raise IndexError(f"source_entity_index_out_of_range:{index}")
    return nodes[index], entity_type


def map_entity_to_occt(
    entity: dict[str, Any],
    source_graph: dict[str, Any],
    occt_graph: dict[str, Any],
) -> dict[str, Any]:
    source_node, entity_type = source_entity_node(source_graph, entity)
    source_extent = source_bbox_extent(source_graph)
    target_extent = float(
        occt_graph["metadata"]["checkpoint_pair_normalization_extent"]
    )
    if source_extent <= 0:
        raise ValueError("source_bbox_extent_not_positive")
    unit_scale = target_extent / source_extent

    source_type = (
        source_node["surface_type"]
        if entity_type == "face"
        else source_node["curve_type"]
    )
    source_measure = float(
        source_node["area"] if entity_type == "face" else source_node["length"]
    )
    target_measure = source_measure * (
        unit_scale**2 if entity_type == "face" else unit_scale
    )
    source_point = np.asarray(
        point_from_source_node(source_node, entity_type), dtype=float
    ) * unit_scale

    candidates = []
    for node in occt_graph["nodes"]:
        if node["entity_type"] != entity_type:
            continue
        if node["joinable_entity_type"] != source_type:
            continue
        measure = float(
            node["area"] if entity_type == "face" else node["length"]
        )
        measure_rel = abs(measure - target_measure) / max(
            abs(target_measure), 1e-9
        )
        target_point = np.asarray(node.get("centroid"), dtype=float)
        centroid_rel = float(
            np.linalg.norm(target_point - source_point)
            / max(target_extent, 1e-9)
        )
        orientation_penalty = 0.0
        if entity_type == "face":
            source_reversed = bool(source_node.get("reversed"))
            target_reversed = bool(node.get("face_reversed"))
            orientation_penalty = 0.05 * (
                source_reversed != target_reversed
            )
        order_penalty = 1e-6 * abs(
            int(node["occt_topology_index"]) - (int(entity["index"]) + 1)
        )
        score = (
            measure_rel
            + centroid_rel
            + orientation_penalty
            + order_penalty
        )
        candidates.append(
            {
                "node_id": node["node_id"],
                "joinable_node_index": int(node["joinable_node_index"]),
                "occt_topology_index": int(node["occt_topology_index"]),
                "score": score,
                "measure_relative_error": measure_rel,
                "centroid_relative_error": centroid_rel,
            }
        )

    candidates.sort(key=lambda row: (row["score"], row["node_id"]))
    if not candidates:
        return {
            "status": "unmapped",
            "reason": f"no_type_compatible_occt_entity:{source_type}",
            "matches": [],
        }
    best = candidates[0]
    if (
        best["measure_relative_error"] > 0.02
        or best["centroid_relative_error"] > 0.05
    ):
        return {
            "status": "unmapped",
            "reason": "best_geometry_match_outside_tolerance",
            "matches": candidates[:3],
        }
    matches = [
        row for row in candidates
        if row["score"] <= best["score"] + 1e-5
    ]
    return {
        "status": "mapped" if len(matches) == 1 else "ambiguous_equivalent",
        "reason": None,
        "unit_scale": unit_scale,
        "source_type": source_type,
        "source_entity_type": entity_type,
        "source_entity_index": int(entity["index"]),
        "matches": matches,
    }


def all_geometry_entities(geometry: dict[str, Any]) -> list[dict[str, Any]]:
    result = [geometry["entity_one"]]
    result.extend(geometry.get("entity_one_equivalents", []))
    seen = set()
    unique = []
    for entity in result:
        key = (
            entity.get("body"),
            entity.get("type"),
            int(entity.get("index", -1)),
        )
        if key not in seen:
            seen.add(key)
            unique.append(entity)
    return unique


def fusion_body_to_data(
    graph: dict[str, Any], pair_scale: float
) -> tuple[Data, list[dict[str, Any]]]:
    nodes = graph["nodes"]
    face_count = int(graph["properties"]["face_count"])
    type_names = []
    public_nodes = []
    for index, node in enumerate(nodes):
        is_face = index < face_count
        type_name = (
            node["surface_type"]
            if is_face
            else (
                "Degenerate3DCurveType"
                if node.get("is_degenerate")
                else node["curve_type"]
            )
        )
        type_names.append(type_name)
        public_nodes.append(
            {
                "node_id": (
                    f"face_{index + 1:06d}"
                    if is_face
                    else f"edge_{index - face_count + 1:06d}"
                ),
                "entity_type": "face" if is_face else "edge",
                "occt_topology_index": (
                    index + 1 if is_face else index - face_count + 1
                ),
                "joinable_node_index": index,
                "joinable_entity_type": type_name,
                "joinable_entity_type_index": ENTITY_TYPE_MAP[type_name],
                "joinable_entity_type_mapping_quality": "source_exact",
                "geometry_signature": f"fusion_node_id:{node['id']}",
                "orientation": (
                    "reversed" if bool(node.get("reversed")) else "forward"
                ),
                "surface_type": (
                    type_name if is_face else None
                ),
                "curve_type": (
                    None if is_face else type_name
                ),
            }
        )
    type_indices = torch.tensor(
        [ENTITY_TYPE_MAP[name] for name in type_names], dtype=torch.long
    )
    is_face_tensor = torch.tensor(
        [int(index < face_count) for index in range(len(nodes))],
        dtype=torch.long,
    )
    lengths = torch.tensor(
        [
            0.0 if index < face_count else float(node.get("length", 0.0))
            for index, node in enumerate(nodes)
        ],
        dtype=torch.float,
    ) * pair_scale
    face_reversed = torch.tensor(
        [
            int(bool(node.get("reversed"))) if index < face_count else 0
            for index, node in enumerate(nodes)
        ],
        dtype=torch.long,
    )
    edge_reversed = torch.tensor(
        [
            0 if index < face_count else int(bool(node.get("reversed")))
            for index, node in enumerate(nodes)
        ],
        dtype=torch.long,
    )
    id_to_index = {
        int(node["id"]): index for index, node in enumerate(nodes)
    }
    directed_edges = []
    for link in graph["links"]:
        source = id_to_index[int(link["source"])]
        target = id_to_index[int(link["target"])]
        directed_edges.extend(((source, target), (target, source)))
    edge_index = torch.tensor(
        directed_edges, dtype=torch.long
    ).t().contiguous()
    return (
        Data(
            num_nodes=len(nodes),
            edge_index=edge_index,
            entity_types=F.one_hot(type_indices, num_classes=16),
            is_face=is_face_tensor,
            length=lengths,
            face_reversed=face_reversed,
            edge_reversed=edge_reversed,
        ),
        public_nodes,
    )


def infer_prepared_pair(
    model: torch.nn.Module,
    device: torch.device,
    data_a: Data,
    data_b: Data,
    nodes_a: list[dict[str, Any]],
    nodes_b: list[dict[str, Any]],
    top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
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
    retained = min(top_k, logits.numel())
    indices = torch.argsort(
        logits, descending=True, stable=True
    )[:retained].detach().cpu().tolist()
    logits = logits.detach().cpu()
    probabilities = probabilities.detach().cpu()
    candidates = []
    for rank, flat_index in enumerate(indices, 1):
        a_index = int(joint_graph.edge_index[0, flat_index])
        b_index = int(joint_graph.edge_index[1, flat_index]) - data_a.num_nodes
        candidates.append(
            {
                "rank": rank,
                "flat_candidate_index": flat_index,
                "part_a_entity": public_entity(nodes_a[a_index]),
                "part_b_entity": public_entity(nodes_b[b_index]),
                "logit": float(logits[flat_index]),
                "softmax_probability": float(probabilities[flat_index]),
            }
        )
    metadata = {
        "part_a_node_count": data_a.num_nodes,
        "part_b_node_count": data_b.num_nodes,
        "combined_node_count": data_a.num_nodes + data_b.num_nodes,
        "candidate_count": int(logits.numel()),
    }
    return candidates, metadata


def infer_step_pair(
    model: torch.nn.Module,
    device: torch.device,
    graph_a: dict[str, Any],
    graph_b: dict[str, Any],
    top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    extent_a = float(
        graph_a["metadata"]["checkpoint_pair_normalization_extent"]
    )
    extent_b = float(
        graph_b["metadata"]["checkpoint_pair_normalization_extent"]
    )
    pair_scale = 0.999999 / max(extent_a, extent_b)
    data_a, nodes_a = body_to_data(graph_a, pair_scale)
    data_b, nodes_b = body_to_data(graph_b, pair_scale)
    candidates, metadata = infer_prepared_pair(
        model, device, data_a, data_b, nodes_a, nodes_b, top_k
    )
    metadata["pair_scale"] = pair_scale
    return candidates, metadata


def infer_fusion_pair(
    model: torch.nn.Module,
    device: torch.device,
    graph_a: dict[str, Any],
    graph_b: dict[str, Any],
    top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    extent_a = source_bbox_extent(graph_a)
    extent_b = source_bbox_extent(graph_b)
    pair_scale = 0.999999 / max(extent_a, extent_b)
    data_a, nodes_a = fusion_body_to_data(graph_a, pair_scale)
    data_b, nodes_b = fusion_body_to_data(graph_b, pair_scale)
    candidates, metadata = infer_prepared_pair(
        model, device, data_a, data_b, nodes_a, nodes_b, top_k
    )
    metadata["pair_scale"] = pair_scale
    return candidates, metadata


def direct_source_index(
    entity: dict[str, Any], graph: dict[str, Any]
) -> int:
    index = int(entity["index"])
    if entity["type"] == "BRepEdge":
        index += int(graph["properties"]["face_count"])
    return index


def rank_for_truth(
    candidates: list[dict[str, Any]],
    a_indices: set[int],
    b_indices: set[int],
    limit: int,
) -> int | None:
    for candidate in candidates[:limit]:
        a_index = int(candidate["part_a_entity"]["joinable_node_index"])
        b_index = int(candidate["part_b_entity"]["joinable_node_index"])
        if a_index in a_indices and b_index in b_indices:
            return int(candidate["rank"])
    return None


def summarize_rates(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    evaluable = [row for row in rows if row[f"{prefix}_evaluable"]]
    result: dict[str, Any] = {
        f"{prefix}_evaluable_joint_count": len(evaluable)
    }
    for k in (1, 5, 10, 20):
        hits = sum(
            row[f"{prefix}_rank"] is not None
            and row[f"{prefix}_rank"] <= k
            for row in evaluable
        )
        result[f"{prefix}_top_{k}_recall"] = (
            hits / len(evaluable) if evaluable else None
        )
    return result


def paired_transfer_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    paired = [
        row for row in rows
        if row["equivalent_evaluable"]
        and row["source_equivalent_evaluable"]
    ]

    def recall(items: list[dict[str, Any]], field: str, k: int) -> float | None:
        if not items:
            return None
        return sum(
            row[field] is not None and row[field] <= k for row in items
        ) / len(items)

    result: dict[str, Any] = {"paired_evaluable_joint_count": len(paired)}
    for k in (1, 5, 10, 20):
        source = recall(paired, "source_equivalent_rank", k)
        step = recall(paired, "equivalent_rank", k)
        result[f"source_equivalent_top_{k}_recall"] = source
        result[f"step_equivalent_top_{k}_recall"] = step
        result[f"step_minus_source_top_{k}"] = (
            step - source
            if source is not None and step is not None
            else None
        )
    result["by_entity_pair_type"] = {}
    for pair_type in sorted(
        {row["truth_entity_pair_type"] for row in paired}
    ):
        subset = [
            row for row in paired
            if row["truth_entity_pair_type"] == pair_type
        ]
        result["by_entity_pair_type"][pair_type] = {
            "count": len(subset),
            "source_equivalent_top_10_recall": recall(
                subset, "source_equivalent_rank", 10
            ),
            "step_equivalent_top_10_recall": recall(
                subset, "equivalent_rank", 10
            ),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-root", type=Path, default=DEFAULT_SAMPLE_ROOT)
    parser.add_argument("--graph-root", type=Path, default=DEFAULT_GRAPH_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-node-count", type=int, default=950)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        default=Path(__file__).with_name("official_step_predictions"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name(
            "official_step_transfer_audit.json"
        ),
    )
    args = parser.parse_args()

    checkpoint, official_args = load_checkpoint(args.checkpoint)
    model = build_model(checkpoint, official_args).eval()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    model.to(device)

    pair_rows = []
    joint_rows = []
    topology_rows = []
    mapping_failures = []
    inference_failures = []
    joint_files = sorted(args.sample_root.glob("joint_set_*.json"))
    args.prediction_dir.mkdir(parents=True, exist_ok=True)

    for pair_index, joint_path in enumerate(joint_files, 1):
        joint_set = read_json(joint_path)
        body_a = str(joint_set["body_one"])
        body_b = str(joint_set["body_two"])
        fusion_a = read_json(args.sample_root / f"{body_a}.json")
        fusion_b = read_json(args.sample_root / f"{body_b}.json")
        graph_a_path = args.graph_root / f"{body_a}.brep_graph.json"
        graph_b_path = args.graph_root / f"{body_b}.brep_graph.json"
        graph_a = read_json(graph_a_path)
        graph_b = read_json(graph_b_path)
        validate_graph(graph_a, graph_a_path)
        validate_graph(graph_b, graph_b_path)

        for body, fusion, graph in (
            (body_a, fusion_a, graph_a),
            (body_b, fusion_b, graph_b),
        ):
            topology_rows.append(
                {
                    "body": body,
                    "fusion_face_count": int(
                        fusion["properties"]["face_count"]
                    ),
                    "occt_face_count": int(graph["metadata"]["num_faces"]),
                    "fusion_edge_count": int(
                        fusion["properties"]["edge_count"]
                    ),
                    "occt_edge_count": int(graph["metadata"]["num_edges"]),
                    "fusion_adjacency_count": len(fusion["links"]),
                    "occt_adjacency_count": len(graph["edges"]),
                }
            )

        combined_nodes = len(graph_a["nodes"]) + len(graph_b["nodes"])
        candidates: list[dict[str, Any]] = []
        inference_status = "success"
        inference_reason = None
        inference_metadata: dict[str, Any] = {
            "combined_node_count": combined_nodes
        }
        if combined_nodes > args.max_node_count:
            inference_status = "unsupported_official_node_limit"
            inference_reason = (
                f"combined_node_count:{combined_nodes}>"
                f"{args.max_node_count}"
            )
        else:
            try:
                candidates, inference_metadata = infer_step_pair(
                    model, device, graph_a, graph_b, args.top_k
                )
            except Exception as exc:
                inference_status = "failed"
                inference_reason = f"{type(exc).__name__}:{exc}"
        if inference_reason:
            inference_failures.append(
                {
                    "joint_set": joint_path.name,
                    "reason": inference_reason,
                }
            )

        prediction_path = (
            args.prediction_dir / f"{joint_path.stem}.prediction.json"
        )
        write_json(
            prediction_path,
            {
                "schema_version": "1.0.0",
                "joint_set": joint_path.name,
                "body_one": body_a,
                "body_two": body_b,
                "inference_status": inference_status,
                "inference_reason": inference_reason,
                "metadata": inference_metadata,
                "candidates": candidates,
                "failure_reasons": (
                    [] if inference_reason is None else [inference_reason]
                ),
                "unavailable_fields": [
                    "functional_assembly_validity",
                    "final_pose_without_external_solver",
                ],
            },
        )

        source_combined_nodes = len(fusion_a["nodes"]) + len(
            fusion_b["nodes"]
        )
        source_candidates: list[dict[str, Any]] = []
        source_inference_status = "success"
        source_inference_reason = None
        source_inference_metadata: dict[str, Any] = {
            "combined_node_count": source_combined_nodes
        }
        if source_combined_nodes > args.max_node_count:
            source_inference_status = "unsupported_official_node_limit"
            source_inference_reason = (
                f"source_combined_node_count:{source_combined_nodes}>"
                f"{args.max_node_count}"
            )
        else:
            try:
                source_candidates, source_inference_metadata = (
                    infer_fusion_pair(
                        model, device, fusion_a, fusion_b, args.top_k
                    )
                )
            except Exception as exc:
                source_inference_status = "failed"
                source_inference_reason = f"{type(exc).__name__}:{exc}"

        body_assets = {
            body_a: (fusion_a, graph_a, "a"),
            body_b: (fusion_b, graph_b, "b"),
        }
        for joint_index, joint in enumerate(joint_set["joints"]):
            exact_indices = {"a": set(), "b": set()}
            equivalent_indices = {"a": set(), "b": set()}
            source_exact_indices = {"a": set(), "b": set()}
            source_equivalent_indices = {"a": set(), "b": set()}
            exact_entity_types: dict[str, str] = {}
            mapping_records = []
            for geometry_key in (
                "geometry_or_origin_one",
                "geometry_or_origin_two",
            ):
                geometry = joint[geometry_key]
                all_entities = all_geometry_entities(geometry)
                for entity_position, entity in enumerate(all_entities):
                    body = str(entity["body"])
                    if body not in body_assets:
                        mapping_records.append(
                            {
                                "status": "unmapped",
                                "reason": f"entity_body_not_in_pair:{body}",
                            }
                        )
                        continue
                    fusion, graph, side = body_assets[body]
                    source_index = direct_source_index(entity, fusion)
                    if entity_position == 0:
                        source_exact_indices[side].add(source_index)
                        exact_entity_types[side] = (
                            "face"
                            if entity["type"] == "BRepFace"
                            else "edge"
                        )
                    source_equivalent_indices[side].add(source_index)
                    mapping = map_entity_to_occt(entity, fusion, graph)
                    mapping["side"] = side
                    mapping["is_exact_designer_entity"] = entity_position == 0
                    mapping_records.append(mapping)
                    mapping_valid = mapping.get("status") in {
                        "mapped",
                        "ambiguous_equivalent",
                    }
                    mapped_indices = (
                        {
                            int(row["joinable_node_index"])
                            for row in mapping.get("matches", [])
                        }
                        if mapping_valid
                        else set()
                    )
                    if entity_position == 0:
                        exact_indices[side].update(mapped_indices)
                    equivalent_indices[side].update(mapped_indices)

            exact_evaluable = (
                inference_status == "success"
                and bool(exact_indices["a"])
                and bool(exact_indices["b"])
            )
            equivalent_evaluable = (
                inference_status == "success"
                and bool(equivalent_indices["a"])
                and bool(equivalent_indices["b"])
            )
            exact_rank = (
                rank_for_truth(
                    candidates,
                    exact_indices["a"],
                    exact_indices["b"],
                    args.top_k,
                )
                if exact_evaluable
                else None
            )
            equivalent_rank = (
                rank_for_truth(
                    candidates,
                    equivalent_indices["a"],
                    equivalent_indices["b"],
                    args.top_k,
                )
                if equivalent_evaluable
                else None
            )
            source_exact_evaluable = (
                source_inference_status == "success"
                and bool(source_exact_indices["a"])
                and bool(source_exact_indices["b"])
            )
            source_equivalent_evaluable = (
                source_inference_status == "success"
                and bool(source_equivalent_indices["a"])
                and bool(source_equivalent_indices["b"])
            )
            source_exact_rank = (
                rank_for_truth(
                    source_candidates,
                    source_exact_indices["a"],
                    source_exact_indices["b"],
                    args.top_k,
                )
                if source_exact_evaluable
                else None
            )
            source_equivalent_rank = (
                rank_for_truth(
                    source_candidates,
                    source_equivalent_indices["a"],
                    source_equivalent_indices["b"],
                    args.top_k,
                )
                if source_equivalent_evaluable
                else None
            )
            for mapping in mapping_records:
                if mapping.get("status") not in {
                    "mapped",
                    "ambiguous_equivalent",
                }:
                    mapping_failures.append(
                        {
                            "joint_set": joint_path.name,
                            "joint_index": joint_index,
                            "mapping": mapping,
                        }
                    )
            joint_rows.append(
                {
                    "joint_set": joint_path.name,
                    "joint_index": joint_index,
                    "joint_type": joint.get("joint_motion", {}).get(
                        "joint_type"
                    )
                    or joint.get("joint_motion", {}).get("type")
                    or joint.get("type"),
                    "inference_status": inference_status,
                    "source_inference_status": source_inference_status,
                    "truth_entity_pair_type": (
                        f"{exact_entity_types.get('a', 'unknown')}-"
                        f"{exact_entity_types.get('b', 'unknown')}"
                    ),
                    "exact_evaluable": exact_evaluable,
                    "exact_rank": exact_rank,
                    "equivalent_evaluable": equivalent_evaluable,
                    "equivalent_rank": equivalent_rank,
                    "source_exact_evaluable": source_exact_evaluable,
                    "source_exact_rank": source_exact_rank,
                    "source_equivalent_evaluable": (
                        source_equivalent_evaluable
                    ),
                    "source_equivalent_rank": source_equivalent_rank,
                    "exact_truth_indices": {
                        key: sorted(value)
                        for key, value in exact_indices.items()
                    },
                    "equivalent_truth_indices": {
                        key: sorted(value)
                        for key, value in equivalent_indices.items()
                    },
                    "mapping_records": mapping_records,
                }
            )

        pair_rows.append(
            {
                "joint_set": joint_path.name,
                "body_one": body_a,
                "body_two": body_b,
                "joint_count": len(joint_set["joints"]),
                "inference_status": inference_status,
                "inference_reason": inference_reason,
                "source_inference_status": source_inference_status,
                "source_inference_reason": source_inference_reason,
                "source_inference_metadata": source_inference_metadata,
                "prediction_path": str(prediction_path.resolve()),
                **inference_metadata,
            }
        )
        print(
            f"[{pair_index}/{len(joint_files)}] {joint_path.name}: "
            f"{inference_status}"
        )

    # Each body appears in only one sample pair here, but de-duplicate defensively.
    unique_topology = {
        row["body"]: row for row in topology_rows
    }
    topology_rows = list(unique_topology.values())
    face_count_matches = sum(
        row["fusion_face_count"] == row["occt_face_count"]
        for row in topology_rows
    )
    edge_count_matches = sum(
        row["fusion_edge_count"] == row["occt_edge_count"]
        for row in topology_rows
    )
    summary = {
        "pair_count": len(pair_rows),
        "inference_success_pair_count": sum(
            row["inference_status"] == "success" for row in pair_rows
        ),
        "official_node_limit_pair_count": sum(
            row["inference_status"] == "unsupported_official_node_limit"
            for row in pair_rows
        ),
        "joint_count": len(joint_rows),
        "body_count": len(topology_rows),
        "face_count_parity_rate": (
            face_count_matches / len(topology_rows) if topology_rows else None
        ),
        "edge_count_parity_rate": (
            edge_count_matches / len(topology_rows) if topology_rows else None
        ),
        "mapping_failure_count": len(mapping_failures),
        **summarize_rates(joint_rows, "exact"),
        **summarize_rates(joint_rows, "equivalent"),
        **summarize_rates(joint_rows, "source_exact"),
        **summarize_rates(joint_rows, "source_equivalent"),
        "paired_transfer": paired_transfer_summary(joint_rows),
        "inference_status_distribution": dict(
            Counter(row["inference_status"] for row in pair_rows)
        ),
    }
    report = {
        "schema_version": "1.0.0",
        "purpose": (
            "Official pretrained JoinABLe transfer audit on paired Fusion "
            "B-Rep graphs and exported STEP"
        ),
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "epoch": checkpoint.get("epoch"),
            "global_step": checkpoint.get("global_step"),
            "input_features": official_args.input_features,
            "strict_weight_load": True,
        },
        "runtime": environment_summary(),
        "device": str(device),
        "policy": {
            "shadow_mode": True,
            "model_score_can_change_final_acceptance": False,
            "max_node_count": args.max_node_count,
            "topology_mapping_does_not_assume_equal_edge_indices": True,
        },
        "summary": summary,
        "pairs": pair_rows,
        "joints": joint_rows,
        "topology_parity": topology_rows,
        "mapping_failures": mapping_failures,
        "failure_reasons": inference_failures,
        "unavailable_fields": [
            "functional_assembly_validity",
            "permanent_topology_ids_across_reexport",
            "mixed_pool_grouping",
        ],
    }
    write_json(args.output, report)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["inference_success_pair_count"] > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
