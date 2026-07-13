"""End-to-end JoinABLe interface ranking and pair-pose proposal for STEP.

This is the canonical reproduction entry point.  It reuses the audited OCCT
B-Rep adapter and released JoinABLe checkpoint, then runs the paper's top-k
axis/offset/rotation/flip search with an SDF overlap/contact objective.

The output is a ranked pose proposal list.  It is not an automatic assembly
acceptance decision; downstream OCCT exact collision and group closure checks
remain mandatory.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Batch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, PROJECT_ROOT / "sw"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cad_assembly_agent.tools.joinable_interface_predictor.pretrained_joinable_predictor import (  # noqa: E402
    body_to_data,
    make_joint_graph,
    validate_graph,
)
from joinable_gpu_reproduction.joinable_compat import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    batch_to_device,
    build_model,
    load_checkpoint,
)
from joinable_migration_audit.step_to_brep_graph_probe import (  # noqa: E402
    extract_graph,
)
from pose_search import (  # noqa: E402
    JointAxisSeed,
    JoinablePoseSearch,
    generate_axial_rotation_hypotheses,
    matrix_to_placement,
)
from placement_validation import exact_shape_collisions  # noqa: E402
from learned_joint import attach_pose_initials, build_joint_hypotheses  # noqa: E402


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def extract_brep_graphs(
    step_a: Path, step_b: Path, cache_dir: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        cache_dir / f"{step_a.stem}.brep_graph.json",
        cache_dir / f"{step_b.stem}.brep_graph.json",
    ]
    graphs = []
    for step_path, cache_path in zip((step_a, step_b), paths):
        if cache_path.exists():
            graph = json.loads(cache_path.read_text(encoding="utf-8"))
            source_hash = graph.get("source_geometry_sha256")
            # The extractor owns the exact hash contract.  Re-extract only if
            # the cached source path no longer refers to this input.
            if (
                graph.get("source_step_path") != str(step_path.resolve())
                or not source_hash
                or graph.get("metadata", {}).get("adapter_version") != "2.1.0"
            ):
                graph = extract_graph(step_path)
                _write_json(cache_path, graph)
        else:
            graph = extract_graph(step_path)
            _write_json(cache_path, graph)
        graphs.append(graph)
    return graphs[0], graphs[1]


def step_to_stl(step_path: Path, output_dir: Path) -> Path:
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.StlAPI import StlAPI_Writer

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{step_path.stem}.stl"
    if output.exists() and output.stat().st_mtime_ns >= step_path.stat().st_mtime_ns:
        return output
    reader = STEPControl_Reader()
    if reader.ReadFile(str(step_path)) != IFSelect_RetDone:
        raise RuntimeError(f"STEP read failed: {step_path}")
    reader.TransferRoots()
    shape = reader.OneShape()
    mesher = BRepMesh_IncrementalMesh(shape, 0.1, False, 0.5)
    mesher.Perform()
    writer = StlAPI_Writer()
    writer.SetASCIIMode(False)
    writer.Write(shape, str(output))
    return output


def _public_node(node: dict[str, Any]) -> dict[str, Any]:
    result = {
        "entity_id": node["node_id"],
        "entity_type": node["entity_type"],
        "topology_index": int(node["occt_topology_index"]),
        "joinable_node_index": int(node["joinable_node_index"]),
        "joinable_entity_type": node["joinable_entity_type"],
        "geometry_type": (
            node.get("surface_type")
            if node["entity_type"] == "face"
            else node.get("curve_type")
        ),
        "geometry_signature": node["geometry_signature"],
    }
    for key in (
        "axis_origin",
        "axis_direction",
        "centroid",
        "normal",
        "radius",
        "orientation",
    ):
        if node.get(key) is not None:
            result[key] = node[key]
    return result


def _orthogonal(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    basis = np.eye(3)[int(np.argmin(np.abs(value)))]
    first = basis - float(np.dot(basis, value)) * value
    first /= np.linalg.norm(first)
    return first, np.cross(value, first)


def _local_asymmetric_direction(
    graph: dict[str, Any], public_node: dict[str, Any]
) -> tuple[list[float] | None, float, int]:
    """Estimate a tangent witness from the selected entity's B-Rep 1-ring."""

    node_id = str(public_node.get("entity_id"))
    nodes = {str(row.get("node_id")): row for row in graph.get("nodes") or []}
    center = public_node.get("axis_origin") or public_node.get("centroid")
    direction = public_node.get("axis_direction") or public_node.get("normal")
    if center is None or direction is None:
        return None, 0.0, 0
    center = np.asarray(center, dtype=float)
    z_axis = np.asarray(direction, dtype=float)
    z_norm = float(np.linalg.norm(z_axis))
    if center.shape != (3,) or z_axis.shape != (3,) or z_norm <= 1e-12:
        return None, 0.0, 0
    z_axis /= z_norm
    first, second = _orthogonal(z_axis)
    neighbors = []
    for edge in graph.get("edges") or []:
        src, dst = str(edge.get("src")), str(edge.get("dst"))
        other = dst if src == node_id else src if dst == node_id else None
        if other is None or other not in nodes:
            continue
        row = nodes[other]
        point = row.get("axis_origin") or row.get("centroid")
        if point is None:
            continue
        delta = np.asarray(point, dtype=float) - center
        delta -= float(np.dot(delta, z_axis)) * z_axis
        if float(np.linalg.norm(delta)) > 1e-8:
            neighbors.append(delta)
    if len(neighbors) < 2:
        return None, 0.0, len(neighbors)
    a = np.asarray([float(np.dot(value, first)) for value in neighbors])
    b = np.asarray([float(np.dot(value, second)) for value in neighbors])
    aa, bb, ab = float(np.dot(a, a)), float(np.dot(b, b)), float(np.dot(a, b))
    trace = aa + bb
    if trace <= 1e-12:
        return None, 0.0, len(neighbors)
    gap = math.sqrt(max(0.0, (aa - bb) ** 2 + 4.0 * ab ** 2))
    asymmetry = gap / trace
    angle = 0.5 * math.atan2(2.0 * ab, aa - bb)
    witness = math.cos(angle) * first + math.sin(angle) * second
    witness /= np.linalg.norm(witness)
    return witness.tolist(), float(asymmetry), len(neighbors)


def _public_node_with_local_patch(
    graph: dict[str, Any], node: dict[str, Any]
) -> dict[str, Any]:
    result = _public_node(node)
    direction, score, count = _local_asymmetric_direction(graph, result)
    if direction is not None:
        result["local_direction"] = direction
    result["local_asymmetry_score"] = score
    result["local_witness_count"] = count
    return result


def run_gnn_inference(
    graph_a: dict[str, Any],
    graph_b: dict[str, Any],
    *,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    device_name: str = "cpu",
    top_k: int = 20,
) -> dict[str, Any]:
    validate_graph(graph_a, Path("part_a"))
    validate_graph(graph_b, Path("part_b"))
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
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device("cuda:0" if device_name == "cuda" else "cpu")
    model.to(device)
    batch = batch_to_device(batch, device)
    with torch.inference_mode():
        logits = model(*batch)
        probabilities = torch.softmax(logits, dim=0)

    retained = min(max(1, int(top_k)), int(logits.numel()))
    top_indices = torch.argsort(
        logits, descending=True, stable=True
    )[:retained].detach().cpu()
    logits = logits.detach().cpu()
    probabilities = probabilities.detach().cpu()
    candidates = []
    for rank, flat_index in enumerate(top_indices.tolist(), 1):
        node_a_index = int(joint_graph.edge_index[0, flat_index])
        node_b_index = (
            int(joint_graph.edge_index[1, flat_index]) - data_a.num_nodes
        )
        candidates.append({
            "rank": rank,
            "node_a": _public_node_with_local_patch(graph_a, nodes_a[node_a_index]),
            "node_b": _public_node_with_local_patch(graph_b, nodes_b[node_b_index]),
            "logit": float(logits[flat_index]),
            "probability": float(probabilities[flat_index]),
        })
    return {
        "checkpoint": str(checkpoint_path.resolve()),
        "device": str(device),
        "input_features": official_args.input_features,
        "pair_scale": pair_scale,
        "total_candidates": int(logits.numel()),
        "top_k": retained,
        "candidates": candidates,
    }


def _axis_from_node(
    node: dict[str, Any]
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    origin = node.get("axis_origin") or node.get("centroid")
    direction = node.get("axis_direction") or node.get("normal")
    if origin is None or direction is None:
        return None
    origin_array = np.asarray(origin, dtype=float)
    direction_array = np.asarray(direction, dtype=float)
    length = float(np.linalg.norm(direction_array))
    if origin_array.shape != (3,) or direction_array.shape != (3,) or length <= 1e-12:
        return None
    direction_array /= length
    return tuple(origin_array.tolist()), tuple(direction_array.tolist())


def joint_axis_seed(candidate: dict[str, Any]) -> JointAxisSeed | None:
    fixed = _axis_from_node(candidate["node_a"])
    moving = _axis_from_node(candidate["node_b"])
    if fixed is None or moving is None:
        return None
    return JointAxisSeed(
        moving_origin=moving[0],
        moving_direction=moving[1],
        fixed_origin=fixed[0],
        fixed_direction=fixed[1],
        prediction_rank=int(candidate["rank"]),
        prediction_score=float(candidate["logit"]),
        entity_a=str(candidate["node_a"]["entity_id"]),
        entity_b=str(candidate["node_b"]["entity_id"]),
    )


def _select_diverse_rotation_rows(
    rows: list[dict[str, Any]], maximum: int
) -> list[dict[str, Any]]:
    """Keep high-evidence rotations plus a bounded angularly diverse frontier."""

    unique: dict[float, dict[str, Any]] = {}
    for row in rows:
        angle = (float(row["rotation_degrees"]) + 180.0) % 360.0 - 180.0
        key = round(180.0 if abs(angle + 180.0) < 1e-9 else angle, 6)
        current = unique.get(key)
        quality = (
            float(row.get("score", 0.0)),
            row.get("evidence_kind") == "circular_pattern_correspondence",
            not bool(row.get("geometry_symmetry_only", False)),
        )
        if current is None or quality > (
            float(current.get("score", 0.0)),
            current.get("evidence_kind") == "circular_pattern_correspondence",
            not bool(current.get("geometry_symmetry_only", False)),
        ):
            unique[key] = row
    pool = list(unique.values())
    if not pool:
        return []
    selected: list[dict[str, Any]] = []
    anchor = min(pool, key=lambda row: abs(float(row["rotation_degrees"])))
    selected.append(anchor)
    while len(selected) < min(maximum, len(pool)):
        remaining = [row for row in pool if row not in selected]

        def diversity(row: dict[str, Any]) -> tuple[float, float, bool]:
            angle = float(row["rotation_degrees"])
            nearest = min(
                abs(((angle - float(other["rotation_degrees"]) + 180.0) % 360.0) - 180.0)
                for other in selected
            )
            return (
                nearest,
                float(row.get("score", 0.0)),
                not bool(row.get("geometry_symmetry_only", False)),
            )

        selected.append(max(remaining, key=diversity))
    return selected


def attach_axial_orientation_hypotheses(
    graph_a: dict[str, Any],
    graph_b: dict[str, Any],
    seeds: list[JointAxisSeed],
    *,
    maximum_rotations_per_seed: int = 8,
) -> tuple[list[JointAxisSeed], list[dict[str, Any]]]:
    """Attach feature-derived axial rotations to each pairwise axis seed.

    The evidence is intentionally proposal-only. In particular, a uniform hole
    pattern produces geometry-symmetry variants, not a claim that those poses
    are functionally equivalent or correct.
    """

    attached: list[JointAxisSeed] = []
    audits: list[dict[str, Any]] = []
    for seed in seeds:
        try:
            evidence = generate_axial_rotation_hypotheses(
                graph_a,
                graph_b,
                fixed_axis_origin=seed.fixed_origin,
                fixed_axis_direction=seed.fixed_direction,
                moving_axis_origin=seed.moving_origin,
                moving_axis_direction=seed.moving_direction,
            )
            rows = _select_diverse_rotation_rows(
                list(evidence.get("rotation_hypotheses") or []),
                max(1, int(maximum_rotations_per_seed)),
            )
            rotations = tuple(float(row["rotation_degrees"]) for row in rows)
            attached.append(replace(
                seed,
                rotation_seed_degrees=rotations or (0.0,),
            ))
            audits.append({
                "entity_a": seed.entity_a,
                "entity_b": seed.entity_b,
                "prediction_rank": seed.prediction_rank,
                "rotation_seed_degrees": list(rotations or (0.0,)),
                "evidence": evidence,
            })
        except Exception as exc:
            attached.append(seed)
            audits.append({
                "entity_a": seed.entity_a,
                "entity_b": seed.entity_b,
                "prediction_rank": seed.prediction_rank,
                "rotation_seed_degrees": [0.0],
                "status": "unavailable",
                "error": str(exc),
            })
    return attached, audits


def run_pipeline(
    step_a: Path,
    step_b: Path,
    *,
    output_dir: Path | None = None,
    checkpoint_path: Path | None = None,
    device: str = "cpu",
    top_k: int = 20,
    pose_top_k: int = 5,
    run_search: bool = True,
    search_budget: int = 80,
    sample_count: int = 2048,
    exact_check_limit: int = 12,
) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir = output_dir or Path("joinable_e2e_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_a, graph_b = extract_brep_graphs(
        step_a, step_b, output_dir / "cache"
    )
    inference = run_gnn_inference(
        graph_a,
        graph_b,
        checkpoint_path=checkpoint_path or DEFAULT_CHECKPOINT,
        device_name=device,
        top_k=top_k,
    )
    raw_axis_seeds = [
        seed
        for candidate in inference["candidates"]
        if (seed := joint_axis_seed(candidate)) is not None
    ]
    axis_seeds, axial_orientation_audit = attach_axial_orientation_hypotheses(
        graph_a, graph_b, raw_axis_seeds
    )
    orientation_by_pair = {
        (str(row.get("entity_a")), str(row.get("entity_b"))): row
        for row in axial_orientation_audit
    }
    manifold_candidates = []
    for candidate in inference["candidates"]:
        enriched = dict(candidate)
        key = (
            str(candidate["node_a"].get("entity_id")),
            str(candidate["node_b"].get("entity_id")),
        )
        orientation = orientation_by_pair.get(key, {})
        enriched["rotation_hypotheses"] = (
            (orientation.get("evidence") or {}).get("rotation_hypotheses")
            or []
        )
        manifold_candidates.append(enriched)
    joint_hypotheses = build_joint_hypotheses(
        "part_0",
        "part_1",
        manifold_candidates,
        maximum_phases_per_entity_pair=4,
        enumerate_polarity=True,
    )
    pose_results: list[dict[str, Any]] = []
    if run_search and axis_seeds:
        stl_a = step_to_stl(step_a, output_dir / "stl")
        stl_b = step_to_stl(step_b, output_dir / "stl")
        searcher = JoinablePoseSearch(
            stl_a,
            stl_b,
            sample_count=sample_count,
            budget=search_budget,
            objective="default",
        )
        for result in searcher.search(axis_seeds, top_k=pose_top_k):
            row = result.to_dict()
            row["placement_part_b_in_part_a_frame"] = matrix_to_placement(
                np.asarray(result.transform, dtype=float)
            )
            pose_results.append(row)

    best = pose_results[0] if pose_results else None
    best_exact_collision_free = None
    for row in pose_results[: max(0, int(exact_check_limit))]:
        components = [
            {
                "id": "fixed",
                "label": step_a.stem,
                "source": str(step_a.resolve()),
                "placement": {"translate": [0.0, 0.0, 0.0]},
            },
            {
                "id": "moving",
                "label": step_b.stem,
                "source": str(step_b.resolve()),
                "placement": row["placement_part_b_in_part_a_frame"],
            },
        ]
        exact = exact_shape_collisions(output_dir, components)
        row["exact_collision"] = exact
        if (
            best_exact_collision_free is None
            and exact.get("status") == "success"
            and not exact.get("collisions")
        ):
            best_exact_collision_free = row
    joint_hypotheses = attach_pose_initials(joint_hypotheses, pose_results)
    output = {
        "schema_version": "joinable_e2e.v2",
        "pipeline": "released_joinable_topk_plus_constraint_manifold_pose_frontier",
        "part_a_fixed": str(step_a.resolve()),
        "part_b_moving": str(step_b.resolve()),
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "gnn_inference": inference,
        "joint_hypotheses": {
            "schema_version": "pair_joint_hypothesis.v1",
            "count": len(joint_hypotheses),
            "rows": [row.to_dict() for row in joint_hypotheses],
            "contract": (
                "Entity pairs are learned by JoinABLe; analytic B-Rep geometry "
                "lifts them to local frames and free-DOF manifolds.  No row is "
                "a fixed assembly answer or an acceptance decision."
            ),
        },
        "pose_search": {
            "enabled": bool(run_search),
            "axis_seed_count": len(axis_seeds),
            "searched_top_k": min(pose_top_k, len(axis_seeds)),
            "budget_per_flip": search_budget,
            "sample_count": sample_count,
            "objective": "JoinABLe default: overlap if severe, else overlap-10*contact",
            "flip_adaptation": (
                "axis-direction sign enumeration using proper rigid rotations; "
                "the released reflection is not emitted as a CAD pose"
            ),
            "axial_orientation_evidence": axial_orientation_audit,
            "rotation_seed_policy": (
                "B-Rep interface periodicity and weak directional witnesses "
                "are retained as pose proposals; neither establishes "
                "functional equivalence or an acceptance decision."
            ),
            "results": pose_results,
            "best": best,
            "exact_check_limit": exact_check_limit,
            "best_exact_collision_free": best_exact_collision_free,
        },
        "acceptance_boundary": {
            "can_auto_accept": False,
            "requires_exact_occt_collision": True,
            "requires_selected_constraint_closure": True,
            "requires_group_consistency": True,
        },
    }
    _write_json(output_dir / "joinable_e2e_result.json", output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step_a", type=Path)
    parser.add_argument("step_b", type=Path)
    parser.add_argument("-o", "--output-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--pose-top-k", type=int, default=5)
    parser.add_argument("--search-budget", type=int, default=80)
    parser.add_argument("--sample-count", type=int, default=2048)
    parser.add_argument("--exact-check-limit", type=int, default=12)
    parser.add_argument("--no-search", action="store_true")
    args = parser.parse_args()
    for path in (args.step_a, args.step_b):
        if not path.exists():
            parser.error(f"STEP file not found: {path}")
    result = run_pipeline(
        args.step_a,
        args.step_b,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint,
        device=args.device,
        top_k=args.top_k,
        pose_top_k=args.pose_top_k,
        run_search=not args.no_search,
        search_budget=args.search_budget,
        sample_count=args.sample_count,
        exact_check_limit=args.exact_check_limit,
    )
    best = result["pose_search"]["best"]
    best_exact = result["pose_search"]["best_exact_collision_free"]
    print(json.dumps({
        "status": "ok",
        "output": str((args.output_dir or Path("joinable_e2e_output")) / "joinable_e2e_result.json"),
        "candidate_count": result["gnn_inference"]["total_candidates"],
        "axis_seed_count": result["pose_search"]["axis_seed_count"],
        "best_cost": best["evaluation"]["cost"] if best else None,
        "best_overlap": best["evaluation"]["overlap"] if best else None,
        "best_contact": best["evaluation"]["contact"] if best else None,
        "exact_collision_free_pose_found": best_exact is not None,
        "exact_collision_free_rank": (
            best_exact["prediction_rank"] if best_exact else None
        ),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
