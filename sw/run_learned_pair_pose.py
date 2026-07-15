"""Generate learned CAD pair-pose hypotheses from cached JoinABLe B-Rep graphs.

This runner is intentionally independent of the global solver.  It creates an
auditable top-k Pose frontier that can be fed to the existing multi-part stage
only after pair-level holdout checks pass.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for item in (PROJECT_ROOT, PROJECT_ROOT / "sw"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from cad_assembly_agent.tools.joinable_interface_predictor.pretrained_joinable_predictor import body_to_data  # noqa: E402
from joinable_gpu_reproduction.joinable_compat import build_model, load_checkpoint  # noqa: E402
from learned_joint.pose_head_adapter import load_pose_heads, propose_for_joinable_candidate  # noqa: E402


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _bbox_diagonal_mm(graph: dict[str, Any]) -> float:
    box = (graph.get("metadata") or {}).get("bounding_box") or {}
    lower, upper = box.get("min") or [], box.get("max") or []
    if len(lower) != 3 or len(upper) != 3:
        raise ValueError("graph_bbox_unavailable_for_patch_pose_scale")
    return max(math.sqrt(sum((float(upper[i]) - float(lower[i])) ** 2 for i in range(3))), 1e-4)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("joinable_result", type=Path, help="Existing geometry-only JoinABLe result JSON")
    parser.add_argument("graph_a", type=Path, help="Cached B-Rep graph for fixed/source part")
    parser.add_argument("graph_b", type=Path, help="Cached B-Rep graph for moving/target part")
    parser.add_argument("head_checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--candidate-limit", type=int, default=10)
    parser.add_argument("--modes-per-candidate", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.candidate_limit < 1 or args.modes_per_candidate < 1:
        raise ValueError("candidate_limits_must_be_positive")
    source = _read(args.joinable_result)
    inference = source["gnn_inference"]
    pair_scale = float(inference["pair_scale"])
    if pair_scale <= 0:
        raise ValueError("invalid_pair_scale")
    graph_a, graph_b = _read(args.graph_a), _read(args.graph_b)
    data_a, _ = body_to_data(graph_a, pair_scale)
    data_b, _ = body_to_data(graph_b, pair_scale)
    checkpoint, official_args = load_checkpoint(Path(inference["checkpoint"]))
    joinable = build_model(checkpoint, official_args).eval()
    heads, head_metadata = load_pose_heads(args.head_checkpoint, device=args.device)
    # Pose-head labels use the larger part bounding-box diagonal, which is
    # deliberately independent from JoinABLe's checkpoint normalization scale.
    pair_extent_mm = max(_bbox_diagonal_mm(graph_a), _bbox_diagonal_mm(graph_b))
    rows = []
    for candidate in inference["candidates"][: args.candidate_limit]:
        proposals = propose_for_joinable_candidate(
            joinable_model=joinable,
            graph_a=data_a,
            graph_b=data_b,
            candidate=candidate,
            pose_model=heads,
            pair_extent_mm=pair_extent_mm,
            raw_graph_a=graph_a,
            raw_graph_b=graph_b,
            device=args.device,
            top_k=args.modes_per_candidate,
        )
        rows.append({
            "joinable_entity_rank": int(candidate["rank"]),
            "joinable_logit": float(candidate["logit"]),
            "entity_a": candidate["node_a"],
            "entity_b": candidate["node_b"],
            "learned_pose_hypotheses": proposals,
        })
    result = {
        "schema_version": "cad_pair_pose_frontier.v2",
        "status": "review_only",
        "input_contract": {
            "joinable_candidate_source": "B-Rep graph only",
            "head_input": "selected entity plus one-ring frozen JoinABLe embedding and local sampled B-Rep patch",
            "forbidden_model_inputs": ["file_name", "part_name", "case_id", "bom", "solidworks_answer", "joint_type"],
        },
        "head_checkpoint_schema": head_metadata["schema_version"],
        "pair_extent_mm": pair_extent_mm,
        "candidate_count": len(rows),
        "pose_hypotheses": rows,
        "acceptance_boundary": "No candidate is accepted here. Global consistency and OCCT validation remain required.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "candidate_count": len(rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
