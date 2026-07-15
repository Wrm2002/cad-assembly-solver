"""Compatibility facade for released JoinABLe checkpoint inference.

The canonical implementation now lives in :mod:`joinable_e2e` and the audited
predictor under ``cad_assembly_agent/tools/joinable_interface_predictor``.
This module keeps the historical CLI/function name without maintaining a
second graph builder.  In particular, it no longer drops edge entities or
constructs cross-body graph edges without the body-B index offset.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from joinable_e2e import run_pipeline


def run_inference_on_pair(
    step1: str | Path,
    step2: str | Path,
    output_dir: str | Path,
    model: Any = None,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    if model is not None:
        raise ValueError(
            "Passing an in-memory legacy model is no longer supported; pass "
            "a checkpoint through joinable_e2e.run_pipeline instead."
        )
    result = run_pipeline(
        Path(step1),
        Path(step2),
        output_dir=Path(output_dir),
        top_k=top_k,
        run_search=False,
    )
    predictions = []
    for candidate in result["gnn_inference"]["candidates"]:
        node_a = candidate["node_a"]
        node_b = candidate["node_b"]
        predictions.append({
            "rank": candidate["rank"],
            "face1_idx": node_a["joinable_node_index"],
            "face2_idx": node_b["joinable_node_index"],
            "entity1_type": node_a["entity_type"],
            "entity2_type": node_b["entity_type"],
            "probability": candidate["probability"],
            "logit": candidate["logit"],
            "node_a": node_a,
            "node_b": node_b,
        })
    return predictions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step1")
    parser.add_argument("step2")
    parser.add_argument("--output", "-o", default="brep_output")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    predictions = run_inference_on_pair(
        args.step1, args.step2, args.output, top_k=args.top_k
    )
    for row in predictions:
        print(
            f"#{row['rank']} p={row['probability']:.6f} "
            f"{row['entity1_type']}:{row['face1_idx']} <-> "
            f"{row['entity2_type']}:{row['face2_idx']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
