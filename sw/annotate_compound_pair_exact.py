"""Annotate rigid compound pair candidates with isolated OCCT collision checks.

The annotation is proposal hygiene, not assembly acceptance.  A pair that is
collision-free can still be separated, have the wrong symmetry phase, or be
functionally invalid.  Failed rigid candidates are retained for audit but are
deprioritized before the multi-part solve.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_pair_probe(
    frontier: dict[str, Any], source: str, target: str
) -> tuple[dict[str, Any], list[int]]:
    rows = (frontier.get("joint_hypotheses") or {}).get("rows") or []
    hypotheses = []
    raw_indices = []
    identity = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    for raw_index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        if not str(row.get("manifold_type", "")).startswith("compound_"):
            continue
        transform = row.get("initial_pose_b_in_a")
        if not isinstance(transform, list) or len(transform) != 4:
            continue
        raw_indices.append(raw_index)
        hypotheses.append({
            "hypothesis_id": f"compound_pair_raw_{raw_index:06d}",
            "raw_row_index": raw_index,
            "part_poses": {source: identity, target: transform},
            "consistent_cycle_count": 0,
            "optimizer": {"status": "not_run", "cost": 0.0},
            "exact_validation": {"status": "not_checked"},
            "accepted": False,
            "review_required": True,
        })
    return {
        "schema_version": "compound_pair_exact_probe.v1",
        "status": "review_required",
        "accepted": False,
        "review_required": True,
        "part_ids": [source, target],
        "hypothesis_count": len(hypotheses),
        "hypotheses": hypotheses,
    }, raw_indices


def annotate_frontier(
    frontier: dict[str, Any], exact_probe: dict[str, Any]
) -> dict[str, Any]:
    output = copy.deepcopy(frontier)
    rows = output.setdefault("joint_hypotheses", {}).setdefault("rows", [])
    counts = {"valid": 0, "failed": 0, "uncertain": 0, "not_checked": 0}
    for hypothesis in exact_probe.get("hypotheses") or []:
        raw_index = hypothesis.get("raw_row_index")
        if not isinstance(raw_index, int) or not 0 <= raw_index < len(rows):
            continue
        exact = hypothesis.get("exact_validation") or {}
        status = str(exact.get("status", "not_checked"))
        counts[status] = counts.get(status, 0) + 1
        collisions = (exact.get("occt") or {}).get("collisions") or []
        volumes = [
            float(row.get("intersection_volume_mm3", 0.0) or 0.0)
            for row in collisions if isinstance(row, dict)
        ]
        provenance = rows[raw_index].setdefault("provenance", {})
        provenance.update({
            "pair_exact_status": status,
            "pair_exact_collision_free": status == "valid",
            "pair_exact_collision_count": len(collisions),
            "pair_exact_intersection_volume_mm3": sum(volumes),
            "pair_exact_is_acceptance": False,
        })
    output["joint_hypotheses"]["compound_pair_exact_audit"] = {
        "schema_version": "compound_pair_exact_annotation.v1",
        "checked_count": sum(counts.values()),
        "status_counts": counts,
        "acceptance_boundary": (
            "Pair non-collision only prioritizes candidates; multi-part "
            "contact, precision, closure and functional gates remain required."
        ),
    }
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frontier", type=Path)
    parser.add_argument("source")
    parser.add_argument("target")
    parser.add_argument("part_a", type=Path)
    parser.add_argument("part_b", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--occt-python", type=Path, required=True)
    args = parser.parse_args()
    frontier = _read(args.frontier)
    probe, raw_indices = build_pair_probe(frontier, args.source, args.target)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    probe_input = args.output.with_suffix(".pair_exact_input.json")
    probe_output = args.output.with_suffix(".pair_exact_output.json")
    probe_input.write_text(
        json.dumps(probe, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if raw_indices:
        command = [
            str(args.occt_python),
            "sw/validate_manifold_pose_exact.py",
            "--input", str(probe_input),
            "--part", f"{args.source}={args.part_a}",
            "--part", f"{args.target}={args.part_b}",
            "--top-n", str(len(raw_indices)),
            "--output", str(probe_output),
        ]
        result = subprocess.run(
            command, cwd=ROOT, text=True, capture_output=True
        )
        if result.returncode:
            raise RuntimeError(
                "pair_exact_subprocess_failed:\n"
                + result.stdout[-2000:] + "\n" + result.stderr[-4000:]
            )
        checked = _read(probe_output)
    else:
        checked = probe
        probe_output.write_text(
            json.dumps(checked, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    annotated = annotate_frontier(frontier, checked)
    args.output.write_text(
        json.dumps(annotated, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    audit = annotated["joint_hypotheses"]["compound_pair_exact_audit"]
    print(json.dumps({**audit, "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
