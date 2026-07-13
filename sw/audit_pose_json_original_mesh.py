"""Audit global poses on original high-resolution meshes without claiming OCCT."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "sw") not in sys.path:
    sys.path.insert(0, str(ROOT / "sw"))

from learned_joint.mesh_residuals import MeshContactResidualProvider  # noqa: E402


@dataclass(frozen=True)
class _Factor:
    source: str
    target: str


def _mesh(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--mesh requires PART_ID=STL_PATH")
    key, raw = value.split("=", 1)
    path = Path(raw)
    if not key or not path.is_file():
        raise argparse.ArgumentTypeError("--mesh requires an existing STL")
    return key, path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--mesh", action="append", type=_mesh, required=True)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    meshes = dict(args.mesh)
    provider = MeshContactResidualProvider(meshes, sample_count=args.samples)
    factors = [
        _Factor(a, b)
        for index, a in enumerate(sorted(meshes))
        for b in sorted(meshes)[index + 1:]
    ]
    data = json.loads(args.input.read_text(encoding="utf-8"))
    rows = []
    for hypothesis in (data.get("hypotheses") or [])[: max(0, args.top_n)]:
        poses = {
            part: np.asarray(matrix, dtype=float)
            for part, matrix in hypothesis["part_poses"].items()
        }
        audit = provider.audit(poses, factors)
        maximum_overlap = audit["maximum_overlap"]
        minimum_contact = audit["minimum_selected_contact"]
        closest = min(
            (row["closest_distance"] for row in audit["pair_scores"]),
            default=float("inf"),
        )
        if maximum_overlap >= 0.10:
            status = "likely_penetrating"
        elif closest <= 1.0 or minimum_contact > 0.0:
            status = "near_contact_no_exact_authority"
        else:
            status = "likely_separated"
        rows.append({
            "hypothesis_id": hypothesis["hypothesis_id"],
            "status": status,
            "maximum_overlap": maximum_overlap,
            "minimum_selected_contact": minimum_contact,
            "closest_distance": closest,
            "audit": audit,
        })
    output = {
        "schema_version": "original_mesh_pose_audit.v1",
        "authority": "sampled_oriented_distance_only_not_occt",
        "cannot_auto_accept": True,
        "samples": args.samples,
        "rows": rows,
        "status_counts": {
            status: sum(row["status"] == status for row in rows)
            for status in (
                "near_contact_no_exact_authority",
                "likely_penetrating",
                "likely_separated",
            )
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["status_counts"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
