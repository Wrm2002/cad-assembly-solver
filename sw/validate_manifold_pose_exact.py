"""Attach OCCT Boolean collision results to a manifold-solver JSON."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from placement_validation import exact_shape_collisions


def _part(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--part requires PART_ID=STEP_PATH")
    key, raw = value.split("=", 1)
    path = Path(raw).resolve()
    if not key or not path.is_file():
        raise argparse.ArgumentTypeError("--part requires an existing STEP path")
    return key, path


def _axis_angle(matrix: list[list[float]]) -> list[float] | None:
    r = matrix
    cosine = max(-1.0, min(1.0, (r[0][0] + r[1][1] + r[2][2] - 1.0) / 2.0))
    angle = math.acos(cosine)
    if angle < 1e-10:
        return None
    if abs(math.pi - angle) < 1e-6:
        axis = [
            math.sqrt(max(0.0, (r[0][0] + 1.0) / 2.0)),
            math.sqrt(max(0.0, (r[1][1] + 1.0) / 2.0)),
            math.sqrt(max(0.0, (r[2][2] + 1.0) / 2.0)),
        ]
        if r[0][1] < 0:
            axis[1] = -axis[1]
        if r[0][2] < 0:
            axis[2] = -axis[2]
    else:
        scale = 2.0 * math.sin(angle)
        axis = [
            (r[2][1] - r[1][2]) / scale,
            (r[0][2] - r[2][0]) / scale,
            (r[1][0] - r[0][1]) / scale,
        ]
    norm = math.sqrt(sum(value * value for value in axis))
    if norm <= 1e-12:
        axis = [1.0, 0.0, 0.0]
    else:
        axis = [value / norm for value in axis]
    return axis + [math.degrees(angle)]


def _placement(matrix: list[list[float]]) -> dict:
    result = {"translate": [float(matrix[i][3]) for i in range(3)]}
    axis_angle = _axis_angle(matrix)
    if axis_angle is not None:
        result["rotate_sequence"] = [{"axis_angle": axis_angle}]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--part", action="append", type=_part, required=True)
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sources = dict(args.part)
    parents = {path.parent for path in sources.values()}
    if len(parents) != 1:
        parser.error("all STEP sources must be in one directory")
    folder = next(iter(parents))
    data = json.loads(args.input.read_text(encoding="utf-8"))
    for row in (data.get("hypotheses") or [])[: max(0, int(args.top_n))]:
        components = [
            {
                "source": sources[part].name,
                "placement": _placement(row["part_poses"][part]),
            }
            for part in sorted(sources)
        ]
        exact = exact_shape_collisions(folder, components)
        if exact.get("status") != "success":
            status = "uncertain"
        else:
            status = "failed" if exact.get("collisions") else "valid"
        row["exact_validation"] = {"status": status, "occt": exact}
    data["hypotheses"].sort(key=lambda row: (
        {"valid": 0, "not_checked": 1, "uncertain": 2, "failed": 3}.get(
            row.get("exact_validation", {}).get("status"), 4
        ),
        -int(row.get("consistent_cycle_count", 0)),
        row.get("optimizer", {}).get("cost", float("inf")),
    ))
    destination = args.output or args.input
    destination.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "checked": min(len(data.get("hypotheses") or []), max(0, int(args.top_n))),
        "valid": sum(row.get("exact_validation", {}).get("status") == "valid" for row in data.get("hypotheses") or []),
        "failed": sum(row.get("exact_validation", {}).get("status") == "failed" for row in data.get("hypotheses") or []),
        "uncertain": sum(row.get("exact_validation", {}).get("status") == "uncertain" for row in data.get("hypotheses") or []),
        "output": str(destination.resolve()),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
