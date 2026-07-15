"""Calibrate Fusion360 recorded contact faces against transformed OBJ geometry.

This is an audit utility, not a training-data builder.  It follows the source
assembly hierarchy, obtains the two faces named by a recorded ``contacts``
entry, transforms both face point sets to the assembly frame, then measures
bidirectional nearest-surface *vertex* gaps.  The result establishes whether
the published occurrence transforms and indexed OBJ faces are suitable for
constructing a true contact-Pose dataset.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from fusion360_common import build_parts, discover_assembly_files, entity_part_id, load_json  # noqa: E402


def _obj_face_points(path: Path, cache: dict[Path, dict[int, np.ndarray]]) -> dict[int, np.ndarray]:
    if path in cache:
        return cache[path]
    vertices: list[list[float]] = []
    faces: dict[int, list[int]] = {}
    current: int | None = None
    with path.open("r", encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            if line.startswith("v "):
                try:
                    vertices.append([float(value) for value in line.split()[1:4]])
                except ValueError:
                    continue
            elif line.startswith("g face "):
                try:
                    current = int(line.split()[2])
                    faces.setdefault(current, [])
                except (IndexError, ValueError):
                    current = None
            elif line.startswith("f ") and current is not None:
                for token in line.split()[1:]:
                    try:
                        index = int(token.split("/")[0])
                        faces[current].append(index - 1 if index > 0 else len(vertices) + index)
                    except ValueError:
                        continue
    array = np.asarray(vertices, dtype=np.float64)
    result = {
        index: array[np.unique(np.asarray(indices, dtype=int))]
        for index, indices in faces.items()
        if indices
    }
    cache[path] = result
    return result


def _obj_path(part: dict[str, Any]) -> Path | None:
    for candidate in (part.get("geometry") or {}).get("candidates") or []:
        if candidate.get("format") == "obj" and candidate.get("exists"):
            return Path(candidate["path"])
    return None


def _transform(points: np.ndarray, matrix: list[list[float]]) -> np.ndarray:
    homo = np.concatenate((points, np.ones((len(points), 1))), axis=1)
    return (np.asarray(matrix, dtype=float) @ homo.T).T[:, :3]


def _bidirectional_gap(left: np.ndarray, right: np.ndarray, *, max_points: int = 384) -> tuple[float, float, float]:
    if len(left) > max_points:
        left = left[np.linspace(0, len(left) - 1, max_points).round().astype(int)]
    if len(right) > max_points:
        right = right[np.linspace(0, len(right) - 1, max_points).round().astype(int)]
    distances = np.linalg.norm(left[:, None, :] - right[None, :, :], axis=-1)
    left_min, right_min = distances.min(axis=1), distances.min(axis=0)
    return float(left_min.mean()), float(right_min.mean()), float(min(left_min.min(), right_min.min()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--limit", type=int, default=500, help="Maximum mapped contact pairs to check; zero means all.")
    args = parser.parse_args()
    files = discover_assembly_files(args.input_root)
    cache: dict[Path, dict[int, np.ndarray]] = {}
    rows, failures = [], Counter()
    for assembly_file in files:
        data = load_json(assembly_file)
        parts, _ = build_parts(data, assembly_file.parent, visible_only=True)
        by_id = {part["part_id"]: part for part in parts}
        for contact_index, contact in enumerate(data.get("contacts") or []):
            if args.limit and len(rows) >= args.limit:
                break
            first, second = contact.get("entity_one") or {}, contact.get("entity_two") or {}
            part_a, part_b = entity_part_id(first), entity_part_id(second)
            if part_a not in by_id or part_b not in by_id:
                failures["contact_part_unmapped"] += 1; continue
            if first.get("type") != "BRepFace" or second.get("type") != "BRepFace":
                failures["contact_not_face_pair"] += 1; continue
            path_a, path_b = _obj_path(by_id[part_a]), _obj_path(by_id[part_b])
            if path_a is None or path_b is None:
                failures["contact_obj_missing"] += 1; continue
            face_a = _obj_face_points(path_a, cache).get(int(first.get("index", -1)))
            face_b = _obj_face_points(path_b, cache).get(int(second.get("index", -1)))
            if face_a is None or face_b is None:
                failures["contact_face_index_missing_in_obj"] += 1; continue
            a_world = _transform(face_a, by_id[part_a]["transform"])
            b_world = _transform(face_b, by_id[part_b]["transform"])
            gap_a, gap_b, minimum = _bidirectional_gap(a_world, b_world)
            rows.append({
                "assembly_id": assembly_file.parent.name,
                "contact_index": contact_index,
                "surface_types": [first.get("surface_type"), second.get("surface_type")],
                "mean_gap_cm": (gap_a + gap_b) / 2.0,
                "minimum_gap_cm": minimum,
                "is_close_at_0_1mm": bool(minimum <= 0.01),
            })
        if args.limit and len(rows) >= args.limit:
            break
    gaps = np.asarray([row["mean_gap_cm"] for row in rows], dtype=float)
    minimums = np.asarray([row["minimum_gap_cm"] for row in rows], dtype=float)
    report = {
        "schema_version": "fusion360_contact_mesh_geometry_audit.v1",
        "input_root": str(args.input_root.resolve()),
        "contacts_checked": len(rows),
        "mean_face_gap_cm_quantiles": np.quantile(gaps, [0, .25, .5, .75, .9, .95, .99, 1]).tolist() if len(gaps) else [],
        "minimum_face_gap_cm_quantiles": np.quantile(minimums, [0, .25, .5, .75, .9, .95, .99, 1]).tolist() if len(minimums) else [],
        "minimum_gap_within_0_1mm_rate": float((minimums <= .01).mean()) if len(minimums) else 0.0,
        "mean_gap_within_1mm_rate": float((gaps <= .1).mean()) if len(gaps) else 0.0,
        "failures": dict(failures),
        "interpretation": "OBJ face samples and occurrence transforms must be calibrated before they become contact-Pose labels.",
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
