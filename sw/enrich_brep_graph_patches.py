"""Add geometry-only face samples to an existing JoinABLe B-Rep graph.

The released JoinABLe checkpoint does not require UV/curve grids, so the
historic graph cache intentionally omitted them.  The contact-pose sidecar
does require a local surface patch.  This utility deterministically
re-imports the same STEP file with OCCT, triangulates each indexed face and
adds sampled points and outward directions without changing node indices or
checkpoint features.  No file name, case id or assembly answer is encoded in
the generated arrays.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_step(path: Path) -> Any:
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.STEPControl import STEPControl_Reader

    reader = STEPControl_Reader()
    if reader.ReadFile(str(path)) != IFSelect_RetDone:
        raise RuntimeError(f"step_read_failed:{path}")
    reader.TransferRoots()
    return reader.OneShape()


def _point(transform: Any, value: Any) -> list[float]:
    transformed = value.Transformed(transform)
    return [float(transformed.X()), float(transformed.Y()), float(transformed.Z())]


def _cross(a: list[float], b: list[float], c: list[float]) -> list[float]:
    first = [b[i] - a[i] for i in range(3)]
    second = [c[i] - a[i] for i in range(3)]
    value = [
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    ]
    length = max(sum(item * item for item in value) ** 0.5, 1e-12)
    return [item / length for item in value]


def _sample_face(face: Any, count: int) -> tuple[list[list[float]], list[list[float]]]:
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.TopAbs import TopAbs_REVERSED
    from OCC.Core.TopLoc import TopLoc_Location

    location = TopLoc_Location()
    mesh = BRep_Tool.Triangulation(face, location)
    if mesh is None or mesh.NbTriangles() < 1:
        return [], []
    transform = location.Transformation()
    samples: list[tuple[list[float], list[float]]] = []
    reversed_face = face.Orientation() == TopAbs_REVERSED
    for triangle_index in range(1, mesh.NbTriangles() + 1):
        indices = mesh.Triangle(triangle_index).Get()
        points = [_point(transform, mesh.Node(index)) for index in indices]
        normal = _cross(points[0], points[1], points[2])
        if reversed_face:
            normal = [-item for item in normal]
        # Triangle centroids are stable interior samples and avoid repeating
        # shared mesh vertices with conflicting per-triangle directions.
        centroid = [sum(point[axis] for point in points) / 3.0 for axis in range(3)]
        samples.append((centroid, normal))
    selected = [round(index * (len(samples) - 1) / max(count - 1, 1)) for index in range(count)]
    return [samples[index][0] for index in selected], [samples[index][1] for index in selected]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step_path", type=Path)
    parser.add_argument("input_graph", type=Path)
    parser.add_argument("output_graph", type=Path)
    parser.add_argument("--sample-count", type=int, default=48)
    parser.add_argument("--deflection", type=float, default=0.25)
    args = parser.parse_args()
    if args.sample_count < 3 or args.deflection <= 0:
        raise ValueError("invalid_sampling_configuration")

    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopExp import topexp
    from OCC.Core.TopTools import TopTools_IndexedMapOfShape
    try:
        from OCC.Core.TopoDS import Face as topods_face
    except ImportError:  # pythonocc < 7.8
        from OCC.Core.TopoDS import topods_Face as topods_face

    payload = json.loads(args.input_graph.read_text(encoding="utf-8"))
    shape = _load_step(args.step_path)
    mesher = BRepMesh_IncrementalMesh(shape, args.deflection, False, 0.5, True)
    mesher.Perform()
    face_map = TopTools_IndexedMapOfShape()
    topexp.MapShapes(shape, TopAbs_FACE, face_map)
    sampled, unavailable = 0, 0
    for node in payload.get("nodes") or []:
        if str(node.get("entity_type")) != "face":
            continue
        topology_index = int(node.get("occt_topology_index", 0))
        if not 1 <= topology_index <= face_map.Size():
            unavailable += 1
            continue
        face = topods_face(face_map.FindKey(topology_index))
        points, directions = _sample_face(face, args.sample_count)
        if not points:
            unavailable += 1
            continue
        node["patch_points"] = points
        node["patch_directions"] = directions
        node["patch_source"] = "occt_face_triangulation_centroids.v1"
        sampled += 1
    metadata = payload.setdefault("metadata", {})
    metadata["pose_patch_enrichment"] = {
        "schema_version": "occt_face_patch_enrichment.v1",
        "sample_count": args.sample_count,
        "sampled_face_count": sampled,
        "unavailable_face_count": unavailable,
        "case_specific_logic": False,
        "semantic_inputs": False,
    }
    args.output_graph.parent.mkdir(parents=True, exist_ok=True)
    args.output_graph.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"sampled_faces": sampled, "unavailable_faces": unavailable, "output": str(args.output_graph)}, ensure_ascii=False))
    return 0 if sampled else 2


if __name__ == "__main__":
    raise SystemExit(main())
