"""Render a global-pose JSON directly from STEP parts using OCCT triangulation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _part(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--part requires PART_ID=STEP_PATH")
    key, raw = value.split("=", 1)
    path = Path(raw)
    if not key or not path.is_file():
        raise argparse.ArgumentTypeError("--part requires an existing STEP path")
    return key, path


def _triangles(path: Path, transform: np.ndarray) -> np.ndarray:
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopoDS import Face
    from OCC.Core.gp import gp_Trsf

    reader = STEPControl_Reader()
    if reader.ReadFile(str(path)) != IFSelect_RetDone:
        raise RuntimeError(f"cannot_read_step:{path}")
    reader.TransferRoots()
    shape = reader.OneShape()
    trsf = gp_Trsf()
    trsf.SetValues(*[float(value) for value in transform[:3, :4].reshape(-1)])
    shape = BRepBuilderAPI_Transform(shape, trsf, True).Shape()
    BRepMesh_IncrementalMesh(shape, 0.25, False, 0.5, True)
    out = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = Face(explorer.Current())
        location = face.Location()
        mesh = BRep_Tool.Triangulation(face, location)
        if mesh is not None:
            location_transform = location.Transformation()
            nodes = []
            for index in range(1, mesh.NbNodes() + 1):
                point = mesh.Node(index).Transformed(location_transform)
                nodes.append((point.X(), point.Y(), point.Z()))
            nodes = np.asarray(nodes, dtype=float)
            for index in range(1, mesh.NbTriangles() + 1):
                a, b, c = mesh.Triangle(index).Get()
                out.append(nodes[[a - 1, b - 1, c - 1]])
        explorer.Next()
    return np.asarray(out, dtype=float)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--part", action="append", type=_part, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hypothesis-index", type=int, default=0)
    parser.add_argument("--title", default="Learned CAD Pose")
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    data = json.loads(args.input.read_text(encoding="utf-8"))
    rows = data.get("hypotheses") or []
    if not rows or not 0 <= args.hypothesis_index < len(rows):
        raise ValueError("hypothesis_index_out_of_range")
    poses = rows[args.hypothesis_index]["part_poses"]
    parts = dict(args.part)
    palette = [(0.15, 0.43, 0.85), (0.16, 0.76, 0.30), (0.94, 0.48, 0.08), (0.62, 0.20, 0.72), (0.10, 0.65, 0.70)]
    meshes = [(key, _triangles(path, np.asarray(poses[key], dtype=float))) for key, path in parts.items()]
    all_points = np.concatenate([mesh.reshape(-1, 3) for _, mesh in meshes])
    center = 0.5 * (all_points.min(axis=0) + all_points.max(axis=0))
    radius = max(float((all_points.max(axis=0) - all_points.min(axis=0)).max()) * 0.58, 1.0)
    views = [(25, -45, "isometric"), (0, -90, "front XY"), (0, 0, "side YZ"), (90, 0, "top XZ")]
    figure = plt.figure(figsize=(15, 12))
    for index, (elevation, azimuth, label) in enumerate(views, 1):
        axis = figure.add_subplot(2, 2, index, projection="3d")
        for part_index, (key, mesh) in enumerate(meshes):
            step = max(1, len(mesh) // 3500)
            collection = Poly3DCollection(mesh[::step], facecolor=palette[part_index % len(palette)], edgecolor="none", alpha=0.82)
            axis.add_collection3d(collection)
        axis.set_xlim(center[0] - radius, center[0] + radius)
        axis.set_ylim(center[1] - radius, center[1] + radius)
        axis.set_zlim(center[2] - radius, center[2] + radius)
        axis.set_box_aspect((1, 1, 1))
        axis.view_init(elevation, azimuth)
        axis.set_title(label)
        axis.set_axis_off()
    exact = rows[args.hypothesis_index].get("exact_validation", {}).get("status", "not_checked")
    figure.suptitle(f"{args.title} | hypothesis={rows[args.hypothesis_index].get('hypothesis_id')} | OCCT={exact}\n" + ", ".join(f"{key}=color{index + 1}" for index, key in enumerate(parts)), fontsize=13)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(args.output, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(json.dumps({"output": str(args.output.resolve()), "exact_status": exact}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
