"""Read-only extraction of cylindrical holes attached to planar flange faces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from interface_geometry_proxy import _bbox_from_shape, _normalise, _occ


def _plane_descriptor(face: Any, occ: dict[str, Any]) -> dict[str, Any] | None:
    adaptor = occ["BRepAdaptor_Surface"](face, True)
    if adaptor.GetType() != occ["GeomAbs_Plane"]:
        return None
    plane = adaptor.Plane()
    normal = _normalise((plane.Axis().Direction().X(), plane.Axis().Direction().Y(), plane.Axis().Direction().Z()))
    lo, hi = _bbox_from_shape(face, occ)
    centre = (lo + hi) / 2.0
    support = np.array([plane.Location().X(), plane.Location().Y(), plane.Location().Z()])
    centre = centre - normal * float(np.dot(centre - support, normal))
    return {"normal": normal, "centre": centre, "bbox_min": lo, "bbox_max": hi}


def extract(path: Path) -> dict[str, Any]:
    occ = _occ()
    reader = occ["STEPControl_Reader"]()
    if reader.ReadFile(str(path)) != occ["IFSelect_RetDone"]:
        raise RuntimeError(f"cannot read {path}")
    reader.TransferRoots()
    shape = reader.OneShape()
    # pythonocc's static helper fills the supplied map; build it explicitly for
    # versions where the return value is None.
    from OCC.Core.TopExp import topexp_MapShapesAndAncestors
    from OCC.Core.TopTools import TopTools_IndexedDataMapOfShapeListOfShape, TopTools_ListIteratorOfListOfShape
    from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.GeomAbs import GeomAbs_Cylinder

    edge_to_faces = TopTools_IndexedDataMapOfShapeListOfShape()
    topexp_MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edge_to_faces)
    holes = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    face_index = 0
    while explorer.More():
        face_index += 1
        face = explorer.Current()
        explorer.Next()
        try:
            adaptor = BRepAdaptor_Surface(face, True)
            if adaptor.GetType() != GeomAbs_Cylinder:
                continue
            cylinder = adaptor.Cylinder()
            radius = float(cylinder.Radius())
            if not 0.8 <= radius <= 8.0:
                continue
            axis = _normalise((cylinder.Axis().Direction().X(), cylinder.Axis().Direction().Y(), cylinder.Axis().Direction().Z()))
            lo, hi = _bbox_from_shape(face, occ)
            midpoint = (lo + hi) / 2.0
            support = np.array([cylinder.Location().X(), cylinder.Location().Y(), cylinder.Location().Z()])
            centre = support + axis * float(np.dot(midpoint - support, axis))
            hosts = []
            edge_explorer = TopExp_Explorer(face, TopAbs_EDGE)
            while edge_explorer.More():
                edge = edge_explorer.Current()
                edge_explorer.Next()
                if not edge_to_faces.Contains(edge):
                    continue
                iterator = TopTools_ListIteratorOfListOfShape(edge_to_faces.FindFromKey(edge))
                while iterator.More():
                    other = iterator.Value()
                    iterator.Next()
                    descriptor = _plane_descriptor(other, occ)
                    if descriptor is not None and abs(float(np.dot(descriptor["normal"], axis))) >= 0.98:
                        hosts.append(descriptor)
            if not hosts:
                continue
            host = max(hosts, key=lambda item: float(np.prod(item["bbox_max"] - item["bbox_min"])))
            holes.append(
                {
                    "face_index": face_index,
                    "radius": radius,
                    "centre": centre.round(6).tolist(),
                    "axis": axis.round(8).tolist(),
                    "host_normal": host["normal"].round(8).tolist(),
                    "host_centre": host["centre"].round(6).tolist(),
                    "host_bbox_min": host["bbox_min"].round(6).tolist(),
                    "host_bbox_max": host["bbox_max"].round(6).tolist(),
                }
            )
        except Exception:
            continue
    return {"source": str(path.resolve()), "hole_count": len(holes), "holes": holes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = extract(args.step)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"hole_count": result["hole_count"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
