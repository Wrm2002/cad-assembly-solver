"""Extract B-Rep free-edge contours from the case-5 chassis.

An opening in a sheet-metal/compound chassis is represented topologically by a
free boundary edge (one adjacent face), unlike a screw hole whose edges have
two or more adjacent faces.  This module only recalls *possible* openings;
their traversability is separately tested with the EAR profile and insertion
path.  No hole centre is used to generate a contour.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[tuple[int, int, int], tuple[int, int, int]] = {}

    def find(self, item):
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, a, b) -> None:
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[b] = a


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_VERTEX
    from OCC.Core.TopExp import TopExp_Explorer, topexp_MapShapesAndAncestors
    from OCC.Core.TopTools import TopTools_IndexedDataMapOfShapeListOfShape, TopTools_ListIteratorOfListOfShape

    reader = STEPControl_Reader()
    if reader.ReadFile(str(args.step)) != IFSelect_RetDone:
        raise RuntimeError(f"cannot read {args.step}")
    reader.TransferRoots()
    shape = reader.OneShape()

    ancestors = TopTools_IndexedDataMapOfShapeListOfShape()
    topexp_MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, ancestors)
    uf = UnionFind()
    free_edges: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = []
    explorer = TopExp_Explorer(shape, TopAbs_EDGE)
    while explorer.More():
        edge = explorer.Current()
        explorer.Next()
        faces = TopTools_ListIteratorOfListOfShape(ancestors.FindFromKey(edge))
        count = 0
        while faces.More():
            count += 1
            faces.Next()
        if count != 1:
            continue
        vertices = []
        ev = TopExp_Explorer(edge, TopAbs_VERTEX)
        while ev.More():
            point = BRep_Tool.Pnt(ev.Current())
            vertices.append(tuple(int(round(v * 10)) for v in (point.X(), point.Y(), point.Z())))
            ev.Next()
        if len(vertices) < 2 or vertices[0] == vertices[-1]:
            continue
        a, b = vertices[0], vertices[-1]
        uf.union(a, b)
        free_edges.append((a, b))

    components: dict[tuple[int, int, int], set[tuple[int, int, int]]] = defaultdict(set)
    edge_count: dict[tuple[int, int, int], int] = defaultdict(int)
    for a, b in free_edges:
        root = uf.find(a)
        components[root].update((a, b))
        edge_count[root] += 1

    contours = []
    raw_components = []
    for root, vertices in components.items():
        pts = np.asarray(list(vertices), dtype=float) / 10.0
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        ext = hi - lo
        raw_components.append({
            "free_edge_count": edge_count[root],
            "vertex_count": len(vertices),
            "bbox_extent_mm": ext.round(3).tolist(),
        })
        # A usable opening contour has several boundary edges and non-trivial
        # extent in at least two directions; short seam fragments are retained
        # in the raw count but not promoted to a candidate.
        if edge_count[root] < 4 or np.count_nonzero(ext >= 5.0) < 2:
            continue
        contours.append({
            "contour_id": f"FREE_LOOP_{len(contours):03d}",
            "free_edge_count": edge_count[root],
            "vertex_count": len(vertices),
            "bbox_min_mm": lo.round(3).tolist(),
            "bbox_max_mm": hi.round(3).tolist(),
            "bbox_extent_mm": ext.round(3).tolist(),
            "centre_mm": ((lo + hi) / 2.0).round(3).tolist(),
            "topology_evidence": "all member edges have exactly one adjacent B-Rep face",
        })

    contours.sort(key=lambda row: (-np.prod(np.sort(np.asarray(row["bbox_extent_mm"]))[-2:]), -row["free_edge_count"]))
    result = {
        "method": "B-Rep_free_edge_connected_contours",
        "source": str(args.step.resolve()),
        "raw_free_edge_count": len(free_edges),
        "raw_connected_component_count": len(raw_components),
        "largest_raw_components": sorted(
            raw_components,
            key=lambda row: (-row["free_edge_count"], -np.prod(np.sort(np.asarray(row["bbox_extent_mm"]))[-2:])),
        )[:20],
        "candidate_contour_count": len(contours),
        "candidate_contours": contours,
        "required_next_gate": "EAR cross-section sweep through each contour; free boundary alone does not prove traversability",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"raw_free_edges": len(free_edges), "candidate_contours": len(contours), "top": contours[:10]}, indent=2))


if __name__ == "__main__":
    main()
