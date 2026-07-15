"""Resource-bounded JoinABLe-style analytic interface baseline.

This is intentionally labelled a rule baseline, not a neural JoinABLe model.
It ranks traceable OCCT face/edge pairs without constructing the full
cross-product in memory.  The output contract is compatible with a later
learned scorer.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import heapq
import json
import math
import subprocess
import sys
import warnings
from pathlib import Path

warnings.simplefilter("ignore")


def dump(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _xyz(value) -> list[float]:
    return [float(value.X()), float(value.Y()), float(value.Z())]


def _bbox(shape) -> tuple[list[float], list[float]]:
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib

    box = Bnd_Box()
    brepbndlib.Add(shape, box)
    x1, y1, z1, x2, y2, z2 = box.Get()
    return [x1, y1, z1], [x2, y2, z2]


def _bucket(value: float) -> int:
    return int(math.floor(math.log10(max(abs(value), 1e-12)) * 4.0))


def _keep(
    reservoirs: dict[tuple[str, int], list[tuple[float, int, dict]]],
    descriptor: dict,
    limit: int,
) -> None:
    key = (descriptor["geometry_type"], _bucket(descriptor["characteristic_size"]))
    heap = reservoirs.setdefault(key, [])
    rank = float(descriptor["salience"])
    tie = int(descriptor["topology_index"])
    item = (rank, tie, descriptor)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif item[:2] > heap[0][:2]:
        heapq.heapreplace(heap, item)


def extract_descriptors(source: Path, output: Path, per_bucket: int) -> int:
    try:
        from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
        from OCC.Core.BRepGProp import brepgprop
        from OCC.Core.GProp import GProp_GProps
        from OCC.Core.GeomAbs import (
            GeomAbs_BSplineCurve,
            GeomAbs_BSplineSurface,
            GeomAbs_BezierCurve,
            GeomAbs_BezierSurface,
            GeomAbs_Circle,
            GeomAbs_Cone,
            GeomAbs_Cylinder,
            GeomAbs_Ellipse,
            GeomAbs_Line,
            GeomAbs_OffsetCurve,
            GeomAbs_OffsetSurface,
            GeomAbs_OtherCurve,
            GeomAbs_OtherSurface,
            GeomAbs_Plane,
            GeomAbs_Sphere,
            GeomAbs_SurfaceOfExtrusion,
            GeomAbs_SurfaceOfRevolution,
            GeomAbs_Torus,
        )
        from OCC.Core.IFSelect import IFSelect_RetDone
        from OCC.Core.STEPControl import STEPControl_Reader
        from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE
        from OCC.Core.TopExp import topexp
        from OCC.Core.TopTools import TopTools_IndexedMapOfShape
        from OCC.Core.TopoDS import topods

        surface_names = {
            int(GeomAbs_Plane): "plane",
            int(GeomAbs_Cylinder): "cylinder",
            int(GeomAbs_Cone): "cone",
            int(GeomAbs_Sphere): "sphere",
            int(GeomAbs_Torus): "torus",
            int(GeomAbs_BezierSurface): "bezier_surface",
            int(GeomAbs_BSplineSurface): "bspline_surface",
            int(GeomAbs_SurfaceOfRevolution): "surface_of_revolution",
            int(GeomAbs_SurfaceOfExtrusion): "surface_of_extrusion",
            int(GeomAbs_OffsetSurface): "offset_surface",
            int(GeomAbs_OtherSurface): "other_surface",
        }
        curve_names = {
            int(GeomAbs_Line): "line",
            int(GeomAbs_Circle): "circle",
            int(GeomAbs_Ellipse): "ellipse",
            int(GeomAbs_BezierCurve): "bezier_curve",
            int(GeomAbs_BSplineCurve): "bspline_curve",
            int(GeomAbs_OffsetCurve): "offset_curve",
            int(GeomAbs_OtherCurve): "other_curve",
        }
        reader = STEPControl_Reader()
        if reader.ReadFile(str(source)) != IFSelect_RetDone:
            raise RuntimeError("STEP_read_failed")
        if reader.TransferRoots() <= 0:
            raise RuntimeError("STEP_transfer_failed")
        shape = reader.OneShape()
        face_map = TopTools_IndexedMapOfShape()
        edge_map = TopTools_IndexedMapOfShape()
        topexp.MapShapes(shape, TopAbs_FACE, face_map)
        topexp.MapShapes(shape, TopAbs_EDGE, edge_map)
        reservoirs: dict[tuple[str, int], list[tuple[float, int, dict]]] = {}
        failures = []

        for index in range(1, face_map.Size() + 1):
            try:
                face = topods.Face(face_map.FindKey(index))
                adaptor = BRepAdaptor_Surface(face, True)
                props = GProp_GProps()
                brepgprop.SurfaceProperties(face, props)
                area = float(props.Mass())
                centroid = _xyz(props.CentreOfMass())
                minimum, maximum = _bbox(face)
                dims = [max(0.0, maximum[i] - minimum[i]) for i in range(3)]
                geometry_type = surface_names.get(int(adaptor.GetType()), "unknown_surface")
                direction = None
                origin = centroid
                radius = None
                if geometry_type == "plane":
                    plane = adaptor.Plane()
                    direction = _xyz(plane.Axis().Direction())
                elif geometry_type == "cylinder":
                    cylinder = adaptor.Cylinder()
                    direction = _xyz(cylinder.Axis().Direction())
                    radius = float(cylinder.Radius())
                    axis_location = _xyz(cylinder.Axis().Location())
                    delta = [centroid[i] - axis_location[i] for i in range(3)]
                    axial = sum(delta[i] * direction[i] for i in range(3))
                    origin = [axis_location[i] + axial * direction[i] for i in range(3)]
                elif geometry_type == "cone":
                    cone = adaptor.Cone()
                    direction = _xyz(cone.Axis().Direction())
                    radius = float(abs(cone.RefRadius()))
                    origin = _xyz(cone.Apex())
                elif geometry_type == "sphere":
                    sphere = adaptor.Sphere()
                    origin = _xyz(sphere.Location())
                    radius = float(sphere.Radius())
                elif geometry_type == "torus":
                    torus = adaptor.Torus()
                    direction = _xyz(torus.Axis().Direction())
                    origin = _xyz(torus.Location())
                    radius = float(torus.MajorRadius())
                size = radius if radius and radius > 0 else math.sqrt(max(area, 1e-12))
                descriptor = {
                    "entity_id": f"face_{index:06d}",
                    "entity_type": "face",
                    "topology_index": index,
                    "geometry_type": geometry_type,
                    "measure": area,
                    "characteristic_size": size,
                    "radius": radius,
                    "centroid": centroid,
                    "axis_origin": origin,
                    "direction": direction,
                    "bbox_min": minimum,
                    "bbox_max": maximum,
                    "bbox_dimensions": dims,
                    "salience": math.log1p(max(area, 0.0)),
                }
                _keep(reservoirs, descriptor, per_bucket)
            except Exception as exc:
                failures.append(f"face_{index}:{type(exc).__name__}:{exc}")

        for index in range(1, edge_map.Size() + 1):
            try:
                edge = topods.Edge(edge_map.FindKey(index))
                adaptor = BRepAdaptor_Curve(edge)
                props = GProp_GProps()
                brepgprop.LinearProperties(edge, props)
                length = float(props.Mass())
                centroid = _xyz(props.CentreOfMass())
                minimum, maximum = _bbox(edge)
                dims = [max(0.0, maximum[i] - minimum[i]) for i in range(3)]
                geometry_type = curve_names.get(int(adaptor.GetType()), "unknown_curve")
                direction = None
                origin = centroid
                radius = None
                if geometry_type == "line":
                    line = adaptor.Line()
                    direction = _xyz(line.Direction())
                elif geometry_type == "circle":
                    circle = adaptor.Circle()
                    direction = _xyz(circle.Axis().Direction())
                    origin = _xyz(circle.Location())
                    radius = float(circle.Radius())
                elif geometry_type == "ellipse":
                    ellipse = adaptor.Ellipse()
                    direction = _xyz(ellipse.Axis().Direction())
                    origin = _xyz(ellipse.Location())
                    radius = float(ellipse.MajorRadius())
                size = radius if radius and radius > 0 else max(length, 1e-12)
                descriptor = {
                    "entity_id": f"edge_{index:06d}",
                    "entity_type": "edge",
                    "topology_index": index,
                    "geometry_type": geometry_type,
                    "measure": length,
                    "characteristic_size": size,
                    "radius": radius,
                    "centroid": centroid,
                    "axis_origin": origin,
                    "direction": direction,
                    "bbox_min": minimum,
                    "bbox_max": maximum,
                    "bbox_dimensions": dims,
                    "salience": math.log1p(max(length, 0.0)),
                }
                _keep(reservoirs, descriptor, per_bucket)
            except Exception as exc:
                failures.append(f"edge_{index}:{type(exc).__name__}:{exc}")

        descriptors = [
            item[2]
            for heap in reservoirs.values()
            for item in sorted(heap, reverse=True)
        ]
        descriptors.sort(key=lambda row: (row["entity_type"], row["topology_index"]))
        result = {
            "schema_version": "1.0.0",
            "extractor": "occt_resource_bounded_analytic_entity_reservoir",
            "source_step_path": str(source.resolve()),
            "source_geometry_sha256": sha256(source),
            "topology_counts": {"faces": face_map.Size(), "edges": edge_map.Size()},
            "retained_entity_count": len(descriptors),
            "per_geometry_scale_bucket_limit": per_bucket,
            "entities": descriptors,
            "failure_reasons": failures,
            "unavailable_fields": (
                []
                if len(descriptors) == face_map.Size() + edge_map.Size()
                else ["non_retained_entity_descriptors"]
            ),
        }
        dump(output, result)
        return 0 if descriptors else 2
    except Exception as exc:
        dump(
            output,
            {
                "schema_version": "1.0.0",
                "source_step_path": str(source.resolve()),
                "failure_reasons": [f"{type(exc).__name__}:{exc}"],
                "unavailable_fields": ["entity_descriptors"],
            },
        )
        return 2


def _compatible(a: dict, b: dict) -> float:
    ga, gb = a["geometry_type"], b["geometry_type"]
    if ga == gb and ga in {
        "plane",
        "cylinder",
        "cone",
        "sphere",
        "torus",
        "line",
        "circle",
        "ellipse",
    }:
        return 1.0
    if {ga, gb} == {"cylinder", "circle"}:
        return 0.92
    if {ga, gb} == {"plane", "line"}:
        return 0.70
    return 0.0


def _ratio_score(x: float | None, y: float | None) -> float:
    if not x or not y or x <= 0.0 or y <= 0.0:
        return 0.0
    return math.exp(-abs(math.log(x / y)))


def score_pair(a: dict, b: dict) -> tuple[float, dict]:
    type_score = _compatible(a, b)
    size_score = _ratio_score(a["characteristic_size"], b["characteristic_size"])
    radius_score = _ratio_score(a.get("radius"), b.get("radius"))
    salience_score = 1.0 - math.exp(
        -min(float(a.get("salience", 0.0)), float(b.get("salience", 0.0))) / 5.0
    )
    if a.get("radius") and b.get("radius"):
        score = (
            0.49 * type_score
            + 0.17 * size_score
            + 0.29 * radius_score
            + 0.05 * salience_score
        )
    else:
        score = 0.57 * type_score + 0.38 * size_score + 0.05 * salience_score
    return score, {
        "type_compatibility": type_score,
        "characteristic_size_compatibility": size_score,
        "radius_compatibility": radius_score,
        "interface_salience": salience_score,
    }


def rank_candidates(a_path: Path, b_path: Path, output: Path, top_k: int, neighbors: int) -> int:
    try:
        a_data = json.loads(a_path.read_text(encoding="utf-8"))
        b_data = json.loads(b_path.read_text(encoding="utf-8"))
        a_entities = a_data["entities"]
        b_entities = b_data["entities"]
        by_type: dict[str, list[tuple[float, dict]]] = {}
        for entity in a_entities:
            by_type.setdefault(entity["geometry_type"], []).append(
                (math.log(max(entity["characteristic_size"], 1e-12)), entity)
            )
        for rows in by_type.values():
            rows.sort(key=lambda item: item[0])

        compatible_types = {
            "plane": ["plane", "line"],
            "line": ["line", "plane"],
            "cylinder": ["cylinder", "circle"],
            "circle": ["circle", "cylinder"],
            "cone": ["cone"],
            "sphere": ["sphere"],
            "torus": ["torus"],
            "ellipse": ["ellipse"],
        }
        heap: list[tuple[float, str, dict]] = []
        evaluated = set()
        for b in b_entities:
            target = math.log(max(b["characteristic_size"], 1e-12))
            for geometry_type in compatible_types.get(b["geometry_type"], []):
                rows = by_type.get(geometry_type, [])
                values = [item[0] for item in rows]
                position = bisect.bisect_left(values, target)
                lo = max(0, position - neighbors)
                hi = min(len(rows), position + neighbors + 1)
                for _, a in rows[lo:hi]:
                    key = (a["entity_id"], b["entity_id"])
                    if key in evaluated:
                        continue
                    evaluated.add(key)
                    score, evidence = score_pair(a, b)
                    if score <= 0:
                        continue
                    candidate = {
                        "candidate_id": f"{a['entity_id']}__{b['entity_id']}",
                        "part_a_entity": a,
                        "part_b_entity": b,
                        "joint_family_candidate": (
                            "coaxial"
                            if {"cylinder", "circle"} & {a["geometry_type"], b["geometry_type"]}
                            else (
                                "planar"
                                if "plane" in {a["geometry_type"], b["geometry_type"]}
                                else "axis_alignment"
                            )
                        ),
                        "score": score,
                        "score_evidence": evidence,
                        "model_kind": "deterministic_rule_baseline",
                    }
                    tie = candidate["candidate_id"]
                    item = (score, tie, candidate)
                    if len(heap) < top_k:
                        heapq.heappush(heap, item)
                    elif item[:2] > heap[0][:2]:
                        heapq.heapreplace(heap, item)
        candidates = [item[2] for item in sorted(heap, reverse=True)]
        for rank, candidate in enumerate(candidates, 1):
            candidate["rank"] = rank
        total_cross_product = len(a_entities) * len(b_entities)
        result = {
            "schema_version": "1.0.0",
            "part_a": a_data["source_step_path"],
            "part_b": b_data["source_step_path"],
            "predictor": "resource_bounded_analytic_rule_baseline",
            "is_pretrained_joinable": False,
            "top_k": top_k,
            "candidate_count": len(candidates),
            "retained_entity_cross_product": total_cross_product,
            "evaluated_compatible_neighbor_pairs": len(evaluated),
            "candidate_reduction_fraction": (
                1.0 - len(evaluated) / total_cross_product
                if total_cross_product
                else None
            ),
            "candidates": candidates,
            "failure_reasons": [],
            "unavailable_fields": [
                "learned_joinable_probability",
                "designer_selected_interface_truth",
            ],
        }
        dump(output, result)
        return 0 if candidates else 2
    except Exception as exc:
        dump(
            output,
            {
                "schema_version": "1.0.0",
                "failure_reasons": [f"{type(exc).__name__}:{exc}"],
                "unavailable_fields": ["ranked_interface_candidates"],
            },
        )
        return 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--source")
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-bucket", type=int, default=12)
    parser.add_argument("--part-a-descriptors")
    parser.add_argument("--part-b-descriptors")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--neighbors", type=int, default=8)
    args = parser.parse_args()
    if args.extract:
        return extract_descriptors(Path(args.source), Path(args.output), args.per_bucket)
    if args.predict:
        return rank_candidates(
            Path(args.part_a_descriptors),
            Path(args.part_b_descriptors),
            Path(args.output),
            args.top_k,
            args.neighbors,
        )
    parser.error("choose --extract or --predict")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
