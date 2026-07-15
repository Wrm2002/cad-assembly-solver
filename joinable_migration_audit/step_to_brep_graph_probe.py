"""Extract auditable JoinABLe-style face/edge graphs from STEP using OCCT."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _round_vector(values: Any) -> list[float] | None:
    try:
        result = [float(value) for value in values]
    except (TypeError, ValueError):
        return None
    if len(result) != 3 or not all(math.isfinite(v) for v in result):
        return None
    return [round(value, 12) for value in result]


def _signature(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:20]


def extract_graph(step_path: Path) -> dict[str, Any]:
    from OCC.Core.BRep import BRep_Tool  # type: ignore
    from OCC.Core.BRepAdaptor import (  # type: ignore
        BRepAdaptor_Curve,
        BRepAdaptor_Surface,
    )
    from OCC.Core.BRepBndLib import brepbndlib  # type: ignore
    from OCC.Core.BRepGProp import (  # type: ignore
        brepgprop_LinearProperties,
        brepgprop_SurfaceProperties,
    )
    from OCC.Core.BRepLProp import BRepLProp_SLProps  # type: ignore
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh  # type: ignore
    from OCC.Core.BRepTools import breptools_UVBounds  # type: ignore
    from OCC.Core.Bnd import Bnd_Box  # type: ignore
    from OCC.Core.GProp import GProp_GProps  # type: ignore
    from OCC.Core.GeomAbs import (  # type: ignore
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
        GeomAbs_Plane,
        GeomAbs_Sphere,
        GeomAbs_SurfaceOfExtrusion,
        GeomAbs_SurfaceOfRevolution,
        GeomAbs_Torus,
    )
    from OCC.Core.IFSelect import IFSelect_RetDone  # type: ignore
    from OCC.Core.STEPControl import STEPControl_Reader  # type: ignore
    from OCC.Core.TopAbs import (  # type: ignore
        TopAbs_EDGE,
        TopAbs_FACE,
        TopAbs_FORWARD,
        TopAbs_REVERSED,
        TopAbs_SOLID,
    )
    from OCC.Core.TopExp import (  # type: ignore
        TopExp_Explorer,
        topexp_MapShapes,
    )
    from OCC.Core.TopTools import TopTools_IndexedMapOfShape  # type: ignore
    from OCC.Core.TopLoc import TopLoc_Location  # type: ignore
    from OCC.Core.TopoDS import topods  # type: ignore

    surface_names = {
        GeomAbs_Plane: "plane",
        GeomAbs_Cylinder: "cylinder",
        GeomAbs_Cone: "cone",
        GeomAbs_Sphere: "sphere",
        GeomAbs_Torus: "torus",
        GeomAbs_BSplineSurface: "bspline",
        GeomAbs_BezierSurface: "bezier",
        GeomAbs_SurfaceOfRevolution: "surface_of_revolution",
        GeomAbs_SurfaceOfExtrusion: "surface_of_extrusion",
        GeomAbs_OffsetSurface: "offset",
    }
    curve_names = {
        GeomAbs_Line: "line",
        GeomAbs_Circle: "circle",
        GeomAbs_Ellipse: "ellipse",
        GeomAbs_BSplineCurve: "bspline",
        GeomAbs_BezierCurve: "bezier",
        GeomAbs_OffsetCurve: "offset",
    }

    reader = STEPControl_Reader()
    read_status = reader.ReadFile(str(step_path))
    if read_status != IFSelect_RetDone:
        raise RuntimeError(f"STEP_read_failed_status:{read_status}")
    transferred = reader.TransferRoots()
    if transferred <= 0:
        raise RuntimeError("STEP_no_roots_transferred")
    shape = reader.OneShape()
    # These samples are not passed to the released JoinABLe checkpoint.  They
    # are retained solely for the learned local interface-patch head, in the
    # same B-Rep coordinate system as its Fusion360 supervision.
    BRepMesh_IncrementalMesh(shape, 0.35, False, 0.5, True).Perform()
    bounding_box = Bnd_Box()
    brepbndlib.Add(shape, bounding_box)
    bbox_values = [float(value) for value in bounding_box.Get()]
    bbox_min = bbox_values[:3]
    bbox_max = bbox_values[3:]
    bbox_center = [
        (low + high) * 0.5 for low, high in zip(bbox_min, bbox_max)
    ]
    bbox_half_extents = [
        max(0.0, (high - low) * 0.5)
        for low, high in zip(bbox_min, bbox_max)
    ]
    normalization_extent = max(bbox_half_extents)
    face_map = TopTools_IndexedMapOfShape()
    edge_map = TopTools_IndexedMapOfShape()
    solid_map = TopTools_IndexedMapOfShape()
    topexp_MapShapes(shape, TopAbs_FACE, face_map)
    topexp_MapShapes(shape, TopAbs_EDGE, edge_map)

    def _sample_indices(count: int, desired: int = 32) -> list[int]:
        if count <= 0:
            return []
        return [int(round(value)) for value in [
            index * (count - 1) / max(desired - 1, 1)
            for index in range(desired)
        ]]

    def _unit_vector(values: list[float]) -> list[float]:
        length = math.sqrt(sum(value * value for value in values))
        if length <= 1e-15:
            raise ValueError("zero_length_vector")
        return [value / length for value in values]

    def _face_patch(face: Any) -> tuple[list[list[float]], list[list[float]]]:
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation(face, location)
        if triangulation is None or triangulation.NbNodes() <= 0:
            raise ValueError("face_triangulation_unavailable")
        transform = location.Transformation()
        points = [
            [float(point.X()), float(point.Y()), float(point.Z())]
            for point in (
                triangulation.Node(index).Transformed(transform)
                for index in range(1, triangulation.NbNodes() + 1)
            )
        ]
        normal = [0.0, 0.0, 0.0]
        for index in range(1, triangulation.NbTriangles() + 1):
            a, b, c = triangulation.Triangle(index).Get()
            pa, pb, pc = (points[a - 1], points[b - 1], points[c - 1])
            ab = [pb[i] - pa[i] for i in range(3)]
            ac = [pc[i] - pa[i] for i in range(3)]
            cross = [
                ab[1] * ac[2] - ab[2] * ac[1],
                ab[2] * ac[0] - ab[0] * ac[2],
                ab[0] * ac[1] - ab[1] * ac[0],
            ]
            normal = [normal[i] + cross[i] for i in range(3)]
        try:
            normal = _unit_vector(normal)
        except ValueError:
            normal = [0.0, 0.0, 0.0]
        if face.Orientation() == TopAbs_REVERSED:
            normal = [-value for value in normal]
        selected = _sample_indices(len(points))
        return ([points[index] for index in selected], [normal for _ in selected])

    def _edge_patch(edge: Any) -> tuple[list[list[float]], list[list[float]]]:
        adaptor = BRepAdaptor_Curve(edge)
        first, last = float(adaptor.FirstParameter()), float(adaptor.LastParameter())
        if not (math.isfinite(first) and math.isfinite(last)) or abs(last - first) <= 1e-12:
            raise ValueError("edge_parameter_range_unavailable")
        points, tangents = [], []
        for step in range(32):
            parameter = first + (last - first) * step / 31.0
            points.append(_round_vector(adaptor.Value(parameter).Coord()) or [0.0, 0.0, 0.0])
            tangent = _round_vector(adaptor.DN(parameter, 1).Coord()) or [0.0, 0.0, 0.0]
            try:
                tangents.append(_unit_vector(tangent))
            except ValueError:
                tangents.append([0.0, 0.0, 0.0])
        if edge.Orientation() == TopAbs_REVERSED:
            tangents = [[-value for value in tangent] for tangent in tangents]
        return points, tangents
    topexp_MapShapes(shape, TopAbs_SOLID, solid_map)

    nodes = []
    unavailable: set[str] = set()
    approximated_checkpoint_entity_mappings: list[str] = []

    surface_checkpoint_map = {
        GeomAbs_Plane: ("PlaneSurfaceType", 0, "exact"),
        GeomAbs_Cylinder: ("CylinderSurfaceType", 1, "exact"),
        GeomAbs_Cone: ("ConeSurfaceType", 2, "exact"),
        GeomAbs_Sphere: ("SphereSurfaceType", 3, "exact"),
        GeomAbs_Torus: ("TorusSurfaceType", 4, "exact"),
        GeomAbs_BSplineSurface: ("NurbsSurfaceType", 7, "exact"),
        GeomAbs_BezierSurface: ("NurbsSurfaceType", 7, "approximated"),
        GeomAbs_SurfaceOfRevolution: (
            "NurbsSurfaceType", 7, "approximated"
        ),
        GeomAbs_SurfaceOfExtrusion: (
            "NurbsSurfaceType", 7, "approximated"
        ),
        GeomAbs_OffsetSurface: ("NurbsSurfaceType", 7, "approximated"),
    }

    def checkpoint_curve_type(
        adaptor: Any, edge_length: float
    ) -> tuple[str, int, str]:
        curve_type = adaptor.GetType()
        if edge_length <= 1e-10:
            return "Degenerate3DCurveType", 15, "exact"
        if curve_type == GeomAbs_Line:
            return "Line3DCurveType", 8, "exact"
        if curve_type == GeomAbs_Circle:
            span = abs(float(adaptor.LastParameter()) - float(
                adaptor.FirstParameter()
            ))
            if abs(span - 2.0 * math.pi) <= 1e-5:
                return "Circle3DCurveType", 10, "exact"
            return "Arc3DCurveType", 9, "exact"
        if curve_type == GeomAbs_Ellipse:
            span = abs(float(adaptor.LastParameter()) - float(
                adaptor.FirstParameter()
            ))
            if abs(span - 2.0 * math.pi) <= 1e-5:
                return "Ellipse3DCurveType", 11, "exact"
            return "EllipticalArc3DCurveType", 12, "exact"
        if curve_type in (
            GeomAbs_BSplineCurve,
            GeomAbs_BezierCurve,
            GeomAbs_OffsetCurve,
        ):
            quality = (
                "exact"
                if curve_type == GeomAbs_BSplineCurve
                else "approximated"
            )
            return "NurbsCurve3DCurveType", 14, quality
        return "NurbsCurve3DCurveType", 14, "approximated"

    def orientation_name(value: Any) -> str:
        if value == TopAbs_FORWARD:
            return "forward"
        if value == TopAbs_REVERSED:
            return "reversed"
        return "unknown"

    for index in range(1, face_map.Size() + 1):
        face = topods.Face(face_map.FindKey(index))
        adaptor = BRepAdaptor_Surface(face, True)
        props = GProp_GProps()
        brepgprop_SurfaceProperties(face, props)
        centroid = _round_vector(props.CentreOfMass().Coord())
        area = float(props.Mass())
        surface_type = surface_names.get(adaptor.GetType(), "unknown")
        checkpoint_surface = surface_checkpoint_map.get(
            adaptor.GetType(),
            ("NurbsSurfaceType", 7, "approximated"),
        )
        if checkpoint_surface[2] != "exact":
            approximated_checkpoint_entity_mappings.append(
                f"face_{index:06d}:{surface_type}"
            )
        normal = None
        try:
            u_min, u_max, v_min, v_max = breptools_UVBounds(face)
            uv_values = (u_min, u_max, v_min, v_max)
            if all(math.isfinite(float(value)) for value in uv_values):
                sl_props = BRepLProp_SLProps(
                    adaptor,
                    (u_min + u_max) * 0.5,
                    (v_min + v_max) * 0.5,
                    1,
                    1e-7,
                )
                if sl_props.IsNormalDefined():
                    normal = _round_vector(sl_props.Normal().Coord())
                    if (
                        normal is not None
                        and face.Orientation() == TopAbs_REVERSED
                    ):
                        normal = [-value for value in normal]
        except Exception:
            normal = None
        if normal is None:
            unavailable.add(f"face_{index:06d}.normal")

        # ── Extract analytic surface parameters (cylinder, cone, sphere, torus) ──
        radius = None
        axis_origin = None
        axis_direction = None
        try:
            st = adaptor.GetType()
            if st == GeomAbs_Cylinder:
                cyl = adaptor.Cylinder()
                radius = round(float(cyl.Radius()), 9)
                ax = cyl.Axis()
                loc = ax.Location()
                di = ax.Direction()
                axis_origin = _round_vector(loc.Coord())
                axis_direction = _round_vector(di.Coord())
            elif st == GeomAbs_Cone:
                cone = adaptor.Cone()
                radius = round(float(cone.RefRadius()), 9)
                ax = cone.Axis()
                loc = ax.Location()
                di = ax.Direction()
                axis_origin = _round_vector(loc.Coord())
                axis_direction = _round_vector(di.Coord())
            elif st == GeomAbs_Sphere:
                sph = adaptor.Sphere()
                radius = round(float(sph.Radius()), 9)
                loc = sph.Location()
                axis_origin = _round_vector(loc.Coord())
            elif st == GeomAbs_Torus:
                tor = adaptor.Torus()
                radius = round(float(tor.MajorRadius()), 9)
                ax = tor.Axis()
                loc = ax.Location()
                di = ax.Direction()
                axis_origin = _round_vector(loc.Coord())
                axis_direction = _round_vector(di.Coord())
        except Exception:
            pass

        signature_payload = {
            "entity_type": "face",
            "surface_type": surface_type,
            "area": round(area, 9),
            "centroid": centroid,
        }
        try:
            patch_points, patch_directions = _face_patch(face)
            patch_status = "success"
        except Exception as exc:
            patch_points, patch_directions = [], []
            patch_status = f"unavailable:{type(exc).__name__}"
        nodes.append({
            "node_id": f"face_{index:06d}",
            "entity_type": "face",
            "occt_topology_index": index,
            "joinable_node_index": index - 1,
            "joinable_entity_type": checkpoint_surface[0],
            "joinable_entity_type_index": checkpoint_surface[1],
            "joinable_entity_type_mapping_quality": checkpoint_surface[2],
            "is_face": 1,
            "length": 0.0,
            "face_reversed": int(
                face.Orientation() == TopAbs_REVERSED
            ),
            "edge_reversed": 0,
            "surface_type": surface_type,
            "area": area,
            "normal": normal,
            "centroid": centroid,
            "radius": radius,
            "axis_origin": axis_origin,
            "axis_direction": axis_direction,
            "orientation": orientation_name(face.Orientation()),
            "geometry_signature": _signature(signature_payload),
            "reverse_lookup_available_in_worker": True,
            "failure_reasons": [],
            "unavailable_fields": (
                [] if normal is not None else ["normal"]
            ),
            "patch_points": patch_points,
            "patch_directions": patch_directions,
            "patch_status": patch_status,
        })

    for index in range(1, edge_map.Size() + 1):
        edge = topods.Edge(edge_map.FindKey(index))
        adaptor = BRepAdaptor_Curve(edge)
        props = GProp_GProps()
        brepgprop_LinearProperties(edge, props)
        centroid = _round_vector(props.CentreOfMass().Coord())
        length = float(props.Mass())
        curve_type = curve_names.get(adaptor.GetType(), "unknown")
        checkpoint_curve = checkpoint_curve_type(adaptor, length)
        if checkpoint_curve[2] != "exact":
            approximated_checkpoint_entity_mappings.append(
                f"edge_{index:06d}:{curve_type}"
            )
        radius = None
        axis_origin = None
        axis_direction = None
        try:
            if adaptor.GetType() == GeomAbs_Circle:
                circle = adaptor.Circle()
                radius = float(circle.Radius())
                axis = circle.Axis()
                axis_origin = _round_vector(axis.Location().Coord())
                axis_direction = _round_vector(axis.Direction().Coord())
            elif adaptor.GetType() == GeomAbs_Ellipse:
                ellipse = adaptor.Ellipse()
                radius = float(ellipse.MajorRadius())
                axis = ellipse.Axis()
                axis_origin = _round_vector(axis.Location().Coord())
                axis_direction = _round_vector(axis.Direction().Coord())
            elif adaptor.GetType() == GeomAbs_Line:
                line = adaptor.Line()
                axis_origin = _round_vector(line.Location().Coord())
                axis_direction = _round_vector(line.Direction().Coord())
        except Exception:
            radius = None
        signature_payload = {
            "entity_type": "edge",
            "curve_type": curve_type,
            "length": round(length, 9),
            "centroid": centroid,
            "radius": round(radius, 9) if radius is not None else None,
        }
        try:
            patch_points, patch_directions = _edge_patch(edge)
            patch_status = "success"
        except Exception as exc:
            patch_points, patch_directions = [], []
            patch_status = f"unavailable:{type(exc).__name__}"
        nodes.append({
            "node_id": f"edge_{index:06d}",
            "entity_type": "edge",
            "occt_topology_index": index,
            "joinable_node_index": face_map.Size() + index - 1,
            "joinable_entity_type": checkpoint_curve[0],
            "joinable_entity_type_index": checkpoint_curve[1],
            "joinable_entity_type_mapping_quality": checkpoint_curve[2],
            "is_face": 0,
            "face_reversed": 0,
            "edge_reversed": int(
                edge.Orientation() == TopAbs_REVERSED
            ),
            "curve_type": curve_type,
            "length": length,
            "radius": radius,
            "centroid": centroid,
            "axis_origin": axis_origin,
            "axis_direction": axis_direction,
            "orientation": orientation_name(edge.Orientation()),
            "geometry_signature": _signature(signature_payload),
            "reverse_lookup_available_in_worker": True,
            "failure_reasons": [],
            "unavailable_fields": [],
            "patch_points": patch_points,
            "patch_directions": patch_directions,
            "patch_status": patch_status,
        })

    adjacency: set[tuple[int, int]] = set()
    face_edge_occurrence_count: dict[tuple[int, int], int] = {}
    # Keep oriented face uses separately from the released JoinABLe graph.
    # The checkpoint continues to see only face--edge adjacency; these records
    # are audit evidence for downstream pose/orientation logic, not a silent
    # change to the pretrained model's input topology.
    edge_to_face_uses: dict[int, list[tuple[int, Any]]] = {}
    for face_index in range(1, face_map.Size() + 1):
        face = face_map.FindKey(face_index)
        explorer = TopExp_Explorer(face, TopAbs_EDGE)
        while explorer.More():
            edge_use = topods.Edge(explorer.Current())
            edge_index = edge_map.FindIndex(edge_use)
            if edge_index > 0:
                adjacency.add((face_index, edge_index))
                key = (face_index, edge_index)
                face_edge_occurrence_count[key] = (
                    face_edge_occurrence_count.get(key, 0) + 1
                )
                use = (face_index, edge_use.Orientation())
                uses = edge_to_face_uses.setdefault(edge_index, [])
                if use not in uses:
                    uses.append(use)
            explorer.Next()

    def _unit(values: list[float]) -> list[float]:
        length = math.sqrt(sum(value * value for value in values))
        if length <= 1e-15:
            raise ValueError("zero_length_vector")
        return [value / length for value in values]

    def _dot(left: list[float], right: list[float]) -> float:
        return sum(a * b for a, b in zip(left, right))

    def _cross(left: list[float], right: list[float]) -> list[float]:
        return [
            left[1] * right[2] - left[2] * right[1],
            left[2] * right[0] - left[0] * right[2],
            left[0] * right[1] - left[1] * right[0],
        ]

    def _normal_at_edge_midpoint(edge: Any, face: Any) -> list[float]:
        curve_2d, first, last = BRep_Tool.CurveOnSurface(edge, face)
        if curve_2d is None or not (
            math.isfinite(float(first)) and math.isfinite(float(last))
        ):
            raise ValueError("curve_on_surface_unavailable")
        uv = curve_2d.Value((float(first) + float(last)) * 0.5)
        props = BRepLProp_SLProps(
            BRepAdaptor_Surface(face, True), uv.X(), uv.Y(), 1, 1e-7
        )
        if not props.IsNormalDefined():
            raise ValueError("face_normal_undefined")
        normal = _unit([float(value) for value in props.Normal().Coord()])
        if face.Orientation() == TopAbs_REVERSED:
            normal = [-value for value in normal]
        return normal

    edge_nodes = {
        int(node["occt_topology_index"]): node
        for node in nodes
        if node["entity_type"] == "edge"
    }
    topology_feature_failures: list[str] = []
    topology_feature_successes = 0
    for edge_index, node in edge_nodes.items():
        uses = edge_to_face_uses.get(edge_index, [])
        node["adjacent_face_ids"] = [
            f"face_{face_index:06d}" for face_index, _ in uses
        ]
        node["adjacent_face_topology_indices"] = [
            face_index for face_index, _ in uses
        ]
        node["topology_feature_status"] = "unavailable"
        node["convexity"] = "unknown"
        node["convexity_confidence"] = "none"
        node["dihedral_angle_degrees"] = None
        node["signed_dihedral_angle_degrees"] = None
        node["topology_failure_reasons"] = []
        if len(uses) == 1:
            node.update({
                "topology_feature_status": "success",
                "convexity": "boundary",
                "convexity_confidence": "topology_only",
            })
            topology_feature_successes += 1
            continue
        if len(uses) != 2:
            reason = (
                "edge_has_no_adjacent_face" if not uses
                else "more_than_two_adjacent_faces"
            )
            node["topology_failure_reasons"].append(reason)
            topology_feature_failures.append(f"edge_{edge_index:06d}:{reason}")
            continue
        try:
            edge = topods.Edge(edge_map.FindKey(edge_index))
            adaptor = BRepAdaptor_Curve(edge)
            first, last = adaptor.FirstParameter(), adaptor.LastParameter()
            if not (math.isfinite(float(first)) and math.isfinite(float(last))):
                raise ValueError("finite_edge_parameter_range_unavailable")
            parameter = (float(first) + float(last)) * 0.5
            tangent = _unit([
                float(value) for value in adaptor.DN(parameter, 1).Coord()
            ])
            if uses[0][1] == TopAbs_REVERSED:
                tangent = [-value for value in tangent]
            face_a = topods.Face(face_map.FindKey(uses[0][0]))
            face_b = topods.Face(face_map.FindKey(uses[1][0]))
            normal_a = _normal_at_edge_midpoint(edge, face_a)
            normal_b = _normal_at_edge_midpoint(edge, face_b)
            sine = max(-1.0, min(1.0, _dot(tangent, _cross(normal_a, normal_b))))
            cosine = max(-1.0, min(1.0, _dot(normal_a, normal_b)))
            signed = math.degrees(math.atan2(sine, cosine))
            magnitude = abs(signed)
            node.update({
                "topology_feature_status": "success",
                "dihedral_angle_degrees": magnitude,
                "signed_dihedral_angle_degrees": signed,
                "convexity": (
                    "smooth" if magnitude <= 1e-3
                    else ("convex" if signed > 0.0 else "concave")
                ),
                "convexity_confidence": "oriented_local_face_normals",
                "edge_midpoint": _round_vector(adaptor.Value(parameter).Coord()),
                "edge_tangent": _round_vector(tangent),
                "adjacent_face_normals": [normal_a, normal_b],
            })
            topology_feature_successes += 1
        except Exception as exc:
            reason = f"{type(exc).__name__}:{exc}"
            node["topology_failure_reasons"].append(reason)
            topology_feature_failures.append(f"edge_{edge_index:06d}:{reason}")
    seam_edge_indices = {
        edge_index
        for (_, edge_index), count in face_edge_occurrence_count.items()
        if count > 1
    }
    checkpoint_node_index = 0
    for node in nodes:
        is_seam = (
            node["entity_type"] == "edge"
            and int(node["occt_topology_index"]) in seam_edge_indices
        )
        node["is_seam_edge"] = is_seam
        node["checkpoint_include"] = not is_seam
        node["checkpoint_node_index"] = (
            checkpoint_node_index if not is_seam else None
        )
        if not is_seam:
            checkpoint_node_index += 1
    links = [{
        "src": f"face_{face_index:06d}",
        "dst": f"edge_{edge_index:06d}",
        "relation": "face_edge_adjacency",
        "failure_reasons": [],
        "unavailable_fields": [],
    } for face_index, edge_index in sorted(adjacency)]
    source_hash = sha256_file(step_path)
    joinable_feature_gaps = [
        "face UV point/normal/trimming-mask grids",
        "edge point/tangent grids",
        "designer-selected joint labels",
        "contact labels",
        "hole labels",
        "assembly transforms and hierarchy",
    ]
    unavailable_fields = sorted(unavailable.union(joinable_feature_gaps))
    if topology_feature_failures:
        unavailable_fields.append("partial_edge_topology_features")
    graph = {
        "schema_version": "1.0.0",
        "part_id": step_path.stem,
        "source_step_path": str(step_path.resolve()),
        "source_geometry_sha256": source_hash,
        "nodes": nodes,
        "edges": links,
        "metadata": {
            "adapter_version": "2.2.0",
            "unit": "STEP_declared_or_importer_interpreted",
            "bounding_box": {
                "min": bbox_min,
                "max": bbox_max,
                "center": bbox_center,
                "half_extents": bbox_half_extents,
            },
            "checkpoint_pair_normalization_extent": normalization_extent,
            "num_faces": face_map.Size(),
            "num_edges": edge_map.Size(),
            "num_solids": solid_map.Size(),
            "num_face_edge_adjacencies": len(links),
            "edge_topology_features": {
                "available": topology_feature_successes > 0,
                "successful_edge_count": topology_feature_successes,
                "failed_edge_count": len(topology_feature_failures),
                "convexity_convention": (
                    "Outward face normals; tangent follows the first adjacent "
                    "face's oriented edge use. Positive signed dihedral is convex."
                ),
                "failure_reasons": topology_feature_failures,
            },
            "extraction_status": "success",
            "failure_reason": None,
            "id_scheme": (
                "1-based OCCT IndexedMap order per entity type"
            ),
            "id_stability_scope": (
                "same input hash, OCCT build, import settings and process"
            ),
            "stable_across_step_rewrite_or_healing": False,
            "reverse_lookup_scope": (
                "exact in worker via IndexedMap.FindKey(index); "
                "after serialization requires deterministic re-import"
            ),
            "joinable_feature_gaps": joinable_feature_gaps,
            "released_checkpoint_input_features": [
                "entity_types",
                "length",
                "face_reversed",
                "edge_reversed",
            ],
            "released_checkpoint_minimal_features_available": True,
            "released_checkpoint_topology_canonicalization": {
                "excluded_seam_edge_count": len(seam_edge_indices),
                "excluded_seam_edge_indices": sorted(seam_edge_indices),
                "checkpoint_node_count": checkpoint_node_index,
                "original_occt_node_count": len(nodes),
                "raw_occt_nodes_preserved": True,
            },
            "released_checkpoint_approximated_entity_mappings": (
                approximated_checkpoint_entity_mappings
            ),
            "released_checkpoint_requires_uv_or_curve_grids": False,
            "failure_reasons": [],
            "unavailable_fields": unavailable_fields,
        },
        "failure_reasons": [],
        "unavailable_fields": unavailable_fields,
    }
    return graph


def worker(input_path: Path, output_path: Path) -> int:
    try:
        graph = extract_graph(input_path)
        write_json(output_path, graph)
        return 0
    except Exception as exc:
        failure = {
            "schema_version": "1.0.0",
            "part_id": input_path.stem,
            "source_step_path": str(input_path.resolve()),
            "nodes": [],
            "edges": [],
            "metadata": {
                "extraction_status": "failed",
                "failure_reason": f"{type(exc).__name__}:{exc}",
                "failure_reasons": [
                    f"{type(exc).__name__}:{exc}"
                ],
                "unavailable_fields": ["brep_graph"],
            },
            "failure_reasons": [f"{type(exc).__name__}:{exc}"],
            "unavailable_fields": ["brep_graph"],
        }
        write_json(output_path, failure)
        return 2


def controller(
    inputs: list[Path],
    out_dir: Path,
    report_path: Path,
    limit: int,
    timeout_seconds: int,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    script = Path(__file__).resolve()
    for source in inputs[:limit]:
        output = out_dir / f"{source.stem}.brep_graph.json"
        command = [
            sys.executable,
            str(script),
            "--worker",
            "--worker-input",
            str(source),
            "--worker-output",
            str(output),
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
            if output.is_file():
                with output.open("r", encoding="utf-8") as stream:
                    graph = json.load(stream)
                status = graph["metadata"]["extraction_status"]
                failures = graph["failure_reasons"]
                unavailable = graph["unavailable_fields"]
                counts = {
                    "num_faces": graph["metadata"].get("num_faces"),
                    "num_edges": graph["metadata"].get("num_edges"),
                    "num_adjacencies": graph["metadata"].get(
                        "num_face_edge_adjacencies"
                    ),
                }
            else:
                status = "worker_failed_without_output"
                failures = [
                    f"worker_exit_code:{completed.returncode}",
                    completed.stderr[-1000:],
                ]
                unavailable = ["brep_graph"]
                counts = {}
        except subprocess.TimeoutExpired:
            status = "worker_timeout"
            failures = [f"worker_timeout_after_{timeout_seconds}_seconds"]
            unavailable = ["brep_graph"]
            counts = {}
        results.append({
            "source_step_path": str(source.resolve()),
            "output_path": str(output.resolve()),
            "status": status,
            **counts,
            "failure_reasons": failures,
            "unavailable_fields": unavailable,
        })
    success_count = sum(row["status"] == "success" for row in results)
    report = {
        "schema_version": "1.0.0",
        "occt_available": success_count > 0,
        "requested_count": min(limit, len(inputs)),
        "attempted_count": len(results),
        "success_count": success_count,
        "failure_count": len(results) - success_count,
        "results": results,
        "entity_id_audit": {
            "traceable_ids_present": success_count > 0,
            "stable_for_unchanged_file_same_environment": True,
            "stable_across_step_reexport_or_topology_healing": False,
            "reverse_lookup_to_occt_shape": (
                "Yes inside extraction worker through IndexedMap.FindKey; "
                "serialized ids require deterministic re-import."
            ),
        },
        "joinable_required_fields_not_extracted": [
            "face UV point/normal/trimming-mask grids",
            "edge point/tangent grids",
            "edge convexity and dihedral angle",
            "designer-selected joint entity labels",
            "joint equivalence labels",
            "contact and hole labels",
            "assembly transform and hierarchy",
        ],
        "acceptance_met": success_count >= min(3, limit),
        "failure_reasons": [
            reason for row in results
            for reason in row["failure_reasons"]
        ],
        "unavailable_fields": sorted({
            field for row in results
            for field in row["unavailable_fields"]
        }),
    }
    write_json(report_path, report)
    print(f"STEP graphs: {success_count}/{len(results)} successful")
    return 0 if report["acceptance_met"] else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*")
    parser.add_argument(
        "--out-dir", default="step_brep_graph_samples"
    )
    parser.add_argument(
        "--report", default="step_brep_graph_probe_report.json"
    )
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--worker-input")
    parser.add_argument("--worker-output")
    args = parser.parse_args()
    if args.worker:
        if not args.worker_input or not args.worker_output:
            parser.error("--worker requires input and output")
        return worker(
            Path(args.worker_input), Path(args.worker_output)
        )
    inputs = [Path(value) for value in args.inputs]
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        write_json(Path(args.report), {
            "schema_version": "1.0.0",
            "success_count": 0,
            "failure_count": len(missing),
            "failure_reasons": [
                f"input_not_found:{path}" for path in missing
            ],
            "unavailable_fields": ["brep_graph"],
        })
        return 2
    return controller(
        inputs,
        Path(args.out_dir),
        Path(args.report),
        max(1, args.limit),
        max(1, args.timeout_seconds),
    )


if __name__ == "__main__":
    raise SystemExit(main())
