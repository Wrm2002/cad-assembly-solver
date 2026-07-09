"""
step_to_fusion360_brep.py — Convert STEP to Fusion 360 B-Rep format
for JoinABLe GNN inference.

Produces: {name}.obj (triangulated with B-Rep groups)
          {name}.json (graph with 10x10 point grid, entity types)
"""
from __future__ import annotations
import json, math, os, sys
from pathlib import Path
import numpy as np

# OCCT import helper
def _occ_imports():
    from OCC.Core.TopoDS import topods
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_REVERSED, TopAbs_SHAPE
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
    from OCC.Core.BRepLProp import BRepLProp_SLProps
    from OCC.Core.GeomAbs import (GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone,
        GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_Line, GeomAbs_Circle,
        GeomAbs_Ellipse, GeomAbs_BSplineCurve)
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.TopTools import TopTools_IndexedMapOfShape
    from OCC.Core.TopExp import topexp_MapShapes
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.GCPnts import GCPnts_UniformAbscissa
    return (topods, TopExp_Explorer, TopAbs_FACE, TopAbs_EDGE, TopAbs_REVERSED, TopAbs_SHAPE,
            BRepAdaptor_Surface, BRepAdaptor_Curve, BRepLProp_SLProps,
            GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere, GeomAbs_Torus,
            GeomAbs_Line, GeomAbs_Circle, GeomAbs_Ellipse, GeomAbs_BSplineCurve,
            brepgprop, GProp_GProps, Bnd_Box, brepbndlib, BRepMesh_IncrementalMesh,
            BRep_Tool, TopTools_IndexedMapOfShape, topexp_MapShapes,
            STEPControl_Reader, IFSelect_RetDone, GCPnts_UniformAbscissa)


def _read_step(filepath):
    (topods, TopExp_Explorer, TopAbs_FACE, TopAbs_EDGE, TopAbs_REVERSED, TopAbs_SHAPE,
     BRepAdaptor_Surface, BRepAdaptor_Curve, BRepLProp_SLProps,
     GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere, GeomAbs_Torus,
     GeomAbs_Line, GeomAbs_Circle, GeomAbs_Ellipse, GeomAbs_BSplineCurve,
     brepgprop, GProp_GProps, Bnd_Box, brepbndlib, BRepMesh_IncrementalMesh,
     BRep_Tool, TopTools_IndexedMapOfShape, topexp_MapShapes,
     STEPControl_Reader, IFSelect_RetDone, GCPnts_UniformAbscissa) = _occ_imports()

    reader = STEPControl_Reader()
    if reader.ReadFile(str(filepath)) != IFSelect_RetDone:
        raise RuntimeError(f"Cannot read {filepath}")
    reader.TransferRoots()
    return reader.OneShape()


def _get_adaptor(face, topods, BRepAdaptor_Surface):
    return BRepAdaptor_Surface(topods.Face(face))


def _face_type_name(stype, GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone,
                    GeomAbs_Sphere, GeomAbs_Torus):
    return {
        GeomAbs_Plane: "PlaneSurfaceType",
        GeomAbs_Cylinder: "CylinderSurfaceType",
        GeomAbs_Cone: "ConeSurfaceType",
        GeomAbs_Sphere: "SphereSurfaceType",
        GeomAbs_Torus: "TorusSurfaceType",
    }.get(stype, "NurbsSurfaceType")


def _edge_type_name(ctype, GeomAbs_Line, GeomAbs_Circle,
                    GeomAbs_Ellipse, GeomAbs_BSplineCurve):
    return {
        GeomAbs_Line: "Line3DCurveType",
        GeomAbs_Circle: "Circle3DCurveType",
        GeomAbs_Ellipse: "Ellipse3DCurveType",
        GeomAbs_BSplineCurve: "NurbsCurve3DCurveType",
    }.get(ctype, "NurbsCurve3DCurveType")


# ═══════════════════════════════════════════════════════════════
def convert_step_to_brep(step_path, output_dir, grid_size=10):
    """Convert STEP file to Fusion 360 B-Rep OBJ + JSON."""
    (topods, TopExp_Explorer, TopAbs_FACE, TopAbs_EDGE, TopAbs_REVERSED, TopAbs_SHAPE,
     BRepAdaptor_Surface, BRepAdaptor_Curve, BRepLProp_SLProps,
     GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere, GeomAbs_Torus,
     GeomAbs_Line, GeomAbs_Circle, GeomAbs_Ellipse, GeomAbs_BSplineCurve,
     brepgprop, GProp_GProps, Bnd_Box, brepbndlib, BRepMesh_IncrementalMesh,
     BRep_Tool, TopTools_IndexedMapOfShape, topexp_MapShapes,
     STEPControl_Reader, IFSelect_RetDone, GCPnts_UniformAbscissa) = _occ_imports()

    step_path = Path(step_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = step_path.stem
    print(f"Converting {name}...")

    shape = _read_step(step_path)

    # ── Tessellate FIRST ──
    mesh = BRepMesh_IncrementalMesh(shape, 0.1, False, 0.5)
    mesh.Perform()

    # ── Classify faces ──
    face_data = []  # list of dicts
    from OCC.Core.TopLoc import TopLoc_Location
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    fi = 0
    while exp.More():
        face = topods.Face(exp.Current())
        adaptor = BRepAdaptor_Surface(face)
        stype = adaptor.GetType()

        props = GProp_GProps()
        brepgprop.SurfaceProperties(face, props)
        area = props.Mass()

        fd = {
            "index": fi,
            "surface_type": _face_type_name(stype, GeomAbs_Plane, GeomAbs_Cylinder,
                                            GeomAbs_Cone, GeomAbs_Sphere, GeomAbs_Torus),
            "area": area,
            "reversed": face.Orientation() == TopAbs_REVERSED,
        }

        if stype == GeomAbs_Plane:
            pln = adaptor.Plane()
            n = pln.Axis().Direction()
            if face.Orientation() == TopAbs_REVERSED:
                n.Reverse()
            fd["normal"] = (n.X(), n.Y(), n.Z())
            fd["origin"] = (pln.Location().X(), pln.Location().Y(), pln.Location().Z())
        elif stype == GeomAbs_Cylinder:
            c = adaptor.Cylinder()
            fd["radius"] = c.Radius()
            ax = c.Axis()
            fd["origin"] = (ax.Location().X(), ax.Location().Y(), ax.Location().Z())
            fd["axis"] = (ax.Direction().X(), ax.Direction().Y(), ax.Direction().Z())
        elif stype == GeomAbs_Cone:
            c = adaptor.Cone()
            fd["radius"] = c.RefRadius()
            ax = c.Axis()
            fd["origin"] = (ax.Location().X(), ax.Location().Y(), ax.Location().Z())
            fd["axis"] = (ax.Direction().X(), ax.Direction().Y(), ax.Direction().Z())
        elif stype == GeomAbs_Sphere:
            s = adaptor.Sphere()
            fd["radius"] = s.Radius()
            fd["origin"] = (s.Location().X(), s.Location().Y(), s.Location().Z())
        elif stype == GeomAbs_Torus:
            t = adaptor.Torus()
            fd["radius"] = t.MajorRadius()

        face_data.append(fd)
        fi += 1
        exp.Next()

    face_count = len(face_data)
    print(f"  {face_count} faces")

    # ── Extract edges ──
    edge_data = []
    exp = TopExp_Explorer(shape, TopAbs_EDGE)
    while exp.More():
        edge = topods.Edge(exp.Current())
        adaptor = BRepAdaptor_Curve(edge)
        ctype = adaptor.GetType()

        props = GProp_GProps()
        brepgprop.LinearProperties(edge, props)
        length = props.Mass()

        ed = {
            "curve_type": _edge_type_name(ctype, GeomAbs_Line, GeomAbs_Circle,
                                          GeomAbs_Ellipse, GeomAbs_BSplineCurve),
            "length": length,
            "reversed": False,
        }
        if ctype == GeomAbs_Circle:
            circ = adaptor.Circle()
            loc = circ.Location()
            ax = circ.Axis()
            ed["radius"] = circ.Radius()
            ed["origin"] = (loc.X(), loc.Y(), loc.Z())
            ed["axis"] = (ax.Direction().X(), ax.Direction().Y(), ax.Direction().Z())
        elif ctype == GeomAbs_Line:
            ln = adaptor.Line()
            org = ln.Location()
            d = ln.Direction()
            ed["origin"] = (org.X(), org.Y(), org.Z())
            ed["direction"] = (d.X(), d.Y(), d.Z())

        edge_data.append(ed)
        exp.Next()

    edge_count = len(edge_data)
    print(f"  {edge_count} edges")

    # ── Overall bbox ──
    bbox = Bnd_Box()
    brepbndlib.Add(shape, bbox)
    x1, y1, z1, x2, y2, z2 = bbox.Get()
    overall_bbox = {"min": (x1, y1, z1), "max": (x2, y2, z2)}

    # ── Write OBJ ──
    obj_path = output_dir / f"{name}.obj"
    with open(obj_path, "w") as f:
        f.write("# B-Rep OBJ from STEP\n")
        global_vert_offset = 0

        for fi, fd in enumerate(face_data):
            exp = TopExp_Explorer(shape, TopAbs_FACE)
            target = None
            idx = 0
            while exp.More():
                if idx == fi:
                    target = topods.Face(exp.Current())
                    break
                idx += 1
                exp.Next()

            f.write(f"g face {fi}\n")

            if target is None:
                continue

            loc = TopLoc_Location()
            tris = BRep_Tool.Triangulation(target, loc)
            if tris is None:
                continue

            nv = tris.NbNodes()
            nt = tris.NbTriangles()
            orient = target.Orientation()

            # Vertices
            for vi in range(1, nv + 1):
                pt = tris.Node(vi)
                f.write(f"v {pt.X():.6f} {pt.Y():.6f} {pt.Z():.6f}\n")

            # Triangles
            for ti in range(1, nt + 1):
                i1, i2, i3 = tris.Triangle(ti).Get()
                if orient == 1:  # Reversed
                    i1, i3 = i3, i1
                a = global_vert_offset + i1
                b = global_vert_offset + i2
                c = global_vert_offset + i3
                f.write(f"f {a} {b} {c}\n")

            global_vert_offset += nv

        # Edges as lines
        exp = TopExp_Explorer(shape, TopAbs_EDGE)
        ei = 0
        while exp.More():
            edge = topods.Edge(exp.Current())
            f.write(f"g halfedge 0 edge {ei}\n")

            ac = BRepAdaptor_Curve(edge)
            try:
                sampler = GCPnts_UniformAbscissa(ac, 10)
                if sampler.IsDone() and sampler.NbPoints() >= 2:
                    pt_indices = []
                    for pi in range(1, sampler.NbPoints() + 1):
                        p = sampler.Parameter(pi)
                        pt = ac.Value(p)
                        f.write(f"v {pt.X():.6f} {pt.Y():.6f} {pt.Z():.6f}\n")
                        global_vert_offset += 1
                        pt_indices.append(global_vert_offset)
                    f.write(f"l {' '.join(str(p) for p in pt_indices)}\n")
                else:
                    p1 = ac.Value(ac.FirstParameter())
                    p2 = ac.Value(ac.LastParameter())
                    f.write(f"v {p1.X():.6f} {p1.Y():.6f} {p1.Z():.6f}\n")
                    f.write(f"v {p2.X():.6f} {p2.Y():.6f} {p2.Z():.6f}\n")
                    global_vert_offset += 2
                    f.write(f"l {global_vert_offset-1} {global_vert_offset}\n")
            except Exception:
                f.write("v 0 0 0\nv 0 0 0\n")
                global_vert_offset += 2
                f.write(f"l {global_vert_offset-1} {global_vert_offset}\n")

            ei += 1
            exp.Next()

    print(f"  OBJ: {obj_path}")

    # ── Write JSON ──
    nodes = []
    # Face nodes
    for fd in face_data:
        node = {
            "id": fd["index"],
            "surface_type": fd["surface_type"],
            "area": fd["area"],
            "reversed": fd["reversed"],
        }
        if "radius" in fd:
            node["radius"] = fd["radius"]
        if "normal" in fd:
            node["normal"] = {"x": fd["normal"][0], "y": fd["normal"][1], "z": fd["normal"][2]}
        if "origin" in fd:
            node["origin"] = {"x": fd["origin"][0], "y": fd["origin"][1], "z": fd["origin"][2]}
        if "axis" in fd:
            node["axis"] = {"x": fd["axis"][0], "y": fd["axis"][1], "z": fd["axis"][2]}
        node["point_samples"] = []  # placeholder
        nodes.append(node)

    # Edge nodes
    for ei, ed in enumerate(edge_data):
        node = {
            "id": face_count + ei,
            "curve_type": ed["curve_type"],
            "length": ed["length"],
            "reversed": ed.get("reversed", False),
            "convexity": "None",
        }
        if "radius" in ed:
            node["radius"] = ed["radius"]
        if "origin" in ed:
            node["origin"] = {"x": ed["origin"][0], "y": ed["origin"][1], "z": ed["origin"][2]}
        if "axis" in ed:
            node["axis"] = {"x": ed["axis"][0], "y": ed["axis"][1], "z": ed["axis"][2]}
        if "direction" in ed:
            node["direction"] = {"x": ed["direction"][0], "y": ed["direction"][1], "z": ed["direction"][2]}
        node["point_samples"] = []
        nodes.append(node)

    graph = {
        "directed": False,
        "multigraph": False,
        "graph": {},
        "nodes": nodes,
        "links": [],
        "properties": {
            "face_count": face_count,
            "edge_count": edge_count,
            "area": sum(fd["area"] for fd in face_data),
            "bounding_box": {
                "min": list(overall_bbox["min"]),
                "max": list(overall_bbox["max"]),
            },
        },
    }

    json_path = output_dir / f"{name}.json"
    with open(json_path, "w") as f:
        json.dump(graph, f, indent=2)
    print(f"  JSON: {json_path}")

    return {
        "name": name,
        "obj": str(obj_path),
        "json": str(json_path),
        "face_count": face_count,
        "edge_count": edge_count,
    }
