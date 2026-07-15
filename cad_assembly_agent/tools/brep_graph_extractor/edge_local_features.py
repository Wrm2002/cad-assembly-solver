"""Extract auditable OCCT dihedral/convexity features for B-Rep edges.

For ordinary parts every edge is evaluated.  Very large parts can be limited
to candidate faces/edges so that feature parity does not require loading a
hundreds-of-megabytes graph into memory.
"""
from __future__ import annotations

import argparse
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


def _vec(v) -> list[float]:
    return [float(v.X()), float(v.Y()), float(v.Z())]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _cross(a: list[float], b: list[float]) -> list[float]:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _unit(v: list[float]) -> list[float]:
    length = math.sqrt(_dot(v, v))
    if length <= 1e-15:
        raise ValueError("zero_length_vector")
    return [x / length for x in v]


def load_shape(path: Path):
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.STEPControl import STEPControl_Reader

    reader = STEPControl_Reader()
    if reader.ReadFile(str(path)) != IFSelect_RetDone:
        raise RuntimeError("STEP_read_failed")
    if reader.TransferRoots() <= 0:
        raise RuntimeError("STEP_transfer_failed")
    return reader.OneShape()


def worker(source: Path, output: Path, face_ids: set[int], edge_ids: set[int], scope: str) -> int:
    try:
        from OCC.Core.BRep import BRep_Tool
        from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
        from OCC.Core.BRepLProp import BRepLProp_SLProps
        from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_REVERSED
        from OCC.Core.TopExp import TopExp_Explorer, topexp
        from OCC.Core.TopTools import TopTools_IndexedMapOfShape
        from OCC.Core.TopoDS import topods

        shape = load_shape(source)
        face_map = TopTools_IndexedMapOfShape()
        edge_map = TopTools_IndexedMapOfShape()
        topexp.MapShapes(shape, TopAbs_FACE, face_map)
        topexp.MapShapes(shape, TopAbs_EDGE, edge_map)

        edge_to_face_uses: dict[int, list[tuple[int, int]]] = {}
        for face_index in range(1, face_map.Size() + 1):
            face = topods.Face(face_map.FindKey(face_index))
            explorer = TopExp_Explorer(face, TopAbs_EDGE)
            while explorer.More():
                edge_use = topods.Edge(explorer.Current())
                edge_index = edge_map.FindIndex(edge_use)
                if edge_index:
                    row = (face_index, int(edge_use.Orientation()))
                    if row not in edge_to_face_uses.setdefault(edge_index, []):
                        edge_to_face_uses[edge_index].append(row)
                explorer.Next()

        selected = set(edge_ids)
        if face_ids:
            for edge_index, uses in edge_to_face_uses.items():
                if any(face_index in face_ids for face_index, _ in uses):
                    selected.add(edge_index)
        if scope == "all":
            selected = set(range(1, edge_map.Size() + 1))

        def normal_at(edge, face):
            curve2d, first, last = BRep_Tool.CurveOnSurface(edge, face)
            if curve2d is None or not (math.isfinite(first) and math.isfinite(last)):
                raise ValueError("curve_on_surface_unavailable")
            uv = curve2d.Value((first + last) / 2.0)
            props = BRepLProp_SLProps(
                BRepAdaptor_Surface(face, True), uv.X(), uv.Y(), 1, 1e-7
            )
            if not props.IsNormalDefined():
                raise ValueError("face_normal_undefined")
            normal = _unit(_vec(props.Normal()))
            if face.Orientation() == TopAbs_REVERSED:
                normal = [-x for x in normal]
            return normal, [float(uv.X()), float(uv.Y())]

        features = []
        failures = []
        for edge_index in sorted(selected):
            row = {
                "edge_id": edge_index,
                "adjacent_face_ids": [],
                "dihedral_angle_degrees": None,
                "signed_dihedral_angle_degrees": None,
                "convexity": None,
                "status": "unprocessed",
                "failure_reasons": [],
            }
            try:
                edge = topods.Edge(edge_map.FindKey(edge_index))
                uses = edge_to_face_uses.get(edge_index, [])
                row["adjacent_face_ids"] = [face_index for face_index, _ in uses]
                if len(uses) == 0:
                    row["convexity"] = "degenerate"
                    row["status"] = "unavailable"
                    row["failure_reasons"].append("edge_has_no_adjacent_face")
                elif len(uses) == 1:
                    row["convexity"] = "boundary"
                    row["status"] = "success"
                elif len(uses) > 2:
                    row["convexity"] = "non_manifold"
                    row["status"] = "unavailable"
                    row["failure_reasons"].append("more_than_two_adjacent_faces")
                else:
                    adaptor = BRepAdaptor_Curve(edge)
                    first, last = adaptor.FirstParameter(), adaptor.LastParameter()
                    if not (math.isfinite(first) and math.isfinite(last)):
                        raise ValueError("finite_edge_parameter_range_unavailable")
                    parameter = (first + last) / 2.0
                    tangent = _unit(_vec(adaptor.DN(parameter, 1)))
                    if uses[0][1] == int(TopAbs_REVERSED):
                        tangent = [-x for x in tangent]
                    face_a = topods.Face(face_map.FindKey(uses[0][0]))
                    face_b = topods.Face(face_map.FindKey(uses[1][0]))
                    normal_a, uv_a = normal_at(edge, face_a)
                    normal_b, uv_b = normal_at(edge, face_b)
                    sine = max(-1.0, min(1.0, _dot(tangent, _cross(normal_a, normal_b))))
                    cosine = max(-1.0, min(1.0, _dot(normal_a, normal_b)))
                    signed = math.degrees(math.atan2(sine, cosine))
                    magnitude = abs(signed)
                    row.update(
                        {
                            "dihedral_angle_degrees": magnitude,
                            "signed_dihedral_angle_degrees": signed,
                            "convexity": (
                                "smooth"
                                if magnitude <= 1e-3
                                else ("convex" if signed > 0.0 else "concave")
                            ),
                            "edge_midpoint": _vec(adaptor.Value(parameter)),
                            "edge_tangent": tangent,
                            "face_normals": [normal_a, normal_b],
                            "face_uv": [uv_a, uv_b],
                            "status": "success",
                        }
                    )
            except Exception as exc:
                row["status"] = "failed"
                row["convexity"] = row["convexity"] or "unknown"
                row["failure_reasons"].append(f"{type(exc).__name__}:{exc}")
            if row["failure_reasons"]:
                failures.extend(
                    f"edge_{edge_index}:{reason}" for reason in row["failure_reasons"]
                )
            features.append(row)

        result = {
            "schema_version": "1.0.0",
            "source_step_path": str(source.resolve()),
            "scope": scope,
            "candidate_face_ids": sorted(face_ids),
            "candidate_edge_ids": sorted(edge_ids),
            "topology_edge_count": edge_map.Size(),
            "selected_edge_count": len(selected),
            "successful_edge_count": sum(row["status"] == "success" for row in features),
            "features": features,
            "convexity_convention": (
                "Outward face normals; tangent follows the first adjacent face's "
                "oriented edge use. Positive signed dihedral is convex."
            ),
            "failure_reasons": failures,
            "unavailable_fields": (
                []
                if len(selected) == edge_map.Size()
                else ["non_candidate_edge_features"]
            ),
        }
        dump(output, result)
        return 0 if features and all(row["status"] != "failed" for row in features) else 2
    except Exception as exc:
        dump(
            output,
            {
                "schema_version": "1.0.0",
                "source_step_path": str(source.resolve()),
                "scope": scope,
                "failure_reasons": [f"{type(exc).__name__}:{exc}"],
                "unavailable_fields": ["edge_local_features"],
            },
        )
        return 2


def candidate_ids(pair_truth: Path) -> dict[str, dict[str, set[int]]]:
    result: dict[str, dict[str, set[int]]] = {}
    if not pair_truth.exists():
        return result
    for path in pair_truth.glob("case_*/*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for suffix in ("a", "b"):
            part = data.get(f"part_{suffix}")
            candidate = data.get(f"candidate_interface_{suffix}", {})
            if not part:
                continue
            key = f"{path.parent.name}:{part}"
            bucket = result.setdefault(key, {"faces": set(), "edges": set()})
            bucket["faces"].update(int(x) for x in candidate.get("face_ids", []))
            bucket["edges"].update(int(x) for x in candidate.get("edge_ids", []))
    return result


def prediction_ids(
    prediction_root: Path, result: dict[str, dict[str, set[int]]], top_k: int = 10
) -> None:
    if not prediction_root.exists():
        return
    for path in prediction_root.glob("case_*/*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for suffix in ("a", "b"):
            source = data.get(f"part_{suffix}")
            if not source:
                continue
            name = Path(source).name
            key = f"{path.parent.name}:{name}"
            bucket = result.setdefault(key, {"faces": set(), "edges": set()})
            for candidate in data.get("candidates", [])[:top_k]:
                entity = candidate.get(f"part_{suffix}_entity", {})
                entity_type = entity.get("entity_type")
                index = entity.get("topology_index")
                if entity_type in ("face", "edge") and index is not None:
                    bucket[f"{entity_type}s"].add(int(index))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--pair-truth", required=True)
    parser.add_argument("--predictions")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--source")
    parser.add_argument("--output")
    parser.add_argument("--face-ids", default="[]")
    parser.add_argument("--edge-ids", default="[]")
    parser.add_argument("--scope", choices=["all", "candidate_local"], default="all")
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    if args.worker:
        return worker(
            Path(args.source),
            Path(args.output),
            set(json.loads(args.face_ids)),
            set(json.loads(args.edge_ids)),
            args.scope,
        )

    inventory = json.loads(Path(args.inventory).read_text(encoding="utf-8"))
    candidates = candidate_ids(Path(args.pair_truth))
    if args.predictions:
        prediction_ids(Path(args.predictions), candidates)
    out_root = Path(args.out_root)
    rows = []
    for case in inventory["cases"]:
        case_id = str(case["case_id"])
        for part in case["part_step_files"]:
            name = part["name"]
            source = Path(part["source_path"])
            key = f"case_{case_id}:{name}"
            ids = candidates.get(key, {"faces": set(), "edges": set()})
            scope = (
                "candidate_local"
                if name == "01-62DC24-MLB-PCBA.stp"
                or int(part.get("bytes", 0)) > 20_000_000
                else "all"
            )
            output = out_root / f"case_{case_id}" / f"{source.stem}.edge_local_features.json"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--inventory",
                args.inventory,
                "--pair-truth",
                args.pair_truth,
                "--out-root",
                args.out_root,
                "--report",
                args.report,
                "--worker",
                "--source",
                str(source),
                "--output",
                str(output),
                "--face-ids",
                json.dumps(sorted(ids["faces"])),
                "--edge-ids",
                json.dumps(sorted(ids["edges"])),
                "--scope",
                scope,
            ]
            try:
                completed = subprocess.run(command, timeout=args.timeout)
                status = "success" if completed.returncode == 0 else "partial_or_failed"
                reason = None if completed.returncode == 0 else f"worker_exit_{completed.returncode}"
            except subprocess.TimeoutExpired:
                status = "failed"
                reason = f"worker_timeout_{args.timeout}s"
                dump(
                    output,
                    {
                        "schema_version": "1.0.0",
                        "source_step_path": str(source.resolve()),
                        "scope": scope,
                        "failure_reasons": [reason],
                        "unavailable_fields": ["edge_local_features"],
                    },
                )
            rows.append(
                {
                    "case_id": case_id,
                    "part": name,
                    "scope": scope,
                    "candidate_face_count": len(ids["faces"]),
                    "candidate_edge_count": len(ids["edges"]),
                    "status": status,
                    "failure_reason": reason,
                    "output": str(output.resolve()),
                }
            )

    report = {
        "schema_version": "1.0.0",
        "part_count": len(rows),
        "success_count": sum(row["status"] == "success" for row in rows),
        "partial_or_failed_count": sum(row["status"] != "success" for row in rows),
        "results": rows,
        "failure_reasons": [
            row["failure_reason"] for row in rows if row["failure_reason"]
        ],
        "unavailable_fields": [],
    }
    dump(Path(args.report), report)
    print(f"edge local features: {report['success_count']}/{report['part_count']}")
    return 0 if report["success_count"] == report["part_count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
