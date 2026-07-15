"""Generate and exactly validate bounded two-part pose hypotheses with OCCT."""
from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

warnings.simplefilter("ignore")


def dump(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_shape(path: Path):
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.STEPControl import STEPControl_Reader

    reader = STEPControl_Reader()
    if reader.ReadFile(str(path)) != IFSelect_RetDone:
        raise RuntimeError("STEP_read_failed")
    if reader.TransferRoots() <= 0:
        raise RuntimeError("STEP_transfer_failed")
    return reader.OneShape()


def _unit(values: list[float]) -> list[float]:
    length = math.sqrt(sum(value * value for value in values))
    if length <= 1e-12:
        raise ValueError("zero_direction")
    return [value / length for value in values]


def _matrix(transform) -> list[list[float]]:
    return [
        [float(transform.Value(row, column)) for column in range(1, 5)]
        for row in range(1, 4)
    ] + [[0.0, 0.0, 0.0, 1.0]]


def _align_transform(
    origin_b: list[float],
    direction_b: list[float],
    origin_a: list[float],
    target_direction: list[float],
):
    from OCC.Core.gp import gp_Ax1, gp_Dir, gp_Pnt, gp_Trsf, gp_Vec

    source = gp_Dir(*_unit(direction_b))
    target = gp_Dir(*_unit(target_direction))
    rotation = gp_Trsf()
    dot = max(-1.0, min(1.0, source.Dot(target)))
    if abs(dot - 1.0) > 1e-12:
        if abs(dot + 1.0) <= 1e-12:
            helper = gp_Dir(1, 0, 0) if abs(source.X()) < 0.9 else gp_Dir(0, 1, 0)
            axis = source.Crossed(helper)
            rotation.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), axis), math.pi)
        else:
            axis = source.Crossed(target)
            rotation.SetRotation(
                gp_Ax1(gp_Pnt(0, 0, 0), axis), math.acos(dot)
            )
    moved = gp_Pnt(*origin_b).Transformed(rotation)
    translation = gp_Trsf()
    translation.SetTranslation(
        gp_Vec(
            float(origin_a[0] - moved.X()),
            float(origin_a[1] - moved.Y()),
            float(origin_a[2] - moved.Z()),
        )
    )
    result = gp_Trsf()
    result.Multiply(translation)
    result.Multiply(rotation)
    return result


def pose_hypotheses(candidate: dict) -> list[dict]:
    from OCC.Core.gp import gp_Ax1, gp_Dir, gp_Pnt, gp_Trsf

    entity_a = candidate["part_a_entity"]
    entity_b = candidate["part_b_entity"]
    direction_a = entity_a.get("direction")
    direction_b = entity_b.get("direction")
    if not direction_a or not direction_b:
        return []
    origin_a = entity_a.get("axis_origin") or entity_a["centroid"]
    origin_b = entity_b.get("axis_origin") or entity_b["centroid"]
    family = candidate["joint_family_candidate"]
    targets = (
        [[-value for value in direction_a]]
        if family == "planar"
        else [direction_a, [-value for value in direction_a]]
    )
    rotations = [0, 90, 180, 270] if family == "planar" else [0, 180]
    results = []
    seen = set()
    for target_index, target in enumerate(targets):
        base = _align_transform(origin_b, direction_b, origin_a, target)
        for angle in rotations:
            around = gp_Trsf()
            if angle:
                around.SetRotation(
                    gp_Ax1(gp_Pnt(*origin_a), gp_Dir(*_unit(target))),
                    math.radians(angle),
                )
            transform = gp_Trsf()
            transform.Multiply(around)
            transform.Multiply(base)
            matrix = _matrix(transform)
            key = tuple(round(value, 9) for row in matrix for value in row)
            if key in seen:
                continue
            seen.add(key)
            results.append(
                {
                    "target_axis_variant": target_index,
                    "axial_rotation_degrees": angle,
                    "transform_b_to_a": matrix,
                    "_transform": transform,
                }
            )
    return results


def validate(part_a: Path, part_b: Path, prediction: Path, rank: int, output: Path) -> int:
    try:
        from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
        from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
        from OCC.Core.BRepGProp import brepgprop
        from OCC.Core.GProp import GProp_GProps

        predictions = json.loads(prediction.read_text(encoding="utf-8"))
        candidates = predictions.get("candidates", [])
        if rank < 1 or rank > len(candidates):
            raise IndexError("candidate_rank_out_of_range")
        candidate = candidates[rank - 1]
        shape_a = load_shape(part_a)
        shape_b = load_shape(part_b)

        def volume(shape) -> float:
            props = GProp_GProps()
            brepgprop.VolumeProperties(shape, props)
            return abs(float(props.Mass()))

        volume_a = volume(shape_a)
        volume_b = volume(shape_b)
        checks = []
        for pose_rank, hypothesis in enumerate(pose_hypotheses(candidate), 1):
            row = {
                "pose_rank": pose_rank,
                "target_axis_variant": hypothesis["target_axis_variant"],
                "axial_rotation_degrees": hypothesis["axial_rotation_degrees"],
                "transform_b_to_a": hypothesis["transform_b_to_a"],
                "status": "unprocessed",
                "failure_reasons": [],
            }
            try:
                transformed_b = BRepBuilderAPI_Transform(
                    shape_b, hypothesis["_transform"], True
                ).Shape()
                distance_solver = BRepExtrema_DistShapeShape(shape_a, transformed_b)
                distance_solver.Perform()
                if not distance_solver.IsDone():
                    raise RuntimeError("distance_solver_not_done")
                clearance = float(distance_solver.Value())
                common_builder = BRepAlgoAPI_Common(shape_a, transformed_b)
                common_builder.Build()
                if not common_builder.IsDone():
                    raise RuntimeError("boolean_common_not_done")
                common_volume = volume(common_builder.Shape())
                denominator = max(min(volume_a, volume_b), 1e-12)
                common_ratio = common_volume / denominator
                collision_free = common_ratio <= 1e-5
                contact = clearance <= 0.05
                row.update(
                    {
                        "clearance_mm": clearance,
                        "occt_common_volume": common_volume,
                        "minimum_part_volume_ratio": common_ratio,
                        "collision_result": (
                            "collision_free" if collision_free else "penetration"
                        ),
                        "contact_result": "contact" if contact else "separated",
                        "status": (
                            "valid"
                            if collision_free and contact
                            else "failed"
                        ),
                    }
                )
                if not collision_free:
                    row["failure_reasons"].append(
                        f"penetration_ratio_{common_ratio:.6g}"
                    )
                if not contact:
                    row["failure_reasons"].append(
                        f"clearance_{clearance:.6g}_mm"
                    )
            except Exception as exc:
                row["status"] = "uncertain"
                row["failure_reasons"].append(f"{type(exc).__name__}:{exc}")
            checks.append(row)

        valid = [row for row in checks if row["status"] == "valid"]
        result = {
            "schema_version": "1.0.0",
            "part_a": str(part_a.resolve()),
            "part_b": str(part_b.resolve()),
            "candidate_rank": rank,
            "candidate_id": candidate["candidate_id"],
            "candidate_score": candidate["score"],
            "selected_entities": {
                "part_a": candidate["part_a_entity"]["entity_id"],
                "part_b": candidate["part_b_entity"]["entity_id"],
            },
            "checked_pose_count": len(checks),
            "pose_checks": checks,
            "best_valid_pose_rank": valid[0]["pose_rank"] if valid else None,
            "final_pose_status": (
                "valid"
                if valid
                else (
                    "uncertain"
                    if any(row["status"] == "uncertain" for row in checks)
                    else "failed"
                )
            ),
            "failure_reasons": (
                []
                if valid
                else sorted(
                    {
                        reason
                        for row in checks
                        for reason in row["failure_reasons"]
                    }
                )
            ),
            "unavailable_fields": ["contact_area"],
        }
        dump(output, result)
        return 0 if result["final_pose_status"] == "valid" else 2
    except Exception as exc:
        dump(
            output,
            {
                "schema_version": "1.0.0",
                "part_a": str(part_a.resolve()),
                "part_b": str(part_b.resolve()),
                "candidate_rank": rank,
                "final_pose_status": "uncertain",
                "failure_reasons": [f"{type(exc).__name__}:{exc}"],
                "unavailable_fields": ["pose_checks"],
            },
        )
        return 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part-a", required=True)
    parser.add_argument("--part-b", required=True)
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    return validate(
        Path(args.part_a),
        Path(args.part_b),
        Path(args.prediction),
        args.rank,
        Path(args.output),
    )


if __name__ == "__main__":
    raise SystemExit(main())
