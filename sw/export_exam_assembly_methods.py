"""Export the short-term exam answer: direct assembly method JSON for cases 1-5.

This is intentionally a delivery-layer script.  It does not try to make the
pose solver perfect.  It reads the current known-group outputs and exports the
part-to-part assembly method that the short-term target needs:

- which two parts directly assemble;
- how they assemble, using one primary label plus supporting labels;
- a concise Chinese explanation;
- pose/collision diagnostics kept separate from the relation answer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CASES = [
    ("case_1", Path("sw/1"), "两个法兰同轴贴合"),
    ("case_2", Path("sw/2"), "轴、两个法兰和键的装配"),
    ("case_3", Path("sw/3"), "风扇模块插入风扇笼"),
    ("case_4", Path("sw/4_lightweight"), "主板、内存条和 CPU 的装配"),
    ("case_5", Path("sw/5_lightweight"), "机箱、挂耳和电源模块的装配"),
]


PRIMARY_PRIORITY = [
    "pocket_mate",
    "clearance",
    "coaxial",
    "planar_mate",
    "planar_align",
]


ZH_LABELS = {
    "coaxial": "同轴",
    "clearance": "轴孔间隙配合/插入",
    "planar_mate": "平面贴合",
    "planar_align": "平面对齐",
    "pocket_mate": "插槽/凹腔配合",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pair_key(parts: list[str] | tuple[str, str]) -> tuple[str, str]:
    return tuple(sorted(str(part) for part in parts))


def _primary(labels: list[str]) -> str:
    present = set(labels)
    for label in PRIMARY_PRIORITY:
        if label in present:
            return label
    return labels[0] if labels else "unknown"


def _method_sentence(parts: list[str], labels: list[str]) -> str:
    relation = " + ".join(ZH_LABELS.get(label, label) for label in labels)
    return f"{parts[0]} 与 {parts[1]} 通过 {relation} 形成直接装配关系。"


def _candidate_label_map(candidate_payload: dict[str, Any]) -> dict[tuple[str, str], list[str]]:
    result = {}
    for row in candidate_payload.get("pair_candidates", []):
        labels = list(row.get("relation_types") or [])
        if labels:
            result[_pair_key(row.get("parts", []))] = labels
    return result


def _fallback_method_labels(
    parts: list[str],
    candidate_labels: list[str],
    connection: dict[str, Any],
) -> tuple[list[str], list[str]]:
    """Generic fallback for outputs produced before method labels existed."""
    present = set(candidate_labels)
    reasons = [f"fallback_candidate_evidence={sorted(present)}"]
    if "clearance" in present:
        return ["clearance"], reasons
    if "coaxial" in present:
        labels = ["coaxial"]
        if "planar_mate" in present:
            labels.append("planar_mate")
        elif "planar_align" in present:
            labels.append("planar_align")
        return labels, reasons
    if "pocket_mate" in present:
        labels = ["pocket_mate"]
        if "planar_mate" in present:
            labels.append("planar_mate")
        elif "planar_align" in present:
            labels.append("planar_align")
        return labels, reasons
    values = list(connection.get("supporting_relation_types") or [])
    if not values and connection.get("primary_relation_type"):
        values = [connection["primary_relation_type"]]
    return values, reasons


def export_case(case_id: str, case_dir: Path, description: str) -> dict[str, Any]:
    output_dir = case_dir / "known_group_output"
    relation_path = output_dir / "assembly_relations.json"
    candidate_path = output_dir / "candidate_relations.json"
    pose_path = output_dir / "pose_validation.json"
    if not relation_path.exists():
        raise FileNotFoundError(f"missing known-group output: {relation_path}")

    relation_payload = _load_json(relation_path)
    candidate_payload = _load_json(candidate_path) if candidate_path.exists() else {}
    candidate_labels = _candidate_label_map(candidate_payload)

    edges = []
    for connection in relation_payload.get("direct_connections", []):
        parts = list(connection.get("parts", []))
        pair = _pair_key(parts)
        labels = list(connection.get("assembly_method_relation_types") or [])
        reasons = list(connection.get("assembly_method_reason") or [])
        if not labels:
            labels, reasons = _fallback_method_labels(
                parts,
                candidate_labels.get(pair) or [],
                connection,
            )
        labels = (
            labels
            or list(connection.get("supporting_relation_types") or [])
            or [connection.get("primary_relation_type")]
        )
        labels = [label for label in labels if label]
        primary = _primary(labels)
        edges.append({
            "connection_id": connection.get("connection_id"),
            "parts": parts,
            "primary_relation_type": primary,
            "relation_types": labels,
            "relation_types_zh": [ZH_LABELS.get(label, label) for label in labels],
            "assembly_method_zh": _method_sentence(parts, labels),
            "score": connection.get("score"),
            "confidence": connection.get("confidence"),
            "selection_role": connection.get("selection_role"),
            "relative_transform_a_to_b": connection.get("relative_transform_a_to_b"),
            "diagnostic_pose_closed": connection.get(
                "constraint_closed_in_selected_pose"
            ),
            "diagnostic_original_primary_relation_type": connection.get(
                "primary_relation_type"
            ),
            "diagnostic_original_supporting_relation_types": connection.get(
                "supporting_relation_types"
            ),
            "method_inference_reason": reasons,
        })

    pose_payload = _load_json(pose_path) if pose_path.exists() else {}
    collision = relation_payload.get("collision_validation", {})
    return {
        "case_id": case_id,
        "source_case_dir": str(case_dir),
        "description_zh": description,
        "input_assumption": "all_parts_belong_to_one_assembly",
        "parts": relation_payload.get("parts", []),
        "direct_assembly_edges": edges,
        "unresolved_parts": relation_payload.get("unresolved_parts", []),
        "diagnostics": {
            "assembly_connected": relation_payload.get("assembly_connected"),
            "pose_status": relation_payload.get("pose_status"),
            "pose_is_delivery_blocker": False,
            "collision_status": collision.get("status"),
            "collision_count": len(collision.get("collisions", [])),
            "checked_pose_count": collision.get("checked_pose_count"),
            "selected_pose_rank": collision.get("selected_pose_rank"),
            "search_status": pose_payload.get("search_status"),
            "complete_pose_candidate_count": pose_payload.get(
                "complete_pose_candidate_count"
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="sw/exam_assembly_methods.json",
        help="Output JSON path.",
    )
    args = parser.parse_args()

    cases = [
        export_case(case_id, case_dir, description)
        for case_id, case_dir, description in CASES
    ]
    payload = {
        "schema_version": "short_term_exam_methods.v1",
        "task": "case12345_direct_assembly_method_export",
        "scope": (
            "Correct direct assembly edges and relation labels. Pose/collision "
            "diagnostics are reported but do not block this short-term target."
        ),
        "human_labels_used_for_inference": False,
        "case_specific_label_overrides_used": False,
        "cases": cases,
        "summary": {
            "case_count": len(cases),
            "direct_edge_count": sum(
                len(case["direct_assembly_edges"]) for case in cases
            ),
            "pose_valid_count": sum(
                1 for case in cases
                if case["diagnostics"].get("pose_status") == "valid"
            ),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "ok",
        "output": str(output),
        "case_count": payload["summary"]["case_count"],
        "direct_edge_count": payload["summary"]["direct_edge_count"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
