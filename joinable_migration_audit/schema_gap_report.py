"""Compare measured Fusion/JoinABLe fields with measured STEP/OCCT graphs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def build_gap(
    fusion_probe: dict[str, Any],
    step_probe: dict[str, Any],
) -> dict[str, Any]:
    fusion_success = fusion_probe.get("parsed_sample_count", 0) > 0
    step_success = step_probe.get("success_count", 0) > 0
    fields = [
        {
            "field": "face nodes",
            "fusion_joinable": fusion_success,
            "current_step_occt": step_success,
            "missing_impact": "none for coarse topology",
            "repair": "retain OCCT IndexedMap face id",
        },
        {
            "field": "edge nodes",
            "fusion_joinable": fusion_success,
            "current_step_occt": step_success,
            "missing_impact": "none for coarse topology",
            "repair": "retain OCCT IndexedMap edge id",
        },
        {
            "field": "stable persistent face/edge id",
            "fusion_joinable": (
                "source index stable inside released record"
            ),
            "current_step_occt": (
                "conditional: same file/importer only"
            ),
            "missing_impact": (
                "predictions cannot safely survive STEP rewrite/healing"
            ),
            "repair": (
                "hash input, store indexed id and geometry signature; "
                "revalidate after every import"
            ),
        },
        {
            "field": "surface/curve type",
            "fusion_joinable": fusion_success,
            "current_step_occt": step_success,
            "missing_impact": "minor enum/domain mapping required",
            "repair": "map OCCT GeomAbs types to JoinABLe vocabulary",
        },
        {
            "field": "face-edge adjacency",
            "fusion_joinable": fusion_success,
            "current_step_occt": step_success,
            "missing_impact": "none",
            "repair": "preserve heterogeneous incidence links",
        },
        {
            "field": "face 10x10 point/normal/trim grid",
            "fusion_joinable": fusion_success,
            "current_step_occt": False,
            "missing_impact": (
                "pretrained JoinABLe feature contract cannot be reproduced"
            ),
            "repair": "sample OCCT parametric faces with trimming mask",
        },
        {
            "field": "edge point/tangent grid",
            "fusion_joinable": fusion_success,
            "current_step_occt": False,
            "missing_impact": "pretrained edge encoder input missing",
            "repair": "sample OCCT curves at normalized parameters",
        },
        {
            "field": "edge convexity/dihedral angle",
            "fusion_joinable": fusion_success,
            "current_step_occt": False,
            "missing_impact": "local interface descriptor domain gap",
            "repair": "estimate from adjacent face normals",
        },
        {
            "field": "designer-selected joint entity label",
            "fusion_joinable": fusion_success,
            "current_step_occt": False,
            "missing_impact": "no supervised truth on user STEP",
            "repair": (
                "prediction only; obtain labels from assembly mates or "
                "manual annotation for evaluation"
            ),
        },
        {
            "field": "contact label",
            "fusion_joinable": fusion_success,
            "current_step_occt": False,
            "missing_impact": "cannot train/evaluate contact localization",
            "repair": (
                "requires assembled pose and exact proximity/contact query"
            ),
        },
        {
            "field": "hole labels",
            "fusion_joinable": fusion_success,
            "current_step_occt": False,
            "missing_impact": "JoinABLe hole augmentation unavailable",
            "repair": "rule-estimate cylindrical loops, mark as inferred",
        },
        {
            "field": "joint/mate type label",
            "fusion_joinable": fusion_success,
            "current_step_occt": False,
            "missing_impact": "cannot evaluate mate type on standalone STEP",
            "repair": "obtain from source CAD assembly or manual ground truth",
        },
        {
            "field": "assembly transform",
            "fusion_joinable": fusion_success,
            "current_step_occt": False,
            "missing_impact": "no ground-truth pose reconstruction target",
            "repair": "requires assembly-level source, not isolated parts",
        },
        {
            "field": "assembly hierarchy",
            "fusion_joinable": "joint set is a body pair, source has context",
            "current_step_occt": False,
            "missing_impact": "cannot infer mixed-pool grouping truth",
            "repair": "ingest assembly tree/BOM or create separate labels",
        },
    ]
    return {
        "schema_version": "1.0.0",
        "status": (
            "success" if fusion_success and step_success else "partial"
        ),
        "fusion_joint_label_maps_to_face_or_edge": fusion_success,
        "current_step_has_faces_and_edges": step_success,
        "current_step_has_conditionally_stable_entity_ids": step_success,
        "current_step_has_edge_curve_type": step_success,
        "current_step_has_face_surface_type": step_success,
        "current_step_has_face_edge_adjacency": step_success,
        "current_step_has_contact_label": False,
        "current_step_has_joint_label": False,
        "current_step_has_mate_type_label": False,
        "current_step_has_assembly_transform": False,
        "current_step_has_assembly_hierarchy": False,
        "field_gap_table": fields,
        "domain_gap": [
            "Fusion graph comes from native Fusion B-Rep; STEP is a neutral exchange import with possible topology splitting/healing.",
            "JoinABLe uses sampled face/edge grids, convexity and dihedral features not yet emitted by the minimal OCCT probe.",
            "Units and normalization must be reproduced exactly before pretrained inference.",
            "Entity ordering and persistent identity differ across Fusion and OCCT.",
            "Standalone STEP has no designer joint, mate, contact, pose, hierarchy or functional-intent labels.",
        ],
        "fields_that_must_be_added_for_pretrained_joinable": [
            "face UV point/normal/trimming-mask grids",
            "edge point/tangent grids",
            "edge convexity and dihedral angle",
            "JoinABLe-compatible normalization and enum mapping",
            "validated entity-index round trip",
        ],
        "fields_rule_estimable": [
            "surface/curve type",
            "area/length/radius",
            "local axis candidates",
            "face-edge adjacency",
            "approximate hole candidates",
            "edge convexity and dihedral angle",
        ],
        "fields_unobtainable_from_isolated_step_parts": [
            "designer-selected joint entities",
            "ground-truth mate type",
            "ground-truth assembly transform",
            "assembly hierarchy and BOM",
            "true contact pairs before pose is known",
            "functional compatibility and source grouping",
        ],
        "failure_reasons": (
            [] if fusion_success and step_success
            else ["one_or_more_probe_inputs_not_successful"]
        ),
        "unavailable_fields": sorted({
            item["field"] for item in fields
            if item["current_step_occt"] is False
        }),
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Fusion / JoinABLe 与 STEP / OCCT Schema Gap",
        "",
        f"状态：`{report['status']}`",
        "",
        "| 字段 | Fusion / JoinABLe 是否有 | 当前 STEP 是否有 | 缺失影响 | 修复建议 |",
        "|---|---|---|---|---|",
    ]
    for row in report["field_gap_table"]:
        lines.append(
            f"| {row['field']} | {row['fusion_joinable']} | "
            f"{row['current_step_occt']} | {row['missing_impact']} | "
            f"{row['repair']} |"
        )
    lines.extend([
        "",
        "## 直接回答",
        "",
        "- Fusion designer-selected joint 能映射到 B-Rep face/edge。",
        "- 当前 STEP graph 有 face、edge、surface/curve type 和 adjacency。",
        "- OCCT id 仅在同文件、同版本、同导入设置下条件稳定；不是 CAD persistent id。",
        "- 单独 STEP 不含 contact、joint、mate、assembly transform 或 hierarchy 真值。",
        "- 因此 JoinABLe 可替代候选接口生成层，不能直接完成 mixed-pool 分组。",
        "",
        "## 推理时 domain gap",
        "",
    ])
    lines.extend(f"- {item}" for item in report["domain_gap"])
    lines.extend([
        "",
        "## 可规则估计",
        "",
    ])
    lines.extend(
        f"- {item}" for item in report["fields_rule_estimable"]
    )
    lines.extend([
        "",
        "## 无法从孤立 STEP 获得",
        "",
    ])
    lines.extend(
        f"- {item}"
        for item in report[
            "fields_unobtainable_from_isolated_step_parts"
        ]
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fusion-probe",
        default="fusion_joint_schema_sample.json",
    )
    parser.add_argument(
        "--step-probe",
        default="step_brep_graph_probe_report.json",
    )
    parser.add_argument(
        "--json-output", default="schema_gap_report.json"
    )
    parser.add_argument(
        "--markdown-output", default="schema_gap_report.md"
    )
    args = parser.parse_args()
    fusion = load_json(Path(args.fusion_probe))
    step = load_json(Path(args.step_probe))
    report = build_gap(fusion, step)
    write_json(Path(args.json_output), report)
    Path(args.markdown_output).write_text(
        markdown(report), encoding="utf-8"
    )
    print(f"Schema gap report: {report['status']}")
    return 0 if report["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
