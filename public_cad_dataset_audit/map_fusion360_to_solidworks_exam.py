"""Build the Fusion360-to-SolidWorks external exam mapping.

This script does not train or tune the matcher.  It creates an auditable
boundary between:

1. Fusion360 assembly graphs used for development and calibration; and
2. user-provided SolidWorks/STEP cases used only as an external exam.

The output answers two practical questions:

- Can the current data be fed through one common relation-label protocol?
- Which SolidWorks exam labels are not yet covered by mapped Fusion360
  development samples?
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRAPH_DIR = (
    PROJECT_ROOT
    / "public_cad_dataset_audit"
    / "outputs"
    / "fusion360_assembly_graphs"
)
DEFAULT_SW_ROOT = PROJECT_ROOT / "sw"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "public_cad_dataset_audit"
    / "outputs"
    / "fusion360_to_solidworks_exam_mapping"
)

COMMON_LABELS = [
    "coaxial",
    "clearance",
    "planar_mate",
    "planar_align",
    "pocket_mate",
]

POCKET_KEYWORDS = {
    "slot",
    "socket",
    "groove",
    "rail",
    "rails",
    "channel",
    "bay",
    "cage",
    "housing",
    "bracket",
    "keyway",
}

POCKET_EXCLUDE_PHRASES = {
    "socket set screw",
    "hex socket",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def step_candidate(part: dict[str, Any]) -> dict[str, Any]:
    geometry = part.get("geometry") or {}
    for candidate in geometry.get("candidates") or []:
        if candidate.get("format") == "step":
            path = candidate.get("path")
            return {
                "path": path,
                "exists": bool(path and Path(path).is_file()),
                "source": "geometry.candidates.step",
            }
    path = geometry.get("path")
    return {
        "path": path if geometry.get("format") == "step" else None,
        "exists": bool(
            path and geometry.get("format") == "step" and Path(path).is_file()
        ),
        "source": "geometry.path" if geometry.get("format") == "step" else None,
    }


def relation_interface_types(edge: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for relation in edge.get("relations") or []:
        for entity in relation.get("interface_entities") or []:
            local = entity.get("local_geometry") or {}
            for key in ("surface_type", "curve_type"):
                value = local.get(key)
                if value:
                    values.append(str(value))
            entity_type = entity.get("entity_type")
            if entity_type:
                values.append(str(entity_type))
    return values


def map_fusion_edge_to_common_labels(
    edge: dict[str, Any],
    part_names: list[str] | None = None,
) -> tuple[list[str], str | None, str, list[str], list[str]]:
    """Map Fusion360 relation/contact records to the common five-label set.

    The mapping is intentionally conservative.  It maps only geometry evidence
    present in the source relation entities; it does not infer functional roles
    from file names or assembly IDs.
    """

    relation_types = {
        str(value)
        for value in edge.get("relation_types") or []
        if value is not None
    }
    relation_kinds = {
        str(value)
        for value in edge.get("relation_kinds") or []
        if value is not None
    }
    interfaces = relation_interface_types(edge)
    interface_text = " ".join(interfaces).lower()
    source_text = " ".join(sorted(relation_types | relation_kinds)).lower()

    labels: list[str] = []
    reasons: list[str] = []
    unavailable: list[str] = []
    part_name_text = " ".join(part_names or []).lower()

    has_plane = "planesurfacetype" in interface_text
    has_cylinder = "cylindersurfacetype" in interface_text
    has_cone = "conesurfacetype" in interface_text
    has_circle_or_edge = (
        "circle3dcurvetype" in interface_text
        or "brepedge" in interface_text
    )

    if any(token in source_text for token in ("revolute", "cylindrical")):
        labels.append("coaxial")
        reasons.append("Fusion joint type indicates rotation/cylindrical axis.")
    if any(token in source_text for token in ("slider", "pin")):
        labels.append("clearance")
        reasons.append("Fusion joint type indicates insertion/sliding freedom.")
    if "rigid" in source_text:
        reasons.append(
            "Rigid joint is positive supervision but does not by itself define one geometry label."
        )

    if has_cylinder or has_cone or has_circle_or_edge:
        if "coaxial" not in labels:
            labels.append("coaxial")
        reasons.append("Mapped from cylindrical/circular interface evidence.")
    if has_plane:
        labels.append("planar_mate")
        reasons.append("Mapped from planar interface/contact evidence.")

    if "contact" in source_text and not labels:
        labels.append("planar_mate")
        reasons.append(
            "Contact-only positive edge without analytic subtype; defaulted to planar_mate for weak supervision."
        )
        unavailable.append("specific_contact_interface_type")

    pocket_name_hit = (
        any(keyword in part_name_text for keyword in POCKET_KEYWORDS)
        and not any(phrase in part_name_text for phrase in POCKET_EXCLUDE_PHRASES)
    )
    multi_planar_contact = has_plane and len(
        [
            value
            for value in interfaces
            if str(value).lower() == "planesurfacetype"
        ]
    ) >= 2
    if pocket_name_hit and ("contact" in source_text or "rigid" in source_text):
        if "pocket_mate" not in labels:
            labels.append("pocket_mate")
        reasons.append(
            "Pocket candidate from Fusion part names plus positive contact/joint evidence; requires audit before use as high-confidence supervision."
        )
        if multi_planar_contact:
            reasons.append(
                "Multiple planar contact/interface records support slot/cavity-style constraint."
            )
        unavailable.append("pocket_mate_human_audit")

    labels = [label for label in COMMON_LABELS if label in set(labels)]
    if not labels:
        confidence = "unmapped"
        primary = None
        unavailable.append("common_relation_label")
    elif "pocket_mate" in labels:
        confidence = "medium"
        primary = "pocket_mate"
    elif len(labels) >= 2:
        confidence = "medium"
        primary = "coaxial" if "coaxial" in labels else labels[0]
    else:
        confidence = "medium" if labels[0] in {"coaxial", "planar_mate"} else "low"
        primary = labels[0]

    if "pocket_mate" not in labels:
        unavailable.append("slot_or_cavity_functional_role")
    return labels, primary, confidence, reasons, sorted(set(unavailable))


def graph_files(graph_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in graph_dir.glob("*.json")
        if path.name != "conversion_manifest.json"
    )


def build_fusion_relation_rows(
    graph_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    assembly_rows: list[dict[str, Any]] = []
    failures: list[str] = []
    label_counter: Counter[str] = Counter()
    source_relation_counter: Counter[str] = Counter()
    split_counter: Counter[str] = Counter()
    assembly_label_support: dict[str, set[str]] = defaultdict(set)
    step_missing_count = 0
    unmapped_positive_count = 0

    files = graph_files(graph_dir)
    split_by_assembly: dict[str, str] = {}
    for index, graph_file in enumerate(files):
        if index % 10 in {0, 1}:
            split = "test"
        elif index % 10 == 2:
            split = "dev"
        else:
            split = "train"
        graph = load_json(graph_file)
        assembly_id = graph.get("assembly_id") or graph_file.stem
        split_by_assembly[assembly_id] = split

        parts = {part["part_id"]: part for part in graph.get("parts", [])}
        part_step = {part_id: step_candidate(part) for part_id, part in parts.items()}
        step_missing_count += sum(
            1 for candidate in part_step.values() if not candidate["exists"]
        )
        pos_count = neg_count = 0

        for edge in graph.get("positive_part_pair_edges") or []:
            pair = edge.get("part_pair") or []
            if len(pair) != 2 or any(part_id not in parts for part_id in pair):
                failures.append(f"{assembly_id}:{edge.get('edge_id')}:bad_positive_pair")
                continue
            part_names = [
                " ".join(
                    str(parts[part_id].get(key) or "")
                    for key in ("part_name", "body_name")
                ).strip()
                for part_id in pair
            ]
            labels, primary, confidence, reasons, unavailable = (
                map_fusion_edge_to_common_labels(edge, part_names)
            )
            if not labels:
                unmapped_positive_count += 1
            for label in labels:
                label_counter[label] += 1
                assembly_label_support[assembly_id].add(label)
            for relation_type in edge.get("relation_types") or []:
                source_relation_counter[str(relation_type)] += 1
            row = {
                "schema_version": "1.0.0",
                "sample_id": f"fusion360:{assembly_id}:{edge.get('edge_id')}",
                "split": split,
                "source_dataset": graph.get("source_dataset"),
                "source_graph_path": str(graph_file.resolve()),
                "ground_truth_scope": "development_only",
                "do_not_use_for_solidworks_exam": True,
                "assembly_id": assembly_id,
                "part_pair": pair,
                "part_names": [parts[part_id].get("part_name") for part_id in pair],
                "part_name_evidence": part_names,
                "solidworks_compatible_geometry_paths": [
                    part_step[part_id]["path"] for part_id in pair
                ],
                "solidworks_compatible_geometry_exists": all(
                    part_step[part_id]["exists"] for part_id in pair
                ),
                "direct_connection": True,
                "source_relation_types": edge.get("relation_types") or [],
                "source_relation_kinds": edge.get("relation_kinds") or [],
                "mapped_relation_types": labels,
                "primary_relation_type": primary,
                "mapping_confidence": confidence,
                "mapping_reasons": reasons,
                "negative_definition": None,
                "failure_reasons": edge.get("failure_reasons") or [],
                "unavailable_fields": sorted(
                    set((edge.get("unavailable_fields") or []) + unavailable)
                ),
            }
            rows.append(row)
            pos_count += 1

        positive_pairs = {
            tuple(edge.get("part_pair") or [])
            for edge in graph.get("positive_part_pair_edges") or []
        }
        for edge in graph.get("negative_part_pair_edges") or []:
            pair = edge.get("part_pair") or []
            if len(pair) != 2 or any(part_id not in parts for part_id in pair):
                failures.append(f"{assembly_id}:{edge.get('edge_id')}:bad_negative_pair")
                continue
            if tuple(pair) in positive_pairs:
                failures.append(f"{assembly_id}:{edge.get('edge_id')}:negative_overlaps_positive")
                continue
            row = {
                "schema_version": "1.0.0",
                "sample_id": f"fusion360:{assembly_id}:{edge.get('edge_id')}",
                "split": split,
                "source_dataset": graph.get("source_dataset"),
                "source_graph_path": str(graph_file.resolve()),
                "ground_truth_scope": "development_only_closed_world_negative",
                "do_not_use_for_solidworks_exam": True,
                "assembly_id": assembly_id,
                "part_pair": pair,
                "part_names": [parts[part_id].get("part_name") for part_id in pair],
                "solidworks_compatible_geometry_paths": [
                    part_step[part_id]["path"] for part_id in pair
                ],
                "solidworks_compatible_geometry_exists": all(
                    part_step[part_id]["exists"] for part_id in pair
                ),
                "direct_connection": False,
                "source_relation_types": ["none_observed"],
                "source_relation_kinds": [],
                "mapped_relation_types": [],
                "primary_relation_type": None,
                "mapping_confidence": "closed_world_negative",
                "mapping_reasons": [
                    "No Fusion joint/as-built joint/contact is recorded for this same-assembly pair."
                ],
                "negative_definition": edge.get("negative_definition"),
                "failure_reasons": edge.get("failure_reasons") or [],
                "unavailable_fields": edge.get("unavailable_fields") or [],
            }
            rows.append(row)
            neg_count += 1

        assembly_rows.append(
            {
                "assembly_id": assembly_id,
                "split": split,
                "source_graph_path": str(graph_file.resolve()),
                "part_count": len(parts),
                "positive_pair_count": pos_count,
                "negative_pair_count": neg_count,
                "step_missing_count": sum(
                    1 for candidate in part_step.values() if not candidate["exists"]
                ),
            }
        )

    train_labels = {
        label
        for row in rows
        if row["split"] == "train"
        for label in (row.get("mapped_relation_types") or [])
    }
    all_labels = {
        label
        for row in rows
        for label in (row.get("mapped_relation_types") or [])
    }
    moved_for_label_coverage = []
    for label in sorted(all_labels - train_labels):
        candidates = [
            assembly_id
            for assembly_id, labels in assembly_label_support.items()
            if label in labels
        ]
        if not candidates:
            continue
        # Move the smallest positive-support assembly for this label to train.
        chosen = sorted(
            candidates,
            key=lambda assembly_id: (
                sum(1 for row in rows if row["assembly_id"] == assembly_id and row["direct_connection"]),
                assembly_id,
            ),
        )[0]
        old_split = split_by_assembly.get(chosen)
        if old_split == "train":
            train_labels.add(label)
            continue
        split_by_assembly[chosen] = "train"
        for row in rows:
            if row["assembly_id"] == chosen:
                row["split"] = "train"
        for row in assembly_rows:
            if row["assembly_id"] == chosen:
                row["split"] = "train"
        moved_for_label_coverage.append({
            "assembly_id": chosen,
            "from": old_split,
            "to": "train",
            "reason": f"ensure_train_support_for_label:{label}",
        })
        train_labels.update(assembly_label_support[chosen])

    split_counter = Counter(row["split"] for row in rows)
    summary = {
        "assembly_count": len(files),
        "sample_count": len(rows),
        "positive_count": sum(1 for row in rows if row["direct_connection"]),
        "negative_count": sum(1 for row in rows if not row["direct_connection"]),
        "split_counts": dict(sorted(split_counter.items())),
        "mapped_label_counts": dict(sorted(label_counter.items())),
        "source_relation_type_counts": dict(sorted(source_relation_counter.items())),
        "step_missing_count": step_missing_count,
        "unmapped_positive_count": unmapped_positive_count,
        "moved_assemblies_for_train_label_coverage": moved_for_label_coverage,
        "failure_reasons": failures,
        "unavailable_fields": sorted(
            {
                field
                for row in rows
                for field in row.get("unavailable_fields", [])
            }
        ),
    }
    return rows, assembly_rows, summary


def numeric_case_dirs(sw_root: Path) -> list[Path]:
    return sorted(
        [path for path in sw_root.iterdir() if path.is_dir() and path.name.isdigit()],
        key=lambda path: int(path.name),
    )


def is_external_input_part(path: Path) -> bool:
    if path.suffix.lower() not in {".step", ".stp"}:
        return False
    lowered = path.stem.lower()
    return not lowered.startswith("assembly")


def external_part_paths(sw_root: Path, case_id: str, parts: list[str]) -> list[str]:
    return [str((sw_root / case_id / part).resolve()) for part in parts]


def build_solidworks_exam_rows(sw_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    failures: list[str] = []
    confirmed_case_count = 0
    all_case_dirs = numeric_case_dirs(sw_root)
    annotated_cases = set()

    for label_file in sorted((sw_root / "phase5_annotation_pack").glob("case_*/human_labels.json")):
        data = load_json(label_file)
        case_id = str(data.get("case_id"))
        annotated_cases.add(case_id)
        status = data.get("annotation_status")
        confirmed = status == "pass_1_confirmed"
        if confirmed:
            confirmed_case_count += 1
        case_rows = []
        for relation in data.get("pass_1_direct_relations") or []:
            pair = relation.get("parts") or []
            paths = external_part_paths(sw_root, case_id, pair)
            exists = [Path(path).is_file() for path in paths]
            row = {
                "schema_version": "1.0.0",
                "sample_id": f"solidworks_exam:case_{case_id}:{relation.get('pair_id')}",
                "source_dataset": "solidworks_step_external_exam",
                "ground_truth_scope": "external_exam_answer_only",
                "must_not_be_read_by_inference": True,
                "case_id": case_id,
                "pair_id": relation.get("pair_id"),
                "part_pair": pair,
                "part_paths": paths,
                "part_files_exist": exists,
                "part_sha256": [sha256_file(Path(path)) for path in paths],
                "direct_connection": relation.get("direct_connection"),
                "mapped_relation_types": relation.get("relation_types") or [],
                "primary_relation_type": relation.get("primary_relation_type"),
                "annotation_status": status,
                "annotation_confidence": relation.get("annotation_confidence"),
                "notes": relation.get("notes"),
                "usable_for_scoring": confirmed and relation.get("direct_connection") is not None and all(exists),
                "failure_reasons": [] if all(exists) else ["part_file_missing"],
                "unavailable_fields": ["interface_entity_ids", "original_solidworks_mates"],
            }
            if not row["usable_for_scoring"]:
                failures.append(f"case_{case_id}:{relation.get('pair_id')}:not_usable_for_scoring")
            rows.append(row)
            case_rows.append(row)
        cases.append(
            {
                "case_id": case_id,
                "annotation_status": status,
                "part_count": len(data.get("parts") or []),
                "pair_count": len(case_rows),
                "usable_pair_count": sum(row["usable_for_scoring"] for row in case_rows),
                "positive_pair_count": sum(
                    row["direct_connection"] is True and row["usable_for_scoring"]
                    for row in case_rows
                ),
                "negative_pair_count": sum(
                    row["direct_connection"] is False and row["usable_for_scoring"]
                    for row in case_rows
                ),
            }
        )

    for case_dir in all_case_dirs:
        if case_dir.name in annotated_cases:
            continue
        step_parts = sorted(
            [
                path.name
                for path in case_dir.iterdir()
                if is_external_input_part(path)
            ]
        )
        pair_count = len(list(combinations(step_parts, 2)))
        cases.append(
            {
                "case_id": case_dir.name,
                "annotation_status": "missing_phase5_human_labels",
                "part_count": len(step_parts),
                "pair_count": pair_count,
                "usable_pair_count": 0,
                "positive_pair_count": None,
                "negative_pair_count": None,
            }
        )
        failures.append(f"case_{case_dir.name}:missing_phase5_human_labels")

    label_counter: Counter[str] = Counter()
    primary_counter: Counter[str] = Counter()
    combo_counter: Counter[str] = Counter()
    positive_rows = [
        row
        for row in rows
        if row["usable_for_scoring"] and row["direct_connection"] is True
    ]
    for row in positive_rows:
        labels = row.get("mapped_relation_types") or []
        for label in labels:
            label_counter[label] += 1
        if row.get("primary_relation_type"):
            primary_counter[str(row["primary_relation_type"])] += 1
        combo_counter["+".join(labels) if labels else "unlabeled_positive"] += 1

    summary = {
        "case_count": len(all_case_dirs),
        "annotated_case_count": len(annotated_cases),
        "confirmed_case_count": confirmed_case_count,
        "scorable_pair_count": sum(row["usable_for_scoring"] for row in rows),
        "scorable_positive_pair_count": len(positive_rows),
        "scorable_negative_pair_count": sum(
            row["usable_for_scoring"] and row["direct_connection"] is False
            for row in rows
        ),
        "positive_label_counts": dict(sorted(label_counter.items())),
        "positive_primary_label_counts": dict(sorted(primary_counter.items())),
        "positive_label_combo_counts": dict(sorted(combo_counter.items())),
        "cases": sorted(cases, key=lambda row: int(row["case_id"])),
        "failure_reasons": failures,
        "unavailable_fields": ["interface_entity_ids", "original_solidworks_mates"],
    }
    return rows, summary


def coverage_summary(
    fusion_summary: dict[str, Any],
    solidworks_summary: dict[str, Any],
) -> dict[str, Any]:
    fusion_labels = set(fusion_summary.get("mapped_label_counts") or {})
    sw_labels = set(solidworks_summary.get("positive_label_counts") or {})
    missing_labels = sorted(sw_labels - fusion_labels)
    covered_labels = sorted(sw_labels & fusion_labels)

    fusion_combos = set()
    # Fusion rows are not passed in here, so combo coverage is intentionally not
    # claimed.  The exam should not rely on exact combo memorisation.
    sw_combos = set(solidworks_summary.get("positive_label_combo_counts") or {})

    can_run_external_exam = (
        solidworks_summary["confirmed_case_count"] > 0
        and solidworks_summary["scorable_pair_count"] > 0
    )
    can_claim_label_coverage = not missing_labels and bool(sw_labels)
    can_claim_full_exam = (
        can_run_external_exam
        and can_claim_label_coverage
        and solidworks_summary["confirmed_case_count"] == solidworks_summary["case_count"]
    )
    return {
        "fusion_mapped_labels": sorted(fusion_labels),
        "solidworks_exam_positive_labels": sorted(sw_labels),
        "covered_exam_labels": covered_labels,
        "missing_exam_labels_in_fusion_mapping": missing_labels,
        "solidworks_exam_positive_label_combos": sorted(sw_combos),
        "fusion_combo_coverage_not_claimed": sorted(fusion_combos),
        "can_run_external_exam": can_run_external_exam,
        "can_score_current_confirmed_cases": can_run_external_exam,
        "can_claim_fusion_to_solidworks_label_coverage": can_claim_label_coverage,
        "can_claim_full_5case_exam_readiness": can_claim_full_exam,
        "readiness_level": (
            "full_5case_ready"
            if can_claim_full_exam
            else (
                "partial_confirmed_cases_ready"
                if can_run_external_exam
                else "not_ready"
            )
        ),
        "blocking_reasons": [
            *(
                [f"Fusion mapping lacks exam labels: {', '.join(missing_labels)}"]
                if missing_labels
                else []
            ),
            *(
                ["Some SolidWorks cases have no confirmed phase5 labels"]
                if solidworks_summary["confirmed_case_count"] != solidworks_summary["case_count"]
                else []
            ),
        ],
    }


def write_report(
    output_dir: Path,
    fusion_summary: dict[str, Any],
    solidworks_summary: dict[str, Any],
    coverage: dict[str, Any],
) -> None:
    report = f"""# Fusion360 到 SolidWorks 外部考试映射报告

## 结论

- 能否运行当前外部考试：{coverage['can_run_external_exam']}
- 能否给当前已确认案例判分：{coverage['can_score_current_confirmed_cases']}
- 能否声称 Fusion360 标签覆盖 SolidWorks 考题：{coverage['can_claim_fusion_to_solidworks_label_coverage']}
- 能否声称完整 5 组 SolidWorks 考试已准备好：{coverage['can_claim_full_5case_exam_readiness']}
- 当前状态：`{coverage['readiness_level']}`

## Fusion360 开发数据映射

- assembly 数量：{fusion_summary['assembly_count']}
- 样本总数：{fusion_summary['sample_count']}
- 正样本边：{fusion_summary['positive_count']}
- closed-world 负样本边：{fusion_summary['negative_count']}
- STEP 兼容几何缺失数：{fusion_summary['step_missing_count']}
- 未能映射到五类通用标签的正样本：{fusion_summary['unmapped_positive_count']}
- `pocket_mate` 候选正样本：{fusion_summary.get('pocket_mate_candidate_count', 0)}
- 映射标签计数：`{fusion_summary['mapped_label_counts']}`

说明：Fusion360 的负样本只是“同一 assembly 内没有记录 joint/contact”，不是物理或功能上绝对不能装。
`pocket_mate` 是由结构名和 contact/joint 证据挖出的候选标签，后续应抽样人工复核后再作为高置信监督。

## SolidWorks/STEP 外部考试集

- 外部 case 数量：{solidworks_summary['case_count']}
- 已有 phase5 标注 case：{solidworks_summary['annotated_case_count']}
- 已确认标注 case：{solidworks_summary['confirmed_case_count']}
- 可判分 pair：{solidworks_summary['scorable_pair_count']}
- 可判分正边：{solidworks_summary['scorable_positive_pair_count']}
- 可判分负边：{solidworks_summary['scorable_negative_pair_count']}
- 考题正边标签计数：`{solidworks_summary['positive_label_counts']}`
- 考题标签组合计数：`{solidworks_summary['positive_label_combo_counts']}`

## 标签覆盖审计

- Fusion360 已映射标签：`{coverage['fusion_mapped_labels']}`
- SolidWorks 考题正边标签：`{coverage['solidworks_exam_positive_labels']}`
- 已覆盖考题标签：`{coverage['covered_exam_labels']}`
- Fusion360 映射中缺失的考题标签：`{coverage['missing_exam_labels_in_fusion_mapping']}`

## 数据边界

1. 推理和阈值调参只能使用 Fusion360 输出。
2. SolidWorks `human_labels.json` 只能在最终评测阶段读取。
3. 不能按 `case_id`、文件名或人工备注写运行时规则。
4. SolidWorks 原始 STEP 文件只读；本脚本只计算 hash 和清单。
5. Fusion360→SolidWorks 的映射是“标签协议和 STEP 兼容几何路径”的映射，不是把 SolidWorks 考题答案混入训练集。

## 阻塞项

{chr(10).join('- ' + reason for reason in coverage['blocking_reasons']) if coverage['blocking_reasons'] else '- 无'}
"""
    (output_dir / "fusion360_to_solidworks_exam_mapping_report.md").write_text(
        report,
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fusion-graph-dir", type=Path, default=DEFAULT_GRAPH_DIR)
    parser.add_argument("--sw-root", type=Path, default=DEFAULT_SW_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fusion_rows, fusion_assemblies, fusion_summary = build_fusion_relation_rows(
        args.fusion_graph_dir.resolve()
    )
    pocket_candidates = [
        row
        for row in fusion_rows
        if row["direct_connection"]
        and "pocket_mate" in (row.get("mapped_relation_types") or [])
    ]
    fusion_summary["pocket_mate_candidate_count"] = len(pocket_candidates)
    solidworks_rows, solidworks_summary = build_solidworks_exam_rows(
        args.sw_root.resolve()
    )
    coverage = coverage_summary(fusion_summary, solidworks_summary)

    by_split: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in fusion_rows:
        by_split[row["split"]].append(row)

    write_jsonl(output_dir / "fusion360_relation_benchmark.jsonl", fusion_rows)
    for split in ("train", "dev", "test"):
        write_jsonl(output_dir / f"fusion360_{split}.jsonl", by_split[split])
    write_json(output_dir / "fusion360_assembly_split_manifest.json", fusion_assemblies)
    write_json(output_dir / "fusion360_pocket_mate_candidates.json", pocket_candidates)
    write_json(output_dir / "fusion360_relation_mapping_summary.json", fusion_summary)
    write_json(output_dir / "solidworks_external_exam_manifest.json", solidworks_rows)
    write_json(output_dir / "solidworks_exam_summary.json", solidworks_summary)
    write_json(output_dir / "exam_readiness.json", coverage)
    write_report(output_dir, fusion_summary, solidworks_summary, coverage)

    print(
        json.dumps(
            {
                "fusion_samples": fusion_summary["sample_count"],
                "fusion_positive": fusion_summary["positive_count"],
                "solidworks_scorable_pairs": solidworks_summary["scorable_pair_count"],
                "readiness_level": coverage["readiness_level"],
                "can_run_external_exam": coverage["can_run_external_exam"],
                "can_claim_full_5case_exam_readiness": coverage[
                    "can_claim_full_5case_exam_readiness"
                ],
                "blocking_reasons": coverage["blocking_reasons"],
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if coverage["can_run_external_exam"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
