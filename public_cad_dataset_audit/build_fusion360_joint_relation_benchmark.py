"""Build a large pair-level relation benchmark from Fusion360 Joint Dataset.

This script is intentionally conservative:

* It does not read SolidWorks exam labels.
* It treats every Fusion360 joint as a positive direct pair sample.
* It does not fabricate physical negatives from the Joint Dataset alone.
* Weakly mined pocket_mate samples are explicitly marked as weak labels.

The goal is Step 1 data engineering for pair-level relation recognition, not
final assembly acceptance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


COMMON_LABELS = [
    "coaxial",
    "planar_mate",
    "clearance",
    "pocket_mate",
    "planar_align",
]

POCKET_TEXT_HINTS = {
    "slot",
    "pin slot",
    "pinslot",
    "keyway",
    "key way",
    "groove",
    "channel",
    "rail",
    "slider",
    "slide",
    "track",
    "pocket",
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
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_bucket(key: str) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 10


def split_for_key(key: str) -> str:
    bucket = stable_bucket(key)
    if bucket == 0:
        return "test"
    if bucket == 1:
        return "dev"
    return "train"


def assembly_id_from_body_id(body_id: str | None) -> str | None:
    if not body_id:
        return None
    # Fusion body ids usually look like: 25370_adbde9bf_0025_1
    match = re.match(r"^(.+?_[0-9a-fA-F]{8})_\d+_\d+$", body_id)
    if match:
        return match.group(1)
    parts = body_id.split("_")
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return body_id


def compact_entity(entity: dict[str, Any] | None) -> dict[str, Any]:
    if not entity:
        return {
            "entity_type": None,
            "body": None,
            "topology_index": None,
            "surface_type": None,
            "curve_type": None,
            "point_on_entity": None,
            "bounding_box": None,
            "failure_reasons": ["entity_missing"],
        }
    return {
        "entity_type": entity.get("type"),
        "body": entity.get("body"),
        "topology_index": entity.get("index"),
        "surface_type": entity.get("surface_type"),
        "curve_type": entity.get("curve_type"),
        "point_on_entity": entity.get("point_on_entity"),
        "bounding_box": entity.get("bounding_box"),
        "failure_reasons": [],
    }


def geometry_summary(geometry_or_origin: dict[str, Any] | None) -> dict[str, Any]:
    geometry_or_origin = geometry_or_origin or {}
    entity = compact_entity(geometry_or_origin.get("entity_one"))
    return {
        "geometry_type": geometry_or_origin.get("geometry_type"),
        "key_point_type": geometry_or_origin.get("key_point_type"),
        "origin": geometry_or_origin.get("origin"),
        "primary_axis_vector": geometry_or_origin.get("primary_axis_vector"),
        "secondary_axis_vector": geometry_or_origin.get("secondary_axis_vector"),
        "tertiary_axis_vector": geometry_or_origin.get("tertiary_axis_vector"),
        "axis_line": geometry_or_origin.get("axis_line"),
        "entity": entity,
    }


def surface_tokens(joint: dict[str, Any]) -> set[str]:
    tokens: set[str] = set()
    for key in ("geometry_or_origin_one", "geometry_or_origin_two"):
        geo = joint.get(key) or {}
        if geo.get("geometry_type"):
            tokens.add(str(geo["geometry_type"]))
        entity = geo.get("entity_one") or {}
        for entity_key in ("surface_type", "curve_type", "type"):
            value = entity.get(entity_key)
            if value:
                tokens.add(str(value))
    return tokens


def classify_joint(
    joint: dict[str, Any],
) -> tuple[list[str], str | None, str, bool, list[str], list[str]]:
    motion = joint.get("joint_motion") or {}
    joint_type = str(motion.get("joint_type") or "UnknownJointType")
    tokens = surface_tokens(joint)
    token_text = " ".join(sorted(tokens)).lower()
    name_text = str(joint.get("name") or "").lower()
    text = f"{joint_type} {name_text}".lower()

    labels: list[str] = []
    reasons: list[str] = []
    unavailable: list[str] = []
    weak_label = False

    has_plane = "planesurfacetype" in token_text
    has_cylinder = "cylindersurfacetype" in token_text
    has_cone = "conesurfacetype" in token_text
    has_circle = "circle3dcurvetype" in token_text
    has_edge = "brepedge" in token_text

    if joint_type in {"RevoluteJointType", "CylindricalJointType"}:
        labels.append("coaxial")
        reasons.append(f"{joint_type} implies an axis-based joint.")
    if has_cylinder or has_cone or has_circle:
        labels.append("coaxial")
        reasons.append("Joint selected cylindrical/conical/circular interface geometry.")

    if joint_type in {"SliderJointType", "CylindricalJointType", "PinSlotJointType"}:
        labels.append("clearance")
        reasons.append(f"{joint_type} implies sliding/insertion freedom.")

    if has_plane:
        labels.append("planar_mate")
        reasons.append("Joint selected planar B-Rep face geometry.")

    if joint_type == "PlanarJointType":
        labels.append("planar_align")
        reasons.append("PlanarJointType maps to planar alignment freedom.")

    pocket_candidate = False
    if joint_type == "PinSlotJointType":
        pocket_candidate = True
        labels.append("pocket_mate")
        reasons.append("PinSlotJointType is direct pin/slot supervision.")
    elif any(hint in text for hint in POCKET_TEXT_HINTS):
        pocket_candidate = True
        labels.append("pocket_mate")
        weak_label = True
        reasons.append("Pocket candidate from joint name/type text hint.")
    elif joint_type == "SliderJointType" and has_plane:
        pocket_candidate = True
        labels.append("pocket_mate")
        weak_label = True
        reasons.append(
            "Weak pocket candidate: SliderJointType with planar guide faces; "
            "may represent rail/slot/channel engagement and requires audit."
        )

    labels = [label for label in COMMON_LABELS if label in set(labels)]
    if not labels:
        unavailable.append("mapped_relation_type")
        confidence = "unmapped"
        primary = None
    elif "pocket_mate" in labels and weak_label:
        confidence = "weak"
        primary = "pocket_mate"
        unavailable.append("pocket_mate_human_audit")
    elif "pocket_mate" in labels:
        confidence = "medium"
        primary = "pocket_mate"
        unavailable.append("pocket_mate_human_audit")
    elif joint_type == "PlanarJointType":
        confidence = "medium"
        primary = "planar_align"
    elif "coaxial" in labels:
        confidence = "medium"
        primary = "coaxial"
    else:
        confidence = "medium"
        primary = labels[0]

    if has_edge and not (has_cylinder or has_cone or has_circle):
        unavailable.append("edge_curve_subtype_if_absent")
    if not pocket_candidate:
        unavailable.append("slot_or_cavity_functional_role")

    return labels, primary, confidence, weak_label, reasons, sorted(set(unavailable))


def body_geometry_paths(joint_dir: Path, body_id: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for suffix in ("step", "smt", "obj", "json"):
        path = joint_dir / f"{body_id}.{suffix}"
        out[suffix] = {
            "path": str(path.resolve()),
            "exists": path.is_file(),
            "sha256": sha256_file(path) if suffix in {"step", "json"} else None,
        }
    return out


def build_rows(joint_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    label_counts: Counter[str] = Counter()
    primary_counts: Counter[str] = Counter()
    joint_type_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    split_label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    assembly_split: dict[str, str] = {}

    for joint_set_path in sorted(joint_dir.glob("joint_set_*.json")):
        try:
            data = load_json(joint_set_path)
        except Exception as exc:  # noqa: BLE001
            failures.append(
                {
                    "source_record_path": str(joint_set_path.resolve()),
                    "failure_reasons": [f"json_load_failed:{type(exc).__name__}:{exc}"],
                }
            )
            continue

        body_one = data.get("body_one")
        body_two = data.get("body_two")
        if not body_one or not body_two:
            failures.append(
                {
                    "source_record_path": str(joint_set_path.resolve()),
                    "failure_reasons": ["body_endpoint_missing"],
                }
            )
            continue
        assembly_ids = sorted(
            {
                value
                for value in (
                    assembly_id_from_body_id(body_one),
                    assembly_id_from_body_id(body_two),
                )
                if value
            }
        )
        assembly_key = "|".join(assembly_ids)
        split = split_for_key(assembly_key)
        for assembly_id in assembly_ids:
            old_split = assembly_split.get(assembly_id)
            if old_split and old_split != split:
                failures.append(
                    {
                        "source_record_path": str(joint_set_path.resolve()),
                        "failure_reasons": [
                            f"assembly_split_conflict:{assembly_id}:{old_split}:{split}"
                        ],
                    }
                )
            assembly_split[assembly_id] = split

        body_paths = {
            body_one: body_geometry_paths(joint_dir, body_one),
            body_two: body_geometry_paths(joint_dir, body_two),
        }
        missing_geometry = [
            f"{body_id}.{suffix}"
            for body_id, paths in body_paths.items()
            for suffix in ("step", "smt", "obj", "json")
            if not paths[suffix]["exists"]
        ]
        joints = data.get("joints") or []
        if not joints:
            failures.append(
                {
                    "source_record_path": str(joint_set_path.resolve()),
                    "failure_reasons": ["joint_set_contains_no_joints"],
                }
            )
            continue

        for joint_index, joint in enumerate(joints):
            labels, primary, confidence, weak_label, reasons, unavailable = classify_joint(joint)
            joint_type = str((joint.get("joint_motion") or {}).get("joint_type") or "UnknownJointType")
            joint_type_counts[joint_type] += 1
            for label in labels:
                label_counts[label] += 1
                split_label_counts[split][label] += 1
            if primary:
                primary_counts[primary] += 1

            row = {
                "schema_version": "1.0.0",
                "sample_id": f"fusion360_joint:{joint_set_path.stem}:joint_{joint_index:02d}",
                "source_dataset": "fusion360_gallery_joint",
                "source_record_path": str(joint_set_path.resolve()),
                "source_record_sha256": sha256_file(joint_set_path),
                "ground_truth_scope": "development_only_pair_joint_supervision",
                "do_not_use_for_solidworks_exam": True,
                "split": split,
                "assembly_id": assembly_key,
                "assembly_ids": assembly_ids,
                "joint_set_id": joint_set_path.stem,
                "joint_index": joint_index,
                "joint_name": joint.get("name"),
                "part_pair": [body_one, body_two],
                "body_one": body_one,
                "body_two": body_two,
                "solidworks_compatible_geometry_paths": [
                    body_paths[body_one]["step"]["path"],
                    body_paths[body_two]["step"]["path"],
                ],
                "solidworks_compatible_geometry_exists": all(
                    body_paths[body]["step"]["exists"] for body in (body_one, body_two)
                ),
                "body_geometry": body_paths,
                "direct_connection": True,
                "negative_definition": None,
                "source_joint_type": joint_type,
                "source_relation_types": [joint_type],
                "source_relation_kinds": [str(joint.get("type") or "Joint")],
                "joint_motion": joint.get("joint_motion"),
                "geometry_or_origin_one": geometry_summary(joint.get("geometry_or_origin_one")),
                "geometry_or_origin_two": geometry_summary(joint.get("geometry_or_origin_two")),
                "mapped_relation_types": labels,
                "primary_relation_type": primary,
                "mapping_confidence": confidence,
                "weak_label": weak_label or confidence == "weak",
                "mapping_reasons": reasons,
                "failure_reasons": missing_geometry,
                "unavailable_fields": sorted(set(unavailable)),
            }
            rows.append(row)
            split_counts[split] += 1

    summary = {
        "schema_version": "1.0.0",
        "source_dataset": "fusion360_gallery_joint",
        "joint_dir": str(joint_dir.resolve()),
        "sample_count": len(rows),
        "positive_count": len(rows),
        "negative_count": 0,
        "negative_policy": (
            "No physical negatives are fabricated from the Joint Dataset alone. "
            "Every row is a positive pair-level joint supervision sample."
        ),
        "assembly_count": len(assembly_split),
        "split_counts": dict(sorted(split_counts.items())),
        "mapped_label_counts": dict(sorted(label_counts.items())),
        "primary_label_counts": dict(sorted(primary_counts.items())),
        "source_joint_type_counts": dict(sorted(joint_type_counts.items())),
        "split_label_counts": {
            split: dict(sorted(counter.items()))
            for split, counter in sorted(split_label_counts.items())
        },
        "weak_label_count": sum(1 for row in rows if row["weak_label"]),
        "pocket_mate_candidate_count": sum(
            1 for row in rows if "pocket_mate" in row["mapped_relation_types"]
        ),
        "pocket_mate_weak_candidate_count": sum(
            1
            for row in rows
            if "pocket_mate" in row["mapped_relation_types"] and row["weak_label"]
        ),
        "unmapped_positive_count": sum(1 for row in rows if not row["mapped_relation_types"]),
        "solidworks_compatible_geometry_missing_count": sum(
            1 for row in rows if not row["solidworks_compatible_geometry_exists"]
        ),
        "failure_count": len(failures),
    }
    return rows, failures, summary


def write_split_report(
    output_dir: Path,
    summary: dict[str, Any],
    split_rows: dict[str, list[dict[str, Any]]],
) -> None:
    lines = [
        "# Fusion360 Joint Dataset split report",
        "",
        "## Summary",
        "",
        f"- sample_count: {summary['sample_count']}",
        f"- positive_count: {summary['positive_count']}",
        f"- negative_count: {summary['negative_count']}",
        f"- assembly_count: {summary['assembly_count']}",
        f"- weak_label_count: {summary['weak_label_count']}",
        f"- pocket_mate_candidate_count: {summary['pocket_mate_candidate_count']}",
        f"- pocket_mate_weak_candidate_count: {summary['pocket_mate_weak_candidate_count']}",
        f"- unmapped_positive_count: {summary['unmapped_positive_count']}",
        "",
        "## Split counts",
        "",
        "| split | samples | unique assemblies | labels |",
        "|---|---:|---:|---|",
    ]
    for split in ("train", "dev", "test"):
        rows = split_rows.get(split, [])
        assemblies = {row["assembly_id"] for row in rows}
        labels = Counter(
            label for row in rows for label in row.get("mapped_relation_types", [])
        )
        lines.append(
            f"| {split} | {len(rows)} | {len(assemblies)} | "
            f"`{dict(sorted(labels.items()))}` |"
        )
    lines += [
        "",
        "## Negative policy",
        "",
        summary["negative_policy"],
        "",
        "## Leakage guard",
        "",
        "- Split is assigned by stable hash of `assembly_id`, not by individual pair.",
        "- SolidWorks `human_labels.json` is not read by this script.",
        "- `do_not_use_for_solidworks_exam` is true for every Fusion360 row.",
        "",
    ]
    (output_dir / "split_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_label_mapping_report(output_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Fusion360 Joint to common relation label mapping report",
        "",
        "## Common label schema",
        "",
        "- `coaxial`: revolute/cylindrical joints or cylindrical/conical/circular selected geometry.",
        "- `planar_mate`: selected planar B-Rep face geometry.",
        "- `clearance`: slider/cylindrical/pin-slot motion implying insertion/sliding freedom.",
        "- `pocket_mate`: pin-slot joints and weakly mined slot/rail/channel style candidates.",
        "- `planar_align`: Fusion360 `PlanarJointType`.",
        "",
        "## Weak label policy",
        "",
        "`pocket_mate` labels mined from `SliderJointType` + planar guide faces are marked as `weak_label=true` and `mapping_confidence=weak`. They are candidates for model training/audit, not hard engineering truth.",
        "",
        "## Negative policy",
        "",
        summary["negative_policy"],
        "",
        "## Counts",
        "",
        f"- source_joint_type_counts: `{summary['source_joint_type_counts']}`",
        f"- mapped_label_counts: `{summary['mapped_label_counts']}`",
        f"- primary_label_counts: `{summary['primary_label_counts']}`",
        f"- weak_label_count: {summary['weak_label_count']}",
        f"- pocket_mate_candidate_count: {summary['pocket_mate_candidate_count']}",
        f"- unmapped_positive_count: {summary['unmapped_positive_count']}",
        "",
        "## Known limitations",
        "",
        "- Joint Dataset gives pair-level joint supervision, not full assembly closed-world negatives.",
        "- Weak `pocket_mate` candidates require human audit before being treated as high-confidence labels.",
        "- Functional roles such as shaft/hub/bearing are generally unavailable in this source and must not be hallucinated.",
        "- This benchmark is for development only and must not include SolidWorks exam answers.",
        "",
    ]
    (output_dir / "label_mapping_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def write_candidate_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "sample_id",
        "split",
        "assembly_id",
        "joint_set_id",
        "joint_index",
        "joint_name",
        "source_joint_type",
        "part_pair",
        "mapped_relation_types",
        "primary_relation_type",
        "mapping_confidence",
        "weak_label",
        "mapping_reasons",
        "source_record_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: json.dumps(row.get(field), ensure_ascii=False)
                    if isinstance(row.get(field), (list, dict))
                    else row.get(field)
                    for field in fields
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--joint-dir",
        type=Path,
        default=Path(r"D:\Model_match_public_data\fusion360_joint_full\j1.0.0\joint"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "public_cad_dataset_audit/outputs/fusion360_joint_relation_benchmark"
        ),
    )
    args = parser.parse_args()

    joint_dir = args.joint_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, failures, summary = build_rows(joint_dir)
    split_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        split_rows[row["split"]].append(row)

    write_jsonl(output_dir / "fusion360_relation_benchmark.jsonl", rows)
    for split in ("train", "dev", "test"):
        write_jsonl(output_dir / f"fusion360_{split}.jsonl", split_rows[split])
    pocket_rows = [
        row for row in rows if "pocket_mate" in row.get("mapped_relation_types", [])
    ]
    write_json(output_dir / "fusion360_pocket_mate_candidates.json", pocket_rows)
    write_candidate_csv(output_dir / "fusion360_pocket_mate_candidates.csv", pocket_rows)
    write_json(output_dir / "fusion360_joint_relation_summary.json", summary)
    write_json(output_dir / "conversion_failures.json", failures)
    write_split_report(output_dir, summary, split_rows)
    write_label_mapping_report(output_dir, summary)

    print(
        json.dumps(
            {
                "sample_count": summary["sample_count"],
                "assembly_count": summary["assembly_count"],
                "mapped_label_counts": summary["mapped_label_counts"],
                "pocket_mate_candidate_count": summary["pocket_mate_candidate_count"],
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    missing_labels = [
        label for label in COMMON_LABELS if summary["mapped_label_counts"].get(label, 0) == 0
    ]
    return 0 if rows and not missing_labels else 2


if __name__ == "__main__":
    raise SystemExit(main())
