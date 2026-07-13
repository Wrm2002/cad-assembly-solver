"""Build a human review pack for mined Fusion360 pocket_mate candidates.

The pack is intentionally simple: CSV/JSON/Markdown plus copied STEP/PNG files
for the selected candidate pairs.  Reviewers only need to fill
`review_label` with one of:

* true_pocket_mate
* false_pocket_mate
* uncertain
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_name(value: str) -> str:
    keep = []
    for char in value:
        if char.isalnum() or char in "-_.":
            keep.append(char)
        else:
            keep.append("_")
    return "".join(keep)[:120]


def candidate_priority(row: dict[str, Any]) -> tuple[int, str, str]:
    reasons = " ".join(row.get("mapping_reasons") or []).lower()
    names = " ".join(row.get("part_name_evidence") or []).lower()
    source_types = " ".join(row.get("source_relation_types") or []).lower()
    score = 0
    if "multiple planar" in reasons:
        score -= 20
    if any(token in names for token in ("slot", "socket", "groove", "channel", "rail", "keyway", "pocket")):
        score -= 15
    if "cylindrical" in reasons or "coaxial" in row.get("mapped_relation_types", []):
        score -= 5
    if "contact" in source_types:
        score -= 3
    return (score, str(row.get("split") or ""), str(row.get("sample_id") or ""))


def pick_review_rows(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    # Keep deterministic and mildly balanced across split.
    by_split: dict[str, list[dict[str, Any]]] = {}
    for row in sorted(rows, key=candidate_priority):
        by_split.setdefault(str(row.get("split") or "unknown"), []).append(row)
    selected: list[dict[str, Any]] = []
    target_order = ["train", "dev", "test", "unknown"]
    while len(selected) < limit:
        changed = False
        for split in target_order:
            bucket = by_split.get(split) or []
            if bucket and len(selected) < limit:
                selected.append(bucket.pop(0))
                changed = True
        if not changed:
            break
    return selected


def copy_if_exists(src: str | None, dst: Path) -> str | None:
    if not src:
        return None
    src_path = Path(src)
    if not src_path.is_file():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst)
    return str(dst.resolve())


def build_pack(input_json: Path, output_dir: Path, limit: int) -> dict[str, Any]:
    rows = load_json(input_json)
    selected = pick_review_rows(rows, limit)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir = output_dir / "candidates"
    sheet_rows: list[dict[str, Any]] = []

    for index, row in enumerate(selected, start=1):
        case_id = f"candidate_{index:03d}_{safe_name(str(row.get('assembly_id')))}"
        case_dir = candidate_dir / case_id
        step_paths = row.get("solidworks_compatible_geometry_paths") or []
        copied_steps = []
        copied_pngs = []
        for part_index, src in enumerate(step_paths, start=1):
            src_path = Path(src)
            copied = copy_if_exists(str(src_path), case_dir / f"part_{part_index:02d}_{safe_name(src_path.stem)}.step")
            copied_steps.append(copied)
            png_src = src_path.with_suffix(".png")
            copied_pngs.append(
                copy_if_exists(str(png_src), case_dir / f"part_{part_index:02d}_{safe_name(src_path.stem)}.png")
            )

        review_row = {
            "review_id": case_id,
            "sample_id": row.get("sample_id"),
            "assembly_id": row.get("assembly_id"),
            "split": row.get("split"),
            "part_names": " | ".join(str(x) for x in row.get("part_name_evidence") or row.get("part_names") or []),
            "source_relation_types": " | ".join(str(x) for x in row.get("source_relation_types") or []),
            "mapped_relation_types": " | ".join(str(x) for x in row.get("mapped_relation_types") or []),
            "mapping_confidence": row.get("mapping_confidence"),
            "weak_label": row.get("weak_label"),
            "mapping_reasons": " | ".join(str(x) for x in row.get("mapping_reasons") or []),
            "original_step_paths": " | ".join(str(x) for x in step_paths),
            "copied_step_paths": " | ".join(str(x) for x in copied_steps if x),
            "copied_png_paths": " | ".join(str(x) for x in copied_pngs if x),
            "review_label": "",
            "review_notes": "",
        }
        sheet_rows.append(review_row)
        (case_dir / "candidate_metadata.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    fields = list(sheet_rows[0].keys()) if sheet_rows else []
    with (output_dir / "review_sheet.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sheet_rows)
    (output_dir / "review_sheet.json").write_text(
        json.dumps(sheet_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    split_counts = Counter(row["split"] for row in sheet_rows)
    readme = [
        "# pocket_mate 人工抽查包",
        "",
        f"- candidate_count: {len(sheet_rows)}",
        f"- split_counts: `{dict(sorted(split_counts.items()))}`",
        "",
        "## 标注方式",
        "",
        "请打开 `review_sheet.csv`，只填写两列：",
        "",
        "- `review_label`: `true_pocket_mate` / `false_pocket_mate` / `uncertain`",
        "- `review_notes`: 可选中文说明",
        "",
        "判断标准：",
        "",
        "- `true_pocket_mate`: 明显是卡槽、插槽、口袋、导轨、凹腔嵌合、pin-slot/keyway 这类嵌入式约束。",
        "- `false_pocket_mate`: 只是普通同轴、普通平面贴合、偶然接触、螺钉孔/销孔但没有口袋/槽语义。",
        "- `uncertain`: 看不清或需要装配上下文。",
        "",
        "每个候选的 STEP/PNG 文件已复制到 `candidates/` 子目录；原始路径也保留在 CSV 中。",
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")

    summary = {
        "input_json": str(input_json.resolve()),
        "output_dir": str(output_dir.resolve()),
        "candidate_count_total": len(rows),
        "review_count": len(sheet_rows),
        "split_counts": dict(sorted(split_counts.items())),
        "review_labels": ["true_pocket_mate", "false_pocket_mate", "uncertain"],
    }
    (output_dir / "review_pack_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-json",
        type=Path,
        default=Path("public_cad_dataset_audit/outputs/fusion360_assembly_benchmark/fusion360_pocket_mate_candidates.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("public_cad_dataset_audit/outputs/pocket_mate_review_pack_50"),
    )
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()
    summary = build_pack(args.input_json.resolve(), args.output_dir.resolve(), args.limit)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
