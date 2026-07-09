"""Prepare an unbiased two-pass human annotation pack for phase 5."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from render_parts_tray import render_parts_tray


LABELS = ["coaxial", "clearance", "planar_mate", "planar_align", "pocket_mate"]


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=".")
    parser.add_argument("--cases", nargs="+", default=["1", "2", "3"])
    parser.add_argument("--output", default="phase5_annotation_pack")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output = root / args.output
    index = []
    for case_id in args.cases:
        case_dir = root / case_id
        prediction_path = case_dir / "known_group_output" / "assembly_relations.json"
        prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
        parts = list(prediction["parts"])
        case_out = output / f"case_{case_id}"
        case_out.mkdir(parents=True, exist_ok=True)
        system_prediction = {
            "case_id": case_id,
            "warning": "系统预测仅供预测对照，不得直接复制为人工真值。",
            "pose_status": prediction["pose_status"],
            "direct_connections": prediction["direct_connections"],
            "assembly_relations": prediction["assembly_relations"],
        }
        write_json(case_out / "system_prediction.json", system_prediction)
        pairs = []
        for index_number, (a, b) in enumerate(itertools.combinations(parts, 2), 1):
            pairs.append({
                "pair_id": f"case_{case_id}_pair_{index_number:02d}",
                "parts": [a, b],
                "direct_connection": None,
                "relation_types": [],
                "primary_relation_type": None,
                "annotation_confidence": None,
                "notes": "",
            })
        blank = {
            "schema_version": "1.0.0",
            "case_id": case_id,
            "annotation_status": "not_started",
            "annotator": None,
            "annotation_date": None,
            "parts": parts,
            "allowed_relation_types": LABELS,
            "pass_1_direct_relations": pairs,
            "pass_2_interfaces": [],
            "assembly_level_notes": "",
        }
        write_json(case_out / "human_labels.json", blank)
        labels = [f"P{number:02d}  {name}" for number, name in enumerate(parts, 1)]
        tray = render_parts_tray(
            [case_dir / name for name in parts],
            labels,
            case_out / "parts_tray.png",
            cols=min(2, len(parts)),
            cell_width=520,
            cell_height=400,
        )
        index.append({
            "case_id": case_id,
            "part_count": len(parts),
            "pair_count": len(pairs),
            "human_labels": str((case_out / "human_labels.json").resolve()),
            "system_prediction": str((case_out / "system_prediction.json").resolve()),
            "parts_tray": str(tray.resolve()),
        })
    write_json(output / "annotation_index.json", {"cases": index})
    instructions = """# 第五阶段人工标注说明

第一轮只填写 `human_labels.json` 中的 `pass_1_direct_relations`：

1. `direct_connection`：两个零件是否直接装配，填写 `true` 或 `false`；不确定时保持 `null`。
2. `relation_types`：可多选 `coaxial`、`clearance`、`planar_mate`、`planar_align`、`pocket_mate`。
3. `primary_relation_type`：上述标签中的主标签。
4. `annotation_confidence`：填写 `high`、`medium` 或 `low`。
5. 不要先查看 `system_prediction.json`；完成真值后再打开它做差异复核。

一条连接可以有多个标签。例如法兰通常同时是同轴和端面贴合。没有直接接触的零件对必须标为 `false`，即使它们属于同一个装配体。

第一轮完成后再生成第二轮接口面/边标注表，避免在错误零件对上浪费face级标注工作。
"""
    (output / "标注说明.md").write_text(instructions, encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
