"""Run Step-3 geometry validation and pose solving on Fusion360 STEP pairs.

This smoke test uses Fusion360 selected geometry only.  It does not read
SolidWorks external exam labels.  Each selected positive pair is copied into an
isolated known-group case and processed by the existing analytic geometry +
OCCT pose pipeline.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SW_ROOT = PROJECT_ROOT / "sw"
if str(SW_ROOT) not in sys.path:
    sys.path.insert(0, str(SW_ROOT))

from known_group_assembly import run_known_group_assembly  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value)[:120]


def existing_positive_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        if not row.get("direct_connection"):
            continue
        paths = [Path(path) for path in row.get("solidworks_compatible_geometry_paths") or []]
        if len(paths) != 2 or not all(path.is_file() for path in paths):
            continue
        size = sum(path.stat().st_size for path in paths)
        item = dict(row)
        item["_step_paths"] = paths
        item["_total_size"] = size
        selected.append(item)
    return sorted(selected, key=lambda row: (row["_total_size"], row["sample_id"]))


def prepare_case(row: dict[str, Any], output_root: Path, index: int) -> Path:
    case_dir = output_root / f"case_{index:02d}_{safe_name(row['sample_id'])}"
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True)
    copied = []
    for part_index, source in enumerate(row["_step_paths"], 1):
        destination = case_dir / f"P{part_index:02d}_{source.name}"
        shutil.copy2(source, destination)
        copied.append(destination.name)
    write_json(
        case_dir / "fusion360_truth_hint.json",
        {
            "sample_id": row["sample_id"],
            "source_graph_path": row.get("source_graph_path"),
            "part_names": row.get("part_names"),
            "copied_parts": copied,
            "fusion360_relation_types": row.get("mapped_relation_types") or [],
            "ground_truth_scope": "fusion360_internal_smoke_only",
            "not_solidworks_exam": True,
        },
    )
    return case_dir


def compare_labels(predicted: list[str], expected: list[str]) -> dict[str, Any]:
    pred = set(predicted)
    exp = set(expected)
    return {
        "expected": sorted(exp),
        "predicted": sorted(pred),
        "overlap": sorted(pred & exp),
        "missing": sorted(exp - pred),
        "extra": sorted(pred - exp),
        "has_any_overlap": bool(pred & exp),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--beam-width", type=int, default=12)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = existing_positive_rows(read_jsonl(args.input_jsonl.resolve()))
    selected = rows[: max(1, args.limit)]

    results = []
    for index, row in enumerate(selected, 1):
        case_dir = prepare_case(row, output_dir / "cases", index)
        try:
            result = run_known_group_assembly(
                case_dir,
                output_dir=case_dir / "known_group_output",
                beam_width=args.beam_width,
            )
            predicted_types = sorted({
                relation
                for connection in result.get("direct_connections", [])
                for relation in (
                    connection.get("supporting_relation_types")
                    or [connection.get("primary_relation_type")]
                )
                if relation
            })
            status = "success"
            failure_reasons = []
        except Exception as exc:  # pragma: no cover - explicit audit output
            result = {}
            predicted_types = []
            status = "failed"
            failure_reasons = [f"{type(exc).__name__}:{exc}"]
        results.append({
            "sample_id": row["sample_id"],
            "case_dir": str(case_dir),
            "status": status,
            "part_names": row.get("part_names"),
            "fusion360_relation_types": row.get("mapped_relation_types") or [],
            "pose_status": result.get("pose_status"),
            "assembly_connected": result.get("assembly_connected"),
            "direct_connection_count": len(result.get("direct_connections", [])),
            "predicted_relation_types": predicted_types,
            "label_overlap": compare_labels(
                predicted_types,
                row.get("mapped_relation_types") or [],
            ),
            "collision_count": len(
                (result.get("collision_validation") or {}).get("collisions") or []
            ),
            "failure_reasons": failure_reasons,
        })

    summary = {
        "schema_version": "1.0.0",
        "task": "step3_fusion_geometry_pose_smoke",
        "input_jsonl": str(args.input_jsonl.resolve()),
        "selected_count": len(selected),
        "success_count": sum(row["status"] == "success" for row in results),
        "pose_valid_count": sum(row.get("pose_status") == "valid" for row in results),
        "pose_failed_count": sum(row.get("pose_status") == "failed" for row in results),
        "pose_uncertain_count": sum(row.get("pose_status") == "uncertain" for row in results),
        "label_any_overlap_count": sum(
            bool(row["label_overlap"]["has_any_overlap"]) for row in results
        ),
        "results": results,
        "limitations": [
            "This is a Fusion360 geometry smoke test, not the SolidWorks external exam.",
            "Fusion360 contact labels are imperfect and used only as internal consistency hints.",
            "Pose success proves physical feasibility, not final functional correctness.",
        ],
    }
    write_json(output_dir / "step3_pose_smoke_summary.json", summary)
    report = f"""# Step 3 几何验证与 pose 求解 smoke report

- 输入：Fusion360 selected STEP 正边
- 样本数：{summary['selected_count']}
- 运行成功：{summary['success_count']}
- pose valid：{summary['pose_valid_count']}
- pose failed：{summary['pose_failed_count']}
- pose uncertain：{summary['pose_uncertain_count']}
- 与 Fusion360 映射标签有交集：{summary['label_any_overlap_count']}

该报告只证明 Step 3 几何/pose 求解链路可运行；不作为 SolidWorks 外部考试分数。
"""
    (output_dir / "step3_pose_smoke_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({
        "selected_count": summary["selected_count"],
        "success_count": summary["success_count"],
        "pose_valid_count": summary["pose_valid_count"],
        "pose_failed_count": summary["pose_failed_count"],
        "pose_uncertain_count": summary["pose_uncertain_count"],
        "output_dir": str(output_dir),
    }, ensure_ascii=False, indent=2))
    return 0 if summary["success_count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
