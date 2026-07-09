"""Summarize known-group relation outputs without inventing ground truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=".")
    parser.add_argument("--cases", nargs="+", default=["1", "2", "3"])
    parser.add_argument("--output", default="KNOWN_GROUP_PHASE1_4_STATUS.json")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    rows = []
    for case in args.cases:
        path = root / case / "known_group_output" / "assembly_relations.json"
        if not path.is_file():
            rows.append({"case_id": case, "status": "missing", "path": str(path)})
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "case_id": case,
            "status": "complete",
            "part_count": len(result["parts"]),
            "assembly_connected": result["assembly_connected"],
            "pose_status": result["pose_status"],
            "direct_connection_count": len(result["direct_connections"]),
            "relation_count": len(result["assembly_relations"]),
            "relation_types": sorted({
                row["relation_type"] for row in result["assembly_relations"]
            }),
            "checked_pose_count": result["collision_validation"].get("checked_pose_count"),
            "collision_count": len(result["collision_validation"].get("collisions", [])),
            "joinable_status": result["candidate_summary"]["joinable"]["status"],
            "unresolved_parts": result["unresolved_parts"],
            "path": str(path),
        })
    payload = {
        "schema_version": "1.0.0",
        "task": "known_group_assembly_relation_recognition_phase_1_4",
        "ground_truth_evaluation_performed": False,
        "cases": rows,
        "summary": {
            "requested_case_count": len(args.cases),
            "completed_case_count": sum(row["status"] == "complete" for row in rows),
            "connected_case_count": sum(bool(row.get("assembly_connected")) for row in rows),
            "pose_valid_case_count": sum(row.get("pose_status") == "valid" for row in rows),
            "pose_failed_case_count": sum(row.get("pose_status") == "failed" for row in rows),
            "pose_uncertain_case_count": sum(row.get("pose_status") == "uncertain" for row in rows),
        },
        "evaluation_boundary": (
            "This report audits execution and physical validation only. "
            "Direct-edge and label accuracy require the user's real labels in phase 5."
        ),
    }
    output = root / args.output
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
