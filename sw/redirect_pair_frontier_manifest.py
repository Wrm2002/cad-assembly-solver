"""Create an auditable manifest pointing at augmented pair-frontier results.

The mapping is structural: each successful record uses a result file with the
given sibling filename in the same pair directory.  It never inspects a part
name, case id, or mechanical-family label.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_manifest", type=Path)
    parser.add_argument("output_manifest", type=Path)
    parser.add_argument("--result-name", default="joinable_e2e_learned_pose_v2.json")
    parser.add_argument("--version", default="v2")
    args = parser.parse_args()
    payload = json.loads(args.input_manifest.read_text(encoding="utf-8"))
    changed = 0
    for record in payload.get("records") or []:
        if record.get("status") != "success":
            continue
        replacement = Path(str(record["result_path"])).with_name(args.result_name)
        if not replacement.is_file():
            raise FileNotFoundError(f"augmented_pair_result_missing:{replacement}")
        record["result_path"] = str(replacement.resolve())
        record["learned_pair_pose"] = {
            "version": str(args.version),
            "role": "soft_prior_and_candidate_ranking_only",
            "case_specific_override": False,
        }
        changed += 1
    payload["schema_version"] = f"known_group_pair_frontier_learned_pose.{args.version}"
    payload["learned_pose_result_name"] = args.result_name
    payload["case_specific_override"] = False
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"changed_records": changed, "output": str(args.output_manifest)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
