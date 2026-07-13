"""Generate an auditable all-pairs JoinABLe Pose frontier for one known group.

The input is an explicit list of parts already known to belong to one assembly.
The script only enumerates unordered pairs; part IDs organise output paths and
are never passed to JoinABLe as inference features.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path


def _part(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--part requires PART_ID=STEP_PATH")
    part_id, raw_path = value.split("=", 1)
    path = Path(raw_path)
    if not part_id or not path.is_file():
        raise argparse.ArgumentTypeError("--part requires an existing STEP path")
    return part_id, path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", action="append", type=_part, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--pose-top-k", type=int, default=5)
    parser.add_argument("--search-budget", type=int, default=8)
    parser.add_argument("--sample-count", type=int, default=1024)
    parser.add_argument("--exact-check-limit", type=int, default=10)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    args = parser.parse_args()

    parts = dict(args.part)
    if not 2 <= len(parts) <= 5 or len(set(parts)) != len(parts):
        raise SystemExit("known-group pair frontier supports 2..5 unique parts")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    e2e = Path(__file__).with_name("joinable_e2e.py")
    records = []
    for source, target in itertools.combinations(parts, 2):
        pair_dir = args.output_dir / f"pair_{source}__{target}"
        result_path = pair_dir / "joinable_e2e_result.json"
        command = [
            sys.executable, str(e2e), str(parts[source]), str(parts[target]),
            "--output-dir", str(pair_dir), "--device", args.device,
            "--top-k", str(args.top_k), "--pose-top-k", str(args.pose_top_k),
            "--search-budget", str(args.search_budget),
            "--sample-count", str(args.sample_count),
            "--exact-check-limit", str(args.exact_check_limit),
        ]
        try:
            completed = subprocess.run(
                command, capture_output=True, text=True,
                timeout=args.timeout_seconds, check=False,
            )
            status = "success" if completed.returncode == 0 and result_path.is_file() else "failed"
            records.append({
                "source": source,
                "target": target,
                "result_path": str(result_path.resolve()),
                "status": status,
                "return_code": completed.returncode,
                "stdout_tail": completed.stdout[-1000:],
                "stderr_tail": completed.stderr[-1000:],
            })
        except subprocess.TimeoutExpired:
            records.append({
                "source": source,
                "target": target,
                "result_path": str(result_path.resolve()),
                "status": "timeout",
                "return_code": None,
            })
    manifest = {
        "schema_version": "known_group_pair_frontier.v1",
        "part_sources": {part: str(path.resolve()) for part, path in parts.items()},
        "pair_count": len(records),
        "records": records,
        "part_ids_are_bookkeeping_only": True,
        "inference_feature_policy": "B-Rep geometry only; no part ID or path tokens.",
    }
    path = args.output_dir / "pair_frontier_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "pair_count": len(records),
        "success_count": sum(row["status"] == "success" for row in records),
        "manifest": str(path),
    }, ensure_ascii=False))
    return 0 if all(row["status"] == "success" for row in records) else 2


if __name__ == "__main__":
    raise SystemExit(main())
