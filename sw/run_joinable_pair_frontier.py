"""Generate an auditable all-pairs JoinABLe frontier for one 2..5-part group.

The input is an explicit list of part *instances*.  The runner enumerates every
unordered pair and invokes the canonical :mod:`joinable_e2e` entry point in an
isolated subprocess.  A pair is complete only when the subprocess succeeds and
its JSON result satisfies the expected contract.  Missing inputs, timeouts,
non-zero exits, and malformed/missing result files remain explicit failures in
the manifest; they are never converted into successful analytic fallbacks.

Part IDs only organise output paths and reports.  They are not passed to the
JoinABLe model as inference features.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Sequence


SCHEMA_VERSION = "known_group_pair_frontier.v2"
CACHE_SCHEMA_VERSION = "known_group_pair_frontier_cache.v1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "joinable_migration_audit"
    / "vendor"
    / "JoinABLe"
    / "pretrained"
    / "paper"
    / "last_run_0.ckpt"
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _part(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--part requires PART_ID=STEP_PATH")
    part_id, raw_path = value.split("=", 1)
    if not part_id or not raw_path:
        raise argparse.ArgumentTypeError("--part requires PART_ID=STEP_PATH")
    # Existence is deliberately checked while producing pair records.  This
    # lets a batch manifest name every pair affected by a missing input instead
    # of aborting before an auditable result can be written.
    return part_id, Path(raw_path)


def _normal_path(path: Path) -> str:
    return os.path.normcase(str(path.resolve()))


def _file_signature(path: Path) -> dict[str, Any]:
    resolved = _normal_path(path)
    try:
        stat = path.stat()
    except OSError:
        return {"path": resolved, "exists": False}
    return {
        "path": resolved,
        "exists": path.is_file(),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _load_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, "result file was not produced"
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, f"result JSON is unreadable: {type(exc).__name__}: {exc}"
    if not isinstance(value, dict):
        return None, "result JSON root must be an object"
    return value, None


def _validate_result(
    result_path: Path,
    step_a: Path,
    step_b: Path,
) -> tuple[bool, str | None]:
    result, error = _load_json(result_path)
    if result is None:
        return False, error
    schema = result.get("schema_version")
    if not isinstance(schema, str) or not schema.startswith("joinable_e2e."):
        return False, "result schema_version is not a joinable_e2e contract"
    expected_a = _normal_path(step_a)
    expected_b = _normal_path(step_b)
    actual_a = result.get("part_a_fixed")
    actual_b = result.get("part_b_moving")
    if not isinstance(actual_a, str) or os.path.normcase(actual_a) != expected_a:
        return False, "result part_a_fixed does not match this pair"
    if not isinstance(actual_b, str) or os.path.normcase(actual_b) != expected_b:
        return False, "result part_b_moving does not match this pair"
    required_objects = (
        "gnn_inference",
        "joint_hypotheses",
        "pose_search",
        "acceptance_boundary",
    )
    missing = [
        name
        for name in required_objects
        if not isinstance(result.get(name), dict)
    ]
    if missing:
        return False, f"result is missing required object(s): {', '.join(missing)}"
    return True, None


def _cache_contract(
    step_a: Path,
    step_b: Path,
    *,
    e2e: Path,
    checkpoint: Path | None,
    device: str,
    top_k: int,
    pose_top_k: int,
    search_budget: int,
    sample_count: int,
    exact_check_limit: int,
    run_search: bool,
) -> dict[str, Any]:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "engine": _file_signature(e2e),
        "inputs": [_file_signature(step_a), _file_signature(step_b)],
        "checkpoint": _file_signature(checkpoint or DEFAULT_CHECKPOINT),
        "options": {
            "device": device,
            "top_k": int(top_k),
            "pose_top_k": int(pose_top_k),
            "search_budget": int(search_budget),
            "sample_count": int(sample_count),
            "exact_check_limit": int(exact_check_limit),
            "run_search": bool(run_search),
        },
    }


def _cache_matches(path: Path, expected: dict[str, Any]) -> bool:
    cached, _ = _load_json(path)
    return cached == expected


def _build_command(
    e2e: Path,
    step_a: Path,
    step_b: Path,
    pair_dir: Path,
    *,
    checkpoint: Path | None,
    device: str,
    top_k: int,
    pose_top_k: int,
    search_budget: int,
    sample_count: int,
    exact_check_limit: int,
    run_search: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(e2e),
        str(step_a),
        str(step_b),
        "--output-dir",
        str(pair_dir),
        "--device",
        device,
        "--top-k",
        str(top_k),
        "--pose-top-k",
        str(pose_top_k),
        "--search-budget",
        str(search_budget),
        "--sample-count",
        str(sample_count),
        "--exact-check-limit",
        str(exact_check_limit),
    ]
    if checkpoint is not None:
        command.extend(("--checkpoint", str(checkpoint)))
    if not run_search:
        command.append("--no-search")
    return command


def run_pair_frontier(
    part_rows: Sequence[tuple[str, Path]],
    output_dir: Path,
    *,
    checkpoint: Path | None = None,
    device: str = "cpu",
    top_k: int = 20,
    pose_top_k: int = 5,
    search_budget: int = 8,
    sample_count: int = 1024,
    exact_check_limit: int = 10,
    timeout_seconds: int = 180,
    run_search: bool = True,
    force: bool = False,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> dict[str, Any]:
    """Run or reuse every unordered pair and return the persisted manifest."""

    if not 2 <= len(part_rows) <= 5:
        raise ValueError("known-group pair frontier supports 2..5 parts")
    part_ids = [part_id for part_id, _ in part_rows]
    if len(set(part_ids)) != len(part_ids):
        raise ValueError("known-group pair frontier requires unique part IDs")
    if any(not part_id for part_id in part_ids):
        raise ValueError("part IDs must be non-empty")

    output_dir.mkdir(parents=True, exist_ok=True)
    runner = command_runner or subprocess.run
    e2e = Path(__file__).with_name("joinable_e2e.py")
    parts = dict(part_rows)
    records: list[dict[str, Any]] = []

    for pair_index, (source, target) in enumerate(itertools.combinations(part_ids, 2)):
        step_a, step_b = parts[source], parts[target]
        pair_dir = output_dir / f"pair_{source}__{target}"
        result_path = pair_dir / "joinable_e2e_result.json"
        cache_path = pair_dir / "pair_cache_manifest.json"
        pair_record: dict[str, Any] = {
            "pair_index": pair_index,
            "source": source,
            "target": target,
            "part_a": str(step_a.resolve()),
            "part_b": str(step_b.resolve()),
            "result_path": str(result_path.resolve()),
            "cache_manifest": str(cache_path.resolve()),
            "pipeline_complete": False,
            "cache_hit": False,
        }

        missing = [
            str(path.resolve())
            for path in (step_a, step_b)
            if not path.is_file()
        ]
        if missing:
            pair_record.update({
                "status": "missing_input",
                "return_code": None,
                "missing_inputs": missing,
                "error": "one or more pair inputs do not exist or are not files",
            })
            records.append(pair_record)
            continue

        cache_contract = _cache_contract(
            step_a,
            step_b,
            e2e=e2e,
            checkpoint=checkpoint,
            device=device,
            top_k=top_k,
            pose_top_k=pose_top_k,
            search_budget=search_budget,
            sample_count=sample_count,
            exact_check_limit=exact_check_limit,
            run_search=run_search,
        )
        result_valid, result_error = _validate_result(result_path, step_a, step_b)
        if (
            not force
            and result_valid
            and _cache_matches(cache_path, cache_contract)
        ):
            pair_record.update({
                "status": "cached",
                "return_code": 0,
                "pipeline_complete": True,
                "cache_hit": True,
            })
            records.append(pair_record)
            continue

        command = _build_command(
            e2e,
            step_a,
            step_b,
            pair_dir,
            checkpoint=checkpoint,
            device=device,
            top_k=top_k,
            pose_top_k=pose_top_k,
            search_budget=search_budget,
            sample_count=sample_count,
            exact_check_limit=exact_check_limit,
            run_search=run_search,
        )
        try:
            completed = runner(
                command,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            pair_record.update({
                "status": "timeout",
                "return_code": None,
                "error": f"joinable_e2e exceeded {timeout_seconds} seconds",
                "stdout_tail": (exc.stdout or "")[-1000:] if isinstance(exc.stdout, str) else "",
                "stderr_tail": (exc.stderr or "")[-1000:] if isinstance(exc.stderr, str) else "",
            })
            records.append(pair_record)
            continue
        except OSError as exc:
            pair_record.update({
                "status": "execution_error",
                "return_code": None,
                "error": f"{type(exc).__name__}: {exc}",
            })
            records.append(pair_record)
            continue

        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        pair_record.update({
            "return_code": int(completed.returncode),
            "stdout_tail": stdout[-1000:],
            "stderr_tail": stderr[-1000:],
        })
        if completed.returncode != 0:
            pair_record.update({
                "status": "failed",
                "error": "joinable_e2e returned a non-zero exit code",
            })
            records.append(pair_record)
            continue

        result_valid, result_error = _validate_result(result_path, step_a, step_b)
        if not result_valid:
            pair_record.update({
                "status": "invalid_result",
                "error": result_error or "joinable_e2e result failed validation",
            })
            records.append(pair_record)
            continue

        _write_json(cache_path, cache_contract)
        pair_record.update({
            "status": "success",
            "pipeline_complete": True,
        })
        records.append(pair_record)

    expected_pair_count = len(part_rows) * (len(part_rows) - 1) // 2
    completed_count = sum(row["pipeline_complete"] for row in records)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "part_sources": {part: str(path.resolve()) for part, path in part_rows},
        "part_count": len(part_rows),
        "expected_pair_count": expected_pair_count,
        "pair_count": len(records),
        "completed_count": completed_count,
        # Kept for consumers of the v1 command summary.  A valid cache hit is a
        # completed pair, not a failed inference.
        "success_count": completed_count,
        "cache_hit_count": sum(row["cache_hit"] for row in records),
        "failed_count": len(records) - completed_count,
        "pipeline_complete": (
            len(records) == expected_pair_count
            and completed_count == expected_pair_count
        ),
        "records": records,
        "part_ids_are_bookkeeping_only": True,
        "inference_feature_policy": "B-Rep geometry only; no part ID or path tokens.",
        "failure_policy": (
            "Missing inputs, timeout/non-zero subprocess exits, and missing or "
            "malformed joinable_e2e results remain explicit incomplete pairs."
        ),
        "acceptance_boundary": (
            "This manifest audits candidate-frontier coverage only and cannot "
            "auto-accept an assembly group."
        ),
    }
    _write_json(output_dir / "pair_frontier_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", action="append", type=_part, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--pose-top-k", type=int, default=5)
    parser.add_argument("--search-budget", type=int, default=8)
    parser.add_argument("--sample-count", type=int, default=1024)
    parser.add_argument("--exact-check-limit", type=int, default=10)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--no-search", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    try:
        manifest = run_pair_frontier(
            args.part,
            args.output_dir,
            checkpoint=args.checkpoint,
            device=args.device,
            top_k=args.top_k,
            pose_top_k=args.pose_top_k,
            search_budget=args.search_budget,
            sample_count=args.sample_count,
            exact_check_limit=args.exact_check_limit,
            timeout_seconds=args.timeout_seconds,
            run_search=not args.no_search,
            force=args.force,
        )
    except ValueError as exc:
        parser.error(str(exc))

    path = args.output_dir / "pair_frontier_manifest.json"
    print(json.dumps({
        "pair_count": manifest["pair_count"],
        "success_count": manifest["success_count"],
        "cache_hit_count": manifest["cache_hit_count"],
        "failed_count": manifest["failed_count"],
        "pipeline_complete": manifest["pipeline_complete"],
        "manifest": str(path),
    }, ensure_ascii=False))
    return 0 if manifest["pipeline_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
