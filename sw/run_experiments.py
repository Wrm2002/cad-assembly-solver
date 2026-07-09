"""Run BFS/scoring/pruning/reliable ablations and snapshot every result."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from evaluate_results import _case_metrics, _mean
from run_cases import _case_id, discover_cases


MODES = {
    "baseline_1_bfs": ["--solver", "bfs"],
    "baseline_2_bfs_scoring": ["--solver", "bfs", "--enable-scoring"],
    "baseline_3_bfs_scoring_pruning": ["--solver", "bfs", "--enable-pruning"],
    "proposed_reliable": ["--solver", "reliable"],
}


def _run(command, cwd, timeout, log):
    started = time.perf_counter()
    result = subprocess.run(
        command, cwd=cwd, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout, check=False,
    )
    runtime = time.perf_counter() - started
    log.write_text(
        f"COMMAND: {subprocess.list2cmdline(command)}\nRETURN: {result.returncode}\n"
        f"RUNTIME: {runtime:.6f}\n--- STDOUT ---\n{result.stdout}\n"
        f"--- STDERR ---\n{result.stderr}\n",
        encoding="utf-8",
    )
    return result, runtime


def _snapshot(case, destination):
    destination.mkdir(parents=True, exist_ok=True)
    names = [
        "assembly_manifest.json", "assembly_diagnostics.json", "assembly_report.txt",
        "assembly_validation.json", "kept_matches.json", "removed_matches.json",
        "scored_matches.json", "search_report.json",
    ]
    for name in names:
        source = case / name
        if source.is_file():
            shutil.copy2(source, destination / name)


def _aggregate(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["mode"], row["group_size"]), []).append(row)
    aggregate = []
    for (mode, size), items in sorted(groups.items()):
        aggregate.append({
            "mode": mode,
            "group_size": size,
            "num_cases": len(items),
            "manifest_success_rate": _mean(x["manifest_success"] for x in items),
            "assembly_step_success_rate": _mean(x["assembly_step_success"] for x in items),
            "all_parts_solved_rate": _mean(x["all_parts_solved"] for x in items),
            "mean_unsolved_parts": _mean(x["unsolved_parts_count"] for x in items),
            "mean_match_precision": _mean(x["match_precision"] for x in items),
            "mean_match_recall": _mean(x["match_recall"] for x in items),
            "mean_false_positive_matches": _mean(x["false_positive_match_count"] for x in items),
            "mean_assembly_graph_f1": _mean(x["assembly_graph_f1"] for x in items),
            "mean_match_confidence_score": _mean(
                x["mean_match_confidence_score"] for x in items
            ),
            "mean_constraint_residual": _mean(x["max_constraint_residual"] for x in items),
            "mean_collision_count": _mean(x["collision_count"] for x in items),
            "mean_runtime_sec": _mean(x["runtime_sec"] for x in items),
            "mean_expanded_states": _mean(x["expanded_states"] for x in items),
            "complete_solution_rate": _mean(x["complete_solution"] for x in items),
        })
    return aggregate


def _write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root")
    parser.add_argument("--modes", nargs="+", choices=list(MODES), default=list(MODES))
    parser.add_argument("--beam-width", type=int, default=20)
    parser.add_argument("--min-score", type=float, default=0.5)
    parser.add_argument("--max-neighbors", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--max-cases-per-group", type=int,
        help="Deterministically evaluate only the first N cases in each group.",
    )
    args = parser.parse_args()
    project = Path(__file__).resolve().parent
    root = Path(args.dataset_root).resolve()
    output = Path(args.output_dir).resolve() if args.output_dir else root / "experiments"
    cases = discover_cases(root)
    if args.max_cases_per_group is not None:
        selected = []
        per_group = {}
        for case in sorted(cases, key=_case_id):
            gt_path = case / "gt.json"
            gt = json.loads(gt_path.read_text(encoding="utf-8")) if gt_path.is_file() else {}
            size = int(gt.get("group_size", 0))
            if per_group.get(size, 0) < args.max_cases_per_group:
                selected.append(case)
                per_group[size] = per_group.get(size, 0) + 1
        cases = selected
    rows = []
    for mode in args.modes:
        for case in cases:
            case_id = _case_id(case)
            mode_dir = output / mode / case_id
            mode_dir.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable, str(project / "compute_manifest.py"), str(case),
                *MODES[mode], "--write-diagnostics",
                "--beam-width", str(args.beam_width),
                "--min-score", str(args.min_score),
                "--max-neighbors", str(args.max_neighbors),
            ]
            result, runtime = _run(command, project, args.timeout, mode_dir / "compute.log")
            validation_rc = None
            build_rc = None
            if result.returncode == 0:
                matches = case / "kept_matches.json"
                validation = [
                    sys.executable, str(project / "placement_validation.py"), str(case)
                ]
                if matches.is_file() and ("pruning" in mode or "reliable" in mode):
                    validation.extend(["--matches", str(matches)])
                validation_result, _ = _run(
                    validation, project, args.timeout, mode_dir / "validation.log"
                )
                validation_rc = validation_result.returncode
                build_result, _ = _run(
                    [sys.executable, str(project / "build_assembly.py"), str(case)],
                    project, args.timeout, mode_dir / "build.log",
                )
                build_rc = build_result.returncode
            runtime_row = {"mode": mode, "runtime_sec": str(runtime)}
            metrics = _case_metrics(case, runtime_row)
            metrics["case_id"] = case_id
            metrics["assembly_step_success"] = build_rc == 0
            metrics["compute_returncode"] = result.returncode
            metrics["validation_returncode"] = validation_rc
            metrics["build_returncode"] = build_rc
            rows.append(metrics)
            _snapshot(case, mode_dir)
            print(
                f"{mode} {case_id}: compute={result.returncode} "
                f"validation={validation_rc} build={build_rc} "
                f"status={metrics['validation_status']}",
                flush=True,
            )
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "experiment_cases.csv", rows)
    _write_csv(output / "experiment_group_summary.csv", _aggregate(rows))
    (output / "experiment_manifest.json").write_text(
        json.dumps(
            {
                "modes": args.modes,
                "beam_width": args.beam_width,
                "min_score": args.min_score,
                "max_neighbors": args.max_neighbors,
                "num_cases": len(cases),
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
