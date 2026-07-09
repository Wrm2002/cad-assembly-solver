"""Aggregate assembly metrics by group size and solver mode."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any] | list[Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    except Exception:
        return None


def _mate_key(item):
    parts = item.get("parts")
    if not parts:
        parts = [item.get("part_a"), item.get("part_b")]
    return tuple(sorted(str(part) for part in parts)), item.get("type")


def _case_metrics(case: Path, runtime_row: dict[str, str] | None):
    inputs = [
        path for path in case.iterdir()
        if path.is_file() and path.suffix.lower() in {".step", ".stp"}
        and not path.name.lower().startswith("assembly")
    ]
    validation = _read_json(case / "assembly_validation.json") or {}
    diagnostics = _read_json(case / "assembly_diagnostics.json") or {}
    search = _read_json(case / "search_report.json") or {}
    manifest = _read_json(case / "assembly_manifest.json") or {}
    gt = _read_json(case / "gt.json") or {}
    mode = runtime_row.get("mode", "unknown") if runtime_row else "unknown"
    if mode != "proposed_reliable":
        # search_report.json may be left by a previous reliable-solver run in
        # the same case directory. Baseline metrics must not consume it.
        search = {}
    if mode in {"baseline_1_bfs", "baseline_2_bfs_scoring"}:
        # These modes deliberately use the complete raw graph. Do not let
        # kept_matches.json left by a later ablation contaminate reruns.
        matches = diagnostics.get("matches", {}).get("items", [])
    else:
        matches = _read_json(case / "kept_matches.json")
        if matches is None:
            matches = _read_json(case / "scored_matches.json")
        if matches is None:
            matches = diagnostics.get("matches", {}).get("items", [])

    true_mates = {_mate_key(item) for item in gt.get("true_mates", [])}
    predicted = {_mate_key(item) for item in matches}
    if true_mates:
        true_positive = len(true_mates & predicted)
        precision = true_positive / len(predicted) if predicted else 0.0
        recall = true_positive / len(true_mates)
        false_positive = len(predicted - true_mates)
        graph_f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )
    else:
        precision = recall = false_positive = graph_f1 = None
    scored_values = [
        float(item["score"]) for item in matches if item.get("score") is not None
    ]

    components = manifest.get("components", [])
    identity_count = sum(
        component.get("placement", {}).get("translate", [0, 0, 0]) == [0, 0, 0]
        and not component.get("placement", {}).get("rotate_sequence")
        for component in components
    )
    unsolved = validation.get(
        "num_unsolved_parts",
        len(search.get("unsolved_parts", diagnostics.get("placement", {}).get("unsolved_parts", []))),
    )
    return {
        "case_id": case.name,
        "group_size": int(gt.get("group_size", len(inputs))),
        "mode": mode,
        "manifest_success": bool(manifest),
        "assembly_step_success": (case / "assembly.step").is_file(),
        "all_parts_solved": unsolved == 0 and bool(manifest),
        "unsolved_parts_count": unsolved,
        "identity_placement_count": identity_count,
        "match_precision": precision,
        "match_recall": recall,
        "false_positive_match_count": false_positive,
        "assembly_graph_f1": graph_f1,
        "mean_match_confidence_score": (
            sum(scored_values) / len(scored_values) if scored_values else None
        ),
        "connected_component_count": validation.get(
            "graph", diagnostics.get("graph", {})
        ).get("connected_component_count"),
        "max_constraint_residual": validation.get("max_constraint_residual"),
        "collision_count": validation.get("collision_count"),
        "severe_penetration_count": validation.get("severe_penetration_count"),
        "runtime_sec": float(runtime_row["runtime_sec"]) if runtime_row and runtime_row.get("runtime_sec") else None,
        "expanded_states": search.get("expanded_states"),
        "complete_solution": search.get("status") == "success" if search else None,
        "search_score": search.get("total_score"),
        "validation_status": validation.get("status"),
    }


def _mean(values):
    values = [float(value) for value in values if value not in (None, "")]
    return sum(values) / len(values) if values else None


def evaluate(dataset_root: Path, summary_path: Path | None):
    runtime = {}
    if summary_path and summary_path.is_file():
        with summary_path.open(encoding="utf-8-sig", newline="") as handle:
            runtime = {row["case_id"]: row for row in csv.DictReader(handle)}
    direct = [path for path in dataset_root.iterdir() if path.is_dir() and any(
            child.suffix.lower() in {".step", ".stp"} and not child.name.lower().startswith("assembly")
            for child in path.iterdir() if child.is_file()
        )]
    nested = [
        path / "step" for path in dataset_root.iterdir()
        if path.is_dir() and (path / "step").is_dir() and any(
            child.suffix.lower() in {".step", ".stp"} and not child.name.lower().startswith("assembly")
            for child in (path / "step").iterdir() if child.is_file()
        )
    ]
    cases = sorted(
        direct + nested,
        key=lambda path: path.name,
    )
    rows = [_case_metrics(case, runtime.get(case.name)) for case in cases]
    groups = defaultdict(list)
    for row in rows:
        groups[(row["mode"], row["group_size"])].append(row)
    aggregate = []
    for (mode, size), items in sorted(groups.items()):
        aggregate.append({
            "mode": mode,
            "group_size": size,
            "num_cases": len(items),
            "manifest_success_rate": _mean(row["manifest_success"] for row in items),
            "assembly_step_success_rate": _mean(row["assembly_step_success"] for row in items),
            "all_parts_solved_rate": _mean(row["all_parts_solved"] for row in items),
            "mean_unsolved_parts": _mean(row["unsolved_parts_count"] for row in items),
            "mean_match_precision": _mean(row["match_precision"] for row in items),
            "mean_match_recall": _mean(row["match_recall"] for row in items),
            "mean_false_positive_matches": _mean(row["false_positive_match_count"] for row in items),
            "mean_assembly_graph_f1": _mean(row["assembly_graph_f1"] for row in items),
            "mean_match_confidence_score": _mean(
                row["mean_match_confidence_score"] for row in items
            ),
            "mean_constraint_residual": _mean(row["max_constraint_residual"] for row in items),
            "mean_collision_count": _mean(row["collision_count"] for row in items),
            "mean_runtime_sec": _mean(row["runtime_sec"] for row in items),
            "mean_expanded_states": _mean(row["expanded_states"] for row in items),
            "complete_solution_rate": _mean(row["complete_solution"] for row in items),
        })
    return rows, aggregate


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
    parser.add_argument("--summary")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    root = Path(args.dataset_root).resolve()
    output = Path(args.output_dir).resolve() if args.output_dir else root / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    rows, groups = evaluate(root, Path(args.summary).resolve() if args.summary else None)
    _write_csv(output / "case_metrics.csv", rows)
    _write_csv(output / "group_summary.csv", groups)
    print(f"cases={len(rows)} groups={len(groups)}")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
