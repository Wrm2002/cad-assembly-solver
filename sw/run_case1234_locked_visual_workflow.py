"""Pinned Case1--4 visual-pose workflow and regression gate.

This is the only supported regression entry point for the four human-confirmed
cases.  Golden poses are used solely to detect regression after solving; they
are never injected into candidate generation, scoring, or a new production
case.  A missing/abstaining visual stage is a hard workflow failure rather than
a silent geometry-only fallback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_CONFIG = HERE / "configs" / "case1234_visual_pose_workflow.v1.json"


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(value: str | None) -> Path | None:
    return (ROOT / value).resolve() if value else None


def _canonical_pose(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for component in manifest.get("components") or []:
        rows.append({
            "part": str(component.get("label") or Path(component["source"]).stem),
            "placement": component.get("placement") or {},
        })
    return sorted(rows, key=lambda row: row["part"])


def _numbers_close(left: Any, right: Any, tolerance: float = 1e-6) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right)) <= tolerance
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            _numbers_close(left[key], right[key], tolerance) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _numbers_close(a, b, tolerance) for a, b in zip(left, right)
        )
    return left == right


def validate_visual_decision(decision: dict[str, Any]) -> list[str]:
    errors = []
    if decision.get("status") not in {"review", "accepted"}:
        errors.append("final status is not review/accepted")
    if not decision.get("visual_semantics_used_for_ranking"):
        errors.append("visual semantics were not used for ranking")
    if decision.get("geometry_scores_exposed_to_visual_model") is not False:
        errors.append("geometry scores leaked into visual model input")
    if decision.get("semantic_auto_accept_enabled") is not False:
        errors.append("semantic auto-accept must remain disabled")
    if not decision.get("selected_manifest"):
        errors.append("no selected review manifest")
    return errors


def verify_lock(config: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    cases: dict[str, Any] = {}
    required_chain = [
        "protected_origin_diverse_pose_frontier",
        "qwen_multiview_role_region_direction_action_review",
        "semantic_guided_topk_without_geometry_score_leakage",
        "bounded_occt_exact_collision_gate",
    ]
    for stage in required_chain:
        if stage not in config.get("solver_chain", []):
            errors.append(f"missing locked stage: {stage}")

    implementation_tokens = {
        HERE / "known_group_assembly.py": [
            "--export-pose-candidates",
            "--defer-exact-collision",
            "_export_pose_candidate_frontier",
        ],
        HERE / "visual_pose_candidate_reranker.py": [
            "geometry_scores_exposed_to_visual_model",
            "semantic_auto_accept_enabled",
            "classify_exact_collision_audit",
        ],
    }
    for path, tokens in implementation_tokens.items():
        text = path.read_text(encoding="utf-8")
        for token in tokens:
            if token not in text:
                errors.append(f"implementation invariant missing: {path.name}:{token}")

    for case_id, row in config.get("cases", {}).items():
        case_errors = []
        manifest_path = _resolve(row["golden_manifest"])
        render_path = _resolve(row["golden_render"])
        for label, path, expected in (
            ("manifest", manifest_path, row["golden_manifest_sha256"]),
            ("render", render_path, row["golden_render_sha256"]),
        ):
            if path is None or not path.is_file():
                case_errors.append(f"missing golden {label}: {path}")
            elif _sha256(path) != expected:
                case_errors.append(f"golden {label} hash changed")

        decision_path = _resolve(row.get("current_route_receipt"))
        visual_path = _resolve(row.get("current_visual_receipt"))
        if decision_path:
            if not decision_path.is_file():
                case_errors.append("missing current visual route decision")
            else:
                case_errors.extend(validate_visual_decision(_read(decision_path)))
        if visual_path:
            if not visual_path.is_file():
                case_errors.append("missing visual API receipt")
            else:
                visual = _read(visual_path)
                if visual.get("status") != "ok":
                    case_errors.append("visual API receipt is not status=ok")
                if not visual.get("model"):
                    case_errors.append("visual API model is missing")
        cases[case_id] = {
            "status": "pass" if not case_errors else "fail",
            "errors": case_errors,
            "human_confirmation": row.get("human_confirmation"),
            "golden_manifest": str(manifest_path),
            "golden_render": str(render_path),
        }
        errors.extend(f"case{case_id}: {error}" for error in case_errors)
    return {
        "schema_version": "case1234_workflow_lock_audit.v1",
        "workflow_id": config.get("workflow_id"),
        "status": "pass" if not errors else "fail",
        "geometry_only_fallback_allowed": False,
        "cases": cases,
        "errors": errors,
    }


def _run_checked(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def run_case(
    case_id: str,
    row: dict[str, Any],
    defaults: dict[str, Any],
    output_root: Path,
    *,
    visual_mode: str,
) -> dict[str, Any]:
    case_output = output_root / f"case{case_id}"
    geometry_output = case_output / "geometry"
    visual_output = case_output / "visual"
    geometry_command = [
        sys.executable,
        str(HERE / "known_group_assembly.py"),
        str(_resolve(row["case_dir"])),
        "--output-dir",
        str(geometry_output),
        "--beam-width",
        str(defaults["beam_width"]),
        "--export-pose-candidates",
        str(defaults["export_pose_candidates"]),
        "--defer-exact-collision",
        "--skip-assembly-step",
    ]
    if row.get("joinable_pose_dir"):
        geometry_command.extend(
            ["--joinable-pose-dir", str(_resolve(row["joinable_pose_dir"]))]
        )
    if row.get("brep_graph_dir"):
        geometry_command.extend(
            ["--brep-graph-dir", str(_resolve(row["brep_graph_dir"]))]
        )
    _run_checked([
        sys.executable,
        str(HERE / "run_safe_occt_worker.py"),
        "--output-root",
        str(case_output),
        "--stage",
        "geometry_candidate_generation",
        "--timeout",
        "1800",
        "--cpu-affinity",
        "0-7",
        "--max-worker-memory-gb",
        "8",
        "--min-free-memory-gb",
        "8",
        "--cooldown-seconds",
        "3",
        "--",
        *geometry_command,
    ])
    _run_checked([
        sys.executable,
        str(HERE / "visual_pose_candidate_reranker.py"),
        "--geometry-output",
        str(geometry_output),
        "--output-dir",
        str(visual_output),
        "--mode",
        visual_mode,
        "--exact-top-n",
        str(defaults["exact_top_n"]),
        "--view-width",
        str(defaults["view_width"]),
        "--view-height",
        str(defaults["view_height"]),
    ])
    decision = _read(visual_output / "final_visual_pose_decision.json")
    errors = validate_visual_decision(decision)
    selected_path = Path(decision.get("selected_manifest") or "")
    golden_path = _resolve(row["golden_manifest"])
    if selected_path.is_file() and golden_path and golden_path.is_file():
        if not _numbers_close(
            _canonical_pose(_read(selected_path)), _canonical_pose(_read(golden_path))
        ):
            errors.append("selected pose differs from human-confirmed golden pose")
    else:
        errors.append("selected or golden manifest missing")
    return {
        "case_id": case_id,
        "status": "pass" if not errors else "regression_failed",
        "errors": errors,
        "decision": str((visual_output / "final_visual_pose_decision.json").resolve()),
        "selected_manifest": str(selected_path.resolve()) if selected_path.is_file() else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--verify-lock", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--cases", default="1,2,3,4")
    parser.add_argument("--visual-mode", choices=("live", "cache_only"), default="live")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    if args.verify_lock == args.run:
        parser.error("choose exactly one of --verify-lock or --run")
    config = _read(args.config.resolve())
    base = ROOT / "sw" / "generalization_work" / "case1234_visual_pose_locked_v1"
    if args.verify_lock:
        audit = verify_lock(config)
        audit_path = base / "workflow_lock_audit.json"
        _write(audit_path, audit)
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        return 0 if audit["status"] == "pass" else 2

    requested = [value.strip() for value in args.cases.split(",") if value.strip()]
    unknown = [value for value in requested if value not in config.get("cases", {})]
    if unknown:
        parser.error(f"unknown locked cases: {unknown}")
    lock_audit = verify_lock(config)
    if lock_audit["status"] != "pass":
        raise RuntimeError("workflow lock verification failed; refusing to run")
    output_root = (
        args.output_root.resolve()
        if args.output_root
        else base / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    results = [
        run_case(
            case_id,
            config["cases"][case_id],
            config["execution_defaults"],
            output_root,
            visual_mode=args.visual_mode,
        )
        for case_id in requested
    ]
    audit = {
        "schema_version": "case1234_locked_run.v1",
        "workflow_id": config["workflow_id"],
        "status": "pass" if all(row["status"] == "pass" for row in results) else "fail",
        "visual_stage_required": True,
        "geometry_only_fallback_allowed": False,
        "results": results,
    }
    _write(output_root / "locked_run_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if audit["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
