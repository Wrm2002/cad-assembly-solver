"""Run one isolated pose-candidate channel for a frozen ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any

from learned_joint.pose_acceptance import contact_supported_exact_pose
from learned_joint.precision_pose_validator import validate_precision_pose


ROOT = Path(__file__).resolve().parents[1]


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run(command: list[str]) -> dict[str, Any]:
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(
            "subprocess_failed:\n" + " ".join(command)
            + "\nstdout:\n" + result.stdout[-3000:]
            + "\nstderr:\n" + result.stderr[-3000:]
        )
    return {"command": command, "stdout_tail": result.stdout[-2000:]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pair_frontier_manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--solver-python", type=Path, required=True)
    parser.add_argument("--occt-python", type=Path, required=True)
    parser.add_argument("--render-python", type=Path, required=True)
    parser.add_argument("--channel", choices=("baseline", "learned"), default="baseline")
    parser.add_argument("--max-compound-pair-candidates", type=int, default=0)
    parser.add_argument("--max-topologies", type=int, default=6)
    parser.add_argument("--max-hypotheses", type=int, default=24)
    parser.add_argument("--max-nfev", type=int, default=120)
    parser.add_argument("--exact-top-n", type=int, default=8)
    parser.add_argument("--closure-enumeration-limit", type=int, default=50000)
    parser.add_argument("--no-mesh-residuals", action="store_true")
    args = parser.parse_args()
    manifest = _read(args.pair_frontier_manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    part_sources = manifest.get("part_sources") or {}
    part_args = [item for key, value in sorted(part_sources.items()) for item in ("--part", f"{key}={value}")]
    commands = []
    baseline_budget = "4" if args.channel == "baseline" else "0"
    learned_budget = "0" if args.channel == "baseline" else "4"
    learned_weight = "0.0" if args.channel == "baseline" else "0.35"
    solved = args.output_dir / "global_pose_baseline.json"
    solver_command = [
        str(args.solver_python), "sw/run_manifold_global_pose.py", *part_args,
        "--pair-frontier-manifest", str(args.pair_frontier_manifest),
        "--output", str(solved), "--max-pair-candidates", baseline_budget,
        "--max-compound-pair-candidates", str(args.max_compound_pair_candidates),
        "--max-learned-pair-candidates", learned_budget,
        "--max-hypotheses", str(args.max_hypotheses),
        "--max-nfev", str(args.max_nfev),
        "--max-topologies", str(args.max_topologies),
        "--closure-enumeration-limit", str(args.closure_enumeration_limit),
        "--learned-pose-prior-weight", learned_weight,
    ]
    if args.no_mesh_residuals:
        solver_command.append("--no-mesh-residuals")
    commands.append(_run(solver_command))
    exact = args.output_dir / "global_pose_baseline_exact.json"
    commands.append(_run([
        str(args.occt_python), "sw/validate_manifold_pose_exact.py", "--input", str(solved),
        *part_args, "--top-n", str(args.exact_top_n), "--output", str(exact),
    ]))
    payload = _read(exact)
    hypotheses = payload.get("hypotheses") or []
    statuses = [row.get("exact_validation", {}).get("status", "not_checked") for row in hypotheses]
    contact_gates = [contact_supported_exact_pose(row) for row in hypotheses]
    precision_gates = [
        validate_precision_pose(row, contact_gate=contact_gate)
        for row, contact_gate in zip(hypotheses, contact_gates)
    ]
    selected_index = next(
        (
            index for index, gate in enumerate(precision_gates)
            if gate["precision_status"] == "valid"
        ),
        next(
            (
                index for index, (row, gate) in enumerate(
                    zip(hypotheses, precision_gates)
                )
                if row.get("exact_validation", {}).get("status") == "valid"
                and gate["precision_status"] == "review"
            ),
            0,
        ),
    )
    render = None
    if hypotheses:
        stl = args.output_dir / "rendered_stl"
        commands.append(_run([
            str(args.occt_python), "sw/export_global_pose_stl.py", "--input", str(exact),
            *part_args, "--output-dir", str(stl), "--hypothesis-index", str(selected_index),
        ]))
        render = args.output_dir / "baseline_pose_render.png"
        commands.append(_run([
            str(args.render_python), "sw/render_pose_stl.py", str(stl), str(render),
            "--title", (
                "Analytic/compound geometry (precision checked)"
                if args.channel == "baseline" else
                "Learned pose sidecar (precision checked)"
            ),
        ]))
    summary = {
        "schema_version": "frozen_pose_baseline.v1",
        "status": payload.get("status"),
        "hypothesis_count": len(hypotheses),
        "exact_status_counts": {value: statuses.count(value) for value in sorted(set(statuses))},
        "best_exact_status": statuses[0] if statuses else "none",
        "contact_supported_valid_count": sum(gate["status"] == "valid" for gate in contact_gates),
        "contact_gate_status_counts": {
            value: sum(gate["status"] == value for gate in contact_gates)
            for value in sorted({gate["status"] for gate in contact_gates})
        },
        "selected_hypothesis_index": selected_index if hypotheses else None,
        "contact_gate_audit": contact_gates,
        "precision_status_counts": {
            value: sum(
                gate["precision_status"] == value for gate in precision_gates
            )
            for value in ("valid", "review", "failed")
        },
        "precision_gate_audit": precision_gates,
        "selected_precision_status": (
            precision_gates[selected_index]["precision_status"]
            if precision_gates else None
        ),
        "render": str(render.resolve()) if render else None,
        "candidate_channel": args.channel,
        "learned_sidecar_enabled": args.channel == "learned",
        "compound_candidate_budget_per_pair": args.max_compound_pair_candidates,
        "acceptance_boundary": (
            "Only precision_status=valid is an automatic physical Pose pass; "
            "review and failed remain non-accepted."
        ),
        "commands": commands,
    }
    (args.output_dir / "exam_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
