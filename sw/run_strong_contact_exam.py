"""Run a geometry-only strong-contact pose head on an isolated known-group exam.

This runner deliberately has no case-specific logic.  It reads an existing
JoinABLe pair-frontier manifest, augments every pair with top-k B-Rep-patch
pose proposals, solves the global SE(3) graph, performs OCCT collision checks,
and renders the best checked hypothesis.  Paths and part IDs are orchestration
metadata only and never reach the learned head.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from learned_joint.pose_acceptance import contact_supported_exact_pose


ROOT = Path(__file__).resolve().parents[1]


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run(command: list[str], *, cwd: Path) -> dict[str, Any]:
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(
            "subprocess_failed:\n" + " ".join(command) + "\n" + result.stderr[-4000:]
        )
    return {"command": command, "stdout_tail": result.stdout[-2000:]}


def _pair_graphs(result_path: Path, payload: dict[str, Any]) -> tuple[Path, Path]:
    cache = result_path.parent / "cache"
    first = cache / f"{Path(payload['part_a_fixed']).stem}.brep_graph.json"
    second = cache / f"{Path(payload['part_b_moving']).stem}.brep_graph.json"
    if not first.is_file() or not second.is_file():
        raise FileNotFoundError(f"pair_graph_missing:{first}:{second}")
    return first, second


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pair_frontier_manifest", type=Path)
    parser.add_argument("head_checkpoint", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--runtime-python", type=Path, required=True)
    parser.add_argument(
        "--solver-python", type=Path,
        help="Numerically stable NumPy/SciPy runtime for the global solver; defaults to runtime-python.",
    )
    parser.add_argument("--occt-python", type=Path, required=True)
    parser.add_argument("--candidate-limit", type=int, default=20)
    parser.add_argument("--modes-per-candidate", type=int, default=6)
    parser.add_argument("--learned-pose-prior-weight", type=float, default=.35)
    args = parser.parse_args()
    solver_python = args.solver_python or args.runtime_python
    manifest = _read(args.pair_frontier_manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    commands = []
    enriched_cache: dict[tuple[str, str], Path] = {}

    def enriched_graph(step_path: Path, graph_path: Path) -> Path:
        key = (str(step_path.resolve()), str(graph_path.resolve()))
        if key in enriched_cache:
            return enriched_cache[key]
        output = args.output_dir / "enriched_graphs" / graph_path.name
        commands.append(_run([
            str(args.occt_python), "sw/enrich_brep_graph_patches.py",
            str(step_path), str(graph_path), str(output),
        ], cwd=ROOT))
        enriched_cache[key] = output
        return output

    for record in manifest.get("records") or []:
        if record.get("status") != "success":
            continue
        base = Path(record["result_path"])
        pair_payload = _read(base)
        graph_a, graph_b = _pair_graphs(base, pair_payload)
        graph_a = enriched_graph(Path(pair_payload["part_a_fixed"]), graph_a)
        graph_b = enriched_graph(Path(pair_payload["part_b_moving"]), graph_b)
        learned = base.with_name("learned_pose_frontier_strong_contact.json")
        augmented = base.with_name("joinable_e2e_learned_pose_strong_contact.json")
        commands.append(_run([
            str(args.runtime_python), "sw/run_learned_pair_pose.py", str(base), str(graph_a), str(graph_b),
            str(args.head_checkpoint), str(learned), "--candidate-limit", str(args.candidate_limit),
            "--modes-per-candidate", str(args.modes_per_candidate), "--device", "cpu",
        ], cwd=ROOT))
        commands.append(_run([
            str(args.runtime_python), "sw/augment_manifold_frontier_with_learned_pose.py", str(base), str(learned), str(augmented),
            "--per-entity-limit", "2",
        ], cwd=ROOT))
    redirected = args.output_dir / "pair_frontier_manifest_strong_contact.json"
    commands.append(_run([
        str(args.runtime_python), "sw/redirect_pair_frontier_manifest.py", str(args.pair_frontier_manifest), str(redirected),
        "--result-name", "joinable_e2e_learned_pose_strong_contact.json", "--version", "strong_contact_v1",
    ], cwd=ROOT))
    part_sources = manifest.get("part_sources") or {}
    part_args = [item for key, value in sorted(part_sources.items()) for item in ("--part", f"{key}={value}")]
    solved = args.output_dir / "global_pose_strong_contact.json"
    commands.append(_run([
        # This stage uses existing sibling STL files and SciPy optimisation;
        # run it in the stable Torch/Numpy runtime.  OCCT is reserved for the
        # exact Boolean validation/export steps below.
        str(solver_python), "sw/run_manifold_global_pose.py", *part_args,
        "--pair-frontier-manifest", str(redirected), "--output", str(solved),
        "--max-pair-candidates", "4", "--max-topologies", "6", "--max-hypotheses", "24", "--max-nfev", "120",
        "--max-learned-pair-candidates", "4",
        "--learned-pose-prior-weight", str(args.learned_pose_prior_weight),
    ], cwd=ROOT))
    exact = args.output_dir / "global_pose_strong_contact_exact.json"
    commands.append(_run([
        str(args.occt_python), "sw/validate_manifold_pose_exact.py", "--input", str(solved), *part_args,
        "--top-n", "8", "--output", str(exact),
    ], cwd=ROOT))
    result = _read(exact)
    hypotheses = result.get("hypotheses") or []
    contact_gates = [contact_supported_exact_pose(row) for row in hypotheses]
    selected_index = next(
        (index for index, gate in enumerate(contact_gates) if gate["status"] == "valid"),
        0,
    )
    render = None
    if hypotheses:
        stl = args.output_dir / "rendered_stl"
        commands.append(_run([
            str(args.occt_python), "sw/export_global_pose_stl.py", "--input", str(exact), *part_args,
            "--output-dir", str(stl), "--hypothesis-index", str(selected_index),
        ], cwd=ROOT))
        render = args.output_dir / "strong_contact_pose_render.png"
        commands.append(_run([
            str(args.runtime_python), "sw/render_pose_stl.py", str(stl), str(render),
            "--title", "Strong-contact B-Rep pose result (OCCT checked)",
        ], cwd=ROOT))
    exact_statuses = [row.get("exact_validation", {}).get("status", "not_checked") for row in hypotheses]
    summary = {
        "schema_version": "strong_contact_exam.v1",
        "status": result.get("status"),
        "pair_count": len([row for row in manifest.get("records") or [] if row.get("status") == "success"]),
        "candidate_fusion": "protected analytic baseline plus additive learned sidecar",
        "hypothesis_count": len(hypotheses),
        "exact_status_counts": {value: exact_statuses.count(value) for value in sorted(set(exact_statuses))},
        "best_exact_status": exact_statuses[0] if exact_statuses else "none",
        "contact_supported_valid_count": sum(gate["status"] == "valid" for gate in contact_gates),
        "contact_gate_status_counts": {
            value: sum(gate["status"] == value for gate in contact_gates)
            for value in sorted({gate["status"] for gate in contact_gates})
        },
        "selected_hypothesis_index": selected_index if hypotheses else None,
        "contact_gate_audit": contact_gates,
        "render": str(render.resolve()) if render else None,
        "acceptance_boundary": "An OCCT-valid pose is physical feasibility only, not an automatic semantic acceptance.",
        "commands": commands,
    }
    (args.output_dir / "exam_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
