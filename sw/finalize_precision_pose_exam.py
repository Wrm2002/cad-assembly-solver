"""Apply the precision gate to an exact pose frontier and render its best tier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any

from learned_joint.pose_acceptance import contact_supported_exact_pose
from learned_joint.precision_pose_validator import validate_precision_pose


ROOT = Path(__file__).resolve().parents[1]


def _part(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--part requires PART_ID=STEP_PATH")
    part_id, raw = value.split("=", 1)
    path = Path(raw).resolve()
    if not part_id or not path.is_file():
        raise argparse.ArgumentTypeError("--part requires an existing STEP path")
    return part_id, path


def _run(command: list[str]) -> dict[str, Any]:
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(
            "subprocess_failed:\n" + " ".join(command)
            + "\n" + result.stdout[-2000:] + "\n" + result.stderr[-4000:]
        )
    return {"command": command, "stdout_tail": result.stdout[-2000:]}


def select_precision_hypothesis(
    hypotheses: list[dict[str, Any]],
    precision_rows: list[dict[str, Any]],
) -> int | None:
    if not hypotheses:
        return None

    def key(index: int) -> tuple[Any, ...]:
        hypothesis, precision = hypotheses[index], precision_rows[index]
        status = precision["precision_status"]
        tier = {"valid": 0, "review": 1, "failed": 2}.get(status, 3)
        exact_valid = (
            hypothesis.get("exact_validation", {}).get("status") == "valid"
        )
        prismatic_count = sum(
            "prismatic" in str(row.get("manifold_type", "")).lower()
            for row in hypothesis.get("factor_residuals") or []
        )
        unresolved = len(hypothesis.get("unresolved_manifold_dofs") or [])
        axis_distance = precision.get("axis_distance_mm")
        axis_distance = float("inf") if axis_distance is None else axis_distance
        cost = float((hypothesis.get("optimizer") or {}).get("cost", float("inf")))
        return (
            tier,
            not exact_valid,
            -prismatic_count if status == "review" else 0,
            unresolved,
            axis_distance,
            cost,
            index,
        )

    return min(range(len(hypotheses)), key=key)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exact_input", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--part", action="append", type=_part, required=True)
    parser.add_argument("--occt-python", type=Path, required=True)
    parser.add_argument("--render-python", type=Path, required=True)
    parser.add_argument("--title", default="Precision-gated CAD pose result")
    args = parser.parse_args()
    payload = json.loads(args.exact_input.read_text(encoding="utf-8"))
    hypotheses = payload.get("hypotheses") or []
    contact_rows = [contact_supported_exact_pose(row) for row in hypotheses]
    precision_rows = [
        validate_precision_pose(row, contact_gate=contact)
        for row, contact in zip(hypotheses, contact_rows)
    ]
    selected = select_precision_hypothesis(hypotheses, precision_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    commands = []
    render = None
    if selected is not None:
        part_args = [
            item for part, path in args.part
            for item in ("--part", f"{part}={path}")
        ]
        stl = args.output_dir / "rendered_stl"
        commands.append(_run([
            str(args.occt_python), "sw/export_global_pose_stl.py",
            "--input", str(args.exact_input), *part_args,
            "--output-dir", str(stl),
            "--hypothesis-index", str(selected),
        ]))
        render = args.output_dir / "precision_pose_render.png"
        commands.append(_run([
            str(args.render_python), "sw/render_pose_stl.py",
            str(stl), str(render), "--title", args.title,
        ]))
    selected_precision = precision_rows[selected] if selected is not None else None
    final_status = (
        selected_precision["precision_status"]
        if selected_precision is not None else "failed"
    )
    summary = {
        "schema_version": "precision_pose_exam.v1",
        "input": str(args.exact_input.resolve()),
        "hypothesis_count": len(hypotheses),
        "exact_status_counts": {
            status: sum(
                row.get("exact_validation", {}).get("status") == status
                for row in hypotheses
            )
            for status in ("valid", "failed", "uncertain", "not_checked")
        },
        "precision_status_counts": {
            status: sum(row["precision_status"] == status for row in precision_rows)
            for status in ("valid", "review", "failed")
        },
        "selected_hypothesis_index": selected,
        "selected_precision_validation": selected_precision,
        "selected_contact_gate": contact_rows[selected] if selected is not None else None,
        "final_pose_status": final_status,
        "accepted": final_status == "valid",
        "review_required": final_status == "review",
        "render": str(render.resolve()) if render else None,
        "precision_gate_audit": precision_rows,
        "contact_gate_audit": contact_rows,
        "commands": commands,
        "acceptance_boundary": (
            "OCCT validity and pose feasibility are necessary. Automatic "
            "acceptance additionally requires a valid multi-evidence precision gate."
        ),
    }
    (args.output_dir / "exam_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        key: summary[key] for key in (
            "hypothesis_count", "exact_status_counts", "precision_status_counts",
            "selected_hypothesis_index", "final_pose_status", "render",
        )
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
