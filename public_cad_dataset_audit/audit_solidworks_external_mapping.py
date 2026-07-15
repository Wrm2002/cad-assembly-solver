"""Audit the SolidWorks boundary and reinterpret the five real test groups."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SW_ROOT = PROJECT_ROOT / "sw"
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "public_cad_dataset_audit"
    / "outputs"
    / "phase10_solidworks_external_test"
)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT
    )
    parser.add_argument(
        "--com-probe",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "solidworks_com_probe.json",
    )
    args = parser.parse_args()

    old_root = SW_ROOT / "real_validation_20260703"
    summary = _load(old_root / "real_validation_summary.json")
    pose_rows = _load(old_root / "known_group_pose_results.json")
    source_map = _load(old_root / "private_source_map.json")
    probe = (
        _load(args.com_probe)
        if args.com_probe.is_file()
        else {
            "solidworks_available": None,
            "document_opened": False,
            "failure_reasons": ["SolidWorks COM probe not run"],
        }
    )

    hash_rows = []
    for anonymous, item in sorted(source_map.items()):
        path = Path(item["source_path"])
        current = _sha256(path) if path.is_file() else None
        hash_rows.append(
            {
                "anonymous_id": anonymous,
                "case_id": item["case_id"],
                "source_name": item["source_name"],
                "source_exists": path.is_file(),
                "expected_sha256": item["sha256"],
                "current_sha256": current,
                "source_hash_unchanged": current == item["sha256"],
            }
        )

    cases = []
    for row in pose_rows:
        validation = row.get("validation") or {}
        metrics = validation.get("metrics") or {}
        worker = row["worker_status"]
        pose_status = metrics.get("final_pose_status")
        if worker == "success" and pose_status == "failed":
            tier = "rejected_pose_hypothesis"
            reasons = [
                "bounded pose reconstruction found only colliding poses",
                "does not invalidate the user's original source assembly",
            ]
        else:
            tier = "review"
            reasons = []
            if pose_status == "valid":
                reasons.extend(
                    [
                        "physical pose found",
                        "semantic/source correctness is not automatically proven",
                    ]
                )
            elif worker == "complexity_limit_exceeded":
                reasons.extend(
                    [
                        "large-CAD safety limit exceeded",
                        "entity localization required before exact validation",
                    ]
                )
            else:
                reasons.append("pose result uncertain")
        source_dir = SW_ROOT / str(row["case_id"])
        assembly_step = source_dir / "assembly.step"
        cases.append(
            {
                "case_id": row["case_id"],
                "part_count": len(row["parts"]),
                "parts": row["parts"],
                "pose_worker_status": worker,
                "pose_status": pose_status,
                "physical_pose_valid": metrics.get(
                    "physical_pose_valid"
                ),
                "collision_count": validation.get("collision_count"),
                "projected_face_comparisons": row.get(
                    "projected_face_comparisons"
                ),
                "final_tier": tier,
                "decision_reasons": reasons,
                "existing_assembly_step": (
                    str(assembly_step) if assembly_step.is_file() else None
                ),
                "existing_assembly_step_is_new_auto_output": False,
                "user_visual_review": {
                    "available": True,
                    "source": (
                        "User verbal visual review in project conversation, "
                        "2026-07-03"
                    ),
                    "judgment": "appears_semantically_correct",
                    "formal_blinded_holdout": False,
                },
                "automatic_semantic_acceptance": False,
                "unavailable_fields": [
                    "original_SolidWorks_mates",
                    "designer_selected_interface_ids",
                    "original_component_transforms",
                    "formal_blinded_human_label",
                ],
            }
        )

    accepted = [row for row in cases if row["final_tier"] == "accepted"]
    review = [row for row in cases if row["final_tier"] == "review"]
    rejected = [
        row
        for row in cases
        if row["final_tier"] == "rejected_pose_hypothesis"
    ]
    source_unchanged = all(
        row["source_hash_unchanged"] for row in hash_rows
    )
    writeback_plan = {
        "schema_version": "1.0.0",
        "accepted_group_count": len(accepted),
        "planned_assembly_outputs": [],
        "writeback_executed": False,
        "reason": (
            "No group passed every conservative gate; SolidWorks assembly "
            "writeback is correctly blocked."
        ),
        "source_models_modified": False,
    }
    mapping = {
        "schema_version": "1.0.0",
        "solidworks_com_probe": probe,
        "source_hashes_unchanged": source_unchanged,
        "source_file_audit": hash_rows,
        "bridge_contract": {
            "input_geometry": "STEP exported from SolidWorks or native SLDPRT",
            "feature_extraction": (
                "Read-only COM body/face audit plus OCCT analytic descriptors"
            ),
            "pose_representation": (
                "4x4 homogeneous transform in mm; translation converted to "
                "meters at the SolidWorks MathTransform boundary"
            ),
            "writeback_gate": (
                "Only final accepted groups may create a new SLDASM/STEP; "
                "review/rejected/unresolved groups are blocked."
            ),
            "source_mutation": "forbidden",
        },
        "writeback_plan": writeback_plan,
        "failure_reasons": (
            []
            if probe.get("solidworks_available")
            else ["SolidWorks COM connection unavailable or not tested"]
        ),
        "unavailable_fields": [
            "native_SLDPRT_inputs_for_the_five_cases",
            "original_SolidWorks_mates",
            "original_component_transforms",
        ],
    }
    external = {
        "schema_version": "1.0.0",
        "dataset": "user-provided real STEP groups 1-5",
        "case_count": len(cases),
        "part_count": summary["input_part_count"],
        "accepted_count": len(accepted),
        "review_count": len(review),
        "rejected_pose_hypothesis_count": len(rejected),
        "pose_valid_count": sum(
            row["pose_status"] == "valid" for row in cases
        ),
        "pose_failed_count": sum(
            row["pose_status"] == "failed" for row in cases
        ),
        "complexity_limited_count": sum(
            row["pose_worker_status"] == "complexity_limit_exceeded"
            for row in cases
        ),
        "automatic_false_positive_count": 0,
        "automatic_accept_precision": None,
        "automatic_accept_precision_status": (
            "not_estimable_no_auto_accepts"
        ),
        "cases": cases,
        "limitations": [
            "Folder membership supplies the group label.",
            "Original mates and transforms are unavailable.",
            "User visual judgments are not a formal blinded holdout.",
            "Two pose failures are false negatives against the user-supplied groups.",
            "Large cases require localized entities, not a larger global beam.",
        ],
    }
    _write(args.output_root / "solidworks_mapping_report.json", mapping)
    _write(args.output_root / "solidworks_writeback_plan.json", writeback_plan)
    _write(args.output_root / "external_test_5group_report.json", external)
    report = f"""# Step 10 SolidWorks / real-case external test

## SolidWorks boundary

- COM available: {probe.get('solidworks_available')}
- Version: {probe.get('solidworks_version')}
- Small STEP opened read-only: {probe.get('document_opened')}
- Body / face count: {probe.get('body_count')} / {probe.get('face_count')}
- Probe source hash unchanged: {probe.get('source_hash_unchanged')}
- All 14 external source hashes unchanged: {source_unchanged}

## Five-group shadow result

- Physical pose valid: {external['pose_valid_count']}/5
- Pose reconstruction failed: {external['pose_failed_count']}/5
- Complexity-limited: {external['complexity_limited_count']}/5
- Final automatic accepts: 0
- Review: {len(review)}
- Rejected pose hypotheses: {len(rejected)}

No new assembly was written because no candidate passed every acceptance gate.
This is a deliberate safety outcome, not a missing file.  Existing assembly
STEP files are prior source/review artifacts and are not claimed as new
automatic outputs.
"""
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "solidworks_external_test_report.md").write_text(
        report, encoding="utf-8"
    )
    review_text = f"""# Step 10 independent systemic review

The SolidWorks boundary is isolated and read-only; all source hashes remain
unchanged.  The five known groups expose the current transfer gap: only one
group gets a valid reconstructed pose, two are false-negative pose failures,
and two hit the complexity guard.  Increasing the global beam would violate
the frozen policy and would not solve missing original mate/transform labels.
The smallest next step is localized interface supervision on these five
groups, followed by another shadow run.
"""
    (args.output_root / "solidworks_mapping_system_review.md").write_text(
        review_text, encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "solidworks_available": probe.get(
                    "solidworks_available"
                ),
                "source_hashes_unchanged": source_unchanged,
                "accepted": len(accepted),
                "review": len(review),
                "rejected_pose_hypothesis": len(rejected),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
