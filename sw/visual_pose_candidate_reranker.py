"""Visual-semantic reranking for an exported geometric pose frontier.

This is the generic bridge between the two branches of the assembly route:

* geometry exports proper SE(3) candidates with machine evidence;
* a vision model compares standardized multiviews without seeing geometry
  scores or a stored final pose;
* semantic results reorder a protected union but never auto-accept it;
* bounded OCCT collision checks run after semantic scheduling.

The output is conservative: a vision-selected, collision-free candidate is
still ``review`` until precision and insertion-path gates are available.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from multimodal_reviewer import QwenVLReviewer


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PartRole(_Strict):
    part_id: str
    role: str
    visible_functional_interfaces: list[str]
    likely_assembly_actions: list[str]
    evidence: list[str]


class PoseAssessment(_Strict):
    candidate_id: str
    semantic_score: float = Field(ge=0.0, le=1.0)
    role_consistency: str
    receiving_region_consistency: str
    orientation_consistency: str
    assembly_action_consistency: str
    containment_or_centering_consistency: str
    service_or_functional_face_visibility: str
    semantic_forbidden: bool
    reason_codes: list[str]
    explanation: str


class PoseBatchReview(_Strict):
    assembly_family: str
    expected_assembly_action: str
    part_roles: list[PartRole]
    candidate_assessments: list[PoseAssessment]
    preferred_candidate_ids: list[str] = Field(max_length=3)
    ambiguity: str
    confidence: float = Field(ge=0.0, le=1.0)
    review_required: bool


POSE_REVIEW_SYSTEM_PROMPT = r"""
You are a conservative visual reviewer of mechanical CAD assembly POSE
candidates. Multiple images are supplied; every image title contains a unique
candidate ID. Compare the candidates against each other.

First infer each part's functional role from visible geometry: carrier,
housing, flange, shaft, key, fan, cage, cover, insert, connector, etc. Then
infer the likely receiving region, assembly direction/action, required
centering or containment, and which faces must remain exposed.

Do not estimate a rotation matrix or translation. Do not reward a candidate
because it looks close in one view. Use all views. Reject obvious floating,
buried, crossed, off-centre, wrong-side, wrong-direction, or unseated poses.
For an axial multi-part assembly, judge whether the shaft/key subgroup is
coaxial and centred through the receiving supports. For an insert/cage
assembly, judge whether the insert is actually inside and seated in the
matching bay rather than hovering outside it. Preserve genuinely equivalent
poses and say when images cannot resolve them.

The prompt intentionally contains no geometry score. Your semantic_score must
come only from visible functional evidence. ``semantic_forbidden=true`` means
the pose visibly contradicts the inferred function; it does not by itself
authorize automatic rejection or acceptance.

Return ONLY strict JSON with exactly these root keys:
assembly_family, expected_assembly_action, part_roles,
candidate_assessments, preferred_candidate_ids, ambiguity, confidence,
review_required.

Each candidate_assessments item must contain exactly:
candidate_id, semantic_score, role_consistency,
receiving_region_consistency, orientation_consistency,
assembly_action_consistency, containment_or_centering_consistency,
service_or_functional_face_visibility, semantic_forbidden, reason_codes,
explanation.
Do not invent candidate IDs or part IDs.
""".strip()


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _fallback(candidate_ids: list[str], part_ids: list[str]) -> dict[str, Any]:
    return {
        "assembly_family": "unknown",
        "expected_assembly_action": "unknown",
        "part_roles": [
            {
                "part_id": part,
                "role": "unknown",
                "visible_functional_interfaces": [],
                "likely_assembly_actions": ["unknown"],
                "evidence": ["visual_review_unavailable"],
            }
            for part in part_ids
        ],
        "candidate_assessments": [
            {
                "candidate_id": candidate_id,
                "semantic_score": 0.5,
                "role_consistency": "unknown",
                "receiving_region_consistency": "unknown",
                "orientation_consistency": "unknown",
                "assembly_action_consistency": "unknown",
                "containment_or_centering_consistency": "unknown",
                "service_or_functional_face_visibility": "unknown",
                "semantic_forbidden": False,
                "reason_codes": ["visual_review_unavailable"],
                "explanation": "Visual review abstained.",
            }
            for candidate_id in candidate_ids
        ],
        "preferred_candidate_ids": [],
        "ambiguity": "visual_review_unavailable",
        "confidence": 0.0,
        "review_required": True,
    }


def _validator(candidate_ids: set[str], part_ids: set[str]):
    def validate(value: Any) -> dict[str, Any]:
        raw = value if isinstance(value, dict) else {}
        raw_roles = raw.get("part_roles") or []
        if isinstance(raw_roles, dict):
            raw_roles = [
                {"part_id": part_id, **(row if isinstance(row, dict) else {"role": row})}
                for part_id, row in raw_roles.items()
            ]
        roles = []
        for row in raw_roles if isinstance(raw_roles, list) else []:
            if not isinstance(row, dict):
                continue
            part_id = str(row.get("part_id") or row.get("id") or "")
            if part_id not in part_ids:
                continue
            roles.append({
                "part_id": part_id,
                "role": str(row.get("role") or row.get("part_role") or "unknown"),
                "visible_functional_interfaces": [
                    str(item) for item in (
                        row.get("visible_functional_interfaces")
                        or row.get("functional_interfaces")
                        or []
                    )
                ],
                "likely_assembly_actions": [
                    str(item) for item in (
                        row.get("likely_assembly_actions")
                        or row.get("assembly_actions")
                        or ["unknown"]
                    )
                ],
                "evidence": [str(item) for item in (row.get("evidence") or [])],
            })

        raw_assessments = raw.get("candidate_assessments") or []
        if isinstance(raw_assessments, dict):
            raw_assessments = [
                {
                    "candidate_id": candidate_id,
                    **(row if isinstance(row, dict) else {"explanation": row}),
                }
                for candidate_id, row in raw_assessments.items()
            ]
        assessments = []
        for row in raw_assessments if isinstance(raw_assessments, list) else []:
            if not isinstance(row, dict):
                continue
            candidate_id = str(row.get("candidate_id") or row.get("id") or "")
            if candidate_id not in candidate_ids:
                continue
            score = row.get(
                "semantic_score",
                row.get("plausibility_score", row.get("score", 0.5)),
            )
            try:
                semantic_score = max(0.0, min(1.0, float(score)))
            except (TypeError, ValueError):
                semantic_score = 0.5
            assessments.append({
                "candidate_id": candidate_id,
                "semantic_score": semantic_score,
                "role_consistency": str(row.get("role_consistency", "unknown")),
                "receiving_region_consistency": str(
                    row.get("receiving_region_consistency", "unknown")
                ),
                "orientation_consistency": str(
                    row.get("orientation_consistency", "unknown")
                ),
                "assembly_action_consistency": str(
                    row.get("assembly_action_consistency", "unknown")
                ),
                "containment_or_centering_consistency": str(
                    row.get(
                        "containment_or_centering_consistency",
                        row.get("centering_consistency", "unknown"),
                    )
                ),
                "service_or_functional_face_visibility": str(
                    row.get("service_or_functional_face_visibility", "unknown")
                ),
                "semantic_forbidden": bool(
                    row.get("semantic_forbidden", row.get("forbidden", False))
                ),
                "reason_codes": [
                    str(item) for item in (row.get("reason_codes") or [])
                ],
                "explanation": str(
                    row.get("explanation") or row.get("reason") or ""
                ),
            })
        preferred = raw.get("preferred_candidate_ids") or []
        if isinstance(preferred, str):
            preferred = [preferred]
        normalized = {
            "assembly_family": str(raw.get("assembly_family", "unknown")),
            "expected_assembly_action": str(
                raw.get("expected_assembly_action", "unknown")
            ),
            "part_roles": roles,
            "candidate_assessments": assessments,
            "preferred_candidate_ids": [
                str(candidate_id) for candidate_id in preferred
                if str(candidate_id) in candidate_ids
            ][:3],
            "ambiguity": str(raw.get("ambiguity", "unknown")),
            "confidence": max(0.0, min(1.0, float(raw.get("confidence", 0.0) or 0.0))),
            "review_required": bool(raw.get("review_required", True)),
        }
        parsed = PoseBatchReview.model_validate(normalized)
        result = parsed.model_dump(mode="json")
        seen = set()
        filtered = []
        for row in result["candidate_assessments"]:
            candidate_id = row["candidate_id"]
            if candidate_id not in candidate_ids or candidate_id in seen:
                continue
            seen.add(candidate_id)
            filtered.append(row)
        result["candidate_assessments"] = filtered
        result["preferred_candidate_ids"] = [
            candidate_id
            for candidate_id in result["preferred_candidate_ids"]
            if candidate_id in candidate_ids
        ][:3]
        result["part_roles"] = [
            row for row in result["part_roles"] if row["part_id"] in part_ids
        ]
        return result

    return validate


def rank_candidates(
    frontier_rows: list[dict[str, Any]],
    visual_output: dict[str, Any],
) -> list[dict[str, Any]]:
    """Rank without permitting semantic evidence to bypass physical gates."""

    visual_by_id = {
        row["candidate_id"]: row
        for row in visual_output.get("candidate_assessments") or []
        if row.get("candidate_id")
    }
    preferred = {
        candidate_id: index
        for index, candidate_id in enumerate(
            visual_output.get("preferred_candidate_ids") or []
        )
    }
    rows = []
    for frontier in frontier_rows:
        candidate_id = str(frontier["candidate_id"])
        visual = visual_by_id.get(candidate_id) or {
            "candidate_id": candidate_id,
            "semantic_score": 0.0,
            "semantic_forbidden": False,
            "reason_codes": ["visual_candidate_missing"],
            "explanation": "No visual assessment was returned.",
        }
        machine = dict(frontier.get("machine_evidence") or {})
        collision_status = str(machine.get("collision_status") or "")
        collision_count = int(machine.get("collision_count", 0) or 0)
        known_collision = collision_status == "success" and collision_count > 0
        semantic_forbidden = bool(visual.get("semantic_forbidden", False))
        raw_semantic_score = visual.get("semantic_score", 0.5)
        semantic_score = float(
            0.5 if raw_semantic_score is None else raw_semantic_score
        )
        closure_ratio = float(machine.get("closure_ratio", 0.0) or 0.0)
        # Sort keys are explicit gates, not one opaque final score.
        row = {
            **frontier,
            "visual_assessment": visual,
            "semantic_score": semantic_score,
            "semantic_forbidden": semantic_forbidden,
            "preferred_rank": preferred.get(candidate_id),
            "known_collision": known_collision,
            "visual_ranking_only": True,
            "can_auto_accept": False,
            "review_required": True,
        }
        row["ranking_key_audit"] = {
            "known_collision_rank": int(known_collision),
            "semantic_forbidden_rank": int(semantic_forbidden),
            "preferred_rank": preferred.get(candidate_id, 999),
            "negative_semantic_score": -semantic_score,
            "negative_closure_ratio": -closure_ratio,
            "source_pose_rank": int(frontier.get("source_pose_rank", 10**9)),
        }
        rows.append(row)
    rows.sort(
        key=lambda row: (
            row["ranking_key_audit"]["known_collision_rank"],
            row["ranking_key_audit"]["semantic_forbidden_rank"],
            row["ranking_key_audit"]["preferred_rank"],
            row["ranking_key_audit"]["negative_semantic_score"],
            row["ranking_key_audit"]["negative_closure_ratio"],
            row["ranking_key_audit"]["source_pose_rank"],
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["visual_reranked_rank"] = rank
    return rows


def classify_exact_collision_audit(
    audit: dict[str, Any],
    *,
    uncertain_collision_ratio: float = 0.002,
) -> dict[str, Any]:
    """Classify exact collision evidence without overstating incomplete CAD.

    Only a complete, empty Boolean audit is collision-free.  A positive
    intersection on complete topology is a failure.  A very small fitted
    intersection on mixed solid/open-shell STEP data remains uncertain and
    may be surfaced for review, but is never upgraded to collision-free or
    Accepted.
    """

    collisions = list(audit.get("collisions") or [])
    complete = bool((audit.get("coverage_audit") or {}).get("complete", False))
    collision_free = (
        audit.get("status") == "success"
        and complete
        and not collisions
        and audit.get("collision_result") in {None, "no_collision_detected"}
    )
    ratios = [
        float(row.get("minimum_part_volume_ratio", float("inf")))
        for row in collisions
        if isinstance(row, dict)
    ]
    maximum_ratio = max(ratios, default=0.0)
    collision_uncertain = (
        not collision_free
        and not complete
        and bool(collisions)
        and maximum_ratio <= max(0.0, float(uncertain_collision_ratio))
    )
    if collision_free:
        outcome = "collision_free"
    elif collision_uncertain:
        outcome = "collision_uncertain_small_overlap_partial_topology"
    elif collisions:
        outcome = "collision_failed"
    else:
        outcome = "collision_uncertain_incomplete_topology"
    return {
        "outcome": outcome,
        "collision_free": collision_free,
        "collision_uncertain": collision_uncertain,
        "topology_coverage_complete": complete,
        "maximum_minimum_part_volume_ratio": maximum_ratio,
        "uncertain_collision_ratio_threshold": float(uncertain_collision_ratio),
    }


def _run_worker(
    output_root: Path,
    stage: str,
    command: list[str],
    *,
    timeout_seconds: int,
    max_memory_gb: float,
) -> None:
    wrapper = HERE / "run_safe_occt_worker.py"
    complete = [
        sys.executable,
        str(wrapper),
        "--output-root",
        str(output_root),
        "--stage",
        stage,
        "--timeout",
        str(timeout_seconds),
        "--cpu-affinity",
        "0-7",
        "--max-worker-memory-gb",
        str(max_memory_gb),
        "--min-free-memory-gb",
        "8",
        "--cooldown-seconds",
        "3",
        "--",
        *command,
    ]
    subprocess.run(complete, cwd=ROOT, check=True)


def run_visual_pose_rerank(
    geometry_output: Path,
    output_dir: Path,
    *,
    mode: str = "live",
    exact_top_n: int = 3,
    view_width: int = 720,
    view_height: int = 520,
) -> dict[str, Any]:
    frontier_path = (
        geometry_output / "pose_candidate_frontier" / "pose_candidate_frontier.json"
    )
    frontier = _read(frontier_path)
    raw_candidates = list(frontier.get("candidates") or [])
    candidates = []
    seen_pose_signatures: set[str] = set()
    for row in raw_candidates:
        manifest = _read(Path(row["manifest"]))
        signature = json.dumps(
            [
                {
                    "source": component.get("source"),
                    "placement": component.get("placement") or {},
                }
                for component in manifest.get("components") or []
            ],
            sort_keys=True,
            separators=(",", ":"),
        )
        if signature in seen_pose_signatures:
            continue
        seen_pose_signatures.add(signature)
        candidates.append(row)
    if not candidates:
        raise ValueError(f"empty pose candidate frontier: {frontier_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = output_dir / "candidate_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    images = []
    rendered_candidates = []
    render_failures = []
    for index, row in enumerate(candidates, start=1):
        candidate_id = str(row["candidate_id"])
        image_path = preview_dir / f"{candidate_id}.png"
        if not image_path.is_file():
            try:
                _run_worker(
                    output_dir,
                    f"render_{index:02d}_{candidate_id}",
                    [
                        sys.executable,
                        str(HERE / "render_assembly_manifest_occt.py"),
                        str(Path(row["manifest"]).resolve()),
                        str(image_path.resolve()),
                        "--view-width",
                        str(view_width),
                        "--view-height",
                        str(view_height),
                        "--expanded-views",
                        "--relationship-view",
                        "--context-transparency",
                        "0.72",
                    ],
                    timeout_seconds=420,
                    max_memory_gb=8.0,
                )
            except (subprocess.CalledProcessError, RuntimeError) as exc:
                render_failures.append({
                    "candidate_id": candidate_id,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                continue
        images.append(image_path.resolve())
        row["preview"] = str(image_path.resolve())
        rendered_candidates.append(row)

    _write(output_dir / "candidate_render_audit.json", {
        "schema_version": "visual_pose_render_audit.v1",
        "input_candidate_count": len(candidates),
        "rendered_candidate_count": len(rendered_candidates),
        "failed_candidate_count": len(render_failures),
        "render_failures": render_failures,
    })
    if not rendered_candidates:
        raise RuntimeError("no pose candidate preview could be rendered")

    first_manifest = _read(Path(rendered_candidates[0]["manifest"]))
    part_ids = [
        str(component.get("label") or component.get("id"))
        for component in first_manifest.get("components") or []
    ]
    candidate_ids = [str(row["candidate_id"]) for row in rendered_candidates]
    text_context = json.dumps(
        {
            "task": "compare mechanical assembly pose candidates",
            "candidate_ids_in_image_order": candidate_ids,
            "part_ids": part_ids,
            "instructions": [
                "infer roles, receiving region, direction and assembly action",
                "judge centering/containment and functional-face visibility",
                "do not infer from geometry scores; none are supplied",
                "preserve equivalent candidates and abstain when ambiguous",
            ],
        },
        ensure_ascii=False,
        indent=2,
    )
    config_path = HERE / "configs" / "pool_pipeline.json"
    config = dict(_read(config_path).get("multimodal_review", {}))
    config["vision_max_tokens"] = max(
        8192, int(config.get("vision_max_tokens", 0) or 0)
    )
    config["vision_timeout_seconds"] = max(
        90, int(config.get("vision_timeout_seconds", 0) or 0)
    )
    reviewer = QwenVLReviewer(config, output_dir / "cache")
    fallback = _fallback(candidate_ids, part_ids)
    record = reviewer.structured_review(
        f"pose_candidate_batch:{frontier.get('assembly_id', geometry_output.name)}",
        images,
        text_context,
        system_prompt=POSE_REVIEW_SYSTEM_PROMPT,
        prompt_version="pose_candidate_semantic_rerank.v1",
        validate_output=_validator(set(candidate_ids), set(part_ids)),
        fallback_output=fallback,
        mode=mode,  # type: ignore[arg-type]
    )
    _write(output_dir / "visual_pose_review.json", record)
    visual_output = record.get("output") or fallback
    reranked = rank_candidates(candidates, visual_output)
    _write(output_dir / "semantic_reranked_frontier.json", {
        "schema_version": "semantic_reranked_pose_frontier.v1",
        "visual_stage_status": record.get("status"),
        "visual_model": record.get("model"),
        "geometry_scores_exposed_to_visual_model": False,
        "semantic_auto_accept_enabled": False,
        "candidates": reranked,
    })

    exact_rows = []
    exact_candidates = [
        row for row in reranked
        if not row["semantic_forbidden"] and not row["known_collision"]
    ][: max(1, int(exact_top_n))]
    for index, row in enumerate(exact_candidates, start=1):
        audit_path = output_dir / "exact_collision" / f"{row['candidate_id']}.json"
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        if not audit_path.is_file():
            _run_worker(
                output_dir,
                f"exact_{index:02d}_{row['candidate_id']}",
                [
                    sys.executable,
                    str(HERE / "case5_collision_audit.py"),
                    str(Path(row["manifest"]).resolve()),
                    str(audit_path.resolve()),
                    "--max-pairs",
                    "256",
                ],
                timeout_seconds=600,
                max_memory_gb=8.0,
            )
        audit = _read(audit_path)
        collision_gate = classify_exact_collision_audit(audit)
        exact_rows.append({
            "candidate_id": row["candidate_id"],
            "audit": str(audit_path.resolve()),
            "status": audit.get("status"),
            "collision_result": audit.get("collision_result"),
            "collision_count": len(audit.get("collisions") or []),
            **collision_gate,
        })

    exact_by_id = {row["candidate_id"]: row for row in exact_rows}
    selected = next(
        (
            row for row in reranked
            if exact_by_id.get(row["candidate_id"], {}).get("collision_free")
        ),
        None,
    )
    selected_collision_uncertain = False
    if selected is None:
        # Preserve an inspectable pose when the exact checker is incomplete
        # and reports only a bounded sub-threshold fitted overlap.  This is a
        # Review outcome, never Accepted or collision-free.
        selected = next(
            (
                row for row in reranked
                if exact_by_id.get(row["candidate_id"], {}).get(
                    "collision_uncertain"
                )
            ),
            None,
        )
        selected_collision_uncertain = selected is not None
    if selected is not None:
        final_manifest = output_dir / "selected_review_manifest.json"
        shutil.copyfile(Path(selected["manifest"]), final_manifest)
        final_status = "review"
        if selected_collision_uncertain:
            reason = (
                "visual-preferred candidate retained for review: OCCT found "
                "only a bounded small overlap on incomplete mixed solid/open-"
                "shell topology; it is not proven collision-free"
            )
        else:
            reason = (
                "visual-preferred candidate passed bounded exact collision; "
                "precision and insertion-path gates remain required"
            )
    else:
        final_manifest = None
        final_status = "unresolved"
        reason = "no visual-scheduled candidate passed complete exact collision"
    decision = {
        "schema_version": "visual_pose_gate.v1",
        "status": final_status,
        "accepted": False,
        "review_required": True,
        "selected_candidate_id": selected.get("candidate_id") if selected else None,
        "selected_source_pose_rank": selected.get("source_pose_rank") if selected else None,
        "selected_manifest": str(final_manifest.resolve()) if final_manifest else None,
        "reason": reason,
        "exact_collision_audits": exact_rows,
        "visual_semantics_used_for_ranking": record.get("status") == "ok",
        "geometry_scores_exposed_to_visual_model": False,
        "semantic_auto_accept_enabled": False,
        "insertion_path_validation": "not_available_review_required",
        "selected_collision_uncertain": selected_collision_uncertain,
    }
    _write(output_dir / "final_visual_pose_decision.json", decision)
    return decision


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("live", "cache_only", "off"), default="live")
    parser.add_argument("--exact-top-n", type=int, default=3)
    parser.add_argument("--view-width", type=int, default=720)
    parser.add_argument("--view-height", type=int, default=520)
    args = parser.parse_args()
    decision = run_visual_pose_rerank(
        args.geometry_output.resolve(),
        args.output_dir.resolve(),
        mode=args.mode,
        exact_top_n=args.exact_top_n,
        view_width=args.view_width,
        view_height=args.view_height,
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0 if decision["status"] in {"review", "accepted"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
