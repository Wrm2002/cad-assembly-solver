"""Fuse JoinABLe pair-pose proposals with visual-region B-Rep candidates.

The bridge is intentionally conservative:

* JoinABLe and analytic/visual providers receive protected quotas;
* visual semantics may add only a small ranking prior and never create an
  acceptance decision;
* every output remains review-only until downstream collision, insertion-path
  and group-consistency gates are complete.

The current CLI targets the three-part Case5 audit, but all fusion and SE(3)
functions use filename-independent part identifiers and proper rigid matrices.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.spatial.transform import Rotation

from global_pose_solver.audited_hypothesis_solver import solve_bounded_global_pose
from global_pose_solver.joinable_adapter import load_joinable_pose_pool
from pose_search import matrix_to_placement


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _matrix_from_candidate(row: dict[str, Any]) -> np.ndarray:
    rotation = np.asarray(row["R"], dtype=float)
    translation = np.asarray(row["t_mm"], dtype=float)
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("visual/analytic candidate requires R(3x3) and t_mm(3)")
    if abs(float(np.linalg.det(rotation)) - 1.0) > 1e-4:
        raise ValueError("candidate rotation must be proper rigid")
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


def _se3_distance(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    relative = np.linalg.inv(left) @ right
    return {
        "translation_mm": float(np.linalg.norm(relative[:3, 3])),
        "rotation_degrees": float(
            np.degrees(Rotation.from_matrix(relative[:3, :3]).magnitude())
        ),
    }


def visual_analytic_pool(
    source: str,
    target: str,
    rows: Iterable[dict[str, Any]],
    *,
    maximum_candidates: int = 8,
    prior_validations: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Convert numbered B-Rep regions into one protected pair-pose pool."""

    candidates = []
    excluded = []
    prior_validations = prior_validations or {}
    for index, row in enumerate(rows):
        candidate_id = str(row.get("candidate_id") or f"analytic:{index:03d}")
        validation = prior_validations.get(candidate_id) or {}
        decision = str(validation.get("decision", ""))
        if (
            decision.startswith("rejected_")
            or validation.get("collision_result") == "collision_detected"
        ):
            excluded.append({
                "index": index,
                "candidate_id": candidate_id,
                "reason": "prior_physical_validation_rejected",
                "validation": validation,
            })
            continue
        try:
            transform = _matrix_from_candidate(row)
        except (KeyError, TypeError, ValueError) as exc:
            excluded.append({"index": index, "reason": str(exc)})
            continue
        geometry = float(row.get("geometry_score", row.get("score", 0.0)) or 0.0)
        semantic = float(row.get("semantic_region_score", 0.0) or 0.0)
        preferred_rank = row.get("semantic_preferred_rank")
        # The visual term is deliberately capped at 0.05.  It changes review
        # order but cannot turn a weak geometric candidate into physical proof.
        visual_bonus = min(0.05, 0.05 * max(0.0, min(1.0, semantic)))
        prior = geometry + visual_bonus
        candidates.append({
            "candidate_id": candidate_id,
            "source": source,
            "target": target,
            "T_rel": transform.tolist(),
            "prior": prior,
            "rank": index + 1,
            "candidate_origin": "analytic_brep_visual_region",
            "provider": "analytic_visual",
            "region_id": row.get("region_id"),
            "geometry_score": geometry,
            "semantic_region_score": semantic,
            "semantic_preferred_rank": preferred_rank,
            "semantic_forbidden": bool(row.get("semantic_forbidden", False)),
            "protected": bool(row.get("protected", False)),
            "candidate_sources": list(row.get("candidate_sources") or []),
            "visual_bonus": visual_bonus,
            "proposal_only": True,
            "can_auto_accept": False,
            "prior_validation": validation or None,
        })
    candidates.sort(key=lambda row: (-float(row["prior"]), row["candidate_id"]))
    retained = candidates[: max(1, int(maximum_candidates))]
    return {"source": source, "target": target, "candidates": retained}, {
        "provider": "analytic_visual",
        "source": source,
        "target": target,
        "input_count": len(candidates) + len(excluded),
        "retained_count": len(retained),
        "excluded": excluded,
        "visual_bonus_cap": 0.05,
        "can_auto_accept": False,
    }


def load_candidate_validations(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    """Load candidate-level physical decisions from prior audited stages."""

    result: dict[str, dict[str, Any]] = {}
    for path in paths:
        payload = _read(path)
        rows = payload.get("candidate_audit") or payload.get("candidates") or []
        for row in rows:
            if not isinstance(row, dict) or not row.get("candidate_id"):
                continue
            result[str(row["candidate_id"])] = {**row, "validation_source": str(path)}
    return result


def _annotate_joinable_visual_agreement(
    joinable_pool: dict[str, Any],
    visual_pool: dict[str, Any] | None,
) -> dict[str, Any]:
    """Record proximity to visual regions without deleting either provider."""

    visual_rows = list((visual_pool or {}).get("candidates") or [])
    result = {**joinable_pool, "candidates": []}
    for row in joinable_pool.get("candidates") or []:
        candidate = dict(row)
        transform = np.asarray(candidate["T_rel"], dtype=float)
        nearest = None
        for visual in visual_rows:
            distance = _se3_distance(
                transform, np.asarray(visual["T_rel"], dtype=float)
            )
            quality = (
                distance["translation_mm"] / 20.0
                + distance["rotation_degrees"] / 45.0
            )
            if nearest is None or quality < nearest[0]:
                nearest = quality, visual, distance
        if nearest is not None:
            _, visual, distance = nearest
            supported = (
                distance["translation_mm"] <= 20.0
                and distance["rotation_degrees"] <= 15.0
            )
            candidate["nearest_visual_region"] = {
                "candidate_id": visual["candidate_id"],
                "region_id": visual.get("region_id"),
                **distance,
                "supported": supported,
            }
            if supported:
                candidate["prior"] = float(candidate.get("prior", 0.0)) + 0.05
        candidate["provider"] = "joinable"
        candidate["proposal_only"] = True
        candidate["can_auto_accept"] = False
        result["candidates"].append(candidate)
    return result


def fuse_provider_quotas(
    pools: Iterable[dict[str, Any]],
    *,
    maximum_per_provider: int = 8,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Merge providers per part pair while guaranteeing each a bounded quota."""

    by_pair: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = {}
    orientation: dict[tuple[str, str], tuple[str, str]] = {}
    for pool in pools:
        source, target = str(pool["source"]), str(pool["target"])
        key = tuple(sorted((source, target)))
        orientation.setdefault(key, (source, target))
        provider = str(
            (pool.get("candidates") or [{}])[0].get("provider", "unknown")
        )
        by_pair.setdefault(key, {}).setdefault(provider, []).extend(
            list(pool.get("candidates") or [])
        )
    fused = []
    pair_audits = []
    for key in sorted(by_pair):
        source, target = orientation[key]
        rows = []
        counts = {}
        for provider, provider_rows in sorted(by_pair[key].items()):
            ordered = sorted(
                provider_rows,
                key=lambda row: (-float(row.get("prior", 0.0)), row["candidate_id"]),
            )
            selected = ordered[: max(1, int(maximum_per_provider))]
            rows.extend(selected)
            counts[provider] = {
                "input": len(provider_rows),
                "retained": len(selected),
            }
        fused.append({"source": source, "target": target, "candidates": rows})
        pair_audits.append({
            "pair": list(key),
            "provider_counts": counts,
            "fused_count": len(rows),
        })
    return fused, {
        "schema_version": "visual_joinable_provider_fusion.v1",
        "maximum_per_provider": int(maximum_per_provider),
        "pairs": pair_audits,
        "provider_scores_are_ranking_priors_only": True,
        "can_auto_accept": False,
    }


def _result_paths(root: Path) -> list[Path]:
    return sorted(root.rglob("joinable_e2e_result.json"))


def run_case5_bridge(
    case_dir: Path,
    joinable_root: Path,
    visual_candidates_path: Path,
    output_dir: Path,
    *,
    maximum_per_provider: int = 8,
    candidate_validation_paths: Iterable[Path] = (),
) -> dict[str, Any]:
    visual = _read(visual_candidates_path)
    prior_validations = load_candidate_validations(candidate_validation_paths)
    carrier = "01-ASSY-CHASSIS-MODULE-R6250H0.stp"
    ear = "01-ASSY-CHASSIS-EAR-L-R620.stp"
    psu = "5-CRPS1300NC.stp"
    sources = {
        carrier: case_dir / carrier,
        ear: case_dir / ear,
        psu: case_dir / psu,
    }
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing Case5 STEP input(s): {missing}")

    analytic_pools = []
    analytic_audits = []
    for target, key in ((ear, "ear"), (psu, "psu")):
        pool, audit = visual_analytic_pool(
            carrier,
            target,
            visual.get(key) or [],
            maximum_candidates=maximum_per_provider,
            prior_validations=prior_validations,
        )
        analytic_pools.append(pool)
        analytic_audits.append(audit)

    visual_by_pair = {
        tuple(sorted((pool["source"], pool["target"]))): pool
        for pool in analytic_pools
    }
    joinable_pools = []
    joinable_audits = []
    for result_path in _result_paths(joinable_root):
        payload = _read(result_path)
        source = Path(str(payload["part_a_fixed"])).name
        target = Path(str(payload["part_b_moving"])).name
        pool, audit = load_joinable_pose_pool(
            source,
            target,
            result_path,
            maximum_candidates=maximum_per_provider,
            include_not_checked=True,
            include_manifold_initials=True,
        )
        key = tuple(sorted((source, target)))
        pool = _annotate_joinable_visual_agreement(
            pool, visual_by_pair.get(key)
        )
        joinable_pools.append(pool)
        joinable_audits.append(audit)

    fused_pools, fusion_audit = fuse_provider_quotas(
        [*analytic_pools, *joinable_pools],
        maximum_per_provider=maximum_per_provider,
    )
    solve = solve_bounded_global_pose(
        [carrier, ear, psu],
        fused_pools,
        anchor_id=carrier,
        max_candidates_per_pair=maximum_per_provider * 2,
        max_topologies=8,
        max_hypotheses=128,
        exact_validator=None,
        validate_top_n=0,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write(output_dir / "unified_candidate_pools.json", fused_pools)
    _write(output_dir / "provider_fusion_audit.json", {
        **fusion_audit,
        "analytic_audits": analytic_audits,
        "joinable_audits": joinable_audits,
        "joinable_report_count": len(joinable_audits),
        "prior_validation_candidate_count": len(prior_validations),
    })
    _write(output_dir / "global_pose_result.json", solve)

    manifests = []
    for rank, hypothesis in enumerate(solve.get("hypotheses") or [], start=1):
        manifest = {
            "schema_version": "2.0.0",
            "assembly_name": f"case5_joinable_visual_hypothesis_{rank:02d}",
            "global_units": "mm",
            "review_required": True,
            "can_auto_accept": False,
            "hypothesis_id": hypothesis["hypothesis_id"],
            "components": [],
        }
        for index, part in enumerate((carrier, ear, psu), start=1):
            matrix = np.asarray(hypothesis["part_poses"][part], dtype=float)
            manifest["components"].append({
                "id": part,
                "label": part,
                "source": str(sources[part].resolve()),
                "placement": matrix_to_placement(matrix),
                "color_index": index,
            })
        path = output_dir / "hypotheses" / f"rank_{rank:02d}.manifest.json"
        _write(path, manifest)
        manifests.append(str(path.resolve()))
        if rank >= 12:
            break
    summary = {
        "schema_version": "case5_visual_joinable_bridge.v1",
        "status": solve.get("status"),
        "accepted": False,
        "review_required": True,
        "joinable_report_count": len(joinable_audits),
        "fused_pair_pool_count": len(fused_pools),
        "global_hypothesis_count": solve.get("hypothesis_count", 0),
        "manifest_count": len(manifests),
        "manifests": manifests,
        "acceptance_boundary": (
            "Visual and JoinABLe scores only rank a protected candidate union. "
            "No hypothesis is accepted before exact collision, insertion-path "
            "and group-consistency validation."
        ),
    }
    _write(output_dir / "bridge_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--joinable-root", type=Path, required=True)
    parser.add_argument("--visual-candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maximum-per-provider", type=int, default=8)
    parser.add_argument(
        "--candidate-validation",
        action="append",
        type=Path,
        default=[],
        help="candidate-level collision/physical audit JSON; rejected rows are removed",
    )
    args = parser.parse_args()
    summary = run_case5_bridge(
        args.case_dir.resolve(),
        args.joinable_root.resolve(),
        args.visual_candidates.resolve(),
        args.output_dir.resolve(),
        maximum_per_provider=args.maximum_per_provider,
        candidate_validation_paths=[path.resolve() for path in args.candidate_validation],
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
