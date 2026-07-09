"""Evaluate the geometry pipeline on the real STEP groups in folders 1-5.

The source folders are never modified. Files are hard-linked into an isolated,
anonymized mixed pool. Ground-truth folder membership is used only for
evaluation and is not exposed to indexing, grouping, or pose workers.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from constraints import match_features
from contracts import CandidateStatus, PartFeature
from conservative_pipeline import geometry_tiers
from global_grouping import evaluate_groups, generate_group_proposals, assign_groups
from match_pruning import prune_match_graph
from match_scoring import score_matches
from pool_index import _contract, _legacy_features, _prescreen


HERE = Path(__file__).resolve().parent
SOURCE_CASES = [str(number) for number in range(1, 6)]


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _source_parts(case_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in case_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".step", ".stp"}
        and not path.name.lower().startswith("assembly")
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_pool(source_root: Path, output_root: Path) -> dict[str, Any]:
    pool = output_root / "mixed_real_pool"
    parts_dir = pool / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    source_rows = []
    for case_id in SOURCE_CASES:
        for path in _source_parts(source_root / case_id):
            source_rows.append((case_id, path))
    shuffled = list(source_rows)
    random.Random(20260703).shuffle(shuffled)
    mapping = {}
    groups: dict[str, list[str]] = {case_id: [] for case_id in SOURCE_CASES}
    for index, (case_id, source) in enumerate(shuffled, start=1):
        anonymous = f"part_{index:03d}{source.suffix.lower()}"
        destination = parts_dir / anonymous
        if not destination.exists():
            try:
                os.link(source, destination)
            except OSError:
                import shutil

                shutil.copy2(source, destination)
        mapping[anonymous] = {
            "case_id": case_id,
            "source_name": source.name,
            "source_path": str(source.resolve()),
            "bytes": source.stat().st_size,
            "sha256": _sha256(source),
        }
        groups[case_id].append(anonymous)
    gt = {
        "schema_version": "1.0.0",
        "pool_id": "mixed_real_pool",
        "parts": sorted(mapping),
        "true_groups": [
            {
                "group_id": f"REAL_{case_id}",
                "parts": sorted(groups[case_id]),
                "true_mates": [],
            }
            for case_id in SOURCE_CASES
        ],
        "evaluation_note": (
            "Folder membership is supplied by the user. Exact source mate "
            "and source pose labels are not available."
        ),
    }
    _write(pool / "pool_gt.json", gt)
    _write(output_root / "private_source_map.json", mapping)
    return {"pool": pool, "mapping": mapping, "groups": groups, "gt": gt}


def extract_features_isolated(
    pool: Path,
    output_root: Path,
    *,
    timeout_seconds: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    feature_dir = pool / "index" / "parts"
    feature_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    features = []
    files = sorted(
        path
        for path in (pool / "parts").iterdir()
        if path.suffix.lower() in {".step", ".stp"}
    )
    for position, path in enumerate(files, start=1):
        output = feature_dir / f"{path.name}.json"
        stdout_path = output_root / "logs" / f"{path.name}.stdout.log"
        stderr_path = output_root / "logs" / f"{path.name}.stderr.log"
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        if output.is_file():
            try:
                feature = _load(output)
                PartFeature.model_validate(feature)
                features.append(feature)
                reports.append(
                    {
                        "part_id": path.name,
                        "status": "resumed",
                        "elapsed_seconds": 0.0,
                    }
                )
                print(
                    f"feature {position}/{len(files)} {path.name}: resumed",
                    flush=True,
                )
                continue
            except Exception:
                output.unlink(missing_ok=True)
        print(
            f"feature {position}/{len(files)} {path.name}: extracting",
            flush=True,
        )
        try:
            process = subprocess.run(
                [
                    sys.executable,
                    str(HERE / "real_part_feature_worker.py"),
                    str(path),
                    path.name,
                    str(output),
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds,
            )
            stdout_path.write_text(process.stdout, encoding="utf-8")
            stderr_path.write_text(process.stderr, encoding="utf-8")
            status = "success" if process.returncode == 0 else "worker_failed"
            report = {
                "part_id": path.name,
                "status": status,
                "returncode": process.returncode,
                "elapsed_seconds": time.perf_counter() - started,
                "stdout_tail": process.stdout[-1000:],
                "stderr_tail": process.stderr[-2000:],
            }
        except subprocess.TimeoutExpired as exc:
            status = "timeout"
            report = {
                "part_id": path.name,
                "status": status,
                "returncode": None,
                "elapsed_seconds": time.perf_counter() - started,
                "stdout_tail": (exc.stdout or "")[-1000:]
                if isinstance(exc.stdout, str)
                else "",
                "stderr_tail": (exc.stderr or "")[-2000:]
                if isinstance(exc.stderr, str)
                else "",
            }
        reports.append(report)
        if status == "success" and output.is_file():
            feature = _load(output)
            PartFeature.model_validate(feature)
            features.append(feature)
        print(
            f"feature {path.name}: {status} "
            f"{report['elapsed_seconds']:.1f}s",
            flush=True,
        )
    _write(output_root / "feature_extraction_report.json", reports)
    return features, reports


def build_index_from_features(
    pool: Path,
    features: list[dict[str, Any]],
    pipeline_config: dict[str, Any],
) -> dict[str, Any]:
    parts = {
        item["part_id"]: PartFeature.model_validate(item)
        for item in features
    }
    prescreen_config = pipeline_config["prescreen"]
    audit = []
    for first_id, second_id in itertools.combinations(parts, 2):
        _, record = _prescreen(
            parts[first_id], parts[second_id], prescreen_config
        )
        audit.append(record)
    maximum_neighbors = int(
        prescreen_config["maximum_coarse_neighbors_per_part"]
    )
    ranked_by_part = {part_id: [] for part_id in parts}
    for record in audit:
        if record["accepted"]:
            for part_id in record["parts"]:
                ranked_by_part[part_id].append(record)
    allowed_pairs = set()
    for records in ranked_by_part.values():
        ranked = sorted(
            records,
            key=lambda item: (
                float(item["coarse_score"]),
                tuple(item["parts"]),
            ),
            reverse=True,
        )[:maximum_neighbors]
        allowed_pairs.update(
            tuple(sorted(item["parts"])) for item in ranked
        )
    accepted_pairs = []
    for record in audit:
        pair = tuple(sorted(record["parts"]))
        if record["accepted"] and pair not in allowed_pairs:
            record["accepted"] = False
            record["rejection_reason"] = (
                f"outside top {maximum_neighbors} coarse neighbors"
            )
        if record["accepted"]:
            accepted_pairs.append(tuple(record["parts"]))

    legacy = {
        part_id: _legacy_features(part)
        for part_id, part in parts.items()
    }
    generation = pipeline_config["candidate_generation"]
    raw_matches = []
    for first_id, second_id in accepted_pairs:
        raw_matches.extend(
            match_features(
                {
                    first_id: legacy[first_id],
                    second_id: legacy[second_id],
                },
                generation.get("detailed_thresholds", {}),
            )
        )
    scored = score_matches(raw_matches, legacy)
    kept, removed = prune_match_graph(
        scored,
        min_score=float(generation["minimum_score"]),
        top_k_pair=int(generation["top_k_per_pair"]),
        max_neighbors=int(generation["maximum_neighbors"]),
    )
    index_dir = pool / "index"
    outputs = {
        "part_features.json": [
            parts[part_id].model_dump(mode="json")
            for part_id in sorted(parts)
        ],
        "screening_audit.json": {
            "config": prescreen_config,
            "total_pairs": len(audit),
            "accepted_pairs": len(accepted_pairs),
            "rejected_pairs": len(audit) - len(accepted_pairs),
            "pairs": audit,
        },
        "geometry_candidates.json": [
            _contract(match, CandidateStatus.generated).model_dump(
                mode="json"
            )
            for match in scored
        ],
        "pruned_candidates.json": [
            _contract(match, CandidateStatus.kept).model_dump(mode="json")
            for match in kept
        ],
        "removed_candidates.json": [
            _contract(
                match,
                CandidateStatus.removed,
                {
                    "removal_reason": match.get("removal_reason"),
                    "removal_detail": match.get("removal_detail"),
                },
            ).model_dump(mode="json")
            for match in removed
        ],
    }
    for name, payload in outputs.items():
        _write(index_dir / name, payload)
    return outputs


def projected_face_comparisons(
    features: list[dict[str, Any]],
    part_ids: list[str] | None = None,
) -> int:
    """Estimate the nested face-pair loops used by the current matcher."""
    allowed = set(part_ids) if part_ids is not None else None
    selected = [
        item
        for item in features
        if allowed is None or item["part_id"] in allowed
    ]
    total = 0
    for first, second in itertools.combinations(selected, 2):
        total += (
            len(first.get("cylindrical_faces", []))
            * len(second.get("cylindrical_faces", []))
        )
        total += (
            len(first.get("planar_faces", []))
            * len(second.get("planar_faces", []))
        )
    return total


def build_index_isolated(
    pool: Path,
    output_root: Path,
    config_path: Path,
    *,
    timeout_seconds: int,
    projected_comparisons: int,
    maximum_comparisons: int,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    stdout_path = output_root / "logs" / "index.stdout.log"
    stderr_path = output_root / "logs" / "index.stderr.log"
    started = time.perf_counter()
    if projected_comparisons > maximum_comparisons:
        report = {
            "status": "complexity_limit_exceeded",
            "returncode": None,
            "elapsed_seconds": 0.0,
            "projected_face_comparisons": projected_comparisons,
            "maximum_safe_comparisons": maximum_comparisons,
            "stdout_tail": "",
            "stderr_tail": (
                "Current prescreen enumerates Cartesian products over all "
                "planar and cylindrical faces."
            ),
        }
        _write(output_root / "index_worker_report.json", report)
        print(
            "detailed indexing: complexity_limit_exceeded "
            f"projected={projected_comparisons:,}",
            flush=True,
        )
        return None, report
    print("detailed mixed-pool indexing: isolated worker", flush=True)
    try:
        process = subprocess.run(
            [
                sys.executable,
                str(HERE / "real_index_worker.py"),
                str(pool),
                "--config",
                str(config_path),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )
        stdout_path.write_text(process.stdout, encoding="utf-8")
        stderr_path.write_text(process.stderr, encoding="utf-8")
        report = {
            "status": (
                "success" if process.returncode == 0 else "worker_failed"
            ),
            "returncode": process.returncode,
            "elapsed_seconds": time.perf_counter() - started,
            "stdout_tail": process.stdout[-2000:],
            "stderr_tail": process.stderr[-4000:],
        }
    except subprocess.TimeoutExpired as exc:
        report = {
            "status": "timeout",
            "returncode": None,
            "elapsed_seconds": time.perf_counter() - started,
            "stdout_tail": (exc.stdout or "")[-2000:]
            if isinstance(exc.stdout, str)
            else "",
            "stderr_tail": (exc.stderr or "")[-4000:]
            if isinstance(exc.stderr, str)
            else "",
        }
    _write(output_root / "index_worker_report.json", report)
    print(
        f"detailed indexing: {report['status']} "
        f"{report['elapsed_seconds']:.1f}s",
        flush=True,
    )
    required = [
        "part_features.json",
        "screening_audit.json",
        "geometry_candidates.json",
        "pruned_candidates.json",
        "removed_candidates.json",
    ]
    index_dir = pool / "index"
    if report["status"] != "success" or not all(
        (index_dir / name).is_file() for name in required
    ):
        return None, report
    return {
        name: _load(index_dir / name) for name in required
    }, report


def run_grouping(
    pool: Path,
    index_outputs: dict[str, Any],
    pipeline_config: dict[str, Any],
    conservative_config: dict[str, Any],
) -> dict[str, Any]:
    part_ids = sorted(
        item["part_id"] for item in index_outputs["part_features.json"]
    )
    proposals = generate_group_proposals(
        part_ids,
        index_outputs["pruned_candidates.json"],
        pipeline_config["global_grouping"],
    )
    assignment = assign_groups(
        part_ids, proposals, pipeline_config["global_grouping"]
    )
    gt = _load(pool / "pool_gt.json")
    metrics = evaluate_groups(assignment["selected_groups"], gt)
    proposal_rows = [
        proposal.model_dump(mode="json") for proposal in proposals
    ]
    accepted, review, rejected = geometry_tiers(
        pool,
        proposal_rows,
        index_outputs["pruned_candidates.json"],
        conservative_config,
    )
    grouping_dir = pool / "grouping"
    _write(grouping_dir / "group_proposals.json", proposal_rows)
    _write(
        grouping_dir / "group_assignment.json",
        {
            **assignment,
            "schema_version": "1.0.0",
            "pool_id": pool.name,
            "metrics": metrics,
        },
    )
    _write(grouping_dir / "accepted_geometry_candidates.json", accepted)
    _write(grouping_dir / "review_geometry_candidates.json", review)
    _write(grouping_dir / "rejected_geometry_candidates.json", rejected)
    truth_sets = {
        frozenset(group["parts"]): group["group_id"]
        for group in gt["true_groups"]
    }
    proposal_sets = {
        frozenset(proposal["parts"]): proposal["group_id"]
        for proposal in proposal_rows
    }
    coverage = [
        {
            "true_group_id": group_id,
            "parts": sorted(parts),
            "proposal_generated": parts in proposal_sets,
            "proposal_id": proposal_sets.get(parts),
        }
        for parts, group_id in truth_sets.items()
    ]
    return {
        "proposals": proposal_rows,
        "assignment": assignment,
        "metrics": metrics,
        "coverage": coverage,
        "geometry_tiers": {
            "accepted": len(accepted),
            "review": len(review),
            "rejected": len(rejected),
        },
    }


def validate_known_groups(
    pool: Path,
    groups: dict[str, list[str]],
    output_root: Path,
    config_path: Path,
    features: list[dict[str, Any]],
    *,
    timeout_seconds: int,
    maximum_comparisons: int,
) -> list[dict[str, Any]]:
    reports = []
    for case_id in SOURCE_CASES:
        group_id = f"REAL_{case_id}"
        parts = sorted(groups[case_id])
        result_path = (
            pool
            / "known_group_validation"
            / group_id
            / "validation_result.json"
        )
        projected = projected_face_comparisons(features, parts)
        started = time.perf_counter()
        print(
            f"known group {case_id}: {len(parts)} parts, "
            f"projected comparisons={projected:,}",
            flush=True,
        )
        if projected > maximum_comparisons:
            report = {
                "case_id": case_id,
                "group_id": group_id,
                "parts": parts,
                "worker_status": "complexity_limit_exceeded",
                "returncode": None,
                "elapsed_seconds": 0.0,
                "validation": None,
                "projected_face_comparisons": projected,
                "maximum_safe_comparisons": maximum_comparisons,
                "stdout_tail": "",
                "stderr_tail": (
                    "Known-group matcher would exceed the experimental "
                    "face-pair safety limit."
                ),
            }
            reports.append(report)
            print(
                f"known group {case_id}: complexity_limit_exceeded",
                flush=True,
            )
            continue
        try:
            process = subprocess.run(
                [
                    sys.executable,
                    str(HERE / "real_known_group_worker.py"),
                    str(pool),
                    group_id,
                    json.dumps(parts),
                    "--config",
                    str(config_path),
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds,
            )
            result = _load(result_path) if result_path.is_file() else None
            report = {
                "case_id": case_id,
                "group_id": group_id,
                "parts": parts,
                "worker_status": (
                    "success" if process.returncode == 0 else "failed"
                ),
                "returncode": process.returncode,
                "elapsed_seconds": time.perf_counter() - started,
                "validation": result,
                "projected_face_comparisons": projected,
                "stdout_tail": process.stdout[-1000:],
                "stderr_tail": process.stderr[-3000:],
            }
        except subprocess.TimeoutExpired as exc:
            report = {
                "case_id": case_id,
                "group_id": group_id,
                "parts": parts,
                "worker_status": "timeout",
                "returncode": None,
                "elapsed_seconds": time.perf_counter() - started,
                "validation": None,
                "projected_face_comparisons": projected,
                "stdout_tail": (exc.stdout or "")[-1000:]
                if isinstance(exc.stdout, str)
                else "",
                "stderr_tail": (exc.stderr or "")[-3000:]
                if isinstance(exc.stderr, str)
                else "",
            }
        reports.append(report)
        print(
            f"known group {case_id}: {report['worker_status']} "
            f"{report['elapsed_seconds']:.1f}s",
            flush=True,
        )
    _write(output_root / "known_group_pose_results.json", reports)
    return reports


def summarize(
    prepared: dict[str, Any],
    feature_reports: list[dict[str, Any]],
    index_outputs: dict[str, Any] | None,
    grouping: dict[str, Any] | None,
    pose_reports: list[dict[str, Any]],
    features: list[dict[str, Any]],
    output_root: Path,
) -> dict[str, Any]:
    successful_features = sum(
        report["status"] in {"success", "resumed"}
        for report in feature_reports
    )
    pose_statuses = {}
    for report in pose_reports:
        validation = report.get("validation") or {}
        metrics = validation.get("metrics", {})
        pose_statuses[report["case_id"]] = {
            "worker_status": report["worker_status"],
            "validation_status": validation.get("status"),
            "physical_pose_valid": metrics.get(
                "physical_pose_valid", metrics.get("accepted")
            ),
            "final_pose_status": metrics.get("final_pose_status"),
            "unsolved_parts": validation.get("unsolved_parts"),
            "collision_count": validation.get("collision_count"),
            "projected_face_comparisons": report.get(
                "projected_face_comparisons"
            ),
        }
    summary = {
        "schema_version": "1.0.0",
        "dataset": "user-provided real cases 1-5",
        "input_part_count": len(prepared["mapping"]),
        "true_group_count": 5,
        "feature_success_count": successful_features,
        "feature_failure_count": len(feature_reports) - successful_features,
        "mixed_pool_projected_face_comparisons": (
            projected_face_comparisons(features)
        ),
        "prescreen_accepted_pairs": (
            index_outputs["screening_audit.json"]["accepted_pairs"]
            if index_outputs
            else None
        ),
        "generated_candidate_count": (
            len(index_outputs["geometry_candidates.json"])
            if index_outputs
            else None
        ),
        "kept_candidate_count": (
            len(index_outputs["pruned_candidates.json"])
            if index_outputs
            else None
        ),
        "true_group_proposal_coverage": (
            sum(row["proposal_generated"] for row in grouping["coverage"])
            / len(grouping["coverage"])
            if grouping and grouping["coverage"]
            else None
        ),
        "mixed_pool_metrics": grouping["metrics"] if grouping else None,
        "geometry_tiers": (
            grouping["geometry_tiers"] if grouping else None
        ),
        "known_group_pose": pose_statuses,
        "semantic_model_used": False,
        "ground_truth_scope": (
            "Folder membership only; exact original mates and original "
            "assembly poses are unavailable."
        ),
    }
    _write(output_root / "real_validation_summary.json", summary)
    rows = []
    for case_id in SOURCE_CASES:
        pose = pose_statuses.get(case_id, {})
        coverage = next(
            (
                row
                for row in grouping["coverage"]
                if row["true_group_id"] == f"REAL_{case_id}"
            ),
            {},
        ) if grouping else {}
        rows.append(
            {
                "case_id": case_id,
                "part_count": len(prepared["groups"][case_id]),
                "true_group_proposed": coverage.get(
                    "proposal_generated"
                ),
                "pose_worker_status": pose.get("worker_status"),
                "pose_status": pose.get("final_pose_status"),
                "physical_pose_valid": pose.get("physical_pose_valid"),
                "collision_count": pose.get("collision_count"),
                "projected_face_comparisons": pose.get(
                    "projected_face_comparisons"
                ),
                "unsolved_parts": "|".join(
                    pose.get("unsolved_parts") or []
                ),
            }
        )
    with (output_root / "real_case_results.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root", default=str(HERE)
    )
    parser.add_argument(
        "--output",
        default=str(HERE / "real_validation_20260703"),
    )
    parser.add_argument("--feature-timeout", type=int, default=900)
    parser.add_argument("--index-timeout", type=int, default=900)
    parser.add_argument("--pose-timeout", type=int, default=900)
    parser.add_argument(
        "--max-face-comparisons",
        type=int,
        default=50_000_000,
        help="Experiment safety limit; does not change algorithm scores.",
    )
    parser.add_argument(
        "--skip-pose", action="store_true"
    )
    args = parser.parse_args()
    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    pipeline_config_path = HERE / "configs" / "pool_pipeline.json"
    pipeline_config = _load(pipeline_config_path)
    conservative_config = _load(
        HERE / "configs" / "conservative_pipeline.json"
    )
    prepared = prepare_pool(source_root, output_root)
    features, feature_reports = extract_features_isolated(
        prepared["pool"],
        output_root,
        timeout_seconds=args.feature_timeout,
    )
    index_outputs = None
    grouping = None
    if len(features) == len(prepared["mapping"]):
        index_outputs, _ = build_index_isolated(
            prepared["pool"],
            output_root,
            pipeline_config_path,
            timeout_seconds=args.index_timeout,
            projected_comparisons=projected_face_comparisons(features),
            maximum_comparisons=args.max_face_comparisons,
        )
        if index_outputs is not None:
            grouping = run_grouping(
                prepared["pool"],
                index_outputs,
                pipeline_config,
                conservative_config,
            )
            _write(
                output_root / "true_group_coverage.json",
                grouping["coverage"],
            )
    pose_reports = []
    if not args.skip_pose:
        pose_reports = validate_known_groups(
            prepared["pool"],
            prepared["groups"],
            output_root,
            pipeline_config_path,
            features,
            timeout_seconds=args.pose_timeout,
            maximum_comparisons=args.max_face_comparisons,
        )
    summary = summarize(
        prepared,
        feature_reports,
        index_outputs,
        grouping,
        pose_reports,
        features,
        output_root,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
