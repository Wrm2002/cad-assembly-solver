"""Prepare, run, and evaluate a leakage-controlled Qwen-VL calibration set.

The input manifest contains anonymous images and production-available context.
Truth labels and geometry baselines live in a separate evaluation-only file.
No live provider call occurs unless ``--mode live`` is explicitly supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from multimodal_reviewer import QwenVLReviewer
from render_parts_tray import render_parts_tray


SPLITS = {1: "calibration", 2: "validation", 3: "test"}


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _sample_id(case_id: str, subject: str) -> str:
    digest = hashlib.sha256(
        f"{case_id}:{subject}".encode("utf-8")
    ).hexdigest()[:16]
    return f"MM_{digest}"


def _positive_geometry_scores(pools_root: Path) -> dict[tuple[int, str], float]:
    scores = {}
    for pool in sorted(pools_root.glob("functional_pool_*")):
        variant = int(pool.name.rsplit("_", 1)[-1])
        gt = _load(pool / "pool_gt.json")
        proposals = _load(pool / "grouping" / "group_proposals.json")
        proposal_by_parts = {
            tuple(sorted(row["parts"])): row for row in proposals
        }
        for group in gt["true_groups"]:
            proposal = proposal_by_parts.get(
                tuple(sorted(group["parts"]))
            )
            if proposal:
                scores[(variant, group["assembly_family"])] = float(
                    proposal["geometry_score"]
                )
    return scores


def prepare(
    dataset_root: Path,
    pools_root: Path,
    forced_negative_audit: Path,
    output_root: Path,
    *,
    force: bool,
) -> dict[str, Any]:
    if output_root.exists():
        if not force:
            raise FileExistsError(
                f"{output_root} exists; use --force only before any live run"
            )
        shutil.rmtree(output_root)
    images = output_root / "images"
    images.mkdir(parents=True)
    positive_scores = _positive_geometry_scores(pools_root)
    negative_rows = _load(forced_negative_audit)["records"]
    negative_scores = {
        (row["case_id"], row["negative_id"]): float(
            row["geometry_score"]
        )
        for row in negative_rows
    }
    inputs = []
    labels = []

    for metadata_path in sorted(dataset_root.glob("*/metadata.json")):
        case_dir = metadata_path.parent
        metadata = _load(metadata_path)
        variant = int(metadata["case_id"].rsplit("_", 1)[-1])
        split = SPLITS[variant]
        positive_id = _sample_id(metadata["case_id"], "positive")
        positive_paths = [
            case_dir / part["file"] for part in metadata["parts"]
        ]
        positive_image = images / f"{positive_id}.png"
        render_parts_tray(
            positive_paths,
            [f"part_{index:02d}" for index in range(1, len(positive_paths) + 1)],
            positive_image,
        )
        inputs.append(
            {
                "sample_id": positive_id,
                "split": split,
                "part_count": len(positive_paths),
                "image": str(positive_image.resolve()),
                "text_context": (
                    f"Anonymous candidate containing {len(positive_paths)} "
                    "STEP parts. Judge functional assembly plausibility from "
                    "visible geometry only. Abstain when function is unclear."
                ),
                "input_policy": "anonymous_geometry_image_only_v1",
            }
        )
        labels.append(
            {
                "sample_id": positive_id,
                "split": split,
                "case_id": metadata["case_id"],
                "assembly_family": metadata["assembly_family"],
                "label": 1,
                "subject_type": "positive",
                "geometry_baseline_score": positive_scores.get(
                    (variant, metadata["assembly_family"])
                ),
            }
        )

        positive_by_id = {
            part["part_id"]: case_dir / part["file"]
            for part in metadata["parts"]
        }
        negative_by_id = {
            part["part_id"]: case_dir / part["file"]
            for part in metadata["negative_parts"]
        }
        for negative in metadata["negative_groups"]:
            sample_id = _sample_id(
                metadata["case_id"], negative["negative_id"]
            )
            paths = [
                positive_by_id.get(part_id)
                or negative_by_id.get(part_id)
                for part_id in negative["parts"]
            ]
            if any(path is None for path in paths):
                raise ValueError(
                    f"unresolved negative parts:{metadata['case_id']}:"
                    f"{negative['negative_id']}"
                )
            image = images / f"{sample_id}.png"
            render_parts_tray(
                paths,
                [f"part_{index:02d}" for index in range(1, len(paths) + 1)],
                image,
            )
            inputs.append(
                {
                    "sample_id": sample_id,
                    "split": split,
                    "part_count": len(paths),
                    "image": str(image.resolve()),
                    "text_context": (
                        f"Anonymous candidate containing {len(paths)} STEP "
                        "parts. Judge functional assembly plausibility from "
                        "visible geometry only. Abstain when function is unclear."
                    ),
                    "input_policy": "anonymous_geometry_image_only_v1",
                }
            )
            labels.append(
                {
                    "sample_id": sample_id,
                    "split": split,
                    "case_id": metadata["case_id"],
                    "assembly_family": metadata["assembly_family"],
                    "label": 0,
                    "subject_type": negative["negative_type"],
                    "geometry_baseline_score": negative_scores.get(
                        (metadata["case_id"], negative["negative_id"])
                    ),
                }
            )

    input_manifest = {
        "schema_version": "1.0.0",
        "artifact_role": "provider_input_no_truth",
        "sample_count": len(inputs),
        "input_policy": "anonymous_geometry_image_only_v1",
        "part_roles_in_provider_input": False,
        "assembly_family_in_provider_input": False,
        "functional_relations_in_provider_input": False,
        "samples": inputs,
    }
    truth = {
        "schema_version": "1.0.0",
        "artifact_role": "evaluation_only_never_provider_input",
        "sample_count": len(labels),
        "labels": labels,
    }
    _write(output_root / "provider_inputs.json", input_manifest)
    _write(output_root / "evaluation_only_labels.json", truth)
    lock_files = [
        output_root / "provider_inputs.json",
        output_root / "evaluation_only_labels.json",
        *sorted(images.glob("*.png")),
    ]
    lock = {
        "schema_version": "1.0.0",
        "policy": "freeze_before_any_live_provider_call",
        "files": {
            str(path.relative_to(output_root)).replace("\\", "/"): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in lock_files
        },
    }
    _write(output_root / "calibration_lock.json", lock)
    return input_manifest


def review(
    output_root: Path,
    config_path: Path,
    *,
    mode: str,
) -> dict[str, Any]:
    config = _load(config_path).get("multimodal_review", {})
    inputs = _load(output_root / "provider_inputs.json")
    reviewer = QwenVLReviewer(config, output_root / "cache")
    records = []
    for sample in inputs["samples"]:
        record = reviewer.review(
            sample["sample_id"],
            [sample["image"]],
            sample["text_context"],
            mode=mode,
        )
        records.append(
            {
                "sample_id": sample["sample_id"],
                "split": sample["split"],
                "input_policy": sample["input_policy"],
                **record,
            }
        )
    payload = {
        "schema_version": "1.0.0",
        "provider": "qwen-vl",
        "mode": mode,
        "provider_call_authorized": mode == "live",
        "sample_count": len(records),
        "records": records,
    }
    _write(output_root / "provider_decisions.json", payload)
    return payload


def _auc(labels: list[int], scores: list[float]) -> float | None:
    positives = [score for label, score in zip(labels, scores) if label == 1]
    negatives = [score for label, score in zip(labels, scores) if label == 0]
    if not positives or not negatives:
        return None
    wins = sum(
        1.0 if positive > negative else 0.5 if positive == negative else 0.0
        for positive in positives
        for negative in negatives
    )
    return wins / (len(positives) * len(negatives))


def _split_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [int(row["label"]) for row in rows]
    scores = [float(row["semantic_score"]) for row in rows]
    geometry_rows = [
        row for row in rows if row["geometry_baseline_score"] is not None
    ]
    semantic_brier = (
        sum((score - label) ** 2 for score, label in zip(scores, labels))
        / len(rows)
        if rows
        else None
    )
    geometry_brier = (
        sum(
            (
                float(row["geometry_baseline_score"])
                - int(row["label"])
            )
            ** 2
            for row in geometry_rows
        )
        / len(geometry_rows)
        if geometry_rows
        else None
    )
    auto_accepted = [
        row
        for row in rows
        if row["verdict"] == "accept"
        and float(row["semantic_score"]) >= 0.75
        and float(row["confidence"]) >= 0.5
    ]
    true_accepts = sum(int(row["label"]) == 1 for row in auto_accepted)
    false_accepts = len(auto_accepted) - true_accepts
    return {
        "sample_count": len(rows),
        "positive_count": sum(labels),
        "negative_count": len(labels) - sum(labels),
        "semantic_auc": _auc(labels, scores),
        "semantic_brier": semantic_brier,
        "geometry_baseline_brier": geometry_brier,
        "brier_improves_over_geometry": (
            semantic_brier < geometry_brier
            if semantic_brier is not None and geometry_brier is not None
            else False
        ),
        "verdict_counts": dict(
            Counter(row["verdict"] for row in rows)
        ),
        "auto_accept_count": len(auto_accepted),
        "auto_accept_precision": (
            true_accepts / len(auto_accepted) if auto_accepted else None
        ),
        "false_positive_count": false_accepts,
    }


def evaluate(output_root: Path) -> dict[str, Any]:
    labels = {
        row["sample_id"]: row
        for row in _load(
            output_root / "evaluation_only_labels.json"
        )["labels"]
    }
    decisions = _load(output_root / "provider_decisions.json")
    rows = []
    for record in decisions["records"]:
        truth = labels[record["sample_id"]]
        decision = record["decision"]
        rows.append(
            {
                **truth,
                "verdict": decision["verdict"],
                "semantic_score": decision["plausibility_score"],
                "confidence": decision["confidence"],
            }
        )
    by_split = {
        split: _split_metrics(
            [row for row in rows if row["split"] == split]
        )
        for split in ("calibration", "validation", "test")
    }
    test = by_split["test"]
    all_verdicts = {
        row["verdict"] for row in rows
    }
    gate_reasons = []
    if test["semantic_auc"] is None or test["semantic_auc"] < 0.70:
        gate_reasons.append("test_semantic_auc_below_0.70")
    if not test["brier_improves_over_geometry"]:
        gate_reasons.append("test_brier_not_better_than_geometry")
    if not {"accept", "reject", "abstain"} <= all_verdicts:
        gate_reasons.append("accept_reject_abstain_not_all_observed")
    if test["false_positive_count"] != 0:
        gate_reasons.append("test_false_positive_count_nonzero")
    if (
        test["auto_accept_precision"] is None
        or test["auto_accept_precision"] < 0.90
    ):
        gate_reasons.append("test_auto_accept_precision_not_established")
    report = {
        "schema_version": "1.0.0",
        "artifact_role": "evaluation_only_calibration_gate",
        "provider": "qwen-vl",
        "provider_mode": decisions["mode"],
        "by_split": by_split,
        "all_verdict_counts": dict(Counter(row["verdict"] for row in rows)),
        "semantic_reranking_enabled": False,
        "calibration_gate_passed": not gate_reasons,
        "gate_failure_reasons": gate_reasons,
        "decision": (
            "eligible_for_separate_human_approval"
            if not gate_reasons
            else "explanation_only"
        ),
    }
    _write(output_root / "multimodal_calibration_report.json", report)
    (output_root / "multimodal_gate_decision.md").write_text(
        "\n".join(
            [
                "# Multimodal Semantic Calibration Gate",
                "",
                f"- Provider mode: {decisions['mode']}",
                f"- Gate passed: {report['calibration_gate_passed']}",
                "- Semantic reranking enabled: false",
                f"- Failure reasons: {gate_reasons}",
                "",
                "Even a passing report requires a separate explicit approval "
                "before any production decision can change.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return report


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action",
        choices=("prepare", "review", "evaluate", "all"),
        default="all",
    )
    parser.add_argument(
        "--mode",
        choices=("off", "cache_only", "live"),
        default="off",
    )
    parser.add_argument(
        "--dataset-root",
        default=str(here / "data" / "functional_dataset_v1"),
    )
    parser.add_argument(
        "--pools-root",
        default=str(here / "data" / "functional_mixed_pools_v1"),
    )
    parser.add_argument(
        "--forced-negative-audit",
        default=str(
            here
            / "data"
            / "functional_results"
            / "forced_hard_negative_audit.json"
        ),
    )
    parser.add_argument(
        "--output-root",
        default=str(here / "data" / "multimodal_calibration_v1"),
    )
    parser.add_argument(
        "--config",
        default=str(here / "configs" / "pool_pipeline.json"),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_root).resolve()
    if args.action in {"prepare", "all"}:
        prepare(
            Path(args.dataset_root).resolve(),
            Path(args.pools_root).resolve(),
            Path(args.forced_negative_audit).resolve(),
            output,
            force=args.force,
        )
    if args.action in {"review", "all"}:
        review(
            output,
            Path(args.config).resolve(),
            mode=args.mode,
        )
    report = None
    if args.action in {"evaluate", "all"}:
        report = evaluate(output)
    print(
        json.dumps(
            report
            or {
                "action": args.action,
                "output_root": str(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
