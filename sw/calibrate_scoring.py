"""Calibrate match-score thresholds against synthetic ground truth."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from constraints import match_features
from features import extract_features
from match_scoring import score_matches
from run_cases import discover_cases, _case_id


def _key(item):
    parts = item.get("parts") or [item.get("part_a"), item.get("part_b")]
    return tuple(sorted(str(part) for part in parts)), item.get("type")


def _metrics(items, threshold, field="score"):
    selected = [item for item in items if item[field] >= threshold]
    tp = sum(item["label"] for item in selected)
    fp = len(selected) - tp
    positives = sum(item["label"] for item in items)
    fn = positives - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": round(threshold, 2),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _fit_platt(items, iterations=4000, learning_rate=0.2, l2=1e-3):
    """Fit p(true)=sigmoid(a*score+b) without an ML dependency."""
    positives = sum(item["label"] for item in items)
    negatives = len(items) - positives
    if not items or positives == 0 or negatives == 0:
        return 1.0, 0.0
    a = 0.0
    b = math.log(positives / negatives)
    count = len(items)
    for _ in range(iterations):
        grad_a = l2 * a
        grad_b = 0.0
        for item in items:
            value = max(-40.0, min(40.0, a * item["score"] + b))
            probability = 1.0 / (1.0 + math.exp(-value))
            error = probability - item["label"]
            grad_a += error * item["score"] / count
            grad_b += error / count
        a -= learning_rate * grad_a
        b -= learning_rate * grad_b
    return a, b


def calibrate(dataset_root):
    observations = []
    cases = discover_cases(dataset_root)
    for case in cases:
        gt_path = case / "gt.json"
        if not gt_path.is_file():
            continue
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        truth = {_key(item) for item in gt.get("true_mates", [])}
        files = sorted(
            path for path in case.iterdir()
            if path.is_file() and path.suffix.lower() in {".step", ".stp"}
            and not path.name.lower().startswith("assembly")
        )
        features = {path.name: extract_features(str(path)) for path in files}
        scored = score_matches(match_features(features), features)
        for item in scored:
            observations.append({
                "case_id": _case_id(case),
                "group_size": gt.get("group_size", len(files)),
                "parts": "|".join(sorted(item["parts"])),
                "type": item["type"],
                "score": float(item["score"]),
                "confidence": item["confidence"],
                "label": int(_key(item) in truth),
            })
    sweep = [_metrics(observations, index / 100) for index in range(0, 101, 5)]
    best = max(sweep, key=lambda row: (row["f1"], row["recall"], row["threshold"]))
    raw_brier = (
        sum((item["score"] - item["label"]) ** 2 for item in observations)
        / len(observations)
        if observations else None
    )
    platt_a, platt_b = _fit_platt(observations)
    for item in observations:
        value = max(-40.0, min(40.0, platt_a * item["score"] + platt_b))
        item["calibrated_probability"] = 1.0 / (1.0 + math.exp(-value))
    calibrated_sweep = [
        _metrics(observations, index / 100, "calibrated_probability")
        for index in range(0, 101, 5)
    ]
    calibrated_best = max(
        calibrated_sweep,
        key=lambda row: (row["f1"], row["recall"], row["threshold"]),
    )
    calibrated_brier = (
        sum(
            (item["calibrated_probability"] - item["label"]) ** 2
            for item in observations
        ) / len(observations)
        if observations else None
    )
    bins = []
    for low in (0.0, 0.2, 0.4, 0.6, 0.8):
        values = [
            item for item in observations
            if low <= item["score"] < low + 0.2 or (low == 0.8 and item["score"] == 1.0)
        ]
        bins.append({
            "score_min": low,
            "score_max": low + 0.2,
            "count": len(values),
            "mean_score": (
                sum(item["score"] for item in values) / len(values) if values else None
            ),
            "empirical_positive_rate": (
                sum(item["label"] for item in values) / len(values) if values else None
            ),
        })
    return observations, sweep, {
        "num_cases": len(cases),
        "num_candidates": len(observations),
        "num_positive_candidates": sum(item["label"] for item in observations),
        "recommended_threshold": best["threshold"],
        "best_threshold_metrics": best,
        "raw_brier_score": raw_brier,
        "platt_scaling": {"coefficient": platt_a, "intercept": platt_b},
        "recommended_calibrated_probability_threshold": calibrated_best["threshold"],
        "best_calibrated_threshold_metrics": calibrated_best,
        "calibrated_brier_score": calibrated_brier,
        "calibration_bins": bins,
    }, calibrated_sweep


def _csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    root = Path(args.dataset_root).resolve()
    output = Path(args.output_dir).resolve() if args.output_dir else root / "calibration"
    output.mkdir(parents=True, exist_ok=True)
    observations, sweep, report, calibrated_sweep = calibrate(root)
    _csv(output / "scored_candidates.csv", observations)
    _csv(output / "threshold_sweep.csv", sweep)
    _csv(output / "calibrated_threshold_sweep.csv", calibrated_sweep)
    (output / "calibration_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
