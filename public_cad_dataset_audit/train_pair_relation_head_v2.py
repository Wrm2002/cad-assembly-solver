"""Step 2 lightweight relation head — optimized for large benchmark (a00_a01).

Strategy:
- Subsample training to 50k rows with stratified positive/negative ratio
- Use sparse features + SGDClassifier for scalability
- Full dev/test evaluation
"""

from __future__ import annotations

import argparse, json, math, pickle, random, re
from collections import Counter
from pathlib import Path
from typing import Any

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

LABELS = ["clearance", "coaxial", "planar_align", "planar_mate", "pocket_mate"]
TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_+-]{1,}")

random.seed(42)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _tokenize(values: list[str | None]) -> list[str]:
    tokens = []
    for value in values:
        for token in TOKEN_RE.findall(str(value or "").lower()):
            tokens.append(token[:40])
    return tokens


def featurize(row: dict[str, Any]) -> dict[str, float]:
    """Extract transferable features only (no source_relation_types/kind)."""
    features: dict[str, float] = {}
    # Limit token features to top keywords only
    top_keywords = {"screw", "bolt", "nut", "washer", "pin", "shaft", "bearing", "housing",
                    "plate", "bracket", "flange", "base", "cover", "body", "ring", "gear",
                    "piston", "cylinder", "valve", "spring", "rail", "slide", "guide",
                    "socket", "clip", "dovetail", "t-nut", "groove", "dowell"}
    for token in _tokenize(row.get("part_names") or []):
        if token in top_keywords:
            features[f"name_{token}"] = features.get(f"name_{token}", 0.0) + 1.0
    for token in _tokenize(row.get("part_name_evidence") or []):
        if token in top_keywords:
            features[f"ev_{token}"] = features.get(f"ev_{token}", 0.0) + 1.0
    paths = row.get("solidworks_compatible_geometry_paths") or []
    features["has_geometry"] = float(len([p for p in paths if p]) >= 2)
    features["all_step_ok"] = float(bool(row.get("solidworks_compatible_geometry_exists")))
    unavailable = row.get("unavailable_fields") or []
    features["unavail_count"] = float(len(unavailable))
    # Add a bias feature that activates only when geometry is available
    # This helps the model distinguish real pairs from metadata-only ones
    features["is_real_pair"] = float(
        bool(row.get("solidworks_compatible_geometry_exists"))
        and len(row.get("solidworks_compatible_geometry_paths") or []) >= 2
    )
    return features


def direct_targets(rows: list[dict[str, Any]]) -> list[int]:
    return [1 if row.get("direct_connection") else 0 for row in rows]


def multilabel_targets(rows: list[dict[str, Any]]) -> list[tuple[str, ...]]:
    targets = []
    for row in rows:
        if row.get("direct_connection"):
            targets.append(tuple(label for label in row.get("mapped_relation_types") or [] if label in LABELS))
        else:
            targets.append(tuple())
    return targets


def stratified_subsample(rows: list[dict[str, Any]], max_samples: int) -> list[dict[str, Any]]:
    pos = [r for r in rows if r.get("direct_connection")]
    neg = [r for r in rows if not r.get("direct_connection")]
    pos_ratio = len(pos) / len(rows) if rows else 0.1
    n_pos = min(len(pos), int(max_samples * pos_ratio))
    n_neg = min(len(neg), max_samples - n_pos)
    sampled = random.sample(pos, n_pos) + random.sample(neg, n_neg)
    random.shuffle(sampled)
    return sampled


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-train-samples", type=int, default=50000)
    parser.add_argument("--max-iter", type=int, default=200)
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {input_dir}...")
    train_all = read_jsonl(input_dir / "fusion360_train.jsonl")
    dev_rows = read_jsonl(input_dir / "fusion360_dev.jsonl")
    test_rows = read_jsonl(input_dir / "fusion360_test.jsonl")

    # Subsample training
    train_rows = stratified_subsample(train_all, args.max_train_samples)
    pos_count = sum(1 for r in train_rows if r.get("direct_connection"))

    print(f"Train: {len(train_rows)} (subsampled from {len(train_all)}, {pos_count} pos)")
    print(f"Dev:   {len(dev_rows)}")
    print(f"Test:  {len(test_rows)}")

    # Featurize
    print("Featurizing...")
    train_x = [featurize(row) for row in train_rows]
    dev_x = [featurize(row) for row in dev_rows]
    test_x = [featurize(row) for row in test_rows]

    # Direct edge model
    print("Training direct edge classifier...")
    direct_model = Pipeline([
        ("vectorizer", DictVectorizer(sparse=False)),
        ("classifier", SGDClassifier(loss="log_loss", max_iter=args.max_iter, random_state=42, n_jobs=-1)),
    ])
    train_y_direct = direct_targets(train_rows)
    direct_model.fit(train_x, train_y_direct)

    dev_y_direct = direct_targets(dev_rows)
    test_y_direct = direct_targets(test_rows)
    dev_pred_direct = direct_model.predict(dev_x)
    test_pred_direct = direct_model.predict(test_x)
    dev_proba_direct = direct_model.predict_proba(dev_x)[:, 1]
    test_proba_direct = direct_model.predict_proba(test_x)[:, 1]

    # Relation type model
    print("Training relation type classifier...")
    train_y_rel = multilabel_targets(train_rows)
    mlb = MultiLabelBinarizer(classes=sorted(LABELS))
    mlb.fit([tuple(LABELS)])
    train_y_rel_bin = mlb.transform(train_y_rel)

    rel_model = Pipeline([
        ("vectorizer", DictVectorizer(sparse=False)),
        ("classifier", OneVsRestClassifier(
            SGDClassifier(loss="log_loss", max_iter=args.max_iter, random_state=42, n_jobs=-1)
        )),
    ])
    rel_model.fit(train_x, train_y_rel_bin)

    dev_y_rel = multilabel_targets(dev_rows)
    test_y_rel = multilabel_targets(test_rows)
    dev_y_rel_bin = mlb.transform(dev_y_rel)
    test_y_rel_bin = mlb.transform(test_y_rel)
    dev_pred_rel_bin = rel_model.predict(dev_x)
    test_pred_rel_bin = rel_model.predict(test_x)

    # Metrics
    metrics = {
        "schema_version": "1.0.0",
        "model_kind": "lightweight_pair_relation_head_sgd",
        "solidworks_labels_used_for_training": False,
        "input_dir": str(input_dir),
        "train_samples": len(train_rows),
        "train_positives": pos_count,
        "labels": LABELS,
        "direct_edge": {
            "dev": {
                "classification_report": classification_report(dev_y_direct, dev_pred_direct, output_dict=True),
            },
            "test": {
                "classification_report": classification_report(test_y_direct, test_pred_direct, output_dict=True),
            },
        },
        "relation_labels": {},
    }

    # Per-label metrics
    for split_name, y_true, y_pred in [
        ("dev", dev_y_rel_bin, dev_pred_rel_bin),
        ("test", test_y_rel_bin, test_pred_rel_bin),
    ]:
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        per_label = {}
        for i, label in enumerate(LABELS):
            per_label[label] = {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
        micro = precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0)
        macro = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        metrics["relation_labels"][split_name] = {
            "micro_f1": float(micro[2]),
            "macro_f1": float(macro[2]),
            "per_label": per_label,
        }

    # Pocket mate specific analysis
    pocket_idx = LABELS.index("pocket_mate")
    for split_name, y_true, y_pred in [
        ("dev", dev_y_rel_bin, dev_pred_rel_bin),
        ("test", test_y_rel_bin, test_pred_rel_bin),
    ]:
        tp = int(((y_true[:, pocket_idx] == 1) & (y_pred[:, pocket_idx] == 1)).sum())
        fn = int(((y_true[:, pocket_idx] == 1) & (y_pred[:, pocket_idx] == 0)).sum())
        fp = int(((y_true[:, pocket_idx] == 0) & (y_pred[:, pocket_idx] == 1)).sum())
        metrics.setdefault("pocket_mate_detail", {})[split_name] = {
            "true_positive": tp,
            "false_negative": fn,
            "false_positive": fp,
            "recall": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
            "precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
        }

    write_json(output_dir / "pair_relation_model_metrics.json", metrics)

    # Save model
    with open(output_dir / "pair_relation_head.pkl", "wb") as f:
        pickle.dump({"direct_model": direct_model, "rel_model": rel_model, "mlb": mlb, "labels": LABELS}, f)

    # Report
    report = [
        "# Step 2 Pair-level 关系模型报告 (a00_a01)",
        "",
        f"- 模型类型：轻量 SGD pair relation head",
        f"- 训练数据：Fusion360 a00_a01 train（子采样 {len(train_rows)}/{len(train_all)}），{pos_count} positives",
        f"- 是否使用 SolidWorks 考试答案训练：否",
        "",
        "## 数据规模",
        f"- train: {len(train_rows)} (subsampled from {len(train_all)})",
        f"- dev: {len(dev_rows)}",
        f"- test: {len(test_rows)}",
        "",
        "## Direct Edge 指标",
        f"- dev: precision={metrics['direct_edge']['dev']['classification_report']['macro avg']['precision']:.4f}",
        f"- test: precision={metrics['direct_edge']['test']['classification_report']['macro avg']['precision']:.4f}",
        "",
        "## Relation Label 指标",
        f"- dev micro-F1: {metrics['relation_labels']['dev']['micro_f1']:.4f}",
        f"- test micro-F1: {metrics['relation_labels']['test']['micro_f1']:.4f}",
        "",
        "### Per-label F1",
    ]
    for label in LABELS:
        dev_f1 = metrics["relation_labels"]["dev"]["per_label"][label]["f1"]
        test_f1 = metrics["relation_labels"]["test"]["per_label"][label]["f1"]
        report.append(f"- {label}: dev={dev_f1:.4f}, test={test_f1:.4f}")

    report.extend([
        "",
        "## Pocket Mate 详情",
        f"- dev: recall={metrics['pocket_mate_detail']['dev']['recall']:.4f}, precision={metrics['pocket_mate_detail']['dev']['precision']:.4f}",
        f"- test: recall={metrics['pocket_mate_detail']['test']['recall']:.4f}, precision={metrics['pocket_mate_detail']['test']['precision']:.4f}",
        f"  TP={metrics['pocket_mate_detail']['test']['true_positive']}, FN={metrics['pocket_mate_detail']['test']['false_negative']}, FP={metrics['pocket_mate_detail']['test']['false_positive']}",
        "",
        "## 边界",
        "这不是最终大模型。subsampled SGD baseline 验证训练-评估闭环在 expanded benchmark 上是否可用。",
    ])

    (output_dir / "pair_relation_model_report.md").write_text("\n".join(report), encoding="utf-8")

    print("\nDone!")
    print(f"  Direct edge - dev F1: {metrics['direct_edge']['dev']['classification_report']['macro avg']['f1-score']:.4f}")
    print(f"  Direct edge - test F1: {metrics['direct_edge']['test']['classification_report']['macro avg']['f1-score']:.4f}")
    print(f"  Relation - dev micro-F1: {metrics['relation_labels']['dev']['micro_f1']:.4f}")
    print(f"  Relation - test micro-F1: {metrics['relation_labels']['test']['micro_f1']:.4f}")
    print(f"  Pocket mate - test recall: {metrics['pocket_mate_detail']['test']['recall']:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
