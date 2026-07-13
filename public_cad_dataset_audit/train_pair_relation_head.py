"""Train a lightweight pair-level relation head on Fusion360 mappings.

This is the Step-2 bridge between Fusion360 data engineering and geometric
pose validation.  It is intentionally small and auditable:

- inputs are only Fusion360 train/dev/test JSONL rows;
- SolidWorks external labels are never read;
- the model predicts direct connection and the five common relation labels;
- JoinABLe remains the interface-localization tool, while this head predicts
  pair-level relation labels from structured metadata/geometry evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer


LABELS = ["coaxial", "clearance", "planar_mate", "planar_align", "pocket_mate"]
TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_+-]{1,}")


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
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _tokenize(values: list[str | None]) -> list[str]:
    tokens = []
    for value in values:
        for token in TOKEN_RE.findall(str(value or "").lower()):
            tokens.append(token[:40])
    return tokens


def featurize(row: dict[str, Any]) -> dict[str, float]:
    features: dict[str, float] = {}
    for relation_type in row.get("source_relation_types") or []:
        features[f"source_relation_type={relation_type}"] = 1.0
    for relation_kind in row.get("source_relation_kinds") or []:
        features[f"source_relation_kind={relation_kind}"] = 1.0
    for token in _tokenize(row.get("part_names") or []):
        features[f"part_token={token}"] = features.get(f"part_token={token}", 0.0) + 1.0
    for token in _tokenize(row.get("part_name_evidence") or []):
        features[f"evidence_token={token}"] = features.get(f"evidence_token={token}", 0.0) + 1.0
    paths = row.get("solidworks_compatible_geometry_paths") or []
    features["geometry_path_count"] = float(len([path for path in paths if path]))
    features["all_step_geometry_exists"] = float(
        bool(row.get("solidworks_compatible_geometry_exists"))
    )
    unavailable = row.get("unavailable_fields") or []
    for field in unavailable:
        features[f"unavailable={field}"] = 1.0
    features["unavailable_count"] = float(len(unavailable))
    features["is_closed_world_negative"] = float(not bool(row.get("direct_connection")))
    return features


def multilabel_targets(rows: list[dict[str, Any]]) -> list[tuple[str, ...]]:
    targets = []
    for row in rows:
        if row.get("direct_connection"):
            targets.append(tuple(label for label in row.get("mapped_relation_types") or [] if label in LABELS))
        else:
            targets.append(tuple())
    return targets


def direct_targets(rows: list[dict[str, Any]]) -> list[int]:
    return [1 if row.get("direct_connection") else 0 for row in rows]


def safe_auc(y_true: list[int], scores: list[float]) -> float | None:
    positives = [(score, label) for score, label in zip(scores, y_true) if label == 1]
    negatives = [(score, label) for score, label in zip(scores, y_true) if label == 0]
    if not positives or not negatives:
        return None
    wins = ties = 0.0
    for ps, _ in positives:
        for ns, _ in negatives:
            if ps > ns:
                wins += 1.0
            elif ps == ns:
                ties += 1.0
    return (wins + 0.5 * ties) / (len(positives) * len(negatives))


def predict_positive_probability(model: Pipeline, rows: list[dict[str, Any]]) -> list[float]:
    probabilities = model.predict_proba([featurize(row) for row in rows])
    return [float(row[1]) for row in probabilities]


def label_metrics(y_true, y_pred, label_names: list[str]) -> dict[str, Any]:
    per_label = {}
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=None,
        zero_division=0,
    )
    for index, label in enumerate(label_names):
        per_label[label] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
    return {
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_label": per_label,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-iter", type=int, default=250)
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = read_jsonl(input_dir / "fusion360_train.jsonl")
    dev_rows = read_jsonl(input_dir / "fusion360_dev.jsonl")
    test_rows = read_jsonl(input_dir / "fusion360_test.jsonl")
    train_x = [featurize(row) for row in train_rows]
    dev_x = [featurize(row) for row in dev_rows]
    test_x = [featurize(row) for row in test_rows]

    direct_model = Pipeline([
        ("vectorizer", DictVectorizer(sparse=True)),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(48,),
            activation="relu",
            alpha=0.0005,
            max_iter=args.max_iter,
            random_state=20260708,
            early_stopping=False,
            n_iter_no_change=20,
        )),
    ])
    direct_model.fit(train_x, direct_targets(train_rows))

    mlb = MultiLabelBinarizer(classes=LABELS)
    train_y_labels = mlb.fit_transform(multilabel_targets(train_rows))
    dev_y_labels = mlb.transform(multilabel_targets(dev_rows))
    test_y_labels = mlb.transform(multilabel_targets(test_rows))
    relation_model = Pipeline([
        ("vectorizer", DictVectorizer(sparse=True)),
        ("ovr_mlp", OneVsRestClassifier(MLPClassifier(
            hidden_layer_sizes=(64,),
            activation="relu",
            alpha=0.0008,
            max_iter=args.max_iter,
            random_state=20260708,
            early_stopping=False,
            n_iter_no_change=20,
        ))),
    ])
    relation_model.fit(train_x, train_y_labels)

    def split_report(name: str, rows: list[dict[str, Any]], x, y_labels) -> dict[str, Any]:
        y_direct = direct_targets(rows)
        direct_pred = [int(value) for value in direct_model.predict(x)]
        direct_scores = predict_positive_probability(direct_model, rows)
        label_pred = relation_model.predict(x)
        positive_true = [row for row in rows if row.get("direct_connection")]
        relation_support = Counter(
            label for row in positive_true for label in (row.get("mapped_relation_types") or [])
        )
        return {
            "sample_count": len(rows),
            "positive_count": sum(y_direct),
            "negative_count": len(rows) - sum(y_direct),
            "relation_support": dict(sorted(relation_support.items())),
            "direct_connection": {
                "auc": safe_auc(y_direct, direct_scores),
                "classification_report": classification_report(
                    y_direct,
                    direct_pred,
                    output_dict=True,
                    zero_division=0,
                ),
            },
            "relation_labels": label_metrics(y_labels, label_pred, LABELS),
        }

    metrics = {
        "schema_version": "1.0.0",
        "model_kind": "lightweight_pair_relation_head",
        "not_a_large_model": True,
        "solidworks_labels_used_for_training": False,
        "input_dir": str(input_dir),
        "labels": LABELS,
        "train": split_report("train", train_rows, train_x, train_y_labels),
        "dev": split_report("dev", dev_rows, dev_x, dev_y_labels),
        "test": split_report("test", test_rows, test_x, test_y_labels),
        "limitations": [
            "Features are structured Fusion360 metadata/contact evidence, not full B-Rep neural encoding.",
            "pocket_mate labels include mined candidates and require audit before high-confidence claims.",
            "JoinABLe is still used as the interface-localization network; this head predicts pair relation labels.",
        ],
    }
    write_json(output_dir / "pair_relation_model_metrics.json", metrics)

    with (output_dir / "pair_relation_head.pkl").open("wb") as handle:
        pickle.dump(
            {
                "direct_model": direct_model,
                "relation_model": relation_model,
                "label_binarizer": mlb,
                "labels": LABELS,
                "metrics": metrics,
            },
            handle,
        )

    predictions = []
    for split_name, rows in (("dev", dev_rows), ("test", test_rows)):
        x = [featurize(row) for row in rows]
        direct_scores = predict_positive_probability(direct_model, rows)
        label_scores = relation_model.predict_proba(x)
        label_pred = relation_model.predict(x)
        for row, direct_score, scores, pred in zip(rows, direct_scores, label_scores, label_pred):
            predictions.append({
                "sample_id": row["sample_id"],
                "split": split_name,
                "true_direct_connection": bool(row.get("direct_connection")),
                "predicted_direct_connection_score": float(direct_score),
                "true_relation_types": row.get("mapped_relation_types") or [],
                "predicted_relation_scores": {
                    label: float(scores[index]) for index, label in enumerate(LABELS)
                },
                "predicted_relation_types": [
                    label for index, label in enumerate(LABELS) if int(pred[index]) == 1
                ],
                "source_relation_types": row.get("source_relation_types") or [],
                "part_names": row.get("part_names") or [],
            })
    write_json(output_dir / "pair_relation_predictions_dev_test.json", predictions)

    report = f"""# Step 2 Pair-level 关系模型报告

## 结论

- 模型类型：轻量 pair-level relation head（sklearn MLP）
- 训练数据：Fusion360 train split
- 是否使用 SolidWorks 考试答案训练：否
- JoinABLe 角色：接口候选定位网络；本模型负责 pair-level 关系标签预测

## 数据规模

- train: {metrics['train']['sample_count']} samples / {metrics['train']['positive_count']} positives
- dev: {metrics['dev']['sample_count']} samples / {metrics['dev']['positive_count']} positives
- test: {metrics['test']['sample_count']} samples / {metrics['test']['positive_count']} positives

## 指标

- dev direct AUC: {metrics['dev']['direct_connection']['auc']}
- test direct AUC: {metrics['test']['direct_connection']['auc']}
- dev relation micro-F1: {metrics['dev']['relation_labels']['micro_f1']}
- test relation micro-F1: {metrics['test']['relation_labels']['micro_f1']}

## 边界

这不是最终大模型。它证明 pair-level 神经/学习式关系头的训练-预测-评估闭环已经打通。
后续应把输入从结构化字段升级为 B-Rep/JoinABLe graph embedding，并人工审核 pocket_mate 候选。
"""
    (output_dir / "pair_relation_model_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({
        "train_samples": metrics["train"]["sample_count"],
        "dev_auc": metrics["dev"]["direct_connection"]["auc"],
        "test_auc": metrics["test"]["direct_connection"]["auc"],
        "dev_relation_micro_f1": metrics["dev"]["relation_labels"]["micro_f1"],
        "test_relation_micro_f1": metrics["test"]["relation_labels"]["micro_f1"],
        "output_dir": str(output_dir),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
