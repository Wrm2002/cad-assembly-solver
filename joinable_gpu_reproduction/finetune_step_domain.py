"""Small, auditable JoinABLe fine-tune on mapped STEP interface labels.

This is a bounded domain-adaptation experiment, not a new grouping model.  The
model still emits only ranked interface candidates and remains shadow-only.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import torch
from torch_geometric.data import Batch, Data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cad_assembly_agent.tools.joinable_interface_predictor.pretrained_joinable_predictor import (  # noqa: E402
    body_to_data,
    make_joint_graph,
)
from joinable_gpu_reproduction.joinable_compat import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    batch_to_device,
    build_model,
    environment_summary,
    load_checkpoint,
    write_json,
)


DEFAULT_SUBSET = Path(
    r"D:\Model_match_public_data\fusion360_joint\domain_adapt_300"
)


@dataclass
class TrainingSample:
    sample_id: str
    graph_a: Data
    graph_b: Data
    joint_graph: Data
    exact_pairs: set[tuple[int, int]]
    equivalent_pairs: set[tuple[int, int]]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_manifest_path(
    raw_path: str,
    source_root: str,
    data_root: Path | None,
) -> Path:
    if data_root is None:
        return Path(raw_path)

    if "\\" in raw_path or (
        len(raw_path) >= 2 and raw_path[1] == ":"
    ):
        pure_path = PureWindowsPath(raw_path)
        pure_root = PureWindowsPath(source_root)
    else:
        pure_path = PurePosixPath(raw_path)
        pure_root = PurePosixPath(source_root)
    try:
        relative = pure_path.relative_to(pure_root)
    except ValueError as exc:
        raise ValueError(
            f"Manifest path is outside subset_root: {raw_path}"
        ) from exc
    return data_root.joinpath(*relative.parts)


def build_sample(
    row: dict[str, Any],
    source_root: str,
    data_root: Path | None,
) -> TrainingSample:
    graph_a_json = read_json(
        resolve_manifest_path(
            row["step_graph_a"], source_root, data_root
        )
    )
    graph_b_json = read_json(
        resolve_manifest_path(
            row["step_graph_b"], source_root, data_root
        )
    )
    extent_a = float(
        graph_a_json["metadata"]["checkpoint_pair_normalization_extent"]
    )
    extent_b = float(
        graph_b_json["metadata"]["checkpoint_pair_normalization_extent"]
    )
    pair_scale = 0.999999 / max(extent_a, extent_b)
    graph_a, _ = body_to_data(graph_a_json, pair_scale)
    graph_b, _ = body_to_data(graph_b_json, pair_scale)
    joint_graph = make_joint_graph(graph_a.num_nodes, graph_b.num_nodes)
    labels = torch.zeros(
        graph_a.num_nodes * graph_b.num_nodes, dtype=torch.long
    )
    for left, right in row["exact_positive_pairs"]:
        labels[int(left) * graph_b.num_nodes + int(right)] = 1
    joint_graph.edge_attr = labels
    return TrainingSample(
        sample_id=str(row["sample_id"]),
        graph_a=graph_a,
        graph_b=graph_b,
        joint_graph=joint_graph,
        exact_pairs={
            (int(left), int(right))
            for left, right in row["exact_positive_pairs"]
        },
        equivalent_pairs={
            (int(left), int(right))
            for left, right in row["equivalent_positive_pairs"]
        },
    )


def load_split_rows(
    manifest: dict[str, Any],
    split: str,
    training: bool,
    data_root: Path | None = None,
) -> list[TrainingSample]:
    rows = []
    source_root = str(manifest["subset_root"])
    for row in manifest["splits"][split]:
        if training and row["status"] != "training_and_evaluation":
            continue
        if not training and row["status"] not in {
            "training_and_evaluation",
            "evaluation_only_no_exact_mapping",
        }:
            continue
        if training and not row["exact_positive_pairs"]:
            continue
        if not training and not row["equivalent_positive_pairs"]:
            continue
        rows.append(build_sample(row, source_root, data_root))
    return rows


def collate(samples: list[TrainingSample]) -> tuple[Batch, Batch, Batch]:
    return (
        Batch.from_data_list([sample.graph_a for sample in samples]),
        Batch.from_data_list([sample.graph_b for sample in samples]),
        Batch.from_data_list([sample.joint_graph for sample in samples]),
    )


def evaluate(
    model: torch.nn.Module,
    samples: list[TrainingSample],
    device: torch.device,
) -> dict[str, Any]:
    ranks: dict[str, list[int | None]] = {
        "exact": [],
        "equivalent": [],
    }
    failures = []
    model.eval()
    with torch.inference_mode():
        for sample in samples:
            try:
                batch = batch_to_device(
                    collate([sample]), device
                )
                logits = model(*batch)
                order = torch.argsort(
                    logits, descending=True, stable=True
                ).detach().cpu().tolist()
                n2 = sample.graph_b.num_nodes
                exact_rank = None
                equivalent_rank = None
                for position, flat_index in enumerate(order, 1):
                    pair = (
                        int(flat_index) // n2,
                        int(flat_index) % n2,
                    )
                    if exact_rank is None and pair in sample.exact_pairs:
                        exact_rank = position
                    if (
                        equivalent_rank is None
                        and pair in sample.equivalent_pairs
                    ):
                        equivalent_rank = position
                    if (
                        (exact_rank is not None or not sample.exact_pairs)
                        and (
                            equivalent_rank is not None
                            or not sample.equivalent_pairs
                        )
                    ):
                        break
                if sample.exact_pairs:
                    ranks["exact"].append(exact_rank)
                if sample.equivalent_pairs:
                    ranks["equivalent"].append(equivalent_rank)
            except Exception as exc:
                failures.append(
                    {
                        "sample_id": sample.sample_id,
                        "reason": f"{type(exc).__name__}:{exc}",
                    }
                )
    def summarize(values: list[int | None]) -> dict[str, Any]:
        finite = [rank for rank in values if rank is not None]
        count = len(values)
        result = {
            "sample_count": count,
            "mean_positive_rank": (
                sum(finite) / len(finite) if finite else None
            ),
            "missing_positive_in_full_ranking": sum(
                rank is None for rank in values
            ),
        }
        for k in (1, 5, 10, 20):
            result[f"top_{k}_recall"] = (
                sum(
                    rank is not None and rank <= k for rank in values
                )
                / count
                if count
                else None
            )
        return result

    return {
        "total_loaded_sample_count": len(samples),
        "exact": summarize(ranks["exact"]),
        "equivalent": summarize(ranks["equivalent"]),
        "failure_count": len(failures),
        "failures": failures,
    }


def train_epoch(
    model: torch.nn.Module,
    samples: list[TrainingSample],
    official_args: object,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    model.train()
    indices = list(range(len(samples)))
    random.Random(seed).shuffle(indices)
    losses = []
    measured_samples = 0
    started = time.perf_counter()
    for start in range(0, len(indices), batch_size):
        selected = [
            samples[index]
            for index in indices[start : start + batch_size]
        ]
        if len(selected) < batch_size:
            continue
        batch = batch_to_device(collate(selected), device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(*batch)
        loss = model.compute_loss(official_args, logits, batch[2])
        if not torch.isfinite(loss):
            raise RuntimeError(f"non_finite_loss:{float(loss.detach().cpu())}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        measured_samples += len(selected)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {
        "sample_count": measured_samples,
        "step_count": len(losses),
        "mean_loss": sum(losses) / len(losses) if losses else None,
        "elapsed_seconds": elapsed,
        "samples_per_second": (
            measured_samples / elapsed if elapsed > 0 else None
        ),
    }


def score_tuple(metrics: dict[str, Any]) -> tuple[float, float, float]:
    exact = metrics["exact"]
    return (
        float(exact.get("top_10_recall") or 0.0),
        float(exact.get("top_5_recall") or 0.0),
        float(exact.get("top_1_recall") or 0.0),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_SUBSET / "domain_adaptation_manifest.json",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=(
            "Optional relocated subset root. Paths stored in the manifest "
            "are resolved relative to manifest.subset_root."
        ),
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument(
        "--train-scope",
        choices=("full", "post_only"),
        default="full",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("domain_finetune_results"),
    )
    args = parser.parse_args()

    torch.set_num_threads(max(1, args.torch_threads))
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.device == "auto":
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(
            "cuda:0" if args.device == "cuda" else "cpu"
        )
    manifest = read_json(args.manifest)
    train_samples = load_split_rows(
        manifest, "train", training=True, data_root=args.data_root
    )
    validation_samples = load_split_rows(
        manifest,
        "validation",
        training=False,
        data_root=args.data_root,
    )
    test_samples = load_split_rows(
        manifest,
        "test",
        training=False,
        data_root=args.data_root,
    )
    if not train_samples or not validation_samples or not test_samples:
        raise RuntimeError(
            "Mapped train/validation/test samples are required"
        )

    checkpoint, official_args = load_checkpoint(args.checkpoint)
    model = build_model(checkpoint, official_args).to(device)
    if args.train_scope == "post_only":
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith("post.")
    trainable_parameters = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad
    ]
    trainable_parameter_count = sum(
        parameter.numel() for parameter in trainable_parameters
    )
    total_parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_dir / "progress.json"
    baseline_validation = evaluate(model, validation_samples, device)
    baseline_test = evaluate(model, test_samples, device)
    optimizer = torch.optim.Adam(
        trainable_parameters, lr=args.learning_rate
    )
    best_score = score_tuple(baseline_validation)
    best_epoch = 0
    best_path = args.output_dir / "best_step_domain_state.pt"
    torch.save(
        {
            "state_dict": {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            },
            "epoch": 0,
            "validation": baseline_validation,
            "source_checkpoint": str(args.checkpoint),
        },
        best_path,
    )
    history = []
    epochs_without_improvement = 0

    for epoch in range(1, max(1, args.epochs) + 1):
        train_metrics = train_epoch(
            model,
            train_samples,
            official_args,
            optimizer,
            device,
            max(1, args.batch_size),
            args.seed + epoch,
        )
        validation_metrics = evaluate(
            model, validation_samples, device
        )
        current_score = score_tuple(validation_metrics)
        improved = current_score > best_score
        if improved:
            best_score = current_score
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "state_dict": {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    },
                    "epoch": epoch,
                    "validation": validation_metrics,
                    "source_checkpoint": str(args.checkpoint),
                },
                best_path,
            )
        else:
            epochs_without_improvement += 1
        row = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
            "improved": improved,
            "epochs_without_improvement": epochs_without_improvement,
        }
        history.append(row)
        write_json(
            progress_path,
            {
                "status": "training",
                "current_epoch": epoch,
                "requested_epochs": args.epochs,
                "best_epoch": best_epoch,
                "baseline_validation": baseline_validation,
                "latest": row,
            },
        )
        print(
            f"epoch {epoch}: loss={train_metrics['mean_loss']:.5f}, "
            f"val_exact_top10="
            f"{validation_metrics['exact']['top_10_recall']:.4f}, "
            f"best={best_epoch}",
            flush=True,
        )
        if epochs_without_improvement >= max(1, args.patience):
            break

    best_payload = torch.load(
        best_path, map_location="cpu", weights_only=False
    )
    model.load_state_dict(best_payload["state_dict"], strict=True)
    model.to(device)
    best_validation = evaluate(model, validation_samples, device)
    best_test = evaluate(model, test_samples, device)
    validation_top10_delta = (
        (best_validation["exact"]["top_10_recall"] or 0.0)
        - (baseline_validation["exact"]["top_10_recall"] or 0.0)
    )
    meaningful_validation_delta = max(
        0.03,
        2.0
        / max(1, baseline_validation["exact"]["sample_count"]),
    )
    meaningful_improvement = (
        validation_top10_delta >= meaningful_validation_delta
    )
    report = {
        "schema_version": "1.0.0",
        "purpose": "Bounded JoinABLe STEP-domain fine-tune",
        "policy": {
            "candidate_provider_only": True,
            "shadow_mode": True,
            "can_change_final_acceptance": False,
            "selection_metric": "validation top10, then top5, then top1",
            "selection_label_scope": "exact designer-selected entity pair",
        },
        "environment": environment_summary(),
        "device": str(device),
        "configuration": {
            "source_checkpoint": str(args.checkpoint),
            "manifest": str(args.manifest),
            "data_root": (
                str(args.data_root.resolve())
                if args.data_root is not None
                else None
            ),
            "epochs_requested": args.epochs,
            "epochs_completed": len(history),
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "patience": args.patience,
            "seed": args.seed,
            "torch_threads": args.torch_threads,
            "train_scope": args.train_scope,
            "trainable_parameter_count": trainable_parameter_count,
            "total_parameter_count": total_parameter_count,
        },
        "dataset": {
            "train_count": len(train_samples),
            "validation_count": len(validation_samples),
            "test_count": len(test_samples),
        },
        "baseline_validation": baseline_validation,
        "baseline_test": baseline_test,
        "best_epoch": best_epoch,
        "best_validation": best_validation,
        "best_test": best_test,
        "history": history,
        "best_checkpoint": str(best_path.resolve()),
        "rent_gpu_trigger": {
            "validation_top10_delta": validation_top10_delta,
            "required_meaningful_delta": meaningful_validation_delta,
            "small_scale_meaningfully_improves_validation_top10": (
                meaningful_improvement
            ),
            "full_scale_training_recommended": meaningful_improvement,
        },
        "failure_reasons": (
            baseline_validation["failures"]
            + baseline_test["failures"]
            + best_validation["failures"]
            + best_test["failures"]
        ),
        "unavailable_fields": [
            "functional_assembly_validity",
            "mixed_pool_grouping_quality",
        ],
    }
    write_json(args.output_dir / "finetune_report.json", report)
    write_json(
        progress_path,
        {
            "status": "complete",
            "best_epoch": best_epoch,
            "baseline_validation": baseline_validation,
            "best_validation": best_validation,
            "baseline_test": baseline_test,
            "best_test": best_test,
        },
    )
    print(json.dumps(report["rent_gpu_trigger"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
