"""Evaluate the released JoinABLe checkpoint on official preprocessed samples."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from joinable_compat import (
    DEFAULT_CHECKPOINT,
    DEFAULT_DATA_ROOT,
    batch_to_device,
    build_model,
    collate_samples,
    environment_summary,
    load_checkpoint,
    load_legacy_samples,
    top_k_hits,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--limit", type=int, default=200, help="0 evaluates the full test cache"
    )
    parser.add_argument("--cpu-compare", type=int, default=10)
    parser.add_argument("--max-node-count", type=int, default=950)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Use CPU for a deterministic full-set compatibility audit.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("official_inference_report.json"),
    )
    return parser.parse_args()


def main() -> int:
    cli = parse_args()
    checkpoint, official_args = load_checkpoint(cli.checkpoint)
    model = build_model(checkpoint, official_args)
    samples, data_audit = load_legacy_samples(
        cli.data_root / "test.pickle",
        limit=cli.limit,
        max_node_count=cli.max_node_count,
        label_scheme="test",
        selection="first",
    )
    if not samples:
        raise RuntimeError("No official test samples were converted")

    cpu_logits: dict[str, torch.Tensor] = {}
    model.eval()
    cpu_started = time.perf_counter()
    with torch.inference_mode():
        for sample in samples[: cli.cpu_compare]:
            batch = collate_samples([sample])
            cpu_logits[sample.name] = model(*batch).detach().cpu()
    cpu_seconds = time.perf_counter() - cpu_started

    if cli.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    if cli.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda:0" if cli.device == "cuda" else "cpu")
    model.to(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    hit_counts = {"top_1": 0, "top_5": 0, "top_10": 0}
    per_sample = []
    max_cpu_cuda_abs_diff = 0.0
    top_10_order_match_count = 0
    missing_positive_count = 0
    started = time.perf_counter()

    with torch.inference_mode():
        for index, sample in enumerate(samples):
            batch = batch_to_device(collate_samples([sample]), device)
            logits = model(*batch)
            labels = batch[2].edge_attr.long()
            if int(labels.sum().item()) == 0:
                missing_positive_count += 1
            hits = top_k_hits(logits, labels)
            for key, hit in hits.items():
                hit_counts[key] += int(hit)

            record = {
                "file": sample.name,
                "graph1_nodes": int(batch[0].num_nodes),
                "graph2_nodes": int(batch[1].num_nodes),
                "candidate_entity_pairs": int(logits.numel()),
                "positive_entity_pairs": int(labels.sum().item()),
                **hits,
            }
            if sample.name in cpu_logits:
                gpu_logits = logits.detach().cpu()
                diff = float((gpu_logits - cpu_logits[sample.name]).abs().max())
                max_cpu_cuda_abs_diff = max(max_cpu_cuda_abs_diff, diff)
                k = min(10, gpu_logits.numel())
                order_match = bool(
                    torch.equal(
                        torch.topk(gpu_logits, k=k).indices,
                        torch.topk(cpu_logits[sample.name], k=k).indices,
                    )
                )
                top_10_order_match_count += int(order_match)
                record["cpu_cuda_max_abs_logit_diff"] = diff
                record["cpu_cuda_top_10_order_match"] = order_match
            per_sample.append(record)
            if (index + 1) % 100 == 0:
                print(f"evaluated {index + 1}/{len(samples)}")

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    sample_count = len(samples)
    report = {
        "schema_version": "1.0.0",
        "purpose": "Official JoinABLe checkpoint compatibility inference",
        "checkpoint": {
            "path": str(cli.checkpoint),
            "epoch": checkpoint.get("epoch"),
            "global_step": checkpoint.get("global_step"),
            "pytorch_lightning_version": checkpoint.get(
                "pytorch-lightning_version"
            ),
            "state_dict_tensor_count": len(checkpoint["state_dict"]),
            "strict_weight_load": True,
        },
        "environment": environment_summary(),
        "compatibility_adapter": {
            "official_checkpoint_modified": False,
            "official_preprocessed_pickle_modified": False,
            "legacy_pyg_data_migrated_in_memory": True,
            "feature_set": official_args.input_features,
            "test_positive_labels": ["Joint", "JointEquivalent"],
        },
        "data_audit": data_audit,
        "evaluation": {
            "device": str(device),
            "sample_count": sample_count,
            "top_1_accuracy": hit_counts["top_1"] / sample_count,
            "top_5_recall": hit_counts["top_5"] / sample_count,
            "top_10_recall": hit_counts["top_10"] / sample_count,
            "missing_positive_sample_count": missing_positive_count,
            "elapsed_seconds": elapsed,
            "samples_per_second": sample_count / elapsed,
            "gpu_peak_allocated_mib": (
                torch.cuda.max_memory_allocated() / 1024**2
                if device.type == "cuda"
                else None
            ),
        },
        "cpu_cuda_consistency": {
            "compared_sample_count": min(cli.cpu_compare, sample_count),
            "cpu_elapsed_seconds": cpu_seconds,
            "max_abs_logit_difference": max_cpu_cuda_abs_diff,
            "top_10_exact_order_match_count": top_10_order_match_count,
        },
        "samples": per_sample,
        "limitations": [
            "This evaluates joint-entity ranking on known part pairs; it is not mixed-pool grouping.",
            "A correct joint entity prediction does not by itself prove a collision-free assembly pose.",
            "The modern runtime is a compatibility reproduction, not the original CUDA 10.2 environment.",
        ],
    }
    write_json(cli.output, report)
    print(f"report: {cli.output}")
    print(
        "top-1/top-5/top-10: "
        f"{report['evaluation']['top_1_accuracy']:.4f}/"
        f"{report['evaluation']['top_5_recall']:.4f}/"
        f"{report['evaluation']['top_10_recall']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
