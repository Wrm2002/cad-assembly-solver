"""Bounded CUDA training benchmark using the released JoinABLe architecture."""

from __future__ import annotations

import argparse
import math
import statistics
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
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=30)
    parser.add_argument("--max-node-count", type=int, default=950)
    parser.add_argument("--official-train-size", type=int, default=13234)
    parser.add_argument(
        "--modes",
        default="fp32",
        help="Comma-separated benchmark modes: fp32,amp_fp16",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("cuda_training_benchmark.json"),
    )
    return parser.parse_args()


def make_batches(samples: list, batch_size: int) -> list[tuple]:
    usable = len(samples) - (len(samples) % batch_size)
    return [
        collate_samples(samples[index : index + batch_size])
        for index in range(0, usable, batch_size)
    ]


def run_mode(
    mode: str,
    checkpoint: dict,
    official_args: object,
    cpu_batches: list[tuple],
    warmup_steps: int,
    measure_steps: int,
) -> dict:
    device = torch.device("cuda:0")
    model = build_model(checkpoint, official_args).to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=official_args.lr)
    use_amp = mode == "amp_fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    times = []
    sample_count = 0
    loss_values = []
    total_steps = warmup_steps + measure_steps
    try:
        for step in range(total_steps):
            batch = batch_to_device(cpu_batches[step % len(cpu_batches)], device)
            batch_size = int(batch[2].num_graphs)
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=use_amp
            ):
                logits = model(*batch)
                loss = model.compute_loss(official_args, logits, batch[2])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            if step >= warmup_steps:
                times.append(elapsed)
                sample_count += batch_size
                loss_values.append(float(loss.detach().cpu()))
    except torch.cuda.OutOfMemoryError as exc:
        return {
            "mode": mode,
            "status": "cuda_out_of_memory",
            "reason": str(exc),
            "completed_measured_steps": len(times),
            "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "gpu_peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
        }
    except Exception as exc:
        return {
            "mode": mode,
            "status": "runtime_failed",
            "reason": f"{type(exc).__name__}: {exc}",
            "completed_measured_steps": len(times),
            "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "gpu_peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
        }

    total_seconds = sum(times)
    return {
        "mode": mode,
        "status": "success",
        "warmup_steps": warmup_steps,
        "measured_steps": len(times),
        "measured_samples": sample_count,
        "total_measured_seconds": total_seconds,
        "mean_step_seconds": statistics.mean(times),
        "median_step_seconds": statistics.median(times),
        "p95_step_seconds": sorted(times)[max(0, math.ceil(len(times) * 0.95) - 1)],
        "steps_per_second": len(times) / total_seconds,
        "samples_per_second": sample_count / total_seconds,
        "mean_loss": statistics.mean(loss_values),
        "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "gpu_peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
    }


def main() -> int:
    cli = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this environment")

    checkpoint, official_args = load_checkpoint(cli.checkpoint)
    # Loading the 16 GB train pickle solely for a speed sample would create
    # avoidable host-memory pressure.  Official test graphs have the same tensor
    # structure, so they are used for bounded forward/backward throughput.
    samples, data_audit = load_legacy_samples(
        cli.data_root / "test.pickle",
        limit=cli.samples,
        max_node_count=cli.max_node_count,
        label_scheme="train",
        selection="uniform",
    )
    cpu_batches = make_batches(samples, cli.batch_size)
    if not cpu_batches:
        raise RuntimeError("Not enough converted samples for one batch")

    requested_modes = [mode.strip() for mode in cli.modes.split(",") if mode.strip()]
    unsupported = sorted(set(requested_modes) - {"fp32", "amp_fp16"})
    if unsupported:
        raise ValueError(f"Unsupported modes: {unsupported}")
    modes = []
    for mode in requested_modes:
        print(f"benchmarking {mode}")
        modes.append(
            run_mode(
                mode,
                checkpoint,
                official_args,
                cpu_batches,
                cli.warmup_steps,
                cli.measure_steps,
            )
        )

    for result in modes:
        if result["status"] == "success":
            epoch_seconds = cli.official_train_size / result["samples_per_second"]
            result["estimated_official_epoch_minutes"] = epoch_seconds / 60
            result["estimated_100_epochs_hours"] = epoch_seconds * 100 / 3600
        else:
            result["estimated_official_epoch_minutes"] = None
            result["estimated_100_epochs_hours"] = None

    report = {
        "schema_version": "1.0.0",
        "purpose": "Bounded local JoinABLe CUDA training throughput benchmark",
        "environment": environment_summary(),
        "checkpoint": str(cli.checkpoint),
        "data_audit": data_audit,
        "benchmark_configuration": {
            "source_split": "official test cache used as representative graph tensors",
            "source_sample_count": len(samples),
            "batch_size": cli.batch_size,
            "max_node_count": cli.max_node_count,
            "official_train_size_for_estimate": cli.official_train_size,
            "reason_train_pickle_not_loaded": (
                "The official train pickle is about 16 GB and unnecessary for "
                "a bounded kernel/VRAM throughput measurement."
            ),
        },
        "results": modes,
        "estimate_limitations": [
            "The estimate excludes full-dataset pickle loading and validation.",
            "Graph sizes vary, so a complete epoch may differ from this bounded sample.",
            "Windows display processes consume part of the 8 GB VRAM.",
            "This is a speed/VRAM test, not a converged retraining experiment.",
            "The released model creates explicit FP32 buffers, so AMP requires "
            "an architecture compatibility change and is not the reference path.",
        ],
    }
    write_json(cli.output, report)
    print(f"report: {cli.output}")
    for result in modes:
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
