"""Run one OCCT command with the repository's crash-safety resource policy."""
from __future__ import annotations

import argparse
from pathlib import Path

from run_case5_visual_semantic_workflow import (
    Checkpoints,
    ResourcePolicy,
    parse_cpu_affinity,
    run_isolated,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--cpu-affinity", type=parse_cpu_affinity, default=parse_cpu_affinity("0-7"))
    parser.add_argument("--max-worker-memory-gb", type=float, default=10.0)
    parser.add_argument("--min-free-memory-gb", type=float, default=8.0)
    parser.add_argument("--cooldown-seconds", type=float, default=12.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("a worker command is required after --")

    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    run_isolated(
        command,
        stage=args.stage,
        checkpoint=Checkpoints(root, resume=True),
        stdout_path=logs / f"{args.stage}.stdout.log",
        stderr_path=logs / f"{args.stage}.stderr.log",
        timeout_seconds=args.timeout,
        resource_policy=ResourcePolicy(
            cpu_affinity=tuple(args.cpu_affinity),
            max_worker_memory_gb=args.max_worker_memory_gb,
            min_free_memory_gb=args.min_free_memory_gb,
            cooldown_seconds=args.cooldown_seconds,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
