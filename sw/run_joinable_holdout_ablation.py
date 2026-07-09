"""One-command analytic-only vs analytic+JoinABLe harder-holdout ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from audit_joinable_holdout import run as run_strict_audit
from candidate_recall_audit import run_audit as run_candidate_audit
from conservative_pipeline import run as run_conservative
from pool_index import index_pool
from prepare_joinable_pool_graphs import prepare as prepare_graphs


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
DEFAULT_JOINABLE_PYTHON = Path(r"D:\Model_match_envs\joinable_gpu\python.exe")
DEFAULT_REAL_REPORT = PROJECT_ROOT / "joinable_gpu_reproduction" / "rule_vs_pretrained_step_report.json"
BATCH_INFERENCE = (
    PROJECT_ROOT
    / "cad_assembly_agent"
    / "tools"
    / "joinable_interface_predictor"
    / "batch_mixed_pool_inference.py"
)
DOMAIN_EVALUATOR = (
    PROJECT_ROOT
    / "joinable_gpu_reproduction"
    / "evaluate_domain_holdout_union.py"
)
DEFAULT_DOMAIN_MANIFEST = Path(
    r"D:\Model_match_public_data\fusion360_joint\domain_adapt_2600"
    r"\domain_adaptation_manifest.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def source_files(root: Path) -> list[Path]:
    files = [root / "mixed_pool_manifest.json"]
    for pool in sorted(root.glob("holdout_pool_*")):
        files.extend(
            path
            for path in (
                pool / "pool_gt.json",
                pool / "pool_input.json",
                pool / "part_semantics.json",
            )
            if path.is_file()
        )
        files.extend(sorted((pool / "parts").glob("*.step")))
    return files


def copy_arm(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copy2(source / "mixed_pool_manifest.json", destination)
    for pool in sorted(source.glob("holdout_pool_*")):
        target = destination / pool.name
        target.mkdir()
        shutil.copytree(pool / "parts", target / "parts")
        for name in ("pool_gt.json", "pool_input.json", "part_semantics.json"):
            path = pool / name
            if path.is_file():
                shutil.copy2(path, target / name)


def run_batch(
    python: Path,
    mixed_root: Path,
    output: Path,
    device: str,
    top_k: int,
) -> float:
    command = [
        str(python),
        str(BATCH_INFERENCE),
        "--mixed-pool-root",
        str(mixed_root),
        "--output",
        str(output),
        "--top-k",
        str(top_k),
        "--device",
        device,
    ]
    started = time.perf_counter()
    subprocess.run(command, cwd=HERE, check=True)
    return time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=HERE / "data" / "functional_cad_holdout_pools_v1",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--joinable-python", type=Path, default=DEFAULT_JOINABLE_PYTHON)
    parser.add_argument("--real-report", type=Path, default=DEFAULT_REAL_REPORT)
    parser.add_argument(
        "--domain-manifest", type=Path, default=DEFAULT_DOMAIN_MANIFEST
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    source = args.source_root.resolve()
    output = args.output_root.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"output root must be absent or empty for a clean ablation: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    pipeline_config = HERE / "configs" / "pool_pipeline.json"
    conservative_config = HERE / "configs" / "conservative_pipeline.json"
    checkpoint = (
        PROJECT_ROOT
        / "joinable_migration_audit"
        / "vendor"
        / "JoinABLe"
        / "pretrained"
        / "paper"
        / "last_run_0.ckpt"
    )
    frozen = source_files(source) + [
        pipeline_config,
        conservative_config,
        checkpoint,
        args.real_report.resolve(),
    ]
    if args.domain_manifest.is_file():
        frozen.append(args.domain_manifest.resolve())
    manifest = {
        "schema_version": "1.0.0",
        "source_root": str(source),
        "output_root": str(output),
        "training_performed": False,
        "files": [
            {"path": str(path), "size": path.stat().st_size, "sha256": sha256(path)}
            for path in frozen
        ],
        "stages": {},
    }
    write_json(output / "run_manifest.json", manifest)

    started = time.perf_counter()
    graph_report = prepare_graphs(source)
    manifest["stages"]["graph_preparation"] = {
        "wall_seconds": time.perf_counter() - started,
        "success_count": graph_report["success_count"],
        "failure_count": graph_report["failure_count"],
    }
    if graph_report["failure_count"]:
        raise RuntimeError("JoinABLe graph preparation had failures")

    cache = output / "cache"
    cache.mkdir()
    main_report = cache / f"joinable_pair_rankings_{args.device}.json"
    wall = run_batch(
        args.joinable_python.resolve(), source, main_report, args.device, args.top_k
    )
    main_payload = json.loads(main_report.read_text(encoding="utf-8"))
    manifest["stages"][f"inference_{args.device}"] = {
        "wall_seconds": wall,
        "model_elapsed_seconds": main_payload["elapsed_seconds"],
        "pair_count": main_payload["pair_count"],
        "success_count": main_payload["success_count"],
        "gpu_peak_allocated_mib": main_payload.get("gpu_peak_allocated_mib"),
    }
    other_device = "cpu" if args.device == "cuda" else "cuda"
    comparison_report = cache / f"joinable_pair_rankings_{other_device}.json"
    comparison_wall = run_batch(
        args.joinable_python.resolve(),
        source,
        comparison_report,
        other_device,
        args.top_k,
    )
    comparison_payload = json.loads(comparison_report.read_text(encoding="utf-8"))
    manifest["stages"][f"inference_{other_device}"] = {
        "wall_seconds": comparison_wall,
        "model_elapsed_seconds": comparison_payload["elapsed_seconds"],
        "pair_count": comparison_payload["pair_count"],
        "success_count": comparison_payload["success_count"],
        "gpu_peak_allocated_mib": comparison_payload.get("gpu_peak_allocated_mib"),
    }
    domain_report = None
    if args.domain_manifest.is_file():
        domain_report = cache / "domain_holdout_union_test_report.json"
        domain_started = time.perf_counter()
        subprocess.run(
            [
                str(args.joinable_python.resolve()),
                str(DOMAIN_EVALUATOR),
                "--manifest",
                str(args.domain_manifest.resolve()),
                "--device",
                args.device,
                "--output",
                str(domain_report),
            ],
            cwd=PROJECT_ROOT / "joinable_gpu_reproduction",
            check=True,
        )
        domain_payload = json.loads(domain_report.read_text(encoding="utf-8"))
        manifest["stages"]["design_disjoint_domain_evaluation"] = {
            "wall_seconds": time.perf_counter() - domain_started,
            "evaluated_count": domain_payload["evaluated_count"],
            "exact_evaluable_count": domain_payload["exact"]["evaluable_count"],
        }

    workspaces = output / "workspaces"
    analytic_root = workspaces / "analytic_only"
    joinable_root = workspaces / "analytic_joinable"
    copy_arm(source, analytic_root)
    copy_arm(source, joinable_root)
    for arm_root, report in ((analytic_root, None), (joinable_root, main_report)):
        for pool in sorted(arm_root.glob("holdout_pool_*")):
            index_pool(
                pool / "parts",
                pool / "index",
                pipeline_config,
                joinable_report_path=report,
            )

    results = output / "results"
    analytic_results = results / "analytic_only"
    joinable_results = results / "analytic_joinable"
    started = time.perf_counter()
    run_candidate_audit(analytic_root, analytic_results)
    run_conservative(
        analytic_root,
        analytic_results,
        pipeline_config_path=pipeline_config,
        conservative_config_path=conservative_config,
    )
    manifest["stages"]["analytic_only_pipeline"] = {
        "wall_seconds": time.perf_counter() - started
    }
    started = time.perf_counter()
    run_candidate_audit(joinable_root, joinable_results)
    run_conservative(
        joinable_root,
        joinable_results,
        pipeline_config_path=pipeline_config,
        conservative_config_path=conservative_config,
    )
    manifest["stages"]["analytic_joinable_pipeline"] = {
        "wall_seconds": time.perf_counter() - started
    }
    audit = run_strict_audit(
        args.real_report.resolve(),
        domain_report,
        analytic_root,
        joinable_root,
        analytic_results,
        joinable_results,
        output / "audit",
    )
    manifest["completed"] = True
    manifest["expert_decision"] = audit["expert_decision"]
    write_json(output / "run_manifest.json", manifest)
    print(json.dumps(audit["expert_decision"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
