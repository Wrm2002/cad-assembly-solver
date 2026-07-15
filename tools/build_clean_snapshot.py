#!/usr/bin/env python3
"""Build a curated, reproducible source snapshot of Model_match."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "release"
STAMP = "20260707"
ZIP_PATH = RELEASE / f"Model_match_clean_{STAMP}.zip"

ROOT_FILES = {
    ".gitignore",
    "AGENTS.md",
    "CLAUDE.md",
    "CLEANUP_SUMMARY.json",
    "IMAGE_TASK_DELIVERY.md",
    "IMAGE_TASK_DELIVERY_STATUS.json",
    "INCIDENT_REPORT.md",
    "ME.md",
    "P0_CONSERVATIVE_DELIVERY.md",
    "P0_CONSERVATIVE_FREEZE.json",
    "P123_DELIVERY.md",
    "P123_DELIVERY_FREEZE.json",
    "PROJECT_STRUCTURE.md",
    "STRICT_ROUTE_ALIGNMENT_REVIEW.md",
    "STRICT_ROUTE_ALIGNMENT_STATUS.json",
    "TECHNICAL_REVIEW_AND_MODIFICATIONS.md",
    "WORKING_PROTOCOL.md",
}

EXPLICIT_FILES = {
    "sw/data/joinable_holdout_ablation_repro_v2/run_manifest.json",
    "sw/data/joinable_holdout_ablation_repro_v2/JOINABLE_HARDER_HOLDOUT_FREEZE.json",
    "sw/data/joinable_holdout_ablation_repro_v2/audit/strict_audit.json",
    "sw/data/joinable_holdout_ablation_repro_v2/audit/EXPERT_REVIEW.md",
    "sw/data/joinable_holdout_ablation_repro_v2/audit/real_joint_rescue_cases.csv",
    "sw/data/joinable_holdout_ablation_repro_v2/audit/false_candidate_routes.csv",
    "sw/data/joinable_holdout_ablation_repro_v2/cache/domain_holdout_union_test_report.json",
    "sw/data/joinable_holdout_ablation_repro_v2/results/analytic_only/conservative_metrics.json",
    "sw/data/joinable_holdout_ablation_repro_v2/results/analytic_joinable/conservative_metrics.json",
}

INCLUDE_TREES = (
    "tools",
    "sw/configs",
    "sw/global_optimizer",
    "sw/schemas",
    "sw/sw_dataset_generator",
    "sw/tests",
    "cad_assembly_agent",
    "joinable_gpu_reproduction",
    "joinable_migration_audit",
    "joinable_step4_addon_20260705",
    "joinable_step4_bundle_20260705",
    "public_cad_dataset_audit",
    "sw/data/functional_dataset_v1",
    "sw/data/functional_cad_holdout_v1",
    "sw/data/multimodal_calibration_v1",
    "sw/data/topology_pose_audit_v1",
    "sw/data/topology_pose_holdout_audit_v1",
)

SW_ROOT_SUFFIXES = {
    ".py",
    ".md",
    ".json",
    ".csv",
    ".yaml",
    ".yml",
    ".toml",
    ".txt",
    ".ps1",
}

BLOCKED_DIR_NAMES = {
    ".git",
    ".conda",
    ".venv",
    ".vscode",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "incident_dumps",
    "release",
    "autodl_2600_results_20260704",
}

BLOCKED_PARTS = {
    "domain_finetune_cpu",
    "domain_finetune_cpu_low_lr",
    "domain_finetune_cpu_post_only",
    "domain_finetune_cpu_seed_7",
    "domain_finetune_cpu_seed_17",
    "domain_finetune_cpu_seed_73",
    "domain_finetune_relocation_smoke",
    "domain_finetune_results",
    "domain_finetune_smoke",
    "domain_finetune_smoke_exact",
}

BLOCKED_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".pid",
    ".log",
    ".rar",
    ".zip",
    ".7z",
    ".pt",
}

MAX_GENERAL_FILE_BYTES = 8 * 1024 * 1024
SELECTED_CHECKPOINT = Path(
    "joinable_migration_audit/vendor/JoinABLe/pretrained/paper/last_run_0.ckpt"
)


def is_allowed(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    parts = set(rel.parts)
    if parts & BLOCKED_DIR_NAMES or parts & BLOCKED_PARTS:
        return False
    if path.suffix.lower() in BLOCKED_SUFFIXES:
        return False
    if path.name.startswith(".env"):
        return False
    if rel == SELECTED_CHECKPOINT:
        return True
    if path.stat().st_size > MAX_GENERAL_FILE_BYTES:
        return False
    return True


def collect_files() -> list[Path]:
    selected: set[Path] = set()

    for name in ROOT_FILES:
        path = ROOT / name
        if path.is_file() and is_allowed(path):
            selected.add(path)

    for name in EXPLICIT_FILES:
        path = ROOT / name
        if path.is_file() and is_allowed(path):
            selected.add(path)

    sw_root = ROOT / "sw"
    for path in sw_root.iterdir():
        if path.is_file() and path.suffix.lower() in SW_ROOT_SUFFIXES and is_allowed(path):
            selected.add(path)

    for tree in INCLUDE_TREES:
        base = ROOT / tree
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file() and is_allowed(path):
                selected.add(path)

    checkpoint = ROOT / SELECTED_CHECKPOINT
    if checkpoint.is_file():
        selected.add(checkpoint)

    return sorted(selected, key=lambda p: p.relative_to(ROOT).as_posix())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    RELEASE.mkdir(exist_ok=True)
    files = collect_files()
    manifest = {
        "snapshot": ZIP_PATH.name,
        "date": "2026-07-07",
        "policy": "curated source, tests, small functional datasets, key reports, and one JoinABLe checkpoint",
        "file_count": len(files),
        "files": [],
    }

    with zipfile.ZipFile(
        ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for path in files:
            rel = path.relative_to(ROOT).as_posix()
            digest = sha256_file(path)
            archive.write(path, rel)
            manifest["files"].append(
                {"path": rel, "size": path.stat().st_size, "sha256": digest}
            )
        archive.writestr(
            "SNAPSHOT_MANIFEST.json",
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        )

    archive_digest = sha256_file(ZIP_PATH)
    checksum_path = ZIP_PATH.with_suffix(ZIP_PATH.suffix + ".sha256")
    checksum_path.write_text(
        f"{archive_digest}  {ZIP_PATH.name}{os.linesep}", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "zip": str(ZIP_PATH),
                "sha256_file": str(checksum_path),
                "archive_sha256": archive_digest,
                "file_count": len(files),
                "size_bytes": ZIP_PATH.stat().st_size,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
