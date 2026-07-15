"""Extract a deterministic Fusion-STEP domain-adaptation subset from j1.0.0.

Only joint-set JSON, body graph JSON, and STEP files are extracted.  OBJ/SMT and
the 16 GB legacy pickle are intentionally avoided.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


DEFAULT_ARCHIVE = Path(
    r"D:\Model_match_public_data\fusion360_joint\j1.0.0.7z"
)
DEFAULT_SPLIT = Path(
    r"D:\Model_match_public_data\fusion360_joint"
    r"\j1.0.0_preprocessed\train_test.json"
)
DEFAULT_OUTPUT = Path(
    r"D:\Model_match_public_data\fusion360_joint\domain_adapt_300"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def extract_members(
    archive: Path,
    output_root: Path,
    members: list[str],
    batch_size: int,
    progress_path: Path,
    stage: str,
) -> list[dict[str, Any]]:
    del batch_size  # Kept in the CLI for backward compatibility.
    member_list = output_root / f".{stage}_members.txt"
    member_list.write_text(
        "\n".join(members) + "\n", encoding="utf-8"
    )
    write_json(
        progress_path,
        {
            "stage": stage,
            "status": "extracting_single_archive_scan",
            "total_members": len(members),
            "member_list": str(member_list),
        },
    )
    completed = subprocess.run(
        [
            "tar",
            "-xf",
            str(archive),
            "-C",
            str(output_root),
            "-T",
            str(member_list),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    failures = []
    if completed.returncode != 0:
        failures.append(
            {
                "stage": stage,
                "returncode": completed.returncode,
                "stderr": completed.stderr[-4000:],
            }
        )
    write_json(
        progress_path,
        {
            "stage": stage,
            "status": (
                "complete" if not failures else "archive_extract_failed"
            ),
            "completed_members": (
                len(members) if not failures else None
            ),
            "total_members": len(members),
            "failure_count": len(failures),
        },
    )
    print(
        f"{stage}: members={len(members)}, failures={len(failures)}",
        flush=True,
    )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-count", type=int, default=200)
    parser.add_argument("--validation-count", type=int, default=50)
    parser.add_argument("--test-count", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=80)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_root / "extraction_progress.json"
    split = read_json(args.split_file)
    requested = {
        "train": max(0, args.train_count),
        "validation": max(0, args.validation_count),
        "test": max(0, args.test_count),
    }
    selected = {
        name: list(split[name][:count])
        for name, count in requested.items()
    }
    joint_ids = [
        joint_id
        for name in ("train", "validation", "test")
        for joint_id in selected[name]
    ]
    joint_members = [
        f"j1.0.0/joint/{joint_id}.json" for joint_id in joint_ids
    ]
    failures = extract_members(
        args.archive,
        args.output_root,
        joint_members,
        max(1, args.batch_size),
        progress_path,
        "joint_json",
    )

    data_root = args.output_root / "j1.0.0" / "joint"
    body_ids = set()
    missing_joint_json = []
    for joint_id in joint_ids:
        path = data_root / f"{joint_id}.json"
        if not path.is_file():
            missing_joint_json.append(str(path))
            continue
        payload = read_json(path)
        body_ids.add(str(payload["body_one"]))
        body_ids.add(str(payload["body_two"]))

    body_members = []
    for body_id in sorted(body_ids):
        body_members.extend(
            (
                f"j1.0.0/joint/{body_id}.json",
                f"j1.0.0/joint/{body_id}.step",
            )
        )
    failures.extend(
        extract_members(
            args.archive,
            args.output_root,
            body_members,
            max(1, args.batch_size),
            progress_path,
            "body_json_and_step",
        )
    )

    missing_body_files = []
    total_bytes = 0
    for body_id in sorted(body_ids):
        for suffix in (".json", ".step"):
            path = data_root / f"{body_id}{suffix}"
            if path.is_file():
                total_bytes += path.stat().st_size
            else:
                missing_body_files.append(str(path))

    subset_split = {
        "schema_version": "1.0.0",
        "source_split_file": str(args.split_file),
        "selection_policy": "first_N_from_official_split_order",
        "splits": selected,
    }
    write_json(args.output_root / "subset_split.json", subset_split)
    manifest = {
        "schema_version": "1.0.0",
        "source_archive": str(args.archive),
        "source_split_file": str(args.split_file),
        "output_root": str(args.output_root),
        "joint_set_count": len(joint_ids),
        "unique_body_count": len(body_ids),
        "body_asset_count": len(body_ids) * 2,
        "extracted_body_bytes": total_bytes,
        "splits": {key: len(value) for key, value in selected.items()},
        "failure_count": (
            len(failures)
            + len(missing_joint_json)
            + len(missing_body_files)
        ),
        "batch_failures": failures,
        "missing_joint_json": missing_joint_json,
        "missing_body_files": missing_body_files,
        "excluded_formats": [".obj", ".smt", "legacy_preprocessed_pickle"],
        "failure_reasons": [
            f"missing_joint_json:{path}" for path in missing_joint_json
        ]
        + [f"missing_body_file:{path}" for path in missing_body_files],
        "unavailable_fields": [],
    }
    write_json(args.output_root / "subset_manifest.json", manifest)
    write_json(
        progress_path,
        {
            "stage": "complete",
            "joint_set_count": len(joint_ids),
            "unique_body_count": len(body_ids),
            "failure_count": manifest["failure_count"],
        },
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if manifest["failure_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
