"""Merge failure-isolated equivalent-pose B-Rep dataset chunks in order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chunk_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--splits", nargs="+", choices=("train", "dev", "test"), default=("train", "dev", "test"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit: dict[str, Any] = {
        "schema_version": "fusion360_equivalent_pose_brep_hard_negative_merged.v1",
        "merge_contract": "Chunks are concatenated only within their official Fusion assembly split and sorted by source record range.",
        "splits": {},
    }
    for split in args.splits:
        pieces: list[tuple[int, Path, dict[str, Any]]] = []
        for audit_path in args.chunk_root.glob(f"{split}_*/dataset_audit.json"):
            row = json.loads(audit_path.read_text(encoding="utf-8"))["splits"][split]
            pieces.append((int(row["source_record_range"]["start"]), audit_path.parent / f"{split}.npz", row))
        pieces.sort(key=lambda item: item[0])
        if not pieces:
            raise RuntimeError(f"no_chunks_for_{split}")
        starts = [item[0] for item in pieces]
        if starts[0] != 0:
            raise RuntimeError(f"chunk_range_does_not_start_at_zero_{split}")
        arrays_by_key: dict[str, list[np.ndarray]] = {}
        expected_end = 0
        chunk_audits: list[dict[str, Any]] = []
        for start, path, row in pieces:
            if start != expected_end:
                raise RuntimeError(f"chunk_range_gap_or_overlap_{split}_{expected_end}_{start}")
            with np.load(path) as payload:
                for key in payload.files:
                    arrays_by_key.setdefault(key, []).append(payload[key])
            expected_end = int(row["source_record_range"]["end_exclusive"])
            chunk_audits.append(row)
        if expected_end != int(chunk_audits[0]["source_record_range"]["total_in_split"]):
            raise RuntimeError(f"chunk_range_incomplete_{split}")
        merged = {key: np.concatenate(value, axis=0) for key, value in arrays_by_key.items()}
        np.savez_compressed(args.output_dir / f"{split}.npz", **merged)
        audit["splits"][split] = {
            "chunks": len(pieces), "source_records": expected_end,
            "samples": int(merged["target_pose"].shape[0]),
            "mean_positive_pose_modes": float(merged["target_pose_mode_mask"].sum(axis=1).mean()),
            "mean_measured_hard_negatives": float(merged["hard_negative_mask"].sum(axis=1).mean()),
            "primary_examples": int(sum(item["counts"].get("emitted_primary", 0) for item in chunk_audits)),
            "equivalent_entity_examples": int(sum(item["counts"].get("emitted_equivalent_entity", 0) for item in chunk_audits)),
            "manifest_validated_supervisions": int(sum(item["counts"].get("manifest_validated_supervisions", 0) for item in chunk_audits)),
        }
        print(json.dumps({"split": split, **audit["splits"][split]}), flush=True)
    (args.output_dir / "dataset_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
