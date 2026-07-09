"""Stable public façade for geometry tools used by later Agent stages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from compute_manifest import generate_manifest
from contracts import SCHEMA_VERSION
from part_index import index_part
from placement_validation import validate_assembly
from pool_index import index_pool
from known_group_assembly import run_known_group_assembly


API_VERSION = "1.0.0"


def extract_part_feature(step_path: str | Path, part_id: str | None = None) -> dict[str, Any]:
    """Return a schema-valid PartFeature document."""
    return index_part(step_path, part_id).model_dump(mode="json")


def index_part_pool(
    parts_dir: str | Path,
    output_dir: str | Path,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Index and prescreen an unordered STEP pool with full audit outputs."""
    project = Path(__file__).resolve().parent
    config = config_path or project / "configs" / "pool_pipeline.json"
    outputs = index_pool(parts_dir, output_dir, config)
    return {
        "api_version": API_VERSION,
        "schema_version": SCHEMA_VERSION,
        "output_dir": str(Path(output_dir).resolve()),
        "num_parts": len(outputs["part_features.json"]),
        "num_candidates": len(outputs["geometry_candidates.json"]),
        "num_kept_candidates": len(outputs["pruned_candidates.json"]),
    }


def solve_known_group(
    case_dir: str | Path,
    *,
    solver: str = "reliable",
    beam_width: int = 20,
    min_score: float = 0.5,
    max_neighbors: int = 4,
) -> dict[str, Any]:
    """Run the existing group solver and validation through a stable API."""
    if solver not in {"bfs", "reliable"}:
        raise ValueError("solver must be 'bfs' or 'reliable'")
    case_dir = Path(case_dir).resolve()
    generate_manifest(
        case_dir,
        write_diagnostics=True,
        solver=solver,
        enable_scoring=solver == "reliable",
        enable_pruning=solver == "reliable",
        beam_width=beam_width,
        min_score=min_score,
        max_neighbors=max_neighbors,
    )
    matches_path = case_dir / "kept_matches.json"
    validation = validate_assembly(
        case_dir,
        matches_path if matches_path.is_file() else None,
    )
    return {
        "api_version": API_VERSION,
        "schema_version": SCHEMA_VERSION,
        "case_dir": str(case_dir),
        "manifest": json.loads(
            (case_dir / "assembly_manifest.json").read_text(encoding="utf-8")
        ),
        "validation": validation,
    }


def recognize_known_group_relations(
    case_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    joinable_report: str | Path | None = None,
    beam_width: int = 20,
) -> dict[str, Any]:
    """Recognize direct labelled relations for parts known to form one assembly.

    This is the preferred public entry point for the narrowed project task. It
    does not run mixed-pool grouping or semantic membership decisions.
    """
    return run_known_group_assembly(
        case_dir,
        output_dir=output_dir,
        joinable_report=joinable_report,
        beam_width=beam_width,
    )
