"""Write a reproducibility manifest for the frozen geometry baseline."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from contracts import ANGLE_UNIT, COORDINATE_FRAME, LENGTH_UNIT, SCHEMA_VERSION
from pipeline_api import API_VERSION


CORE_FILES = [
    "features.py",
    "constraints.py",
    "coordinate_solver.py",
    "refinement.py",
    "match_scoring.py",
    "match_pruning.py",
    "small_assembly_solver.py",
    "placement_validation.py",
    "compute_manifest.py",
    "pipeline_api.py",
    "contracts.py",
    "configs/pool_pipeline.json",
]


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_revision(project):
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        ).stdout.strip() or None
    except Exception:
        return None


def freeze(project):
    project = Path(project).resolve()
    files = {
        name: _sha256(project / name)
        for name in CORE_FILES
        if (project / name).is_file()
    }
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "api_version": API_VERSION,
        "schema_version": SCHEMA_VERSION,
        "git_revision": _git_revision(project),
        "python": sys.version,
        "platform": platform.platform(),
        "conventions": {
            "length_unit": LENGTH_UNIT,
            "angle_unit": ANGLE_UNIT,
            "coordinate_frame": COORDINATE_FRAME,
            "part_id": "stable pool-local identifier; never inferred from function",
        },
        "core_file_sha256": files,
        "known_limitations": [
            "Synthetic v0 cases share one incremental parameterized family.",
            "Hole and hole-pattern classification is heuristic and labelled as such.",
            "Bounding-box collision is a conservative precheck, not exact penetration.",
            "Pool prescreen prioritizes recall and does not decide final grouping.",
            "No LLM, learned scorer, global grouping, or Agent action is used in D1-D3.",
        ],
    }


if __name__ == "__main__":
    project = Path(__file__).resolve().parent
    output = project / "baseline_freeze.json"
    output.write_text(
        json.dumps(freeze(project), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(output)
