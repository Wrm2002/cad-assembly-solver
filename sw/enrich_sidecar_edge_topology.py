"""Enrich B-Rep sidecar graphs with edge topology features from edge_local_features.

Merges the edge convexity/dihedral/adjacent_face output with the existing
_brep_graphs311 sidecar, producing enriched graphs that key_slot_features
and axial_group_centering can consume.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EDGE_FEATURES_SCRIPT = (
    ROOT
    / "cad_assembly_agent"
    / "tools"
    / "brep_graph_extractor"
    / "edge_local_features.py"
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def enrich_one(
    step_path: Path,
    sidecar_path: Path,
    output_path: Path,
    *,
    timeout: int = 300,
) -> dict[str, Any]:
    """Extract edge features and merge into sidecar graph."""
    sidecar = _read(sidecar_path)
    temp_dir = output_path.parent / "_edge_features_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    edge_out = temp_dir / f"{step_path.stem}_edge_features.json"

    # Only run edge features if not cached
    if not edge_out.is_file():
        cmd = [
            sys.executable,
            str(EDGE_FEATURES_SCRIPT),
            "--inventory", str(temp_dir / "_dummy_inventory.json"),
            "--pair-truth", str(temp_dir / "_dummy_pair_truth"),
            "--out-root", str(temp_dir),
            "--report", str(temp_dir / "_dummy_report.json"),
            "--worker",
            "--source", str(step_path),
            "--output", str(edge_out),
            "--scope", "all",
        ]
        # Ensure dummy args exist
        for dummy in ["_dummy_inventory.json", "_dummy_pair_truth"]:
            dummy_path = temp_dir / dummy
            if not dummy_path.exists():
                if dummy.endswith(".json"):
                    _write(dummy_path, {"cases": []})
                else:
                    dummy_path.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if result.returncode != 0:
            return {
                "status": "edge_features_failed",
                "stderr": result.stderr[-2000:],
                "returncode": result.returncode,
            }

    edge_data = _read(edge_out)
    features = edge_data.get("features", [])
    if not features:
        return {
            "status": "no_edge_features",
            "edge_count": 0,
        }

    # Build lookup: occt_topology_index -> feature dict
    edge_by_occt_index: dict[int, dict[str, Any]] = {}
    for feat in features:
        eid = feat.get("edge_id")
        if eid is not None:
            edge_by_occt_index[int(eid)] = feat

    # Merge edge features into sidecar edge nodes
    enriched_count = 0
    for node in sidecar.get("nodes", []):
        if node.get("entity_type") != "edge":
            continue
        occt_idx = node.get("occt_topology_index")
        if occt_idx is None:
            continue
        feat = edge_by_occt_index.get(int(occt_idx))
        if feat is None:
            continue
        node["convexity"] = feat.get("convexity", "unknown")
        node["dihedral_angle"] = feat.get("dihedral_angle_degrees")
        node["signed_dihedral_angle"] = feat.get("signed_dihedral_angle_degrees")
        node["adjacent_face_ids"] = [
            f"face_{fidx:06d}" for fidx in feat.get("adjacent_face_ids", [])
        ]
        node["topology_feature_status"] = feat.get("status", "unknown")
        node["edge_midpoint"] = feat.get("edge_midpoint")
        node["edge_tangent"] = feat.get("edge_tangent")
        node["face_normals"] = feat.get("face_normals")
        enriched_count += 1

    # Set edge_topology_features metadata
    meta = sidecar.setdefault("metadata", {})
    meta["edge_topology_features"] = {
        "available": True,
        "source": "edge_local_features worker (scope=all)",
        "total_edges": edge_data.get("topology_edge_count", 0),
        "selected_edges": edge_data.get("selected_edge_count", 0),
        "successful_edges": edge_data.get("successful_edge_count", 0),
        "enriched_in_sidecar": enriched_count,
        "convexity_convention": edge_data.get("convexity_convention", ""),
    }

    # Remove edge_convexity from unavailable_fields
    unavailable = sidecar.get("unavailable_fields", [])
    sidecar["unavailable_fields"] = [
        f for f in unavailable
        if f not in ("edge convexity", "exact_edge_convexity")
    ]

    _write(output_path, sidecar)
    return {
        "status": "enriched",
        "total_nodes": len(sidecar.get("nodes", [])),
        "total_edges": edge_data.get("topology_edge_count", 0),
        "enriched_edge_nodes": enriched_count,
        "concave_edges": sum(
            1 for n in sidecar.get("nodes", [])
            if n.get("entity_type") == "edge"
            and n.get("convexity") == "concave"
            and n.get("topology_feature_status") == "success"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, required=True,
                        help="Case directory containing STEP files and _brep_graphs311/")
    parser.add_argument("--output-dir", type=Path,
                        help="Output directory for enriched sidecars (default: case_dir/_brep_graphs_enriched/)")
    args = parser.parse_args()

    case_dir = args.case_dir.resolve()
    graph_dir = case_dir / "_brep_graphs311"
    if not graph_dir.is_dir():
        print(f"ERROR: graph_dir not found: {graph_dir}", file=sys.stderr)
        return 1

    output_dir = args.output_dir or (case_dir / "_brep_graphs_enriched")
    output_dir.mkdir(parents=True, exist_ok=True)

    step_files = sorted(
        p for p in case_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".step", ".stp"}
        and not p.name.lower().startswith("assembly")
    )

    results = {}
    for step_path in step_files:
        sidecar_path = graph_dir / f"{step_path.stem}_graph.json"
        if not sidecar_path.is_file():
            print(f"SKIP {step_path.name}: no sidecar at {sidecar_path}")
            continue
        output_path = output_dir / f"{step_path.stem}_graph.json"
        print(f"Enriching {step_path.name}...")
        result = enrich_one(step_path, sidecar_path, output_path)
        results[step_path.name] = result
        status = result["status"]
        if status == "enriched":
            print(f"  OK: {result['enriched_edge_nodes']} edges enriched, "
                  f"{result['concave_edges']} concave")
        else:
            print(f"  FAIL: {status}")

    summary = {
        "schema_version": "1.0.0",
        "case_dir": str(case_dir),
        "output_dir": str(output_dir),
        "results": results,
    }
    _write(output_dir / "_enrichment_summary.json", summary)

    success = all(r.get("status") == "enriched" for r in results.values())
    print(f"\nEnrichment: {'ALL OK' if success else 'SOME FAILED'}")
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
