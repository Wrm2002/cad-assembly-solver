"""Generate diagnostics for the unchanged legacy CAD assembly pipeline."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

from constraints import (
    CLEARANCE,
    COAXIAL,
    PLANAR_ALIGN,
    PLANAR_MATE,
    POCKET_MATE,
    _classify_part,
    match_features,
)
from features import extract_features
from match_scoring import score_matches


STEP_SUFFIXES = {".step", ".stp"}
STRONG_MATCH_TYPES = {COAXIAL, CLEARANCE, POCKET_MATE}
WEAK_MATCH_TYPES = {PLANAR_ALIGN, PLANAR_MATE}


def _step_inputs(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in STEP_SUFFIXES
        and not path.name.lower().startswith("assembly")
    )


def _bbox_size(features: dict[str, Any]) -> list[float] | None:
    bbox = features.get("bbox")
    if not bbox or not bbox.get("min") or not bbox.get("max"):
        return None
    return [
        round(float(bbox["max"][index]) - float(bbox["min"][index]), 6)
        for index in range(3)
    ]


def _bbox_diagonal(features: dict[str, Any]) -> float:
    size = _bbox_size(features)
    return math.sqrt(sum(value * value for value in size)) if size else 0.0


def _identity_placement(placement: dict[str, Any]) -> bool:
    translation = placement.get("translate", [0.0, 0.0, 0.0])
    rotations = placement.get("rotate_sequence", [])
    translation_identity = all(abs(float(value)) < 1e-9 for value in translation)
    rotation_identity = True
    for rotation in rotations:
        angle = rotation.get("axis_angle", [0.0, 0.0, 1.0, 0.0])
        if len(angle) >= 4 and abs(float(angle[3])) >= 1e-9:
            rotation_identity = False
            break
    return translation_identity and rotation_identity


def _rotation_magnitude(placement: dict[str, Any]) -> float:
    total = 0.0
    for rotation in placement.get("rotate_sequence", []):
        angle = rotation.get("axis_angle")
        if angle and len(angle) >= 4:
            total += abs(float(angle[3]))
        elif "axis_to" in rotation:
            source = rotation["axis_to"].get("from", [0.0, 0.0, 1.0])
            target = rotation["axis_to"].get("to", [0.0, 0.0, 1.0])
            source_norm = math.sqrt(sum(float(value) ** 2 for value in source))
            target_norm = math.sqrt(sum(float(value) ** 2 for value in target))
            if source_norm and target_norm:
                dot = sum(float(a) * float(b) for a, b in zip(source, target))
                total += math.degrees(
                    math.acos(max(-1.0, min(1.0, dot / source_norm / target_norm)))
                )
    return total


def _choose_reference(parts_features: dict[str, dict[str, Any]], matches: list[dict[str, Any]]) -> str:
    adjacency = defaultdict(list)
    for match in matches:
        a, b = match["parts"]
        adjacency[a].append(match)
        adjacency[b].append(match)

    def weight(part: str) -> tuple[int, float]:
        axial_edges = sum(
            match["type"] in {COAXIAL, CLEARANCE}
            for match in adjacency.get(part, [])
        )
        radius = max(
            (float(cylinder["radius"]) for cylinder in parts_features[part].get("cylinders", [])),
            default=0.0,
        )
        return axial_edges, radius

    return max(parts_features, key=weight)


def _graph_analysis(
    part_names: list[str],
    matches: list[dict[str, Any]],
    reference: str,
) -> tuple[dict[str, Any], set[str]]:
    neighbors: dict[str, set[str]] = {part: set() for part in part_names}
    edge_types: dict[str, list[str]] = {part: [] for part in part_names}
    pair_set: set[tuple[str, str]] = set()
    for match in matches:
        a, b = match["parts"]
        neighbors[a].add(b)
        neighbors[b].add(a)
        edge_types[a].append(match["type"])
        edge_types[b].append(match["type"])
        pair_set.add(tuple(sorted((a, b))))

    components: list[list[str]] = []
    unseen = set(part_names)
    while unseen:
        start = min(unseen)
        component = []
        queue = deque([start])
        unseen.remove(start)
        while queue:
            part = queue.popleft()
            component.append(part)
            for neighbor in sorted(neighbors[part]):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)
        components.append(sorted(component))

    reachable: set[str] = set()
    queue = deque([reference])
    while queue:
        part = queue.popleft()
        if part in reachable:
            continue
        reachable.add(part)
        queue.extend(neighbors[part] - reachable)

    high_degree_threshold = max(3, min(5, len(part_names) - 1))
    high_degree = sorted(
        part for part in part_names if len(neighbors[part]) >= high_degree_threshold
    )
    weak_only = sorted(
        part
        for part in part_names
        if edge_types[part]
        and not any(match_type in STRONG_MATCH_TYPES for match_type in edge_types[part])
    )

    return {
        "num_parts": len(part_names),
        "num_edges": len(pair_set),
        "connected_components": components,
        "connected_component_count": len(components),
        "isolated_parts": sorted(part for part in part_names if not neighbors[part]),
        "degrees": {part: len(neighbors[part]) for part in part_names},
        "high_degree_parts": high_degree,
        "suspicious_weak_only_parts": weak_only,
        "reference_part": reference,
    }, reachable


def _load_manifest(folder: Path) -> tuple[dict[str, Any] | None, str | None]:
    path = folder / "assembly_manifest.json"
    if not path.is_file():
        return None, "assembly_manifest.json is missing"
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as exc:
        return None, f"assembly_manifest.json is invalid: {exc}"


def _placement_analysis(
    manifest: dict[str, Any] | None,
    part_names: list[str],
    reachable: set[str],
    parts_features: dict[str, dict[str, Any]],
    reference: str,
) -> dict[str, Any]:
    components = manifest.get("components", []) if manifest else []
    by_source = {component.get("source"): component for component in components}
    maximum_diagonal = max(
        (_bbox_diagonal(features) for features in parts_features.values()),
        default=0.0,
    )
    excessive_threshold = max(1000.0, maximum_diagonal * 10.0)

    solved = []
    unsolved = []
    identity = []
    excessive_translation = []
    excessive_rotation = []
    statuses = {}

    for part in part_names:
        component = by_source.get(part)
        placement = component.get("placement", {}) if component else {}
        is_identity = _identity_placement(placement)
        if is_identity:
            identity.append(part)

        if component is None:
            status = "unsolved"
            reason = "missing from manifest"
        elif part not in reachable:
            status = "unsolved"
            reason = "not reachable from solver reference in the match graph"
        elif part == reference:
            status = "reference"
            reason = "reference part is intentionally fixed at identity"
        else:
            status = "solved"
            reason = "reachable from reference; placement present"

        statuses[part] = {
            "status": status,
            "reason": reason,
            "identity_placement": is_identity,
        }
        if status == "unsolved":
            unsolved.append(part)
        else:
            solved.append(part)

        translation = placement.get("translate", [0.0, 0.0, 0.0])
        magnitude = math.sqrt(sum(float(value) ** 2 for value in translation))
        if magnitude > excessive_threshold:
            excessive_translation.append({"part": part, "magnitude": magnitude})
        rotation = _rotation_magnitude(placement)
        if rotation > 360.0 + 1e-6:
            excessive_rotation.append({"part": part, "degrees": rotation})

    return {
        "solved_parts": sorted(solved),
        "unsolved_parts": sorted(unsolved),
        "identity_placements": sorted(identity),
        "excessive_translation": excessive_translation,
        "excessive_rotation": excessive_rotation,
        "translation_warning_threshold": excessive_threshold,
        "part_status": statuses,
    }


def analyze_case(folder: Path | str) -> dict[str, Any]:
    folder = Path(folder).resolve()
    inputs = _step_inputs(folder)
    if not inputs:
        raise FileNotFoundError(f"No STEP part files in {folder}")

    parts_features: dict[str, dict[str, Any]] = {}
    feature_summary = {}
    for path in inputs:
        features = extract_features(str(path))
        parts_features[path.name] = features
        feature_summary[path.name] = {
            "cylinders": len(features.get("cylinders", [])),
            "planes": len(features.get("planes", [])),
            "cones": len(features.get("cones", [])),
            "torii": len(features.get("torii", [])),
            "spheres": len(features.get("spheres", [])),
            "bbox_size": _bbox_size(features),
            "classified_type": _classify_part(features),
            "occt_stats": features.get("occt_stats", {}),
        }

    raw_matches = match_features(parts_features)
    matches = score_matches(raw_matches, parts_features)
    match_types = Counter(match["type"] for match in matches)
    confidence_counts = Counter(match["confidence"] for match in matches)
    scores = [float(match["score"]) for match in matches]
    reference = _choose_reference(parts_features, matches)
    graph, reachable = _graph_analysis(list(parts_features), matches, reference)
    manifest, manifest_error = _load_manifest(folder)
    placement = _placement_analysis(
        manifest,
        list(parts_features),
        reachable,
        parts_features,
        reference,
    )

    warnings = []
    if manifest_error:
        warnings.append(manifest_error)
    if graph["connected_component_count"] > 1:
        warnings.append(
            f"disconnected match graph: {graph['connected_component_count']} components"
        )
    if graph["isolated_parts"]:
        warnings.append(f"isolated parts: {', '.join(graph['isolated_parts'])}")
    if graph["high_degree_parts"]:
        warnings.append(f"high-degree parts: {', '.join(graph['high_degree_parts'])}")
    if graph["suspicious_weak_only_parts"]:
        warnings.append(
            "weak planar-only evidence: "
            + ", ".join(graph["suspicious_weak_only_parts"])
        )
    if placement["unsolved_parts"]:
        warnings.append(f"unsolved parts: {', '.join(placement['unsolved_parts'])}")
    ambiguous_identity = [
        part
        for part in placement["identity_placements"]
        if part != reference and part not in placement["unsolved_parts"]
    ]
    if ambiguous_identity:
        warnings.append(
            "identity placement, success is ambiguous: " + ", ".join(ambiguous_identity)
        )
    if placement["excessive_translation"]:
        warnings.append("one or more placements have excessive translation")
    if placement["excessive_rotation"]:
        warnings.append("one or more placements have excessive rotation")
    warnings.append("collision validation is not available in the legacy baseline")

    return {
        "schema_version": 1,
        "case_id": folder.name,
        "folder": str(folder),
        "features": feature_summary,
        "matches": {
            "raw_match_count": len(matches),
            "counts_by_type": dict(sorted(match_types.items())),
            "confidence_counts": dict(sorted(confidence_counts.items())),
            "score_summary": {
                "minimum": min(scores) if scores else None,
                "maximum": max(scores) if scores else None,
                "mean": sum(scores) / len(scores) if scores else None,
            },
            "items": matches,
        },
        "graph": graph,
        "placement": placement,
        "outputs": {
            "manifest_exists": (folder / "assembly_manifest.json").is_file(),
            "assembly_step_exists": (folder / "assembly.step").is_file(),
            "manifest_error": manifest_error,
        },
        "warnings": warnings,
    }


def _render_report(diagnostics: dict[str, Any]) -> str:
    graph = diagnostics["graph"]
    placement = diagnostics["placement"]
    matches = diagnostics["matches"]
    lines = [
        f"Assembly diagnostics: {diagnostics['case_id']}",
        "=" * 72,
        f"Parts: {graph['num_parts']}",
        f"Matches: {matches['raw_match_count']} {matches['counts_by_type']}",
        f"Confidence: {matches['confidence_counts']}",
        f"Score summary: {matches['score_summary']}",
        f"Graph edges: {graph['num_edges']}",
        f"Connected components: {graph['connected_component_count']}",
        f"Reference part: {graph['reference_part']}",
        f"Solved/reference parts: {len(placement['solved_parts'])}",
        f"Unsolved parts: {len(placement['unsolved_parts'])}",
        f"Identity placements: {len(placement['identity_placements'])}",
        "",
        "Part features:",
    ]
    for part, features in diagnostics["features"].items():
        lines.append(
            f"- {part}: type={features['classified_type']}, "
            f"cylinders={features['cylinders']}, planes={features['planes']}, "
            f"cones={features['cones']}, torii={features['torii']}, "
            f"spheres={features['spheres']}, bbox={features['bbox_size']}"
        )
    lines.extend(["", "Warnings:"])
    lines.extend(f"- {warning}" for warning in diagnostics["warnings"])
    return "\n".join(lines) + "\n"


def write_diagnostics(folder: Path | str) -> tuple[Path, Path]:
    folder = Path(folder).resolve()
    diagnostics = analyze_case(folder)
    json_path = folder / "assembly_diagnostics.json"
    report_path = folder / "assembly_report.txt"
    json_path.write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(_render_report(diagnostics), encoding="utf-8")
    return json_path, report_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder")
    args = parser.parse_args()
    folder = Path(args.folder).resolve()
    json_path, report_path = write_diagnostics(folder)
    print(json_path)
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
