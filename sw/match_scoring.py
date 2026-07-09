"""Explainable rule-based scoring for legacy CAD mate candidates.

This module does not generate, prune, or solve matches.  It annotates copies
of the candidates produced by ``constraints.py`` with score, confidence, and
reason fields so its effect can be measured before changing the solver.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from constraints import CLEARANCE, COAXIAL, PLANAR_ALIGN, PLANAR_MATE, POCKET_MATE
from features import extract_features
from constraints import match_features


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _dot(a: list[float] | tuple[float, ...], b: list[float] | tuple[float, ...]) -> float:
    norm_a = math.sqrt(sum(float(value) ** 2 for value in a))
    norm_b = math.sqrt(sum(float(value) ** 2 for value in b))
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return sum(float(x) * float(y) for x, y in zip(a, b)) / norm_a / norm_b


def _area_ratio(area_a: float, area_b: float) -> float:
    if area_a <= 0.0 or area_b <= 0.0:
        return 0.0
    return min(area_a, area_b) / max(area_a, area_b)


def _area_reliability(area_a: float, area_b: float, scale: float = 1000.0) -> float:
    representative = math.sqrt(max(0.0, area_a) * max(0.0, area_b))
    return _clamp(math.log1p(representative) / math.log1p(scale))


def _feature(
    match: dict[str, Any],
    parts_features: dict[str, dict[str, Any]],
    side: str,
    collection: str,
) -> dict[str, Any]:
    part_index = 0 if side == "a" else 1
    feature_key = "feat_a_idx" if side == "a" else "feat_b_idx"
    part = match["parts"][part_index]
    index = int(match.get(feature_key, -1))
    features = parts_features.get(part, {}).get(collection, [])
    return features[index] if 0 <= index < len(features) else {}


def _confidence(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.5:
        return "medium"
    return "low"


def _score_coaxial(
    match: dict[str, Any],
    parts_features: dict[str, dict[str, Any]],
) -> tuple[float, dict[str, Any]]:
    a = _feature(match, parts_features, "a", "cylinders")
    b = _feature(match, parts_features, "b", "cylinders")
    radius_a = float(a.get("radius", match.get("_radius_a", 0.0)))
    radius_b = float(b.get("radius", radius_a + float(match.get("radius_match", 0.0))))
    radius_diff = abs(radius_a - radius_b)
    radius_scale = max(radius_a, radius_b, 1.0)
    radius_similarity = math.exp(-4.0 * radius_diff / radius_scale)
    axis_dot = abs(_dot(a.get("axis", [0, 0, 1]), b.get("axis", [0, 0, 1])))
    area_a = float(a.get("area", 0.0))
    area_b = float(b.get("area", 0.0))
    area_signal = _area_reliability(area_a, area_b)
    max_radius_a = max(
        (float(item.get("radius", 0.0)) for item in parts_features[match["parts"][0]].get("cylinders", [])),
        default=radius_a,
    )
    max_radius_b = max(
        (float(item.get("radius", 0.0)) for item in parts_features[match["parts"][1]].get("cylinders", [])),
        default=radius_b,
    )
    main_cylinder_ratio = min(
        radius_a / max(max_radius_a, 1e-9),
        radius_b / max(max_radius_b, 1e-9),
    )
    score = (
        0.50 * radius_similarity
        + 0.20 * axis_dot
        + 0.15 * area_signal
        + 0.15 * main_cylinder_ratio
    )
    return score, {
        "radius_a": radius_a,
        "radius_b": radius_b,
        "radius_diff": radius_diff,
        "radius_similarity": radius_similarity,
        "axis_dot_abs": axis_dot,
        "area_a": area_a,
        "area_b": area_b,
        "area_reliability": area_signal,
        "main_cylinder_ratio": main_cylinder_ratio,
    }


def _score_clearance(
    match: dict[str, Any],
    parts_features: dict[str, dict[str, Any]],
) -> tuple[float, dict[str, Any]]:
    inner = _feature(match, parts_features, "a", "cylinders")
    outer = _feature(match, parts_features, "b", "cylinders")
    radius_inner = float(inner.get("radius", match.get("_radius_a", 0.0)))
    radius_outer = float(outer.get("radius", radius_inner + float(match.get("gap", 0.0))))
    gap = max(0.0, radius_outer - radius_inner)
    gap_ratio = gap / max(radius_inner, 1.0)
    # Broad engineering prior: a small positive gap is strongest; a gap near
    # or larger than the shaft radius is usually a false candidate.
    gap_quality = math.exp(-3.0 * gap_ratio)
    positive_gap = 1.0 if gap >= 0.05 else _clamp(gap / 0.05)
    axis_dot = abs(
        _dot(inner.get("axis", [0, 0, 1]), outer.get("axis", [0, 0, 1]))
    )
    area_inner = float(inner.get("area", 0.0))
    area_outer = float(outer.get("area", 0.0))
    area_signal = _area_reliability(area_inner, area_outer)
    structural_radius = _clamp(radius_inner / 20.0)
    base_score = (
        0.40 * gap_quality
        + 0.10 * positive_gap
        + 0.20 * axis_dot
        + 0.15 * area_signal
        + 0.15 * structural_radius
    )
    oversize_penalty = math.exp(-1.5 * max(0.0, gap_ratio - 0.25))
    score = base_score * oversize_penalty
    return score, {
        "radius_inner": radius_inner,
        "radius_outer": radius_outer,
        "gap": gap,
        "gap_ratio": gap_ratio,
        "gap_quality": gap_quality,
        "oversize_gap_penalty": oversize_penalty,
        "axis_dot_abs": axis_dot,
        "area_inner": area_inner,
        "area_outer": area_outer,
        "area_reliability": area_signal,
        "structural_radius": structural_radius,
    }


def _score_planar(
    match: dict[str, Any],
    parts_features: dict[str, dict[str, Any]],
) -> tuple[float, dict[str, Any]]:
    a = _feature(match, parts_features, "a", "planes")
    b = _feature(match, parts_features, "b", "planes")
    normal_dot = _dot(a.get("normal", [0, 0, 1]), b.get("normal", [0, 0, 1]))
    expected = -1.0 if match["type"] == PLANAR_MATE else 1.0
    # Candidate features are expressed in independent part-local frames.
    # Pose search may flip either normal, so local parallelism magnitude is
    # evidence; final orientation correctness belongs to validation.
    normal_quality = abs(normal_dot)
    area_a = float(a.get("area", 0.0))
    area_b = float(b.get("area", 0.0))
    ratio = _area_ratio(area_a, area_b)
    area_signal = _area_reliability(area_a, area_b, scale=5000.0)
    distance = abs(float(match.get("distance", 0.0)))
    distance_scale = max(
        math.sqrt(max(area_a, area_b, 1.0)),
        1.0,
    )
    distance_quality = math.exp(-distance / distance_scale)
    base = (
        0.35 * normal_quality
        + 0.30 * ratio
        + 0.20 * area_signal
        + 0.15 * distance_quality
    )
    # Same-normal planar alignment is deliberately weaker when used alone.
    score = base if match["type"] == PLANAR_MATE else 0.75 * base
    return score, {
        "normal_dot": normal_dot,
        "expected_normal_dot": expected,
        "normal_quality": normal_quality,
        "area_a": area_a,
        "area_b": area_b,
        "area_ratio": ratio,
        "area_reliability": area_signal,
        "distance": distance,
        "distance_quality": distance_quality,
        "planar_align_discount": 0.75 if match["type"] == PLANAR_ALIGN else 1.0,
    }


def _score_pocket(match: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    size_a = match.get("_size_a") or []
    size_b = match.get("_size_b") or []
    comparable = min(len(size_a), len(size_b))
    if comparable:
        ratios = [
            min(abs(float(size_a[index])), abs(float(size_b[index])))
            / max(abs(float(size_a[index])), abs(float(size_b[index])), 1e-9)
            for index in range(comparable)
        ]
        size_similarity = sum(ratios) / len(ratios)
    else:
        size_similarity = float(match.get("size_similarity", 0.5))
    direction_a = match.get("_dir_a", [0, 0, 1])
    direction_b = match.get("_dir_b", [0, 0, 1])
    direction_compatibility = abs(_dot(direction_a, direction_b))
    wall_a = match.get("_wall_a", [0, 0, 1])
    wall_b = match.get("_wall_b", [0, 0, 1])
    wall_compatibility = abs(_dot(wall_a, wall_b))
    score = (
        0.50 * size_similarity
        + 0.25 * direction_compatibility
        + 0.25 * wall_compatibility
    )
    return score, {
        "size_a": size_a,
        "size_b": size_b,
        "pocket_size_similarity": size_similarity,
        "direction_compatibility": direction_compatibility,
        "wall_normal_compatibility": wall_compatibility,
    }


def score_match(
    match: dict[str, Any],
    parts_features: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    match_type = match.get("type")
    if match_type == COAXIAL:
        raw_score, reason = _score_coaxial(match, parts_features)
    elif match_type == CLEARANCE:
        raw_score, reason = _score_clearance(match, parts_features)
    elif match_type in {PLANAR_MATE, PLANAR_ALIGN}:
        raw_score, reason = _score_planar(match, parts_features)
    elif match_type == POCKET_MATE:
        raw_score, reason = _score_pocket(match)
    else:
        raw_score, reason = 0.0, {"unsupported_match_type": match_type}

    candidate_origin = match.get("candidate_origin")
    if candidate_origin:
        reason["candidate_origin"] = candidate_origin

    score = round(_clamp(raw_score), 6)
    annotated = dict(match)
    annotated["score"] = score
    annotated["confidence"] = _confidence(score)
    annotated["reason"] = reason
    return annotated


def score_matches(
    matches: list[dict[str, Any]],
    parts_features: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    return [score_match(match, parts_features) for match in matches]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder")
    parser.add_argument(
        "--output",
        help="Default: <folder>/scored_matches.json",
    )
    args = parser.parse_args()
    folder = Path(args.folder).resolve()
    step_files = sorted(
        path
        for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".step", ".stp"}
        and not path.name.lower().startswith("assembly")
    )
    parts_features = {path.name: extract_features(str(path)) for path in step_files}
    scored = score_matches(match_features(parts_features), parts_features)
    output = Path(args.output).resolve() if args.output else folder / "scored_matches.json"
    output.write_text(
        json.dumps(scored, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
