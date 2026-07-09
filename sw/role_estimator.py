"""Geometry-only role and center hypotheses for functional assemblies.

Evaluation semantics embedded in synthetic features are intentionally ignored.
Only measured geometry plus the provider-separated pair graph are consumed.
"""

from __future__ import annotations

from math import prod
from typing import Any

from pair_edge import canonical_pair, index_pair_edges


ROLES = (
    "base",
    "cover",
    "shaft",
    "hub",
    "key",
    "bearing",
    "housing",
    "end_cover",
    "locating_pin",
    "fastener",
    "axial_retainer",
    "bearing_retainer",
    "unknown",
)
CENTER_ROLES = {"base", "cover", "shaft", "hub", "housing"}


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _ratio(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return _clamp((value - low) / (high - low))


def _feature_vector(feature: dict[str, Any], max_volume: float) -> dict[str, float]:
    dims = sorted(float(value) for value in feature["bbox"]["size"])
    small, middle, large = (max(value, 1e-6) for value in dims)
    cylinders = feature.get("cylindrical_faces", [])
    holes = feature.get("holes", [])
    planes = feature.get("planar_faces", [])
    convex = [
        row for row in cylinders
        if row.get("parameters", {}).get("surface_polarity") == "convex"
    ]
    concave = [
        row for row in cylinders
        if row.get("parameters", {}).get("surface_polarity") == "concave"
    ]
    volume = float(feature.get("volume") or prod(dims))
    plate = _clamp(1.0 - 1.8 * small / middle)
    elongated = _clamp((large / middle - 1.2) / 2.8)
    compact_ring = _clamp(1.0 - abs(dims[1] / dims[2] - 1.0) * 3.0)
    radial_thinness = _clamp(1.0 - dims[0] / dims[2])
    return {
        "small": small,
        "middle": middle,
        "large": large,
        "volume_norm": _clamp(volume / max(max_volume, 1e-6)),
        "plate": plate,
        "elongated": elongated,
        "compact_ring": compact_ring,
        "radial_thinness": radial_thinness,
        "plane_density": _clamp(len(planes) / 12.0),
        "hole_density": _clamp(len(holes) / 6.0),
        "convex_cylinder": _clamp(len(convex) / 2.0),
        "concave_cylinder": _clamp(len(concave) / 3.0),
        "cylinder_count_norm": _clamp(len(cylinders) / 6.0),
        "hole_count_norm": _clamp(len(holes) / 6.0),
        "has_cylinder": float(bool(cylinders)),
        "has_hole": float(bool(holes)),
        "has_convex_cylinder": float(bool(convex)),
        "has_concave_cylinder": float(bool(concave)),
        "no_cylinder": float(not cylinders),
        "many_planes": _clamp(len(planes) / 8.0),
        "is_small": _clamp(1.0 - volume / max(max_volume * 0.08, 1e-6)),
    }


def _intrinsic_scores(v: dict[str, float]) -> dict[str, float]:
    plate_holes = v["plate"] * v["hole_density"]
    shaft_form = v["elongated"] * v["has_cylinder"]
    ring_form = (
        v["compact_ring"]
        * v["has_cylinder"]
        * v["has_hole"]
    )
    scores = {
        "base": (
            0.30 * v["plate"]
            + 0.25 * v["volume_norm"]
            + 0.30 * v["hole_density"]
            + 0.15 * v["many_planes"]
        ),
        "cover": (
            0.40 * v["plate"]
            + 0.30 * v["hole_density"]
            + 0.20 * v["many_planes"]
            + 0.10 * (1.0 - v["volume_norm"])
        ),
        "shaft": (
            0.52 * shaft_form
            + 0.20 * v["elongated"]
            + 0.18 * (1.0 - v["plate"])
            + 0.10 * (1.0 - v["plate"])
        ),
        "hub": (
            0.38 * ring_form
            + 0.20 * v["compact_ring"]
            + 0.25 * v["many_planes"]
            + 0.17 * v["cylinder_count_norm"]
        ),
        "key": (
            0.38 * v["is_small"]
            + 0.30 * v["elongated"]
            + 0.22 * v["no_cylinder"]
            + 0.10 * v["many_planes"]
        ),
        "bearing": (
            0.62 * ring_form
            + 0.18 * v["radial_thinness"]
            + 0.12 * (1.0 - v["many_planes"])
            + 0.08 * v["has_hole"]
        ),
        "housing": (
            0.30 * v["volume_norm"]
            + 0.24 * v["has_cylinder"]
            + 0.22 * v["hole_density"]
            + 0.14 * v["many_planes"]
            + 0.10 * (1.0 - v["elongated"])
        ),
        "end_cover": (
            0.40 * v["plate"]
            + 0.25 * v["has_hole"]
            + 0.25 * v["hole_density"]
            + 0.10 * v["many_planes"]
        ),
        "locating_pin": (
            0.50 * v["is_small"]
            + 0.30 * v["has_convex_cylinder"]
            + 0.20 * (1.0 - v["has_concave_cylinder"])
        ),
        "fastener": (
            0.45 * v["is_small"]
            + 0.30 * v["has_convex_cylinder"]
            + 0.25 * v["elongated"]
        ),
        "axial_retainer": (
            0.38 * ring_form
            + 0.28 * v["plate"]
            + 0.18 * v["has_hole"]
            + 0.16 * (1.0 - v["many_planes"])
        ),
        "bearing_retainer": (
            0.40 * ring_form
            + 0.30 * v["plate"]
            + 0.18 * v["has_hole"]
            + 0.12 * (1.0 - v["hole_density"])
        ),
    }
    # A plate with repeated holes is much more likely to be a structural plate
    # than an annular bearing/hub, even when it contains a central cylindrical face.
    if plate_holes >= 0.35:
        scores["bearing"] *= 0.35
        scores["hub"] *= 0.55
    return {role: round(_clamp(score), 8) for role, score in scores.items()}


def estimate_roles(
    part_features: list[dict[str, Any]],
    pair_edges: list[dict[str, Any]],
    *,
    top_k: int = 3,
) -> dict[str, dict[str, Any]]:
    """Return multi-label role hypotheses without reading semantic truth."""

    max_volume = max(
        (float(row.get("volume") or 0.0) for row in part_features),
        default=1.0,
    )
    pair_index = index_pair_edges(pair_edges)
    analytic_degree: dict[str, int] = {row["part_id"]: 0 for row in part_features}
    learned_only_degree: dict[str, int] = {
        row["part_id"]: 0 for row in part_features
    }
    for edge in pair_edges:
        for part in edge["parts"]:
            if edge["learned_only"]:
                learned_only_degree[part] = learned_only_degree.get(part, 0) + 1
            elif edge["best_analytic_geometry_score"] > 0.0:
                analytic_degree[part] = analytic_degree.get(part, 0) + 1
    max_degree = max(analytic_degree.values(), default=1)

    result = {}
    for feature in part_features:
        part_id = str(feature["part_id"])
        vector = _feature_vector(feature, max_volume)
        intrinsic = _intrinsic_scores(vector)
        graph_support = _clamp(analytic_degree.get(part_id, 0) / max(max_degree, 1))
        learned_risk = _clamp(
            learned_only_degree.get(part_id, 0)
            / max(analytic_degree.get(part_id, 0) + 1, 1)
        )
        combined = {
            role: round(
                _clamp(
                    0.90 * score
                    + (0.10 * graph_support if role in CENTER_ROLES else 0.0)
                ),
                8,
            )
            for role, score in intrinsic.items()
        }
        ordered = sorted(combined, key=lambda role: (-combined[role], role))
        top_roles = [
            {"role": role, "score": combined[role]}
            for role in ordered[:top_k]
        ]
        center_score = max(
            (combined[role] for role in CENTER_ROLES), default=0.0
        )
        result[part_id] = {
            "part_id": part_id,
            "role_scores": combined,
            "intrinsic_role_scores": intrinsic,
            "top_roles": top_roles,
            "center_score": round(center_score, 8),
            "analytic_graph_support": round(graph_support, 8),
            "learned_only_graph_risk": round(learned_risk, 8),
            "geometry_features": vector,
            "production_fields_used": [
                "bbox",
                "volume",
                "planar_faces",
                "cylindrical_faces",
                "holes",
                "pair_edge_provider_tags",
            ],
            "evaluation_semantics_used": False,
            "audit_trace": [
                "functional_semantics intentionally ignored",
                f"analytic_degree={analytic_degree.get(part_id, 0)}",
                f"learned_only_degree={learned_only_degree.get(part_id, 0)}",
            ],
        }
    return result


def select_center_seeds(
    role_table: dict[str, dict[str, Any]],
    *,
    maximum: int = 8,
    minimum_score: float = 0.45,
) -> list[str]:
    candidates = [
        row for row in role_table.values()
        if float(row["center_score"]) >= minimum_score
    ]
    candidates.sort(
        key=lambda row: (
            -float(row["center_score"]),
            float(row["learned_only_graph_risk"]),
            row["part_id"],
        )
    )
    return [row["part_id"] for row in candidates[:maximum]]
