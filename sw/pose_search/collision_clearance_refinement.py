"""Bounded, review-only proposals derived from exact-collision clearances.

This module deliberately does not run collision detection and does not decide
that an assembly is collision-free.  It consumes one already-produced exact
collision report and proposes at most one translation per movable part for one
refinement round.  A caller must run exact collision validation again before
using a proposal for any downstream decision.

Part identifiers are opaque mapping keys.  No filename token, case identifier,
or semantic label participates in selection or scoring.
"""

from __future__ import annotations

import copy
import math
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from typing import Any


SCHEMA_VERSION = "collision_clearance_refinement.v1"
DEFAULT_MAXIMUM_ITERATIONS = 3
HARD_MAXIMUM_ITERATIONS = 3
DEFAULT_MAXIMUM_TRANSLATION_FRACTION_OF_OBB_DIAGONAL = 0.10
HARD_MAXIMUM_TRANSLATION_FRACTION_OF_OBB_DIAGONAL = 0.10
_EPSILON = 1e-12


def _sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _vector3(value: Any, *, allow_zero: bool) -> list[float] | None:
    if not _sequence(value) or len(value) != 3:
        return None
    vector = [_finite_float(item) for item in value]
    if any(item is None for item in vector):
        return None
    result = [float(item) for item in vector]
    if not allow_zero and math.dist(result, [0.0, 0.0, 0.0]) <= _EPSILON:
        return None
    return result


def _obb_dimensions(value: Any) -> list[float] | None:
    if not isinstance(value, Mapping):
        return None
    raw = value.get("dimensions")
    if not _sequence(raw) or len(raw) != 3:
        return None
    dimensions = [_finite_float(item) for item in raw]
    if any(item is None or item <= 0.0 for item in dimensions):
        return None
    return [float(item) for item in dimensions]


def _history_rows(value: Any) -> list[Mapping[str, Any]]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        for key in (
            "candidate_interface_history",
            "interface_history",
            "history",
            "records",
        ):
            nested = value.get(key)
            if _sequence(nested):
                return [row for row in nested if isinstance(row, Mapping)]
        return [value]
    if _sequence(value):
        return [row for row in value if isinstance(row, Mapping)]
    return []


def _record_parts(record: Mapping[str, Any]) -> list[Any]:
    parts = record.get("parts")
    if _sequence(parts) and len(parts) == 2:
        return list(parts)
    first = record.get("part_a", record.get("first_part"))
    second = record.get("part_b", record.get("second_part"))
    if first is not None and second is not None:
        return [first, second]
    return []


def _same_pair(left: Sequence[Any], right: Sequence[Any]) -> bool:
    return (
        len(left) == 2
        and len(right) == 2
        and (
            (left[0] == right[0] and left[1] == right[1])
            or (left[0] == right[1] and left[1] == right[0])
        )
    )


def _matching_history(
    history: Sequence[Mapping[str, Any]], pair: Sequence[Any]
) -> list[Mapping[str, Any]]:
    return [row for row in history if _same_pair(_record_parts(row), pair)]


def _register_obb(
    output: dict[Any, Mapping[str, Any]],
    sources: dict[Any, str],
    part: Any,
    obb: Any,
    source: str,
) -> None:
    if part is None or part in output or _obb_dimensions(obb) is None:
        return
    output[part] = obb
    sources[part] = source


def _collect_part_obbs(
    placements: Mapping[Any, Any],
    history: Sequence[Mapping[str, Any]],
    explicit: Mapping[Any, Any] | None,
) -> tuple[dict[Any, Mapping[str, Any]], dict[Any, str]]:
    output: dict[Any, Mapping[str, Any]] = {}
    sources: dict[Any, str] = {}
    if isinstance(explicit, Mapping):
        for part, obb in explicit.items():
            _register_obb(output, sources, part, obb, "explicit_part_obbs")
    for part, placement in placements.items():
        if isinstance(placement, Mapping):
            _register_obb(
                output, sources, part, placement.get("obb"), "placement.obb"
            )
    for record_index, record in enumerate(history):
        for key in ("part_obbs", "obb_by_part", "obbs"):
            values = record.get(key)
            if isinstance(values, Mapping):
                for part, obb in values.items():
                    _register_obb(
                        output,
                        sources,
                        part,
                        obb,
                        f"interface_history[{record_index}].{key}",
                    )
        for part_key, obb_keys in (
            ("moving_part", ("moving_obb", "movable_obb")),
            ("movable_part", ("movable_obb", "moving_obb")),
            ("fixed_part", ("fixed_obb", "stationary_obb")),
            ("stationary_part", ("stationary_obb", "fixed_obb")),
        ):
            part = record.get(part_key)
            for obb_key in obb_keys:
                _register_obb(
                    output,
                    sources,
                    part,
                    record.get(obb_key),
                    f"interface_history[{record_index}].{obb_key}",
                )
        parts = _record_parts(record)
        if len(parts) == 2:
            for index, keys in enumerate((
                ("part_a_obb", "first_part_obb"),
                ("part_b_obb", "second_part_obb"),
            )):
                for key in keys:
                    _register_obb(
                        output,
                        sources,
                        parts[index],
                        record.get(key),
                        f"interface_history[{record_index}].{key}",
                    )
    return output, sources


def _history_moving_claims(
    rows: Sequence[Mapping[str, Any]], pair: Sequence[Any]
) -> list[Any]:
    claims: list[Any] = []
    for row in rows:
        for key in ("moving_part", "movable_part"):
            value = row.get(key)
            if value in pair and value not in claims:
                claims.append(value)
        for key in ("fixed_part", "stationary_part"):
            fixed = row.get(key)
            if fixed in pair:
                other = pair[1] if fixed == pair[0] else pair[0]
                if other not in claims:
                    claims.append(other)
    return claims


def _choose_moving_part(
    pair: Sequence[Any],
    movable_parts: set[Any],
    matching_history: Sequence[Mapping[str, Any]],
) -> tuple[Any | None, str | None, str]:
    claims = _history_moving_claims(matching_history, pair)
    if len(claims) > 1:
        return None, "ambiguous_moving_part_history", "none"
    if claims:
        if claims[0] not in movable_parts:
            return None, "history_moving_part_not_allowed", "history"
        return claims[0], None, "history"
    allowed = [part for part in pair if part in movable_parts]
    if not allowed:
        return None, "collision_pair_has_no_allowed_moving_part", "none"
    if len(allowed) == 1:
        return allowed[0], None, "allowed_set_only_member"
    # The source vector is explicitly defined for the second reported part.
    # Prefer it when both parts are movable rather than adding an arbitrary
    # filename- or case-dependent tie-break.
    return pair[1], None, "reported_second_part"


def _largest_solid_intersection(
    collision: Mapping[str, Any],
) -> tuple[Mapping[str, Any] | None, int | None, str | None]:
    rows = collision.get("solid_intersections")
    if not _sequence(rows) or not rows:
        return None, None, "no_solid_intersections"
    ranked: list[tuple[float, int, Mapping[str, Any]]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            continue
        volume = _finite_float(row.get("intersection_volume_mm3"))
        if volume is not None and volume >= 0.0:
            ranked.append((volume, index, row))
    if not ranked:
        return None, None, "no_finite_solid_intersection_volume"
    volume, index, row = max(ranked, key=lambda item: (item[0], -item[1]))
    if volume <= _EPSILON:
        return row, index, "maximum_volume_intersection_not_positive"
    vector = _vector3(
        row.get("clearance_translation_for_second_part_mm"),
        allow_zero=False,
    )
    if vector is None:
        return (
            row,
            index,
            "maximum_volume_intersection_has_no_valid_clearance_vector",
        )
    return row, index, None


def _rejection(
    collision_index: int | None,
    pair: Sequence[Any] | None,
    reason: str,
    **extra: Any,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "collision_index": collision_index,
        "parts": list(pair or []),
        "reason": reason,
    }
    result.update(extra)
    return result


def _base_result(
    *,
    placements: Any,
    iteration_index: Any,
    maximum_iterations: Any,
    report_status: Any,
    maximum_fraction: Any,
) -> dict[str, Any]:
    copied = copy.deepcopy(dict(placements)) if isinstance(placements, Mapping) else {}
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "abstain",
        "reason": "No safe clearance translation was proposed.",
        "iteration_index": iteration_index,
        "maximum_iterations": maximum_iterations,
        "source_exact_collision_status": report_status,
        "proposed_placements": copied,
        "proposals": [],
        "rejections": [],
        "rejection_reasons": [],
        "decision_reasons": ["no_safe_clearance_translation_proposal"],
        "proposal_only": True,
        "review_required": True,
        "can_auto_accept": False,
        "collision_free_claimed": False,
        "collision_status_after_proposal": "not_evaluated",
        "exact_revalidation_required": True,
        "audit": {
            "selection_rule": (
                "largest_intersection_volume_solid_per_collision_pair"
            ),
            "maximum_translation_fraction_of_obb_diagonal": maximum_fraction,
            "one_translation_vector_per_part_per_round": True,
            "maximum_iterations_hard_guard": HARD_MAXIMUM_ITERATIONS,
            "maximum_translation_fraction_hard_guard": (
                HARD_MAXIMUM_TRANSLATION_FRACTION_OF_OBB_DIAGONAL
            ),
            "part_identifiers_treated_as_opaque": True,
            "filename_or_case_id_heuristics_used": False,
            "input_placements_mutated": False,
            "exact_collision_rerun_performed": False,
        },
    }


def _finish_abstain(
    result: dict[str, Any], reason: str, *, rejection: dict[str, Any] | None = None
) -> dict[str, Any]:
    if rejection is not None:
        result["rejections"].append(rejection)
    reasons = [row["reason"] for row in result["rejections"]]
    if reason not in reasons:
        reasons.insert(0, reason)
    result["rejection_reasons"] = list(dict.fromkeys(reasons))
    result["decision_reasons"] = [
        "no_safe_clearance_translation_proposal",
        *result["rejection_reasons"],
    ]
    result["reason"] = reason
    return result


def propose_collision_clearance_refinement(
    placements: Mapping[Any, Mapping[str, Any]],
    exact_collision_report: Mapping[str, Any],
    movable_parts: Collection[Any],
    candidate_interface_history: (
        Sequence[Mapping[str, Any]] | Mapping[str, Any] | None
    ) = None,
    *,
    part_obbs: Mapping[Any, Mapping[str, Any]] | None = None,
    iteration_index: int = 0,
    maximum_iterations: int = DEFAULT_MAXIMUM_ITERATIONS,
    maximum_translation_fraction_of_obb_diagonal: float = (
        DEFAULT_MAXIMUM_TRANSLATION_FRACTION_OF_OBB_DIAGONAL
    ),
) -> dict[str, Any]:
    """Propose one conservative clearance-refinement round.

    ``clearance_translation_for_second_part_mm`` is interpreted in the world
    frame used by the placements.  If the first reported part is the only
    allowed mover, the vector is negated, which preserves the pairwise relative
    displacement.  When both parts are movable the reported second part is
    selected unless an unambiguous interface-history record names the mover.

    The function never clips an over-limit vector, never falls back from the
    maximum-volume solid intersection to a smaller one, and never emits more
    than one vector for a part in a round.
    """

    report_status = (
        exact_collision_report.get("status")
        if isinstance(exact_collision_report, Mapping)
        else None
    )
    result = _base_result(
        placements=placements,
        iteration_index=iteration_index,
        maximum_iterations=maximum_iterations,
        report_status=report_status,
        maximum_fraction=maximum_translation_fraction_of_obb_diagonal,
    )

    iteration = _finite_float(iteration_index)
    iteration_limit = _finite_float(maximum_iterations)
    fraction = _finite_float(maximum_translation_fraction_of_obb_diagonal)
    if not isinstance(placements, Mapping):
        return _finish_abstain(result, "invalid_placements_mapping")
    if not isinstance(exact_collision_report, Mapping):
        return _finish_abstain(result, "invalid_exact_collision_report")
    if (
        isinstance(movable_parts, (str, bytes, bytearray))
        or not isinstance(movable_parts, Collection)
    ):
        return _finish_abstain(result, "invalid_movable_parts_collection")
    if iteration is None or int(iteration) != iteration or iteration < 0:
        return _finish_abstain(result, "invalid_iteration_index")
    if (
        iteration_limit is None
        or int(iteration_limit) != iteration_limit
        or iteration_limit <= 0
    ):
        return _finish_abstain(result, "invalid_maximum_iterations")
    if iteration_limit > HARD_MAXIMUM_ITERATIONS:
        return _finish_abstain(
            result, "maximum_iterations_exceeds_hard_guard"
        )
    if iteration >= iteration_limit:
        return _finish_abstain(result, "fixed_iteration_limit_reached")
    if fraction is None or fraction <= 0.0:
        return _finish_abstain(result, "invalid_relative_obb_limit")
    if fraction > HARD_MAXIMUM_TRANSLATION_FRACTION_OF_OBB_DIAGONAL:
        return _finish_abstain(
            result, "relative_obb_limit_exceeds_hard_guard"
        )
    if report_status != "success":
        return _finish_abstain(
            result, "exact_collision_report_not_complete_success"
        )

    collisions = exact_collision_report.get("collisions")
    if not _sequence(collisions):
        return _finish_abstain(result, "invalid_collision_rows")
    if not collisions:
        # Absence of reported collision rows is not promoted into a new
        # collision-free claim by this proposal primitive.
        return _finish_abstain(result, "no_reported_collision_pairs")

    history = _history_rows(candidate_interface_history)
    obbs, obb_sources = _collect_part_obbs(placements, history, part_obbs)
    movable_set = set(movable_parts)
    candidate_rows: list[dict[str, Any]] = []
    matching_history_counts: dict[int, int] = {}

    for collision_index, collision in enumerate(collisions):
        if not isinstance(collision, Mapping):
            result["rejections"].append(_rejection(
                collision_index, None, "invalid_collision_row"
            ))
            continue
        pair = collision.get("parts")
        if not _sequence(pair) or len(pair) != 2 or pair[0] == pair[1]:
            result["rejections"].append(_rejection(
                collision_index, None, "invalid_collision_pair"
            ))
            continue
        pair = list(pair)
        if any(part not in placements for part in pair):
            result["rejections"].append(_rejection(
                collision_index, pair, "collision_part_missing_from_placements"
            ))
            continue

        pair_history = _matching_history(history, pair)
        matching_history_counts[collision_index] = len(pair_history)
        moving_part, moving_error, mover_source = _choose_moving_part(
            pair, movable_set, pair_history
        )
        if moving_error is not None:
            result["rejections"].append(_rejection(
                collision_index,
                pair,
                moving_error,
                matching_interface_history_count=len(pair_history),
            ))
            continue

        solid, solid_index, solid_error = _largest_solid_intersection(collision)
        if solid_error is not None:
            result["rejections"].append(_rejection(
                collision_index,
                pair,
                solid_error,
                selected_solid_intersection_index=solid_index,
            ))
            continue
        assert solid is not None and solid_index is not None
        vector_for_second = _vector3(
            solid.get("clearance_translation_for_second_part_mm"),
            allow_zero=False,
        )
        assert vector_for_second is not None
        vector = (
            vector_for_second
            if moving_part == pair[1]
            else [-value for value in vector_for_second]
        )

        obb = obbs.get(moving_part)
        dimensions = _obb_dimensions(obb)
        if dimensions is None:
            result["rejections"].append(_rejection(
                collision_index,
                pair,
                "missing_valid_moving_part_obb",
                moving_part=moving_part,
            ))
            continue
        diagonal = math.sqrt(sum(value * value for value in dimensions))
        maximum_norm = fraction * diagonal
        vector_norm = math.sqrt(sum(value * value for value in vector))
        if vector_norm > maximum_norm + _EPSILON:
            result["rejections"].append(_rejection(
                collision_index,
                pair,
                "clearance_translation_exceeds_relative_obb_limit",
                moving_part=moving_part,
                translation_norm_mm=vector_norm,
                maximum_translation_norm_mm=maximum_norm,
                moving_obb_dimensions_mm=dimensions,
            ))
            continue

        placement = placements.get(moving_part)
        if not isinstance(placement, Mapping):
            result["rejections"].append(_rejection(
                collision_index,
                pair,
                "invalid_moving_part_placement",
                moving_part=moving_part,
            ))
            continue
        current_translation = _vector3(
            placement.get("translate", [0.0, 0.0, 0.0]), allow_zero=True
        )
        if current_translation is None:
            result["rejections"].append(_rejection(
                collision_index,
                pair,
                "invalid_moving_part_translation",
                moving_part=moving_part,
            ))
            continue

        selected_volume = float(solid["intersection_volume_mm3"])
        candidate_rows.append({
            "collision_index": collision_index,
            "parts": pair,
            "moving_part": moving_part,
            "moving_part_selection_source": mover_source,
            "vector_inverted_from_second_part": moving_part == pair[0],
            "translation_mm": vector,
            "translation_norm_mm": vector_norm,
            "maximum_translation_norm_mm": maximum_norm,
            "moving_obb_dimensions_mm": dimensions,
            "moving_obb_source": obb_sources.get(moving_part),
            "selected_solid_intersection_index": solid_index,
            "selected_solid_indices": copy.deepcopy(solid.get("solid_indices")),
            "selected_intersection_volume_mm3": selected_volume,
            "matching_interface_history_count": len(pair_history),
            "current_translation_mm": current_translation,
        })

    # A part may be involved in several collision pairs.  Select the proposal
    # backed by the largest selected solid intersection and explicitly reject
    # every other vector for that part in this round.  Input report order is the
    # deterministic tie-break; opaque identifier text is never inspected.
    candidates_by_part: dict[Any, list[dict[str, Any]]] = {}
    for row in candidate_rows:
        candidates_by_part.setdefault(row["moving_part"], []).append(row)
    selected_rows: list[dict[str, Any]] = []
    for part, rows in candidates_by_part.items():
        rows.sort(key=lambda row: (
            -row["selected_intersection_volume_mm3"], row["collision_index"]
        ))
        selected_rows.append(rows[0])
        for rejected in rows[1:]:
            result["rejections"].append(_rejection(
                rejected["collision_index"],
                rejected["parts"],
                "one_translation_vector_per_part_per_round_guard",
                moving_part=part,
                selected_collision_index=rows[0]["collision_index"],
                rejected_translation_mm=rejected["translation_mm"],
            ))
    selected_rows.sort(key=lambda row: row["collision_index"])

    proposed_placements = result["proposed_placements"]
    for row in selected_rows:
        part = row["moving_part"]
        proposed = dict(proposed_placements[part])
        current = row.pop("current_translation_mm")
        proposed_translation = [
            current[index] + row["translation_mm"][index]
            for index in range(3)
        ]
        proposed["translate"] = proposed_translation
        proposed_placements[part] = proposed
        result["proposals"].append({
            **row,
            "proposed_translation_mm": proposed_translation,
            "proposal_only": True,
            "review_required": True,
            "can_auto_accept": False,
            "collision_status_after_proposal": "not_evaluated",
            "exact_revalidation_required": True,
        })

    rejection_reasons = [row["reason"] for row in result["rejections"]]
    result["rejection_reasons"] = list(dict.fromkeys(rejection_reasons))
    result["audit"].update({
        "source_collision_pair_count": len(collisions),
        "candidate_interface_history_count": len(history),
        "matching_interface_history_count_by_collision_index": (
            matching_history_counts
        ),
        "resolved_obb_source_by_part": {
            str(part): source for part, source in obb_sources.items()
        },
        "candidate_translation_count_before_part_guard": len(candidate_rows),
        "proposed_translation_count": len(result["proposals"]),
        "rejected_translation_count": len(result["rejections"]),
        "rejection_reason_counts": dict(Counter(rejection_reasons)),
        "next_iteration_index": int(iteration) + 1,
        "remaining_iteration_budget_after_proposal": max(
            0, int(iteration_limit) - int(iteration) - 1
        ),
    })
    if not result["proposals"]:
        return _finish_abstain(
            result,
            "no_safe_clearance_translation_proposal",
        )

    result["status"] = "proposed"
    result["reason"] = (
        "Review-only clearance translations proposed; exact collision "
        "revalidation has not been run."
    )
    result["decision_reasons"] = [
        "proposal_only_clearance_translation",
        "exact_collision_revalidation_required",
        *result["rejection_reasons"],
    ]
    return result


__all__ = [
    "DEFAULT_MAXIMUM_ITERATIONS",
    "DEFAULT_MAXIMUM_TRANSLATION_FRACTION_OF_OBB_DIAGONAL",
    "HARD_MAXIMUM_ITERATIONS",
    "HARD_MAXIMUM_TRANSLATION_FRACTION_OF_OBB_DIAGONAL",
    "SCHEMA_VERSION",
    "propose_collision_clearance_refinement",
]
