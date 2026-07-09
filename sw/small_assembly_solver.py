"""Reliable solver for assemblies of at most six parts."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field
from typing import Any

from constraints import (
    COAXIAL,
    CLEARANCE,
    PLANAR_ALIGN,
    PLANAR_MATE,
    POCKET_MATE,
)
from coordinate_solver import (
    _compute_combined_placement,
    _compute_placement_from_match,
    placements_to_manifest,
)
from diagnostics import _choose_reference, _rotation_magnitude
from placement_validation import (
    bbox_collisions,
    constraint_residual,
)

COLLISION_BBOX_PENALTY_WEIGHT = 2.0
SATISFIED_INTERFACE_COLLISION_DISCOUNT = 0.05
EXPECTED_OVERLAP_TYPES = {CLEARANCE, PLANAR_MATE, POCKET_MATE}


@dataclass
class SearchState:
    placed_parts: tuple[str, ...]
    unplaced_parts: tuple[str, ...]
    placements: dict[str, dict[str, Any]]
    selected_mates: list[dict[str, Any]] = field(default_factory=list)
    score: float = 0.0
    penalty: float = 0.0
    penalty_details: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def total_score(self) -> float:
        return self.score - self.penalty


def _pair_matches(matches, a, b):
    return [match for match in matches if set(match["parts"]) == {a, b}]


def _placement_key(placement):
    translation = tuple(round(float(value), 4) for value in placement.get("translate", [0, 0, 0]))
    rotations = []
    for rotation in placement.get("rotate_sequence", []):
        if "axis_angle" in rotation:
            rotations.append(tuple(round(float(value), 4) for value in rotation["axis_angle"]))
        elif "axis_to" in rotation:
            rotations.append(json.dumps(rotation["axis_to"], sort_keys=True))
    return translation, tuple(rotations)


def _bbox_diagonal(features):
    bbox = features.get("bbox")
    if not bbox:
        return 1.0
    return max(
        1.0,
        math.sqrt(
            sum((float(bbox["max"][i]) - float(bbox["min"][i])) ** 2 for i in range(3))
        ),
    )


def _bbox_collision_penalty(collision_items, satisfied_interface_pairs):
    """Score broad-phase overlaps without suppressing valid insertions.

    AABB overlap is expected for a shaft inside a bore and often for registered
    planar contacts. Exact OCCT intersection remains the final authority. The
    discount applies only when the current pose actually satisfies a matching
    clearance/planar/pocket constraint; unrelated overlaps keep the full
    penalty.
    """
    total = 0.0
    for item in collision_items:
        pair = tuple(sorted(item["parts"]))
        discount = (
            SATISFIED_INTERFACE_COLLISION_DISCOUNT
            if pair in satisfied_interface_pairs
            else 1.0
        )
        total += (
            COLLISION_BBOX_PENALTY_WEIGHT
            * discount
            * min(1.0, float(item["minimum_part_volume_ratio"]))
        )
    return total


def _flip_placement_around_face_normal(
    placement: dict[str, Any],
    evidence: list[dict[str, Any]],
    features: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """Return a copy of *placement* rotated 180° around the matched face normal.

    Falls back to rotating around the placement's primary axis if the face
    normal cannot be determined from the evidence.
    """
    import copy as _copy2
    import math as _math2

    # Try to find the face normal from evidence
    for match in evidence:
        if match.get("type") not in {PLANAR_MATE, PLANAR_ALIGN}:
            continue
        for part_name in match["parts"]:
            if part_name not in features:
                continue
            planes = features[part_name].get("planes", [])
            feat_idx = None
            if match.get("feat_a_idx") is not None and match["parts"][0] == part_name:
                feat_idx = match["feat_a_idx"]
            elif match.get("feat_b_idx") is not None and match["parts"][1] == part_name:
                feat_idx = match["feat_b_idx"]
            if feat_idx is not None and 0 <= feat_idx < len(planes):
                normal = planes[feat_idx].get("normal", [0, 0, 1])
                # Check if normal is non-zero
                if any(abs(v) > 1e-6 for v in normal):
                    return _rotate_placement_180(placement, normal)

    # Fallback: try Z axis (most common mating face orientation)
    return _rotate_placement_180(placement, [0, 0, 1])


def _rotate_placement_180(
    placement: dict[str, Any],
    axis: list[float],
) -> dict[str, Any]:
    """Return a new placement with an added 180° rotation around *axis*."""
    import copy as _copy
    import math as _math
    new_placement = _copy.deepcopy(placement)
    rotate_seq = list(new_placement.get("rotate_sequence", []))
    # Normalize axis
    length = _math.sqrt(sum(v * v for v in axis))
    if length < 1e-9:
        return new_placement
    axis_unit = [v / length for v in axis]
    rotate_seq.append({
        "axis_angle": [axis_unit[0], axis_unit[1], axis_unit[2], 180.0]
    })
    new_placement["rotate_sequence"] = rotate_seq
    return new_placement


def _hypotheses(
    target,
    state,
    matches,
    features,
    preferred_axial_pairs=frozenset(),
):
    hypotheses = []
    seen = set()
    for reference in state.placed_parts:
        pair = _pair_matches(matches, reference, target)
        if not pair:
            continue
        pair_key = tuple(sorted((reference, target)))
        prefer_axial = (
            pair_key in preferred_axial_pairs
            and any(
                match["type"] in {COAXIAL, CLEARANCE, POCKET_MATE}
                for match in pair
            )
        )
        planar_fingerprints = {
            (
                match["type"],
                match.get("feat_a_idx"),
                match.get("feat_b_idx"),
            )
            for match in pair
            if match["type"] in {PLANAR_ALIGN, PLANAR_MATE}
        }
        # Parallel faces in local frames are alternative placements, not
        # simultaneous constraints. Combining them averages contradictory
        # offsets and commonly collapses every part onto the origin.
        has_planar_alternatives = len(planar_fingerprints) > 1
        combined = None
        if not has_planar_alternatives or prefer_axial:
            combined = _compute_combined_placement(
                pair, reference, target, features, state.placements[reference]
            )
        candidates = []
        if combined:
            candidates.append(("combined", combined, pair))
        for match in sorted(pair, key=lambda item: float(item.get("score", 0.0)), reverse=True):
            if (
                prefer_axial
                and match["type"] in {PLANAR_ALIGN, PLANAR_MATE}
            ):
                continue
            placement = _compute_placement_from_match(
                match, reference, target, features, state.placements[reference]
            )
            if placement:
                candidates.append((match["type"], placement, [match]))
        for source, placement, evidence in candidates:
            key = _placement_key(placement)
            if key in seen:
                continue
            seen.add(key)
            hypotheses.append((source, placement, evidence))
            # ── Flip hypothesis for planar constraints ──
            # A planar_mate / planar_align only constrains face orientation
            # up to 180° ambiguity around the face normal.  The solver must
            # explicitly try the flipped pose so that it can discover the
            # collision‑free side (e.g. flange facing the shaft instead of
            # facing away, or fan inserted into the cage instead of outside).
            if source in {"planar_mate", "planar_align"}:
                flip = _flip_placement_around_face_normal(placement, evidence, features)
                if flip:
                    flip_key = _placement_key(flip)
                    if flip_key not in seen:
                        seen.add(flip_key)
                        hypotheses.append((f"{source}_flipped", flip, evidence))
    return hypotheses


def _evaluate_hypothesis(target, placement, evidence, state, matches, features):
    placements = copy.deepcopy(state.placements)
    placements[target] = placement
    supporting = [
        match
        for match in matches
        if target in match["parts"]
        and any(part in state.placed_parts for part in match["parts"] if part != target)
    ]
    residual_candidates = [
        (
            next(part for part in match["parts"] if part != target),
            match,
            constraint_residual(match, features, placements),
        )
        for match in supporting
    ]
    # Candidate matches are alternative hypotheses, not simultaneous hard
    # constraints. For each already placed neighbour, retain only its best
    # satisfiable supporting edge. Penalising every unselected parallel face
    # makes correct long chains look worse as group size grows.
    best_by_neighbor = {}
    for neighbor, match, residual in residual_candidates:
        if not residual.get("valid"):
            continue
        key = (
            float(residual["residual"]),
            -float(match.get("score", 0.0)),
        )
        current = best_by_neighbor.get(neighbor)
        if current is None or key < current[0]:
            best_by_neighbor[neighbor] = (key, residual)
    residual_items = [
        item[1] for item in best_by_neighbor.values()
    ]
    valid_residuals = [
        float(item["residual"]) for item in residual_items if item.get("valid")
    ]
    mate_score = max((float(match.get("score", 0.0)) for match in evidence), default=0.0)
    multi_support = 0.1 * max(0, sum(value <= 5.0 for value in valid_residuals) - 1)
    residual_penalty = sum(min(value, 100.0) for value in valid_residuals) * 0.02
    # ── Axial reference bonus ──
    # Prefer placements that use coaxial/clearance/pocket_mate references
    # over planar-only references.  This prevents the solver from placing
    # a flange relative to another flange (planar) instead of the shaft
    # (clearance), which causes error propagation when the first flange
    # is at the wrong axial position.
    has_strong_reference = any(
        match["type"] in {COAXIAL, CLEARANCE, POCKET_MATE}
        for match in evidence
    )
    axial_bonus = 0.2 if has_strong_reference else 0.0
    # ──────────────────────────────
    weak_penalty = (
        0.15
        if supporting and all(match["type"] in {PLANAR_ALIGN, PLANAR_MATE} for match in supporting)
        else 0.0
    )
    translation = placement.get("translate", [0, 0, 0])
    translation_magnitude = math.sqrt(sum(float(value) ** 2 for value in translation))
    unrealistic = max(
        0.0,
        translation_magnitude / (10.0 * _bbox_diagonal(features[target])) - 1.0,
    )
    rotation = _rotation_magnitude(placement)
    unrealistic += max(0.0, rotation / 360.0 - 1.0)
    identity_penalty = 0.2 if _placement_key(placement) == ((0.0, 0.0, 0.0), ()) else 0.0
    collision_items = bbox_collisions(
        {part: features[part] for part in (*state.placed_parts, target)},
        placements,
    )
    satisfied_interface_pairs = {
        tuple(sorted(match["parts"]))
        for _, match, residual in residual_candidates
        if (
            match["type"] in EXPECTED_OVERLAP_TYPES
            and residual.get("valid")
            and float(residual["residual"]) <= 5.0
        )
    }
    collision_penalty = _bbox_collision_penalty(
        collision_items,
        satisfied_interface_pairs,
    )
    penalties = {
        "constraint_residual_penalty": residual_penalty,
        "collision_bbox_penalty": collision_penalty,
        "unrealistic_transform_penalty": unrealistic,
        "weak_match_penalty": weak_penalty,
        "identity_placement_penalty": identity_penalty,
    }
    return mate_score + multi_support + axial_bonus, penalties, residual_items


def solve_small_assembly(
    parts_features: dict[str, dict[str, Any]],
    matches: list[dict[str, Any]],
    *,
    beam_width: int = 20,
    target_branching: int = 1,
    placement_priority: dict[str, float] | None = None,
    preferred_axial_pairs: set[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    if not 1 <= len(parts_features) <= 6:
        raise ValueError("reliable solver supports 1..6 parts")
    if beam_width < 1:
        raise ValueError("beam_width must be positive")
    if target_branching < 1:
        raise ValueError("target_branching must be positive")
    placement_priority = placement_priority or {}
    preferred_axial_pairs = {
        tuple(sorted(pair)) for pair in (preferred_axial_pairs or set())
    }
    parts = tuple(parts_features)
    reference = _choose_reference(parts_features, matches)
    initial = SearchState(
        placed_parts=(reference,),
        unplaced_parts=tuple(part for part in parts if part != reference),
        placements={reference: {"translate": [0.0, 0.0, 0.0]}},
    )
    beam = [initial]
    best_complete = None
    complete_states = []
    best_partial = initial
    expanded_states = 0
    bound_pruned_states = 0

    while beam:
        next_beam = []
        for state in beam:
            if not state.unplaced_parts:
                complete_states.append(state)
                if best_complete is None or state.total_score > best_complete.total_score:
                    best_complete = state
                continue
            connectivity = {
                target: sum(
                    1
                    for match in matches
                    if target in match["parts"]
                    and any(part in state.placed_parts for part in match["parts"] if part != target)
                )
                for target in state.unplaced_parts
            }
            targets = sorted(
                (
                    part for part in state.unplaced_parts
                    if connectivity[part] > 0
                ),
                key=lambda part: (
                    float(placement_priority.get(part, 0.0)),
                    connectivity[part],
                    part,
                ),
                reverse=True,
            )
            if targets:
                highest_priority = float(
                    placement_priority.get(targets[0], 0.0)
                )
                if highest_priority > 0.0:
                    targets = [
                        target for target in targets
                        if float(placement_priority.get(target, 0.0))
                        == highest_priority
                    ]
            targets = targets[:target_branching]
            for target in targets:
                for source, placement, evidence in _hypotheses(
                    target,
                    state,
                    matches,
                    parts_features,
                    preferred_axial_pairs,
                ):
                    expanded_states += 1
                    gain, penalties, residuals = _evaluate_hypothesis(
                        target, placement, evidence, state, matches, parts_features
                    )
                    child = SearchState(
                        placed_parts=state.placed_parts + (target,),
                        unplaced_parts=tuple(
                            part for part in state.unplaced_parts
                            if part != target
                        ),
                        placements={
                            **copy.deepcopy(state.placements),
                            target: placement,
                        },
                        selected_mates=state.selected_mates
                        + [{
                            "target": target,
                            "source": source,
                            "evidence": evidence,
                            "residuals": residuals,
                        }],
                        score=state.score + gain,
                        penalty=state.penalty + sum(penalties.values()),
                        penalty_details=dict(state.penalty_details),
                        warnings=list(state.warnings),
                    )
                    for name, value in penalties.items():
                        child.penalty_details[name] = (
                            child.penalty_details.get(name, 0.0) + value
                        )
                    upper_bound = (
                        child.total_score + len(child.unplaced_parts) * 1.1
                    )
                    if (
                        best_complete is not None
                        and upper_bound < best_complete.total_score
                    ):
                        bound_pruned_states += 1
                        continue
                    next_beam.append(child)
                    if (
                        len(child.placed_parts) > len(best_partial.placed_parts)
                        or (
                            len(child.placed_parts)
                            == len(best_partial.placed_parts)
                            and child.total_score > best_partial.total_score
                        )
                    ):
                        best_partial = child
        if not next_beam:
            break
        beam = sorted(next_beam, key=lambda state: state.total_score, reverse=True)[:beam_width]

    complete_states.sort(key=lambda state: state.total_score, reverse=True)
    solution = (complete_states[0] if complete_states else None) or best_partial
    unsolved = [part for part in parts if part not in solution.placements]
    for part in unsolved:
        solution.placements[part] = {"translate": [0.0, 0.0, 0.0]}
    status = "success" if not unsolved else ("partial_success" if len(unsolved) < len(parts) else "failed")
    def serialize_pose(state):
        return {
            "placements": state.placements,
            "components": placements_to_manifest(
                parts_features, state.placements
            ),
            "selected_mates": state.selected_mates,
            "score": state.score,
            "penalty": state.penalty,
            "total_score": state.total_score,
            "penalty_details": state.penalty_details,
        }

    return {
        "status": status,
        "reference_part": reference,
        "placements": solution.placements,
        "components": placements_to_manifest(parts_features, solution.placements),
        "selected_mates": solution.selected_mates,
        "score": solution.score,
        "penalty": solution.penalty,
        "total_score": solution.total_score,
        "penalty_details": solution.penalty_details,
        "unsolved_parts": unsolved,
        "expanded_states": expanded_states,
        "bound_pruned_states": bound_pruned_states,
        "beam_width": beam_width,
        "target_branching": target_branching,
        "placement_priority": placement_priority,
        "preferred_axial_pairs": sorted(preferred_axial_pairs),
        "complete_pose_candidate_count": len(complete_states),
        "pose_candidates": [
            serialize_pose(state) for state in complete_states[:beam_width]
        ],
    }
