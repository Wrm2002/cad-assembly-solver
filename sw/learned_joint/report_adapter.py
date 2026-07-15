"""Read pair-joint manifold frontiers emitted by ``joinable_e2e.py``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def load_manifold_pool(
    source: str,
    target: str,
    result_path: str | Path,
    *,
    maximum_candidates: int = 8,
    maximum_compound_candidates: int = 0,
    maximum_learned_candidates: int = 0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = Path(result_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_rows = (payload.get("joint_hypotheses") or {}).get("rows") or []
    valid_rows = []
    excluded = []
    for index, raw in enumerate(raw_rows):
        if not isinstance(raw, dict):
            excluded.append({"index": index, "reason": "not_an_object"})
            continue
        candidate = dict(raw)
        candidate.update({
            "candidate_id": f"{source}:{target}:manifold:{index:04d}",
            "source": str(source),
            "target": str(target),
        })
        valid_rows.append(candidate)
    def select_channel(candidates: list[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
        """Diversify one channel without allowing it to consume another."""
        if budget <= 0:
            return []
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for candidate in candidates:
            key = (str(candidate.get("entity_a")), str(candidate.get("entity_b")))
            grouped.setdefault(key, []).append(candidate)
        for values in grouped.values():
            values.sort(key=lambda row: (
                not bool((row.get("provenance") or {}).get("pose_search_initial")),
                not bool((row.get("provenance") or {}).get("pair_exact_collision_free")),
                -float((row.get("provenance") or {}).get("learned_pose_score", float("-inf"))),
                float((row.get("provenance") or {}).get("pair_search_cost", float("inf"))),
                abs(float(row.get("phase_degrees", 0.0))),
                -int(row.get("polarity", 1)),
                str(row.get("candidate_id")),
            ))
        selected: list[dict[str, Any]] = []
        keys = list(grouped)  # insertion order preserves the structural rank
        if keys:
            # Axis/plane direction is unoriented at the entity-recall stage.
            # Preserve both polarities of the strongest entity pair before
            # spending the whole small budget on lower-ranked pairs.  The old
            # one-row-per-pair pass silently removed the physically correct
            # polarity for already-opposed flange normals in case1.
            strongest = grouped[keys[0]]
            selected.append(strongest[0])
            opposite = next(
                (
                    row for row in strongest[1:]
                    if int(row.get("polarity", 1)) != int(strongest[0].get("polarity", 1))
                ),
                None,
            )
            if opposite is not None and len(selected) < budget:
                selected.append(opposite)
        for key in keys[1:]:
            if len(selected) >= budget:
                return selected
            selected.append(grouped[key][0])
            if len(selected) >= budget:
                return selected
        depth = 1
        while len(selected) < budget:
            added = False
            for key in keys:
                if depth < len(grouped[key]):
                    selected.append(grouped[key][depth])
                    added = True
                    if len(selected) >= budget:
                        return selected
            if not added:
                break
            depth += 1
        return selected

    def is_compound(row: dict[str, Any]) -> bool:
        provenance = row.get("provenance") or {}
        return bool(
            str(row.get("manifold_type", "")).startswith("compound_")
            or provenance.get("multi_interface_ransac")
            or provenance.get("multi_interface_prismatic")
        )

    compound = [
        row for row in valid_rows
        if is_compound(row)
        and not bool((row.get("provenance") or {}).get("learned_pose_initial"))
    ]
    compound.sort(key=lambda row: (
        not bool((row.get("provenance") or {}).get("pair_exact_collision_free")),
        -int((row.get("provenance") or {}).get("independent_evidence_count", 0) or 0),
        -float(row.get("confidence", 0.0) or 0.0),
        int(row.get("rank", 0) or 0),
    ))
    baseline = [
        row for row in valid_rows
        if not bool((row.get("provenance") or {}).get("learned_pose_initial"))
        and not is_compound(row)
    ]
    learned = [
        row for row in valid_rows
        if bool((row.get("provenance") or {}).get("learned_pose_initial"))
    ]
    baseline_rows = select_channel(baseline, max(0, int(maximum_candidates)))
    compound_rows = select_channel(
        compound, max(0, int(maximum_compound_candidates))
    )
    learned_rows = select_channel(learned, max(0, int(maximum_learned_candidates)))
    # Analytic single-interface rows, geometry-only compound rows and learned
    # rows have independent budgets.  A repeated-hole or prismatic bundle is
    # therefore additive: it cannot erase the protected analytic baseline,
    # and the learned sidecar cannot erase either geometry channel.
    rows = baseline_rows + compound_rows + learned_rows
    grouped_count = len({(str(row.get("entity_a")), str(row.get("entity_b"))) for row in valid_rows})
    return {
        "source": str(source),
        "target": str(target),
        "candidates": rows,
    }, {
        "source": str(source),
        "target": str(target),
        "result_path": str(path.resolve()),
        "input_count": len(raw_rows),
        "distinct_entity_pair_count": grouped_count,
        "retained_count": len(rows),
        "retained_baseline_count": len(baseline_rows),
        "retained_compound_count": len(compound_rows),
        "retained_learned_count": len(learned_rows),
        "baseline_protected": True,
        "compound_geometry_additive": True,
        "excluded": excluded,
    }


def build_manifold_pools(
    records: Iterable[dict[str, Any]],
    *,
    maximum_candidates_per_pair: int = 8,
    maximum_compound_candidates_per_pair: int = 0,
    maximum_learned_candidates_per_pair: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pools, audits = [], []
    for record in records:
        pool, audit = load_manifold_pool(
            str(record["source"]),
            str(record["target"]),
            record["result_path"],
            maximum_candidates=maximum_candidates_per_pair,
            maximum_compound_candidates=maximum_compound_candidates_per_pair,
            maximum_learned_candidates=maximum_learned_candidates_per_pair,
        )
        if pool["candidates"]:
            pools.append(pool)
        audits.append(audit)
    return pools, audits
