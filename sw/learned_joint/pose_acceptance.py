"""Conservative physical gate for globally solved CAD poses."""

from __future__ import annotations

import math
from typing import Any


def contact_supported_exact_pose(
    hypothesis: dict[str, Any], *, maximum_selected_gap_normalized: float = 0.05
) -> dict[str, Any]:
    """Require exact non-collision *and* support on every selected graph edge.

    OCCT Boolean validation only proves that parts do not severely intersect.
    A separated assembly can therefore be ``exact_validation=valid``.  This
    gate uses the generic sampled surface audit to reject that false success;
    it does not use part names, case ids or mechanical-family labels.
    """
    exact_status = str((hypothesis.get("exact_validation") or {}).get("status", "not_checked"))
    audit = hypothesis.get("geometry_residual_audit") or {}
    selected = [
        row for row in (audit.get("pair_scores") or [])
        if bool(row.get("selected_constraint_edge"))
    ]
    gaps = []
    for row in selected:
        try:
            value = float(row.get("contact_gap_normalized"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            gaps.append(value)
    maximum_gap = max(gaps, default=float("inf"))
    supported = (
        exact_status == "valid"
        and bool(selected)
        and len(gaps) == len(selected)
        and maximum_gap <= float(maximum_selected_gap_normalized)
    )
    reason = (
        "exact_noncollision_and_all_selected_edges_contact_supported"
        if supported else
        "occt_valid_but_selected_edges_are_separated"
        if exact_status == "valid" else
        f"exact_validation_{exact_status}"
    )
    return {
        "status": "valid" if supported else "review" if exact_status == "valid" else "failed",
        "exact_status": exact_status,
        "selected_edge_count": len(selected),
        "maximum_selected_contact_gap_normalized": maximum_gap if math.isfinite(maximum_gap) else None,
        "threshold": float(maximum_selected_gap_normalized),
        "reason": reason,
    }
