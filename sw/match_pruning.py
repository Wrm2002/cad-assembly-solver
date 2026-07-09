"""Auditable pruning for scored CAD mate candidates."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from constraints import CLEARANCE, COAXIAL, PLANAR_ALIGN, PLANAR_MATE, POCKET_MATE
from features import extract_features
from constraints import match_features
from match_scoring import score_matches


TYPE_PRIORITY = {
    POCKET_MATE: 5,
    CLEARANCE: 4,
    COAXIAL: 3,
    PLANAR_MATE: 2,
    PLANAR_ALIGN: 1,
}
STRONG_TYPES = {POCKET_MATE, CLEARANCE, COAXIAL}


def _pair(match: dict[str, Any]) -> tuple[str, str]:
    return tuple(sorted(str(part) for part in match["parts"]))


def _fingerprint(match: dict[str, Any]) -> tuple[Any, ...]:
    a, b = match["parts"]
    ia = match.get("feat_a_idx")
    ib = match.get("feat_b_idx")
    if a <= b:
        return match.get("type"), a, b, ia, ib
    return match.get("type"), b, a, ib, ia


def _rank(match: dict[str, Any]) -> tuple[float, int, float]:
    reason = match.get("reason", {})
    area = max(
        float(reason.get("area_a", reason.get("area_inner", 0.0)) or 0.0),
        float(reason.get("area_b", reason.get("area_outer", 0.0)) or 0.0),
    )
    return (
        float(match.get("score", 0.0)),
        TYPE_PRIORITY.get(match.get("type"), 0),
        area,
    )


def _removed(match: dict[str, Any], reason: str, detail: str | None = None) -> dict[str, Any]:
    item = dict(match)
    item["removal_reason"] = reason
    if detail:
        item["removal_detail"] = detail
    return item


def prune_match_graph(
    matches: list[dict[str, Any]],
    *,
    min_score: float = 0.5,
    top_k_pair: int = 3,
    max_neighbors: int = 4,
    planar_hypotheses_per_pair: int = 1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return ``(kept, removed)`` without mutating input matches."""
    if top_k_pair < 1:
        raise ValueError("top_k_pair must be at least 1")
    if max_neighbors < 1:
        raise ValueError("max_neighbors must be at least 1")
    if planar_hypotheses_per_pair < 1:
        raise ValueError("planar_hypotheses_per_pair must be at least 1")

    removed: list[dict[str, Any]] = []
    unique: list[dict[str, Any]] = []
    seen = set()
    for match in sorted(matches, key=_rank, reverse=True):
        fingerprint = _fingerprint(match)
        if fingerprint in seen:
            removed.append(_removed(match, "duplicate"))
            continue
        seen.add(fingerprint)
        if float(match.get("score", 0.0)) < min_score:
            removed.append(
                _removed(
                    match,
                    "low_score",
                    f"score {float(match.get('score', 0.0)):.6f} < {min_score:.6f}",
                )
            )
            continue
        unique.append(dict(match))

    by_pair: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for match in unique:
        by_pair[_pair(match)].append(match)

    pair_filtered: list[dict[str, Any]] = []
    for pair, group in sorted(by_pair.items()):
        ranked = sorted(group, key=_rank, reverse=True)
        has_strong = any(match["type"] in STRONG_TYPES for match in ranked)
        planar_only = all(match["type"] in {PLANAR_MATE, PLANAR_ALIGN} for match in ranked)

        candidates = []
        for match in ranked:
            if has_strong and match["type"] == PLANAR_ALIGN:
                removed.append(
                    _removed(match, "weak_planar_only", "planar_align dominated by strong evidence")
                )
                continue
            candidates.append(match)

        if planar_only and candidates:
            # The baseline keeps one candidate. Reliable pose search may ask
            # for several distinct face hypotheses because local-coordinate
            # geometry cannot know which of two parallel end faces will mate.
            limit = min(top_k_pair, planar_hypotheses_per_pair)
            pair_filtered.extend(candidates[:limit])
            for match in candidates[limit:]:
                removed.append(
                    _removed(
                        match,
                        "weak_planar_only",
                        f"outside top {limit} planar pose hypotheses",
                    )
                )
            continue

        # A high-scoring axial hypothesis must not consume the complete
        # per-pair budget. Registered covers often need both a repeated-hole
        # alignment and one of several planar seating hypotheses. Reserve a
        # bounded planar-mate quota while retaining at least one non-planar
        # candidate. PLANAR_ALIGN remains removable when strong evidence is
        # present because it does not establish contact.
        planar_mates = [
            match for match in candidates
            if match["type"] == PLANAR_MATE
        ]
        reserved_count = min(
            planar_hypotheses_per_pair,
            max(0, top_k_pair - 1),
            len(planar_mates),
        )
        selected_ids = {
            id(match) for match in planar_mates[:reserved_count]
        }
        selected = list(planar_mates[:reserved_count])
        for match in candidates:
            if id(match) in selected_ids:
                continue
            if len(selected) >= top_k_pair:
                break
            selected.append(match)
            selected_ids.add(id(match))
        selected = sorted(selected, key=_rank, reverse=True)
        pair_filtered.extend(selected)
        for match in candidates:
            if id(match) in selected_ids:
                continue
            removed.append(
                _removed(
                    match,
                    "exceeded_top_k_pair",
                    f"pair {pair} already retained top {top_k_pair}",
                )
            )

    pair_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for match in pair_filtered:
        pair_groups[_pair(match)].append(match)
    ranked_pairs = sorted(
        pair_groups,
        key=lambda pair: max(_rank(match) for match in pair_groups[pair]),
        reverse=True,
    )

    neighbors: dict[str, set[str]] = defaultdict(set)
    accepted_pairs = set()
    for pair in ranked_pairs:
        a, b = pair
        new_for_a = b not in neighbors[a]
        new_for_b = a not in neighbors[b]
        if (
            (new_for_a and len(neighbors[a]) >= max_neighbors)
            or (new_for_b and len(neighbors[b]) >= max_neighbors)
        ):
            for match in pair_groups[pair]:
                removed.append(
                    _removed(
                        match,
                        "exceeded_max_neighbors",
                        f"max_neighbors={max_neighbors}",
                    )
                )
            continue
        neighbors[a].add(b)
        neighbors[b].add(a)
        accepted_pairs.add(pair)

    kept = [
        match
        for pair in ranked_pairs
        if pair in accepted_pairs
        for match in sorted(pair_groups[pair], key=_rank, reverse=True)
    ]
    return kept, removed


def write_pruning_logs(
    folder: Path | str,
    kept: list[dict[str, Any]],
    removed: list[dict[str, Any]],
) -> tuple[Path, Path]:
    folder = Path(folder).resolve()
    kept_path = folder / "kept_matches.json"
    removed_path = folder / "removed_matches.json"
    kept_path.write_text(json.dumps(kept, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    removed_path.write_text(
        json.dumps(removed, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return kept_path, removed_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder")
    parser.add_argument("--min-score", type=float, default=0.5)
    parser.add_argument("--top-k-pair", type=int, default=3)
    parser.add_argument("--max-neighbors", type=int, default=4)
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
    kept, removed = prune_match_graph(
        scored,
        min_score=args.min_score,
        top_k_pair=args.top_k_pair,
        max_neighbors=args.max_neighbors,
    )
    kept_path, removed_path = write_pruning_logs(folder, kept, removed)
    print(f"kept={len(kept)} removed={len(removed)}")
    print(kept_path)
    print(removed_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
