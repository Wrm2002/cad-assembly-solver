"""
prescreen.py — Cross-group bounding-box prescreening.

Filters sub-part pairs before expensive feature extraction:
  - Only cross-parent pairs (never same parent)
  - Size-similarity scoring: max of per-axis min/max ratios.
    Mating parts often share one dominant dimension (e.g. thickness)
    while differing in width/height — max catches this, average misses it.
  - Extreme aspect-ratio filter to skip needle/wafer degenerate solids.

All thresholds are parameterised — zero hard-coded magic numbers.
"""


def prescreen_candidates(sub_parts_by_parent, similarity_threshold=0.5,
                         max_size_ratio=50.0):
    """
    Return candidate sub-part pairs sorted by size similarity (descending).

    Args:
        sub_parts_by_parent: {parent_name: [{index, bbox:(dx,dy,dz), path, ...}, ...]}
        similarity_threshold: min size-similarity score (0.0–1.0)
        max_size_ratio: skip pairs where either sub-part's own aspect ratio
            (longest_dim / shortest_dim) exceeds this — filters needle-like
            or wafer-thin degenerate solids.

    Returns:
        [(sub_a, sub_b, score), ...]  sorted highest score first
        Each sub is the original dict from sub_parts_by_parent, augmented
        with 'parent' key.
    """
    # Flatten and tag each sub with its parent
    all_subs = []
    for parent, subs in sub_parts_by_parent.items():
        for s in subs:
            s_tagged = dict(s)
            s_tagged['parent'] = parent
            all_subs.append(s_tagged)

    candidates = []

    for i in range(len(all_subs)):
        for j in range(i + 1, len(all_subs)):
            sa = all_subs[i]
            sb = all_subs[j]

            # Only cross-parent
            if sa['parent'] == sb['parent']:
                continue

            ba = sa['bbox']
            bb = sb['bbox']

            # Extreme size-ratio filter
            ratio_a = max(ba) / max(min(ba), 1e-6)
            ratio_b = max(bb) / max(min(bb), 1e-6)
            if ratio_a > max_size_ratio or ratio_b > max_size_ratio:
                continue

            # Size similarity (0–1): best-matching axis score.
            # Uses max rather than average because mating parts often
            # share one dominant dimension (e.g. thickness) while
            # differing in width/height (CPU vs socket: 73==73).
            axis_scores = []
            for k in range(3):
                mn = min(ba[k], bb[k])
                mx = max(ba[k], bb[k])
                if mx > 0.1:
                    axis_scores.append(mn / mx)
            if not axis_scores:
                continue
            score = max(axis_scores)

            if score >= similarity_threshold:
                candidates.append((sa, sb, score))

    # Sort best matches first
    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates
