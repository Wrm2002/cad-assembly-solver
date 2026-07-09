"""Auditable subset handling, proposal clustering, and review ordering."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any

from contracts import GroupProposal


def _parts(proposal: GroupProposal) -> frozenset[str]:
    return frozenset(proposal.parts)


def attach_subset_superset_links(
    proposals: list[GroupProposal],
) -> list[GroupProposal]:
    updates: dict[str, dict[str, list[str]]] = {
        row.group_id: {"subset_of": [], "supersets": [], "status_modifiers": []}
        for row in proposals
    }
    for left in proposals:
        left_parts = _parts(left)
        for right in proposals:
            if left.group_id == right.group_id:
                continue
            right_parts = _parts(right)
            if left_parts < right_parts:
                updates[left.group_id]["subset_of"].append(right.group_id)
                updates[right.group_id]["supersets"].append(left.group_id)
                if (
                    right.completeness_status == "family_complete"
                    and right.assembly_family == left.assembly_family
                    and len(right.parts) > len(left.parts)
                    and right.relation_coverage >= left.relation_coverage
                    and right.independent_evidence_count
                    >= left.independent_evidence_count + 1
                    and right.geometry_score >= left.geometry_score - 0.08
                    and not bool(
                        right.ranking_features.get(
                            "learned_only_critical_edge", 0.0
                        )
                    )
                ):
                    updates[left.group_id]["status_modifiers"].append(
                        "demotion_pending_pose_valid_superset"
                    )
    return [
        row.model_copy(
            update={
                "subset_of": sorted(set(updates[row.group_id]["subset_of"])),
                "supersets": sorted(set(updates[row.group_id]["supersets"])),
                "status_modifiers": sorted(
                    set(row.status_modifiers)
                    | set(updates[row.group_id]["status_modifiers"])
                ),
            }
        )
        for row in proposals
    ]


def _should_cluster(left: GroupProposal, right: GroupProposal) -> bool:
    # Family clustering is a presentation/review compression mechanism.  Do
    # not create transitive mega-clusters across incompatible templates merely
    # because one ambiguous part appeared under two role hypotheses.
    if left.assembly_family != right.assembly_family:
        return False
    left_parts, right_parts = _parts(left), _parts(right)
    overlap = left_parts & right_parts
    if not overlap:
        return False
    jaccard = len(overlap) / len(left_parts | right_parts)
    shared_center = bool(set(left.center_part_ids) & set(right.center_part_ids))
    return (
        left_parts <= right_parts
        or right_parts <= left_parts
        or jaccard >= 0.5
        or shared_center
    )


def cluster_proposals(
    proposals: list[GroupProposal],
) -> tuple[list[GroupProposal], list[dict[str, Any]]]:
    parent = {row.group_id: row.group_id for row in proposals}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    inverted: dict[str, list[GroupProposal]] = defaultdict(list)
    for proposal in proposals:
        for part in proposal.parts:
            inverted[part].append(proposal)
    compared: set[tuple[str, str]] = set()
    for candidates in inverted.values():
        for index, left in enumerate(candidates):
            for right in candidates[index + 1 :]:
                key = tuple(sorted((left.group_id, right.group_id)))
                if key in compared:
                    continue
                compared.add(key)
                if _should_cluster(left, right):
                    union(*key)

    members: dict[str, list[GroupProposal]] = defaultdict(list)
    for proposal in proposals:
        members[find(proposal.group_id)].append(proposal)

    cluster_ids = {}
    cluster_rows = []
    for component in members.values():
        ordered_ids = sorted(row.group_id for row in component)
        digest = hashlib.sha256(
            "|".join(ordered_ids).encode("utf-8")
        ).hexdigest()[:12]
        cluster_id = f"PC_{digest}"
        for group_id in ordered_ids:
            cluster_ids[group_id] = cluster_id
        cluster_rows.append(
            {
                "proposal_cluster_id": cluster_id,
                "proposal_ids": ordered_ids,
                "proposal_count": len(component),
                "families": sorted({row.assembly_family for row in component}),
                "center_part_ids": sorted(
                    {part for row in component for part in row.center_part_ids}
                ),
                "contains_complete_family_proposal": any(
                    row.completeness_status == "family_complete"
                    for row in component
                ),
            }
        )
    updated = [
        row.model_copy(
            update={"proposal_cluster_id": cluster_ids[row.group_id]}
        )
        for row in proposals
    ]
    cluster_rows.sort(key=lambda row: row["proposal_cluster_id"])
    return updated, cluster_rows


def review_rank_score(proposal: GroupProposal) -> float:
    features = proposal.ranking_features
    return (
        2.2 * float(proposal.completeness_status == "family_complete")
        + 1.8 * proposal.relation_coverage
        + 1.3 * proposal.geometry_score
        + 0.8 * min(proposal.independent_evidence_count / 6.0, 1.0)
        + 0.7 * float(features.get("mean_role_score", 0.0))
        + 0.4 * float(features.get("mean_analytic_score", 0.0))
        - 1.2 * float(bool(proposal.missing_required_relations))
        - 1.0 * float(features.get("learned_only_critical_edge", 0.0))
    )


def build_clustered_review_queue(
    proposals: list[GroupProposal],
    *,
    maximum: int = 50,
    representatives_per_cluster: int = 8,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select diverse representatives; this ordering cannot auto-accept."""

    by_cluster: dict[str, list[GroupProposal]] = defaultdict(list)
    for proposal in proposals:
        by_cluster[proposal.proposal_cluster_id or proposal.group_id].append(
            proposal
        )
    representatives = []
    deferred = []
    for cluster_id, rows in sorted(by_cluster.items()):
        ranked = sorted(
            rows,
            key=lambda row: (
                review_rank_score(row),
                row.completeness_score,
                row.independent_evidence_count,
                row.group_id,
            ),
            reverse=True,
        )
        # Clustering supplies a per-family/per-center quota; ordering inside the
        # cluster remains the explicit, auditable review score.  Independent
        # feature selectors caused arbitrary variants to displace stronger
        # complete proposals on topology-varied holdout cases.
        representatives.extend(ranked[:representatives_per_cluster])
        deferred.extend(ranked[representatives_per_cluster:])

    representatives.sort(
        key=lambda row: (
            review_rank_score(row),
            row.completeness_score,
            row.geometry_score,
            row.group_id,
        ),
        reverse=True,
    )
    selected, overflow = representatives[:maximum], representatives[maximum:]
    deferred.extend(overflow)
    queue = []
    for rank, row in enumerate(selected, 1):
        item = row.model_dump(mode="json")
        item["review_rank_score"] = round(review_rank_score(row), 8)
        item["review_rank"] = rank
        item["review_queue_state"] = "selected"
        item["affects_auto_accept"] = False
        queue.append(item)
    deferred_rows = []
    for row in sorted(
        deferred,
        key=lambda item: (review_rank_score(item), item.group_id),
        reverse=True,
    ):
        item = row.model_dump(mode="json")
        item["review_rank_score"] = round(review_rank_score(row), 8)
        item["review_rank"] = None
        item["review_queue_state"] = "deferred"
        item["affects_auto_accept"] = False
        deferred_rows.append(item)
    return queue, deferred_rows
