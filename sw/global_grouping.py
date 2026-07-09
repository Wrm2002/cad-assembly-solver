"""Deterministic, auditable global grouping over a scored candidate graph."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from functools import lru_cache
from pathlib import Path

from contracts import GroupProposal


def _pair(parts):
    return tuple(sorted(parts))


def _edge_map(candidates):
    edges = {}
    for item in candidates:
        pair = _pair(item["parts"])
        current = edges.get(pair)
        if current is None or item["geometry_score"] > current["geometry_score"]:
            edges[pair] = item
    return edges


def _mst_scores(parts, edges):
    visited = {parts[0]}
    scores, edge_ids = [], []
    while len(visited) < len(parts):
        choices = []
        for pair, edge in edges.items():
            a, b = pair
            if a not in parts or b not in parts:
                continue
            if (a in visited) ^ (b in visited):
                choices.append((float(edge["geometry_score"]), pair, edge))
        if not choices:
            return None
        score, pair, edge = max(choices, key=lambda item: (item[0], item[1]))
        visited.update(pair)
        scores.append(score)
        edge_ids.append(edge["candidate_id"])
    return scores, edge_ids


def generate_group_proposals(part_ids, candidates, config):
    edges = _edge_map(candidates)
    minimum = int(config["minimum_group_size"])
    maximum = min(int(config["maximum_group_size"]), len(part_ids))
    proposals = []
    for size in range(minimum, maximum + 1):
        for subset in itertools.combinations(sorted(part_ids), size):
            mst = _mst_scores(subset, edges)
            if mst is None:
                continue
            mst_scores, mst_edge_ids = mst
            internal = [
                edge for pair, edge in edges.items()
                if pair[0] in subset and pair[1] in subset
            ]
            possible = size * (size - 1) / 2
            density = len({_pair(edge["parts"]) for edge in internal}) / possible
            mean_mst = sum(mst_scores) / len(mst_scores)
            mean_internal = sum(
                float(edge["geometry_score"]) for edge in internal
            ) / len(internal)
            score = (
                float(config["mst_weight"]) * mean_mst
                + float(config["internal_edge_weight"]) * mean_internal
                + float(config["density_weight"]) * density
            )
            if score < float(config["minimum_group_score"]):
                continue
            group_id = "G_" + hashlib.sha256(
                "|".join(subset).encode("utf-8")
            ).hexdigest()[:12]
            proposals.append(
                GroupProposal(
                    group_id=group_id,
                    parts=list(subset),
                    candidate_edges=sorted(
                        {edge["candidate_id"] for edge in internal}
                    ),
                    geometry_score=min(1.0, score),
                    connected=True,
                    status="candidate",
                    reasons=[
                        f"mst_mean={mean_mst:.6f}",
                        f"internal_mean={mean_internal:.6f}",
                        f"edge_density={density:.6f}",
                    ],
                )
            )
    proposals.sort(
        key=lambda item: (item.geometry_score, len(item.parts), item.group_id),
        reverse=True,
    )
    return proposals[: int(config["maximum_proposals"])]


def assign_groups(part_ids, proposals, config, utility_overrides=None):
    utility_overrides = utility_overrides or {}
    index = {part: position for position, part in enumerate(sorted(part_ids))}
    entries = []
    for proposal in proposals:
        mask = sum(1 << index[part] for part in proposal.parts)
        utility = utility_overrides.get(
            proposal.group_id,
            (
                proposal.geometry_score
                - float(config["minimum_group_score"])
            )
            * len(proposal.parts),
        )
        entries.append((mask, utility, proposal))
    by_part = {part: [] for part in index}
    for entry in entries:
        for part in entry[2].parts:
            by_part[part].append(entry)

    @lru_cache(maxsize=None)
    def solve(remaining_mask):
        if not remaining_mask:
            return 0.0, ()
        bit = remaining_mask & -remaining_mask
        position = bit.bit_length() - 1
        part = sorted(index, key=index.get)[position]
        best_score, best_groups = solve(remaining_mask ^ bit)
        for mask, utility, proposal in by_part[part]:
            if mask & remaining_mask != mask:
                continue
            score, groups = solve(remaining_mask ^ mask)
            score += utility
            if score > best_score:
                best_score, best_groups = score, groups + (proposal.group_id,)
        return best_score, best_groups

    full_mask = (1 << len(index)) - 1
    objective, selected_ids = solve(full_mask)
    selected_ids = set(selected_ids)
    selected_parts = {
        part
        for proposal in proposals
        if proposal.group_id in selected_ids
        for part in proposal.parts
    }
    selected = []
    audit = []
    for proposal in proposals:
        item = proposal.model_dump(mode="json")
        if proposal.group_id in selected_ids:
            item["status"] = "selected"
            selected.append(item)
        else:
            conflicts = sorted(set(proposal.parts) & selected_parts)
            item["status"] = "not_selected"
            item["reasons"].append(
                "global objective selected a higher-utility compatible partition"
            )
            if conflicts:
                item["reasons"].append(f"conflicting_parts={','.join(conflicts)}")
        audit.append(item)
    singletons = [
        {
            "group_id": f"S_{part}",
            "parts": [part],
            "candidate_edges": [],
            "geometry_score": 0.0,
            "connected": True,
            "status": "selected_singleton",
            "reasons": ["no positive-utility selected group contained this part"],
        }
        for part in sorted(set(part_ids) - selected_parts)
    ]
    return {
        "objective": objective,
        "selected_groups": selected + singletons,
        "proposal_audit": audit,
    }


def evaluate_groups(selected, gt):
    predicted = {
        frozenset(group["parts"])
        for group in selected
        if len(group["parts"]) > 1
    }
    truth = {
        frozenset(group["parts"]) for group in gt.get("true_groups", [])
    }
    tp = len(predicted & truth)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(truth) if truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    predicted_pairs = {
        frozenset(pair)
        for group in selected
        for pair in itertools.combinations(group["parts"], 2)
    }
    truth_pairs = {
        frozenset(pair)
        for group in gt.get("true_groups", [])
        for pair in itertools.combinations(group["parts"], 2)
    }
    pair_tp = len(predicted_pairs & truth_pairs)
    pair_precision = pair_tp / len(predicted_pairs) if predicted_pairs else 0.0
    pair_recall = pair_tp / len(truth_pairs) if truth_pairs else 0.0
    pair_f1 = (
        2 * pair_precision * pair_recall / (pair_precision + pair_recall)
        if pair_precision + pair_recall
        else 0.0
    )
    return {
        "predicted_groups": len(predicted),
        "true_groups": len(truth),
        "exact_group_true_positive": tp,
        "exact_group_precision": precision,
        "exact_group_recall": recall,
        "exact_group_f1": f1,
        "predicted_copart_pairs": len(predicted_pairs),
        "true_copart_pairs": len(truth_pairs),
        "copart_pair_true_positive": pair_tp,
        "copart_pair_precision": pair_precision,
        "copart_pair_recall": pair_recall,
        "copart_pair_f1": pair_f1,
    }


def run(pool_dir, config_path):
    pool = Path(pool_dir).resolve()
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    candidates = json.loads(
        (pool / "index" / "pruned_candidates.json").read_text(encoding="utf-8")
    )
    part_ids = sorted(
        path.name
        for path in (pool / "parts").iterdir()
        if path.is_file() and path.suffix.lower() in {".step", ".stp"}
    )
    if not part_ids:
        raise FileNotFoundError(f"no STEP parts found in {pool / 'parts'}")
    proposals = generate_group_proposals(
        part_ids, candidates, config["global_grouping"]
    )
    result = assign_groups(part_ids, proposals, config["global_grouping"])
    result["schema_version"] = "1.0.0"
    result["pool_id"] = pool.name
    gt_path = pool / "pool_gt.json"
    if gt_path.is_file():
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        result["metrics"] = evaluate_groups(result["selected_groups"], gt)
        result["evaluation"] = {"available": True, "source": "pool_gt.json"}
    else:
        result["evaluation"] = {
            "available": False,
            "reason": "pool_gt.json absent; inference completed without labels",
        }
    output = pool / "grouping"
    output.mkdir(exist_ok=True)
    (output / "group_proposals.json").write_text(
        json.dumps([item.model_dump(mode="json") for item in proposals], indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "group_assignment.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pool_dir")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "configs" / "pool_pipeline.json"),
    )
    args = parser.parse_args()
    result = run(args.pool_dir, args.config)
    print(
        json.dumps(
            result.get("metrics", result["evaluation"]),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
