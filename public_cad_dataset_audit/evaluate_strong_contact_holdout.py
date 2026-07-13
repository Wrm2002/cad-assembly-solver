"""Evaluate real-contact versus near-gap/slip/flip pose ranking on assembly holdout.

The held-out unit is an Assembly Dataset assembly, never an individual contact
pair.  Candidate zero is the recorded occurrence-relative contact pose; every
other candidate is a generic local SE(3) perturbation.  This reports the
limited but important claim needed before any SolidWorks exam: whether the
model ranks the recorded contact over small geometric near misses.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "sw") not in sys.path:
    sys.path.insert(0, str(ROOT / "sw"))
from learned_joint.pose_learning import CADPairPosePatchModel, contact_hard_pose_perturbations  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="held-out test.npz")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--negatives", type=int, default=8)
    args = parser.parse_args()
    torch.manual_seed(20260712)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not payload.get("patch_geometry"):
        raise ValueError("strong_contact_evaluation_requires_patch_checkpoint")
    arrays = np.load(args.dataset)
    model = CADPairPosePatchModel(int(payload["embedding_dim"]), modes=int(payload["modes"]))
    model.load_state_dict(payload["state_dict"], strict=bool(payload.get("contact_target")))
    model.eval()
    embedding = torch.from_numpy(arrays["pair_embedding"]).float()
    target = torch.from_numpy(arrays["target_pose"]).float()
    dof = torch.from_numpy(arrays["free_dof_mask"]).float()
    patch_a = torch.from_numpy(arrays["patch_a"]).float()
    patch_b = torch.from_numpy(arrays["patch_b"]).float()
    reference = torch.from_numpy(arrays["contact_reference"]).float()
    rank1, ranks, gaps = [], [], []
    with torch.inference_mode():
        for start in range(0, len(target), args.batch_size):
            end = min(len(target), start + args.batch_size)
            candidates, _ = contact_hard_pose_perturbations(target[start:end], dof[start:end], negatives=args.negatives)
            scores = model.score(embedding[start:end], patch_a[start:end], patch_b[start:end], candidates)
            order = torch.argsort(scores, dim=1, descending=True)
            ranks.extend((order == 0).nonzero(as_tuple=False)[:, 1].add(1).tolist())
            rank1.extend((scores.argmax(dim=1) == 0).tolist())
            gaps.extend(reference[start:end, 0].tolist())
    ranks_np, gaps_np = np.asarray(ranks), np.asarray(gaps)
    near_mask = gaps_np <= max(0.01, float(np.quantile(gaps_np, .5)))
    report = {
        "schema_version": "strong_contact_holdout.v1",
        "unit_of_split": "assembly_id",
        "examples": int(len(ranks_np)),
        "candidate_contract": "recorded contact pose vs generic small-gap/slip/flip/penetration local SE(3) perturbations",
        "overall": {
            "true_contact_rank1_rate": float(np.mean(rank1)),
            "mean_true_contact_rank": float(ranks_np.mean()),
            "top3_rate": float(np.mean(ranks_np <= 3)),
        },
        "dense_contact_subset": {
            "examples": int(near_mask.sum()),
            "true_contact_rank1_rate": float(np.mean(np.asarray(rank1)[near_mask])) if near_mask.any() else None,
            "mean_true_contact_rank": float(ranks_np[near_mask].mean()) if near_mask.any() else None,
            "gap_threshold_normalised": float(max(0.01, np.quantile(gaps_np, .5))),
        },
        "interpretation": "A high rank-1 rate establishes local physical-contact discrimination on unseen assemblies only. It does not establish functional semantics or guarantee a SolidWorks case pose.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
