"""Evaluate CAD pair-pose scoring on held-out B-Rep contact hard negatives.

The test split is assembly-disjoint.  The evaluator uses local SE(3) gap,
slip and flip perturbations generated from the held-out recorded Pose; no
SolidWorks case, part name, mate type or functional label is read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SW_ROOT = PROJECT_ROOT / "sw"
if str(SW_ROOT) not in sys.path:
    sys.path.insert(0, str(SW_ROOT))
from learned_joint.pose_head_adapter import load_pose_heads  # noqa: E402
from learned_joint.pose_learning import CADPairPosePatchModel, contact_hard_pose_perturbations  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="test.npz with local B-Rep patches")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--negatives", type=int, default=12)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    arrays = np.load(args.dataset)
    required = ("pair_embedding", "target_pose", "free_dof_mask", "patch_a", "patch_b", "contact_reference")
    missing = [key for key in required if key not in arrays]
    if missing:
        raise ValueError(f"contact_evaluation_fields_missing:{missing}")
    mask = np.linalg.norm(arrays["target_pose"][:, :3], axis=1) <= 32.0
    tensors = [torch.from_numpy(arrays[key][mask]).float() for key in required]
    loader = DataLoader(TensorDataset(*tensors), batch_size=args.batch_size, shuffle=False)
    device = torch.device(args.device)
    model, metadata = load_pose_heads(args.checkpoint, device=str(device))
    if not isinstance(model, CADPairPosePatchModel):
        raise ValueError("patch_model_required")
    totals = {"examples": 0, "contact_examples": 0, "rank1": 0.0, "contact_rank1": 0.0, "gap_rejected": 0.0, "contact_mae": 0.0}
    with torch.no_grad():
        for embedding, pose, dof, patch_a, patch_b, reference in loader:
            embedding, pose, dof = embedding.to(device), pose.to(device), dof.to(device)
            patch_a, patch_b, reference = patch_a.to(device), patch_b.to(device), reference.to(device)
            candidates, _ = contact_hard_pose_perturbations(pose, dof, negatives=args.negatives)
            scores = model.score(embedding, patch_a, patch_b, candidates)
            targets = model.contact_targets_from_geometry(patch_a, patch_b, candidates)
            predicted = model.predict_contact(embedding, patch_a, patch_b, candidates)
            winners = scores.argmax(dim=1) == 0
            contact_mask = reference[:, 3] > 0.5
            size = int(embedding.shape[0])
            totals["examples"] += size
            totals["contact_examples"] += int(contact_mask.sum())
            totals["rank1"] += float(winners.sum())
            totals["contact_rank1"] += float(winners[contact_mask].sum())
            # candidate 1 is always a small local gap/slip perturbation.
            totals["gap_rejected"] += float((scores[:, 0] > scores[:, 1]).sum())
            totals["contact_mae"] += float((predicted - targets).abs().mean()) * size
    report = {
        "schema_version": "cad_pair_contact_hard_negative_evaluation.v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_contact_target": bool(metadata.get("contact_target")),
        "split": "heldout_fusion360_test",
        "examples": totals["examples"],
        "contact_examples": totals["contact_examples"],
        "all_hard_negative_rank1_rate": totals["rank1"] / max(1, totals["examples"]),
        "contact_subset_rank1_rate": totals["contact_rank1"] / max(1, totals["contact_examples"]),
        "small_gap_or_slip_rejection_rate": totals["gap_rejected"] / max(1, totals["examples"]),
        "contact_target_mae": totals["contact_mae"] / max(1, totals["examples"]),
        "case_specific_input": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
