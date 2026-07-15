"""Train lightweight CAD pair-pose proposal and interface-score heads.

The JoinABLe encoder embeddings in the input dataset are frozen.  This first
training stage therefore measures whether the new supervision is useful before
any fine-tuning can damage the published pair-entity predictor.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SW_ROOT = PROJECT_ROOT / "sw"
if str(SW_ROOT) not in sys.path:
    sys.path.insert(0, str(SW_ROOT))
from learned_joint.pose_learning import (  # noqa: E402
    CADPairPoseModel,
    CADPairPosePatchModel,
    contact_hard_pose_perturbations,
    generic_pose_perturbations,
    proposal_loss,
    proposal_equivalence_loss,
    proposal_topk_pose_error,
)


class PoseArrayDataset(Dataset):
    def __init__(self, path: Path, *, max_normalized_translation: float) -> None:
        arrays = np.load(path)
        raw_target = arrays["target_pose"]
        translation_norm = np.linalg.norm(raw_target[:, :3], axis=1)
        self.keep_mask = np.isfinite(translation_norm) & (translation_norm <= max_normalized_translation)
        self.excluded_outliers = int((~self.keep_mask).sum())
        self.embedding = torch.from_numpy(arrays["pair_embedding"][self.keep_mask]).float()
        self.target = torch.from_numpy(raw_target[self.keep_mask]).float()
        self.dof = torch.from_numpy(arrays["free_dof_mask"][self.keep_mask]).float()
        self.patch_a = (
            torch.from_numpy(arrays["patch_a"][self.keep_mask]).float()
            if "patch_a" in arrays else None
        )
        self.patch_b = (
            torch.from_numpy(arrays["patch_b"][self.keep_mask]).float()
            if "patch_b" in arrays else None
        )
        self.contact_reference = (
            torch.from_numpy(arrays["contact_reference"][self.keep_mask]).float()
            if "contact_reference" in arrays else None
        )
        # v2 data keeps all pose representatives in the same selected entity
        # frame.  Alternate B-Rep entities are separate rows, never targets in
        # a mismatched local coordinate system.
        self.target_modes = (
            torch.from_numpy(arrays["target_pose_modes"][self.keep_mask]).float()
            if "target_pose_modes" in arrays else None
        )
        self.target_mode_mask = (
            torch.from_numpy(arrays["target_pose_mode_mask"][self.keep_mask]).bool()
            if "target_pose_mode_mask" in arrays else None
        )
        self.hard_negative_pose = (
            torch.from_numpy(arrays["hard_negative_pose"][self.keep_mask]).float()
            if "hard_negative_pose" in arrays else None
        )
        self.hard_negative_mask = (
            torch.from_numpy(arrays["hard_negative_mask"][self.keep_mask]).bool()
            if "hard_negative_mask" in arrays else None
        )
        if not (len(self.embedding) == len(self.target) == len(self.dof)):
            raise ValueError("pose_dataset_length_mismatch")

    def __len__(self) -> int:
        return int(self.embedding.shape[0])

    def __getitem__(self, index: int):
        suffix = ()
        if self.target_modes is not None:
            suffix = (self.target_modes[index], self.target_mode_mask[index],
                      self.hard_negative_pose[index], self.hard_negative_mask[index])
        if self.patch_a is None:
            return (self.embedding[index], self.target[index], self.dof[index], *suffix)
        if self.contact_reference is None:
            return (self.embedding[index], self.target[index], self.dof[index], self.patch_a[index], self.patch_b[index], *suffix)
        return (
            self.embedding[index], self.target[index], self.dof[index],
            self.patch_a[index], self.patch_b[index], self.contact_reference[index],
            *suffix,
        )


def _run_epoch(
    model: CADPairPoseModel | CADPairPosePatchModel,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    negatives: int,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {
        "loss": 0.0,
        "pose": 0.0,
        "score": 0.0,
        "contact": 0.0,
        "contact_rank": 0.0,
        "topk_error": 0.0,
        "score_positive_rank1_rate": 0.0,
        "examples": 0,
    }
    for batch in loader:
        embedding, target, dof = batch[:3]
        patched = isinstance(model, CADPairPosePatchModel)
        patch_a, patch_b = (batch[3], batch[4]) if patched else (None, None)
        contact_reference = batch[5] if patched else None
        # Legacy payloads contain 3 tensors (or 6 patched tensors).
        # Equivalence data adds four tensors after that payload.
        has_equivalence = len(batch) == (10 if patched else 7)
        mode_offset = 6 if patched else 3
        target_modes = target_mode_mask = hard_negative_pose = hard_negative_mask = None
        if has_equivalence:
            target_modes, target_mode_mask, hard_negative_pose, hard_negative_mask = batch[mode_offset:mode_offset + 4]
        embedding, target, dof = embedding.to(device), target.to(device), dof.to(device)
        if patch_a is not None:
            patch_a, patch_b = patch_a.to(device), patch_b.to(device)
        if contact_reference is not None:
            contact_reference = contact_reference.to(device)
        if has_equivalence:
            target_modes = target_modes.to(device)
            target_mode_mask = target_mode_mask.to(device)
            hard_negative_pose = hard_negative_pose.to(device)
            hard_negative_mask = hard_negative_mask.to(device)
        with torch.set_grad_enabled(training):
            proposal = (
                model.propose(embedding, patch_a, patch_b)
                if isinstance(model, CADPairPosePatchModel) else model.propose(embedding)
            )
            pose_terms = (proposal_equivalence_loss(proposal, target_modes, target_mode_mask, dof)
                          if has_equivalence else proposal_loss(proposal, target, dof))
            if has_equivalence:
                candidates = torch.cat((target_modes, hard_negative_pose), dim=1)
                labels = torch.cat((target_mode_mask.float(), torch.zeros_like(hard_negative_mask, dtype=torch.float)), dim=1)
                candidate_mask = torch.cat((target_mode_mask, hard_negative_mask), dim=1)
            else:
                candidates, labels = (
                    contact_hard_pose_perturbations(target, dof, negatives=negatives)
                    if isinstance(model, CADPairPosePatchModel) else
                    generic_pose_perturbations(target, dof, negatives=negatives)
                )
                candidate_mask = torch.ones_like(labels, dtype=torch.bool)
            if isinstance(model, CADPairPosePatchModel):
                # ``propose`` already encoded pair + patches once.  Reusing
                # its latent and one geometric-evidence pass avoids doing the
                # expensive B×candidate×32×32 nearest-neighbour computation
                # three times per batch (score, contact prediction, target).
                evidence = model.geometric_evidence(patch_a, patch_b, candidates)
                repeated = proposal["latent"][:, None, :].expand(-1, candidates.shape[1], -1)
                candidate_features = torch.cat((repeated, candidates, evidence), dim=-1)
                score_logits = model.interface_scorer(candidate_features).squeeze(-1)
            else:
                score_logits = model.score(embedding, candidates)
            score_elements = torch.nn.functional.binary_cross_entropy_with_logits(score_logits, labels, reduction="none")
            score_loss = (score_elements * candidate_mask.float()).sum() / candidate_mask.float().sum().clamp_min(1.0)
            contact_loss = torch.zeros((), device=device)
            contact_rank_loss = torch.zeros((), device=device)
            if isinstance(model, CADPairPosePatchModel):
                predicted_contact = torch.sigmoid(model.contact_predictor(candidate_features))
                target_contact = model.contact_targets_from_evidence(evidence).detach()
                # Candidate zero is the unperturbed occurrence-relative pose.
                # Use its measured transformed-OBJ face gap/coverage/normal
                # target directly; derive only the synthetic alternatives from
                # the same B-Rep geometry.  This prevents a Joint-Dataset
                # coordinate frame from becoming the final contact truth.
                if contact_reference is not None and not has_equivalence:
                    target_contact[:, 0, :] = contact_reference[:, :3].clamp(0.0, 1.0)
                contact_elements = torch.nn.functional.smooth_l1_loss(predicted_contact, target_contact, reduction="none").mean(dim=-1)
                contact_loss = (contact_elements * candidate_mask.float()).sum() / candidate_mask.float().sum().clamp_min(1.0)
                # Only known, dense real contacts impose a strong preference
                # over near-gap/slip/flip alternatives.  Other joint types are
                # not incorrectly forced into a planar contact interpretation.
                contact_mask = contact_reference[:, 3] > 0.5 if contact_reference is not None else torch.zeros_like(labels[:, 0], dtype=torch.bool)
                if contact_mask.any():
                    if has_equivalence:
                        positive_count = target_modes.shape[1]
                        best_positive = score_logits[:, :positive_count].masked_fill(~target_mode_mask, float("-inf")).amax(dim=1, keepdim=True)
                        valid_negative = hard_negative_mask
                        margin = 0.75 - best_positive + score_logits[:, positive_count:]
                        contact_rank_loss = (torch.relu(margin) * valid_negative.float())[contact_mask].sum() / valid_negative[contact_mask].float().sum().clamp_min(1.0)
                    else:
                        margin = 0.75 - score_logits[contact_mask, :1] + score_logits[contact_mask, 1:]
                        contact_rank_loss = torch.relu(margin).mean()
            loss = pose_terms.total + score_loss + 0.50 * contact_loss + 0.75 * contact_rank_loss
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        size = int(embedding.shape[0])
        totals["loss"] += float(loss.detach()) * size
        totals["pose"] += float(pose_terms.pose.detach()) * size
        totals["score"] += float(score_loss.detach()) * size
        totals["contact"] += float(contact_loss.detach()) * size
        totals["contact_rank"] += float(contact_rank_loss.detach()) * size
        totals["topk_error"] += float(proposal_topk_pose_error(proposal, target).mean().detach()) * size
        ranked_logits = score_logits.masked_fill(~candidate_mask, float("-inf"))
        if has_equivalence:
            chosen = ranked_logits.argmax(dim=1)
            rank1_positive = labels.gather(1, chosen[:, None]).squeeze(1) > 0.5
        else:
            rank1_positive = ranked_logits.argmax(dim=1) == 0
        totals["score_positive_rank1_rate"] += float(rank1_positive.float().mean().detach()) * size
        totals["examples"] += size
    count = max(1, totals.pop("examples"))
    return {key: value / count for key, value in totals.items()} | {"examples": count}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--modes", type=int, default=8)
    parser.add_argument("--negatives", type=int, default=5)
    parser.add_argument("--max-normalized-translation", type=float, default=32.0)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--patch-geometry", action="store_true")
    parser.add_argument("--contact-target", action="store_true")
    parser.add_argument(
        "--geometry-only-zero-embedding", action="store_true",
        help=("Train/infer from local B-Rep patches only.  The retained embedding "
              "slot is explicitly zeroed rather than carrying JoinABLe features."),
    )
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    train_data = PoseArrayDataset(
        args.dataset_dir / "train.npz", max_normalized_translation=args.max_normalized_translation
    )
    dev_data = PoseArrayDataset(
        args.dataset_dir / "dev.npz", max_normalized_translation=args.max_normalized_translation
    )
    test_data = PoseArrayDataset(
        args.dataset_dir / "test.npz", max_normalized_translation=args.max_normalized_translation
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, generator=generator)
    dev_loader = DataLoader(dev_data, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    if args.patch_geometry and train_data.patch_a is None:
        parser.error("--patch-geometry requires patch_a/patch_b arrays")
    if args.contact_target and train_data.contact_reference is None:
        parser.error("--contact-target requires contact_reference arrays")
    if args.contact_target and not args.patch_geometry:
        parser.error("--contact-target requires --patch-geometry")
    if args.geometry_only_zero_embedding and not args.patch_geometry:
        parser.error("--geometry-only-zero-embedding requires --patch-geometry")
    model = (
        CADPairPosePatchModel(embedding_dim=int(train_data.embedding.shape[1]), modes=args.modes)
        if args.patch_geometry else
        CADPairPoseModel(embedding_dim=int(train_data.embedding.shape[1]), modes=args.modes)
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(model, train_loader, device=device, optimizer=optimizer, negatives=args.negatives)
        with torch.no_grad():
            dev_metrics = _run_epoch(model, dev_loader, device=device, optimizer=None, negatives=args.negatives)
        row = {"epoch": epoch, "train": train_metrics, "dev": dev_metrics}
        history.append(row)
        print(json.dumps(row), flush=True)
        if dev_metrics["loss"] < best:
            best = dev_metrics["loss"]
            torch.save({
                "schema_version": "cad_pair_pose_and_interface_heads.v1",
                "state_dict": model.state_dict(),
                "embedding_dim": model.embedding_dim,
                "modes": model.modes,
                "patch_geometry": bool(args.patch_geometry),
                "contact_target": bool(args.contact_target),
                "geometry_only_zero_embedding": bool(args.geometry_only_zero_embedding),
                "epoch": epoch,
                "dev_metrics": dev_metrics,
                "training_contract": {
                    "encoder": "frozen official JoinABLe B-Rep embeddings",
                    "forbidden_inputs": ["file_name", "part_name", "case_id", "bom", "solidworks_answer", "joint_type"],
                    "solidworks_exam_used_for_training": False,
                },
            }, args.output_dir / "best.pt")
    checkpoint = torch.load(args.output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        test_metrics = _run_epoch(model, test_loader, device=device, optimizer=None, negatives=args.negatives)
    report = {
        "schema_version": "cad_pair_pose_and_interface_training_report.v1",
        "device": str(device),
        "patch_geometry": bool(args.patch_geometry),
        "contact_target": bool(args.contact_target),
        "geometry_only_zero_embedding": bool(args.geometry_only_zero_embedding),
        "train_examples": len(train_data),
        "dev_examples": len(dev_data),
        "test_examples": len(test_data),
        "excluded_pose_outliers": {
            "rule": "normalised_local_translation_norm <= %.3f" % args.max_normalized_translation,
            "train": train_data.excluded_outliers,
            "dev": dev_data.excluded_outliers,
            "test": test_data.excluded_outliers,
        },
        "best_epoch": checkpoint["epoch"],
        "best_dev": checkpoint["dev_metrics"],
        "test": test_metrics,
        "history": history,
        "interpretation": {
            "topk_error": "minimum normalised local-frame SE(3) proxy error among proposal modes; lower is better",
            "score": "binary loss separating recorded pose from generic local SE(3) hard perturbations; lower is better",
            "contact": "regression loss for B-Rep local gap, coverage and normal mismatch; lower is better",
            "not_a_claim": "Contact supervision improves physical fit ranking but does not by itself prove functional semantic correctness on SolidWorks exams.",
        },
    }
    (args.output_dir / "training_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"best_epoch": checkpoint["epoch"], "test": test_metrics}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
