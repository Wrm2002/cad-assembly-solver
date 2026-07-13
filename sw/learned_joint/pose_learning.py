"""Learning heads for CAD pair-pose proposal and interface scoring.

The frozen JoinABLe encoder supplies local B-Rep embeddings for an entity pair
and its one-ring neighbourhood.  This module deliberately does *not* consume
file names, assembly ids, joint-type strings, or SolidWorks exam metadata.

Two heads share the same geometric embedding:

* ``PoseProposalHead`` emits a small, multi-modal set of relative SE(3) poses;
* ``InterfaceScoreHead`` ranks a proposed pose against local B-Rep evidence.

The scorer is trained with generic perturbations of known Fusion 360 poses.
It is therefore an interface-pose compatibility score, not a hard-coded
mechanical-family decoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch
from torch import Tensor, nn
import torch.nn.functional as F


POSE_DIM: Final[int] = 9  # translation (normalised xyz) + Zhou 6D rotation


def rotation_6d_to_matrix(value: Tensor) -> Tensor:
    """Convert the continuous 6D rotation representation to SO(3)."""

    first = F.normalize(value[..., :3], dim=-1, eps=1e-8)
    second_raw = value[..., 3:6]
    second = F.normalize(
        second_raw - (first * second_raw).sum(dim=-1, keepdim=True) * first,
        dim=-1,
        eps=1e-8,
    )
    third = torch.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-1)


def matrix_to_rotation_6d(matrix: Tensor) -> Tensor:
    """Return the first two *columns* of a rotation matrix."""

    return torch.cat((matrix[..., :, 0], matrix[..., :, 1]), dim=-1)


def _mlp(input_dim: int, hidden_dim: int, output_dim: int, *, layers: int = 3) -> nn.Sequential:
    blocks: list[nn.Module] = []
    current = input_dim
    for _ in range(max(1, layers - 1)):
        blocks.extend((nn.Linear(current, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()))
        current = hidden_dim
    blocks.append(nn.Linear(current, output_dim))
    return nn.Sequential(*blocks)


class CADPairPoseModel(nn.Module):
    """Small heads placed after a frozen or fine-tuned JoinABLe encoder.

    ``pair_embedding`` is the concatenation of four JoinABLe embeddings:
    selected entity and one-ring pooled entity for both parts.  The architecture
    is intentionally lightweight so the original encoder can initially remain
    frozen while the new supervision is validated.
    """

    def __init__(
        self,
        embedding_dim: int = 1536,
        hidden_dim: int = 512,
        modes: int = 8,
    ) -> None:
        super().__init__()
        if modes < 2:
            raise ValueError("modes_must_be_at_least_two")
        self.embedding_dim = int(embedding_dim)
        self.modes = int(modes)
        self.trunk = _mlp(self.embedding_dim, hidden_dim, hidden_dim, layers=3)
        self.pose_modes = nn.Linear(hidden_dim, self.modes * POSE_DIM)
        self.mode_logits = nn.Linear(hidden_dim, self.modes)
        self.free_dof_logits = nn.Linear(hidden_dim, 6)
        self.interface_scorer = _mlp(hidden_dim + POSE_DIM, hidden_dim, 1, layers=3)

    def encode(self, pair_embedding: Tensor) -> Tensor:
        if pair_embedding.ndim != 2 or pair_embedding.shape[1] != self.embedding_dim:
            raise ValueError("pair_embedding_shape_mismatch")
        return self.trunk(pair_embedding)

    def propose(self, pair_embedding: Tensor) -> dict[str, Tensor]:
        latent = self.encode(pair_embedding)
        raw = self.pose_modes(latent).view(-1, self.modes, POSE_DIM)
        # Normalising the 6D part prevents an unconstrained scale from becoming
        # a shortcut for the proposal loss.
        rotations = matrix_to_rotation_6d(rotation_6d_to_matrix(raw[..., 3:]))
        poses = torch.cat((raw[..., :3], rotations), dim=-1)
        return {
            "latent": latent,
            "pose_modes": poses,
            "mode_logits": self.mode_logits(latent),
            "free_dof_logits": self.free_dof_logits(latent),
        }

    def score(self, pair_embedding: Tensor, candidate_pose: Tensor) -> Tensor:
        """Return an uncalibrated compatibility logit for each candidate pose."""

        if candidate_pose.shape[-1] != POSE_DIM:
            raise ValueError("candidate_pose_dimension_mismatch")
        latent = self.encode(pair_embedding)
        if candidate_pose.ndim == 2:
            return self.interface_scorer(torch.cat((latent, candidate_pose), dim=-1)).squeeze(-1)
        if candidate_pose.ndim != 3:
            raise ValueError("candidate_pose_rank_mismatch")
        batch, count, _ = candidate_pose.shape
        repeated = latent[:, None, :].expand(batch, count, latent.shape[-1])
        return self.interface_scorer(torch.cat((repeated, candidate_pose), dim=-1)).squeeze(-1)


class CADPairPosePatchModel(nn.Module):
    """Pose heads with candidate-conditioned local B-Rep patch evidence.

    Each patch contains points and boundary directions expressed in the
    selected entity's local frame.  For every candidate pose the scorer
    transforms patch B into frame A and derives smooth contact/normal evidence.
    This lets the network learn gap and insertion compatibility without named
    flange/key/slot factors.
    """

    def __init__(self, embedding_dim: int = 1536, hidden_dim: int = 512, modes: int = 8) -> None:
        super().__init__()
        self.embedding_dim, self.modes = int(embedding_dim), int(modes)
        self.point_encoder = _mlp(6, 64, 96, layers=3)
        self.trunk = _mlp(self.embedding_dim + 192, hidden_dim, hidden_dim, layers=3)
        self.pose_modes = nn.Linear(hidden_dim, self.modes * POSE_DIM)
        self.mode_logits = nn.Linear(hidden_dim, self.modes)
        self.free_dof_logits = nn.Linear(hidden_dim, 6)
        self.interface_scorer = _mlp(hidden_dim + POSE_DIM + 6, hidden_dim, 1, layers=3)
        # Candidate-conditioned contact targets: normalised gap, two-sided
        # coverage and normal mismatch.  These are trained from the recorded
        # B-Rep pose and synthetic geometric perturbations, never mate names.
        self.contact_predictor = _mlp(hidden_dim + POSE_DIM + 6, hidden_dim, 3, layers=3)

    def encode(self, pair_embedding: Tensor, patch_a: Tensor, patch_b: Tensor) -> Tensor:
        if pair_embedding.ndim != 2 or pair_embedding.shape[1] != self.embedding_dim:
            raise ValueError("pair_embedding_shape_mismatch")
        if patch_a.ndim != 3 or patch_a.shape[-1] != 6 or patch_b.shape != patch_a.shape:
            raise ValueError("local_patch_shape_mismatch")
        encoded_a = self.point_encoder(patch_a).amax(dim=1)
        encoded_b = self.point_encoder(patch_b).amax(dim=1)
        return self.trunk(torch.cat((pair_embedding, encoded_a, encoded_b), dim=-1))

    def propose(self, pair_embedding: Tensor, patch_a: Tensor, patch_b: Tensor) -> dict[str, Tensor]:
        latent = self.encode(pair_embedding, patch_a, patch_b)
        raw = self.pose_modes(latent).view(-1, self.modes, POSE_DIM)
        rotations = matrix_to_rotation_6d(rotation_6d_to_matrix(raw[..., 3:]))
        return {
            "latent": latent,
            "pose_modes": torch.cat((raw[..., :3], rotations), dim=-1),
            "mode_logits": self.mode_logits(latent),
            "free_dof_logits": self.free_dof_logits(latent),
        }

    @staticmethod
    def geometric_evidence(patch_a: Tensor, patch_b: Tensor, candidate_pose: Tensor) -> Tensor:
        if candidate_pose.ndim == 2:
            candidate_pose = candidate_pose[:, None, :]
        rotation = rotation_6d_to_matrix(candidate_pose[..., 3:])
        translation = candidate_pose[..., :3]
        points_a, normals_a = patch_a[..., :3], patch_a[..., 3:]
        points_b, normals_b = patch_b[..., :3], patch_b[..., 3:]
        points_b_moved = torch.einsum("bcij,bpj->bcpi", rotation, points_b) + translation[:, :, None, :]
        normals_b_moved = torch.einsum("bcij,bpj->bcpi", rotation, normals_b)
        delta = points_a[:, None, :, None, :] - points_b_moved[:, :, None, :, :]
        distances = torch.linalg.vector_norm(delta, dim=-1)
        min_a, index_b = distances.min(dim=-1)
        min_b, index_a = distances.min(dim=-2)
        # Smooth coverage remains differentiable and avoids a hard tolerance.
        coverage_a = torch.exp(-12.0 * min_a).mean(dim=-1)
        coverage_b = torch.exp(-12.0 * min_b).mean(dim=-1)
        gap_a, gap_b = min_a.mean(dim=-1), min_b.mean(dim=-1)
        gather_b = torch.gather(
            normals_b_moved,
            2,
            index_b[..., None].expand(*index_b.shape, 3),
        )
        normal_dot_a = (normals_a[:, None] * gather_b).sum(dim=-1)
        # Invalid/zero exported directions carry no normal penalty.
        valid_a = (normals_a.norm(dim=-1)[:, None] > 0.5) & (gather_b.norm(dim=-1) > 0.5)
        opposition = torch.where(valid_a, (1.0 + normal_dot_a).abs(), 0.0)
        opposition = opposition.sum(dim=-1) / valid_a.sum(dim=-1).clamp_min(1)
        centroid_gap = torch.linalg.vector_norm(
            points_a.mean(dim=1)[:, None] - points_b_moved.mean(dim=2), dim=-1
        )
        return torch.stack((gap_a, gap_b, coverage_a, coverage_b, opposition, centroid_gap), dim=-1)

    def _candidate_features(
        self, pair_embedding: Tensor, patch_a: Tensor, patch_b: Tensor, candidate_pose: Tensor
    ) -> Tensor:
        latent = self.encode(pair_embedding, patch_a, patch_b)
        poses = candidate_pose[:, None, :] if candidate_pose.ndim == 2 else candidate_pose
        evidence = self.geometric_evidence(patch_a, patch_b, poses)
        repeated = latent[:, None, :].expand(-1, poses.shape[1], -1)
        features = torch.cat((repeated, poses, evidence), dim=-1)
        return features

    @staticmethod
    def contact_targets_from_geometry(patch_a: Tensor, patch_b: Tensor, candidate_pose: Tensor) -> Tensor:
        """Continuous B-Rep contact targets for a candidate local pose.

        The target is deliberately descriptive rather than a named mate class:
        [normalised bidirectional gap, contact coverage, normal mismatch].
        """
        evidence = CADPairPosePatchModel.geometric_evidence(patch_a, patch_b, candidate_pose)
        return CADPairPosePatchModel.contact_targets_from_evidence(evidence)

    @staticmethod
    def contact_targets_from_evidence(evidence: Tensor) -> Tensor:
        """Convert cached local B-Rep evidence into continuous contact targets."""
        gap = ((evidence[..., 0] + evidence[..., 1]) * 0.5 / 0.20).clamp(0.0, 1.0)
        coverage = ((evidence[..., 2] + evidence[..., 3]) * 0.5).clamp(0.0, 1.0)
        normal_mismatch = (evidence[..., 4] / 2.0).clamp(0.0, 1.0)
        return torch.stack((gap, coverage, normal_mismatch), dim=-1)

    def score(
        self, pair_embedding: Tensor, patch_a: Tensor, patch_b: Tensor, candidate_pose: Tensor
    ) -> Tensor:
        features = self._candidate_features(pair_embedding, patch_a, patch_b, candidate_pose)
        logits = self.interface_scorer(features).squeeze(-1)
        return logits[:, 0] if candidate_pose.ndim == 2 else logits

    def predict_contact(
        self, pair_embedding: Tensor, patch_a: Tensor, patch_b: Tensor, candidate_pose: Tensor
    ) -> Tensor:
        features = self._candidate_features(pair_embedding, patch_a, patch_b, candidate_pose)
        result = torch.sigmoid(self.contact_predictor(features))
        return result[:, 0] if candidate_pose.ndim == 2 else result


@dataclass(frozen=True)
class PoseLoss:
    total: Tensor
    pose: Tensor
    mode: Tensor
    free_dof: Tensor
    winner_index: Tensor


def proposal_loss(
    proposal: dict[str, Tensor],
    target_pose: Tensor,
    target_free_dof: Tensor,
    *,
    translation_weight: float = 2.0,
    mode_weight: float = 0.25,
    dof_weight: float = 0.1,
) -> PoseLoss:
    """Best-of-K pose supervision with a calibrated winning-mode loss."""

    poses = proposal["pose_modes"]
    target = target_pose[:, None, :]
    # Fusion exports occasionally contain a numerically valid but extremely
    # distant local frame.  Smooth-L1 prevents a remaining long-tail sample
    # from dominating every mini-batch while preserving the local optimum.
    translation_error = F.smooth_l1_loss(
        poses[..., :3], target[..., :3].expand_as(poses[..., :3]), reduction="none"
    ).mean(dim=-1)
    rotation_error = F.smooth_l1_loss(
        poses[..., 3:], target[..., 3:].expand_as(poses[..., 3:]), reduction="none"
    ).mean(dim=-1)
    per_mode = float(translation_weight) * translation_error + rotation_error
    winner_error, winner_index = per_mode.min(dim=1)
    pose = winner_error.mean()
    mode = F.cross_entropy(proposal["mode_logits"], winner_index)
    free_dof = F.binary_cross_entropy_with_logits(
        proposal["free_dof_logits"], target_free_dof.float()
    )
    return PoseLoss(
        total=pose + float(mode_weight) * mode + float(dof_weight) * free_dof,
        pose=pose,
        mode=mode,
        free_dof=free_dof,
        winner_index=winner_index,
    )


def proposal_equivalence_loss(
    proposal: dict[str, Tensor],
    target_pose_modes: Tensor,
    target_mode_mask: Tensor,
    target_free_dof: Tensor,
    *,
    translation_weight: float = 2.0,
    mode_weight: float = 0.25,
    dof_weight: float = 0.1,
) -> PoseLoss:
    """Best-of-K supervision against a *set* of equivalent valid poses.

    ``target_pose_modes`` contains poses expressed in the **same selected
    entity frames as the input**.  This qualification matters: a pose derived
    from a different B-Rep entity is emitted as a separate training example by
    the dataset builder, never silently mixed into this coordinate system.

    Invalid/padded target slots are excluded before the minimum is taken.  A
    freely rotating or sliding Joint-Dataset example can therefore teach the
    proposal head that several physical SE(3) states are acceptable instead of
    treating all but the recorded occurrence as negative labels.
    """

    if target_pose_modes.ndim != 3 or target_pose_modes.shape[-1] != POSE_DIM:
        raise ValueError("target_pose_modes_shape_mismatch")
    if target_mode_mask.shape != target_pose_modes.shape[:2]:
        raise ValueError("target_mode_mask_shape_mismatch")
    poses = proposal["pose_modes"]  # [B, K, 9]
    targets = target_pose_modes[:, None, :, :]  # [B, 1, M, 9]
    proposed = poses[:, :, None, :]             # [B, K, 1, 9]
    translation_error = F.smooth_l1_loss(
        proposed[..., :3].expand(-1, -1, targets.shape[2], -1),
        targets[..., :3].expand(-1, poses.shape[1], -1, -1),
        reduction="none",
    ).mean(dim=-1)
    rotation_error = F.smooth_l1_loss(
        proposed[..., 3:].expand(-1, -1, targets.shape[2], -1),
        targets[..., 3:].expand(-1, poses.shape[1], -1, -1),
        reduction="none",
    ).mean(dim=-1)
    per_pair = float(translation_weight) * translation_error + rotation_error
    valid = target_mode_mask[:, None, :].bool()
    if not bool(target_mode_mask.any(dim=1).all()):
        raise ValueError("each_example_requires_one_equivalent_pose")
    per_pair = per_pair.masked_fill(~valid, float("inf"))
    per_mode = per_pair.amin(dim=2)
    winner_error, winner_index = per_mode.min(dim=1)
    pose = winner_error.mean()
    mode = F.cross_entropy(proposal["mode_logits"], winner_index)
    free_dof = F.binary_cross_entropy_with_logits(
        proposal["free_dof_logits"], target_free_dof.float()
    )
    return PoseLoss(
        total=pose + float(mode_weight) * mode + float(dof_weight) * free_dof,
        pose=pose,
        mode=mode,
        free_dof=free_dof,
        winner_index=winner_index,
    )


def generic_pose_perturbations(
    target_pose: Tensor,
    free_dof_mask: Tensor,
    *,
    negatives: int = 5,
    translation_std: float = 0.18,
    rotation_std: float = 0.45,
) -> tuple[Tensor, Tensor]:
    """Make pose-scoring examples without named mate templates.

    The positive is the recorded assembly pose.  Negatives perturb only the
    constrained coordinates; free dimensions remain permissible variations.
    This makes an axial sliding freedom different from an erroneous axial gap.
    """

    if negatives < 1:
        raise ValueError("negatives_must_be_positive")
    batch = target_pose.shape[0]
    constrained_translation = 1.0 - free_dof_mask[:, :3].float()
    constrained_rotation = 1.0 - free_dof_mask[:, 3:].float()
    candidates = target_pose[:, None, :].repeat(1, negatives + 1, 1)
    translation_noise = torch.randn(
        batch, negatives, 3, device=target_pose.device, dtype=target_pose.dtype
    ) * float(translation_std)
    rotation_noise = torch.randn(
        batch, negatives, 6, device=target_pose.device, dtype=target_pose.dtype
    ) * float(rotation_std)
    # The 6D representation has no coordinate-wise physical axes.  We still
    # suppress rotational perturbations for fully rotationally-free joints.
    rotationally_constrained = constrained_rotation.max(dim=-1, keepdim=True).values
    candidates[:, 1:, :3] += translation_noise * constrained_translation[:, None, :]
    candidates[:, 1:, 3:] += rotation_noise * rotationally_constrained[:, None, :]
    candidates[..., 3:] = matrix_to_rotation_6d(rotation_6d_to_matrix(candidates[..., 3:]))
    labels = torch.zeros((batch, negatives + 1), device=target_pose.device, dtype=target_pose.dtype)
    labels[:, 0] = 1.0
    return candidates, labels


def contact_hard_pose_perturbations(
    target_pose: Tensor,
    free_dof_mask: Tensor,
    *,
    negatives: int = 8,
) -> tuple[Tensor, Tensor]:
    """Near-contact gap, slip and flip negatives in local B-Rep coordinates.

    The perturbation families are generic SE(3) operations.  They do not
    encode a flange, key, hole, pocket or any case-specific mechanism.
    """
    if negatives < 1:
        raise ValueError("negatives_must_be_positive")
    batch = target_pose.shape[0]
    candidates = target_pose[:, None, :].repeat(1, negatives + 1, 1)
    constrained_translation = 1.0 - free_dof_mask[:, :3].float()
    constrained_rotation = (1.0 - free_dof_mask[:, 3:].float()).max(dim=-1).values > 0.5
    magnitudes = (0.0125, 0.03, 0.075, 0.15)
    angles = (10.0, 25.0, 55.0, 180.0)
    original_rotation = rotation_6d_to_matrix(target_pose[:, 3:])
    for offset in range(negatives):
        index = offset + 1
        axis = offset % 3
        sign = -1.0 if (offset // 3) % 2 else 1.0
        # Local normal/tangent displacement; select a constrained coordinate
        # whenever the nominal axis is free.
        usable = constrained_translation.clone()
        fallback = usable.argmax(dim=1)
        chosen = torch.where(usable[:, axis] > 0.5, torch.full_like(fallback, axis), fallback)
        translate = torch.zeros((batch, 3), device=target_pose.device, dtype=target_pose.dtype)
        translate.scatter_(1, chosen[:, None], sign * magnitudes[offset % len(magnitudes)])
        translate *= (usable.sum(dim=1, keepdim=True) > 0).to(target_pose.dtype)
        candidates[:, index, :3] += translate
        # Alternate pure translations with orientation errors.  Rotations are
        # composed in SO(3), avoiding a 6D-coordinate noise shortcut.
        if offset % 2:
            theta = torch.tensor(angles[offset % len(angles)] * torch.pi / 180.0, device=target_pose.device, dtype=target_pose.dtype)
            cosine, sine = torch.cos(theta), torch.sin(theta)
            delta = torch.eye(3, device=target_pose.device, dtype=target_pose.dtype).repeat(batch, 1, 1)
            other_a, other_b = (axis + 1) % 3, (axis + 2) % 3
            delta[:, other_a, other_a] = cosine
            delta[:, other_b, other_b] = cosine
            delta[:, other_a, other_b] = -sine
            delta[:, other_b, other_a] = sine
            moved = torch.matmul(delta, original_rotation)
            six = matrix_to_rotation_6d(moved)
            candidates[:, index, 3:] = torch.where(
                constrained_rotation[:, None], six, candidates[:, index, 3:]
            )
    candidates[..., 3:] = matrix_to_rotation_6d(rotation_6d_to_matrix(candidates[..., 3:]))
    labels = torch.zeros((batch, negatives + 1), device=target_pose.device, dtype=target_pose.dtype)
    labels[:, 0] = 1.0
    return candidates, labels


def proposal_topk_pose_error(proposal: dict[str, Tensor], target_pose: Tensor) -> Tensor:
    """Per-item minimum squared pose error among proposal modes for auditing."""

    poses = proposal["pose_modes"]
    target = target_pose[:, None, :]
    return (2.0 * (poses[..., :3] - target[..., :3]).square().mean(dim=-1) +
            (poses[..., 3:] - target[..., 3:]).square().mean(dim=-1)).min(dim=1).values
