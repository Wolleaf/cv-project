"""
Contrastive losses for dense descriptor fine-tuning.

Implements:
  - InfoNCE (NT-Xent) loss on patch-level descriptors
  - Hard-negative aware sampling
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as cfg


class InfoNCELoss(nn.Module):
    """
    InfoNCE / NT-Xent loss for patch-level contrastive learning.

    Given a set of corresponding patch descriptor pairs (anchors, positives)
    from two views of the same scene, the loss pushes corresponding
    descriptors closer while pushing non-corresponding ones apart.

    The negative pairs are sampled in-batch: for each anchor, all other
    descriptors in the batch serve as negatives.
    """

    def __init__(self, temperature: float = cfg.TEMPERATURE):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        desc_a: torch.Tensor,
        desc_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            desc_a: (N, D) L2-normalised descriptors from image A (anchors).
            desc_b: (N, D) L2-normalised descriptors from image B (positives).
                    desc_a[i] and desc_b[i] are a corresponding pair.

        Returns:
            Scalar loss.
        """
        N = desc_a.shape[0]
        if N == 0:
            return torch.tensor(0.0, device=desc_a.device, requires_grad=True)

        # Similarity matrix: (N, N)
        # sim[i, j] = cos(desc_a[i], desc_b[j]) / temperature
        sim = torch.mm(desc_a, desc_b.t()) / self.temperature  # (N, N)

        # Positive pairs are on the diagonal
        labels = torch.arange(N, device=sim.device)

        # Cross-entropy loss treats the diagonal as the correct class
        loss_a2b = F.cross_entropy(sim, labels)
        loss_b2a = F.cross_entropy(sim.t(), labels)

        return (loss_a2b + loss_b2a) / 2


class HardInfoNCELoss(nn.Module):
    """
    InfoNCE with hard-negative mining.

    Same as InfoNCE but only keeps the top-K hardest negatives per anchor
    to focus learning on the most confusing pairs.
    """

    def __init__(
        self,
        temperature: float = cfg.TEMPERATURE,
        hard_neg_ratio: float = 0.5,
    ):
        super().__init__()
        self.temperature = temperature
        self.hard_neg_ratio = hard_neg_ratio

    def forward(
        self,
        desc_a: torch.Tensor,
        desc_b: torch.Tensor,
    ) -> torch.Tensor:
        N = desc_a.shape[0]
        if N <= 1:
            return torch.tensor(0.0, device=desc_a.device, requires_grad=True)

        sim = torch.mm(desc_a, desc_b.t()) / self.temperature

        # Number of hard negatives to keep
        K = max(int(N * self.hard_neg_ratio), 1)

        # For each row, keep only top-K negative similarities + the positive
        labels = torch.arange(N, device=sim.device)

        # Mask out positives temporarily to find hard negatives
        pos_sim = sim[labels, labels].clone()  # (N,)
        sim_neg = sim.clone()
        sim_neg[labels, labels] = float("-inf")

        # Top-K hardest negatives per row
        topk_vals, topk_idx = sim_neg.topk(K, dim=1)  # (N, K)

        # Build reduced logits: [positive, hard_neg_1, ..., hard_neg_K]
        logits = torch.cat([pos_sim.unsqueeze(1), topk_vals], dim=1)  # (N, 1+K)
        target = torch.zeros(N, dtype=torch.long, device=sim.device)  # positive is index 0

        return F.cross_entropy(logits, target)


class MatchingLoss(nn.Module):
    """
    Combined loss for the fine-tuning pipeline.

    Uses InfoNCE as the primary loss with an optional regularisation term
    that encourages descriptor diversity.
    """

    def __init__(
        self,
        temperature: float = cfg.TEMPERATURE,
        use_hard_negatives: bool = True,
        diversity_weight: float = 0.01,
    ):
        super().__init__()
        if use_hard_negatives:
            self.contrastive = HardInfoNCELoss(temperature)
        else:
            self.contrastive = InfoNCELoss(temperature)
        self.diversity_weight = diversity_weight

    def forward(
        self,
        desc_a: torch.Tensor,
        desc_b: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Returns:
            Dictionary with 'total', 'contrastive', and optionally 'diversity' losses.
        """
        loss_c = self.contrastive(desc_a, desc_b)
        losses = {"contrastive": loss_c}

        total = loss_c

        # Diversity regularisation: penalise if descriptors collapse
        if self.diversity_weight > 0 and desc_a.shape[0] > 1:
            # Correlation matrix of descriptors
            corr_a = torch.mm(desc_a.t(), desc_a) / desc_a.shape[0]
            corr_b = torch.mm(desc_b.t(), desc_b) / desc_b.shape[0]
            identity = torch.eye(corr_a.shape[0], device=corr_a.device)
            div_loss = (
                (corr_a - identity).pow(2).mean() +
                (corr_b - identity).pow(2).mean()
            ) / 2
            total = total + self.diversity_weight * div_loss
            losses["diversity"] = div_loss

        losses["total"] = total
        return losses
