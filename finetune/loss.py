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
    InfoNCE with hard-negative mining and Safe Radius.
    
    Crucial Fix: When batch_size=1, negatives are sampled from the same image.
    We MUST mask out spatially close patches (Safe Radius) so the model is not
    forced to push adjacent (similar) patches apart, which causes mode collapse.
    """

    def __init__(
        self,
        temperature: float = cfg.TEMPERATURE,
        hard_neg_ratio: float = 0.5,
        safe_radius: float = 5.0,  # in patches
    ):
        super().__init__()
        self.temperature = temperature
        self.hard_neg_ratio = hard_neg_ratio
        self.safe_radius = safe_radius

    def forward(
        self,
        desc_a: torch.Tensor,
        desc_b: torch.Tensor,
        idx_a: torch.Tensor | None = None,
        idx_b: torch.Tensor | None = None,
        w_patches: int = 28,
    ) -> torch.Tensor:
        N = desc_a.shape[0]
        if N <= 1:
            return torch.tensor(0.0, device=desc_a.device, requires_grad=True)

        sim = torch.mm(desc_a, desc_b.t()) / self.temperature

        # Safe Radius Masking
        if idx_a is not None and self.safe_radius > 0:
            # Convert 1D patch index to 2D grid coordinates
            x_a = (idx_a % w_patches).float()
            y_a = (idx_a // w_patches).float()
            # Compute pairwise distance matrix
            dist_sq = (x_a.unsqueeze(1) - x_a.unsqueeze(0))**2 + (y_a.unsqueeze(1) - y_a.unsqueeze(0))**2
            # Mask out points closer than safe_radius (except diagonal which is the positive pair)
            mask = (dist_sq < self.safe_radius**2)
            mask.fill_diagonal_(False)
            sim.masked_fill_(mask, float('-inf'))

        K = max(int(N * self.hard_neg_ratio), 1)
        labels = torch.arange(N, device=sim.device)

        pos_sim = sim[labels, labels].clone()
        sim_neg = sim.clone()
        sim_neg[labels, labels] = float("-inf")

        topk_vals, topk_idx = sim_neg.topk(K, dim=1)
        logits = torch.cat([pos_sim.unsqueeze(1), topk_vals], dim=1)
        target = torch.zeros(N, dtype=torch.long, device=sim.device)

        return F.cross_entropy(logits, target)


class MatchingLoss(nn.Module):
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
        idx_a: torch.Tensor | None = None,
        idx_b: torch.Tensor | None = None,
        patch_size: int = 16,
        img_size: int = 448,
    ) -> dict[str, torch.Tensor]:
        
        w_patches = img_size // patch_size
        
        if isinstance(self.contrastive, HardInfoNCELoss):
            loss_c = self.contrastive(desc_a, desc_b, idx_a, idx_b, w_patches)
        else:
            loss_c = self.contrastive(desc_a, desc_b)
            
        losses = {"contrastive": loss_c}
        total = loss_c

        if self.diversity_weight > 0 and desc_a.shape[0] > 1:
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
