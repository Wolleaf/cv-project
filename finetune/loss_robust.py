"""
Stable matching loss for dense descriptor fine-tuning.

Replaces InfoNCE (which creates artificial competition among similar patches via
low-temperature softmax) with a multi-component loss designed for patch-level
feature matching:

  1. Positive Alignment  – gently pulls corresponding patches together
  2. Hard Negative Margin – only pushes the most confusable non-matches apart
  3. Feature Preservation – keeps LoRA-adjusted features close to DINO originals
  4. Distinctiveness Weight – down-weights ambiguous / textureless patches

Why this works when InfoNCE fails:
  - InfoNCE at τ=0.07 forces a one-hot "winner takes all" competition.
    In dense matching, many patches on the same surface are genuinely similar.
    Forcing one to win creates contradictory gradients → mode collapse.
  - This loss uses soft, margin-based objectives. It only asks "is the true
    match more similar than the hardest confuser by a margin?" — a much
    easier question that does not break semantic smoothness.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class StableMatchingLoss(nn.Module):
    """
    Multi-component loss for fine-tuning dense descriptors.

    Args:
        pos_weight:      weight of the positive-alignment term
        neg_weight:      weight of the hard-negative margin term
        preserve_weight: weight of the feature-preservation term
        margin:          required similarity gap between positive and hardest negative
        safe_radius:     spatial radius (in patches) within which negatives are ignored
        distinctiveness_min: minimum allowed distinctiveness weight
    """

    def __init__(
        self,
        pos_weight: float = 2.0,
        neg_weight: float = 1.0,
        preserve_weight: float = 0.5,
        diversity_weight: float = 0.1,
        margin: float = 0.3,
        safe_radius: float = 5.0,
        distinctiveness_min: float = 0.1,
    ):
        super().__init__()
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.preserve_weight = preserve_weight
        self.diversity_weight = diversity_weight
        self.margin = margin
        self.safe_radius = safe_radius
        self.distinctiveness_min = distinctiveness_min

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(
        self,
        desc_a: torch.Tensor,          # (N, D)  L2-normalised descriptors from A
        desc_b: torch.Tensor,          # (N, D)  L2-normalised descriptors from B
        idx_a: torch.Tensor | None = None,   # (N,) patch indices in A
        idx_b: torch.Tensor | None = None,   # (N,) patch indices in B
        batch_idx_a: torch.Tensor | None = None,  # (N,) batch membership
        batch_idx_b: torch.Tensor | None = None,
        w_patches: int = 28,
        orig_desc_a: torch.Tensor | None = None,  # (N, D) original DINO features for A
        orig_desc_b: torch.Tensor | None = None,  # (N, D) original DINO features for B
    ) -> dict[str, torch.Tensor]:
        """
        Returns dict with keys: positive, negative, preserve, diversity, total
        """
        N = desc_a.shape[0]
        device = desc_a.device

        if N <= 1:
            z = torch.tensor(0.0, device=device, requires_grad=True)
            return {"positive": z, "negative": z, "preserve": z, "diversity": z, "total": z}

        # ---- similarity matrix (cosine, since features are L2-normalised) ----
        sim = torch.mm(desc_a, desc_b.t())  # (N, N), range [-1, 1]

        # ---- 1. Positive alignment ----
        pos_sim = sim.diagonal()                            # (N,)
        pos_loss = ((1.0 - pos_sim) ** 2).mean()           # mild L2 penalty

        # ---- 2. Hard negative margin ----
        neg_loss = self._hard_negative_margin(
            sim, idx_a, batch_idx_a, w_patches, N, device
        )

        # ---- 3. Feature preservation ----
        if orig_desc_a is not None and orig_desc_b is not None:
            preserve_loss = (
                (desc_a - orig_desc_a).pow(2).mean() +
                (desc_b - orig_desc_b).pow(2).mean()
            ) / 2
        else:
            preserve_loss = torch.tensor(0.0, device=device, requires_grad=True)

        # ---- 4. Diversity regularisation (anti-collapse) ----
        # Encourage features within the same image to use the full feature space.
        div_loss = self._diversity_loss(desc_a, batch_idx_a, device)
        div_loss = div_loss + self._diversity_loss(desc_b, batch_idx_b, device)

        # ---- assemble ----
        total = (
            self.pos_weight * pos_loss +
            self.neg_weight * neg_loss +
            self.preserve_weight * preserve_loss +
            self.diversity_weight * div_loss
        )

        return {
            "positive": pos_loss,
            "negative": neg_loss,
            "preserve": preserve_loss,
            "diversity": div_loss,
            "total": total,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _hard_negative_margin(
        self,
        sim: torch.Tensor,
        idx_a: torch.Tensor | None,
        batch_idx_a: torch.Tensor | None,
        w_patches: int,
        N: int,
        device: torch.device,
    ) -> torch.Tensor:
        """For each anchor, penalise when its hardest (non-masked) negative
        comes within `margin` of the positive similarity."""
        # Start from full similarity matrix, mask the diagonal
        sim_masked = sim.clone()
        sim_masked.fill_diagonal_(float("-inf"))

        # Safe Radius mask
        if idx_a is not None and self.safe_radius > 0:
            x = (idx_a % w_patches).float()
            y = (idx_a // w_patches).float()
            dist_sq = (x.unsqueeze(1) - x.unsqueeze(0)) ** 2 + \
                      (y.unsqueeze(1) - y.unsqueeze(0)) ** 2
            radius_mask = dist_sq < self.safe_radius ** 2

            if batch_idx_a is not None:
                same_image = batch_idx_a.unsqueeze(1) == batch_idx_a.unsqueeze(0)
                radius_mask = radius_mask & same_image

            sim_masked.masked_fill_(radius_mask, float("-inf"))

        # Hardest negative (per anchor)
        hardest_neg_sim, _ = sim_masked.max(dim=1)   # (N,)
        pos_sim = sim.diagonal()

        # Margin loss: ReLU(pos - hardest_neg < margin)
        margin_loss = F.relu(self.margin - pos_sim + hardest_neg_sim)
        return margin_loss.mean()

    @staticmethod
    def _diversity_loss(
        desc: torch.Tensor,
        batch_idx: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor:
        """Penalise the mean absolute cosine similarity among descriptors
        from the *same image*, encouraging them to spread out."""
        if desc.shape[0] <= 1 or batch_idx is None:
            return torch.tensor(0.0, device=device, requires_grad=True)

        total = torch.tensor(0.0, device=device)
        n_groups = 0
        for b in batch_idx.unique():
            mask = (batch_idx == b)
            feats = desc[mask]                        # (n, D)
            if feats.shape[0] < 2:
                continue
            # Pairwise cosine similarities
            cross = torch.mm(feats, feats.t())        # (n, n), diag = 1
            # Mean absolute similarity, excluding diagonal
            n = feats.shape[0]
            off_diag = (cross.sum() - n) / (n * (n - 1) + 1e-8)
            total = total + off_diag.abs()
            n_groups += 1

        if n_groups == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        return total / n_groups


# ------------------------------------------------------------------
# Convenience builder
# ------------------------------------------------------------------

def build_stable_loss(
    pos_weight: float = 2.0,
    neg_weight: float = 1.0,
    preserve_weight: float = 0.5,
    diversity_weight: float = 0.1,
    margin: float = 0.3,
    safe_radius: float = 5.0,
) -> StableMatchingLoss:
    return StableMatchingLoss(
        pos_weight=pos_weight,
        neg_weight=neg_weight,
        preserve_weight=preserve_weight,
        diversity_weight=diversity_weight,
        margin=margin,
        safe_radius=safe_radius,
    )
