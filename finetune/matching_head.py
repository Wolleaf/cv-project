"""
Frozen DINOv3 + Learnable Attention-Based Matching Head — Route B.

This is fundamentally different from all previous approaches in this project:
  - DINOv3 backbone is COMPLETELY FROZEN (zero gradient touches it).
  - A lightweight Transformer-based matching head learns to refine descriptors
    using self-attention (within-image context) and cross-attention (between-image
    matching).  This directly addresses why Route A (Matchability Predictor) failed:
    a single patch in isolation cannot judge its own distinctiveness; it needs
    self-attention over the whole image to discover whether it sits in a
    textureless region surrounded by near-identical neighbours.

Training directly optimises matching accuracy through a dual-softmax loss —
not feature similarity, not BCE on pseudo-labels.

Usage:
  python -m finetune.matching_head train \\
      --checkpoint dinov3_weights/...pth \\
      --train_pairs finetune/navi_train_pairs.txt \\
      --data_root full_dataset/navi_v1.5 \\
      --depth_root full_dataset/navi_v1.5 \\
      --output_dir matching_head_navi \\
      --epochs 20 --batch_pairs 4

  python -m finetune.matching_head extract \\
      --checkpoint dinov3_weights/...pth \\
      --matcher matching_head_navi/matcher_best.pth \\
      --pairs datasets/navi_with_gt.txt \\
      --data_root datasets/test/navi_resized \\
      --output_dir mnn_matching_head_navi/navi \\
      --img_size 448 --eval_resize 640 480
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from .config import VIT_PATCH_SIZE
from .model import DINOv3Backbone
from .dataset import MatchingPairDataset, collate_matching_pairs


# =========================================================================
# Positional Encoding: 2D sinusoidal for the 28×28 patch grid
# =========================================================================

class PatchPositionalEncoding(nn.Module):
    """Sinusoidal 2-D positional encoding for a grid of patches."""

    def __init__(self, d_model: int = 256, max_h: int = 64, max_w: int = 64):
        super().__init__()
        self.d_model = d_model
        self.max_h = max_h
        self.max_w = max_w
        half = d_model // 2
        half2 = half // 2
        div = 10000.0 ** (torch.arange(0, half2, dtype=torch.float32) * 2 / half2)
        # Build once for rows and columns up to max
        row_enc = torch.zeros(max_h, half2 * 2)
        col_enc = torch.zeros(max_w, half2 * 2)
        for i in range(half2):
            row_enc[:, 2 * i] = torch.sin(torch.arange(max_h, dtype=torch.float32) / div[i])
            row_enc[:, 2 * i + 1] = torch.cos(torch.arange(max_h, dtype=torch.float32) / div[i])
            col_enc[:, 2 * i] = torch.sin(torch.arange(max_w, dtype=torch.float32) / div[i])
            col_enc[:, 2 * i + 1] = torch.cos(torch.arange(max_w, dtype=torch.float32) / div[i])
        self.register_buffer("row_enc", row_enc)  # (max_h, half)
        self.register_buffer("col_enc", col_enc)  # (max_w, half)

    def forward(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        """Return (h*w, d_model) positional encoding for an h×w grid."""
        enc = torch.zeros(h * w, self.d_model, device=device)
        half = self.d_model // 2
        for r in range(h):
            enc[r * w:(r + 1) * w, :half] = self.row_enc[r, :half]
        for c in range(w):
            enc[c::w, half:] = self.col_enc[c, :half]
        return enc


# =========================================================================
# Matching Block (self-attention + cross-attention + FFN)
# =========================================================================

class MatchingBlock(nn.Module):
    """One matching refinement block: self-attn → cross-attn → FFN."""

    def __init__(self, d_model: int = 256, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm_sa = nn.LayerNorm(d_model)
        self.norm_ca = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)

    def forward(self, desc_a: torch.Tensor, desc_b: torch.Tensor):
        """
        Args:
            desc_a: (B, N, d_model)  descriptors from image A
            desc_b: (B, N, d_model)  descriptors from image B
        Returns:
            desc_a, desc_b after self-attn, cross-attn, and FFN.
        """
        # Self-attention (within each image, independently)
        a2, _ = self.self_attn(desc_a, desc_a, desc_a)
        desc_a = self.norm_sa(desc_a + a2)
        b2, _ = self.self_attn(desc_b, desc_b, desc_b)
        desc_b = self.norm_sa(desc_b + b2)

        # Cross-attention: A ⟷ B
        ca_a, _ = self.cross_attn(desc_a, desc_b, desc_b)
        desc_a = self.norm_ca(desc_a + ca_a)
        ca_b, _ = self.cross_attn(desc_b, desc_a, desc_a)
        desc_b = self.norm_ca(desc_b + ca_b)

        # FFN
        desc_a = self.norm_ffn(desc_a + self.ffn(desc_a))
        desc_b = self.norm_ffn(desc_b + self.ffn(desc_b))

        return desc_a, desc_b


# =========================================================================
# Full Matcher
# =========================================================================

class LearnableMatcher(nn.Module):
    """
    Frozen DINOv3 backbone + learnable attention-based matching head.

    Pipeline:
      1. Frozen DINO extracts 1024-dim features.
      2. Shared linear projects 1024 → d_model.
      3. 2D sinusoidal positional encoding added.
      4. N matching blocks (self-attn + cross-attn + FFN).
      5. Final linear + L2 normalisation → refined descriptors.
      6. Dual-softmax matching produces an assignment matrix.
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 4,
        num_blocks: int = 4,
        dropout: float = 0.1,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.temperature = temperature

        self.input_proj = nn.Linear(1024, d_model)
        self.pos_enc = PatchPositionalEncoding(d_model)
        self.blocks = nn.ModuleList([
            MatchingBlock(d_model, nhead, dropout) for _ in range(num_blocks)
        ])
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, desc_a_raw: torch.Tensor, desc_b_raw: torch.Tensor):
        """
        Args:
            desc_a_raw: (B, N, 1024) frozen DINO features from image A
            desc_b_raw: (B, N, 1024) frozen DINO features from image B
        Returns:
            desc_a: (B, N, d_model) refined descriptors
            desc_b: (B, N, d_model)
            score_matrix: (B, N, N)  matching scores (desc_a @ desc_b.T / τ)
        """
        B, N, _ = desc_a_raw.shape
        h = w = int(N ** 0.5)  # 28

        # Project + add positional encoding
        pos = self.pos_enc(h, w, desc_a_raw.device).unsqueeze(0)  # (1, N, d_model)
        desc_a = self.input_proj(desc_a_raw) + pos
        desc_b = self.input_proj(desc_b_raw) + pos

        # Refinement blocks
        for block in self.blocks:
            desc_a, desc_b = block(desc_a, desc_b)

        # Final projection + normalisation
        desc_a = F.normalize(self.output_proj(desc_a), p=2, dim=-1)
        desc_b = F.normalize(self.output_proj(desc_b), p=2, dim=-1)

        # Score matrix
        score_matrix = torch.bmm(desc_a, desc_b.transpose(1, 2)) / self.temperature

        return desc_a, desc_b, score_matrix


# =========================================================================
# Dual-softmax matching (used in training and inference)
# =========================================================================

def dual_softmax_matching(
    score_matrix: torch.Tensor,  # (B, N, N)
    confidence_threshold: float = 0.01,
) -> list[list[tuple[int, int, float]]]:
    """
    Compute matches via dual-softmax + mutual argmax.

    Returns list (over batch) of lists of (idx_a, idx_b, confidence) matches.
    """
    B = score_matrix.shape[0]
    device = score_matrix.device

    all_matches = []
    with torch.no_grad():
        # Dual softmax
        p_a = F.softmax(score_matrix, dim=2)   # P(j | i)
        p_b = F.softmax(score_matrix, dim=1)   # P(i | j)
        p = p_a * p_b                           # mutual

        for b in range(B):
            pm = p[b]  # (N, N)
            # Argmax in each direction
            best_b = pm.argmax(dim=1)  # (N,) best B for each A
            best_a = pm.argmax(dim=0)  # (N,) best A for each B
            # Mutual check
            mutual = best_a[best_b] == torch.arange(pm.shape[0], device=device)
            # Confidence = p[i, j] for mutual matches
            batch_matches = []
            for i in range(pm.shape[0]):
                if mutual[i]:
                    j = best_b[i].item()
                    conf = pm[i, j].item()
                    if conf > confidence_threshold:
                        batch_matches.append((i, j, conf))
            all_matches.append(batch_matches)

    return all_matches


# =========================================================================
# Training
# =========================================================================

def train_matching_head(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load frozen DINO ----
    print("Loading frozen DINOv3 backbone...")
    backbone = DINOv3Backbone(
        patch_size=16, embed_dim=1024, depth=24,
        num_heads=16, mlp_ratio=4.0,
    )
    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    elif "teacher" in sd and isinstance(sd["teacher"], dict):
        sd = sd["teacher"]
        sd = {k.replace("backbone.", ""): v for k, v in sd.items()}
    backbone.load_state_dict(sd, strict=False)
    backbone = backbone.to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    print("Backbone frozen.")

    # ---- Matching head ----
    matcher = LearnableMatcher(
        d_model=args.d_model,
        nhead=args.nhead,
        num_blocks=args.num_blocks,
        dropout=args.dropout,
        temperature=args.temperature,
    ).to(device)
    n_params = sum(p.numel() for p in matcher.parameters()) / 1e6
    print(f"Matcher params: {n_params:.2f}M")

    # ---- Dataset ----
    dataset = MatchingPairDataset(
        pairs_path=args.train_pairs,
        data_root=args.data_root,
        depth_root=args.depth_root,
        img_size=args.img_size,
        training=False,  # no augmentation on images — backbone is frozen
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_pairs,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_matching_pairs,
        drop_last=True,
    )
    print(f"Training pairs: {len(dataset)}, batches: ~{len(dataloader)}")

    # ---- Optimiser ----
    optimizer = optim.AdamW(matcher.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=max(len(dataloader), 1),
        pct_start=0.1,
    )

    # ---- Training loop ----
    best_loss = float("inf")
    history = []

    for epoch in range(args.epochs):
        matcher.train()
        total_loss = 0.0
        total_pairs = 0
        num_batches = 0
        t0 = time.time()

        for batch in dataloader:
            if not batch:
                continue
            valid = [s for s in batch if len(s["idx_a"]) > 0]
            if not valid:
                continue

            # Extract frozen DINO features for all images in batch
            all_imgs = []
            for sample in valid:
                all_imgs.append(sample["img_a"])
                all_imgs.append(sample["img_b"])
            all_imgs = torch.stack(all_imgs).to(device)  # (2*B, 3, H, W)

            with torch.no_grad():
                out_all = backbone.forward_features(all_imgs)
                desc_all = F.normalize(out_all["x_norm_patchtokens"], p=2, dim=-1)

            # Split into A and B
            B = len(valid)
            desc_a = desc_all[0::2]  # (B, N, 1024)
            desc_b = desc_all[1::2]  # (B, N, 1024)

            # Forward through matcher
            _, _, score_matrix = matcher(desc_a, desc_b)  # (B, N, N)

            # Dual softmax loss
            loss = torch.tensor(0.0, device=device)
            count = 0
            for b_idx, sample in enumerate(valid):
                idx_a = sample["idx_a"].to(device)
                idx_b = sample["idx_b"].to(device)
                if len(idx_a) == 0:
                    continue
                s = score_matrix[b_idx]  # (N, N)
                p_a = F.log_softmax(s, dim=1)  # log P(j|i)
                p_b = F.log_softmax(s, dim=0)  # log P(i|j)
                # For each GT correspondence, compute log(P_mutual)
                log_p = p_a[idx_a, idx_b] + p_b[idx_a, idx_b]  # (K,)
                loss = loss - log_p.mean()
                count += 1

            if count > 0:
                loss = loss / count
            else:
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(matcher.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total_pairs += B
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        elapsed = time.time() - t0

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"loss={avg_loss:.4f} | "
              f"pairs={total_pairs} | "
              f"batches={num_batches} | "
              f"lr={scheduler.get_last_lr()[0]:.2e} | "
              f"time={elapsed:.1f}s")

        history.append({"epoch": epoch, "loss": avg_loss})
        (output_dir / "training_log.json").write_text(
            json.dumps(history, indent=2))

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(matcher.state_dict(), output_dir / "matcher_best.pth")
            print(f"  → saved best (loss={best_loss:.4f})")

        # Save latest
        torch.save(matcher.state_dict(), output_dir / "matcher_latest.pth")

    print(f"\nTraining done. Best loss={best_loss:.4f}")
    print(f"Model saved to {output_dir}/matcher_best.pth")
    return matcher


# =========================================================================
# Inference: extract features + refine + match + output CSV
# =========================================================================

def get_patch_coords_eval(
    h_patches: int, w_patches: int, patch_size: int,
    img_size: int, eval_w: int, eval_h: int,
) -> np.ndarray:
    """Patch centres mapped to evaluation coordinate space (e.g., 640×480)."""
    ys = np.arange(h_patches) * patch_size + patch_size // 2
    xs = np.arange(w_patches) * patch_size + patch_size // 2
    scale_x = eval_w / img_size
    scale_y = eval_h / img_size
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    return np.stack([grid_x.ravel() * scale_x, grid_y.ravel() * scale_y], axis=1)


def extract_with_matching_head(args):
    """Extract DINO features, refine via matching head, run dual-softmax
    matching, output standard-format CSV files."""
    import cv2
    import torchvision.transforms as T

    from .extract_and_match import pair_output_id

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_w, eval_h = args.eval_resize
    img_size = args.img_size
    patch_size = VIT_PATCH_SIZE
    h_patches = w_patches = img_size // patch_size

    # ---- Load frozen DINO ----
    print("Loading frozen DINOv3 backbone...")
    backbone = DINOv3Backbone(
        patch_size=16, embed_dim=1024, depth=24,
        num_heads=16, mlp_ratio=4.0,
    )
    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    elif "teacher" in sd and isinstance(sd["teacher"], dict):
        sd = sd["teacher"]
        sd = {k.replace("backbone.", ""): v for k, v in sd.items()}
    backbone.load_state_dict(sd, strict=False)
    backbone = backbone.to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    # ---- Load matcher ----
    print(f"Loading matcher from {args.matcher}...")
    matcher = LearnableMatcher(
        d_model=args.d_model,
        nhead=args.nhead,
        num_blocks=args.num_blocks,
    ).to(device)
    matcher.load_state_dict(torch.load(args.matcher, map_location=device))
    matcher.eval()

    # ---- Pre-compute eval coords ----
    coords_eval = get_patch_coords_eval(
        h_patches, w_patches, patch_size, img_size, eval_w, eval_h)

    # ---- Image transform ----
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # ---- Load pairs ----
    with open(args.pairs, "r") as f:
        pairs = [line.strip().split() for line in f if line.strip()]
    pairs = [p for p in pairs if len(p) == 38]
    print(f"Processing {len(pairs)} pairs...")
    print(f"Matching: dual-softmax (confidence_threshold={args.conf_threshold})")

    t0 = time.time()
    n_skipped = 0
    total_matches = 0

    for i, pair in enumerate(pairs):
        name0, name1 = pair[0], pair[1]
        pid = pair_output_id(name0, name1)

        path0 = str(Path(args.data_root) / name0)
        path1 = str(Path(args.data_root) / name1)

        img0 = cv2.imread(path0)
        img1 = cv2.imread(path1)
        if img0 is None or img1 is None:
            n_skipped += 1
            continue

        img0_rgb = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        t0_img = transform(img0_rgb).unsqueeze(0).to(device)
        t1_img = transform(img1_rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            out0 = backbone.forward_features(t0_img)
            out1 = backbone.forward_features(t1_img)
            desc0 = F.normalize(out0["x_norm_patchtokens"], p=2, dim=-1)
            desc1 = F.normalize(out1["x_norm_patchtokens"], p=2, dim=-1)

            # Add batch dim for matcher
            _, _, score_matrix = matcher(desc0, desc1)
            matches = dual_softmax_matching(
                score_matrix, confidence_threshold=args.conf_threshold)

        # Write CSV
        csv_path = output_dir / f"{pid}_matches.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(csv_path), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["left_idx", "right_idx", "x1", "y1", "x2", "y2", "score"])
            for ai, bi, conf in matches[0]:
                x1, y1 = coords_eval[ai]
                x2, y2 = coords_eval[bi]
                writer.writerow([int(ai), int(bi),
                                 f"{x1:.1f}", f"{y1:.1f}",
                                 f"{x2:.1f}", f"{y2:.1f}", f"{conf:.4f}"])
        total_matches += len(matches[0])

        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(pairs)}] {elapsed:.1f}s | {pid} | "
                  f"matches={len(matches[0])}")

    elapsed = time.time() - t0
    avg_m = total_matches / max(len(pairs) - n_skipped, 1)
    print(f"\nDone! {len(pairs)} pairs in {elapsed:.1f}s "
          f"(skipped {n_skipped})")
    print(f"Avg matches per pair: {avg_m:.0f}")
    print(f"Results saved to {output_dir}")


# =========================================================================
# CLI
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Frozen DINO + Learnable Attention Matching Head — Route B")
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- train ----
    p_train = sub.add_parser("train", help="Train matching head")
    p_train.add_argument("--checkpoint", type=str, required=True)
    p_train.add_argument("--train_pairs", type=str, required=True)
    p_train.add_argument("--data_root", type=str, required=True)
    p_train.add_argument("--depth_root", type=str, default="")
    p_train.add_argument("--output_dir", type=str, default="matching_head_output")
    p_train.add_argument("--batch_pairs", type=int, default=4)
    p_train.add_argument("--epochs", type=int, default=20)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--wd", type=float, default=1e-3)
    p_train.add_argument("--d_model", type=int, default=256)
    p_train.add_argument("--nhead", type=int, default=4)
    p_train.add_argument("--num_blocks", type=int, default=4)
    p_train.add_argument("--dropout", type=float, default=0.1)
    p_train.add_argument("--temperature", type=float, default=0.1)
    p_train.add_argument("--num_workers", type=int, default=4)
    p_train.add_argument("--img_size", type=int, default=448)
    p_train.add_argument("--gpu", type=int, default=0)

    # ---- extract ----
    p_ext = sub.add_parser("extract", help="Extract + refine + match")
    p_ext.add_argument("--checkpoint", type=str, required=True)
    p_ext.add_argument("--matcher", type=str, required=True,
                       help="Trained matcher weights")
    p_ext.add_argument("--pairs", type=str, required=True)
    p_ext.add_argument("--data_root", type=str, required=True)
    p_ext.add_argument("--output_dir", type=str, required=True)
    p_ext.add_argument("--img_size", type=int, default=448)
    p_ext.add_argument("--eval_resize", type=int, nargs=2, default=[640, 480])
    p_ext.add_argument("--conf_threshold", type=float, default=0.01)
    p_ext.add_argument("--d_model", type=int, default=256)
    p_ext.add_argument("--nhead", type=int, default=4)
    p_ext.add_argument("--num_blocks", type=int, default=4)
    p_ext.add_argument("--gpu", type=int, default=0)

    args = parser.parse_args()

    if args.command == "train":
        train_matching_head(args)
    elif args.command == "extract":
        extract_with_matching_head(args)


if __name__ == "__main__":
    main()
