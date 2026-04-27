"""
LoRA fine-tuning for DINOv3 on local feature matching.

Key differences from the original Projection Head approach:
  1. No random Projection Head — uses DINO's native 1024-dim features
  2. LoRA adapters injected into ALL 24 attention blocks (not just last 2)
  3. Zero-init guarantees the model starts from the working zero-shot baseline
  4. Only ~200K trainable params vs 25M in the original approach

Usage:
    python -m finetune.train_lora \
        --checkpoint dinov3_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
        --train_pairs finetune/navi_train_pairs.txt \
        --data_root full_dataset/navi_v1.5 \
        --depth_root full_dataset/navi_v1.5 \
        --output_dir finetune_output_lora_navi \
        --epochs 15 --batch_size 1 --img_size 448
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from .config import LOG_EVERY, SAVE_EVERY, WARMUP_EPOCHS
from .model import DINOv3Backbone
from .lora import inject_lora, get_lora_parameters
from .dataset import MatchingPairDataset, collate_matching_pairs
from .loss import MatchingLoss

import argparse


# =====================================================================
# LoRA DINOv3 Matcher — no Projection Head
# =====================================================================

class LoRADINOv3Matcher(nn.Module):
    """
    DINOv3 backbone with LoRA adapters. No Projection Head.

    At initialisation, the model output is IDENTICAL to zero-shot DINOv3
    because LoRA's B matrix is zero. Training gently adjusts the attention
    patterns to improve geometry-aware matching.
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        lora_rank: int = 4,
        lora_alpha: float = 1.0,
        lora_targets: tuple[str, ...] = ("qkv",),
    ):
        super().__init__()

        # 1) Build backbone
        self.backbone = DINOv3Backbone(
            patch_size=16,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            mlp_ratio=4.0,
        )

        # 2) Load pre-trained weights
        if checkpoint_path is not None:
            self._load_weights(checkpoint_path)

        # 3) Freeze EVERYTHING first
        for p in self.parameters():
            p.requires_grad = False

        # 4) Inject LoRA adapters (these are the ONLY trainable params)
        n_lora = inject_lora(self.backbone, rank=lora_rank, alpha=lora_alpha,
                             target_modules=lora_targets)
        print(f"[LoRA] Injected adapters into {len(lora_targets)} modules × 24 blocks")
        print(f"[LoRA] rank={lora_rank}, alpha={lora_alpha}")

        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        print(f"[LoRA] Trainable: {n_train/1e6:.3f}M / {n_total/1e6:.1f}M params "
              f"({100*n_train/n_total:.2f}%)")

        self.patch_size = 16
        self.embed_dim = 1024

    def _load_weights(self, path: str):
        state_dict = torch.load(path, map_location="cpu", weights_only=False)
        if "model" in state_dict and isinstance(state_dict["model"], dict):
            state_dict = state_dict["model"]
        elif "teacher" in state_dict and isinstance(state_dict["teacher"], dict):
            state_dict = state_dict["teacher"]
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

        missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
        if unexpected:
            print(f"[model] Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
        print(f"[model] Loaded pre-trained weights from {path}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns (B, h*w, 1024) L2-normalised descriptors.
        No Projection Head — stays in DINO's original 1024-dim space.
        """
        out = self.backbone.forward_features(x)
        patch_tokens = out["x_norm_patchtokens"]
        return F.normalize(patch_tokens, p=2, dim=-1)

    @torch.no_grad()
    def get_patch_coords(self, H: int, W: int) -> torch.Tensor:
        ps = self.patch_size
        h, w = H // ps, W // ps
        ys = torch.arange(h) * ps + ps // 2
        xs = torch.arange(w) * ps + ps // 2
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1).float()


# =====================================================================
# Training
# =====================================================================

def get_lora_train_args():
    parser = argparse.ArgumentParser(
        description="LoRA fine-tune DINOv3 for local feature matching",
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--train_pairs", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--depth_root", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="finetune_output_lora")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for LoRA params (can be high since params are few)")
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--img_size", type=int, default=448)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, scheduler=None):
    model.train()
    total_loss = 0.0
    total_contrastive = 0.0
    total_diversity = 0.0
    num_valid = 0
    step = 0

    for batch in dataloader:
        if not batch:
            continue
            
        # Filter valid samples and stack them
        valid_samples = [s for s in batch if len(s["idx_a"]) > 0]
        if not valid_samples:
            continue

        img_a = torch.stack([s["img_a"] for s in valid_samples]).to(device)
        img_b = torch.stack([s["img_b"] for s in valid_samples]).to(device)

        # Forward pass for the whole batch (inter-image)
        desc_a_all = model(img_a)  # (B, H*W, 1024)
        desc_b_all = model(img_b)

        desc_a_list = []
        desc_b_list = []
        batch_idx_a_list = []
        batch_idx_b_list = []

        for b, sample in enumerate(valid_samples):
            idx_a = sample["idx_a"].to(device)
            idx_b = sample["idx_b"].to(device)

            desc_a_list.append(desc_a_all[b, idx_a])
            desc_b_list.append(desc_b_all[b, idx_b])
            batch_idx_a_list.append(torch.full((len(idx_a),), b, dtype=torch.long, device=device))
            batch_idx_b_list.append(torch.full((len(idx_b),), b, dtype=torch.long, device=device))

        desc_a_flat = torch.cat(desc_a_list, dim=0)
        desc_b_flat = torch.cat(desc_b_list, dim=0)
        idx_a_flat = torch.cat([s["idx_a"].to(device) for s in valid_samples], dim=0)
        idx_b_flat = torch.cat([s["idx_b"].to(device) for s in valid_samples], dim=0)
        batch_idx_a_flat = torch.cat(batch_idx_a_list, dim=0)
        batch_idx_b_flat = torch.cat(batch_idx_b_list, dim=0)

        losses = criterion(
            desc_a_flat, desc_b_flat, 
            idx_a_flat, idx_b_flat, 
            batch_idx_a_flat, batch_idx_b_flat,
            model.patch_size, 448
        )
        
        loss = losses["total"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        total_contrastive += losses["contrastive"].item()
        if "diversity" in losses:
            total_diversity += losses["diversity"].item()
        num_valid += len(valid_samples)
        step += 1

        if step % LOG_EVERY == 0:
            print(
                f"  [epoch {epoch}] step {step} | "
                f"loss={loss.item():.4f} | "
                f"contrastive={losses['contrastive'].item():.4f} | "
                f"batch_pairs={len(valid_samples)} | "
                f"total_pts={len(idx_a_flat)}",
                flush=True
            )

    if num_valid == 0:
        return {"loss": 0, "contrastive": 0, "diversity": 0, "num_valid": 0}

    return {
        "loss": total_loss / num_valid,
        "contrastive": total_contrastive / num_valid,
        "diversity": total_diversity / num_valid,
        "num_valid": num_valid,
    }


def main():
    args = get_lora_train_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "config.json"
    config_path.write_text(json.dumps(vars(args), indent=2, default=str))

    # ── Model ────────────────────────────────────────────────────────
    print("Building LoRA model...")
    model = LoRADINOv3Matcher(
        checkpoint_path=args.checkpoint,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_targets=("qkv",),  # Only QKV for memory efficiency
    )
    model = model.to(device)

    # ── Dataset ──────────────────────────────────────────────────────
    print("Loading dataset...")
    dataset = MatchingPairDataset(
        pairs_path=args.train_pairs,
        data_root=args.data_root,
        depth_root=args.depth_root,
        img_size=args.img_size,
        training=True,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, collate_fn=collate_matching_pairs, drop_last=True,
    )

    # ── Loss ─────────────────────────────────────────────────────────
    criterion = MatchingLoss(
        temperature=args.temperature,
        use_hard_negatives=True,
    )

    # ── Optimiser (only LoRA params) ─────────────────────────────────
    lora_params = get_lora_parameters(model)
    optimizer = optim.AdamW(lora_params, lr=args.lr, weight_decay=1e-4)

    # Per-step cosine annealing with linear warm-up
    steps_per_epoch = len(dataset)  # batch_size=1, one step per sample
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = min(2 * steps_per_epoch, total_steps // 5)

    import math
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.01, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Resume ───────────────────────────────────────────────────────
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    # ── Training loop ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  LoRA Training: {args.epochs} epochs")
    print(f"  Pairs: {len(dataset)} | Batch size: {args.batch_size}")
    print(f"  LR: {args.lr} | Rank: {args.lora_rank} | Alpha: {args.lora_alpha}")
    print(f"  Feature dim: 1024 (native DINO, no projection head)")
    print(f"  Warmup steps: {warmup_steps} | Total steps: {total_steps}")
    print(f"{'='*60}\n")

    log_path = output_dir / "training_log.json"
    training_log = []

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        metrics = train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, scheduler)
        elapsed = time.time() - t0

        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"loss={metrics['loss']:.4f} | "
            f"contrastive={metrics['contrastive']:.4f} | "
            f"valid_pairs={metrics['num_valid']} | "
            f"lr={lr_now:.2e} | time={elapsed:.1f}s"
        )

        log_entry = {
            "epoch": epoch,
            **metrics,
            "lr": lr_now,
            "elapsed_s": elapsed,
        }
        training_log.append(log_entry)
        log_path.write_text(json.dumps(training_log, indent=2))

        # Save checkpoint
        if (epoch + 1) % SAVE_EVERY == 0 or epoch == args.epochs - 1:
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": metrics,
                "lora_config": {
                    "rank": args.lora_rank,
                    "alpha": args.lora_alpha,
                    "targets": ["qkv"],
                },
            }
            path = output_dir / f"checkpoint_epoch{epoch:03d}.pth"
            torch.save(ckpt, path)
            print(f"  Saved checkpoint: {path}")

            latest = output_dir / "checkpoint_latest.pth"
            torch.save(ckpt, latest)

    print(f"\nTraining complete. Checkpoints saved to {output_dir}")


if __name__ == "__main__":
    main()
