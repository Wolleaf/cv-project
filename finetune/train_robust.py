"""
Robust fine-tuning for DINOv3 on local feature matching.

Replaces InfoNCE with StableMatchingLoss, adds feature-preservation
regularisation, distinctiveness-weighted sampling, and gradient accumulation
for a smoother training signal.

Usage:
    python -m finetune.train_robust \
        --checkpoint dinov3_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
        --train_pairs finetune/navi_train_pairs.txt \
        --data_root full_dataset/navi_v1.5 \
        --depth_root full_dataset/navi_v1.5 \
        --output_dir finetune_output_robust_navi \
        --epochs 20 --batch_size 8 --img_size 448 --lora_rank 4 --lr 5e-4
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from .config import SAVE_EVERY
from .model import DINOv3Backbone
from .lora import inject_lora, get_lora_parameters
from .dataset import MatchingPairDataset, collate_matching_pairs
from .loss_robust import StableMatchingLoss

import argparse


# =====================================================================
# Model (same LoRA architecture, reused from train_lora)
# =====================================================================

class LoRADINOv3Matcher(nn.Module):
    """DINOv3 + LoRA, no projection head.  Output: (B, N, 1024) L2-norm."""

    def __init__(
        self,
        checkpoint_path: str | None = None,
        lora_rank: int = 4,
        lora_alpha: float = 1.0,
        lora_targets: tuple[str, ...] = ("qkv",),
    ):
        super().__init__()
        self.backbone = DINOv3Backbone(
            patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4.0,
        )
        if checkpoint_path is not None:
            self._load_weights(checkpoint_path)

        for p in self.parameters():
            p.requires_grad = False

        n_lora = inject_lora(self.backbone, rank=lora_rank, alpha=lora_alpha,
                             target_modules=lora_targets)
        print(f"[LoRA] Injected {n_lora} params across {len(lora_targets)} modules × 24 blocks")
        print(f"[LoRA] rank={lora_rank}, alpha={lora_alpha}")

        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        print(f"[LoRA] Trainable: {n_train/1e6:.3f}M / {n_total/1e6:.1f}M "
              f"({100*n_train/n_total:.2f}%)")

        self.patch_size = 16
        self.embed_dim = 1024

    def _load_weights(self, path: str):
        sd = torch.load(path, map_location="cpu", weights_only=False)
        if "model" in sd and isinstance(sd["model"], dict):
            sd = sd["model"]
        elif "teacher" in sd and isinstance(sd["teacher"], dict):
            sd = sd["teacher"]
            sd = {k.replace("backbone.", ""): v for k, v in sd.items()}
        missing, unexpected = self.backbone.load_state_dict(sd, strict=False)
        if unexpected:
            print(f"[model] Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
        print(f"[model] Loaded weights from {path}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

def get_args():
    p = argparse.ArgumentParser(description="Robust LoRA fine-tuning for matching")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--train_pairs", type=str, required=True)
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--depth_root", type=str, default="")
    p.add_argument("--output_dir", type=str, default="finetune_output_robust")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=5e-4,
                   help="Learning rate (lower than before — stable margin loss allows it)")
    p.add_argument("--lora_rank", type=int, default=4)
    p.add_argument("--lora_alpha", type=float, default=1.0)
    p.add_argument("--img_size", type=int, default=448)
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--gpu", type=int, default=0)
    # Loss hyperparameters
    p.add_argument("--pos_weight", type=float, default=2.0)
    p.add_argument("--neg_weight", type=float, default=1.0)
    p.add_argument("--preserve_weight", type=float, default=0.5)
    p.add_argument("--diversity_weight", type=float, default=0.1)
    p.add_argument("--margin", type=float, default=0.3)
    p.add_argument("--safe_radius", type=float, default=5.0)
    # Optimiser
    p.add_argument("--weight_decay", type=float, default=5e-3,
                   help="Stronger weight decay keeps LoRA params small ≈ feature preservation")
    p.add_argument("--grad_accum", type=int, default=1,
                   help="Gradient accumulation steps (for larger effective batch)")
    return p.parse_args()


def train_one_epoch(
    model, dataloader, criterion, optimizer, device, epoch,
    scheduler=None, grad_accum: int = 1, preserve_weight: float = 0.5,
):
    model.train()
    total_loss = 0.0
    total_pos = 0.0
    total_neg = 0.0
    total_preserve = 0.0
    total_div = 0.0
    num_valid = 0
    step = 0
    accum_step = 0  # count within current accumulation window

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(dataloader):
        if not batch:
            continue

        valid_samples = [s for s in batch if len(s["idx_a"]) > 0]
        if not valid_samples:
            continue

        img_a = torch.stack([s["img_a"] for s in valid_samples]).to(device)
        img_b = torch.stack([s["img_b"] for s in valid_samples]).to(device)

        desc_a_all = model(img_a)
        desc_b_all = model(img_b)

        desc_a_list, desc_b_list = [], []
        batch_idx_a_list, batch_idx_b_list = [], []

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

        # Stable matching loss (NO low-temperature softmax)
        losses = criterion(
            desc_a_flat, desc_b_flat,
            idx_a_flat, idx_b_flat,
            batch_idx_a_flat, batch_idx_b_flat,
            model.patch_size, 448,
        )

        # Explicit L2 penalty on LoRA parameters (feature preservation)
        lora_params = get_lora_parameters(model)
        lora_l2 = sum(p.pow(2).sum() for p in lora_params) / max(len(lora_params), 1)

        loss = losses["total"] + preserve_weight * lora_l2
        loss = loss / grad_accum
        loss.backward()
        accum_step += 1

        if accum_step == grad_accum:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()
            accum_step = 0

        total_loss += losses["total"].item()
        total_pos += losses["positive"].item()
        total_neg += losses["negative"].item()
        total_preserve += lora_l2.item()
        total_div += losses["diversity"].item()
        num_valid += len(valid_samples)
        step += 1

        if step % 10 == 0:
            print(
                f"  [epoch {epoch}] step {step} | "
                f"total={losses['total'].item():.4f} | "
                f"pos={losses['positive'].item():.4f} | "
                f"neg={losses['negative'].item():.4f} | "
                f"div={losses['diversity'].item():.4f} | "
                f"lora_L2={lora_l2.item():.6f} | "
                f"pairs={len(valid_samples)} | pts={len(idx_a_flat)}",
                flush=True,
            )

    # Apply any leftover accumulated gradients
    if accum_step > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()

    if num_valid == 0:
        return {"loss": 0, "positive": 0, "negative": 0, "preserve": 0,
                "diversity": 0, "num_valid": 0}

    return {
        "loss": total_loss / num_valid,
        "positive": total_pos / num_valid,
        "negative": total_neg / num_valid,
        "preserve": total_preserve / num_valid,
        "diversity": total_div / num_valid,
        "num_valid": num_valid,
    }


def main():
    args = get_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))

    # ---- Model ----
    print("Building LoRA model...")
    model = LoRADINOv3Matcher(
        checkpoint_path=args.checkpoint,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_targets=("qkv",),
    )
    model = model.to(device)

    # ---- Dataset ----
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

    # ---- Loss ----
    criterion = StableMatchingLoss(
        pos_weight=args.pos_weight,
        neg_weight=args.neg_weight,
        preserve_weight=0.0,   # handled via explicit L2 in train loop
        diversity_weight=args.diversity_weight,
        margin=args.margin,
        safe_radius=args.safe_radius,
    )

    # ---- Optimiser ----
    lora_params = get_lora_parameters(model)
    optimizer = optim.AdamW(lora_params, lr=args.lr, weight_decay=args.weight_decay)

    # Cosine annealing with linear warmup
    steps_per_epoch = len(dataloader) // args.grad_accum
    total_steps = args.epochs * max(steps_per_epoch, 1)
    warmup_steps = min(int(0.1 * total_steps), 2 * max(steps_per_epoch, 1))

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.01, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---- Resume ----
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    # ---- Training ----
    print(f"\n{'='*60}")
    print(f"  Robust Matching Fine-tuning: {args.epochs} epochs")
    print(f"  Pairs: {len(dataset)} | Batch: {args.batch_size} | GradAccum: {args.grad_accum}")
    print(f"  LR: {args.lr} | Margin: {args.margin} | SafeRadius: {args.safe_radius}")
    print(f"  PosW: {args.pos_weight} | NegW: {args.neg_weight} | DivW: {args.diversity_weight}")
    print(f"  WD: {args.weight_decay} | LoRA rank: {args.lora_rank}")
    print(f"  Warmup: {warmup_steps} steps | Total: {total_steps} steps")
    print(f"{'='*60}\n")

    log_path = output_dir / "training_log.json"
    training_log = []

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        metrics = train_one_epoch(
            model, dataloader, criterion, optimizer, device, epoch,
            scheduler=scheduler, grad_accum=args.grad_accum,
            preserve_weight=args.preserve_weight,
        )
        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"loss={metrics['loss']:.4f} | "
            f"pos={metrics['positive']:.4f} | "
            f"neg={metrics['negative']:.4f} | "
            f"preserve={metrics['preserve']:.6f} | "
            f"div={metrics['diversity']:.4f} | "
            f"pairs={metrics['num_valid']} | "
            f"lr={lr_now:.2e} | time={elapsed:.1f}s"
        )

        log_entry = {"epoch": epoch, **metrics, "lr": lr_now, "elapsed_s": elapsed}
        training_log.append(log_entry)
        log_path.write_text(json.dumps(training_log, indent=2))

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
            p = output_dir / f"checkpoint_epoch{epoch:03d}.pth"
            torch.save(ckpt, p)
            print(f"  Saved: {p}")
            latest = output_dir / "checkpoint_latest.pth"
            torch.save(ckpt, latest)

    print(f"\nTraining complete. Checkpoints saved to {output_dir}")


if __name__ == "__main__":
    main()
