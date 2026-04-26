"""
Training script for DINOv2 fine-tuning on local feature matching.

Usage:
    python -m finetune.train \
        --checkpoint dinov3_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
        --train_pairs datasets/scannet_with_gt.txt \
        --data_root datasets/test \
        --output_dir finetune_output \
        --epochs 15 --batch_size 2
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from .config import get_train_args, LOG_EVERY, SAVE_EVERY, WARMUP_EPOCHS
from .model import DINOv2Matcher
from .dataset import MatchingPairDataset, collate_matching_pairs
from .loss import MatchingLoss


def build_optimizer(model: DINOv2Matcher, lr_backbone: float, lr_proj: float, wd: float):
    """Separate learning rates for backbone and projection head."""
    backbone_params = []
    proj_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "proj_head" in name:
            proj_params.append(param)
        else:
            backbone_params.append(param)

    param_groups = [
        {"params": backbone_params, "lr": lr_backbone, "weight_decay": wd},
        {"params": proj_params, "lr": lr_proj, "weight_decay": wd},
    ]

    return optim.AdamW(param_groups)


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor=0.001):
    """Linear warm-up scheduler."""
    def f(x):
        if x >= warmup_iters:
            return 1.0
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    return optim.lr_scheduler.LambdaLR(optimizer, f)


def train_one_epoch(
    model: DINOv2Matcher,
    dataloader: DataLoader,
    criterion: MatchingLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    scheduler=None,
) -> dict:
    """Train for one epoch, return average losses."""
    model.train()
    total_loss = 0.0
    total_contrastive = 0.0
    total_diversity = 0.0
    num_valid = 0
    step = 0

    for batch in dataloader:
        # batch is a list of valid samples (from custom collate)
        if not batch:
            continue

        for sample in batch:
            img_a = sample["img_a"].unsqueeze(0).to(device)
            img_b = sample["img_b"].unsqueeze(0).to(device)
            idx_a = sample["idx_a"].to(device)
            idx_b = sample["idx_b"].to(device)

            if len(idx_a) == 0:
                continue

            # Forward: extract descriptors
            desc_a_all = model(img_a)  # (1, N, D)
            desc_b_all = model(img_b)  # (1, N, D)

            # Gather correspondences
            desc_a = desc_a_all[0, idx_a]  # (M, D)
            desc_b = desc_b_all[0, idx_b]  # (M, D)

            # Compute loss
            losses = criterion(desc_a, desc_b)
            loss = losses["total"]

            # Backward
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
            num_valid += 1
            step += 1

            if step % LOG_EVERY == 0:
                print(
                    f"  [epoch {epoch}] step {step} | "
                    f"loss={loss.item():.4f} | "
                    f"contrastive={losses['contrastive'].item():.4f} | "
                    f"corr={len(idx_a)}"
                )

    if num_valid == 0:
        return {"loss": 0, "contrastive": 0, "diversity": 0, "num_valid": 0}

    return {
        "loss": total_loss / num_valid,
        "contrastive": total_contrastive / num_valid,
        "diversity": total_diversity / num_valid,
        "num_valid": num_valid,
    }


def save_checkpoint(model, optimizer, epoch, output_dir, metrics=None):
    """Save a training checkpoint."""
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics or {},
    }
    path = Path(output_dir) / f"checkpoint_epoch{epoch:03d}.pth"
    torch.save(ckpt, path)
    print(f"  Saved checkpoint: {path}")

    # Also save as 'best' / 'latest'
    latest = Path(output_dir) / "checkpoint_latest.pth"
    torch.save(ckpt, latest)

    return path


def main():
    args = get_train_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Save config ──────────────────────────────────────────────────
    config_path = output_dir / "config.json"
    config_path.write_text(json.dumps(vars(args), indent=2, default=str))

    # ── Model ────────────────────────────────────────────────────────
    print("Building model...")
    model = DINOv2Matcher(
        checkpoint_path=args.checkpoint,
        freeze_blocks=args.freeze_blocks,
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
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,      # Windows compatibility
        collate_fn=collate_matching_pairs,
        drop_last=True,
    )

    # ── Loss ─────────────────────────────────────────────────────────
    criterion = MatchingLoss(
        temperature=args.temperature,
        use_hard_negatives=True,
    )

    # ── Optimiser ────────────────────────────────────────────────────
    optimizer = build_optimizer(model, args.lr_backbone, args.lr_proj, 1e-4)

    # ── LR Scheduler ─────────────────────────────────────────────────
    warmup_iters = WARMUP_EPOCHS * len(dataloader)
    scheduler = warmup_lr_scheduler(optimizer, warmup_iters)
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - WARMUP_EPOCHS, eta_min=1e-7
    )

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
    print(f"  Starting training: {args.epochs} epochs")
    print(f"  Pairs: {len(dataset)} | Batch size: {args.batch_size}")
    print(f"  LR backbone: {args.lr_backbone} | LR proj: {args.lr_proj}")
    print(f"  Frozen blocks: {args.freeze_blocks}/{24}")
    print(f"{'='*60}\n")

    log_path = output_dir / "training_log.json"
    training_log = []

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        metrics = train_one_epoch(
            model, dataloader, criterion, optimizer, device, epoch,
            scheduler=scheduler if epoch < WARMUP_EPOCHS else None,
        )
        elapsed = time.time() - t0

        # Step cosine scheduler after warmup
        if epoch >= WARMUP_EPOCHS:
            cosine_scheduler.step()

        lr_bb = optimizer.param_groups[0]["lr"]
        lr_ph = optimizer.param_groups[1]["lr"]

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"loss={metrics['loss']:.4f} | "
            f"contrastive={metrics['contrastive']:.4f} | "
            f"valid_pairs={metrics['num_valid']} | "
            f"lr_bb={lr_bb:.2e} | lr_ph={lr_ph:.2e} | "
            f"time={elapsed:.1f}s"
        )

        # Log
        log_entry = {
            "epoch": epoch,
            **metrics,
            "lr_backbone": lr_bb,
            "lr_proj_head": lr_ph,
            "elapsed_s": elapsed,
        }
        training_log.append(log_entry)
        log_path.write_text(json.dumps(training_log, indent=2))

        # Save checkpoint
        if (epoch + 1) % SAVE_EVERY == 0 or epoch == args.epochs - 1:
            save_checkpoint(model, optimizer, epoch, output_dir, metrics)

    print(f"\nTraining complete. Checkpoints saved to {output_dir}")


if __name__ == "__main__":
    main()
