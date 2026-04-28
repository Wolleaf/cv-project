"""
Extract features from LoRA-finetuned DINOv3 and perform MNN matching.

Same output format as the original extract_and_match.py but uses the
LoRA model (no Projection Head, 1024-dim features).

Usage:
    python -m finetune.extract_lora \
        --checkpoint finetune_output_lora_navi/checkpoint_latest.pth \
        --pretrained dinov3_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
        --pairs datasets/navi_with_gt.txt \
        --data_root datasets/test/navi_resized \
        --output_dir mnn_matching_lora_navi/navi \
        --img_size 448 --eval_resize 640 480
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T

from .extract_and_match import (
    mutual_nearest_neighbors,
    get_patch_coordinates_eval,
    pair_output_id,
)
from .train_lora import LoRADINOv3Matcher


def load_and_preprocess(image_path: str, img_size: int = 448):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(img_rgb)


def get_args():
    parser = argparse.ArgumentParser(
        description="Extract features from LoRA-finetuned DINOv3",
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="LoRA fine-tuned checkpoint")
    parser.add_argument("--pretrained", type=str, required=True,
                        help="Original DINOv3 pretrained weights (needed to rebuild backbone)")
    parser.add_argument("--pairs", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=448)
    parser.add_argument("--eval_resize", type=int, nargs=2, default=[640, 480])
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=float, default=1.0)
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def main():
    args = get_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_w, eval_h = args.eval_resize
    print(f"Model input size: {args.img_size}x{args.img_size}")
    print(f"Eval coordinate system: {eval_w}x{eval_h}")

    # ── Load model ───────────────────────────────────────────────────
    print("Building LoRA model...")
    model = LoRADINOv3Matcher(
        checkpoint_path=args.pretrained,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_targets=("qkv",),
    )

    # Load LoRA-finetuned weights
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded LoRA checkpoint (epoch {ckpt.get('epoch', '?')})")
        if "lora_config" in ckpt:
            print(f"  LoRA config: {ckpt['lora_config']}")
    else:
        model.load_state_dict(ckpt, strict=False)

    model = model.to(device)
    model.eval()

    patch_size = model.patch_size
    h_patches = args.img_size // patch_size
    w_patches = args.img_size // patch_size

    coords_eval = get_patch_coordinates_eval(
        h_patches, w_patches, patch_size, args.img_size, eval_w, eval_h
    )

    # ── Load pairs ───────────────────────────────────────────────────
    with open(args.pairs, "r") as f:
        pairs = [line.strip().split() for line in f if line.strip()]
    pairs = [p for p in pairs if len(p) == 38]
    print(f"Processing {len(pairs)} pairs...")

    print(f"Patch coord range: x=[{coords_eval[:, 0].min():.1f}, {coords_eval[:, 0].max():.1f}], "
          f"y=[{coords_eval[:, 1].min():.1f}, {coords_eval[:, 1].max():.1f}]")

    t0 = time.time()
    n_skipped = 0
    for i, pair in enumerate(pairs):
        name0, name1 = pair[0], pair[1]
        pid = pair_output_id(name0, name1)

        path0 = str(Path(args.data_root) / name0)
        path1 = str(Path(args.data_root) / name1)

        try:
            img_a = load_and_preprocess(path0, args.img_size)
            img_b = load_and_preprocess(path1, args.img_size)
        except FileNotFoundError:
            n_skipped += 1
            continue

        with torch.no_grad():
            desc_a = model(img_a.unsqueeze(0).to(device))[0]
            desc_b = model(img_b.unsqueeze(0).to(device))[0]

        idx_a, idx_b, scores = mutual_nearest_neighbors(desc_a, desc_b)

        csv_path = output_dir.resolve() / f"{pid}_matches.csv"
        safe_csv_path = "\\\\?\\" + str(csv_path) if os.name == "nt" else str(csv_path)
        with open(safe_csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["left_idx", "right_idx", "x1", "y1", "x2", "y2", "score"])
            for a, b, s in zip(idx_a, idx_b, scores):
                x1, y1 = coords_eval[a]
                x2, y2 = coords_eval[b]
                writer.writerow([int(a), int(b), f"{x1:.1f}", f"{y1:.1f}",
                                 f"{x2:.1f}", f"{y2:.1f}", f"{s}"])

        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(pairs)}] {elapsed:.1f}s | {pid} | {len(idx_a)} matches")

    elapsed = time.time() - t0
    print(f"\nDone! {len(pairs)} pairs in {elapsed:.1f}s (skipped {n_skipped})")
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
