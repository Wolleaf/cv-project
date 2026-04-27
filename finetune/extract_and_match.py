"""
Extract dense features from fine-tuned DINOv2 and perform MNN matching.

Produces CSV files in exactly the same format used by lmz's zero-shot
pipeline, so they can be evaluated directly with evaluate_csv_essential.py.

CRITICAL: The output keypoint coordinates must be in the **evaluation resize
coordinate system** (default 640x480), NOT the model's internal resolution
(448x448) or the original image dimensions. The evaluate script reads images
at the --resize resolution and scales intrinsics accordingly.

CSV format:
    left_idx, right_idx, x1, y1, x2, y2, score

Usage:
    python -m finetune.extract_and_match \
        --checkpoint finetune_output/checkpoint_latest.pth \
        --pairs datasets/navi_with_gt.txt \
        --data_root datasets/test/navi_resized \
        --output_dir mnn_matching_finetuned/navi \
        --img_size 448 \
        --eval_resize 640 480
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T

from .config import get_extract_args, VIT_PATCH_SIZE, PROJ_DIM, IMG_SIZE
from .model import DINOv2Matcher


# ═════════════════════════════════════════════════════════════════════
# MNN matching
# ═════════════════════════════════════════════════════════════════════

def mutual_nearest_neighbors(
    desc_a: torch.Tensor,
    desc_b: torch.Tensor,
    threshold: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Mutual Nearest Neighbor matching between two sets of L2-normalised descriptors.

    Args:
        desc_a: (N, D) descriptors from image A
        desc_b: (M, D) descriptors from image B
        threshold: minimum cosine similarity to accept a match

    Returns:
        idx_a: matched indices in A
        idx_b: matched indices in B
        scores: cosine similarity scores for each match
    """
    # Cosine similarity
    sim = torch.mm(desc_a, desc_b.t())  # (N, M)

    # Nearest in B for each A
    nn_b = sim.argmax(dim=1)  # (N,)
    # Nearest in A for each B
    nn_a = sim.argmax(dim=0)  # (M,)

    # Mutual check
    idx_a = torch.arange(desc_a.shape[0], device=desc_a.device)
    mutual_mask = nn_a[nn_b] == idx_a

    # Score threshold
    scores = sim[idx_a, nn_b]
    score_mask = scores >= threshold

    valid = mutual_mask & score_mask
    idx_a_np = idx_a[valid].cpu().numpy()
    idx_b_np = nn_b[valid].cpu().numpy()
    scores_np = scores[valid].cpu().numpy()

    return idx_a_np, idx_b_np, scores_np


# ═════════════════════════════════════════════════════════════════════
# Feature extraction
# ═════════════════════════════════════════════════════════════════════

def load_and_preprocess(
    image_path: str,
    img_size: int = IMG_SIZE,
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Load an image, resize to img_size, and normalise."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    orig_h, orig_w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(img_rgb)
    return tensor, (orig_h, orig_w)


def get_patch_coordinates_eval(
    h_patches: int,
    w_patches: int,
    patch_size: int,
    img_size: int,
    eval_w: int,
    eval_h: int,
) -> np.ndarray:
    """
    Compute pixel coordinates of patch centres, mapped to the **evaluation
    resize coordinate system** (e.g., 640x480).

    The model processes images at img_size x img_size. The evaluate script
    processes images at eval_w x eval_h. We map patch centres from model
    space to evaluation space.

    Returns:
        (h_patches * w_patches, 2) array of (x, y) coordinates in eval space
    """
    # Patch centres in model's internal resolution (img_size x img_size)
    ys = np.arange(h_patches) * patch_size + patch_size // 2
    xs = np.arange(w_patches) * patch_size + patch_size // 2

    # Scale from model resolution to evaluation resolution
    scale_x = eval_w / img_size
    scale_y = eval_h / img_size

    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    coords = np.stack([
        grid_x.ravel() * scale_x,
        grid_y.ravel() * scale_y,
    ], axis=1)

    return coords


# ═════════════════════════════════════════════════════════════════════
# Pair ID (must match lmz's naming convention)
# ═════════════════════════════════════════════════════════════════════

def image_output_id(name: str) -> str:
    """Convert image path to a flat ID string, matching evaluate script convention."""
    path = Path(name)
    scene = next((part for part in path.parts if part.startswith("scene")), None)
    if scene is not None:
        return f"{scene}_{path.stem}"

    parent_parts = [part for part in path.parts[:-1] if part not in ("", ".")]
    if parent_parts:
        return "{}_{}".format("_".join(parent_parts), path.stem)
    return path.stem


def pair_output_id(name0: str, name1: str) -> str:
    return f"{image_output_id(name0)}_{image_output_id(name1)}"


# ═════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════

def main():
    args = get_extract_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_w, eval_h = args.eval_resize
    print(f"Model input size: {args.img_size}x{args.img_size}")
    print(f"Eval coordinate system: {eval_w}x{eval_h}")

    # ── Load model ───────────────────────────────────────────────────
    print("Loading model...")
    model = DINOv2Matcher(checkpoint_path=None, freeze_blocks=0)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded fine-tuned checkpoint (epoch {ckpt.get('epoch', '?')})")
    else:
        # Might be a raw state_dict
        model.load_state_dict(ckpt, strict=False)
        print("Loaded state dict directly")

    model = model.to(device)
    model.eval()

    patch_size = model.patch_size
    h_patches = args.img_size // patch_size
    w_patches = args.img_size // patch_size

    # Pre-compute patch coordinates in evaluation space (same for all images)
    coords_eval = get_patch_coordinates_eval(
        h_patches, w_patches, patch_size, args.img_size, eval_w, eval_h
    )

    # ── Load pairs ───────────────────────────────────────────────────
    with open(args.pairs, "r") as f:
        pairs = [line.strip().split() for line in f if line.strip()]

    pairs = [p for p in pairs if len(p) == 38]
    print(f"Processing {len(pairs)} pairs...")

    # Validate coordinate range
    print(f"Patch coordinate range: x=[{coords_eval[:, 0].min():.1f}, {coords_eval[:, 0].max():.1f}], "
          f"y=[{coords_eval[:, 1].min():.1f}, {coords_eval[:, 1].max():.1f}]")

    t0 = time.time()
    n_skipped = 0
    for i, pair in enumerate(pairs):
        name0, name1 = pair[0], pair[1]
        pid = pair_output_id(name0, name1)

        path0 = str(Path(args.data_root) / name0)
        path1 = str(Path(args.data_root) / name1)

        try:
            img_a, _ = load_and_preprocess(path0, args.img_size)
            img_b, _ = load_and_preprocess(path1, args.img_size)
        except FileNotFoundError as e:
            n_skipped += 1
            continue

        # Extract features
        with torch.no_grad():
            desc_a = model(img_a.unsqueeze(0).to(device))[0]  # (N, D)
            desc_b = model(img_b.unsqueeze(0).to(device))[0]  # (N, D)

        # MNN matching
        idx_a, idx_b, scores = mutual_nearest_neighbors(desc_a, desc_b)

        # Write CSV — coordinates are in EVAL space (e.g. 640x480)
        csv_path = output_dir.resolve() / f"{pid}_matches.csv"
        safe_csv_path = "\\\\?\\" + str(csv_path)
        with open(safe_csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["left_idx", "right_idx", "x1", "y1", "x2", "y2", "score"])
            for a, b, s in zip(idx_a, idx_b, scores):
                x1, y1 = coords_eval[a]
                x2, y2 = coords_eval[b]
                writer.writerow([int(a), int(b), f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}", f"{s}"])

        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(pairs)}] {elapsed:.1f}s | {pid} | {len(idx_a)} matches")

    elapsed = time.time() - t0
    print(f"\nDone! {len(pairs)} pairs in {elapsed:.1f}s (skipped {n_skipped})")
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
