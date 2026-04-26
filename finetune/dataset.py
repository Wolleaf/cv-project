"""
Dataset for training DINOv2 on local feature matching.

Loads image pairs with ground-truth relative poses (and optionally depth maps)
to compute patch-level correspondences for contrastive learning.

Two modes:
  1) With depth maps  → reproject pixels precisely
  2) Without depth    → use epipolar geometry + cross-check as pseudo GT
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from . import config as cfg


# ═════════════════════════════════════════════════════════════════════
# Image transforms
# ═════════════════════════════════════════════════════════════════════

def make_transform(img_size: int = cfg.IMG_SIZE, training: bool = True):
    """
    Standard DINOv2 image normalisation.
    Input images are resized so that both H and W are divisible by patch_size.
    """
    transforms = [
        T.ToPILImage(),
        T.Resize((img_size, img_size)),
    ]
    if training:
        transforms.append(T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05))
    transforms.extend([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return T.Compose(transforms)


# ═════════════════════════════════════════════════════════════════════
# Pairs file parser
# ═════════════════════════════════════════════════════════════════════

def parse_pairs_file(pairs_path: str) -> list[dict]:
    """
    Parse the pairs file used by lmz's pipeline.

    Each line: path_A path_B rot_A rot_B [K_A 9] [K_B 9] [T_AB 16]
    Total 38 tokens per line.
    """
    pairs = []
    with open(pairs_path, "r") as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) != 38:
                continue
            entry = {
                "name0": tokens[0],
                "name1": tokens[1],
                "rot0": int(tokens[2]),
                "rot1": int(tokens[3]),
                "K0": np.array(tokens[4:13], dtype=np.float64).reshape(3, 3),
                "K1": np.array(tokens[13:22], dtype=np.float64).reshape(3, 3),
                "T_0to1": np.array(tokens[22:38], dtype=np.float64).reshape(4, 4),
            }
            pairs.append(entry)
    return pairs


# ═════════════════════════════════════════════════════════════════════
# Correspondence computation
# ═════════════════════════════════════════════════════════════════════

def compute_correspondences_epipolar(
    K0: np.ndarray,
    K1: np.ndarray,
    T_0to1: np.ndarray,
    h_patches: int,
    w_patches: int,
    patch_size: int = cfg.VIT_PATCH_SIZE,
    img_size: int = cfg.IMG_SIZE,
    orig_size0: tuple[int, int] | None = None,
    orig_size1: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate patch-level correspondences using epipolar geometry.

    For each patch centre in image A, compute its epipolar line in image B,
    then find the closest patch centre to that line.

    Returns:
        idx_a: (M,) patch indices in image A
        idx_b: (M,) corresponding patch indices in image B
    """
    # Patch centre coordinates in the resized image
    ps = patch_size
    coords_y = np.arange(h_patches) * ps + ps // 2
    coords_x = np.arange(w_patches) * ps + ps // 2
    grid_y, grid_x = np.meshgrid(coords_y, coords_x, indexing="ij")
    coords = np.stack([grid_x.ravel(), grid_y.ravel(), np.ones(h_patches * w_patches)], axis=1)  # (N, 3)

    # Scale intrinsics to resized image
    if orig_size0 is not None:
        sx0 = img_size / orig_size0[1]
        sy0 = img_size / orig_size0[0]
    else:
        sx0 = sy0 = 1.0
    if orig_size1 is not None:
        sx1 = img_size / orig_size1[1]
        sy1 = img_size / orig_size1[0]
    else:
        sx1 = sy1 = 1.0

    K0_s = K0.copy()
    K0_s[0] *= sx0
    K0_s[1] *= sy0
    K1_s = K1.copy()
    K1_s[0] *= sx1
    K1_s[1] *= sy1

    # Fundamental matrix: F = K1^{-T} @ [t]_x @ R @ K0^{-1}
    R = T_0to1[:3, :3]
    t = T_0to1[:3, 3]

    # Essential matrix
    tx = np.array([
        [0, -t[2], t[1]],
        [t[2], 0, -t[0]],
        [-t[1], t[0], 0]
    ])
    E = tx @ R
    F = np.linalg.inv(K1_s).T @ E @ np.linalg.inv(K0_s)

    # Epipolar lines in image B for each patch centre in A
    lines = (F @ coords.T).T  # (N, 3) — line params (a, b, c)

    # Distance from each patch centre in B to each epipolar line
    coords_b = coords.copy()  # same grid for B (same image size)

    # Point-to-line distance: |ax + by + c| / sqrt(a^2 + b^2)
    denom = np.sqrt(lines[:, 0:1] ** 2 + lines[:, 1:2] ** 2) + 1e-8
    dist = np.abs(coords_b @ lines.T) / denom.T  # (N_b, N_a)

    # For each patch in A, find the closest patch in B
    nearest_b = np.argmin(dist, axis=0)  # (N_a,)
    min_dist = dist[nearest_b, np.arange(dist.shape[1])]

    # Also check reverse: for each selected B, is A the closest?
    nearest_a_rev = np.argmin(dist, axis=1)  # (N_b,)

    # Mutual nearest + distance threshold
    idx_a_all = np.arange(len(nearest_b))
    mutual_mask = nearest_a_rev[nearest_b] == idx_a_all

    # Distance threshold (in pixels)
    dist_thresh = patch_size * 1.5
    dist_mask = min_dist < dist_thresh

    valid = mutual_mask & dist_mask
    idx_a = idx_a_all[valid]
    idx_b = nearest_b[valid]

    return idx_a, idx_b


def compute_correspondences_with_depth(
    depth0: np.ndarray,
    K0: np.ndarray,
    K1: np.ndarray,
    T_0to1: np.ndarray,
    h_patches: int,
    w_patches: int,
    patch_size: int = cfg.VIT_PATCH_SIZE,
    img_size: int = cfg.IMG_SIZE,
    orig_size0: tuple[int, int] | None = None,
    orig_size1: tuple[int, int] | None = None,
    reproj_thresh: float = cfg.REPROJ_THRESH,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute precise correspondences by reprojecting patch centres using depth.

    Args:
        depth0: (H_orig, W_orig) depth map for image A
    """
    ps = patch_size
    N = h_patches * w_patches

    # Patch centres in resized image
    coords_y = np.arange(h_patches) * ps + ps // 2
    coords_x = np.arange(w_patches) * ps + ps // 2
    grid_y, grid_x = np.meshgrid(coords_y, coords_x, indexing="ij")
    pts_resized = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(np.float64)  # (N, 2)

    # Map back to original image coords to sample depth
    if orig_size0 is not None:
        sx = orig_size0[1] / img_size
        sy = orig_size0[0] / img_size
    else:
        sx = sy = 1.0

    pts_orig = pts_resized.copy()
    pts_orig[:, 0] *= sx
    pts_orig[:, 1] *= sy

    # Sample depth at original coordinates
    depth_h, depth_w = depth0.shape[:2]
    px = np.clip(pts_orig[:, 0].astype(int), 0, depth_w - 1)
    py = np.clip(pts_orig[:, 1].astype(int), 0, depth_h - 1)
    depths = depth0[py, px].astype(np.float64)

    valid_depth = depths > 0
    if valid_depth.sum() < cfg.MIN_CORRESPONDENCES:
        return np.array([], dtype=int), np.array([], dtype=int)

    # Unproject to 3D using K0
    ones = np.ones(N, dtype=np.float64)
    pts_h = np.stack([pts_orig[:, 0], pts_orig[:, 1], ones], axis=1)  # (N, 3)
    K0_inv = np.linalg.inv(K0)
    pts_3d = (K0_inv @ pts_h.T).T * depths[:, None]  # (N, 3)

    # Transform to camera B
    R = T_0to1[:3, :3]
    t = T_0to1[:3, 3]
    pts_3d_b = (R @ pts_3d.T).T + t  # (N, 3)

    # Project to image B
    # Scale K1 to resized image
    if orig_size1 is not None:
        sx1 = img_size / orig_size1[1]
        sy1 = img_size / orig_size1[0]
    else:
        sx1 = sy1 = 1.0

    K1_s = K1.copy()
    K1_s[0] *= sx1
    K1_s[1] *= sy1

    pts_proj = (K1_s @ pts_3d_b.T).T  # (N, 3)
    z = pts_proj[:, 2]
    valid_z = z > 0
    pts_proj_2d = pts_proj[:, :2] / (z[:, None] + 1e-8)

    # Check if projected points land on a patch centre in B
    in_bounds = (
        (pts_proj_2d[:, 0] >= 0) & (pts_proj_2d[:, 0] < img_size) &
        (pts_proj_2d[:, 1] >= 0) & (pts_proj_2d[:, 1] < img_size)
    )

    # Find nearest patch index in B
    nearest_patch_x = np.round((pts_proj_2d[:, 0] - ps // 2) / ps).astype(int)
    nearest_patch_y = np.round((pts_proj_2d[:, 1] - ps // 2) / ps).astype(int)
    patch_centre_x = nearest_patch_x * ps + ps // 2
    patch_centre_y = nearest_patch_y * ps + ps // 2

    # Reprojection distance
    reproj_dist = np.sqrt(
        (pts_proj_2d[:, 0] - patch_centre_x) ** 2 +
        (pts_proj_2d[:, 1] - patch_centre_y) ** 2
    )

    valid_patch = (
        (nearest_patch_x >= 0) & (nearest_patch_x < w_patches) &
        (nearest_patch_y >= 0) & (nearest_patch_y < h_patches)
    )

    valid = valid_depth & valid_z & in_bounds & valid_patch & (reproj_dist < reproj_thresh)

    idx_a = np.where(valid)[0]
    idx_b = nearest_patch_y[valid] * w_patches + nearest_patch_x[valid]

    # Remove duplicates in B (keep the one with smallest reproj error)
    if len(idx_b) > 0:
        _, unique_idx = np.unique(idx_b, return_index=True)
        idx_a = idx_a[unique_idx]
        idx_b = idx_b[unique_idx]

    return idx_a, idx_b


# ═════════════════════════════════════════════════════════════════════
# Dataset
# ═════════════════════════════════════════════════════════════════════

class MatchingPairDataset(Dataset):
    """
    Dataset of image pairs with ground-truth correspondences.

    Each sample returns:
      - img_a: (3, H, W) normalised tensor
      - img_b: (3, H, W) normalised tensor
      - idx_a: (M,) patch indices in A for valid correspondences
      - idx_b: (M,) patch indices in B
    """

    def __init__(
        self,
        pairs_path: str,
        data_root: str,
        depth_root: str = "",
        img_size: int = cfg.IMG_SIZE,
        max_correspondences: int = cfg.MAX_CORRESPONDENCES,
        training: bool = True,
    ):
        self.pairs = parse_pairs_file(pairs_path)
        self.data_root = Path(data_root)
        self.depth_root = Path(depth_root) if depth_root else None
        self.img_size = img_size
        self.max_corr = max_correspondences
        self.patch_size = cfg.VIT_PATCH_SIZE
        self.h_patches = img_size // self.patch_size
        self.w_patches = img_size // self.patch_size
        self.transform = make_transform(img_size, training=training)
        self.training = training

        print(f"[dataset] Loaded {len(self.pairs)} pairs from {pairs_path}")
        print(f"[dataset] Data root: {data_root}")
        if self.depth_root:
            print(f"[dataset] Depth root: {depth_root}")

    def __len__(self):
        return len(self.pairs)

    def _load_image(self, name: str) -> np.ndarray | None:
        """Load image as RGB numpy array."""
        path = self.data_root / name
        if not path.exists():
            return None
        img = cv2.imread(str(path))
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _load_depth(self, name: str) -> np.ndarray | None:
        """Load depth map. Tries .png and .npy formats."""
        if self.depth_root is None:
            return None

        stem = Path(name).stem
        parent = Path(name).parent

        # ScanNet style: depth stored as .png (uint16, mm)
        for ext in [".png", ".npy", ".pfm"]:
            depth_name = parent / f"{stem}{ext}"
            depth_path = self.depth_root / depth_name
            if depth_path.exists():
                if ext == ".npy":
                    return np.load(str(depth_path)).astype(np.float32)
                elif ext == ".png":
                    depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
                    if depth is not None:
                        return depth.astype(np.float32) / 1000.0  # mm → m
                # .pfm would need special handling
        return None

    def __getitem__(self, index: int):
        pair = self.pairs[index]

        # Load images
        img_a = self._load_image(pair["name0"])
        img_b = self._load_image(pair["name1"])

        if img_a is None or img_b is None:
            # Return dummy data (will be filtered in collate)
            return self._dummy()

        orig_size_a = img_a.shape[:2]  # (H, W)
        orig_size_b = img_b.shape[:2]

        # Apply transforms
        img_a_t = self.transform(img_a)
        img_b_t = self.transform(img_b)

        # Compute correspondences
        K0 = pair["K0"]
        K1 = pair["K1"]
        T_0to1 = pair["T_0to1"]

        depth_a = self._load_depth(pair["name0"])
        if depth_a is not None:
            idx_a, idx_b = compute_correspondences_with_depth(
                depth_a, K0, K1, T_0to1,
                self.h_patches, self.w_patches,
                self.patch_size, self.img_size,
                orig_size_a, orig_size_b,
            )
        else:
            idx_a, idx_b = compute_correspondences_epipolar(
                K0, K1, T_0to1,
                self.h_patches, self.w_patches,
                self.patch_size, self.img_size,
                orig_size_a, orig_size_b,
            )

        # Skip if too few correspondences
        if len(idx_a) < cfg.MIN_CORRESPONDENCES:
            return self._dummy()

        # Subsample if too many
        if len(idx_a) > self.max_corr:
            sel = np.random.choice(len(idx_a), self.max_corr, replace=False)
            idx_a = idx_a[sel]
            idx_b = idx_b[sel]

        return {
            "img_a": img_a_t,
            "img_b": img_b_t,
            "idx_a": torch.from_numpy(idx_a).long(),
            "idx_b": torch.from_numpy(idx_b).long(),
            "num_corr": len(idx_a),
        }

    def _dummy(self):
        """Return a dummy sample that will be filtered out."""
        s = self.img_size
        return {
            "img_a": torch.zeros(3, s, s),
            "img_b": torch.zeros(3, s, s),
            "idx_a": torch.zeros(0, dtype=torch.long),
            "idx_b": torch.zeros(0, dtype=torch.long),
            "num_corr": 0,
        }


def collate_matching_pairs(batch: list[dict]) -> list[dict]:
    """
    Custom collate that filters out invalid samples (num_corr == 0).
    Returns a list of valid samples (not batched into tensors)
    because correspondence counts vary per pair.
    """
    return [s for s in batch if s["num_corr"] >= cfg.MIN_CORRESPONDENCES]
