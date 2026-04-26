"""
DINOv2-v3 model wrapper with projection head for local feature matching.

Builds the ViT-L/16 architecture from scratch to match the weight file
exactly (RoPE, layer scale, storage tokens, etc.), with no external
dependency on torch.hub or timm.

Architecture:
  DINOv2-v3 ViT-L/16 backbone (partially frozen)
  -> Dense patch tokens (H/16 x W/16 x 1024)
  -> Projection Head (1024 -> 640 -> 256, L2-normalized)
"""

from __future__ import annotations

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as cfg


# =====================================================================
# Building blocks for ViT (matching DINOv2-v3 weight format)
# =====================================================================

class RoPE2D(nn.Module):
    """Rotary Position Embedding for 2-D grids (image patches)."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.half_head_dim = head_dim // 2
        # 'periods' loaded from checkpoint; checkpoint has 16 values
        # (covers half of half_head_dim for y + x split)
        self.n_periods = self.half_head_dim // 2  # 16 for ViT-L
        self.register_buffer("periods", torch.ones(self.n_periods))

    @torch.no_grad()
    def _build_freqs(self, h: int, w: int, device: torch.device):
        """Build frequency grid for h x w patches."""
        freqs = 2.0 * math.pi / self.periods  # (n_periods,)
        ys = torch.arange(h, device=device, dtype=torch.float32)
        xs = torch.arange(w, device=device, dtype=torch.float32)
        # Split periods: half for y, half for x
        n = len(self.periods)
        d2 = n // 2
        freq_y = freqs[:d2]
        freq_x = freqs[d2:]
        # outer products
        angles_y = torch.einsum("i,j->ij", ys, freq_y)  # (h, d2)
        angles_x = torch.einsum("i,j->ij", xs, freq_x)  # (w, n-d2)
        # broadcast to (h, w, d)
        ay = angles_y[:, None, :].expand(h, w, d2)
        ax = angles_x[None, :, :].expand(h, w, n - d2)
        angles = torch.cat([ay, ax], dim=-1)  # (h, w, n)
        return angles.reshape(h * w, n)  # (N, n)

    def forward(self, q: torch.Tensor, k: torch.Tensor, h: int, w: int):
        """Apply RoPE to q and k tensors.
        q, k: (B, num_heads, N, head_dim) where N = 1(cls) + num_patches + num_storage
        """
        B, H, N, D = q.shape
        n_patches = h * w
        n = len(self.periods)  # number of dims to rotate

        angles = self._build_freqs(h, w, q.device)  # (n_patches, n)
        cos_a = angles.cos()
        sin_a = angles.sin()

        def rotate(x):
            # x: (B, H, N, D)
            # Only rotate the patch tokens (skip cls + storage)
            # Identify patch token positions: typically positions 1..n_patches
            # (cls at 0, storage tokens at end)
            x_cls = x[:, :, :1, :]
            x_patches = x[:, :, 1:1 + n_patches, :]
            x_rest = x[:, :, 1 + n_patches:, :]

            # Apply rotation to first 2*n dims of patch tokens
            x1 = x_patches[..., :n]
            x2 = x_patches[..., n:2 * n]
            c = cos_a[None, None, :, :]  # (1, 1, n_patches, n)
            s = sin_a[None, None, :, :]
            rx1 = x1 * c - x2 * s
            rx2 = x1 * s + x2 * c
            x_patches_rot = torch.cat([rx1, rx2, x_patches[..., 2 * n:]], dim=-1)
            return torch.cat([x_cls, x_patches_rot, x_rest], dim=2)

        return rotate(q), rotate(k)


class Attention(nn.Module):
    """Multi-head self-attention with optional RoPE."""

    def __init__(self, dim: int, num_heads: int = 16, qkv_bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # DINOv2-v3 has a bias_mask stored alongside qkv; register under
        # the same sub-path the checkpoint uses: attn.qkv.bias_mask
        self.qkv.register_buffer(
            "bias_mask", torch.ones(dim * 3, dtype=torch.bool), persistent=False
        )
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, rope: RoPE2D | None = None, h: int = 0, w: int = 0):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, D)
        q, k, v = qkv.unbind(0)

        if rope is not None and h > 0 and w > 0:
            q, k = rope(q, k, h, w)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class Mlp(nn.Module):
    """Standard 2-layer MLP (matching DINOv2 fc1/fc2 naming)."""

    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class LayerScale(nn.Module):
    """Per-channel scaling (DINOv2 uses this in every block)."""

    def __init__(self, dim: int, init_value: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class Block(nn.Module):
    """Transformer block matching DINOv2-v3 structure."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.ls1 = LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.ls2 = LayerScale(dim)

    def forward(self, x, rope=None, h=0, w=0):
        x = x + self.ls1(self.attn(self.norm1(x), rope, h, w))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """Image to patch embedding."""

    def __init__(self, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 1024):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, 3, H, W) -> (B, N, C)
        x = self.proj(x)  # (B, C, h, w)
        x = x.flatten(2).transpose(1, 2)
        return x


class DINOv3Backbone(nn.Module):
    """
    ViT-L/16 backbone that exactly matches the DINOv2-v3 weight file.

    Keys expected in state_dict:
      cls_token, storage_tokens, mask_token,
      patch_embed.proj.{weight,bias},
      rope_embed.periods,
      blocks.{0..23}.{norm1,attn,ls1,norm2,mlp,ls2}.*,
      norm.{weight,bias}
    """

    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_storage_tokens: int = 4,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Patch embedding
        self.patch_embed = PatchEmbed(patch_size, 3, embed_dim)

        # Special tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.storage_tokens = nn.Parameter(torch.zeros(1, num_storage_tokens, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))  # 2D to match checkpoint

        # RoPE
        self.rope_embed = RoPE2D(embed_dim, num_heads)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])

        # Final norm
        self.norm = nn.LayerNorm(embed_dim)

    def forward_features(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: (B, 3, H, W) — H, W must be divisible by patch_size
        Returns:
            dict with 'x_norm_patchtokens' (B, N, C)
        """
        B, _, H, W = x.shape
        h = H // self.patch_size
        w = W // self.patch_size

        # Patch embed
        x = self.patch_embed(x)  # (B, h*w, C)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, 1+h*w, C)

        # Append storage tokens
        storage = self.storage_tokens.expand(B, -1, -1)
        x = torch.cat([x, storage], dim=1)  # (B, 1+h*w+S, C)

        # Forward through blocks with RoPE
        for blk in self.blocks:
            x = blk(x, rope=self.rope_embed, h=h, w=w)

        x = self.norm(x)

        # Extract patch tokens (skip cls at 0, skip storage at end)
        n_patches = h * w
        patch_tokens = x[:, 1:1 + n_patches, :]
        cls_token = x[:, 0:1, :]

        return {
            "x_norm_patchtokens": patch_tokens,
            "x_norm_clstoken": cls_token,
        }


# =====================================================================
# Projection Head
# =====================================================================

class ProjectionHead(nn.Module):
    """Small MLP projecting DINOv2 patch tokens to a lower-dim space
    with L2 normalisation for contrastive learning."""

    def __init__(self, in_dim: int = cfg.EMBED_DIM, proj_dim: int = cfg.PROJ_DIM):
        super().__init__()
        mid_dim = (in_dim + proj_dim) // 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, C) patch tokens from DINOv2
        Returns:
            (B, N, proj_dim) L2-normalised descriptors
        """
        x = self.net(x)
        return F.normalize(x, p=2, dim=-1)


# =====================================================================
# DINOv2Matcher (backbone + projection head)
# =====================================================================

class DINOv2Matcher(nn.Module):
    """
    DINOv2-v3 ViT backbone + learnable projection head.

    The backbone can be partially frozen (first `freeze_blocks` transformer
    blocks stay fixed) so that only the last few blocks and the projection
    head are trained.
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        freeze_blocks: int = cfg.FREEZE_BLOCKS,
        proj_dim: int = cfg.PROJ_DIM,
    ):
        super().__init__()

        # 1) Build backbone
        self.backbone = DINOv3Backbone(
            patch_size=cfg.VIT_PATCH_SIZE,
            embed_dim=cfg.EMBED_DIM,
            depth=cfg.NUM_BLOCKS,
            num_heads=16,
            mlp_ratio=4.0,
        )

        # 2) Load pre-trained weights
        if checkpoint_path is not None:
            self._load_weights(checkpoint_path)

        # 3) Freeze layers
        self._freeze(freeze_blocks)

        # 4) Projection head
        self.proj_head = ProjectionHead(in_dim=cfg.EMBED_DIM, proj_dim=proj_dim)

        self.patch_size = cfg.VIT_PATCH_SIZE
        self.embed_dim = cfg.EMBED_DIM

    def _load_weights(self, path: str):
        """Load pre-trained DINOv2-v3 weights."""
        state_dict = torch.load(path, map_location="cpu", weights_only=False)

        # Handle nested state_dict
        if "model" in state_dict and isinstance(state_dict["model"], dict):
            state_dict = state_dict["model"]
        elif "teacher" in state_dict and isinstance(state_dict["teacher"], dict):
            state_dict = state_dict["teacher"]
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

        # Filter out keys we don't use (e.g. qkv.bias_mask we register as buffer)
        missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[model] Missing keys ({len(missing)}): {missing[:5]}...")
        if unexpected:
            print(f"[model] Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
        print(f"[model] Loaded pre-trained weights from {path}")

    def _freeze(self, freeze_blocks: int):
        """Freeze patch embed + first `freeze_blocks` transformer blocks."""
        # Freeze patch embedding
        for p in self.backbone.patch_embed.parameters():
            p.requires_grad = False

        # Freeze special tokens
        self.backbone.cls_token.requires_grad = False
        self.backbone.storage_tokens.requires_grad = False
        self.backbone.mask_token.requires_grad = False

        # RoPE is not trainable (buffer)

        # Freeze first N blocks
        for i, block in enumerate(self.backbone.blocks):
            requires_grad = i >= freeze_blocks
            for p in block.parameters():
                p.requires_grad = requires_grad

        # Norm is trainable
        for p in self.backbone.norm.parameters():
            p.requires_grad = True

        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        print(f"[model] Trainable: {n_train/1e6:.1f}M / {n_total/1e6:.1f}M params "
              f"({100*n_train/n_total:.1f}%)")

    def extract_dense_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract dense patch-level descriptors (before projection).

        Args:
            x: (B, 3, H, W) normalised images (H, W divisible by 16)
        Returns:
            (B, h*w, embed_dim) raw patch tokens
        """
        out = self.backbone.forward_features(x)
        return out["x_norm_patchtokens"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward: backbone -> projection head.

        Returns:
            (B, h*w, proj_dim) L2-normalised descriptors
        """
        patch_tokens = self.extract_dense_features(x)
        return self.proj_head(patch_tokens)

    def get_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Returns descriptors reshaped to (B, D, h, w)."""
        h = x.shape[2] // self.patch_size
        w = x.shape[3] // self.patch_size
        desc = self.forward(x)
        return desc.permute(0, 2, 1).reshape(-1, desc.shape[-1], h, w)

    @torch.no_grad()
    def get_patch_coords(self, H: int, W: int) -> torch.Tensor:
        """Return (h*w, 2) pixel coordinates of patch centres — (x, y)."""
        ps = self.patch_size
        h, w = H // ps, W // ps
        ys = torch.arange(h) * ps + ps // 2
        xs = torch.arange(w) * ps + ps // 2
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1).float()
