# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Course project: fine-tune DINOv3 (ViT-L/16) for local feature matching on two datasets — NAVI (object-level, 3000 test pairs) and ScanNet (indoor scenes, 1500 test pairs). The pipeline: train with contrastive loss → extract dense features → MNN matching → essential matrix pose evaluation.

Evaluation metrics: AUC@5, AUC@10, AUC@20 (pose error from estimated essential matrix), and Precision (fraction of matches satisfying epipolar constraint). These are computed by `evaluate/evaluate_csv_essential.py`, which depends on SuperGlue utilities in `Superglue/models/utils.py`.

## Environment

All commands run inside conda environment `llmdevelop`. CUDA is required. On Windows, use `conda run -n llmdevelop python ...`; on Linux, the shell script `run_all_5090.sh` wraps everything with `conda run`.

## Pipeline stages (per dataset)

### 1. Generate training pairs (NAVI only)
```
python -m finetune.generate_train_pairs --data_root full_dataset/navi_v1.5 --output finetune/navi_train_pairs.txt --max_pairs_per_scene 20
```
Extracts camera parameters from NAVI's `annotations.json`, computes relative poses, outputs 38-token-per-line pairs file. ScanNet reuses `scannet_with_gt.txt` directly.

### 2. LoRA training
```
python -m finetune.train_lora \
    --checkpoint dinov3_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
    --train_pairs finetune/navi_train_pairs.txt \
    --data_root full_dataset/navi_v1.5 \
    --depth_root full_dataset/navi_v1.5 \
    --output_dir finetune_output_lora_navi \
    --epochs 15 --batch_size 8 --img_size 448 --lora_rank 4 --lr 1e-3
```
Key entry point: `finetune/train_lora.py` → `LoRADINOv3Matcher` (a custom ViT-L/16 backbone built from scratch in `finetune/model.py`, with LoRA adapters injected via `finetune/lora.py`). No external dependency on torch.hub or timm — the ViT architecture (RoPE, LayerScale, storage tokens) is hand-built to match the DINOv3 weight file exactly.

The legacy Projection Head approach (`finetune/train.py` → `DINOv2Matcher`) is obsolete and should not be used; it caused catastrophic mode collapse.

### 3. Feature extraction + MNN matching
```
python -m finetune.extract_lora \
    --checkpoint finetune_output_lora_navi/checkpoint_latest.pth \
    --pretrained dinov3_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
    --pairs datasets/navi_with_gt.txt \
    --data_root datasets/test/navi_resized \
    --output_dir mnn_matching_lora_navi/navi \
    --img_size 448 --eval_resize 640 480
```
The `--eval_resize` parameter maps patch coordinates from the model's 448×448 space to the evaluation 640×480 coordinate system. **This must match what the evaluate script uses** — mismatched coordinate systems were a previously discovered bug.

### 4. Evaluation
```
python evaluate/evaluate_csv_essential.py \
    --input_pairs datasets/navi_with_gt.txt \
    --input_dir datasets/test/navi_resized \
    --input_csv_dir mnn_matching_lora_navi/navi \
    --output_dir evaluate/navi_lora
```
Results written to `evaluation_results.txt` and `evaluation_results.json` in the output dir.

### One-shot full pipeline (Linux, RTX 5090)
```bash
chmod +x run_all_5090.sh && ./run_all_5090.sh
```
Runs all four stages for both datasets with batch_size=8 (inter-image negatives).

## Architecture

### Core model: `finetune/model.py`
`DINOv3Backbone` — a hand-built ViT-L/16 (24 blocks, 1024-dim, 16 heads, RoPE 2D positional encoding, LayerScale, 4 storage tokens). Loads weights from the `.pth` file that uses key prefixes `model.*` or `teacher.backbone.*`. The checkpoint contains RoPE `periods` as a buffer. The backbone outputs `{"x_norm_patchtokens": (B, h*w, 1024), "x_norm_clstoken": ...}`.

`DINOv2Matcher` — wraps backbone + `ProjectionHead` (1024→640→256). **Obsolete; use LoRA instead.**

### LoRA: `finetune/lora.py`
`LoRALinear` wraps a frozen `nn.Linear` with trainable low-rank matrices A (Kaiming init) and B (zero init). `inject_lora()` replaces QKV and proj layers across all 24 blocks. Only `lora_A` and `lora_B` parameters are trainable (~0.39M params).

### LoRA trainer: `finetune/train_lora.py`
`LoRADINOv3Matcher` — backbone + LoRA, **no projection head**. Outputs are 1024-dim native DINO features (L2-normalized). The forward path stays in DINO's original feature space; LoRA's zero-init guarantees training starts at zero-shot quality.

### Dataset: `finetune/dataset.py`
`MatchingPairDataset` parses the 38-token pairs format. Compute correspondences either via depth reprojection (`compute_correspondences_with_depth`) or epipolar geometry with mutual nearest-neighbor cross-check (`compute_correspondences_epipolar`). Returns (img_a, img_b, idx_a, idx_b) where idx_a/b are patch indices for valid correspondences.

### Loss: `finetune/loss.py`
`MatchingLoss` wraps `HardInfoNCELoss` — InfoNCE with hard-negative mining. **Critical detail: Safe Radius masking.** When `batch_size=1`, negatives come from the same image. The Safe Radius (default 5 patches) prevents spatially-nearby patches from being used as negatives, which would otherwise force the model to push similar features apart and cause mode collapse (loss converging to `ln(K+1) ≈ 4.86`). The `batch_idx_a/batch_idx_b` parameters ensure masking only applies within the same image — inter-image negatives (when batch_size > 1) are never masked.

### Feature extraction: `finetune/extract_lora.py` and `finetune/extract_and_match.py`
Both produce CSV files with columns `left_idx, right_idx, x1, y1, x2, y2, score`. `extract_lora.py` uses the LoRA model (1024-dim); `extract_and_match.py` uses the legacy Projection Head model (256-dim). Both use `get_patch_coordinates_eval()` to map coordinates from model space to eval space.

### Evaluation: `evaluate/evaluate_csv_essential.py`
Loads CSV matches, estimates essential matrix via OpenCV USAC, computes pose errors, derives AUC and Precision. Depends on `Superglue/models/utils.py` for `pose_auc`, `compute_epipolar_error`, `compute_pose_error`, `scale_intrinsics`, `rotate_intrinsics`, `rotate_pose_inplane`.

### Presentation tools: `presentation/`
- `pca_visualizer.py` — PCA feature visualization comparing zero-shot vs LoRA features
- `plot_results.py` — generates precision comparison bar charts and loss curves for PPT

## Data files

- `datasets/navi_with_gt.txt` (3000 lines × 38 tokens) — NAVI test pairs with ground-truth poses
- `datasets/scannet_with_gt.txt` (1500 lines × 38 tokens) — ScanNet test pairs
- `finetune/navi_train_pairs.txt` — generated training pairs from full NAVI dataset (5596 pairs)
- `datasets/test/navi_resized/` — NAVI test images (longest edge scaled to 1024)
- `datasets/test/scans_test/` — ScanNet test images (640×480)
- `full_dataset/navi_v1.5/` — full NAVI dataset with depth maps and annotations
- `dinov3_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` — DINOv3 ViT-L/16 pretrained weights (~1.2 GB, gitignored)

## Key architectural decisions and pitfalls

1. **No Projection Head**: The Projection Head approach was abandoned because random initialization destroyed DINO's pretrained features, causing precision to drop from 49% to 3%. LoRA preserves the native 1024-dim space.

2. **Safe Radius in loss**: Without it, single-image InfoNCE forces spatially-adjacent similar patches apart, causing mode collapse (loss stuck at ln(K+1)). The Safe Radius mask excludes negatives within 5 patches that belong to the same image.

3. **Coordinate system alignment**: The `--eval_resize` parameter must match between extraction and evaluation scripts. Mismatch was a previously fixed critical bug.

4. **Batch size matters**: batch_size=1 forces intra-image negatives (risky); batch_size≥2 enables inter-image negatives (safer). The RTX 5090 (32GB) script uses batch_size=8.

5. **Checkpoint loading**: The DINOv3 weight file may have nested keys (`model.*` or `teacher.backbone.*`). Both `train_lora.py` and `model.py` handle both formats.
