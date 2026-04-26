"""Quick smoke test: load weights, do a dummy forward pass."""
import torch
import sys
sys.path.insert(0, '.')

from finetune.model import DINOv2Matcher

print("=" * 60)
print("  DINOv2 Fine-tuning Smoke Test")
print("=" * 60)

# 1. Load model with pre-trained weights
print("\n[1/3] Loading model with pre-trained weights...")
model = DINOv2Matcher(
    checkpoint_path="dinov3_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    freeze_blocks=22,
)
print("  OK - Model loaded successfully.")

# 2. Move to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n[2/3] Moving to device: {device}")
model = model.to(device)
model.eval()
print("  OK - Model on device.")

# 3. Dummy forward pass
print("\n[3/3] Running dummy forward pass (1 x 3 x 448 x 448)...")
with torch.no_grad():
    dummy = torch.randn(1, 3, 448, 448, device=device)
    desc = model(dummy)
    print(f"  Output shape: {desc.shape}")
    print(f"  Expected: (1, {448//16 * 448//16}, 256) = (1, {28*28}, 256)")
    assert desc.shape == (1, 28*28, 256), f"Shape mismatch: {desc.shape}"
    print("  OK - Forward pass successful!")

# Memory info
if torch.cuda.is_available():
    mem_alloc = torch.cuda.max_memory_allocated() / 1024**3
    mem_reserved = torch.cuda.max_memory_reserved() / 1024**3
    print(f"\n  GPU memory: {mem_alloc:.2f} GB allocated, {mem_reserved:.2f} GB reserved")

print("\n" + "=" * 60)
print("  ALL TESTS PASSED")
print("=" * 60)
