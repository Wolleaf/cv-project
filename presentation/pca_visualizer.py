import argparse
import os
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# We need to import the models
import sys
# Ensure the root directory is in the path so we can import finetune
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from finetune.train_lora import LoRADINOv3Matcher
from finetune.model import DINOv3Backbone

def load_image(path, size=448):
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not load image at {path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    img_tensor = transform(img_rgb)
    return img_rgb, img_tensor

def extract_features_zero_shot(model, img_tensor, device):
    with torch.no_grad():
        out = model.forward_features(img_tensor.unsqueeze(0).to(device))
        features = out["x_norm_patchtokens"][0] # (H*W, D)
        return F.normalize(features, p=2, dim=-1).cpu().numpy()

def extract_features_lora(model, img_tensor, device):
    with torch.no_grad():
        features = model(img_tensor.unsqueeze(0).to(device))[0] # (H*W, D)
        return features.cpu().numpy()

def pca_to_rgb(features, h_patches, w_patches):
    """
    Apply PCA to reduce features to 3 dimensions, and normalize to RGB.
    """
    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(features) # (H*W, 3)
    
    # Normalize to [0, 1] per channel for visualization
    for i in range(3):
        pca_features[:, i] = (pca_features[:, i] - pca_features[:, i].min()) / \
                             (pca_features[:, i].max() - pca_features[:, i].min() + 1e-8)
                             
    pca_img = pca_features.reshape(h_patches, w_patches, 3)
    return pca_img

def plot_pca(original_img, pca_img, title, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(original_img)
    axes[0].set_title("Original Image", fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Resize PCA image to match original for overlay/display
    h, w = original_img.shape[:2]
    pca_resized = cv2.resize(pca_img, (w, h), interpolation=cv2.INTER_NEAREST)
    
    axes[1].imshow(pca_resized)
    axes[1].set_title(title, fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    out_dir = Path("presentation/result")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Pretrained weights path
    pretrained_path = "dinov3_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
    
    # Find a sample image
    import glob
    img_files = glob.glob("datasets/test/navi_resized/**/*.jpg", recursive=True)
    if not img_files:
        print("No test images found!")
        return
        
    sample_path = img_files[0]
    # Try to find a visually interesting one if possible
    for f in img_files:
        if "dino" in f.lower() or "duck" in f.lower() or "dog" in f.lower():
            sample_path = f
            break
            
    print(f"Using sample image: {sample_path}")
    orig_rgb, img_tensor = load_image(sample_path, size=448)
    
    # Patch grid dimensions
    patch_size = 16
    h_patches = 448 // patch_size
    w_patches = 448 // patch_size

    # -------------------------------------------------------------
    # 1. Zero-Shot PCA
    # -------------------------------------------------------------
    print("Generating Zero-Shot PCA...")
    zero_shot_model = DINOv3Backbone(patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4.0)
    
    state_dict = torch.load(pretrained_path, map_location="cpu", weights_only=False)
    if "model" in state_dict and isinstance(state_dict["model"], dict):
        state_dict = state_dict["model"]
    elif "teacher" in state_dict and isinstance(state_dict["teacher"], dict):
        state_dict = state_dict["teacher"]
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        
    zero_shot_model.load_state_dict(state_dict, strict=False)
    zero_shot_model = zero_shot_model.to(device)
    zero_shot_model.eval()
    
    features_zs = extract_features_zero_shot(zero_shot_model, img_tensor, device)
    pca_img_zs = pca_to_rgb(features_zs, h_patches, w_patches)
    plot_pca(orig_rgb, pca_img_zs, "Zero-Shot DINOv3 PCA (Rich Semantics)", out_dir / "pca_zero_shot.png")
    
    # Cleanup memory
    del zero_shot_model
    torch.cuda.empty_cache()

    # -------------------------------------------------------------
    # 2. LoRA (Collapsed) PCA - If available
    # -------------------------------------------------------------
    collapsed_path = "finetune_output_lora_scannet/checkpoint_latest.pth"
    if os.path.exists(collapsed_path):
        print("Generating LoRA (Collapsed) PCA...")
        lora_model = LoRADINOv3Matcher(
            checkpoint_path=pretrained_path,
            lora_rank=4,
            lora_alpha=1.0,
            lora_targets=("qkv",)
        )
        
        ckpt = torch.load(collapsed_path, map_location="cpu")
        if "model_state_dict" in ckpt:
            lora_model.load_state_dict(ckpt["model_state_dict"])
        else:
            lora_model.load_state_dict(ckpt, strict=False)
            
        lora_model = lora_model.to(device)
        lora_model.eval()
        
        features_lora = extract_features_lora(lora_model, img_tensor, device)
        pca_img_lora = pca_to_rgb(features_lora, h_patches, w_patches)
        plot_pca(orig_rgb, pca_img_lora, "LoRA Collapsed PCA (Loss Paradox)", out_dir / "pca_lora_collapsed.png")
        
        del lora_model
        torch.cuda.empty_cache()

    print(f"PCA visualizations saved to {out_dir}")

if __name__ == "__main__":
    main()
