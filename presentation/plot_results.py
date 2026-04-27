import matplotlib.pyplot as plt
import numpy as np
import os

# Set style for academic/presentation look
plt.style.use('ggplot')

# Ensure output directory exists
os.makedirs('presentation/result', exist_ok=True)

# ---------------------------------------------------------
# 1. Plot Precision Comparison (The Collapse)
# ---------------------------------------------------------
labels = ['NAVI Dataset', 'ScanNet Dataset']
zero_shot = [49.33, 32.20]
proj_head = [3.32, 3.47]
lora_collapse = [5.37, 5.51]

x = np.arange(len(labels))
width = 0.25

fig, ax = plt.subplots(figsize=(9, 6))
rects1 = ax.bar(x - width, zero_shot, width, label='Zero-Shot Baseline', color='#2ca02c', alpha=0.85)
rects2 = ax.bar(x, proj_head, width, label='Projection Head (Collapsed)', color='#d62728', alpha=0.85)
rects3 = ax.bar(x + width, lora_collapse, width, label='LoRA (Collapsed)', color='#ff7f0e', alpha=0.85)

ax.set_ylabel('Matching Precision (%)', fontsize=13, fontweight='bold')
ax.set_title('Matching Precision Catastrophe During Initial Fine-Tuning', fontsize=15, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=13, fontweight='bold')
ax.legend(fontsize=12)

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

fig.tight_layout()
plt.savefig('presentation/result/precision_catastrophe.png', dpi=300, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------
# 2. Plot Loss Paradox (Mode Collapse vs Safe Radius Fix)
# ---------------------------------------------------------
# Epochs 0 to 4
epochs = np.arange(5)
# Real log data from previous tests
loss_collapsed = [4.8552, 4.8550, 4.8551, 4.8551, 4.8552]  # Stuck at ln(129)
loss_safe_radius = [5.4252, 5.2680, 5.1137, 5.1123, 5.1167] # Fixed

fig, ax = plt.subplots(figsize=(9, 6))

ax.plot(epochs, loss_collapsed, marker='o', linestyle='-', color='#d62728', linewidth=3, markersize=9, label='Original HardInfoNCE (Collapsed)')
ax.plot(epochs, loss_safe_radius, marker='s', linestyle='-', color='#1f77b4', linewidth=3, markersize=9, label='With Safe Radius Masking (Fixed)')

# Add the theoretical limit line
ax.axhline(y=4.8598, color='black', linestyle='--', linewidth=2.5, label='Theoretical Mode Collapse Limit: ln(129) ≈ 4.8598')

ax.set_xlabel('Training Epochs (Mini Test Run)', fontsize=13, fontweight='bold')
ax.set_ylabel('Contrastive Loss', fontsize=13, fontweight='bold')
ax.set_title('Loss Paradox: Breaking the Mathematical Collapse Trap', fontsize=15, fontweight='bold', pad=20)
ax.set_xticks(epochs)
ax.legend(fontsize=12, loc='upper right')
ax.grid(True, linestyle=':', alpha=0.7)

# Add text box explaining the paradox
textstr = 'Model gets trapped at ln(K+1) due to\ncontradicting spatial semantics in\nIntra-Image negative sampling.'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.6, edgecolor='gray')
ax.text(0.03, 0.45, textstr, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=props, fontweight='bold')

fig.tight_layout()
plt.savefig('presentation/result/loss_paradox.png', dpi=300, bbox_inches='tight')
plt.close()

print("Presentation plots successfully generated in presentation/result/")
