import matplotlib.pyplot as plt
import numpy as np
import os

# Set style for academic/presentation look
plt.style.use('ggplot')
plt.rcParams['font.size'] = 12

# Ensure output directory exists
os.makedirs('presentation/result', exist_ok=True)

# =============================================================================
# Data: All experimental results
# =============================================================================
# NAVI Dataset (3000 pairs)
navi_zs     = {'auc5': 0.24, 'auc10': 1.07, 'auc20': 3.31, 'prec': 49.17}
navi_proj   = {'auc5': 0.00, 'auc10': 0.02, 'auc20': 0.15, 'prec': 3.32}
navi_lora_c = {'auc5': 0.00, 'auc10': 0.04, 'auc20': 0.42, 'prec': 5.37}
navi_5090   = {'auc5': 0.00, 'auc10': 0.06, 'auc20': 0.33, 'prec': 14.95}

# ScanNet Dataset (1500 pairs)
scan_zs     = {'auc5': 0.23, 'auc10': 1.18, 'auc20': 4.44, 'prec': 32.20}
scan_proj   = {'auc5': 0.00, 'auc10': 0.07, 'auc20': 0.25, 'prec': 3.47}
scan_lora_c = {'auc5': 0.00, 'auc10': 0.09, 'auc20': 0.96, 'prec': 5.51}
scan_safe   = {'auc5': 0.00, 'auc10': 0.26, 'auc20': 1.84, 'prec': 28.83}
scan_5090   = {'auc5': 0.08, 'auc10': 0.24, 'auc20': 1.84, 'prec': 26.99}

# =============================================================================
# Figure 1: Precision Comparison — Full Evolution (Both Datasets)
# =============================================================================
labels = ['NAVI Dataset', 'ScanNet Dataset']
zero_shot    = [navi_zs['prec'],     scan_zs['prec']]
proj_head    = [navi_proj['prec'],   scan_proj['prec']]
lora_collapse= [navi_lora_c['prec'], scan_lora_c['prec']]
safe_radius  = [None,                scan_safe['prec']]  # NAVI didn't run this config
inter_image  = [navi_5090['prec'],   scan_5090['prec']]

x = np.arange(len(labels))
width = 0.15

fig, ax = plt.subplots(figsize=(12, 7))
rects1 = ax.bar(x - 2*width, zero_shot,     width, label='Zero-Shot (Baseline)', color='#2ca02c', alpha=0.9, edgecolor='#1a7a1a', linewidth=1.2)
rects2 = ax.bar(x - 1*width, proj_head,     width, label='Proj Head (Collapsed)', color='#d62728', alpha=0.85)
rects3 = ax.bar(x,           lora_collapse,  width, label='LoRA (Collapsed)',      color='#ff7f0e', alpha=0.85)
# Safe Radius bar only for ScanNet
ax.bar(x[1] + 1*width, scan_safe['prec'], width, label='LoRA + Safe Radius (B=1)', color='#9467bd', alpha=0.85)
rects5 = ax.bar(x + 2*width, inter_image,   width, label='LoRA + Inter-Image (B=8, 5090)', color='#1f77b4', alpha=0.85)

ax.set_ylabel('Matching Precision (%)', fontsize=14, fontweight='bold')
ax.set_title('Precision Across All Fine-Tuning Strategies\n(Zero-Shot Baseline is Unbeatable)', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='upper right')
ax.set_ylim(0, 60)

# Add value labels
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        if height > 0.5:
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects5)
# Manual label for Safe Radius single bar
ax.annotate(f'{scan_safe["prec"]:.1f}%',
            xy=(x[1] + 1*width, scan_safe['prec']),
            xytext=(0, 3), textcoords="offset points",
            ha='center', va='bottom', fontsize=9, fontweight='bold')

# Add horizontal line for Zero-Shot reference
ax.axhline(y=49.17, color='#2ca02c', linestyle=':', linewidth=1, alpha=0.5)
ax.axhline(y=32.20, color='#2ca02c', linestyle=':', linewidth=1, alpha=0.5)

fig.tight_layout()
plt.savefig('presentation/result/precision_all_methods.png', dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# Figure 2: Loss Paradox (Mode Collapse vs Safe Radius Fix)
# =============================================================================
epochs = np.arange(5)
loss_collapsed   = [4.8552, 4.8550, 4.8551, 4.8551, 4.8552]
loss_safe_radius = [5.4252, 5.2680, 5.1137, 5.1123, 5.1167]

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(epochs, loss_collapsed, marker='o', linestyle='-', color='#d62728', linewidth=3, markersize=9,
        label='Original HardInfoNCE (Collapsed)')
ax.plot(epochs, loss_safe_radius, marker='s', linestyle='-', color='#1f77b4', linewidth=3, markersize=9,
        label='With Safe Radius Masking (Fixed)')

ax.axhline(y=4.8598, color='black', linestyle='--', linewidth=2.5,
           label='Theoretical Collapse Limit: ln(129) ≈ 4.8598')

ax.set_xlabel('Training Epochs (Mini Test Run)', fontsize=13, fontweight='bold')
ax.set_ylabel('Contrastive Loss', fontsize=13, fontweight='bold')
ax.set_title('Loss Paradox: Breaking the Mathematical Collapse Trap', fontsize=15, fontweight='bold', pad=20)
ax.set_xticks(epochs)
ax.legend(fontsize=12, loc='upper right')
ax.grid(True, linestyle=':', alpha=0.7)

textstr = 'Model gets trapped at ln(K+1) due to\ncontradicting spatial semantics in\nIntra-Image negative sampling.'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.6, edgecolor='gray')
ax.text(0.03, 0.45, textstr, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=props, fontweight='bold')

fig.tight_layout()
plt.savefig('presentation/result/loss_paradox.png', dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# Figure 3: AUC@20 Comparison — Shows degradation across all methods
# =============================================================================
methods = ['Zero-Shot', 'Proj Head', 'LoRA\n(Collapsed)', 'LoRA+SafeR\n(B=1)', 'LoRA+InterImg\n(B=8, 5090)']
scannet_auc20 = [scan_zs['auc20'], scan_proj['auc20'], scan_lora_c['auc20'], scan_safe['auc20'], scan_5090['auc20']]
navi_auc20    = [navi_zs['auc20'], navi_proj['auc20'], navi_lora_c['auc20'], None,               navi_5090['auc20']]

fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(len(methods))
width = 0.35

bars_s = ax.bar(x - width/2, scannet_auc20, width, label='ScanNet', color='#1f77b4', alpha=0.85)
# For NAVI, skip the Safe Radius bar
navi_vals_plot = [v if v is not None else 0 for v in navi_auc20]
bars_n = ax.bar(x + width/2, navi_vals_plot, width, label='NAVI', color='#ff7f0e', alpha=0.85)

ax.set_ylabel('Pose AUC@20 (%)', fontsize=13, fontweight='bold')
ax.set_title('Pose Estimation Quality (AUC@20): Every Fine-Tuning Degrades Performance',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=11, fontweight='bold')
ax.legend(fontsize=12)
ax.set_ylim(0, 6)

for bar in bars_s:
    h = bar.get_height()
    if h > 0.1:
        ax.annotate(f'{h:.2f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9, fontweight='bold')
for bar in bars_n:
    h = bar.get_height()
    if h > 0.1:
        ax.annotate(f'{h:.2f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9, fontweight='bold')

fig.tight_layout()
plt.savefig('presentation/result/auc20_all_methods.png', dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# Figure 4: The Fundamental Conflict (Conceptual Diagram as Table)
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 5))
ax.axis('off')

col_labels = ['Aspect', 'DINOv3 Pre-training Objective', 'InfoNCE Fine-tuning Objective']
table_data = [
    ['Goal',           'Encode rich semantic similarity', 'Push non-matching patches apart'],
    ['Same surface\n(e.g. white wall)', 'Features SHOULD be similar\n(semantic continuity)', 'Features MUST be different\n(negative pairs)'],
    ['Result',         'Well-structured feature space', 'Destroys semantic structure'],
    ['Outcome',        'Good zero-shot matching', 'Worse matching after training'],
]

table = ax.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.2)

# Style header
for j in range(3):
    table[(0, j)].set_facecolor('#2c3e50')
    table[(0, j)].set_text_props(color='white', fontweight='bold')
# Style rows
for i in range(1, len(table_data) + 1):
    table[(i, 0)].set_facecolor('#ecf0f1')
    table[(i, 0)].set_text_props(fontweight='bold')
    table[(i, 1)].set_facecolor('#d5f4e6')
    table[(i, 2)].set_facecolor('#fad4d4')

ax.set_title('Root Cause: Fundamental Objective Conflict\nbetween DINOv3 Pre-training and InfoNCE Fine-tuning',
             fontsize=14, fontweight='bold', pad=20)

fig.tight_layout()
plt.savefig('presentation/result/objective_conflict.png', dpi=300, bbox_inches='tight')
plt.close()

print("All presentation plots generated in presentation/result/:")
for f in sorted(os.listdir('presentation/result')):
    if f.endswith('.png'):
        print(f"  - {f}")
