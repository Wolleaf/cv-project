import matplotlib.pyplot as plt
import numpy as np
import os

plt.style.use('ggplot')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.unicode_minus'] = False

os.makedirs('report/figures', exist_ok=True)
SAVE = 'report/figures'

# =============================================================================
# Complete data: ALL 7 experiments
# =============================================================================
# Format: {method: {dataset: [AUC5, AUC10, AUC20, Precision]}}
# NAVI Dataset
navi = {
    'Zero-Shot':           [0.24, 1.07, 3.31, 49.17],
    'Projection Head':     [0.00, 0.03, 0.10,  3.16],
    'StableMatchingLoss':  [0.00, 0.02, 0.11,  5.13],
    'Matchability (A)':    [0.00, 0.02, 0.17,  5.55],
    'Matching Head (B)':   [0.00, 0.00, 0.00,  0.00],
}

scan = {
    'Zero-Shot':           [0.23, 1.18, 4.44, 32.20],
    'Projection Head':     [0.00, 0.07, 0.25,  3.47],
    'LoRA (no SafeR)':     [0.00, 0.09, 0.96,  5.51],
    'LoRA + Safe Radius':  [0.00, 0.26, 1.84, 28.83],
    'StableMatchingLoss':  [0.04, 0.26, 1.43, 10.97],
    'Matchability (A)':    [0.00, 0.00, 0.20,  5.95],
    'Matching Head (B)':   [0.00, 0.00, 0.00,  0.00],
}

# =============================================================================
# Figure 1: Precision comparison — ALL methods, both datasets
# =============================================================================
fig, ax = plt.subplots(figsize=(14, 7))

# Build unified method list
all_methods = ['Zero-Shot', 'Projection\nHead', 'LoRA\n(no SafeR)', 'LoRA +\nSafe Radius',
               'Stable\nMatchingLoss', 'Matchability\nPredictor (A)', 'Matching\nHead (B)']
navi_prec = [49.17, 3.16, None, None, 5.13, 5.55, 0.00]
scan_prec = [32.20, 3.47, 5.51, 28.83, 10.97, 5.95, 0.00]

x = np.arange(len(all_methods))
width = 0.32
colors = ['#2ca02c', '#d62728', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

for i, (method, n_val, s_val) in enumerate(zip(all_methods, navi_prec, scan_prec)):
    if n_val is not None:
        ax.bar(i - width/2, n_val, width, color=colors[i], alpha=0.9, edgecolor='black', linewidth=0.5)
    if s_val is not None:
        ax.bar(i + width/2, s_val, width, color=colors[i], alpha=0.45, edgecolor='black', linewidth=0.5)

# Legend patches
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='gray', alpha=0.9, label='NAVI'),
                   Patch(facecolor='gray', alpha=0.45, label='ScanNet')]
ax.legend(handles=legend_elements, fontsize=11, loc='upper right')

# Value labels
for i, (n_val, s_val) in enumerate(zip(navi_prec, scan_prec)):
    if n_val is not None:
        ax.text(i - width/2, n_val + 0.8, f'{n_val:.1f}%', ha='center', va='bottom', fontsize=7.5, fontweight='bold')
    if s_val is not None:
        ax.text(i + width/2, s_val + 0.8, f'{s_val:.1f}%', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

# Zero-Shot reference lines
ax.axhline(y=49.17, color='#2ca02c', linestyle=':', linewidth=1, alpha=0.4)
ax.axhline(y=32.20, color='#2ca02c', linestyle=':', linewidth=1, alpha=0.4)

ax.set_ylabel('Matching Precision (%)', fontsize=13, fontweight='bold')
ax.set_title('Precision Across All Seven Experiments\n(Zero-Shot baseline remains unbeaten)', fontsize=15, fontweight='bold', pad=18)
ax.set_xticks(x)
ax.set_xticklabels(all_methods, fontsize=9)
ax.set_ylim(0, 58)
ax.grid(axis='y', linestyle=':', alpha=0.3)

fig.tight_layout()
plt.savefig(f'{SAVE}/precision_all_methods.png', dpi=300, bbox_inches='tight')
plt.close()
print('Saved: precision_all_methods.png')

# =============================================================================
# Figure 2: AUC@20 comparison
# =============================================================================
fig, ax = plt.subplots(figsize=(14, 7))

navi_auc20 = [3.31, 0.10, None, None, 0.11, 0.17, 0.00]
scan_auc20 = [4.44, 0.25, 0.96, 1.84, 1.43, 0.20, 0.00]

for i, (method, n_val, s_val) in enumerate(zip(all_methods, navi_auc20, scan_auc20)):
    if n_val is not None:
        ax.bar(i - width/2, n_val, width, color=colors[i], alpha=0.9, edgecolor='black', linewidth=0.5)
    if s_val is not None:
        ax.bar(i + width/2, s_val, width, color=colors[i], alpha=0.45, edgecolor='black', linewidth=0.5)

ax.legend(handles=legend_elements, fontsize=11, loc='upper right')

for i, (n_val, s_val) in enumerate(zip(navi_auc20, scan_auc20)):
    if n_val is not None and n_val > 0.05:
        ax.text(i - width/2, n_val + 0.06, f'{n_val:.2f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')
    if s_val is not None and s_val > 0.05:
        ax.text(i + width/2, s_val + 0.06, f'{s_val:.2f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

ax.axhline(y=3.31, color='#ff7f0e', linestyle=':', linewidth=1, alpha=0.4)
ax.axhline(y=4.44, color='#1f77b4', linestyle=':', linewidth=1, alpha=0.4)

ax.set_ylabel('Pose AUC@20 (%)', fontsize=13, fontweight='bold')
ax.set_title('Pose Estimation AUC@20 Across All Seven Experiments', fontsize=15, fontweight='bold', pad=18)
ax.set_xticks(x)
ax.set_xticklabels(all_methods, fontsize=9)
ax.grid(axis='y', linestyle=':', alpha=0.3)

fig.tight_layout()
plt.savefig(f'{SAVE}/auc20_all_methods.png', dpi=300, bbox_inches='tight')
plt.close()
print('Saved: auc20_all_methods.png')

# =============================================================================
# Figure 3: NAVI Precision trend (chronological)
# =============================================================================
fig, ax = plt.subplots(figsize=(11, 6))
navi_stages = ['Zero-Shot', 'Proj Head', 'Stable\nMatchingLoss', 'Matchability\nPredictor (A)', 'Matching\nHead (B)']
navi_stage_prec = [49.17, 3.16, 5.13, 5.55, 0.00]
navi_stage_auc = [3.31, 0.10, 0.11, 0.17, 0.00]
navi_colors = ['#2ca02c', '#d62728', '#8c564b', '#e377c2', '#7f7f7f']

ax2 = ax.twinx()
bars = ax.bar(navi_stages, navi_stage_prec, color=navi_colors, alpha=0.85, edgecolor='black', linewidth=0.5)
line, = ax2.plot(range(len(navi_stages)), navi_stage_auc, 'o-', color='#1f77b4', linewidth=3, markersize=10, zorder=5)

for bar, val in zip(bars, navi_stage_prec):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.5, f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')
for i, val in enumerate(navi_stage_auc):
    ax2.annotate(f'{val:.2f}', (i, val), textcoords="offset points", xytext=(0, -14), ha='center', fontsize=9, fontweight='bold', color='#1f77b4')

ax.set_ylabel('Matching Precision (%)', fontsize=13, fontweight='bold', color='black')
ax2.set_ylabel('Pose AUC@20 (%)', fontsize=13, fontweight='bold', color='#1f77b4')
ax.set_title('NAVI Dataset: Chronological Precision and AUC Trends', fontsize=14, fontweight='bold', pad=18)
ax.set_ylim(0, 58)
ax2.set_ylim(0, 4.5)

# Legend
from matplotlib.lines import Line2D
leg = [Line2D([0],[0], color=c, lw=10, alpha=0.85) for c in navi_colors] + [Line2D([0],[0], color='#1f77b4', lw=3, marker='o')]
ax.legend(leg, navi_stages + ['AUC@20'], fontsize=9, loc='upper right')

fig.tight_layout()
plt.savefig(f'{SAVE}/navi_chronological.png', dpi=300, bbox_inches='tight')
plt.close()
print('Saved: navi_chronological.png')

# =============================================================================
# Figure 4: ScanNet Precision trend (chronological)
# =============================================================================
fig, ax = plt.subplots(figsize=(11, 6))
scan_stages = ['Zero-Shot', 'Proj Head', 'LoRA\n(no SafeR)', 'LoRA +\nSafe Radius',
               'Stable\nMatchingLoss', 'Matchability\nPredictor (A)', 'Matching\nHead (B)']
scan_stage_prec = [32.20, 3.47, 5.51, 28.83, 10.97, 5.95, 0.00]
scan_stage_auc = [4.44, 0.25, 0.96, 1.84, 1.43, 0.20, 0.00]
scan_colors = ['#2ca02c', '#d62728', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

ax2 = ax.twinx()
bars = ax.bar(scan_stages, scan_stage_prec, color=scan_colors, alpha=0.85, edgecolor='black', linewidth=0.5)
line, = ax2.plot(range(len(scan_stages)), scan_stage_auc, 'o-', color='#1f77b4', linewidth=3, markersize=10, zorder=5)

for bar, val in zip(bars, scan_stage_prec):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.4, f'{val:.1f}%', ha='center', fontsize=8.5, fontweight='bold')
for i, val in enumerate(scan_stage_auc):
    color = 'white' if i == 6 else '#1f77b4'
    ax2.annotate(f'{val:.2f}', (i, val), textcoords="offset points", xytext=(0, -14), ha='center', fontsize=9, fontweight='bold', color='#1f77b4')

ax.set_ylabel('Matching Precision (%)', fontsize=13, fontweight='bold', color='black')
ax2.set_ylabel('Pose AUC@20 (%)', fontsize=13, fontweight='bold', color='#1f77b4')
ax.set_title('ScanNet Dataset: Chronological Precision and AUC Trends', fontsize=14, fontweight='bold', pad=18)
ax.set_ylim(0, 38)
ax2.set_ylim(0, 5.5)

leg = [Line2D([0],[0], color=c, lw=10, alpha=0.85) for c in scan_colors] + [Line2D([0],[0], color='#1f77b4', lw=3, marker='o')]
ax.legend(leg, [s.replace('\n',' ') for s in scan_stages] + ['AUC@20'], fontsize=8, loc='upper right')

fig.tight_layout()
plt.savefig(f'{SAVE}/scannet_chronological.png', dpi=300, bbox_inches='tight')
plt.close()
print('Saved: scannet_chronological.png')

# =============================================================================
# Figure 5: Loss convergence — Mode Collapse proof
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

# Subplot 1: Mode Collapse
epochs_c = np.arange(1, 11)
loss_collapsed = [4.87, 4.86, 4.858, 4.856, 4.8555, 4.8552, 4.8551, 4.8551, 4.8551, 4.8552]
ax1.plot(epochs_c, loss_collapsed, 'o-', color='#d62728', linewidth=2.5, markersize=7, label='Projection Head + InfoNCE')
ax1.axhline(y=4.8598, color='black', linestyle='--', linewidth=2, label=r'$\ln(129) \approx 4.8598$ (random)')
ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax1.set_ylabel('InfoNCE Loss', fontsize=12, fontweight='bold')
ax1.set_title('Mode Collapse: Loss → ln(K+1)', fontsize=13, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, linestyle=':', alpha=0.5)

# Subplot 2: Safe Radius recovery
epochs_s = np.arange(1, 6)
loss_safe = [5.43, 5.27, 5.11, 5.11, 5.12]
ax2.plot(epochs_s, loss_safe, 's-', color='#9467bd', linewidth=2.5, markersize=7, label='LoRA + Safe Radius')
ax2.axhline(y=4.8598, color='black', linestyle='--', linewidth=2, alpha=0.5, label=r'$\ln(129)$ collapse limit')
ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax2.set_ylabel('InfoNCE Loss', fontsize=12, fontweight='bold')
ax2.set_title('Safe Radius: Loss Stays Above Collapse', fontsize=13, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, linestyle=':', alpha=0.5)

fig.suptitle('Loss Dynamics: Proof of Mathematical Mode Collapse vs Safe Radius Rescue', fontsize=14, fontweight='bold', y=1.02)
fig.tight_layout()
plt.savefig(f'{SAVE}/loss_convergence.png', dpi=300, bbox_inches='tight')
plt.close()
print('Saved: loss_convergence.png')

# =============================================================================
# Figure 6: Degradation severity — method vs Zero-Shot
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 7))

methods_short = ['Proj\nHead', 'LoRA\nnoSafeR', 'LoRA+\nSafeR', 'Stable\nMatch', 'Matchab.\n(A)', 'Match\nHead (B)']
navi_degradation = [49.17-3.16, None, None, 49.17-5.13, 49.17-5.55, 49.17-0.00]
scan_degradation = [32.20-3.47, 32.20-5.51, 32.20-28.83, 32.20-10.97, 32.20-5.95, 32.20-0.00]

x = np.arange(len(methods_short))
width = 0.32
for i in range(len(methods_short)):
    if navi_degradation[i] is not None:
        ax.bar(i - width/2, navi_degradation[i], width, color='#ff7f0e', alpha=0.85, edgecolor='black', linewidth=0.5)
    if scan_degradation[i] is not None:
        ax.bar(i + width/2, scan_degradation[i], width, color='#1f77b4', alpha=0.65, edgecolor='black', linewidth=0.5)

for i, (n, s) in enumerate(zip(navi_degradation, scan_degradation)):
    if n is not None:
        ax.text(i - width/2, n + 0.5, f'{n:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold', color='#ff7f0e')
    if s is not None:
        color = 'white' if s > 30 else '#1f77b4'
        ax.text(i + width/2, s + 0.5, f'{s:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold', color='#1f77b4')

legend2 = [Patch(facecolor='#ff7f0e', alpha=0.85, label='NAVI degradation'),
           Patch(facecolor='#1f77b4', alpha=0.65, label='ScanNet degradation')]
ax.legend(handles=legend2, fontsize=10, loc='upper left')

ax.set_ylabel('Precision Drop from Zero-Shot (pp)', fontsize=13, fontweight='bold')
ax.set_title('Degradation Severity: How Much Each Method Destroyed Matching Ability', fontsize=14, fontweight='bold', pad=18)
ax.set_xticks(x)
ax.set_xticklabels(methods_short, fontsize=9)
ax.grid(axis='y', linestyle=':', alpha=0.3)

fig.tight_layout()
plt.savefig(f'{SAVE}/degradation_severity.png', dpi=300, bbox_inches='tight')
plt.close()
print('Saved: degradation_severity.png')

# =============================================================================
# Figure 7: Three-Paradigm Summary Table
# =============================================================================
fig, ax = plt.subplots(figsize=(11, 4))
ax.axis('off')

col_labels = ['Paradigm', 'Method', 'DINO Features', 'Best Precision', 'Verdict']
table_data = [
    ['Modify Features\n(Sec 3-5)', 'Projection Head\nLoRA + InfoNCE\nLoRA + Safe Radius\nStableMatchingLoss', 'Modified\n(damaged)', '3-5%\n5.5%\n28.8%\n11.0%', 'Catastrophic\ncollapse\nPartial recovery\nStill degraded'],
    ['Filter Patches\n(Sec 6)', 'Matchability\nPredictor (A)', 'Frozen\n(pristine)', '5-6%', 'Lacks context;\nfilters good patches'],
    ['Learn to Match\n(Sec 7)', 'Attention Matching\nHead (B)', 'Frozen\n(pristine)', '0.00%', 'Zero geometric\ndistinctiveness'],
]

table = ax.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(9.5)
table.scale(1, 2.2)

for j in range(5):
    table[(0, j)].set_facecolor('#2c3e50')
    table[(0, j)].set_text_props(color='white', fontweight='bold', fontsize=10)
for i in [1, 2, 3]:
    table[(i, 0)].set_facecolor('#ecf0f1')
    table[(i, 0)].set_text_props(fontweight='bold')
    for j in range(1, 5):
        if i == 1:
            table[(i, j)].set_facecolor('#fad4d4')
        elif i == 2:
            table[(i, j)].set_facecolor('#fff3cd')
        else:
            table[(i, j)].set_facecolor('#d4edda')

ax.set_title('Three Paradigms, Seven Experiments, Zero Successes', fontsize=14, fontweight='bold', pad=14)

fig.tight_layout()
plt.savefig(f'{SAVE}/three_paradigms.png', dpi=300, bbox_inches='tight')
plt.close()
print('Saved: three_paradigms.png')

# =============================================================================
# Figure 8: Semantic vs Geometric features conceptual diagram (scatter-style)
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

np.random.seed(42)

# Semantic feature space (DINOv3)
n_objects = 4
n_points_per_object = 30
colors_obj = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
centers = [(-3, 2), (3, 2.5), (-2, -3), (3, -2)]
for i, (cx, cy) in enumerate(centers):
    pts = np.random.randn(n_points_per_object, 2) * 0.3 + np.array([cx, cy])
    ax1.scatter(pts[:, 0], pts[:, 1], c=colors_obj[i], s=30, alpha=0.7, edgecolors='black', linewidth=0.3)
ax1.set_title('DINOv3 Semantic Features\n(Smooth, clustered by object category)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Dim 1'); ax1.set_ylabel('Dim 2')
ax1.set_xlim(-5, 5); ax1.set_ylim(-5, 5)
ax1.text(0.02, 0.98, '"door ≈ door, wall ≈ wall"\nCross-instance generalization', transform=ax1.transAxes,
         fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Geometric feature space (what matching needs)
for i in range(20):
    theta = i * 2*np.pi/20 + np.random.randn()*0.05
    r = 3 + np.random.randn()*0.15
    ax2.scatter([r*np.cos(theta)], [r*np.sin(theta)], c='#e74c3c', s=25, alpha=0.8, edgecolors='black', linewidth=0.3)
    ax2.scatter([r*np.cos(theta+0.15)], [r*np.sin(theta+0.15)], c='#3498db', s=25, alpha=0.8, edgecolors='black', linewidth=0.3)
    ax2.plot([r*np.cos(theta), r*np.cos(theta+0.15)], [r*np.sin(theta), r*np.sin(theta+0.15)],
             'gray', linewidth=0.5, alpha=0.4)
ax2.set_title('Geometric Features (needed for matching)\n(Sparse, distinct, view-covariant)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Dim 1'); ax2.set_ylabel('Dim 2')
ax2.set_xlim(-5, 5); ax2.set_ylim(-5, 5)
ax2.text(0.02, 0.98, '"this specific corner ≠ that specific corner"\nIntra-instance distinction', transform=ax2.transAxes,
         fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

fig.suptitle('Fundamental Mismatch: Semantic Features vs Geometric Features', fontsize=14, fontweight='bold', y=1.02)
fig.tight_layout()
plt.savefig(f'{SAVE}/semantic_vs_geometric.png', dpi=300, bbox_inches='tight')
plt.close()
print('Saved: semantic_vs_geometric.png')

# =============================================================================
# Figure 9: Information bottleneck illustration
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis('off')

# Draw concentric conceptual regions
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch, Arc

# Large outer circle: "DINOv3 Pretraining"
outer = Circle((5, 4), 3.5, fill=True, facecolor='#d5f5e3', edgecolor='#2ecc71', linewidth=3, alpha=0.6)
ax.add_patch(outer)
ax.text(5, 7.2, 'DINOv3 Pretraining (142M images)', ha='center', fontsize=13, fontweight='bold', color='#27ae60')

# Middle circle: "Semantic Features"
middle = Circle((5, 4), 2.5, fill=True, facecolor='#d6eaf8', edgecolor='#3498db', linewidth=2.5, alpha=0.7)
ax.add_patch(middle)
ax.text(5, 6.2, 'Semantic Features', ha='center', fontsize=12, fontweight='bold', color='#2980b9')

# Small inner circle: "Geometric Info"
inner = Circle((5, 4), 1.0, fill=True, facecolor='#fadbd8', edgecolor='#e74c3c', linewidth=2.5, alpha=0.9)
ax.add_patch(inner)
ax.text(5, 4, 'Geometric\nResidual', ha='center', fontsize=10, fontweight='bold', color='#c0392b')

# Annotation for "eliminated info"
ax.annotate('Geometric information\nELIMINATED during\npretraining', xy=(8.2, 2.5), fontsize=9, ha='center',
            bbox=dict(boxstyle='round', facecolor='#f9e79f', alpha=0.8),
            fontweight='bold', color='#c0392b')
ax.annotate('', xy=(7.2, 2.8), xytext=(5.6, 3.2),
            arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.5))

# Zero-Shot performance annotation
ax.annotate('Zero-Shot Precision:\nNAVI 49% / ScanNet 32%\n(uses residual geometric trace)', xy=(1.5, 1.5), fontsize=9, ha='center',
            bbox=dict(boxstyle='round', facecolor='#d5f5e3', alpha=0.8),
            fontweight='bold', color='#27ae60')

# Fine-tuning annotation
ax.annotate('Fine-tuning attempts\nto recover geometric info\n→ Impossible (information\nirreversibly lost)', xy=(8, 6), fontsize=9, ha='center',
            bbox=dict(boxstyle='round', facecolor='#fadbd8', alpha=0.8),
            fontweight='bold', color='#c0392b')

ax.set_title('Information-Theoretic View: Why Fine-Tuning Cannot Succeed', fontsize=14, fontweight='bold', pad=18)

fig.tight_layout()
plt.savefig(f'{SAVE}/information_bottleneck.png', dpi=300, bbox_inches='tight')
plt.close()
print('Saved: information_bottleneck.png')

# =============================================================================
# Figure 10: Complete results table as a styled matplotlib figure
# =============================================================================
fig, ax = plt.subplots(figsize=(14, 7))
ax.axis('off')

headers = ['#', 'Method', 'Dataset', 'AUC@5', 'AUC@10', 'AUC@20', 'Precision', 'vs ZS']
all_results = [
    ['0', 'Zero-Shot (baseline)',        'NAVI',    '0.24', '1.07', '3.31', '49.17%', '—'],
    ['0', 'Zero-Shot (baseline)',        'ScanNet', '0.23', '1.18', '4.44', '32.20%', '—'],
    ['1', 'Projection Head + InfoNCE',   'NAVI',    '0.00', '0.03', '0.10', '3.16%',  '↓93.6%'],
    ['1', 'Projection Head + InfoNCE',   'ScanNet', '0.00', '0.07', '0.25', '3.47%',  '↓89.2%'],
    ['2', 'LoRA + InfoNCE (no SafeR)',   'ScanNet', '0.00', '0.09', '0.96', '5.51%',  '↓82.9%'],
    ['3', 'LoRA + Safe Radius (bs=1)',   'ScanNet', '0.00', '0.26', '1.84', '28.83%', '↓10.5%'],
    ['4', 'LoRA + StableMatchingLoss',   'NAVI',    '0.00', '0.02', '0.11', '5.13%',  '↓89.6%'],
    ['4', 'LoRA + StableMatchingLoss',   'ScanNet', '0.04', '0.26', '1.43', '10.97%', '↓65.9%'],
    ['5', 'Matchability Predictor (A)',  'NAVI',    '0.00', '0.02', '0.17', '5.55%',  '↓88.7%'],
    ['5', 'Matchability Predictor (A)',  'ScanNet', '0.00', '0.00', '0.20', '5.95%',  '↓81.5%'],
    ['6', 'Matching Head (B)',           'NAVI',    '0.00', '0.00', '0.00', '0.00%',  '↓100%'],
    ['6', 'Matching Head (B)',           'ScanNet', '0.00', '0.00', '0.00', '0.00%',  '↓100%'],
]

all_data = [headers] + all_results
table = ax.table(cellText=all_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(8.5)
table.scale(1, 1.6)

# Style header
for j in range(len(headers)):
    table[(0, j)].set_facecolor('#2c3e50')
    table[(0, j)].set_text_props(color='white', fontweight='bold')

# Style rows: Zero-Shot green, others by degradation
for i in range(1, len(all_data)):
    prec_str = all_data[i-1][6]
    if 'Zero-Shot' in all_data[i-1][1]:
        bg = '#d5f5e3'
    elif '0.00%' in all_data[i-1][6] and '—' not in all_data[i-1][6]:
        bg = '#f5b7b1'
    elif '28.83%' in all_data[i-1][6]:
        bg = '#f9e79f'
    else:
        bg = '#fadbd8'
    for j in range(len(headers)):
        table[(i, j)].set_facecolor(bg)

ax.set_title('Complete Experimental Results: Seven Attempts Across Two Datasets',
             fontsize=14, fontweight='bold', pad=18)

fig.tight_layout()
plt.savefig(f'{SAVE}/complete_results_table.png', dpi=300, bbox_inches='tight')
plt.close()
print('Saved: complete_results_table.png')

print('\n=== All figures generated successfully in report/figures/ ===')
