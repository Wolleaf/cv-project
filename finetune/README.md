# DINOv3 微调 — 完整操作流程

> **你的环境**: Windows 11 / RTX 5060 (8GB) / conda `llmdevelop` 环境  
> **已验证**: PyTorch 2.9.1+cu130, CUDA 13.0, OpenCV 4.13, torchvision 0.24.1  
> **模型加载 + 前向传递仅需 1.24 GB 显存** ✅

---

## 📋 总体流程概览

```
步骤 1  检查环境              ← 已完成 ✅
步骤 2  理解项目结构           ← 已完成 ✅  
步骤 3  准备训练数据           ← 你需要做
步骤 4  测试训练 (小样本)      ← 你需要运行命令
步骤 5  正式训练              ← 你需要运行命令
步骤 6  提取特征 + 匹配        ← 你需要运行命令
步骤 7  评估 (对比 zero-shot)  ← 你需要运行命令
```

---

## 步骤 1：环境检查 ✅ 已完成

你的环境已验证可用，不需要安装额外依赖。

---

## 步骤 2：项目结构 ✅ 已完成

```
lmz/
├── dinov3_weights/
│   └── dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth  (1.2GB, 预训练权重)
├── datasets/
│   ├── test/                  ← 测试图片 (lmz 已提供)
│   ├── navi_with_gt.txt       ← NAVI 3000对测试 pairs
│   └── scannet_with_gt.txt    ← ScanNet 1500对测试 pairs
├── evaluate/                  ← lmz 的评估脚本
├── mnn_matching/              ← lmz 的 zero-shot 匹配结果
│
├── finetune/                  ← 【新增】你的微调代码
│   ├── model.py               ← DINOv3 模型定义
│   ├── dataset.py             ← 数据集加载
│   ├── loss.py                ← 对比学习损失
│   ├── train.py               ← 训练脚本
│   ├── extract_and_match.py   ← 特征提取+匹配
│   └── config.py              ← 配置参数
│
├── finetune_output/           ← 训练输出 (自动创建)
├── mnn_matching_finetuned/    ← 微调后的匹配结果 (自动创建)
└── evaluate/
    ├── navi/                  ← lmz 的 zero-shot 评估结果
    └── scannet/               ← lmz 的 zero-shot 评估结果
```

---

## 步骤 3：准备训练数据 ⚠️ 你需要做

### 情况 A：使用现有测试数据训练（推荐先试）

lmz 提供的测试数据已经包含图片和带 GT pose 的 pairs 文件，可以直接用来训练（虽然不是理想的训练/测试分割，但足以验证流程和看到提升）。

**检查你的数据是否就绪**：

在 Anaconda Prompt 中运行：

```powershell
conda activate llmdevelop
cd "D:\Admin\Documents\港中深学习\SecondTerm\Image Processing and Computer Vision\lmz"

# 检查 NAVI 测试图片
python -c "from pathlib import Path; imgs = list(Path('datasets/test').rglob('*.jpg')); print(f'Found {len(imgs)} images')"

# 检查 pairs 文件行数
python -c "print('NAVI pairs:', sum(1 for _ in open('datasets/navi_with_gt.txt'))); print('ScanNet pairs:', sum(1 for _ in open('datasets/scannet_with_gt.txt')))"
```

如果图片数量 > 0 且 pairs 行数正确（NAVI=3000, ScanNet=1500），就可以直接进入步骤 4。

### 情况 B：下载完整训练数据（可选，效果更好）

如果想用更多数据训练，需要下载完整数据集：

```
NAVI:    https://github.com/google/navi.git
ScanNet: https://github.com/ScanNet/ScanNet.git
```

下载后放在 `datasets/` 下，并准备对应的 pairs 文件。

---

## 步骤 4：测试训练（小样本验证） 🔥 重要

**目的**：用少量数据跑 2-3 个 epoch，确认整个训练流程没问题。

打开 **Anaconda Prompt**，运行：

```powershell
conda activate llmdevelop
cd "D:\Admin\Documents\港中深学习\SecondTerm\Image Processing and Computer Vision\lmz"

python -m finetune.train `
    --checkpoint dinov3_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth `
    --train_pairs datasets/scannet_with_gt.txt `
    --data_root datasets/test `
    --output_dir finetune_output_test `
    --epochs 2 `
    --batch_size 1 `
    --freeze_blocks 22 `
    --img_size 448
```

**你应该看到**：
- `[model] Loaded pre-trained weights...`
- `[model] Trainable: 25.2M / 303.2M params (8.3%)`
- 每 10 步打印一次 loss（应该逐渐下降）
- 2 个 epoch 大约 5-15 分钟

**如果显存不足（OOM）**，改为：
```powershell
python -m finetune.train `
    --checkpoint dinov3_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth `
    --train_pairs datasets/scannet_with_gt.txt `
    --data_root datasets/test `
    --output_dir finetune_output_test `
    --epochs 2 `
    --batch_size 1 `
    --freeze_blocks 23 `
    --img_size 336
```
（freeze_blocks=23 表示只训练最后 1 层 + projection head，img_size 更小）

---

## 步骤 5：正式训练 🚀

测试通过后，用完整数据跑 15 个 epoch：

```powershell
conda activate llmdevelop
cd "D:\Admin\Documents\港中深学习\SecondTerm\Image Processing and Computer Vision\lmz"

# 用 ScanNet pairs 训练
python -m finetune.train `
    --checkpoint dinov3_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth `
    --train_pairs datasets/scannet_with_gt.txt `
    --data_root datasets/test `
    --output_dir finetune_output `
    --epochs 15 `
    --batch_size 1 `
    --freeze_blocks 22 `
    --img_size 448
```

**训练时间估计**：
- ScanNet 1500对 × 15 epochs ≈ 1-3 小时（取决于实际有效对数量）
- 训练期间可以看 `finetune_output/training_log.json` 观察 loss 变化

**输出文件**：
```
finetune_output/
├── config.json                  ← 训练参数记录
├── training_log.json            ← 每 epoch 的 loss 记录
├── checkpoint_epoch004.pth      ← 第 5 轮检查点
├── checkpoint_epoch009.pth      ← 第 10 轮检查点  
├── checkpoint_epoch014.pth      ← 第 15 轮检查点
└── checkpoint_latest.pth        ← 最新检查点
```

**如果训练中断**，可以恢复：
```powershell
python -m finetune.train `
    --checkpoint dinov3_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth `
    --train_pairs datasets/scannet_with_gt.txt `
    --data_root datasets/test `
    --output_dir finetune_output `
    --epochs 15 `
    --resume finetune_output/checkpoint_latest.pth
```

---

## 步骤 6：提取特征 + MNN 匹配

训练完成后，用微调模型对两个测试数据集分别提取特征并匹配：

```powershell
conda activate llmdevelop
cd "D:\Admin\Documents\港中深学习\SecondTerm\Image Processing and Computer Vision\lmz"

# 6a. NAVI 数据集
python -m finetune.extract_and_match `
    --checkpoint finetune_output/checkpoint_latest.pth `
    --pairs datasets/navi_with_gt.txt `
    --data_root datasets/test/navi_resized `
    --output_dir mnn_matching_finetuned/navi `
    --img_size 448

# 6b. ScanNet 数据集
python -m finetune.extract_and_match `
    --checkpoint finetune_output/checkpoint_latest.pth `
    --pairs datasets/scannet_with_gt.txt `
    --data_root datasets/test `
    --output_dir mnn_matching_finetuned/scannet `
    --img_size 448
```

**输出**：每个图像对生成一个 CSV 文件，格式与 lmz 的 zero-shot 结果完全一致。

---

## 步骤 7：评估 + 对比

使用 lmz 提供的评估脚本对比 zero-shot vs fine-tuned：

```powershell
# 注意：评估脚本依赖 Superglue 工具函数
# 如果你有 Superglue 的代码，运行以下命令：

# 7a. 评估 NAVI
python evaluate/evaluate_csv_essential.py `
    --input_pairs datasets/navi_with_gt.txt `
    --input_dir datasets/test/navi_resized `
    --input_csv_dir mnn_matching_finetuned/navi `
    --output_dir evaluate/navi_finetuned

# 7b. 评估 ScanNet
python evaluate/evaluate_csv_essential.py `
    --input_pairs datasets/scannet_with_gt.txt `
    --input_csv_dir mnn_matching_finetuned/scannet `
    --output_dir evaluate/scannet_finetuned
```

> ⚠️ 提示：评估脚本已经过修改，可以直接使用你放在项目根目录下的 `Superglue` 官方代码库，无需原先的 `_paths.py` 文件。

---

## 📊 预期结果对比

| 指标 | NAVI (zero-shot) | ScanNet (zero-shot) | 微调后预期 |
|------|:---:|:---:|:---:|
| AUC@5 | 0.24 | 0.23 | **1-5+** |
| AUC@10 | 1.07 | 1.18 | **5-15+** |
| AUC@20 | 3.31 | 4.44 | **15-30+** |
| Precision | 49.17 | 32.20 | **60-80+** |

---

## 🔧 常见问题

### Q1: 显存不足 (CUDA out of memory)
**解决**: 减小参数
- `--img_size 336` (从 448 降到 336)
- `--freeze_blocks 23` (只训最后 1 层)

### Q2: 训练 loss 不下降
**可能原因**: 对应关系太少（图片路径不匹配）
**检查**: 看训练日志中 `valid_pairs` 数量是否 > 0

### Q3: 评估脚本报错找不到 Superglue
**解决**: 需要 lmz 提供 `Superglue/` 和 `third_party/` 目录，或单独下载 SuperGlue

### Q4: 想用 NAVI 数据训练
```powershell
python -m finetune.train `
    --train_pairs datasets/navi_with_gt.txt `
    --data_root datasets/test `
    ...其他参数同上
```

---

## 📁 你需要提交的文件

给报告/PPT 用：

1. `finetune/` — 完整微调代码
2. `finetune_output/training_log.json` — 训练 loss 曲线数据
3. `evaluate/navi_finetuned/evaluation_results.json` — NAVI 微调后评估结果
4. `evaluate/scannet_finetuned/evaluation_results.json` — ScanNet 微调后评估结果
5. 对比表格（zero-shot vs fine-tuned）
