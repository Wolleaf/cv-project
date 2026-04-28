#!/bin/bash
# =========================================================================
# 自动化脚本：DINOv3 LoRA 跨图对比微调与评测 (RTX 5090 Linux 专版)
# =========================================================================
# 说明：
# 本脚本旨在充分利用大显存（如 RTX 5090 的 32GB），通过开启 Batch Size = 8，
# 激活真正的 Inter-Image Negatives（跨图像负样本采样），从而彻底解决单图语义坍塌问题。
# =========================================================================

set -e

echo -e "\e[36m=================================================================\e[0m"
echo -e "\e[36m开始全自动流程：NAVI & ScanNet 的 LoRA 微调与评测 (Inter-Image)\e[0m"
echo -e "\e[36m=================================================================\e[0m"

# 修复 libgomp 警告
# 修复 libgomp 警告（AutoDL 环境下设为 8 或 16 既能避免崩溃又能保持多线程性能）
export OMP_NUM_THREADS=8

# 确保在对应的 conda 环境中执行
# 如果环境名称不同，请自行修改 llmdevelop
CONDA_ENV="llmdevelop"

echo -e "\n\e[35m[0/4] 正在执行环境与依赖严格自检...\e[0m"

# 1. 检查 conda 是否安装
if ! command -v conda &> /dev/null; then
    echo -e "\e[31m[错误] 未检测到 conda，请确保 Miniconda/Anaconda 已安装并加入环境变量。\e[0m"
    exit 1
fi

# 2. 检查对应的 conda 环境是否存在
if ! conda info --envs | grep -q "^$CONDA_ENV\b"; then
    echo -e "\e[31m[错误] Conda 环境 '$CONDA_ENV' 不存在，请先创建环境并安装依赖。\e[0m"
    exit 1
fi

# 3. 检查必备库 (torch, cv2, numpy)
echo "检查 Python 依赖包..."
conda run -n $CONDA_ENV python -c "import torch, cv2, numpy; print(f'PyTorch {torch.__version__} 检查通过.')" || {
    echo -e "\e[31m[错误] 缺少必要的 Python 库 (torch, opencv-python, numpy)！\e[0m"
    exit 1
}

# 4. 检查 CUDA 是否可用
conda run -n $CONDA_ENV python -c "import torch; assert torch.cuda.is_available(), 'CUDA 不可用，请检查驱动！'" || {
    echo -e "\e[31m[错误] 当前环境无法使用 GPU，训练将被拒绝。\e[0m"
    exit 1
}

# 5. 检查 DINOv3 预训练权重是否存在
WEIGHTS_PATH="dinov3_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
if [ ! -f "$WEIGHTS_PATH" ]; then
    echo -e "\e[31m[错误] 缺失 DINOv3 预训练权重文件: $WEIGHTS_PATH\e[0m"
    exit 1
fi

# 6. 检查数据集目录和文件
REQUIRED_FILES=(
    "finetune/navi_train_pairs.txt"
    "datasets/navi_with_gt.txt"
    "datasets/scannet_with_gt.txt"
)
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo -e "\e[31m[错误] 缺失必须的数据集文件: $file\e[0m"
        exit 1
    fi
done

REQUIRED_DIRS=(
    "full_dataset/navi_v1.5"
    "datasets/test/navi_resized"
    "datasets/test/scans_test"
)
for dir in "${REQUIRED_DIRS[@]}"; do
    if [ ! -d "$dir" ]; then
        echo -e "\e[31m[错误] 缺失必须的数据集目录: $dir\e[0m"
        exit 1
    fi
done

echo -e "\e[32m[自检通过] 所有环境、依赖及数据均已就绪！\e[0m"

# -------------------------------------------------------------------------
# 第一阶段：NAVI 数据集
# -------------------------------------------------------------------------
echo -e "\n\e[33m[1/4] (已完成) 跳过 NAVI 数据集微调 LoRA...\e[0m"
# conda run -n $CONDA_ENV python -m finetune.train_lora \
#     --checkpoint dinov3_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
#     --train_pairs finetune/navi_train_pairs.txt \
#     --data_root full_dataset/navi_v1.5 \
#     --depth_root full_dataset/navi_v1.5 \
#     --output_dir finetune_output_lora_navi_5090 \
#     --epochs 15 --batch_size 8 --img_size 448 --lora_rank 4 --lr 1e-3

echo -e "\n\e[33m[2/4] 正在提取 NAVI 验证集特征并使用 MNN 进行匹配...\e[0m"
conda run -n $CONDA_ENV python -m finetune.extract_lora \
    --checkpoint finetune_output_lora_navi_5090/checkpoint_latest.pth \
    --pretrained dinov3_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
    --pairs datasets/navi_with_gt.txt \
    --data_root datasets/test/navi_resized \
    --output_dir mnn_matching_lora_navi_5090/navi \
    --img_size 448 --eval_resize 640 480

echo -e "\n\e[32m>> 评测 NAVI 最终结果...\e[0m"
conda run -n $CONDA_ENV python evaluate/evaluate_csv_essential.py \
    --input_pairs datasets/navi_with_gt.txt \
    --input_dir datasets/test/navi_resized \
    --input_csv_dir mnn_matching_lora_navi_5090/navi \
    --output_dir evaluate/navi_lora_5090

# -------------------------------------------------------------------------
# 第二阶段：ScanNet 数据集
# -------------------------------------------------------------------------
echo -e "\n\e[33m[3/4] 正在使用 Batch Size 8 在 ScanNet 数据集上微调 LoRA...\e[0m"
conda run -n $CONDA_ENV python -m finetune.train_lora \
    --checkpoint dinov3_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
    --train_pairs datasets/scannet_with_gt.txt \
    --data_root datasets/test \
    --output_dir finetune_output_lora_scannet_5090 \
    --epochs 15 --batch_size 8 --img_size 448 --lora_rank 4 --lr 1e-3

echo -e "\n\e[33m[4/4] 正在提取 ScanNet 验证集特征并使用 MNN 进行匹配...\e[0m"
conda run -n $CONDA_ENV python -m finetune.extract_lora \
    --checkpoint finetune_output_lora_scannet_5090/checkpoint_latest.pth \
    --pretrained dinov3_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
    --pairs datasets/scannet_with_gt.txt \
    --data_root datasets/test \
    --output_dir mnn_matching_lora_scannet_5090/scannet \
    --img_size 448 --eval_resize 640 480

echo -e "\n\e[32m>> 评测 ScanNet 最终结果...\e[0m"
conda run -n $CONDA_ENV python evaluate/evaluate_csv_essential.py \
    --input_pairs datasets/scannet_with_gt.txt \
    --input_dir datasets/test \
    --input_csv_dir mnn_matching_lora_scannet_5090/scannet \
    --output_dir evaluate/scannet_lora_5090

echo -e "\n\e[36m=================================================================\e[0m"
echo -e "\e[36m所有流程执行完毕！请查看 evaluate 文件夹下的最终定量结果报告。\e[0m"
echo -e "\e[36m=================================================================\e[0m"
