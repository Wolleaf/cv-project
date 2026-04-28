#!/bin/bash
# =========================================================================
# 鲁棒微调方案：StableMatchingLoss + LoRA (RTX 5090 32GB)
# =========================================================================
# 核心改进（相比 run_all_5090.sh）：
#   1. 使用 StableMatchingLoss 替代 InfoNCE
#      - 不再强制 "赢家通吃" 的 softmax 竞争
#      - 改用 margin-based 软间隔：只要求正样本比最难负样本相似度高 margin
#   2. 更强的 LoRA 权重衰减 (5e-3) 保持特征接近预训练 DINO
#   3. Diversity 正则化防止所有特征坍缩到同一点
#   4. 更低的学习率 (5e-4) + 更长的训练 (20 epochs)
# =========================================================================

set -e

echo -e "\e[36m=================================================================\e[0m"
echo -e "\e[36m鲁棒微调全流程：NAVI & ScanNet (StableMatchingLoss + LoRA)\e[0m"
echo -e "\e[36m=================================================================\e[0m"

export OMP_NUM_THREADS=8
CONDA_ENV="llmdevelop"

echo -e "\n\e[35m[0/4] 环境自检...\e[0m"

if ! command -v conda &> /dev/null; then
    echo -e "\e[31m[错误] 未检测到 conda\e[0m"
    exit 1
fi

conda run -n $CONDA_ENV python -c "import torch; assert torch.cuda.is_available(), 'CUDA 不可用'" || exit 1

WEIGHTS_PATH="dinov3_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
if [ ! -f "$WEIGHTS_PATH" ]; then
    echo -e "\e[31m[错误] 缺失权重: $WEIGHTS_PATH\e[0m"
    exit 1
fi

REQUIRED_FILES=(
    "finetune/navi_train_pairs.txt"
    "datasets/navi_with_gt.txt"
    "datasets/scannet_with_gt.txt"
)
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo -e "\e[31m[错误] 缺失文件: $file\e[0m"
        exit 1
    fi
done

echo -e "\e[32m[自检通过]\e[0m"

# =========================================================================
# 第一阶段：NAVI
# =========================================================================
echo -e "\n\e[33m[1/4] NAVI 鲁棒微调 (StableMatchingLoss)...\e[0m"
conda run -n $CONDA_ENV python -m finetune.train_robust \
    --checkpoint "$WEIGHTS_PATH" \
    --train_pairs finetune/navi_train_pairs.txt \
    --data_root full_dataset/navi_v1.5 \
    --depth_root full_dataset/navi_v1.5 \
    --output_dir finetune_output_robust_navi \
    --epochs 20 --batch_size 8 --img_size 448 \
    --lora_rank 4 --lr 5e-4 \
    --pos_weight 2.0 --neg_weight 1.0 \
    --diversity_weight 0.1 --margin 0.3 --safe_radius 5.0 \
    --weight_decay 5e-3

echo -e "\n\e[33m[2/4] NAVI 提取特征 + MNN 匹配...\e[0m"
conda run -n $CONDA_ENV python -m finetune.extract_lora \
    --checkpoint finetune_output_robust_navi/checkpoint_latest.pth \
    --pretrained "$WEIGHTS_PATH" \
    --pairs datasets/navi_with_gt.txt \
    --data_root datasets/test/navi_resized \
    --output_dir mnn_matching_robust_navi/navi \
    --img_size 448 --eval_resize 640 480

echo -e "\n\e[32m>> 评测 NAVI...\e[0m"
conda run -n $CONDA_ENV python evaluate/evaluate_csv_essential.py \
    --input_pairs datasets/navi_with_gt.txt \
    --input_dir datasets/test/navi_resized \
    --input_csv_dir mnn_matching_robust_navi/navi \
    --output_dir evaluate/navi_robust

# =========================================================================
# 第二阶段：ScanNet
# =========================================================================
echo -e "\n\e[33m[3/4] ScanNet 鲁棒微调 (StableMatchingLoss)...\e[0m"
conda run -n $CONDA_ENV python -m finetune.train_robust \
    --checkpoint "$WEIGHTS_PATH" \
    --train_pairs datasets/scannet_with_gt.txt \
    --data_root datasets/test \
    --output_dir finetune_output_robust_scannet \
    --epochs 20 --batch_size 8 --img_size 448 \
    --lora_rank 4 --lr 5e-4 \
    --pos_weight 2.0 --neg_weight 1.0 \
    --diversity_weight 0.1 --margin 0.3 --safe_radius 5.0 \
    --weight_decay 5e-3

echo -e "\n\e[33m[4/4] ScanNet 提取特征 + MNN 匹配...\e[0m"
conda run -n $CONDA_ENV python -m finetune.extract_lora \
    --checkpoint finetune_output_robust_scannet/checkpoint_latest.pth \
    --pretrained "$WEIGHTS_PATH" \
    --pairs datasets/scannet_with_gt.txt \
    --data_root datasets/test \
    --output_dir mnn_matching_robust_scannet/scannet \
    --img_size 448 --eval_resize 640 480

echo -e "\n\e[32m>> 评测 ScanNet...\e[0m"
conda run -n $CONDA_ENV python evaluate/evaluate_csv_essential.py \
    --input_pairs datasets/scannet_with_gt.txt \
    --input_dir datasets/test \
    --input_csv_dir mnn_matching_robust_scannet/scannet \
    --output_dir evaluate/scannet_robust

echo -e "\n\e[36m=================================================================\e[0m"
echo -e "\e[36m全流程完成！查看结果：\e[0m"
echo -e "\e[36m  evaluate/navi_robust/evaluation_results.txt\e[0m"
echo -e "\e[36m  evaluate/scannet_robust/evaluation_results.txt\e[0m"
echo -e "\e[36m=================================================================\e[0m"
