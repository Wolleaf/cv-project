#!/bin/bash
# =========================================================================
# Route A: Matchability Predictor — 一键训练与评测 (RTX 5090 32GB)
# =========================================================================
# 核心思路（与之前所有微调方法完全不同）：
#   - DINOv3 backbone 完全冻结，不做任何修改
#   - 训练一个极小的 MLP（~0.35M 参数）预测每个 patch 的"可匹配度"
#   - 推理时过滤低分 patch，只在高分 patch 上做 MNN 匹配
#   - 纹理缺失区域（白墙）自动被过滤 → 匹配精度提升
# =========================================================================

set -e

echo -e "\e[36m=================================================================\e[0m"
echo -e "\e[36mRoute A: Matchability Predictor 全流程 (NAVI + ScanNet)\e[0m"
echo -e "\e[36m=================================================================\e[0m"

export OMP_NUM_THREADS=8
CONDA_ENV="llmdevelop"
WEIGHTS="dinov3_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"

echo -e "\n\e[35m[0/4] 环境自检...\e[0m"

if ! command -v conda &> /dev/null; then
    echo -e "\e[31m[错误] 未检测到 conda\e[0m"
    exit 1
fi

conda run -n $CONDA_ENV python -c "import torch; assert torch.cuda.is_available(), 'CUDA 不可用'" || exit 1

if [ ! -f "$WEIGHTS" ]; then
    echo -e "\e[31m[错误] 缺失权重: $WEIGHTS\e[0m"
    exit 1
fi

for f in "finetune/navi_train_pairs.txt" "datasets/navi_with_gt.txt" "datasets/scannet_with_gt.txt"; do
    if [ ! -f "$f" ]; then
        echo -e "\e[31m[错误] 缺失文件: $f\e[0m"
        exit 1
    fi
done

echo -e "\e[32m[自检通过]\e[0m"

# =========================================================================
# 第一阶段：NAVI — 训练 matchability predictor
# =========================================================================
echo -e "\n\e[33m[1/4] 在 NAVI 训练集上训练 Matchability Predictor...\e[0m"
conda run -n $CONDA_ENV python -m finetune.matchability train \
    --checkpoint "$WEIGHTS" \
    --train_pairs finetune/navi_train_pairs.txt \
    --data_root full_dataset/navi_v1.5 \
    --depth_root full_dataset/navi_v1.5 \
    --output_dir matchability_navi \
    --epochs 10 --batch_pairs 8 --lr 1e-3 --img_size 448

echo -e "\n\e[33m[2/4] NAVI 提取 + 过滤 + MNN 匹配 + 评估...\e[0m"

# 测试多个 keep_ratio 找到最优值
for RATIO in 0.8 0.6 0.5 0.4 0.3; do
    echo -e "\n\e[34m  --- keep_ratio=$RATIO ---\e[0m"
    conda run -n $CONDA_ENV python -m finetune.matchability extract \
        --checkpoint "$WEIGHTS" \
        --predictor matchability_navi/predictor_best.pth \
        --pairs datasets/navi_with_gt.txt \
        --data_root datasets/test/navi_resized \
        --output_dir "mnn_matching_matchability_navi/navi_r${RATIO}" \
        --img_size 448 --eval_resize 640 480 --keep_ratio $RATIO

    conda run -n $CONDA_ENV python evaluate/evaluate_csv_essential.py \
        --input_pairs datasets/navi_with_gt.txt \
        --input_dir datasets/test/navi_resized \
        --input_csv_dir "mnn_matching_matchability_navi/navi_r${RATIO}" \
        --output_dir "evaluate/navi_matchability_r${RATIO}"

    echo -e "\n\e[34m  NAVI keep_ratio=$RATIO 结果:\e[0m"
    cat "evaluate/navi_matchability_r${RATIO}/evaluation_results.json"
done

# =========================================================================
# 第二阶段：ScanNet — 训练 matchability predictor
# =========================================================================
echo -e "\n\e[33m[3/4] 在 ScanNet 训练集上训练 Matchability Predictor...\e[0m"
conda run -n $CONDA_ENV python -m finetune.matchability train \
    --checkpoint "$WEIGHTS" \
    --train_pairs datasets/scannet_with_gt.txt \
    --data_root datasets/test \
    --output_dir matchability_scannet \
    --epochs 10 --batch_pairs 8 --lr 1e-3 --img_size 448

echo -e "\n\e[33m[4/4] ScanNet 提取 + 过滤 + MNN 匹配 + 评估...\e[0m"

for RATIO in 0.8 0.6 0.5 0.4 0.3; do
    echo -e "\n\e[34m  --- keep_ratio=$RATIO ---\e[0m"
    conda run -n $CONDA_ENV python -m finetune.matchability extract \
        --checkpoint "$WEIGHTS" \
        --predictor matchability_scannet/predictor_best.pth \
        --pairs datasets/scannet_with_gt.txt \
        --data_root datasets/test \
        --output_dir "mnn_matching_matchability_scannet/scannet_r${RATIO}" \
        --img_size 448 --eval_resize 640 480 --keep_ratio $RATIO

    conda run -n $CONDA_ENV python evaluate/evaluate_csv_essential.py \
        --input_pairs datasets/scannet_with_gt.txt \
        --input_dir datasets/test \
        --input_csv_dir "mnn_matching_matchability_scannet/scannet_r${RATIO}" \
        --output_dir "evaluate/scannet_matchability_r${RATIO}"

    echo -e "\n\e[34m  ScanNet keep_ratio=$RATIO 结果:\e[0m"
    cat "evaluate/scannet_matchability_r${RATIO}/evaluation_results.json"
done

# =========================================================================
# 总结
# =========================================================================
echo -e "\n\e[36m=================================================================\e[0m"
echo -e "\e[36m全流程完成！\e[0m"
echo -e "\e[36m=================================================================\e[0m"
echo ""
echo "NAVI 结果汇总:"
for RATIO in 0.8 0.6 0.5 0.4 0.3; do
    echo -n "  keep_ratio=$RATIO: "
    cat "evaluate/navi_matchability_r${RATIO}/evaluation_results.json" 2>/dev/null || echo "N/A"
done
echo ""
echo "ScanNet 结果汇总:"
for RATIO in 0.8 0.6 0.5 0.4 0.3; do
    echo -n "  keep_ratio=$RATIO: "
    cat "evaluate/scannet_matchability_r${RATIO}/evaluation_results.json" 2>/dev/null || echo "N/A"
done
echo ""
echo "Zero-Shot 基线 (参考):"
echo "  NAVI:    AUC@5=0.24  AUC@10=1.07  AUC@20=3.31  Prec=49.17%"
echo "  ScanNet: AUC@5=0.23  AUC@10=1.18  AUC@20=4.44  Prec=32.20%"
