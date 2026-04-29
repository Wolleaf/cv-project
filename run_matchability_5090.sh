#!/bin/bash
# =========================================================================
# Route A: Matchability Predictor — 并行训练与评测 (RTX 5090 32GB)
# =========================================================================
# 优化：NAVI 和 ScanNet 的 predictor 并行训练，充分利用 GPU。
# 每个 predictor 仅 ~280K 参数，GPU 完全有能力同时训练两个。
#
# 流程：
#   Phase 1 [并行] → NAVI 训练 + ScanNet 训练
#   Phase 2 [并行] → NAVI 提取评估 + ScanNet 提取评估
#   Phase 3         → 结果汇总
# =========================================================================

set -e

# -------- 阶段控制 --------
SKIP_TRAIN=false
for arg in "$@"; do
    case "$arg" in
        --skip-train) SKIP_TRAIN=true ;;
        --train-only) TRAIN_ONLY=true ;;
        --eval-only) EVAL_ONLY=true ;;
    esac
done

# -------- 配置 --------
export OMP_NUM_THREADS=8
CONDA_ENV="llmdevelop"
WEIGHTS="dinov3_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
RATIOS=(0.8 0.6 0.5 0.4 0.3)
TRAIN_EPOCHS=10
TRAIN_BATCH=8

# -------- 颜色 --------
RED='\e[31m'
GRN='\e[32m'
YLW='\e[33m'
CYN='\e[36m'
MAG='\e[35m'
RST='\e[0m'

# -------- 日志文件（避免并行输出混乱） --------
LOG_DIR="./logs_matchability"
mkdir -p "$LOG_DIR"

echo -e "${CYN}=================================================================${RST}"
echo -e "${CYN}Route A: Matchability Predictor 并行全流程 (NAVI + ScanNet)${RST}"
echo -e "${CYN}=================================================================${RST}"

# =========================================================================
# Phase 0: 环境自检
# =========================================================================
echo -e "\n${MAG}[0/3] 环境自检...${RST}"

if ! command -v conda &> /dev/null; then
    echo -e "${RED}[错误] 未检测到 conda${RST}"
    exit 1
fi

conda run -n $CONDA_ENV python -c "
import torch
assert torch.cuda.is_available(), 'CUDA 不可用'
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
" || exit 1

if [ ! -f "$WEIGHTS" ]; then
    echo -e "${RED}[错误] 缺失权重: $WEIGHTS${RST}"
    exit 1
fi

for f in "finetune/navi_train_pairs.txt" "datasets/navi_with_gt.txt" "datasets/scannet_with_gt.txt"; do
    if [ ! -f "$f" ]; then
        echo -e "${RED}[错误] 缺失文件: $f${RST}"
        exit 1
    fi
done

echo -e "${GRN}[自检通过]${RST}"

# =========================================================================
# Phase 1: 并行训练 (NAVI + ScanNet)
# =========================================================================
if [ "$SKIP_TRAIN" = true ]; then
    echo -e "\n${YLW}[1/3] 跳过训练 — 使用已有模型${RST}"
    if [ ! -f "matchability_navi/predictor_best.pth" ]; then
        echo -e "${RED}[错误] 缺失: matchability_navi/predictor_best.pth${RST}"
        exit 1
    fi
    if [ ! -f "matchability_scannet/predictor_best.pth" ]; then
        echo -e "${RED}[错误] 缺失: matchability_scannet/predictor_best.pth${RST}"
        exit 1
    fi
    echo -e "${GRN}模型文件检查通过${RST}"
else
    echo -e "\n${YLW}[1/3] 并行训练 NAVI 和 ScanNet 的 Matchability Predictor...${RST}"
    echo -e "       日志: ${LOG_DIR}/train_navi.log  +  ${LOG_DIR}/train_scannet.log"
    echo -e "       Predictor 仅 ~280K 参数，双任务并行 GPU 绰绰有余"

    TRAIN_START=$(date +%s)

    # --- NAVI (后台) ---
    conda run -n $CONDA_ENV python -m finetune.matchability train \
        --checkpoint "$WEIGHTS" \
        --train_pairs finetune/navi_train_pairs.txt \
        --data_root full_dataset/navi_v1.5 \
        --depth_root full_dataset/navi_v1.5 \
        --output_dir matchability_navi \
        --epochs $TRAIN_EPOCHS --batch_pairs $TRAIN_BATCH --lr 1e-3 --img_size 448 \
        > "$LOG_DIR/train_navi.log" 2>&1 &
    PID_NAVI=$!

    # --- ScanNet (后台) ---
    conda run -n $CONDA_ENV python -m finetune.matchability train \
        --checkpoint "$WEIGHTS" \
        --train_pairs datasets/scannet_with_gt.txt \
        --data_root datasets/test \
        --output_dir matchability_scannet \
        --epochs $TRAIN_EPOCHS --batch_pairs $TRAIN_BATCH --lr 1e-3 --img_size 448 \
        > "$LOG_DIR/train_scannet.log" 2>&1 &
    PID_SCANNET=$!

    # 实时显示进度
    echo -e "\n  等待两个训练任务完成..."
    while kill -0 $PID_NAVI 2>/dev/null || kill -0 $PID_SCANNET 2>/dev/null; do
        N_DONE=""
        S_DONE=""
        kill -0 $PID_NAVI 2>/dev/null && N_DONE="训练中" || N_DONE="${GRN}完成${RST}"
        kill -0 $PID_SCANNET 2>/dev/null && S_DONE="训练中" || S_DONE="${GRN}完成${RST}"
        echo -ne "\r  NAVI: $N_DONE  |  ScanNet: $S_DONE  "
        sleep 2
    done
    echo ""

    # 检查是否有失败
    wait $PID_NAVI; NAVI_EXIT=$?
    wait $PID_SCANNET; SCANNET_EXIT=$?

    TRAIN_END=$(date +%s)
    TRAIN_ELAPSED=$((TRAIN_END - TRAIN_START))

    echo -e "\n${GRN}训练完成！${RST} 耗时 ${TRAIN_ELAPSED}s (并行执行)"

    if [ $NAVI_EXIT -ne 0 ]; then
        echo -e "${RED}[错误] NAVI 训练失败，查看日志: ${LOG_DIR}/train_navi.log${RST}"
        tail -20 "$LOG_DIR/train_navi.log"
        exit 1
    fi
    if [ $SCANNET_EXIT -ne 0 ]; then
        echo -e "${RED}[错误] ScanNet 训练失败，查看日志: ${LOG_DIR}/train_scannet.log${RST}"
        tail -20 "$LOG_DIR/train_scannet.log"
        exit 1
    fi

    # 快速展示训练结果
    echo ""
    echo "NAVI 训练最终 loss:"
    grep "Epoch" "$LOG_DIR/train_navi.log" | tail -3
    echo ""
    echo "ScanNet 训练最终 loss:"
    grep "Epoch" "$LOG_DIR/train_scannet.log" | tail -3
fi

# =========================================================================
# Phase 2: 并行提取 + 评估 (NAVI + ScanNet 同时跑)
# =========================================================================
if [ "$SKIP_TRAIN" = true ]; then
    TRAIN_START=$(date +%s)  # for total time tracking in Phase 3
fi

echo -e "\n${YLW}[2/3] 并行提取特征 + 评估 (测试 5 个 keep_ratio)...${RST}"

EXTRACT_START=$(date +%s)

# 每个数据集跑 5 个 ratio 的函数
run_dataset() {
    local NAME=$1
    local PAIRS=$2
    local DATA=$3
    local PREDICTOR=$4
    local OUT_PREFIX=$5
    local EVAL_PREFIX=$6
    local LOGF="$LOG_DIR/extract_${NAME}.log"

    echo "[$NAME] 开始提取与评估..." > "$LOGF"

    for RATIO in "${RATIOS[@]}"; do
        echo "[$NAME] keep_ratio=$RATIO 提取中..." >> "$LOGF"
        conda run -n $CONDA_ENV python -m finetune.matchability extract \
            --checkpoint "$WEIGHTS" \
            --predictor "$PREDICTOR" \
            --pairs "$PAIRS" \
            --data_root "$DATA" \
            --output_dir "${OUT_PREFIX}_r${RATIO}" \
            --img_size 448 --eval_resize 640 480 --keep_ratio $RATIO \
            >> "$LOGF" 2>&1

        echo "[$NAME] keep_ratio=$RATIO 评估中..." >> "$LOGF"
        conda run -n $CONDA_ENV python evaluate/evaluate_csv_essential.py \
            --input_pairs "$PAIRS" \
            --input_dir "$DATA" \
            --input_csv_dir "${OUT_PREFIX}_r${RATIO}" \
            --output_dir "${EVAL_PREFIX}_r${RATIO}" \
            >> "$LOGF" 2>&1

        echo "[$NAME] keep_ratio=$RATIO 完成" >> "$LOGF"
    done

    echo "[$NAME] 全部完成" >> "$LOGF"
}

# 并行运行两个数据集
run_dataset "NAVI" \
    "datasets/navi_with_gt.txt" \
    "datasets/test/navi_resized" \
    "matchability_navi/predictor_best.pth" \
    "mnn_matching_matchability_navi/navi" \
    "evaluate/navi_matchability" &
PID_NAVI_EXTRACT=$!

run_dataset "ScanNet" \
    "datasets/scannet_with_gt.txt" \
    "datasets/test" \
    "matchability_scannet/predictor_best.pth" \
    "mnn_matching_matchability_scannet/scannet" \
    "evaluate/scannet_matchability" &
PID_SCANNET_EXTRACT=$!

echo -e "\n  等待提取与评估完成..."
while kill -0 $PID_NAVI_EXTRACT 2>/dev/null || kill -0 $PID_SCANNET_EXTRACT 2>/dev/null; do
    N_PROG=$(grep -c "完成" "$LOG_DIR/extract_NAVI.log" 2>/dev/null | tr -d '\n' || echo 0)
    S_PROG=$(grep -c "完成" "$LOG_DIR/extract_ScanNet.log" 2>/dev/null | tr -d '\n' || echo 0)
    echo -ne "\r  NAVI: ${N_PROG}/10 步  |  ScanNet: ${S_PROG}/10 步  "
    sleep 3
done
echo ""

wait $PID_NAVI_EXTRACT
wait $PID_SCANNET_EXTRACT

EXTRACT_END=$(date +%s)
EXTRACT_ELAPSED=$((EXTRACT_END - EXTRACT_START))
echo -e "\n${GRN}提取评估完成！${RST} 耗时 ${EXTRACT_ELAPSED}s (两个数据集并行)"

# =========================================================================
# Phase 3: 结果汇总
# =========================================================================
echo -e "\n${YLW}[3/3] 结果汇总${RST}"

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END - TRAIN_START))

echo -e "\n${CYN}=================================================================${RST}"
echo -e "${CYN}                    最 终 结 果 汇 总${RST}"
echo -e "${CYN}=================================================================${RST}"
echo -e "总耗时: ${TOTAL_ELAPSED}s (~$((TOTAL_ELAPSED / 60)) 分钟)"
echo ""

# -------- NAVI --------
echo -e "${YLW}═══ NAVI (3000 pairs) ═══${RST}"
printf "  %-12s  %8s  %8s  %8s  %10s\n" "keep_ratio" "AUC@5" "AUC@10" "AUC@20" "Precision"
printf "  %-12s  %8s  %8s  %8s  %10s\n" "----------" "------" "------" "------" "--------"
for RATIO in "${RATIOS[@]}"; do
    JSON="evaluate/navi_matchability_r${RATIO}/evaluation_results.json"
    if [ -f "$JSON" ]; then
        AUC5=$(python3 -c "import json; d=json.load(open('$JSON')); print(f\"{d['auc@5']:.2f}\")" 2>/dev/null || echo "N/A")
        AUC10=$(python3 -c "import json; d=json.load(open('$JSON')); print(f\"{d['auc@10']:.2f}\")" 2>/dev/null || echo "N/A")
        AUC20=$(python3 -c "import json; d=json.load(open('$JSON')); print(f\"{d['auc@20']:.2f}\")" 2>/dev/null || echo "N/A")
        PREC=$(python3 -c "import json; d=json.load(open('$JSON')); print(f\"{d['precision']:.2f}%\")" 2>/dev/null || echo "N/A")
        printf "  %-12s  %8s  %8s  %8s  %10s\n" "$RATIO" "$AUC5" "$AUC10" "$AUC20" "$PREC"
    else
        printf "  %-12s  %8s  %8s  %8s  %10s\n" "$RATIO" "N/A" "N/A" "N/A" "N/A"
    fi
done
echo ""

# -------- ScanNet --------
echo -e "${YLW}═══ ScanNet (1500 pairs) ═══${RST}"
printf "  %-12s  %8s  %8s  %8s  %10s\n" "keep_ratio" "AUC@5" "AUC@10" "AUC@20" "Precision"
printf "  %-12s  %8s  %8s  %8s  %10s\n" "----------" "------" "------" "------" "--------"
for RATIO in "${RATIOS[@]}"; do
    JSON="evaluate/scannet_matchability_r${RATIO}/evaluation_results.json"
    if [ -f "$JSON" ]; then
        AUC5=$(python3 -c "import json; d=json.load(open('$JSON')); print(f\"{d['auc@5']:.2f}\")" 2>/dev/null || echo "N/A")
        AUC10=$(python3 -c "import json; d=json.load(open('$JSON')); print(f\"{d['auc@10']:.2f}\")" 2>/dev/null || echo "N/A")
        AUC20=$(python3 -c "import json; d=json.load(open('$JSON')); print(f\"{d['auc@20']:.2f}\")" 2>/dev/null || echo "N/A")
        PREC=$(python3 -c "import json; d=json.load(open('$JSON')); print(f\"{d['precision']:.2f}%\")" 2>/dev/null || echo "N/A")
        printf "  %-12s  %8s  %8s  %8s  %10s\n" "$RATIO" "$AUC5" "$AUC10" "$AUC20" "$PREC"
    else
        printf "  %-12s  %8s  %8s  %8s  %10s\n" "$RATIO" "N/A" "N/A" "N/A" "N/A"
    fi
done
echo ""

# -------- 参考基线 --------
echo -e "${CYN}═══ Zero-Shot 基线 (参考) ═══${RST}"
echo -e "  NAVI:    AUC@5=0.24  AUC@10=1.07  AUC@20=3.31  Precision=49.17%"
echo -e "  ScanNet: AUC@5=0.23  AUC@10=1.18  AUC@20=4.44  Precision=32.20%"
echo ""

# 自动识别最优 ratio
echo -e "${CYN}═══ 最优 keep_ratio 分析 ═══${RST}"

for DS in "navi" "scannet"; do
    BEST_R=""
    BEST_P=0
    for RATIO in "${RATIOS[@]}"; do
        if [ "$DS" = "navi" ]; then
            JSON="evaluate/navi_matchability_r${RATIO}/evaluation_results.json"
        else
            JSON="evaluate/scannet_matchability_r${RATIO}/evaluation_results.json"
        fi
        if [ -f "$JSON" ]; then
            P=$(python3 -c "import json; print(json.load(open('$JSON'))['precision'])" 2>/dev/null || echo "0")
            if [ "$(echo "$P > $BEST_P" | bc -l 2>/dev/null || echo 0)" = "1" ]; then
                BEST_P=$P
                BEST_R=$RATIO
            fi
        fi
    done
    if [ -n "$BEST_R" ]; then
        echo "  ${DS^^}: 最优 keep_ratio = $BEST_R (Precision = ${BEST_P}%)"
    fi
done

echo -e "\n${CYN}=================================================================${RST}"
echo -e "${CYN}全流程完成！${RST}"
echo -e "${CYN}=================================================================${RST}"
