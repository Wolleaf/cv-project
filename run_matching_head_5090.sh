#!/bin/bash
# =========================================================================
# Route B: Frozen DINO + Learnable Attention Matching Head (RTX 5090 32GB)
# =========================================================================
# 这是最后一搏：SuperGlue/LightGlue 风格的注意力匹配头，不改 DINO 特征。
#
# 核心思路：
#   - DINOv3 backbone 完全冻结，只提取 1024-dim 特征
#   - Self-attention 让每个 patch 看到同图所有 patch → 感知自身"独特性"
#   - Cross-attention 让两图 patch 互相交流 → 学习跨图对应
#   - Dual-softmax 匹配替代 MNN → 全局分配问题
#
# 为什么应该有效（之前的都失败了）：
#   - Route A(Predictor)：单 patch 看不出自己是否独特 → 失败
#   - Self-attention：patch 看到同图所有邻居 → 能判断独特性
#   - Cross-attention：patch 看到对方图中的候选 → 能消解歧义
#   - 不改特征空间 → 零语义破坏风险
#
# 流程：
#   Phase 0 → 环境自检
#   Phase 1 → [并行] NAVI 训练 + ScanNet 训练
#   Phase 2 → [并行] NAVI 提取评估 + ScanNet 提取评估
#   Phase 3 → 结果汇总
#
# 用法：
#   ./run_matching_head_5090.sh              # 完整流程
#   ./run_matching_head_5090.sh --skip-train # 跳过训练，只评估
# =========================================================================

set -e

# -------- 阶段控制 --------
SKIP_TRAIN=false
for arg in "$@"; do
    case "$arg" in
        --skip-train) SKIP_TRAIN=true ;;
    esac
done

# -------- 配置 --------
export OMP_NUM_THREADS=4
CONDA_ENV="llmdevelop"
WEIGHTS="dinov3_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
TRAIN_EPOCHS=20
TRAIN_BATCH=4
D_MODEL=256
NHEAD=4
NUM_BLOCKS=4

# -------- 颜色 --------
RED='\e[31m'
GRN='\e[32m'
YLW='\e[33m'
CYN='\e[36m'
MAG='\e[35m'
RST='\e[0m'

# -------- 日志目录 --------
LOG_DIR="./logs_matching_head"
mkdir -p "$LOG_DIR"

echo -e "${CYN}=================================================================${RST}"
echo -e "${CYN}Route B: Frozen DINO + Attention Matching Head (NAVI + ScanNet)${RST}"
echo -e "${CYN}Matcher: d_model=${D_MODEL} nhead=${NHEAD} blocks=${NUM_BLOCKS} (~1M params)${RST}"
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
    for f in "matching_head_navi/matcher_best.pth" "matching_head_scannet/matcher_best.pth"; do
        if [ ! -f "$f" ]; then
            echo -e "${RED}[错误] 缺失: $f${RST}"
            exit 1
        fi
    done
    echo -e "${GRN}模型文件检查通过${RST}"
else
    echo -e "\n${YLW}[1/3] 并行训练 NAVI 和 ScanNet 的 Matching Head...${RST}"
    echo -e "       日志: ${LOG_DIR}/train_navi.log  +  ${LOG_DIR}/train_scannet.log"
    echo -e "       Matcher ~1M 参数，双任务并行 GPU 绰绰有余"

    TRAIN_START=$(date +%s)

    # --- NAVI (后台) ---
    conda run -n $CONDA_ENV python -m finetune.matching_head train \
        --checkpoint "$WEIGHTS" \
        --train_pairs finetune/navi_train_pairs.txt \
        --data_root full_dataset/navi_v1.5 \
        --depth_root full_dataset/navi_v1.5 \
        --output_dir matching_head_navi \
        --epochs $TRAIN_EPOCHS --batch_pairs $TRAIN_BATCH \
        --lr 1e-3 --wd 1e-3 \
        --d_model $D_MODEL --nhead $NHEAD --num_blocks $NUM_BLOCKS \
        --dropout 0.1 --temperature 0.1 --num_workers 4 --img_size 448 \
        > "$LOG_DIR/train_navi.log" 2>&1 &
    PID_NAVI=$!

    # --- ScanNet (后台) ---
    conda run -n $CONDA_ENV python -m finetune.matching_head train \
        --checkpoint "$WEIGHTS" \
        --train_pairs datasets/scannet_with_gt.txt \
        --data_root datasets/test \
        --output_dir matching_head_scannet \
        --epochs $TRAIN_EPOCHS --batch_pairs $TRAIN_BATCH \
        --lr 1e-3 --wd 1e-3 \
        --d_model $D_MODEL --nhead $NHEAD --num_blocks $NUM_BLOCKS \
        --dropout 0.1 --temperature 0.1 --num_workers 4 --img_size 448 \
        > "$LOG_DIR/train_scannet.log" 2>&1 &
    PID_SCANNET=$!

    echo -e "\n  等待两个训练任务完成..."
    while kill -0 $PID_NAVI 2>/dev/null || kill -0 $PID_SCANNET 2>/dev/null; do
        N_DONE=""; S_DONE=""
        kill -0 $PID_NAVI 2>/dev/null && N_DONE="训练中" || N_DONE="${GRN}完成${RST}"
        kill -0 $PID_SCANNET 2>/dev/null && S_DONE="训练中" || S_DONE="${GRN}完成${RST}"
        echo -ne "\r  NAVI: $N_DONE  |  ScanNet: $S_DONE  "
        sleep 2
    done
    echo ""

    wait $PID_NAVI; NAVI_EXIT=$?
    wait $PID_SCANNET; SCANNET_EXIT=$?

    TRAIN_END=$(date +%s)
    TRAIN_ELAPSED=$((TRAIN_END - TRAIN_START))
    echo -e "\n${GRN}训练完成！${RST} 耗时 ${TRAIN_ELAPSED}s (并行执行)"

    if [ $NAVI_EXIT -ne 0 ]; then
        echo -e "${RED}[错误] NAVI 训练失败: ${LOG_DIR}/train_navi.log${RST}"
        tail -20 "$LOG_DIR/train_navi.log"
        exit 1
    fi
    if [ $SCANNET_EXIT -ne 0 ]; then
        echo -e "${RED}[错误] ScanNet 训练失败: ${LOG_DIR}/train_scannet.log${RST}"
        tail -20 "$LOG_DIR/train_scannet.log"
        exit 1
    fi

    echo ""
    echo "NAVI 训练最终 loss:"
    grep "Epoch" "$LOG_DIR/train_navi.log" | tail -3
    echo ""
    echo "ScanNet 训练最终 loss:"
    grep "Epoch" "$LOG_DIR/train_scannet.log" | tail -3
fi

# =========================================================================
# Phase 2: 并行提取 + 评估
# =========================================================================
if [ "$SKIP_TRAIN" = true ]; then
    TRAIN_START=$(date +%s)
fi

echo -e "\n${YLW}[2/3] 并行提取特征 + 评估...${RST}"

EXTRACT_START=$(date +%s)

run_dataset() {
    local NAME=$1 PAIRS=$2 DATA=$3 MATCHER=$4 OUT_DIR=$5 EVAL_DIR=$6
    local LOGF="$LOG_DIR/extract_${NAME}.log"

    echo "[$NAME] 开始提取与评估..." > "$LOGF"

    echo "[$NAME] 提取 + 匹配中..." >> "$LOGF"
    conda run -n $CONDA_ENV python -m finetune.matching_head extract \
        --checkpoint "$WEIGHTS" \
        --matcher "$MATCHER" \
        --pairs "$PAIRS" \
        --data_root "$DATA" \
        --output_dir "$OUT_DIR" \
        --d_model $D_MODEL --nhead $NHEAD --num_blocks $NUM_BLOCKS \
        --img_size 448 --eval_resize 640 480 --conf_threshold 0.01 \
        >> "$LOGF" 2>&1

    echo "[$NAME] 评估中..." >> "$LOGF"
    conda run -n $CONDA_ENV python evaluate/evaluate_csv_essential.py \
        --input_pairs "$PAIRS" \
        --input_dir "$DATA" \
        --input_csv_dir "$OUT_DIR" \
        --output_dir "$EVAL_DIR" \
        >> "$LOGF" 2>&1

    echo "[$NAME] 全部完成" >> "$LOGF"
}

# 并行运行
run_dataset "NAVI" \
    "datasets/navi_with_gt.txt" \
    "datasets/test/navi_resized" \
    "matching_head_navi/matcher_best.pth" \
    "mnn_matching_head_navi/navi" \
    "evaluate/navi_matching_head" &
PID_NAVI_EXT=$!

run_dataset "ScanNet" \
    "datasets/scannet_with_gt.txt" \
    "datasets/test" \
    "matching_head_scannet/matcher_best.pth" \
    "mnn_matching_head_scannet/scannet" \
    "evaluate/scannet_matching_head" &
PID_SCN_EXT=$!

echo -e "\n  等待提取与评估完成..."
while kill -0 $PID_NAVI_EXT 2>/dev/null || kill -0 $PID_SCN_EXT 2>/dev/null; do
    N_PROG=$(grep -c "完成" "$LOG_DIR/extract_NAVI.log" 2>/dev/null | tr -d '\n' || echo 0)
    S_PROG=$(grep -c "完成" "$LOG_DIR/extract_ScanNet.log" 2>/dev/null | tr -d '\n' || echo 0)
    echo -ne "\r  NAVI: ${N_PROG}/1 步  |  ScanNet: ${S_PROG}/1 步  "
    sleep 3
done
echo ""

wait $PID_NAVI_EXT
wait $PID_SCN_EXT

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
JSON_NAVI="evaluate/navi_matching_head/evaluation_results.json"
if [ -f "$JSON_NAVI" ]; then
    AUC5=$(python3 -c "import json; d=json.load(open('$JSON_NAVI')); print(f\"{d['auc@5']:.2f}\")" 2>/dev/null || echo "N/A")
    AUC10=$(python3 -c "import json; d=json.load(open('$JSON_NAVI')); print(f\"{d['auc@10']:.2f}\")" 2>/dev/null || echo "N/A")
    AUC20=$(python3 -c "import json; d=json.load(open('$JSON_NAVI')); print(f\"{d['auc@20']:.2f}\")" 2>/dev/null || echo "N/A")
    PREC=$(python3 -c "import json; d=json.load(open('$JSON_NAVI')); print(f\"{d['precision']:.2f}%\")" 2>/dev/null || echo "N/A")
    printf "  %-12s  %8s  %8s  %8s  %10s\n" "AUC@5" "AUC@10" "AUC@20" "Precision" ""
    printf "  %-12s  %8s  %8s  %8s  %10s\n" "----------" "------" "------" "--------"
    printf "  %-12s  %8s  %8s  %8s  %10s\n" "结果" "$AUC5" "$AUC10" "$AUC20" "$PREC"
else
    echo "  (无结果文件)"
fi
echo ""

# -------- ScanNet --------
echo -e "${YLW}═══ ScanNet (1500 pairs) ═══${RST}"
JSON_SCN="evaluate/scannet_matching_head/evaluation_results.json"
if [ -f "$JSON_SCN" ]; then
    AUC5=$(python3 -c "import json; d=json.load(open('$JSON_SCN')); print(f\"{d['auc@5']:.2f}\")" 2>/dev/null || echo "N/A")
    AUC10=$(python3 -c "import json; d=json.load(open('$JSON_SCN')); print(f\"{d['auc@10']:.2f}\")" 2>/dev/null || echo "N/A")
    AUC20=$(python3 -c "import json; d=json.load(open('$JSON_SCN')); print(f\"{d['auc@20']:.2f}\")" 2>/dev/null || echo "N/A")
    PREC=$(python3 -c "import json; d=json.load(open('$JSON_SCN')); print(f\"{d['precision']:.2f}%\")" 2>/dev/null || echo "N/A")
    printf "  %-12s  %8s  %8s  %8s  %10s\n" "AUC@5" "AUC@10" "AUC@20" "Precision" ""
    printf "  %-12s  %8s  %8s  %8s  %10s\n" "----------" "------" "------" "--------"
    printf "  %-12s  %8s  %8s  %8s  %10s\n" "结果" "$AUC5" "$AUC10" "$AUC20" "$PREC"
else
    echo "  (无结果文件)"
fi
echo ""

# -------- Zero-Shot 基线 --------
echo -e "${CYN}═══ 参考基线 ═══${RST}"
echo -e "  Zero-Shot (MNN):    NAVI AUC@5=0.24 AUC@10=1.07 AUC@20=3.31 Prec=49.17%"
echo -e "  Zero-Shot (MNN):    ScanNet AUC@5=0.23 AUC@10=1.18 AUC@20=4.44 Prec=32.20%"
echo -e "  LoRA+SafeRadius:    ScanNet AUC@5=0.00 AUC@10=0.26 AUC@20=1.84 Prec=28.83%"
echo -e "  Matchability(RouteA): 全部退化到 5-6%"
echo ""

echo -e "${CYN}=================================================================${RST}"
echo -e "${CYN}全流程完成！${RST}"
echo -e "${CYN}=================================================================${RST}"
