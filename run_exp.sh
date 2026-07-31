#!/bin/bash
# 一次性并行启动多个实验目录下的 submit_luban.sh
# 用法: bash run_exp.sh <实验文件夹名1> [实验文件夹名2] ...
# 示例: bash run_exp.sh br_nearby_rank_lhuc_gate br_nearby_rank_mmoe br_nearby_rank_din_wide br_nearby_rank_ext_focal_loss

BASE_DIR="/home/luban/rank-ssl/chenpinyuan/tf_rank_BR"

if [[ $# -eq 0 ]]; then
    echo "用法: bash run_exp.sh <实验文件夹名1> [实验文件夹名2] ..."
    echo "示例: bash run_exp.sh br_nearby_rank_lhuc_gate br_nearby_rank_mmoe"
    exit 1
fi

declare -A PIDS  # 记录 实验名 -> PID

run_one_experiment() {
    local EXPERIMENT="$1"
    local EXP_DIR="${BASE_DIR}/${EXPERIMENT}"

    if [[ ! -d "$EXP_DIR" ]]; then
        echo ">>> [${EXPERIMENT}] 目录不存在: $EXP_DIR，跳过" >&2
        return 1
    fi

    if [[ ! -f "${EXP_DIR}/submit_luban.sh" ]]; then
        echo ">>> [${EXPERIMENT}] 找不到 submit_luban.sh，跳过" >&2
        return 1
    fi

    echo ">>> [${EXPERIMENT}] 启动中..." >&2

    # 子 shell 内 cd，避免污染其他实验的工作目录；日志落在各自实验文件夹内部
    (
        cd "$EXP_DIR" || exit 1
        nohup bash submit_luban.sh > "nohup_submit.log" 2>&1 &
        echo $!
    )
}

# ---- 主循环：依次拉起，全部放后台并行跑 ----
for EXPERIMENT in "$@"; do
    pid=$(run_one_experiment "$EXPERIMENT")
    if [[ -n "$pid" ]]; then
        PIDS["$EXPERIMENT"]="$pid"
    fi
    echo "--------------------------------------"
done

# ---- 汇总 ----
echo ""
echo "=== 所有实验启动情况汇总 ==="
for exp in "${!PIDS[@]}"; do
    echo "实验: $exp  PID: ${PIDS[$exp]}  日志: ${BASE_DIR}/${exp}/nohup_submit.log"
done
