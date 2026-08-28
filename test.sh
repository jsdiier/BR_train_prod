#!/bin/bash

source /etc/profile
source ./common.conf

nowt=`date +"%Y%m%d%H%M"`

set -x
set -e

CONF_FILE=./common.conf

exec 1>"./log/test_log_${test_end_day}_$nowt" 2>&1

# ── 自动还原 model_conf.py（与 train.sh 保持一致）──────────────────────────────
if [ -f "model_conf.py.bak" ]; then
    cp model_conf.py.bak model_conf.py
    echo "[自动还原] 检测到 model_conf.py.bak，已将 model_conf.py 还原为完整特征集"
fi

# ── 特征选择：根据 common.conf 中 test_method 指定用哪个方法剪枝 ──
export saliency_select_stage=0
export fea_select_stage=0

case "${test_method:-}" in
    bn)
        if [ -n "$fea_importance_file" ] && [ -f "$fea_importance_file" ]; then
            export fea_select_stage=2 fea_prune_num
            export fea_importance_file
            echo "[BN-gamma test] test_method=bn，重要性文件: $fea_importance_file，触发剪枝"
        else
            echo "[BN-gamma test] test_method=bn 但重要性文件不存在或为空，跳过剪枝"
        fi
        ;;
    senet)
        if [ -n "$se_importance_file" ] && [ -f "$se_importance_file" ]; then
            export fea_select_stage=2 fea_prune_num se_reduction_ratio
            export SE_IMPORTANCE_FILE="$se_importance_file"
            echo "[SENet test] test_method=senet，重要性文件: $se_importance_file，触发剪枝"
        else
            echo "[SENet test] test_method=senet 但重要性文件不存在或为空，跳过剪枝"
        fi
        ;;
    saliency)
        if [ -n "$saliency_importance_file" ] && [ -f "$saliency_importance_file" ]; then
            export saliency_select_stage=2 saliency_prune_num
            export SALIENCY_IMPORTANCE_FILE="$saliency_importance_file"
            echo "[Saliency test] test_method=saliency，重要性文件: $saliency_importance_file，触发剪枝"
        else
            echo "[Saliency test] test_method=saliency 但重要性文件不存在或为空，跳过剪枝"
        fi
        ;;
    *)
        echo "[test.sh] test_method 未设置或为空，不做特征剪枝，使用完整特征集评估"
        ;;
esac

# ── 自定义 checkpoint 加载路径（从 common.conf 读取，与 train.sh 保持一致）──
: ${custom_checkpoint_path:=}
if [ -n "$custom_checkpoint_path" ]; then
    export CUSTOM_CHECKPOINT_PATH="$custom_checkpoint_path"
    echo "[Custom Checkpoint] 从 common.conf 读取到自定义路径: $CUSTOM_CHECKPOINT_PATH"
fi

HADOOP_HDFS_HOME=/usr/local/hadoop-current
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HADOOP_HDFS_HOME/lib/native:${JAVA_HOME}/jre/lib/amd64/server
CLASSPATH=$(${HADOOP_HDFS_HOME}/bin/hadoop classpath --glob) \
$python -u test.py -data $train_hdfs_dir -start_day $test_start_day -end_day $test_end_day

# ── 跑完恢复 model_conf.py ──────────────────────────────────────────────────
if [ -f "model_conf.py.bak" ]; then
    cp model_conf.py.bak model_conf.py
    echo "[test.sh] 已从 model_conf.py.bak 还原完整特征集"
fi
