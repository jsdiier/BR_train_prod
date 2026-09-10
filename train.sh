#!/bin/bash

source ./common.conf

nowt=`date +"%Y%m%d%H%M"`

set -x
set -e

CONF_FILE=./common.conf
ckpt_day=$train_end_day
: ${is_auto_train:=0}
if [ $is_auto_train -eq 1 ];then
    current_date=$(date +%Y%m%d)
    temp_date="$current_date"
    max_days=20
    found=0
    eday=""
    bday=$(date -d "$train_end_day +1 day" +%Y%m%d)
    count=0

    while [[ "$temp_date" -ge "$bday" ]]; do
        if [ $count -ge $max_days ]; then
            echo "例行时间太久"
            exit 1
        fi

        done_file_path=${train_hdfs_dir}/${temp_date}"/_SUCCESS"
        if $hadoop fs -test -e "$done_file_path"; then
            echo "temp_date is exist"
            eday="$temp_date"
            found=1
            break
        fi

        temp_date=$(date -d "$temp_date -1 day" +%Y%m%d)
        if [[ $? -ne 0 ]]; then
            echo "错误: 日期计算失败"
            exit 1
        fi

        count=$((count + 1))
    done

    if [[ $found -eq 0 ]]; then
        echo "no ready day"
        exit 1
    fi

    if [[ ! -d "./model/checkpoints/${ckpt_day}" ]]; then
        echo "ckpt dir not exist"
        exit 1
    fi

    train_start_day=$bday
    train_end_day=$eday
fi

exec 1>"./log/train_log_${train_end_day}_$nowt" 2>&1

bday=`date -d"$train_start_day" +%Y%m%d`
eday=`date -d"$train_end_day" +%Y%m%d`

# ── Saliency Map 特征选择开关：全部来自 common.conf，export 给 Python ──────────
: ${saliency_select_stage:=0}
: ${saliency_prune_num:=50}
: ${enable_eval:=1}
export saliency_select_stage saliency_prune_num enable_eval

# 按"方法名_起始日_结束日"构造 Saliency 重要性文件路径（放 log/）
SALIENCY_IMP_FILE_AUTO="./log/fea_importance_saliency_${bday}_${eday}.txt"

# Saliency Map 阶段一：告知 Python 把输出写到哪里，并开启梯度收集模式
if [ "$saliency_select_stage" -eq 1 ]; then
    export SALIENCY_IMPORTANCE_FILE="$SALIENCY_IMP_FILE_AUTO"
    export saliency_collect=1
    echo "[Saliency Map 阶段一] 将收集 ∂loss/∂emb 梯度范数，收敛后写出: $SALIENCY_IMPORTANCE_FILE"
fi

# Saliency Map 阶段二：只读 common.conf 里 saliency_importance_file（阶段一已自动回写）
if [ "$saliency_select_stage" -eq 2 ]; then
    : ${saliency_importance_file:=}
    if [ -n "$saliency_importance_file" ] && [ -f "$saliency_importance_file" ]; then
        export SALIENCY_IMPORTANCE_FILE="$saliency_importance_file"
        echo "[Saliency Map 阶段二] 使用 common.conf 指定的重要性文件: $SALIENCY_IMPORTANCE_FILE，将删掉最不重要的 ${saliency_prune_num} 个 slot"
        # 备份当前 model_conf.py，阶段二完成后还原
        cp ./model_conf.py ./model_conf.py.bak
        echo "[Saliency Map 阶段二] 已备份 model_conf.py -> model_conf.py.bak"
    else
        echo "[Saliency Map 阶段二] common.conf 中 saliency_importance_file 未设置或文件不存在('$saliency_importance_file')，请先跑阶段一(saliency_select_stage=1)，退出"
        exit 1
    fi
fi

# ── 自定义 checkpoint 加载路径（从 common.conf 读取）──────────────────────────
: ${custom_checkpoint_path:=}
if [ -n "$custom_checkpoint_path" ]; then
    export CUSTOM_CHECKPOINT_PATH="$custom_checkpoint_path"
    echo "[Custom Checkpoint] 从 common.conf 读取到自定义路径: $CUSTOM_CHECKPOINT_PATH"
fi

# ── 启动训练 ──────────────────────────────────────────────────────────────────
HADOOP_HDFS_HOME=/usr/local/hadoop-current
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HADOOP_HDFS_HOME/lib/native:${JAVA_HOME}/jre/lib/amd64/server
CLASSPATH=$(${HADOOP_HDFS_HOME}/bin/hadoop classpath --glob) \
$python -u train.py -data $train_hdfs_dir -start_day "$bday" -end_day "$eday"

# ── 训练成功后处理 ─────────────────────────────────────────────────────────────
if [ $? -eq 0 ]; then
    # 例行训练更新日期
    if [ $is_auto_train -eq 1 ];then
        echo "train succ"
        if grep -q "^train_start_day=" "$CONF_FILE"; then
            sed -i "s/^train_start_day=.*/train_start_day=$bday/" "$CONF_FILE"
        else
            echo "train_start_day=$bday" >> "$CONF_FILE"
        fi

        if grep -q "^train_end_day=" "$CONF_FILE"; then
            sed -i "s/^train_end_day=.*/train_end_day=$eday/" "$CONF_FILE"
        else
            echo "train_end_day=$eday" >> "$CONF_FILE"
        fi
    fi

    # Saliency Map 阶段一：跳过 serving，把文件路径回写 common.conf
    if [ "$saliency_select_stage" -eq 1 ]; then
        echo "[Saliency Map 阶段一] 重要性文件: $SALIENCY_IMP_FILE_AUTO，跳过 put_serving.sh"
        if grep -q "^saliency_importance_file=" "$CONF_FILE"; then
            sed -i "s|^saliency_importance_file=.*|saliency_importance_file=$SALIENCY_IMP_FILE_AUTO|" "$CONF_FILE"
        else
            echo "saliency_importance_file=$SALIENCY_IMP_FILE_AUTO" >> "$CONF_FILE"
        fi
        echo "已将路径写入 $CONF_FILE: saliency_importance_file=$SALIENCY_IMP_FILE_AUTO"
        echo "下一步：将 common.conf 中 saliency_select_stage 改为 2，重新运行 train.sh 即可完成特征剪枝重训"

    # 正常训练 / Saliency Map 阶段二：上传 serving，阶段二完成后自动重置开关
    else
        bash put_serving.sh
        # 阶段二训练+上线完成后：还原 model_conf.py（从 .bak 恢复完整特征集），
        # 保证下次跑其他方法时仍从原始完整特征集出发
        CONF_BAK="./model_conf.py.bak"
        if [ -f "$CONF_BAK" ]; then
            cp "$CONF_BAK" ./model_conf.py
            echo "已从 $CONF_BAK 还原 model_conf.py（完整特征集）"
        fi
        # 自动把开关重置为 0，避免下次跑其他方法时仍带着剪枝逻辑
        if [ "$saliency_select_stage" -eq 2 ]; then
            sed -i "s/^saliency_select_stage=.*/saliency_select_stage=0/" "$CONF_FILE"
            sed -i "s/^test_method=.*/test_method=saliency/" "$CONF_FILE"
            echo "[Saliency Map 阶段二] 训练完成，已将 common.conf 中 saliency_select_stage 重置为 0，test_method 设为 saliency"
        fi
    fi
else
     echo "train failed"
fi

if [ $need_test -eq 1 ];then
    bash test.sh
fi
