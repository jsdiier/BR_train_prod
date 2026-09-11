#!/bin/bash

source ./common.conf

nowt=`date +"%Y%m%d%H%M"`

set -x
set -e

CONF_FILE=./common.conf
: ${is_auto_train:=0}
if [ $is_auto_train -eq 1 ];then
    DONE_FILE="./model/model.done"
    if [[ ! -s "$DONE_FILE" ]]; then
        echo "model.done is missing or empty: $DONE_FILE"
        exit 1
    fi

    last_model_line=$(awk 'NF >= 2 { line = $0 } END { print line }' "$DONE_FILE")
    ckpt_day=$(echo "$last_model_line" | awk '{print $1}')
    ckpt_path=$(echo "$last_model_line" | awk '{print $2}')
    if [[ ! "$ckpt_day" =~ ^[0-9]{8}$ || -z "$ckpt_path" ]]; then
        echo "invalid model.done last line: $last_model_line"
        exit 1
    fi

    if [[ "$ckpt_path" = /* ]]; then
        local_ckpt_path="$ckpt_path"
    else
        local_ckpt_path="./${ckpt_path#./}"
    fi
    if [[ ! -d "$local_ckpt_path" ]]; then
        echo "checkpoint from model.done does not exist: $local_ckpt_path"
        exit 1
    fi

    current_date=$(date +%Y%m%d)
    temp_date="$current_date"
    max_days=20
    found=0
    eday=""
    bday=$(date -d "$ckpt_day +1 day" +%Y%m%d)
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

    check_day="$bday"
    while [[ "$check_day" -le "$eday" ]]; do
        done_file_path=${train_hdfs_dir}/${check_day}"/_SUCCESS"
        if ! $hadoop fs -test -e "$done_file_path"; then
            echo "training data day is incomplete: $done_file_path"
            exit 1
        fi
        check_day=$(date -d "$check_day +1 day" +%Y%m%d)
    done

    train_start_day=$bday
    train_end_day=$eday
    echo "AUTO_TRAIN_RANGE checkpoint_day=$ckpt_day start_day=$bday end_day=$eday"
fi

exec 1>"./log/train_log_${train_end_day}_$nowt" 2>&1

bday=`date -d"$train_start_day" +%Y%m%d`
eday=`date -d"$train_end_day" +%Y%m%d`

HADOOP_HDFS_HOME=/usr/local/hadoop-current
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HADOOP_HDFS_HOME/lib/native:${JAVA_HOME}/jre/lib/amd64/server
CLASSPATH=$(${HADOOP_HDFS_HOME}/bin/hadoop classpath --glob) \
$python -u train.py -data $train_hdfs_dir -start_day $bday -end_day $eday \
    -dump_serving_model $need_dump_serving_model

if [ $? -eq 0 ]; then
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
    bash put_serving.sh
else
     echo "train failed"
fi

if [ $need_test -eq 1 ];then
    bash test.sh
fi
