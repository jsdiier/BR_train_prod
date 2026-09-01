#!/bin/bash

source /etc/profile
source ./common.conf

set -euo pipefail
set -x

mkdir -p log model

date_next() {
    date -d "$1 +1 day" +%Y%m%d
}

checkpoint_dir() {
    echo "model/checkpoints/$1/"
}

checkpoint_ready() {
    [[ -f "$(checkpoint_dir "$1")/checkpoint" ]]
}

assert_ready_range() {
    local data_dir=$1
    local day=$2
    local end_day=$3
    while [[ "$day" -le "$end_day" ]]; do
        if ! $hadoop fs -test -e "${data_dir}/${day}/_SUCCESS"; then
            echo "data day is not ready: ${data_dir}/${day}"
            exit 1
        fi
        day=$(date_next "$day")
    done
}

HADOOP_HDFS_HOME=/usr/local/hadoop-current
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:$HADOOP_HDFS_HOME/lib/native:${JAVA_HOME:-/usr/lib/jvm/java-8-openjdk-amd64}/jre/lib/amd64/server

if ! checkpoint_ready "$extension_restore_day"; then
    echo "restore checkpoint is not ready: $extension_restore_day"
    exit 1
fi

assert_ready_range "$train_hdfs_dir" "$extension_train_start_day" "$extension_train_end_day"
assert_ready_range "$fixed_test_hdfs_dir" "$extension_test_start_day" "$extension_test_end_day"

if ! checkpoint_ready "$extension_train_end_day"; then
    nowt=$(date +%Y%m%d%H%M%S)
    CLASSPATH=$(${HADOOP_HDFS_HOME}/bin/hadoop classpath --glob) \
    $python -u train.py \
        -data "$train_hdfs_dir" \
        -start_day "$extension_train_start_day" \
        -end_day "$extension_train_end_day" \
        -checkpoint_path "$(checkpoint_dir "$extension_restore_day")" \
        -dump_serving_model 0 \
        > "log/extension_train_${extension_train_start_day}_${extension_train_end_day}_${nowt}" 2>&1
fi

if ! checkpoint_ready "$extension_train_end_day"; then
    echo "training finished but checkpoint is missing: $extension_train_end_day"
    exit 1
fi

marker="log/.fixed_test_ckpt_${extension_train_end_day}_${extension_test_start_day}_${extension_test_end_day}.done"
if [[ ! -f "$marker" ]]; then
    bash test.sh \
        "$extension_test_start_day" \
        "$extension_test_end_day" \
        "$(checkpoint_dir "$extension_train_end_day")" \
        "fixed_test_ckpt_${extension_train_end_day}_from_${extension_test_start_day}_to" \
        "$fixed_test_hdfs_dir"
    touch "$marker"
fi

echo "historical baseline extension completed: checkpoint=$extension_train_end_day test=[$extension_test_start_day,$extension_test_end_day]"
