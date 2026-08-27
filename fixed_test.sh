#!/bin/bash

source /etc/profile
source ./common.conf

set -euo pipefail
set -x

mkdir -p log model

checkpoint_dir() {
    echo "model/checkpoints/$1/"
}

checkpoint_ready() {
    [[ -f "$(checkpoint_dir "$1")/checkpoint" ]]
}

data_day_ready() {
    local data_root=$1
    local day=$2
    $hadoop fs -ls "${data_root%/}/${day}/part*" >/dev/null 2>&1
}

assert_data_day() {
    local data_root=$1
    local day=$2
    if ! data_day_ready "$data_root" "$day"; then
        echo "data day has no part files: ${data_root%/}/${day}/part*"
        exit 1
    fi
}

assert_data_range() {
    local data_root=$1
    local day=$2
    local end_day=$3
    while [[ "$day" -le "$end_day" ]]; do
        assert_data_day "$data_root" "$day"
        day=$(date -d "$day +1 day" +%Y%m%d)
    done
}

HADOOP_HDFS_HOME=/usr/local/hadoop-current
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:$HADOOP_HDFS_HOME/lib/native:${JAVA_HOME:-/usr/lib/jvm/java-8-openjdk-amd64}/jre/lib/amd64/server

# A fresh experiment directory starts from random initialization. train.py
# tolerates unavailable historical dates and still requires the target day.
assert_data_day "$old_train_hdfs_dir" "$train_end_day"
if ! checkpoint_ready "$train_end_day"; then
    nowt=$(date +%Y%m%d%H%M%S)
    CLASSPATH=$(${HADOOP_HDFS_HOME}/bin/hadoop classpath --glob) \
    $python -u train.py -data "$old_train_hdfs_dir" -start_day "$train_start_day" \
        -end_day "$train_end_day" -dump_serving_model 0 \
        > "log/fixed_train_${train_start_day}_${train_end_day}_${nowt}" 2>&1
fi
if ! checkpoint_ready "$train_end_day"; then
    echo "training finished but target checkpoint was not created: $train_end_day"
    exit 1
fi

assert_data_range "$new_test_hdfs_dir" "$test_start_day" "$test_end_day"
marker="log/.fixed_test_ckpt_${train_end_day}_${test_start_day}_${test_end_day}.done"
if [[ "${fixed_test_resume:-0}" -ne 1 || ! -f "$marker" ]]; then
    bash test.sh "$test_start_day" "$test_end_day" "$(checkpoint_dir "$train_end_day")" \
        "fixed_test_ckpt_${train_end_day}_from_${test_start_day}_to" "$new_test_hdfs_dir"
    touch "$marker"
fi

echo "fixed-only experiment completed: checkpoint=${train_end_day}, test=[${test_start_day},${test_end_day}]"
