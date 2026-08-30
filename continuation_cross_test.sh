#!/bin/bash

source /etc/profile
source ./common.conf

set -euo pipefail
set -x

mkdir -p log model

HADOOP_HDFS_HOME=/usr/local/hadoop-current
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:$HADOOP_HDFS_HOME/lib/native:${JAVA_HOME:-/usr/lib/jvm/java-8-openjdk-amd64}/jre/lib/amd64/server

checkpoint_dir() {
    echo "model/checkpoints/$1/"
}

checkpoint_ready() {
    [[ -f "$(checkpoint_dir "$1")/checkpoint" ]]
}

data_day_ready() {
    local data_root=$1
    local day=$2
    $hadoop fs -test -e "${data_root%/}/${day}/_SUCCESS" && \
        $hadoop fs -ls "${data_root%/}/${day}/part*" >/dev/null 2>&1
}

assert_data_range() {
    local data_root=$1
    local day=$2
    local end_day=$3
    while [[ "$day" -le "$end_day" ]]; do
        if ! data_day_ready "$data_root" "$day"; then
            echo "data day is incomplete (_SUCCESS and part files required): ${data_root%/}/${day}"
            exit 1
        fi
        day=$(date -d "$day +1 day" +%Y%m%d)
    done
}

run_cross_test() {
    local source_name=$1
    local data_root=$2
    local marker="log/.continuation_cross_${source_name}_ckpt_${train_end_day}_${test_start_day}_${test_end_day}.done"

    assert_data_range "$data_root" "$test_start_day" "$test_end_day"
    if [[ "${fixed_test_resume:-0}" -ne 1 || ! -f "$marker" ]]; then
        bash test.sh \
            "$test_start_day" \
            "$test_end_day" \
            "$(checkpoint_dir "$train_end_day")" \
            "continuation_cross_${source_name}_ckpt_${train_end_day}_from_${test_start_day}_to" \
            "$data_root"
        touch "$marker"
    fi
}

if [[ ! -f "${source_checkpoint_dir%/}/checkpoint" ]]; then
    echo "common source checkpoint is missing: ${source_checkpoint_dir%/}/checkpoint"
    exit 1
fi

assert_data_range "$continuation_train_hdfs_dir" "$train_start_day" "$train_end_day"

if ! checkpoint_ready "$train_end_day"; then
    nowt=$(date +%Y%m%d%H%M%S)
    CLASSPATH=$(${HADOOP_HDFS_HOME}/bin/hadoop classpath --glob) \
    $python -u train.py \
        -data "$continuation_train_hdfs_dir" \
        -start_day "$train_start_day" \
        -end_day "$train_end_day" \
        -checkpoint_path "$source_checkpoint_dir" \
        -dump_serving_model 0 \
        > "log/continuation_train_${train_start_day}_${train_end_day}_${nowt}" 2>&1
fi

if ! checkpoint_ready "$train_end_day"; then
    echo "continuation finished but target checkpoint was not created: $train_end_day"
    exit 1
fi

run_cross_test "jiazhuo" "$jiazhuo_test_hdfs_dir"
run_cross_test "chenpinyuan_fixed" "$chenpinyuan_fixed_test_hdfs_dir"

echo "common-checkpoint continuation completed: source=${source_checkpoint_dir}, train=${continuation_train_hdfs_dir}, test=[${test_start_day},${test_end_day}]"
