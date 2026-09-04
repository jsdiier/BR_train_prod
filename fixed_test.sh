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

assert_data_day() {
    local data_root=$1
    local day=$2
    $hadoop fs -test -e "${data_root%/}/${day}/_SUCCESS" && \
        $hadoop fs -ls "${data_root%/}/${day}/part*" >/dev/null 2>&1 || {
        echo "data day is incomplete: ${data_root%/}/${day}"
        exit 1
    }
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

run_train_range() {
    local data_root=$1
    local start_day=$2
    local end_day=$3
    local restore_day=${4:-}
    local nowt
    nowt=$(date +%Y%m%d%H%M%S)
    local train_args=(-data "$data_root" -start_day "$start_day" -end_day "$end_day" -dump_serving_model 0)
    if [[ -n "$restore_day" ]]; then
        checkpoint_ready "$restore_day" || {
            echo "restore checkpoint is not ready: $restore_day"
            exit 1
        }
        train_args+=( -checkpoint_path "$(checkpoint_dir "$restore_day")" )
    fi
    CLASSPATH=$(${HADOOP_HDFS_HOME}/bin/hadoop classpath --glob) \
    $python -u train.py "${train_args[@]}" > "log/fixed_train_${start_day}_${end_day}_${nowt}" 2>&1
    checkpoint_ready "$end_day" || {
        echo "target checkpoint was not created: $end_day"
        exit 1
    }
}

HADOOP_HDFS_HOME=/usr/local/hadoop-current
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:$HADOOP_HDFS_HOME/lib/native:${JAVA_HOME:-/usr/lib/jvm/java-8-openjdk-amd64}/jre/lib/amd64/server

# All groups initialize independently with their final slot registration.
assert_data_range "$old_train_hdfs_dir" "$train_start_day" "$old_train_end_day"
if ! checkpoint_ready "$old_train_end_day"; then
    run_train_range "$old_train_hdfs_dir" "$train_start_day" "$old_train_end_day"
fi

assert_data_range "$new_train_hdfs_dir" "$new_train_start_day" "$train_end_day"
if ! checkpoint_ready "$train_end_day"; then
    run_train_range "$new_train_hdfs_dir" "$new_train_start_day" "$train_end_day" "$old_train_end_day"
fi

assert_data_range "$new_train_hdfs_dir" "$test_start_day" "$test_end_day"
marker="log/.fixed_test_ckpt_${train_end_day}_${test_start_day}_${test_end_day}.done"
if [[ "${fixed_test_resume:-0}" -ne 1 || ! -f "$marker" ]]; then
    bash test.sh "$test_start_day" "$test_end_day" "$(checkpoint_dir "$train_end_day")" \
        "fixed_test_ckpt_${train_end_day}_from_${test_start_day}_to" "$new_train_hdfs_dir"
    touch "$marker"
fi

echo "JZ v3 factorial completed: checkpoint=${train_end_day}, test=[${test_start_day},${test_end_day}]"
