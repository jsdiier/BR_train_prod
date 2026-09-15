#!/bin/bash

source /etc/profile
source ./common.conf

set -euo pipefail
set -x

mkdir -p log model/checkpoints

date_next() { date -d "$1 +1 day" +%Y%m%d; }
checkpoint_dir() { echo "model/checkpoints/$1/"; }
checkpoint_ready() { [[ -f "$(checkpoint_dir "$1")/checkpoint" ]]; }

assert_data_day() {
    local root=$1
    local day=$2
    $hadoop fs -test -e "${root%/}/${day}/_SUCCESS" || {
        echo "data day is incomplete: ${root%/}/${day}"
        exit 1
    }
}

assert_data_range() {
    local root=$1
    local day=$2
    local end=$3
    while [[ "$day" -le "$end" ]]; do
        assert_data_day "$root" "$day"
        day=$(date_next "$day")
    done
}

run_train_range() {
    local root=$1
    local start=$2
    local end=$3
    local restore=${4:-}
    local stamp
    stamp=$(date +%Y%m%d%H%M%S)
    local args=(-data "$root" -start_day "$start" -end_day "$end" -dump_serving_model 0)
    if [[ -n "$restore" ]]; then
        checkpoint_ready "$restore" || {
            echo "restore checkpoint is missing: $restore"
            exit 1
        }
        args+=( -checkpoint_path "$(checkpoint_dir "$restore")" )
    fi
    CLASSPATH=$(${HADOOP_HDFS_HOME}/bin/hadoop classpath --glob) \
        $python -u train.py "${args[@]}" > "log/train_${start}_${end}_${stamp}" 2>&1
    checkpoint_ready "$end" || {
        echo "target checkpoint was not created: $end"
        exit 1
    }
}

run_test_window() {
    local ckpt=$1
    local start=$2
    local end=$3
    local kind=$4
    local marker="log/.${kind}_test_ckpt_${ckpt}_${start}_${end}.done"
    checkpoint_ready "$ckpt" || {
        echo "test checkpoint is missing: $ckpt"
        exit 1
    }
    if [[ "${auto_test_resume:-0}" -eq 1 && -f "$marker" ]]; then
        echo "test already completed: ckpt=$ckpt test=${start}..${end}"
        return
    fi
    bash test.sh "$start" "$end" "$(checkpoint_dir "$ckpt")" \
        "${kind}_test_ckpt_${ckpt}_from_${start}_to" "$new_train_hdfs_dir"
    touch "$marker"
}

HADOOP_HDFS_HOME=/usr/local/hadoop-current
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:$HADOOP_HDFS_HOME/lib/native:${JAVA_HOME:-/usr/lib/jvm/java-8-openjdk-amd64}/jre/lib/amd64/server

# JZ history is calendar-sparse and train.py skips absent part files. The
# boundary day is required so the first phase can establish its checkpoint.
assert_data_day "$old_train_hdfs_dir" "$old_train_end_day"
if ! checkpoint_ready "$old_train_end_day"; then
    run_train_range "$old_train_hdfs_dir" "$train_start_day" "$old_train_end_day"
fi

assert_data_range "$new_train_hdfs_dir" "$new_train_start_day" "$auto_test_end_day"
if ! checkpoint_ready "$train_end_day"; then
    run_train_range "$new_train_hdfs_dir" "$new_train_start_day" "$train_end_day" "$old_train_end_day"
fi

run_test_window "$train_end_day" "$test_start_day" "$test_end_day" fixed

if ! checkpoint_ready "$auto_test_start_ckpt_day"; then
    run_train_range "$new_train_hdfs_dir" "$(date_next "$train_end_day")" \
        "$auto_test_start_ckpt_day" "$train_end_day"
fi

ckpt_day=$auto_test_start_ckpt_day
test_day=$(date_next "$ckpt_day")
while [[ "$test_day" -le "$auto_test_end_day" ]]; do
    run_test_window "$ckpt_day" "$test_day" "$test_day" rolling
    if [[ "$test_day" -lt "$auto_test_end_day" ]] && ! checkpoint_ready "$test_day"; then
        run_train_range "$new_train_hdfs_dir" "$test_day" "$test_day" "$ckpt_day"
    fi
    ckpt_day=$test_day
    test_day=$(date_next "$ckpt_day")
done

$python rolling_test_summary.py --log-dir log --output model/rolling_metrics.tsv
echo "BR_100SEQ_TWO_STAGE_COMPLETED fixed=${test_start_day}..${test_end_day} rolling_end=${auto_test_end_day}"
