#!/bin/bash

source /etc/profile
source ./common.conf

set -euo pipefail
set -x

mkdir -p log model

date_next() { date -d "$1 +1 day" +%Y%m%d; }
checkpoint_dir() { echo "model/checkpoints/$1/"; }
checkpoint_ready() { [[ -f "$(checkpoint_dir "$1")/checkpoint" ]]; }

assert_zero005_day() {
    local day=$1
    if ! $hadoop fs -test -e "${zero005_hdfs_dir}/${day}/_SUCCESS"; then
        echo "zero005 data day is incomplete: $day"
        exit 1
    fi
}

assert_zero005_range() {
    local day=$1
    local end_day=$2
    while [[ "$day" -le "$end_day" ]]; do
        assert_zero005_day "$day"
        day=$(date_next "$day")
    done
}

run_train_range() {
    local data_root=$1
    local start_day=$2
    local end_day=$3
    local restore_day=${4:-}
    local stamp
    stamp=$(date +%Y%m%d%H%M%S)
    local args=(-data "$data_root" -start_day "$start_day" -end_day "$end_day" -dump_serving_model 0)
    if [[ -n "$restore_day" ]]; then
        checkpoint_ready "$restore_day" || { echo "missing restore checkpoint: $restore_day"; exit 1; }
        args+=( -checkpoint_path "$(checkpoint_dir "$restore_day")" )
    fi
    CLASSPATH=$(${HADOOP_HDFS_HOME}/bin/hadoop classpath --glob) \
        $python -u train.py "${args[@]}" > "log/train_${start_day}_${end_day}_${stamp}" 2>&1
    checkpoint_ready "$end_day" || { echo "target checkpoint missing: $end_day"; exit 1; }
}

run_test_window() {
    local ckpt_day=$1
    local start_day=$2
    local end_day=$3
    local kind=$4
    checkpoint_ready "$ckpt_day" || { echo "test checkpoint missing: $ckpt_day"; exit 1; }
    bash test.sh "$start_day" "$end_day" "$(checkpoint_dir "$ckpt_day")" \
        "${kind}_test_ckpt_${ckpt_day}_from_${start_day}_to" "$zero005_hdfs_dir"
}

HADOOP_HDFS_HOME=/usr/local/hadoop-current
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:$HADOOP_HDFS_HOME/lib/native:${JAVA_HOME:-/usr/lib/jvm/java-8-openjdk-amd64}/jre/lib/amd64/server

# Stage 1: legacy JZ sampling, end-to-end from scratch.
if ! checkpoint_ready "$legacy_train_end_day"; then
    run_train_range "$legacy_train_hdfs_dir" "$train_start_day" "$legacy_train_end_day"
fi

# Stage 2: switch to zero005 and continue from the stage-1 checkpoint.
assert_zero005_range "$zero005_train_start_day" "$train_end_day"
if ! checkpoint_ready "$train_end_day"; then
    run_train_range "$zero005_hdfs_dir" "$zero005_train_start_day" "$train_end_day" "$legacy_train_end_day"
fi

# Frozen fixed window on zero005.
assert_zero005_range "$test_start_day" "$test_end_day"
run_test_window "$train_end_day" "$test_start_day" "$test_end_day" fixed

# Consume 0826..0828 only after fixed evaluation to create the rolling seed.
if ! checkpoint_ready "$auto_test_start_ckpt_day"; then
    run_train_range "$zero005_hdfs_dir" "$(date_next "$train_end_day")" \
        "$auto_test_start_ckpt_day" "$train_end_day"
fi

# Prequential rolling test: checkpoint(D-1) tests D, then D is trained.
ckpt_day=$auto_test_start_ckpt_day
test_day=$(date_next "$ckpt_day")
while [[ "$test_day" -le "$auto_test_end_day" ]]; do
    assert_zero005_day "$test_day"
    run_test_window "$ckpt_day" "$test_day" "$test_day" rolling
    if [[ "$test_day" -lt "$auto_test_end_day" ]] && ! checkpoint_ready "$test_day"; then
        run_train_range "$zero005_hdfs_dir" "$test_day" "$test_day" "$ckpt_day"
    fi
    ckpt_day=$test_day
    test_day=$(date_next "$test_day")
done

$python rolling_test_summary.py --log-dir log --output model/rolling_metrics.tsv
echo "ZERO005_E2E_ROLLING_COMPLETED fixed=${test_start_day}..${test_end_day} rolling=20260829..${auto_test_end_day}"
