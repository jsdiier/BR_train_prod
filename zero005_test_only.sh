#!/bin/bash

source /etc/profile
source ./common.conf

set -euo pipefail
set -x

mkdir -p log model/checkpoints

date_next() { date -d "$1 +1 day" +%Y%m%d; }

link_checkpoint() {
    local day=$1
    local source_dir="../${source_experiment}/model/checkpoints/${day}"
    if [[ ! -f "${source_dir}/checkpoint" ]]; then
        echo "source checkpoint is missing: ${source_dir}"
        exit 1
    fi
    local absolute_source
    absolute_source=$(cd "$source_dir" && pwd)
    ln -sfn "$absolute_source" "model/checkpoints/${day}"
}

assert_test_day() {
    local day=$1
    if ! $hadoop fs -test -e "${zero005_hdfs_dir}/${day}/_SUCCESS"; then
        echo "zero005 test day is incomplete: $day"
        exit 1
    fi
}

run_test_window() {
    local ckpt_day=$1
    local start_day=$2
    local end_day=$3
    local kind=$4
    link_checkpoint "$ckpt_day"
    bash test.sh "$start_day" "$end_day" "model/checkpoints/${ckpt_day}/" \
        "${kind}_test_ckpt_${ckpt_day}_from_${start_day}_to" "$zero005_hdfs_dir"
}

day=$test_start_day
while [[ "$day" -le "$test_end_day" ]]; do
    assert_test_day "$day"
    day=$(date_next "$day")
done
run_test_window "$train_end_day" "$test_start_day" "$test_end_day" fixed

ckpt_day=$auto_test_start_ckpt_day
test_day=$(date_next "$ckpt_day")
while [[ "$test_day" -le "$auto_test_end_day" ]]; do
    assert_test_day "$test_day"
    run_test_window "$ckpt_day" "$test_day" "$test_day" rolling
    ckpt_day=$test_day
    test_day=$(date_next "$test_day")
done

$python rolling_test_summary.py --log-dir log --output model/rolling_metrics.tsv
echo "ZERO005_TEST_ONLY_COMPLETED source=${source_experiment} test=${test_start_day}..${auto_test_end_day}"
