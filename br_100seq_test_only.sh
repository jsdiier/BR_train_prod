#!/bin/bash

source /etc/profile
source ./common.conf

set -euo pipefail
set -x

mkdir -p log model/checkpoints

date_next() { date -d "$1 +1 day" +%Y%m%d; }

link_checkpoint() {
    local day=$1
    local host_source="../${source_experiment}/model/checkpoints/${day}"
    if [[ ! -f "${host_source}/checkpoint" ]]; then
        echo "source checkpoint is missing: ${host_source}"
        exit 1
    fi
    ln -sfn "../../../${source_experiment}/model/checkpoints/${day}" \
        "model/checkpoints/${day}"
}

assert_test_day() {
    local day=$1
    $hadoop fs -test -e "${br_100seq_hdfs_dir%/}/${day}/_SUCCESS" || {
        echo "BR 100seq test day is incomplete: $day"
        exit 1
    }
}

run_test_window() {
    local ckpt_day=$1
    local start_day=$2
    local end_day=$3
    local kind=$4
    local marker="log/.${kind}_test_ckpt_${ckpt_day}_${start_day}_${end_day}.done"
    link_checkpoint "$ckpt_day"
    if [[ "${auto_test_resume:-0}" -eq 1 && -f "$marker" ]]; then
        echo "test already completed: ckpt=$ckpt_day test=${start_day}..${end_day}"
        return
    fi
    bash test.sh "$start_day" "$end_day" "model/checkpoints/${ckpt_day}/" \
        "${kind}_test_ckpt_${ckpt_day}_from_${start_day}_to" "$br_100seq_hdfs_dir"
    touch "$marker"
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
echo "BR_100SEQ_TEST_ONLY_COMPLETED source=${source_experiment} test=${test_start_day}..${auto_test_end_day}"
