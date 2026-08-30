#!/bin/bash

source /etc/profile
source ./common.conf

set -euo pipefail
set -x

mkdir -p log

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
            echo "test data day is incomplete (_SUCCESS and part files required): ${data_root%/}/${day}"
            exit 1
        fi
        day=$(date -d "$day +1 day" +%Y%m%d)
    done
}

run_cross_test() {
    local source_name=$1
    local data_root=$2
    local marker="log/.cross_source_${source_name}_ckpt_${train_end_day}_${test_start_day}_${test_end_day}.done"

    assert_data_range "$data_root" "$test_start_day" "$test_end_day"
    if [[ "${fixed_test_resume:-0}" -ne 1 || ! -f "$marker" ]]; then
        bash test.sh \
            "$test_start_day" \
            "$test_end_day" \
            "$source_checkpoint_dir" \
            "cross_source_${source_name}_ckpt_${train_end_day}_from_${test_start_day}_to" \
            "$data_root"
        touch "$marker"
    fi
}

if [[ ! -f "${source_checkpoint_dir%/}/checkpoint" ]]; then
    echo "source checkpoint is missing: ${source_checkpoint_dir%/}/checkpoint"
    exit 1
fi

run_cross_test "jiazhuo" "$jiazhuo_test_hdfs_dir"
run_cross_test "chenpinyuan_fixed" "$chenpinyuan_fixed_test_hdfs_dir"

echo "cross-source test completed: source_checkpoint=${source_checkpoint_dir}, test=[${test_start_day},${test_end_day}]"
