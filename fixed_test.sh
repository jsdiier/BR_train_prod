#!/bin/bash

source /etc/profile
source ./common.conf

set -euo pipefail
set -x

# Do not inherit an accidental collector flag into validation or fixed testing.
unset SALIENCY_COLLECT SALIENCY_EXPECTED_SLOT_COUNT SALIENCY_REMAIN_COUNT || true

mkdir -p log model

checkpoint_dir() {
    echo "model/checkpoints/$1/"
}

checkpoint_ready() {
    [[ -f "$(checkpoint_dir "$1")/checkpoint" ]]
}

source_checkpoint_ready() {
    [[ -d "$source_checkpoint_path" && -f "$source_checkpoint_path/checkpoint" ]]
}

saliency_assets_ready() {
    [[ -s "log/fea_importance_saliency_${train_start_day}_${train_end_day}.tsv" \
       && -s "log/saliency_top${saliency_remain_count}_slots.txt" \
       && -s "log/saliency_pruned$((saliency_expected_slot_count - saliency_remain_count))_slots.txt" \
       && -s "log/saliency_slot_audit.json" ]]
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

run_saliency_train() {
    local nowt
    nowt=$(date +%Y%m%d%H%M%S)
    local train_args=(
        -data "$new_train_hdfs_dir"
        -start_day "$train_start_day"
        -end_day "$train_end_day"
        -checkpoint_path "$source_checkpoint_path"
        -dump_serving_model 0
    )
    export SALIENCY_COLLECT=1
    export SALIENCY_EXPECTED_SLOT_COUNT="$saliency_expected_slot_count"
    export SALIENCY_REMAIN_COUNT="$saliency_remain_count"
    CLASSPATH=$(${HADOOP_HDFS_HOME}/bin/hadoop classpath --glob) \
    $python -u train.py "${train_args[@]}" \
        > "log/saliency_train_${train_start_day}_${train_end_day}_${nowt}" 2>&1
    unset SALIENCY_COLLECT SALIENCY_EXPECTED_SLOT_COUNT SALIENCY_REMAIN_COUNT
    if ! checkpoint_ready "$train_end_day"; then
        echo "training finished but target checkpoint was not created: $train_end_day"
        exit 1
    fi
    if ! saliency_assets_ready; then
        echo "training finished but one or more Saliency assets are missing"
        exit 1
    fi
}

HADOOP_HDFS_HOME=/usr/local/hadoop-current
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:$HADOOP_HDFS_HOME/lib/native:${JAVA_HOME:-/usr/lib/jvm/java-8-openjdk-amd64}/jre/lib/amd64/server

# The parent checkpoint is an immutable input artifact. Never infer the source
# from this run's model.done, and never write into the parent experiment.
if ! source_checkpoint_ready; then
    echo "source checkpoint is not ready: $source_checkpoint_path"
    exit 1
fi

assert_data_range "$new_train_hdfs_dir" "$train_start_day" "$train_end_day"
if ! checkpoint_ready "$train_end_day"; then
    run_saliency_train
elif ! saliency_assets_ready; then
    echo "checkpoint $train_end_day exists but Saliency assets are incomplete"
    echo "refusing to reuse a checkpoint without its in-memory gradient accumulation"
    exit 1
fi

if [[ "${run_fixed_test:-0}" -eq 1 ]]; then
    assert_data_range "$new_train_hdfs_dir" "$test_start_day" "$test_end_day"
    marker="log/.fixed_test_ckpt_${train_end_day}_${test_start_day}_${test_end_day}.done"
    if [[ "${fixed_test_resume:-0}" -ne 1 || ! -f "$marker" ]]; then
        bash test.sh "$test_start_day" "$test_end_day" "$(checkpoint_dir "$train_end_day")" \
            "fixed_test_ckpt_${train_end_day}_from_${test_start_day}_to" "$new_train_hdfs_dir"
        touch "$marker"
    fi
    echo "Saliency full-slot control completed: checkpoint=${train_end_day}, test=[${test_start_day},${test_end_day}], remain=${saliency_remain_count}"
else
    echo "Saliency collection completed: checkpoint=${train_end_day}, remain=${saliency_remain_count}"
    echo "Fixed test is deferred. Run bash run_fixed_test.sh after data [${test_start_day},${test_end_day}] is ready."
fi
