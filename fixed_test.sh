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
    $hadoop fs -test -e "${data_root%/}/${day}/_SUCCESS" && \
        $hadoop fs -ls "${data_root%/}/${day}/part*" >/dev/null 2>&1
}

assert_data_day() {
    local data_root=$1
    local day=$2
    if ! data_day_ready "$data_root" "$day"; then
        echo "data day is incomplete (_SUCCESS and part files required): ${data_root%/}/${day}"
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

run_train_range() {
    local data_root=$1
    local start_day=$2
    local end_day=$3
    local restore_day=${4:-}
    local nowt
    nowt=$(date +%Y%m%d%H%M%S)
    local train_args=(-data "$data_root" -start_day "$start_day" -end_day "$end_day" -dump_serving_model 0)
    if [[ -n "$restore_day" ]]; then
        if ! checkpoint_ready "$restore_day"; then
            echo "restore checkpoint is not ready: $restore_day"
            exit 1
        fi
        train_args+=( -checkpoint_path "$(checkpoint_dir "$restore_day")" )
    fi
    CLASSPATH=$(${HADOOP_HDFS_HOME}/bin/hadoop classpath --glob) \
    $python -u train.py "${train_args[@]}" > "log/fixed_train_${start_day}_${end_day}_${nowt}" 2>&1
    if ! checkpoint_ready "$end_day"; then
        echo "training finished but target checkpoint was not created: $end_day"
        exit 1
    fi
}

HADOOP_HDFS_HOME=/usr/local/hadoop-current
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:$HADOOP_HDFS_HOME/lib/native:${JAVA_HOME:-/usr/lib/jvm/java-8-openjdk-amd64}/jre/lib/amd64/server

# Fail before submitting the expensive historical training if the frozen
# Saliency artifact and every derived model view are not exactly aligned.
$python - <<'PY'
import model_conf

assert len(model_conf.saliency_top1700_slots) == 1700
assert len(set(model_conf.saliency_top1700_slots)) == 1700
assert len(model_conf.all_slot_ids) == 1700
assert set(model_conf.all_slot_ids) == set(model_conf.saliency_top1700_slots)
assert len(model_conf.interest_slot_ids) == 504
assert len(model_conf.interest_stat_slot_ids) == 8
assert not model_conf.user_pay_seq
assert not model_conf.u_12h_click_cateIds
assert 'user_pay_seq' not in model_conf.seq_slot_dict
assert 'user_12h_click_cateid' not in model_conf.seq_slot_dict
print('Top1700 preflight passed: registered=1700 interest=504 stats=8')
PY

# Slots are registered from day one, while old-source samples keep the gated
# residual exactly zero. The run starts independently from random weights.
assert_data_day "$old_train_hdfs_dir" "$old_train_end_day"
if ! checkpoint_ready "$old_train_end_day"; then
    run_train_range "$old_train_hdfs_dir" "$train_start_day" "$old_train_end_day"
fi

# Restore only this run's own checkpoint before switching to the new source.
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

echo "fixed-only experiment completed: checkpoint=${train_end_day}, test=[${test_start_day},${test_end_day}]"
