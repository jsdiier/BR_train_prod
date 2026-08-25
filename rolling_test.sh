#!/bin/bash

source /etc/profile
source ./common.conf

set -euo pipefail
set -x

mkdir -p log model

: "${need_auto_test:=0}"
if [[ "$need_auto_test" -ne 1 ]]; then
    echo "need_auto_test is not enabled"
    exit 1
fi

date_next() {
    date -d "$1 +1 day" +%Y%m%d
}

checkpoint_dir() {
    echo "model/checkpoints/$1/"
}

checkpoint_ready() {
    [[ -f "$(checkpoint_dir "$1")/checkpoint" ]]
}

sampled_train_day_ready() {
    local day=$1
    local day_path="${train_hdfs_dir%/}/${day}"
    $hadoop fs -ls "${day_path}/part*" >/dev/null 2>&1
}

sampled_train_range_has_data() {
    local day=$1
    local end_day=$2
    while [[ "$day" -le "$end_day" ]]; do
        if sampled_train_day_ready "$day"; then
            return 0
        fi
        day=$(date_next "$day")
    done
    return 1
}

carry_forward_checkpoint() {
    local restore_day=$1
    local target_day=$2
    local source_dir
    local target_dir
    source_dir=$(checkpoint_dir "$restore_day")
    target_dir=$(checkpoint_dir "$target_day")
    mkdir -p "$target_dir"
    cp -R "${source_dir%/}/." "$target_dir/"
    printf '%s\t%s\n' "$target_day" "$target_dir" >> model/model.done
    echo "no sampled data in target range; carried checkpoint $restore_day -> $target_day"
}

assert_eval_day() {
    local day=$1
    local success_path="${eval_hdfs_dir%/}/${day}/_SUCCESS"
    if ! $hadoop fs -test -e "$success_path"; then
        echo "canonical BR evaluation data day is not ready: $day ($success_path)"
        exit 1
    fi
}

assert_eval_range() {
    local day=$1
    local end_day=$2
    while [[ "$day" -le "$end_day" ]]; do
        assert_eval_day "$day"
        day=$(date_next "$day")
    done
}

run_train_range() {
    local start_day=$1
    local end_day=$2
    local restore_day=${3:-}
    local nowt
    nowt=$(date +%Y%m%d%H%M%S)

    if [[ -n "$restore_day" ]]; then
        if ! checkpoint_ready "$restore_day"; then
            echo "restore checkpoint is not ready: $restore_day"
            exit 1
        fi
    fi

    if ! sampled_train_range_has_data "$start_day" "$end_day"; then
        if [[ -z "$restore_day" ]]; then
            echo "initial training range contains no negative-sampled part files"
            exit 1
        fi
        carry_forward_checkpoint "$restore_day" "$end_day"
        return
    fi

    local train_args=(-data "$train_hdfs_dir" -start_day "$start_day" -end_day "$end_day" -dump_serving_model 0)
    if [[ -n "$restore_day" ]]; then
        train_args+=( -checkpoint_path "$(checkpoint_dir "$restore_day")" )
    fi

    CLASSPATH=$(${HADOOP_HDFS_HOME}/bin/hadoop classpath --glob) \
    $python -u train.py "${train_args[@]}" > "log/rolling_train_${start_day}_${end_day}_${nowt}" 2>&1

    if ! checkpoint_ready "$end_day"; then
        echo "training finished but target checkpoint was not created: $end_day"
        exit 1
    fi
}

run_test_window() {
    local ckpt_day=$1
    local start_day=$2
    local end_day=$3
    local marker="log/.rolling_test_ckpt_${ckpt_day}_test_${start_day}_${end_day}.done"
    if [[ "${auto_test_resume:-0}" -eq 1 && -f "$marker" ]]; then
        echo "rolling test already completed, skip: ckpt=$ckpt_day test=[$start_day,$end_day]"
        return
    fi
    if ! checkpoint_ready "$ckpt_day"; then
        echo "test checkpoint is not ready: $ckpt_day"
        exit 1
    fi

    bash test.sh "$start_day" "$end_day" "$(checkpoint_dir "$ckpt_day")" \
        "rolling_test_ckpt_${ckpt_day}_from_${start_day}_to"
    touch "$marker"
}

HADOOP_HDFS_HOME=/usr/local/hadoop-current
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:$HADOOP_HDFS_HOME/lib/native:${JAVA_HOME:-/usr/lib/jvm/java-8-openjdk-amd64}/jre/lib/amd64/server

echo "negative-sampled training data: $train_hdfs_dir"
echo "canonical BR evaluation data: $eval_hdfs_dir"

if [[ "$auto_test_start_ckpt_day" -lt "$train_end_day" ]]; then
    echo "auto_test_start_ckpt_day must be >= train_end_day"
    exit 1
fi
first_rolling_test_day=$(date_next "$auto_test_start_ckpt_day")
if [[ "$auto_test_end_day" -lt "$first_rolling_test_day" ]]; then
    echo "auto_test_end_day must be >= $first_rolling_test_day"
    exit 1
fi

# Phase 1: train on negative-sampled data and evaluate the fixed window on
# canonical unsampled BR data.
if ! checkpoint_ready "$train_end_day"; then
    run_train_range "$train_start_day" "$train_end_day"
fi
assert_eval_range "$test_start_day" "$test_end_day"
run_test_window "$train_end_day" "$test_start_day" "$test_end_day"

# Phase 2: consume the fixed-window dates from the sampled training source and
# create the rolling seed checkpoint.
if ! checkpoint_ready "$auto_test_start_ckpt_day"; then
    seed_train_start=$(date_next "$train_end_day")
    run_train_range "$seed_train_start" "$auto_test_start_ckpt_day" "$train_end_day"
fi

# Phase 3: checkpoint(D) tests canonical BR data(D+1), then sampled data(D+1)
# advances the continuation state.
ckpt_day=$auto_test_start_ckpt_day
test_day=$(date_next "$ckpt_day")
while [[ "$test_day" -le "$auto_test_end_day" ]]; do
    assert_eval_day "$test_day"
    run_test_window "$ckpt_day" "$test_day" "$test_day"

    if [[ "$test_day" -lt "$auto_test_end_day" ]] && ! checkpoint_ready "$test_day"; then
        run_train_range "$test_day" "$test_day" "$ckpt_day"
    fi

    ckpt_day=$test_day
    test_day=$(date_next "$ckpt_day")
done

$python rolling_test_summary.py --log-dir log --output model/rolling_metrics.tsv
echo "rolling test completed through test day ${auto_test_end_day}"
