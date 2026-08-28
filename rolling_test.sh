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

day_in_list() {
    local target=$1
    shift
    local value
    for value in "$@"; do
        if [[ "$value" == "$target" ]]; then
            return 0
        fi
    done
    return 1
}

read_missing_days() {
    local key=$1
    "$python" -c 'import json, sys; value=json.load(open("experiment.json")).get(sys.argv[1], []); print(" ".join(value))' "$key"
}

latest_checkpoint_before() {
    local test_day=$1
    local path
    local day
    local latest=""
    for path in model/checkpoints/[0-9]*; do
        [[ -d "$path" ]] || continue
        day=$(basename "$path")
        [[ "$day" =~ ^[0-9]{8}$ ]] || continue
        if [[ "$day" < "$test_day" ]] && checkpoint_ready "$day"; then
            if [[ -z "$latest" || "$day" > "$latest" ]]; then
                latest=$day
            fi
        fi
    done
    if [[ -z "$latest" ]]; then
        echo "no real checkpoint exists before test day $test_day" >&2
        return 1
    fi
    echo "$latest"
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

assert_eval_day() {
    local day=$1
    local success_path="${eval_hdfs_dir%/}/${day}/_SUCCESS"
    if ! $hadoop fs -test -e "$success_path"; then
        echo "Shulan evaluation data day is not ready: $day ($success_path)"
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
        echo "training range contains no Shulan part files: [$start_day,$end_day]"
        exit 1
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
echo "negative-sampled evaluation data: $eval_hdfs_dir"

read -r -a allowed_missing_train_days <<< "$(read_missing_days allowed_missing_train_days)"
read -r -a allowed_missing_test_days <<< "$(read_missing_days allowed_missing_test_days)"
echo "allowed missing train days: ${allowed_missing_train_days[*]:-none}"
echo "allowed missing test days: ${allowed_missing_test_days[*]:-none}"

for missing_day in "${allowed_missing_train_days[@]}"; do
    if checkpoint_ready "$missing_day"; then
        echo "declared missing train day must not have a checkpoint: $missing_day"
        exit 1
    fi
done

if [[ "$auto_test_start_ckpt_day" -lt "$train_end_day" ]]; then
    echo "auto_test_start_ckpt_day must be >= train_end_day"
    exit 1
fi
first_rolling_test_day=$(date_next "$auto_test_start_ckpt_day")
if [[ "$auto_test_end_day" -lt "$first_rolling_test_day" ]]; then
    echo "auto_test_end_day must be >= $first_rolling_test_day"
    exit 1
fi

# Phase 1: train and evaluate on the same Shulan negative-sampled distribution.
if ! checkpoint_ready "$train_end_day"; then
    run_train_range "$train_start_day" "$train_end_day"
fi
assert_eval_range "$test_start_day" "$test_end_day"
run_test_window "$train_end_day" "$test_start_day" "$test_end_day"

# Phase 2: consume the fixed-window dates and create the real rolling seed.
if ! checkpoint_ready "$auto_test_start_ckpt_day"; then
    seed_train_start=$(date_next "$train_end_day")
    run_train_range "$seed_train_start" "$auto_test_start_ckpt_day" "$train_end_day"
fi

# Phase 3: each available test day uses the latest real checkpoint strictly
# before it. Missing days produce no test row and no checkpoint.
test_day=$(date_next "$auto_test_start_ckpt_day")
while [[ "$test_day" -le "$auto_test_end_day" ]]; do
    ckpt_day=$(latest_checkpoint_before "$test_day")

    if $hadoop fs -test -e "${eval_hdfs_dir%/}/${test_day}/_SUCCESS"; then
        run_test_window "$ckpt_day" "$test_day" "$test_day"
    elif day_in_list "$test_day" "${allowed_missing_test_days[@]}"; then
        echo "declared missing evaluation day, skip metrics: $test_day"
    else
        assert_eval_day "$test_day"
    fi

    if [[ "$test_day" -lt "$auto_test_end_day" ]] && ! checkpoint_ready "$test_day"; then
        if sampled_train_day_ready "$test_day"; then
            run_train_range "$test_day" "$test_day" "$ckpt_day"
        elif day_in_list "$test_day" "${allowed_missing_train_days[@]}"; then
            echo "declared missing training day, no checkpoint created: $test_day"
        else
            echo "unexpected missing Shulan training day: $test_day"
            exit 1
        fi
    fi

    test_day=$(date_next "$test_day")
done

$python rolling_test_summary.py --log-dir log --output model/rolling_metrics.tsv
echo "rolling test completed through test day ${auto_test_end_day}"
