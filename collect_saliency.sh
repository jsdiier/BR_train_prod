#!/bin/bash

source /etc/profile
source ./common.conf

set -euo pipefail
set -x

mkdir -p log model

checkpoint_dir="model/checkpoints/${train_end_day}"
importance_file="log/fea_importance_saliency_${train_start_day}_${train_end_day}.tsv"
top_file="log/saliency_top${saliency_remain_count}_slots.txt"
pruned_count=$((saliency_expected_slot_count - saliency_remain_count))
pruned_file="log/saliency_pruned${pruned_count}_slots.txt"
audit_file="log/saliency_slot_audit.json"

assets_ready() {
    [[ -f "${checkpoint_dir}/checkpoint" \
       && -s "$importance_file" \
       && -s "$top_file" \
       && -s "$pruned_file" \
       && -s "$audit_file" ]]
}

validate_assets() {
    local importance_rows top_rows pruned_rows
    importance_rows=$(grep -cv '^#' "$importance_file")
    top_rows=$(grep -cve '^[[:space:]]*$' "$top_file")
    pruned_rows=$(grep -cve '^[[:space:]]*$' "$pruned_file")
    [[ "$importance_rows" -eq "$saliency_expected_slot_count" ]]
    [[ "$top_rows" -eq "$saliency_remain_count" ]]
    [[ "$pruned_rows" -eq "$pruned_count" ]]

    "$python" - "$audit_file" "$top_file" "$pruned_file" "$importance_file" \
        "$saliency_expected_slot_count" "$saliency_remain_count" \
        "$train_start_day" "$train_end_day" "$source_checkpoint_path" <<'PY'
import json
import sys

audit_path, top_path, pruned_path, importance_path = sys.argv[1:5]
expected, remain = map(int, sys.argv[5:7])
start_day, end_day, source_checkpoint_path = sys.argv[7:10]
audit = json.load(open(audit_path))
top = [int(line.strip()) for line in open(top_path) if line.strip()]
pruned = [int(line.strip()) for line in open(pruned_path) if line.strip()]
ranked = [int(line.split('\t', 1)[0]) for line in open(importance_path)
          if line.strip() and not line.startswith('#')]
assert audit['registered_slot_count'] == expected, audit
assert audit['unique_slot_count'] == expected, audit
assert audit['remain_count'] == remain, audit
assert audit['pruned_count'] == expected - remain, audit
assert audit['gradient_steps'] > 0, audit
assert audit['collection_start_day'] == start_day, audit
assert audit['collection_end_day'] == end_day, audit
assert audit['source_checkpoint'].startswith(source_checkpoint_path.rstrip('/') + '/'), audit
assert len(top) == len(set(top)) == remain
assert len(pruned) == len(set(pruned)) == expected - remain
assert len(ranked) == len(set(ranked)) == expected
assert set(top).isdisjoint(pruned)
assert len(set(top) | set(pruned)) == expected
assert set(top) | set(pruned) == set(ranked)
print('Saliency assets validated:', json.dumps(audit, sort_keys=True))
PY
}

if assets_ready; then
    validate_assets
    echo "collector already completed; no retraining required"
    exit 0
fi

if [[ -e "$checkpoint_dir" || -e "$importance_file" || -e "$top_file" \
      || -e "$pruned_file" || -e "$audit_file" ]]; then
    echo "partial collector outputs detected; refusing to mix artifacts from different attempts"
    exit 1
fi

if [[ ! -f "${source_checkpoint_path%/}/checkpoint" ]]; then
    echo "source checkpoint is not ready: ${source_checkpoint_path%/}/checkpoint"
    exit 1
fi

day="$train_start_day"
while [[ "$day" -le "$train_end_day" ]]; do
    day_root="${new_train_hdfs_dir%/}/${day}"
    "$hadoop" fs -test -e "${day_root}/_SUCCESS"
    "$hadoop" fs -ls "${day_root}/part*" >/dev/null
    day=$(date -d "$day +1 day" +%Y%m%d)
done

nowt=$(date +%Y%m%d%H%M%S)
export SALIENCY_COLLECT=1
export SALIENCY_EXPECTED_SLOT_COUNT="$saliency_expected_slot_count"
export SALIENCY_REMAIN_COUNT="$saliency_remain_count"

HADOOP_HDFS_HOME=/usr/local/hadoop-current
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${HADOOP_HDFS_HOME}/lib/native:${JAVA_HOME:-/usr/lib/jvm/java-8-openjdk-amd64}/jre/lib/amd64/server"
CLASSPATH=$(${HADOOP_HDFS_HOME}/bin/hadoop classpath --glob) \
"$python" -u train.py \
    -data "$new_train_hdfs_dir" \
    -start_day "$train_start_day" \
    -end_day "$train_end_day" \
    -checkpoint_path "$source_checkpoint_path" \
    -dump_serving_model 0 \
    > "log/saliency_collect_${train_start_day}_${train_end_day}_${nowt}" 2>&1

if ! assets_ready; then
    echo "collector finished without the complete checkpoint/Saliency asset set"
    exit 1
fi
validate_assets
echo "Saliency collector completed: checkpoint=${train_end_day}, top=${saliency_remain_count}, pruned=${pruned_count}"
