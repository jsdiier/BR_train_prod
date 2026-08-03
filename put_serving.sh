#!/bin/bash

source /etc/profile
source ./common.conf

nowt=`date +"%Y%m%d%H%M"`

set -x
set -e

if [ ! -d "./log" ]; then
    mkdir -p "./log"
fi

exec 1>"./log/put_serving_log_$nowt" 2>&1

if [ "$need_dump_serving_model" -ne 1 ]; then
    echo "need_dump_serving_model != 1, skip"
    exit 0
fi

DONE_FILE="model/model.done"
if [ ! -f "$DONE_FILE" ]; then
    echo "model.done not found: $DONE_FILE"
    exit 1
fi

last_line=$(tail -n 1 "$DONE_FILE")
if [ -z "$last_line" ]; then
    echo "model.done is empty"
    exit 1
fi

day=$(echo "$last_line" | awk -F'\t' '{print $1}')
if [ -z "$day" ]; then
    echo "failed to parse day from model.done last line: $last_line"
    exit 1
fi

# dump_serving_model(end_day, epo) 落盘路径是 serving_model_<epo>/<end_day>00，epo 固定传0
serving_day="${day}00"
serving_model_path="./serving_model_0/${serving_day}"

if [ ! -d "$serving_model_path" ]; then
    echo "serving model dir not exist: $serving_model_path"
    exit 1
fi

echo "found latest trained day=$day, uploading $serving_model_path -> ${hadoop_serving_model}/${serving_day}"

if ! $hadoop fs -test -e "${hadoop_serving_model}"; then
    $hadoop fs -mkdir -p "${hadoop_serving_model}"
fi

$hadoop fs -rm -r -f "${hadoop_serving_model}/${serving_day}" || true
$hadoop fs -put "$serving_model_path" "${hadoop_serving_model}/${serving_day}"

echo "put_serving done: ${hadoop_serving_model}/${serving_day}"
