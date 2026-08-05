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

# 不再按 dt 分区，统一固定发布到 export_model，每次用最新 ckpt 整体覆盖
export_model_dir="${hadoop_serving_model}/export_model"

echo "found latest trained day=$day, uploading $serving_model_path -> ${export_model_dir}"

if ! $hadoop fs -test -e "${hadoop_serving_model}"; then
    $hadoop fs -mkdir -p "${hadoop_serving_model}"
fi

$hadoop fs -rm -r -f "${export_model_dir}" || true
$hadoop fs -put "$serving_model_path" "${export_model_dir}"

echo "put_serving done: ${export_model_dir}"

# 把 model.done 最后一天写成一个 tag 文件，跟 export_model 平级，用于确认线上模型对应哪个训练日期
version_tag_local="./model_version"
echo -n "$day" > "$version_tag_local"

$hadoop fs -rm -f "${hadoop_serving_model}/model_version" || true
$hadoop fs -put "$version_tag_local" "${hadoop_serving_model}/model_version"

echo "put_serving model_version done: ${hadoop_serving_model}/model_version (day=$day)"

# 生产 _SUCCESS 标记文件，路径 hadoop_root/BR/year=YYYY/month=MM/day=DD/_SUCCESS（跟 serving_model 平级），
# dt 取 model.done 最后一天($day)，不用 common.conf 的 train_end_day(手动跑非 is_auto_train 场景可能没同步更新)
success_year="${day:0:4}"
success_month="${day:4:2}"
success_day="${day:6:2}"
success_dir="${hadoop_root}BR/year=${success_year}/month=${success_month}/day=${success_day}"

$hadoop fs -mkdir -p "${success_dir}"
$hadoop fs -touchz "${success_dir}/_SUCCESS"

echo "put_serving _SUCCESS done: ${success_dir}/_SUCCESS"
