#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"

source ./common.conf

RESTORE_DAY=20260731
START_DAY=20260801
END_DAY=20260906
RESTORE_DIR="model/checkpoints/${RESTORE_DAY}"

mkdir -p log model

echo "PAL_CONTINUATION_BEGIN restore=${RESTORE_DAY} train=${START_DAY}..${END_DAY}"
echo "PAL_CONTINUATION_DATA ${train_hdfs_dir}"
echo "PAL_CONTINUATION_MODE checkpoint_only no_test no_serving"

if [[ ! -f "${RESTORE_DIR}/checkpoint" ]]; then
    echo "ERROR missing restore checkpoint metadata: ${RESTORE_DIR}/checkpoint"
    exit 1
fi

if ! compgen -G "${RESTORE_DIR}/*.index" >/dev/null; then
    echo "ERROR missing restore checkpoint index: ${RESTORE_DIR}/*.index"
    exit 1
fi

if ! compgen -G "${RESTORE_DIR}/*.data-*" >/dev/null; then
    echo "ERROR missing restore checkpoint data: ${RESTORE_DIR}/*.data-*"
    exit 1
fi

for day in "${START_DAY}" "${END_DAY}"; do
    if ! "${hadoop}" fs -test -e "${train_hdfs_dir}${day}/_SUCCESS"; then
        echo "ERROR training boundary day is not ready: ${train_hdfs_dir}${day}/_SUCCESS"
        exit 1
    fi
done

HADOOP_HDFS_HOME=/usr/local/hadoop-current
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${HADOOP_HDFS_HOME}/lib/native:${JAVA_HOME}/jre/lib/amd64/server"
export CLASSPATH
CLASSPATH=$("${HADOOP_HDFS_HOME}/bin/hadoop" classpath --glob)

"${python}" -u train.py \
    -data "${train_hdfs_dir}" \
    -start_day "${START_DAY}" \
    -end_day "${END_DAY}" \
    -checkpoint_path "${RESTORE_DIR}/" \
    -dump_serving_model 0

if [[ ! -f "model/checkpoints/${END_DAY}/checkpoint" ]]; then
    echo "ERROR target checkpoint was not produced: model/checkpoints/${END_DAY}/checkpoint"
    exit 1
fi

if ! compgen -G "model/checkpoints/${END_DAY}/*.index" >/dev/null || \
   ! compgen -G "model/checkpoints/${END_DAY}/*.data-*" >/dev/null; then
    echo "ERROR target checkpoint is incomplete: model/checkpoints/${END_DAY}"
    exit 1
fi

echo "PAL_CONTINUATION_COMPLETED checkpoint=model/checkpoints/${END_DAY}/"
