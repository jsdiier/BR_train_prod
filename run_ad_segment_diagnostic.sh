#!/bin/bash

set -euo pipefail

source /etc/profile

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"
source ./common.conf

BASE_DIR=/home/luban/rank-ssl/chenpinyuan/tf_rank_BR_prod
DATA_ROOT=hdfs://DClusterUS1/user/prod_soda_trade_strategy/rank/chenpinyuan/hash_fea_new_fixed/train
BASELINE_DIR=${BASE_DIR}/BR_train_prod_bs_lr_ema_weights
TOP1700_DIR=${BASE_DIR}/BR_train_prod_ema_interest_feature_tfrecord_fixed_remain1700_e2e
OUTPUT_DIR=${SCRIPT_DIR}/log/ad_segment_ckpt_20260825_20260826_20260828

HADOOP_HDFS_HOME=/usr/local/hadoop-current
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:${HADOOP_HDFS_HOME}/lib/native:${JAVA_HOME}/jre/lib/amd64/server
export CLASSPATH=$(${HADOOP_HDFS_HOME}/bin/hadoop classpath --glob)

mkdir -p "${OUTPUT_DIR}"

MODE=${1:-}
if [[ "${MODE}" == "inspect" ]]; then
    exec "${python}" -u "${SCRIPT_DIR}/inspect_is_ad_fids.py" \
        --experiment-dir "${BASELINE_DIR}" \
        --data-root "${DATA_ROOT}" \
        --start-day 20260826 \
        --end-day 20260828 \
        --max-samples 100000 \
        --output "${OUTPUT_DIR}/is_ad_fid_audit.tsv"
fi

if [[ "${MODE}" != "run" ]]; then
    echo "usage: bash run_ad_segment_diagnostic.sh inspect" >&2
    echo "   or: bash run_ad_segment_diagnostic.sh run <confirmed_ad_fid>" >&2
    exit 2
fi

AD_FID=${2:-}
if [[ -z "${AD_FID}" ]]; then
    echo "confirmed ad FID is required; run inspect first" >&2
    exit 2
fi

"${python}" -u "${SCRIPT_DIR}/dump_ad_segment_predictions.py" \
    --experiment-dir "${BASELINE_DIR}" \
    --checkpoint "${BASELINE_DIR}/model/checkpoints/20260825" \
    --data-root "${DATA_ROOT}" \
    --start-day 20260826 \
    --end-day 20260828 \
    --label baseline \
    --output "${OUTPUT_DIR}/baseline_predictions.npz"

"${python}" -u "${SCRIPT_DIR}/dump_ad_segment_predictions.py" \
    --experiment-dir "${TOP1700_DIR}" \
    --checkpoint "${TOP1700_DIR}/model/checkpoints/20260825" \
    --data-root "${DATA_ROOT}" \
    --start-day 20260826 \
    --end-day 20260828 \
    --label top1700 \
    --output "${OUTPUT_DIR}/top1700_predictions.npz"

"${python}" -u "${SCRIPT_DIR}/compare_ad_segments.py" \
    --baseline "${OUTPUT_DIR}/baseline_predictions.npz" \
    --top1700 "${OUTPUT_DIR}/top1700_predictions.npz" \
    --ad-fid "${AD_FID}" \
    --output-dir "${OUTPUT_DIR}"

echo "AD_SEGMENT_DIAGNOSTIC_COMPLETED output=${OUTPUT_DIR}"
