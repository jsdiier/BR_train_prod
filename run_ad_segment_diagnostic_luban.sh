#!/bin/bash

set -euo pipefail

source /etc/profile

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"

# slot 1044 / dummy,1：桶 0 对应原始 is_ad=1。
CONFIRMED_AD_FID=293859875685924864

echo "LUBAN_AD_SEGMENT_DIAGNOSTIC_BEGIN ad_fid=${CONFIRMED_AD_FID}"
bash "${SCRIPT_DIR}/run_ad_segment_diagnostic.sh" run "${CONFIRMED_AD_FID}"
echo "LUBAN_AD_SEGMENT_DIAGNOSTIC_COMPLETED"
