#!/bin/bash
set -euo pipefail

BASE_DIR=${BASE_DIR:-/home/luban/rank-ssl/chenpinyuan/tf_rank_BR}
TOOLS_DIR=${TOOLS_DIR:-${BASE_DIR}/shared_tools}
STATE_DIR=${STATE_DIR:-${BASE_DIR}/automation_state}
PYTHON_BIN=${PYTHON_BIN:-/nfs/volume-100003-1/raochongzhi/envs/tf_a6000/bin/python}
REPO_URL=${REPO_URL:-https://github.com/jsdiier/tf_rank_BR.git}

mkdir -p "${STATE_DIR}/runs" "${STATE_DIR}/batches" "${STATE_DIR}/locks" "${STATE_DIR}/logs"

"${PYTHON_BIN}" "${TOOLS_DIR}/automation/discover_runs.py" \
  --base-dir "${BASE_DIR}" --state-dir "${STATE_DIR}" --tools-dir "${TOOLS_DIR}" \
  --repo-url "${REPO_URL}"

"${PYTHON_BIN}" "${TOOLS_DIR}/automation/monitor_runs.py" \
  --base-dir "${BASE_DIR}" --state-dir "${STATE_DIR}"

"${PYTHON_BIN}" "${TOOLS_DIR}/automation/manage_batch.py" \
  --base-dir "${BASE_DIR}" --state-dir "${STATE_DIR}" \
  --tools-dir "${TOOLS_DIR}" --python "${PYTHON_BIN}"
