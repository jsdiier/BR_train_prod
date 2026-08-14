#!/bin/bash
set -eo pipefail

# crontab does not load the interactive shell environment. Activate the same
# environment used to run shared_tools manually so dependencies such as
# tabulate are available to test_multi.py and all aggregation subprocesses.
CONDA_ACTIVATE=${CONDA_ACTIVATE:-/home/luban/rank-ssl/chenpinyuan/miniconda_base/bin/activate}
AUTOMATION_ENV=${AUTOMATION_ENV:-SFT_A6000}
if [[ ! -f "${CONDA_ACTIVATE}" ]]; then
  echo "automation conda activate script is missing: ${CONDA_ACTIVATE}" >&2
  exit 1
fi
source "${CONDA_ACTIVATE}" "${AUTOMATION_ENV}"
set -u

BASE_DIR=${BASE_DIR:-/home/luban/rank-ssl/chenpinyuan/tf_rank_BR}
TOOLS_DIR=${TOOLS_DIR:-${BASE_DIR}/shared_tools}
STATE_DIR=${STATE_DIR:-${BASE_DIR}/automation_state}
PYTHON_BIN=${PYTHON_BIN:-$(command -v python)}
REPO_URL=${REPO_URL:-https://github.com/jsdiier/tf_rank_BR.git}

mkdir -p "${STATE_DIR}/runs" "${STATE_DIR}/batches" "${STATE_DIR}/locks" "${STATE_DIR}/logs"

echo "[AUTOMATION] env=${AUTOMATION_ENV} python=${PYTHON_BIN}"

"${PYTHON_BIN}" "${TOOLS_DIR}/automation/discover_runs.py" \
  --base-dir "${BASE_DIR}" --state-dir "${STATE_DIR}" --tools-dir "${TOOLS_DIR}" \
  --repo-url "${REPO_URL}"

"${PYTHON_BIN}" "${TOOLS_DIR}/automation/monitor_runs.py" \
  --base-dir "${BASE_DIR}" --state-dir "${STATE_DIR}"

"${PYTHON_BIN}" "${TOOLS_DIR}/automation/manage_batch.py" \
  --base-dir "${BASE_DIR}" --state-dir "${STATE_DIR}" \
  --tools-dir "${TOOLS_DIR}" --python "${PYTHON_BIN}"
