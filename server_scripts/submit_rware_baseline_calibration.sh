#!/usr/bin/env bash
set -euo pipefail

if [ "${1:-}" = "" ]; then
    echo "Usage: bash server_scripts/submit_rware_baseline_calibration.sh SUITE_ID [PARALLEL_JOBS]" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MANIFEST_PATH="${REPO_ROOT}/optuna/study_library/rware_baseline_calibration.json" \
JOB_NAME="${JOB_NAME:-rware_baseline_cal}" \
bash "${SCRIPT_DIR}/submit_optuna_calibration.sh" "$1" "${2:-4}"
