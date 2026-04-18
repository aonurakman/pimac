#!/usr/bin/env bash
set -euo pipefail

if [ "${1:-}" = "" ]; then
    echo "Usage: bash server_scripts/submit_hard_baseline_calibrations.sh SUITE_PREFIX [PARALLEL_JOBS]" >&2
    exit 1
fi

SUITE_PREFIX="$1"
PARALLEL_JOBS="${2:-4}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/submit_lbf_hard_baseline_calibration.sh" "${SUITE_PREFIX}_lbf_hard" "${PARALLEL_JOBS}"
bash "${SCRIPT_DIR}/submit_spread_hard_baseline_calibration.sh" "${SUITE_PREFIX}_spread_hard" "${PARALLEL_JOBS}"
bash "${SCRIPT_DIR}/submit_rware_baseline_calibration.sh" "${SUITE_PREFIX}_rware" "${PARALLEL_JOBS}"
