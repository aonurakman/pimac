#!/usr/bin/env bash
set -euo pipefail

if [ "${1:-}" = "" ]; then
    echo "Usage: bash server_scripts/submit_v67_active_calibrations.sh SUITE_PREFIX [PARALLEL_JOBS]" >&2
    exit 1
fi

SUITE_PREFIX="$1"
PARALLEL_JOBS="${2:-2}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/submit_lbf_hard_v67_active_calibration.sh" "${SUITE_PREFIX}_lbf_hard" "${PARALLEL_JOBS}"
bash "${SCRIPT_DIR}/submit_spread_hard_v67_active_calibration.sh" "${SUITE_PREFIX}_spread_hard" "${PARALLEL_JOBS}"
bash "${SCRIPT_DIR}/submit_rware_v67_active_calibration.sh" "${SUITE_PREFIX}_rware" "${PARALLEL_JOBS}"
