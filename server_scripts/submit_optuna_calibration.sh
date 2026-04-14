#!/usr/bin/env bash
set -euo pipefail

if [ "${1:-}" = "" ]; then
    echo "Usage: MANIFEST_PATH=/abs/path/to/manifest.json bash server_scripts/submit_optuna_calibration.sh SUITE_ID [PARALLEL_JOBS]" >&2
    exit 1
fi

SUITE_ID="$1"
PARALLEL_JOBS="${2:-3}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${REPO_ROOT}/slurm_logs"

MANIFEST_PATH="${MANIFEST_PATH:-}"
PARTITION="${PARTITION:-rknodes}"
QOS="${QOS:-big_bonk}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEM="${MEM:-40G}"
JOB_NAME="${JOB_NAME:-optuna_calibration}"

if [ -z "${MANIFEST_PATH}" ]; then
    echo "MANIFEST_PATH is required." >&2
    exit 1
fi

mkdir -p "${LOG_DIR}"

SBATCH_ARGS=(
    --job-name "${JOB_NAME}"
    --partition "${PARTITION}"
    --cpus-per-task "${CPUS_PER_TASK}"
    --mem "${MEM}"
    --output "${LOG_DIR}/%x_%j.out"
    --error "${LOG_DIR}/%x_%j.err"
    --chdir "${REPO_ROOT}"
)

if [ -n "${QOS}" ]; then
    SBATCH_ARGS+=(--qos "${QOS}")
fi

sbatch "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/run_optuna_calibration.sh" "${SUITE_ID}" "${PARALLEL_JOBS}"
