#!/usr/bin/env bash
set -euo pipefail

if [ "${1:-}" = "" ]; then
    echo "Usage: bash server_scripts/submit_rware_optuna.sh SUITE_ID [PARALLEL_JOBS]" >&2
    exit 1
fi

SUITE_ID="$1"
PARALLEL_JOBS="${2:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${REPO_ROOT}/slurm_logs"

PARTITION="${PARTITION:-rknodes}"
QOS="${QOS:-big_bonk}"
GRES="${GRES:-gpu:1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEM="${MEM:-64G}"
JOB_NAME="${JOB_NAME:-rware_optuna}"

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

if [ -n "${GRES}" ]; then
    SBATCH_ARGS+=(--gres "${GRES}")
fi

sbatch "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/rware_optuna.sbatch" "${SUITE_ID}" "${PARALLEL_JOBS}"
