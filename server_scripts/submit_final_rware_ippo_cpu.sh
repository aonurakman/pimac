#!/usr/bin/env bash
set -euo pipefail

if [ "${1:-}" = "" ]; then
    echo "Usage: bash server_scripts/submit_final_rware_ippo_cpu.sh RUN_NAME [MAX_PARALLEL]" >&2
    exit 1
fi

RUN_NAME="$1"
MAX_PARALLEL="${2:-5}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${REPO_ROOT}/slurm_logs"

PARTITION="${PARTITION:-rknodes}"
QOS="${QOS:-big_bonk}"
CPUS_PER_TASK="${CPUS_PER_TASK:-5}"
MEM="${MEM:-120G}"
JOB_NAME="${JOB_NAME:-final_rware_ippo_long_cpu}"

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

sbatch "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/run_final_rware_ippo_cpu.sh" "${RUN_NAME}" "${MAX_PARALLEL}"
