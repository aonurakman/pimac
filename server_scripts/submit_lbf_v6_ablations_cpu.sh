#!/usr/bin/env bash
set -euo pipefail

if [ "${1:-}" = "" ]; then
    echo "Usage: bash server_scripts/submit_lbf_v6_ablations_cpu.sh RUN_NAME [MAX_PARALLEL]" >&2
    exit 1
fi

RUN_NAME="$1"
MAX_PARALLEL="${2:-15}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${REPO_ROOT}/slurm_logs"
mkdir -p "${LOG_DIR}"

PARTITION="${PARTITION:-rknodes}"
QOS="${QOS:-big_bonk}"
MEM="${MEM:-80G}"
CPUS="${CPUS:-15}"
JOB_NAME="${JOB_NAME:-lbf_v6_ablations_cpu}"

SBATCH_ARGS=(
    --job-name "${JOB_NAME}"
    --output "${LOG_DIR}/${RUN_NAME}_%j.out"
    --error "${LOG_DIR}/${RUN_NAME}_%j.err"
    --partition "${PARTITION}"
    --qos "${QOS}"
    --nodes 1
    --ntasks 1
    --cpus-per-task "${CPUS}"
    --mem "${MEM}"
    --chdir "${REPO_ROOT}"
)

sbatch "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/run_lbf_v6_ablations_cpu.sh" "${RUN_NAME}" "${MAX_PARALLEL}"
