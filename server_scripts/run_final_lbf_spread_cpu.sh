#!/usr/bin/env bash
set -euo pipefail

if [ "${1:-}" = "" ]; then
    echo "Usage: bash server_scripts/run_final_lbf_spread_cpu.sh RUN_NAME [MAX_PARALLEL]" >&2
    exit 1
fi

RUN_NAME="$1"
MAX_PARALLEL="${2:-16}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SLURM_SUBMIT_DIR:-${PWD}}"
if [ ! -f "${REPO_ROOT}/requirements.txt" ]; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
RESULTS_ROOT="${RESULTS_ROOT:-results/${RUN_NAME}}"
SEEDS="${SEEDS:-42 43 44 45 46}"
DRY_RUN="${DRY_RUN:-0}"

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "${REPO_ROOT}"

if [ "${DRY_RUN}" != "1" ]; then
    if [ ! -d "venv" ]; then
        "${PYTHON_BIN}" -m venv venv
    fi

    venv/bin/python -m pip install --upgrade pip
    venv/bin/pip install -r requirements.txt
fi

mkdir -p "${RESULTS_ROOT}/logs" "${RESULTS_ROOT}/_task_configs"

venv/bin/python - <<'PY' "${RESULTS_ROOT}"
import json
import sys
from pathlib import Path

results_root = Path(sys.argv[1])
repo_root = Path.cwd()

jobs = [
    (
        repo_root / "lbf_hard" / "task.json",
        results_root / "_task_configs" / "lbf_hard_final.json",
        {
            "episodes": 12000,
            "eval_every_episodes": 0,
            "test_rollouts": 100,
        },
    ),
    (
        repo_root / "simple_spread_dynamic_hard" / "task.json",
        results_root / "_task_configs" / "spread_hard_final.json",
        {
            "episodes": 20000,
            "eval_every_episodes": 0,
            "test_rollouts": 100,
        },
    ),
]

for src, dst, overrides in jobs:
    config = json.loads(src.read_text())
    config.update(overrides)
    dst.write_text(json.dumps(config, indent=2))
    print(dst)
PY

discover_configs() {
    local task_dir="$1"
    local algorithm="$2"
    find "${task_dir}/configs/${algorithm}" -maxdepth 1 -type f -name '*.json' | sort
}

launch_job() {
    local task_dir="$1"
    local algorithm="$2"
    local config_path="$3"
    local task_config_path="$4"
    local seed="$5"

    local task_name
    task_name="$(basename "${task_dir}")"
    local config_name
    config_name="$(basename "${config_path}" .json)"
    local run_id="final_${task_name}_${algorithm}_${config_name}_s${seed}"
    local log_path="${RESULTS_ROOT}/logs/${run_id}.log"

    local cmd=(
        venv/bin/python
        "${task_dir}/run.py"
        --algorithm "${algorithm}"
        --alg-config "${config_path}"
        --task-config "${task_config_path}"
        --seed "${seed}"
        --device cpu
        --results-root "${RESULTS_ROOT}"
        --run-id "${run_id}"
        --skip-gif
    )

    if [ "${DRY_RUN}" = "1" ]; then
        printf '%q ' "${cmd[@]}"
        printf '\n'
        return 0
    fi

    echo "[launch] ${run_id}"
    "${cmd[@]}" >"${log_path}" 2>&1
}

TASKS=(
    "lbf_hard"
    "simple_spread_dynamic_hard"
)
ALGORITHMS=(
    "mappo"
    "pimac_v0"
    "pimac_v6"
    "pimac_v7"
)

declare -a PIDS=()
declare -a FAILED=()
running_jobs=0

wait_for_one() {
    local status=0
    if ! wait -n; then
        status=$?
    fi
    running_jobs=$((running_jobs - 1))
    if [ ${status} -ne 0 ]; then
        FAILED+=("1")
    fi
}

echo "Repository: ${REPO_ROOT}"
echo "Results root: ${RESULTS_ROOT}"
echo "Seeds: ${SEEDS}"
echo "Max parallel jobs: ${MAX_PARALLEL}"
date

for task in "${TASKS[@]}"; do
    case "${task}" in
        lbf_hard)
            task_config_path="${RESULTS_ROOT}/_task_configs/lbf_hard_final.json"
            ;;
        simple_spread_dynamic_hard)
            task_config_path="${RESULTS_ROOT}/_task_configs/spread_hard_final.json"
            ;;
        *)
            echo "Unknown task: ${task}" >&2
            exit 1
            ;;
    esac

    for algorithm in "${ALGORITHMS[@]}"; do
        configs=()
        while IFS= read -r config_path; do
            configs+=("${config_path}")
        done < <(discover_configs "${task}" "${algorithm}")
        if [ "${#configs[@]}" -eq 0 ]; then
            echo "No configs found under ${task}/configs/${algorithm}" >&2
            exit 1
        fi

        for config_path in "${configs[@]}"; do
            for seed in ${SEEDS}; do
                if [ "${DRY_RUN}" = "1" ]; then
                    launch_job "${task}" "${algorithm}" "${config_path}" "${task_config_path}" "${seed}"
                    continue
                fi

                while [ "${running_jobs}" -ge "${MAX_PARALLEL}" ]; do
                    wait_for_one
                done

                launch_job "${task}" "${algorithm}" "${config_path}" "${task_config_path}" "${seed}" &
                PIDS+=("$!")
                running_jobs=$((running_jobs + 1))
            done
        done
    done
done

if [ "${DRY_RUN}" != "1" ]; then
    while [ "${running_jobs}" -gt 0 ]; do
        wait_for_one
    done
fi

if [ "${#FAILED[@]}" -ne 0 ]; then
    echo "One or more runs failed." >&2
    exit 1
fi

echo "All runs completed."
date
