#!/usr/bin/env bash
set -euo pipefail

if [ "${1:-}" = "" ]; then
    echo "Usage: bash server_scripts/run_final_lbf_spread_checkpoint_cpu.sh RUN_NAME [MAX_PARALLEL]" >&2
    exit 1
fi

RUN_NAME="$1"
MAX_PARALLEL="${2:-30}"

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

ensure_venv() {
    if [ ! -d "venv" ]; then
        "${PYTHON_BIN}" -m venv venv
        return 0
    fi

    if [ ! -x "venv/bin/python" ] || ! venv/bin/python -c 'import sys; print(sys.executable)' >/dev/null 2>&1; then
        echo "Recreating incompatible repo-local venv under ${REPO_ROOT}/venv" >&2
        rm -rf venv
        "${PYTHON_BIN}" -m venv venv
    fi
}

if [ "${DRY_RUN}" != "1" ]; then
    ensure_venv
    venv/bin/python -m pip install --upgrade pip
    venv/bin/python -m pip install -r requirements.txt
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
        results_root / "_task_configs" / "lbf_hard_final_with_checkpoints.json",
        {
            "episodes": 40000,
            "eval_every_episodes": 2000,
            "checkpoint_selection_mode": "final",
            "validation_rollouts": 10,
            "test_rollouts": 100,
            "save_checkpoint_every_episodes": 2000,
        },
    ),
    (
        repo_root / "simple_spread_dynamic_hard" / "task.json",
        results_root / "_task_configs" / "spread_hard_final_with_checkpoints.json",
        {
            "episodes": 40000,
            "eval_every_episodes": 2000,
            "checkpoint_selection_mode": "final",
            "validation_rollouts": 10,
            "test_rollouts": 100,
            "save_checkpoint_every_episodes": 2000,
        },
    ),
]

for src, dst, overrides in jobs:
    config = json.loads(src.read_text())
    config.update(overrides)
    dst.write_text(json.dumps(config, indent=2))
    print(dst)
PY

launch_job() {
    local task_dir="$1"
    local algorithm="$2"
    local config_name="$3"
    local task_config_path="$4"
    local seed="$5"

    local task_name
    task_name="$(basename "${task_dir}")"
    local config_path="${task_dir}/configs/${algorithm}/${config_name}.json"
    if [ ! -f "${config_path}" ]; then
        echo "Missing config: ${config_path}" >&2
        exit 1
    fi

    local run_id="final_${task_name}_${algorithm}_${config_name}_ckpts_s${seed}"
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

LBF_SPECS=(
    "mappo:best_01"
    "ippo:best_01"
    "pimac_v0:best_01"
    "pimac_v6:active_03"
)

SPREAD_SPECS=(
    "mappo:best_01"
    "ippo:best_01"
    "pimac_v0:best_01"
    "pimac_v6:active_03"
)

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

for spec in "${LBF_SPECS[@]}"; do
    IFS=: read -r algorithm config_name <<< "${spec}"
    for seed in ${SEEDS}; do
        if [ "${DRY_RUN}" = "1" ]; then
            launch_job "lbf_hard" "${algorithm}" "${config_name}" "${RESULTS_ROOT}/_task_configs/lbf_hard_final_with_checkpoints.json" "${seed}"
            continue
        fi
        while [ "${running_jobs}" -ge "${MAX_PARALLEL}" ]; do
            wait_for_one
        done
        launch_job "lbf_hard" "${algorithm}" "${config_name}" "${RESULTS_ROOT}/_task_configs/lbf_hard_final_with_checkpoints.json" "${seed}" &
        running_jobs=$((running_jobs + 1))
    done
done

for spec in "${SPREAD_SPECS[@]}"; do
    IFS=: read -r algorithm config_name <<< "${spec}"
    for seed in ${SEEDS}; do
        if [ "${DRY_RUN}" = "1" ]; then
            launch_job "simple_spread_dynamic_hard" "${algorithm}" "${config_name}" "${RESULTS_ROOT}/_task_configs/spread_hard_final_with_checkpoints.json" "${seed}"
            continue
        fi
        while [ "${running_jobs}" -ge "${MAX_PARALLEL}" ]; do
            wait_for_one
        done
        launch_job "simple_spread_dynamic_hard" "${algorithm}" "${config_name}" "${RESULTS_ROOT}/_task_configs/spread_hard_final_with_checkpoints.json" "${seed}" &
        running_jobs=$((running_jobs + 1))
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
