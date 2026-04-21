#!/usr/bin/env bash
set -euo pipefail

if [ "${1:-}" = "" ]; then
    echo "Usage: bash server_scripts/run_final_rware_cpu.sh RUN_NAME [MAX_PARALLEL]" >&2
    exit 1
fi

RUN_NAME="$1"
MAX_PARALLEL="${2:-32}"

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
src = repo_root / "robotic_warehouse_dynamic" / "task.json"
dst = results_root / "_task_configs" / "robotic_warehouse_dynamic_final.json"
config = json.loads(src.read_text())
config.update(
    {
        "episodes": 3000,
        "eval_every_episodes": 0,
        "test_rollouts": 100,
    }
)
dst.write_text(json.dumps(config, indent=2))
print(dst)
PY

discover_configs() {
    local algorithm="$1"
    find "robotic_warehouse_dynamic/configs/${algorithm}" -maxdepth 1 -type f -name '*.json' | sort
}

launch_job() {
    local algorithm="$1"
    local config_path="$2"
    local task_config_path="$3"
    local seed="$4"

    local config_name
    config_name="$(basename "${config_path}" .json)"
    local run_id="final_robotic_warehouse_dynamic_${algorithm}_${config_name}_s${seed}"
    local log_path="${RESULTS_ROOT}/logs/${run_id}.log"

    local cmd=(
        venv/bin/python
        robotic_warehouse_dynamic/run.py
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

TASK_CONFIG_PATH="${RESULTS_ROOT}/_task_configs/robotic_warehouse_dynamic_final.json"
ALGORITHMS=(
    "mappo"
    "pimac_v0"
    "pimac_v6"
    "pimac_v7"
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

for algorithm in "${ALGORITHMS[@]}"; do
    configs=()
    while IFS= read -r config_path; do
        configs+=("${config_path}")
    done < <(discover_configs "${algorithm}")
    if [ "${#configs[@]}" -eq 0 ]; then
        echo "No configs found under robotic_warehouse_dynamic/configs/${algorithm}" >&2
        exit 1
    fi

    for config_path in "${configs[@]}"; do
        for seed in ${SEEDS}; do
            if [ "${DRY_RUN}" = "1" ]; then
                launch_job "${algorithm}" "${config_path}" "${TASK_CONFIG_PATH}" "${seed}"
                continue
            fi

            while [ "${running_jobs}" -ge "${MAX_PARALLEL}" ]; do
                wait_for_one
            done

            launch_job "${algorithm}" "${config_path}" "${TASK_CONFIG_PATH}" "${seed}" &
            running_jobs=$((running_jobs + 1))
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
