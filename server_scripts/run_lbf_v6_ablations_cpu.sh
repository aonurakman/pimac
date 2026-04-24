#!/usr/bin/env bash
set -euo pipefail

if [ "${1:-}" = "" ]; then
    echo "Usage: bash server_scripts/run_lbf_v6_ablations_cpu.sh RUN_NAME [MAX_PARALLEL]" >&2
    exit 1
fi

RUN_NAME="$1"
MAX_PARALLEL="${2:-15}"

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
src = repo_root / "lbf_hard" / "task.json"
dst = results_root / "_task_configs" / "lbf_hard_final_ablation.json"
config = json.loads(src.read_text())
config.pop("checkpoint_selection_mode", None)
config.pop("save_checkpoint_every_episodes", None)
config.pop("save_checkpoint_episodes", None)
config.update(
    {
        "episodes": 12000,
        "eval_every_episodes": 0,
        "validation_rollouts": 5,
        "test_rollouts": 100,
    }
)
dst.write_text(json.dumps(config, indent=2))
print(dst)
PY

discover_configs() {
    find "lbf_hard/configs/pimac_v6_ablation" -maxdepth 1 -type f -name '*.json' | sort
}

launch_job() {
    local config_path="$1"
    local task_config_path="$2"
    local seed="$3"

    local config_name
    config_name="$(basename "${config_path}" .json)"
    local run_id="final_lbf_hard_pimac_v6_ablation_${config_name}_s${seed}"
    local log_path="${RESULTS_ROOT}/logs/${run_id}.log"

    local cmd=(
        venv/bin/python
        lbf_hard/run.py
        --algorithm pimac_v6_ablation
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

TASK_CONFIG_PATH="${RESULTS_ROOT}/_task_configs/lbf_hard_final_ablation.json"
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

configs=()
while IFS= read -r config_path; do
    configs+=("${config_path}")
done < <(discover_configs)
if [ "${#configs[@]}" -eq 0 ]; then
    echo "No configs found under lbf_hard/configs/pimac_v6_ablation" >&2
    exit 1
fi

for config_path in "${configs[@]}"; do
    for seed in ${SEEDS}; do
        if [ "${DRY_RUN}" = "1" ]; then
            launch_job "${config_path}" "${TASK_CONFIG_PATH}" "${seed}"
            continue
        fi

        while [ "${running_jobs}" -ge "${MAX_PARALLEL}" ]; do
            wait_for_one
        done

        launch_job "${config_path}" "${TASK_CONFIG_PATH}" "${seed}" &
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
