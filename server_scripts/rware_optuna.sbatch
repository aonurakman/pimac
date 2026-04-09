#!/usr/bin/env bash
set -euo pipefail

SUITE_ID="${1:?suite id is required}"
PARALLEL_JOBS="${2:-1}"

REPO_ROOT="${SLURM_SUBMIT_DIR:-${PWD}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MANIFEST_PATH="${MANIFEST_PATH:-${REPO_ROOT}/optuna/study_library/rware_full.json}"

echo "Job started on $(hostname)"
echo "Repository: ${REPO_ROOT}"
echo "Suite id: ${SUITE_ID}"
echo "Parallel jobs: ${PARALLEL_JOBS}"
date

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi

if [ ! -f "${REPO_ROOT}/requirements.txt" ]; then
    echo "requirements.txt not found under ${REPO_ROOT}" >&2
    echo "Submit the job from the repo root or keep sbatch --chdir pointed at the repo root." >&2
    exit 1
fi

cd "${REPO_ROOT}"

if [ ! -d "venv" ]; then
    "${PYTHON_BIN}" -m venv venv
fi

venv/bin/python -m pip install --upgrade pip
venv/bin/pip install -r requirements.txt

venv/bin/python optuna/study.py \
    --manifest "${MANIFEST_PATH}" \
    --suite-id "${SUITE_ID}" \
    --parallel-jobs "${PARALLEL_JOBS}"
