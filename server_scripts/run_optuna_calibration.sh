#!/usr/bin/env bash
set -euo pipefail

SUITE_ID="${1:?suite id is required}"
PARALLEL_JOBS="${2:-3}"

REPO_ROOT="${SLURM_SUBMIT_DIR:-${PWD}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MANIFEST_PATH="${MANIFEST_PATH:-}"
SHOW_CHILD_OUTPUT="${SHOW_CHILD_OUTPUT:-0}"

echo "Job started on $(hostname)"
echo "Repository: ${REPO_ROOT}"
echo "Manifest: ${MANIFEST_PATH}"
echo "Suite id: ${SUITE_ID}"
echo "Parallel jobs: ${PARALLEL_JOBS}"
date

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi

if [ ! -f "${REPO_ROOT}/requirements.txt" ]; then
    echo "requirements.txt not found under ${REPO_ROOT}" >&2
    exit 1
fi

if [ -z "${MANIFEST_PATH}" ]; then
    echo "MANIFEST_PATH is required." >&2
    exit 1
fi

if [ ! -f "${MANIFEST_PATH}" ]; then
    echo "Manifest not found: ${MANIFEST_PATH}" >&2
    exit 1
fi

cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if [ ! -d "venv" ]; then
    "${PYTHON_BIN}" -m venv venv
fi

venv/bin/python -m pip install --upgrade pip
venv/bin/pip install -r requirements.txt

ARGS=(
    optuna/study.py
    --manifest "${MANIFEST_PATH}"
    --suite-id "${SUITE_ID}"
    --parallel-jobs "${PARALLEL_JOBS}"
)

if [ "${SHOW_CHILD_OUTPUT}" = "1" ]; then
    ARGS+=(--show-child-output)
fi

venv/bin/python "${ARGS[@]}"
