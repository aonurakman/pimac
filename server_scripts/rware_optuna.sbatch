#!/usr/bin/env bash
set -euo pipefail

SUITE_ID="${1:?suite id is required}"
PARALLEL_JOBS="${2:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MANIFEST_PATH="${MANIFEST_PATH:-optuna/study_library/rware_full.json}"

echo "Job started on $(hostname)"
echo "Repository: ${REPO_ROOT}"
echo "Suite id: ${SUITE_ID}"
echo "Parallel jobs: ${PARALLEL_JOBS}"
date

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
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
