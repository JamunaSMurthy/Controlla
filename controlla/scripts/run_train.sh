#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_DIR}/.." && pwd)"

VENV_PYTHON="${REPO_ROOT}/.venv/bin/python"
PYTHON_BIN="${PYTHON_BIN:-${VENV_PYTHON}}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python)"
fi

CONFIG="${1:-configs/train.yaml}"

cd "${REPO_ROOT}"

echo "[Controlla] Running training smoke test"
echo "[Controlla] Python: ${PYTHON_BIN}"
echo "[Controlla] Config: ${CONFIG}"

"${PYTHON_BIN}" -m controlla.pipelines.train_controlla \
  --config "${CONFIG}" \
  --dummy-data \
  "${@:2}"