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

CONFIG="${1:-configs/infer.yaml}"
PROMPT="${PROMPT:-calm portrait}"

cd "${REPO_ROOT}"

echo "[Controlla] Running inference"
echo "[Controlla] Python: ${PYTHON_BIN}"
echo "[Controlla] Config: ${CONFIG}"
echo "[Controlla] Prompt: ${PROMPT}"

"${PYTHON_BIN}" -m controlla.pipelines.infer_controlla \
  --config "${CONFIG}" \
  --prompt "${PROMPT}" \
  "${@:2}"