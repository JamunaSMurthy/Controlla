#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${1:-experiments/configs/eval_paper.yaml}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

echo "[Controlla] Running paper-level evaluation"
echo "[Controlla] Config: ${CONFIG}"

"${PYTHON_BIN}" -m controlla.experiments.runners.run_paper_experiments \
  --config "${CONFIG}" \
  "${@:2}"