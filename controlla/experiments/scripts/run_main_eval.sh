#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
PAPER_CONFIG="${1:-experiments/configs/eval_paper.yaml}"
ABLATION_CONFIG="${2:-experiments/configs/eval_ablation.yaml}"
SENSITIVITY_CONFIG="${3:-experiments/configs/eval_sensitivity.yaml}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

echo "[Controlla] Running full evaluation suite"
echo "[Controlla] Paper config: ${PAPER_CONFIG}"
echo "[Controlla] Ablation config: ${ABLATION_CONFIG}"
echo "[Controlla] Sensitivity config: ${SENSITIVITY_CONFIG}"

"${PYTHON_BIN}" -m controlla.experiments.runners.run_paper_experiments \
  --config "${PAPER_CONFIG}"

"${PYTHON_BIN}" -m controlla.experiments.runners.run_ablations \
  --config "${ABLATION_CONFIG}"

"${PYTHON_BIN}" -m controlla.experiments.runners.run_sensitivity \
  --config "${SENSITIVITY_CONFIG}"

echo "[Controlla] Evaluation suite finished"