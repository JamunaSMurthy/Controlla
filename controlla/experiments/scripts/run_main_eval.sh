#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
MAIN_CONFIG="${1:-experiments/configs/eval_main.yaml}"
ABLATION_CONFIG="${2:-experiments/configs/eval_ablation.yaml}"
SENSITIVITY_CONFIG="${3:-experiments/configs/eval_sensitivity.yaml}"

"${PYTHON_BIN}" -m controlla.experiments.runners.run_main_table --config "${MAIN_CONFIG}"
"${PYTHON_BIN}" -m controlla.experiments.runners.run_ablations --config "${ABLATION_CONFIG}"
"${PYTHON_BIN}" -m controlla.experiments.runners.run_sensitivity --config "${SENSITIVITY_CONFIG}"
