#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${1:-controlla/experiments/configs/eval_paper.yaml}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

cd "${WORKSPACE_DIR}"
"${PYTHON_BIN}" -m controlla.experiments.runners.run_paper_experiments --config "${CONFIG}" "${@:2}"
