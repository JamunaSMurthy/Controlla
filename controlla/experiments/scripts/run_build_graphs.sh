#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="${1:-experiments/configs/eval_main.yaml}"

"${PYTHON_BIN}" -m controlla.experiments.graphs.build_emotion_graph --config "${CONFIG_PATH}"
"${PYTHON_BIN}" -m controlla.experiments.graphs.build_identity_graph --config "${CONFIG_PATH}"
"${PYTHON_BIN}" -m controlla.experiments.graphs.build_cross_modal_graph --config "${CONFIG_PATH}"
